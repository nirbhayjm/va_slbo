# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import pickle
from collections import deque

import lunzi.nn as nn
import numpy as np
import tensorflow as tf
import wandb
from lunzi.Logger import logger
from slbo.algos.TRPO import TRPO
from slbo.dynamics_model import DynamicsModel
from slbo.envs.virtual_env import VirtualEnv
from slbo.loss.mle_loss import MLELoss
from slbo.loss.multi_step_loss import MultiStepLoss
from slbo.loss.va_loss import ValueAwareLoss
from slbo.partial_envs import make_env
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.utils.average_meter import AverageMeter
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.utils.normalizer import Normalizers
from slbo.utils.OU_noise import OUNoise
from slbo.utils.runner import Runner
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.wandb_utils import init_wandb
from slbo.v_function.mlp_v_function import MLPVFunction


def evaluate(settings, tag, use_wandb=False, wandb_log_dict={}):
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
        returns = np.array([ep_info["return"] for ep_info in ep_infos])
        logger.info(
            "Tag = %s, Reward on %s (%d episodes): mean = %.6f, std = %.6f",
            tag,
            name,
            len(returns),
            np.mean(returns),
            np.std(returns),
        )
        if use_wandb:
            if len(returns) > 0:
                sem = 0
            else:
                sem = scipy.stats.sem(returns)
            curr_log_dict = {
                f"return mean {name}": np.mean(returns),
                f"return sem {name}": sem,
            }
            curr_log_dict.update(wandb_log_dict)
            wandb.log(curr_log_dict, commit=False)
            if wandb_log_dict:
                wandb.log(
                    {
                        "stage_" + key.replace(" ", "_"): val
                        for key, val in curr_log_dict.items()
                    },
                    commit=True,
                )


def add_multi_step(src: Dataset, dst: Dataset):
    n_envs = 1
    dst.extend(src[:-n_envs])

    ending = src[-n_envs:].copy()
    ending.timeout = True
    dst.extend(ending)


def make_real_runner(n_envs):
    from slbo.envs.batched_env import BatchedEnv

    batched_env = BatchedEnv([make_env(FLAGS.env.id) for _ in range(n_envs)])
    return Runner(batched_env, rescale_action=True, **FLAGS.runner.as_dict())


def main(FLAGS):
    if FLAGS.use_wandb:
        wandb_name = f"{FLAGS.run_id}"
        init_wandb(
            wandb_name=wandb_name,
            project_name=FLAGS.wandb_project_name,
            config=FLAGS.as_dict(),
            group=FLAGS.wandb_group_name,
        )

    env = make_env(FLAGS.env.id)
    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))

    env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)

    dtype = gen_dtype(env, "state action next_state reward done timeout")
    train_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    dev_set = Dataset(dtype, FLAGS.rollout.max_buf_size)

    policy = GaussianMLPPolicy(
        dim_state, dim_action, normalizer=normalizers.state, **FLAGS.policy.as_dict()
    )
    # batched noises
    noise = OUNoise(
        env.action_space,
        theta=FLAGS.OUNoise.theta,
        sigma=FLAGS.OUNoise.sigma,
        shape=(1, dim_action),
    )
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    model = DynamicsModel(
        dim_state=dim_state,
        dim_action=dim_action,
        normalizers=normalizers,
        hidden_sizes=FLAGS.model.hidden_sizes,
    )

    virt_env = VirtualEnv(
        model, make_env(FLAGS.env.id), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model
    )
    virt_runner = Runner(
        virt_env, **{**FLAGS.runner.as_dict(), "max_steps": FLAGS.plan.max_steps}
    )
    va_virt_runner = Runner(
        virt_env, **{**FLAGS.runner.as_dict(), "max_steps": FLAGS.plan.max_steps}
    )

    size_average = FLAGS.algorithm == "va"
    criterion_map = {
        "L1": nn.L1Loss(size_average=size_average),
        "L2": nn.L2Loss(size_average=size_average),
        "MSE": nn.MSELoss(size_average=size_average),
    }
    criterion = criterion_map[FLAGS.model.loss]
    if FLAGS.algorithm == "va":
        loss_mod = ValueAwareLoss(
            model=model,
            value_function=vfn,
            normalizers=normalizers,
            dim_state=dim_state,
            dim_action=dim_action,
            criterion=criterion,
            step=FLAGS.model.multi_step,
            va_loss_coeff=float(FLAGS.model.va_loss_coeff),
            va_norm=FLAGS.model.va_norm,
            virt_runner=va_virt_runner,
        )
    elif FLAGS.algorithm == "mle":
        loss_mod = MLELoss(
            model=model,
            # value_function=vfn,
            normalizers=normalizers,
            dim_state=dim_state,
            dim_action=dim_action,
            criterion=criterion,
            step=FLAGS.model.multi_step,
        )
    elif FLAGS.algorithm == "slbo":
        loss_mod = MultiStepLoss(
            model=model,
            # value_function=vfn,
            normalizers=normalizers,
            dim_state=dim_state,
            dim_action=dim_action,
            criterion=criterion,
            step=FLAGS.model.multi_step,
        )
    else:
        raise ValueError(f"{FLAGS.algorithm}")

    loss_mod.build_backward(FLAGS.model.lr, FLAGS.model.weight_decay)
    algo = TRPO(
        vfn=vfn,
        policy=policy,
        dim_state=dim_state,
        dim_action=dim_action,
        **FLAGS.TRPO.as_dict(),
    )

    tf.get_default_session().run(tf.global_variables_initializer())

    runners = {
        "test": make_real_runner(4),
        "collect": make_real_runner(1),
        "dev": make_real_runner(1),
        "train": make_real_runner(FLAGS.plan.n_envs)
        if FLAGS.algorithm == "MF"
        else virt_runner,
    }
    settings = [
        (runners["test"], policy, "Real Env"),
        (runners["train"], policy, "Virt Env"),
    ]

    saver = nn.ModuleDict({"policy": policy, "model": model, "vfn": vfn})
    policy_saver = nn.ModuleDict({"policy": policy})
    vfn_saver = nn.ModuleDict({"vfn": vfn})
    model_saver = nn.ModuleDict({"model": model})
    print(saver)

    if FLAGS.ckpt.model_load:
        assert os.path.isfile(FLAGS.ckpt.model_load), f"{FLAGS.ckpt.model_load}"
        # saver.load_state_dict(np.load(FLAGS.ckpt.model_load)[()])
        model_saver.load_state_dict(
            np.load(FLAGS.ckpt.model_load, allow_pickle=True)[()]
        )
        logger.warning("Load model from %s", FLAGS.ckpt.model_load)

    if FLAGS.ckpt.buf_load:
        n_samples = 0
        for i in range(FLAGS.ckpt.buf_load_index):
            data = pickle.load(
                open(f"{FLAGS.ckpt.buf_load}/stage-{i}.inc-buf.pkl", "rb")
            )
            add_multi_step(data, train_set)
            n_samples += len(data)
        logger.warning("Loading %d samples from %s", n_samples, FLAGS.ckpt.buf_load)

    max_ent_coef = FLAGS.TRPO.ent_coef
    cum_steps = 0

    for T in range(FLAGS.slbo.n_stages):
        logger.info("------ Starting Stage %d --------", T)

        wandb_log_dict = {"cum_steps": cum_steps}
        evaluate(
            settings,
            "episode",
            use_wandb=FLAGS.use_wandb,
            wandb_log_dict=wandb_log_dict,
        )

        if not FLAGS.use_prev:
            train_set.clear()
            dev_set.clear()

        # collect data
        recent_train_set, ep_infos = runners["collect"].run(
            noise.make(policy), FLAGS.rollout.n_train_samples
        )
        cum_steps += recent_train_set.size()
        add_multi_step(recent_train_set, train_set)
        add_multi_step(
            runners["dev"].run(noise.make(policy), FLAGS.rollout.n_dev_samples)[0],
            dev_set,
        )

        returns = np.array([ep_info["return"] for ep_info in ep_infos])
        if len(returns) > 0:
            logger.info("episode: %s", np.mean(returns))

        if T == 0:  # check
            samples = train_set.sample_multi_step(100, 1, FLAGS.model.multi_step)
            for i in range(FLAGS.model.multi_step - 1):
                masks = 1 - (samples.done[i] | samples.timeout[i])[..., np.newaxis]
                assert np.allclose(
                    samples.state[i + 1] * masks, samples.next_state[i] * masks
                )

        # recent_states = obsvs
        # ref_actions = policy.eval('actions_mean actions_std', states=recent_states)
        if (
            FLAGS.rollout.normalizer == "policy"
            or FLAGS.rollout.normalizer == "uniform"
            and T == 0
        ):
            normalizers.state.update(recent_train_set.state)
            normalizers.action.update(recent_train_set.action)
            normalizers.diff.update(
                recent_train_set.next_state - recent_train_set.state
            )

        if T >= 50:
            max_ent_coef = 0.0

        for i in range(FLAGS.slbo.n_iters):
            if i % FLAGS.slbo.n_evaluate_iters == 0 and i != 0:
                # cur_actions = policy.eval('actions_mean actions_std', states=recent_states)
                # kl_old_new = gaussian_kl(*ref_actions, *cur_actions).sum(axis=1).mean()
                # logger.info('KL(old || cur) = %.6f', kl_old_new)
                evaluate(
                    settings,
                    "iteration",
                    wandb_step=WANDB_STEP,
                    use_wandb=FLAGS.use_wandb,
                )

            losses = deque(maxlen=FLAGS.slbo.n_model_iters)
            va_losses = deque(maxlen=FLAGS.slbo.n_model_iters)
            vf_losses = deque(
                maxlen=FLAGS.slbo.n_model_iters
                // max(FLAGS.model.value_update_interval, 1)
            )
            vf_grad_norms = deque(
                maxlen=FLAGS.slbo.n_model_iters
                // max(FLAGS.model.value_update_interval, 1)
            )
            grad_norm_meter = AverageMeter()
            # --------------------------------------------------
            #  Model learning loop
            # --------------------------------------------------
            n_model_iters = FLAGS.slbo.n_model_iters
            for model_iter_idx in range(n_model_iters):
                samples = train_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step
                )
                _, train_loss, grad_norm, va_loss = loss_mod.get_loss(
                    states=samples.state,
                    next_states_=samples.next_state,
                    actions=samples.action,
                    rewards=samples.reward,
                    masks=~samples.done & ~samples.timeout,
                    fetch="train loss grad_norm va_loss",
                )
                losses.append(train_loss.mean())
                va_losses.append(va_loss.mean())
                grad_norm_meter.update(grad_norm)
                # ideally, we should define an Optimizer class, which takes parameters as inputs.
                # The `update` method of `Optimizer` will invalidate all parameters during updates.
                for param in model.parameters():
                    param.invalidate()

                # ------------------------------
                # VA-SLBO: Value-function refit
                # ------------------------------
                if (
                    FLAGS.model.value_update_interval > 0
                    and (model_iter_idx + 1) % FLAGS.model.value_update_interval == 0
                ):
                    runners["train"].reset()
                    data, ep_infos = runners["train"].run(
                        policy, FLAGS.plan.n_trpo_samples
                    )
                    advantages, values = runners["train"].compute_advantage(vfn, data)
                    _, _, vf_loss, vf_grad_norm = algo.train(
                        max_ent_coef, data, advantages, values, skip_policy_update=True,
                    )
                    logger.info(f"Vfn Loss at {model_iter_idx}: {vf_loss:.3g}")

                    vf_losses.append(vf_loss)
                    vf_grad_norms.append(vf_grad_norm)

            # --------------------------------------------------
            #  Model validation
            # --------------------------------------------------
            if i % FLAGS.model.validation_freq == 0:
                samples = train_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step
                )
                loss, va_loss = loss_mod.get_loss(
                    states=samples.state,
                    next_states_=samples.next_state,
                    actions=samples.action,
                    rewards=samples.reward,
                    masks=~samples.done & ~samples.timeout,
                    fetch="loss va_loss",
                )
                loss = loss.mean()
                if np.isnan(loss) or np.isnan(np.mean(losses)):
                    logger.info("nan! %s %s", np.isnan(loss), np.isnan(np.mean(losses)))
                # Prevent nan values being sent to W&B
                if np.isnan(loss):
                    loss = -1.0
                if np.isnan(np.mean(losses)):
                    losses = [-1.0]

                logger.info(
                    "# Iter %3d: Loss = [train = %.3f, dev = %.3f], after %d steps, grad_norm = %.6f",
                    i,
                    np.mean(losses),
                    loss,
                    n_model_iters,
                    grad_norm_meter.get(),
                )
                if FLAGS.use_wandb:
                    wandb.log(
                        {
                            "slbo_n_stage": T,
                            "slbo_n_iter": i,
                            # Train
                            "model_train_loss": np.mean(losses),
                            "model_grad_norm": grad_norm_meter.get(),
                            "model_va_loss": np.mean(va_losses),
                            # Dev
                            "model_dev_train_loss": loss,
                            "model_dev_ma_loss": va_loss,
                            # Value function
                            "model_loop_vf_loss": np.mean(vf_losses),
                            "model_loop_vf_grad_norm": np.mean(vf_grad_norms),
                        },
                    )

            # --------------------------------------------------
            #  Policy update loop
            # --------------------------------------------------
            for policy_iter_idx in range(FLAGS.slbo.n_policy_iters):
                if FLAGS.algorithm != "MF" and FLAGS.slbo.start == "buffer":
                    runners["train"].set_state(
                        train_set.sample(FLAGS.plan.n_envs).state
                    )
                else:
                    runners["train"].reset()

                data, ep_infos = runners["train"].run(policy, FLAGS.plan.n_trpo_samples)
                advantages, values = runners["train"].compute_advantage(vfn, data)
                dist_mean, dist_std, vf_loss, vf_grad_norm = algo.train(
                    max_ent_coef, data, advantages, values
                )
                returns = [info["return"] for info in ep_infos]
                logger.info(
                    "[TRPO] # %d: n_episodes = %d, returns: {mean = %.0f, std = %.0f}, "
                    "dist std = %.10f, dist mean = %.10f, vf_loss = %.3f",
                    policy_iter_idx,
                    len(returns),
                    np.mean(returns),
                    np.std(returns) / np.sqrt(len(returns)),
                    dist_std,
                    dist_mean,
                    vf_loss,
                )

        if T % FLAGS.ckpt.n_save_stages == 0:
            np.save(f"{FLAGS.log_dir}/stage-{T}", saver.state_dict())
            np.save(f"{FLAGS.log_dir}/final", saver.state_dict())
        if FLAGS.ckpt.n_save_stages == 1:
            pickle.dump(
                recent_train_set, open(f"{FLAGS.log_dir}/stage-{T}.inc-buf.pkl", "wb")
            )


if __name__ == "__main__":
    from lunzi.config import parse
    from slbo.utils.flags import FLAGS

    parse(FLAGS)

    FLAGS.set_seed()
    FLAGS.freeze()

    with tf.Session(config=get_tf_config()):
        main(FLAGS)
