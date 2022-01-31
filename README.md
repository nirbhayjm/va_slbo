
# Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice

Code for the SLBO experiments in [Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice](https://arxiv.org/abs/2106.14080). This repository is a clone of [slbo](https://github.com/facebookresearch/slbo).


## Install

The experiments in this paper were specifically ran with Python 3.6, TensorFlow version `tensorflow-gpu==1.13.1` and uses Weights and Biases](https://wandb.ai/) for tracking experiments on top of slbo. For installation in a new conda environment, run the following commands.

```
conda create -n va_slbo python=3.6
conda activate va_slbo
conda install tensorflow-gpu==1.13.1
pip install -r rllab_requirements.txt
pip install -r va_slbo_requirements.txt
```

[Rllab](https://github.com/rll/rllab) at commit `b3a2899` needs to be installed. This may be done by locally cloning the repo, switching to commit `b3a2899` and using `pip install -e .` inside the repo. It may require a copy of the MuJoCo license to be present at `rllab/vendor/mujoco/mjkey.txt`.

Further, the conda environment file `va_slbo.yaml` is provided as reference for the package list obtained after the full install.

## Run

Make the `experiments` directory for logging.

```
mkdir experiments
```

The following command runs SLBO with a selectedvalue-aware model learning objective.
The `algorithm` flag has three options - `slbo` for the default SLBO algorithm, `mle` for the SLBO ablation with just MLE in the model learning objective and `va` for value-aware loss as the solve model learning loss. For selecting the type of value-aware loss function, `model.va_norm` may be set to either `l1` or `l2` (defaults to `l1` when unspecified).

```
# $ALGO can be set to "slbo", "mle" or "va"
ALGO=va
python slbo/main.py \
    --config \
        configs/algos/${ALGO}.yml \
        configs/env_tingwu/gym_cheetah.yml \
    --set \
        algorithm=${ALGO} \
        model.va_norm=l1 \
        model.value_update_interval=20 \
        seed=9553987 \
        log_dir=experiments/${ALGO}_halfcheetah \
        run_id=va_halfcheetah
```

If tracking experiments with Weights and Biases (wandb), edit `slbo/utils/flags.py` and set the `wandb_project_name` variable. The following command activates wandb for tracking with the `use_wandb=1` option, in the HalfCheetah environment for the default SLBO algorithm.

```
ALGO=slbo
python slbo/main.py \
    --config \
        configs/algos/${ALGO}.yml \
        configs/env_tingwu/gym_cheetah.yml \
    --set \
        algorithm=${ALGO} \
        use_wandb=1 \
        log_dir=experiments/${ALGO}_halfcheetah \
        run_id=my_wandb_experiment
```

### Additional hyper-parameters for value-aware losses

1. `model.va_loss_coeff` (default `0.01`): Sets the scaling coefficient for value-aware model learning losses.
1. `model.value_update_interval` (default `0`): Sets the interval for number of model updates in between value function refitting in the model learning loop of training. Setting this to `0` deactivates value network refitting within model learning. The recommended value for using value-aware objectives is `20`.
