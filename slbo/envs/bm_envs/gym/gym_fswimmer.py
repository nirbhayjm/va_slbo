from __future__ import absolute_import, division, print_function

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class fixedSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, f"{dir_path}/assets/fixed_swimmer.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        ctrl_cost_coeff = 0.0001

        """
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        """

        self.xposbefore = self.model.data.site_xpos[0][0] / self.dt
        self.do_simulation(a, self.frame_skip)
        self.xposafter = self.model.data.site_xpos[0][0] / self.dt
        self.pos_diff = self.xposafter - self.xposbefore

        reward_fwd = self.xposafter - self.xposbefore
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat, self.pos_diff.flat])

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, "action_space", None):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        rewards = -self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nv),
        )
        return self._get_obs()

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.0001 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, -1]
        reward = reward_run + reward_ctrl
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        raise NotImplementedError

    def verify(self):
        pass
