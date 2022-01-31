from __future__ import absolute_import, division, print_function

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self, frame_skip=4):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mujoco_env.MujocoEnv.__init__(
            self, f"{dir_path}/assets/hopper.xml", frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def _step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        if getattr(self, "action_space", None):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[5]
        reward_height = -3.0 * np.square(old_ob[0] - 1.3)
        height, ang = ob[0], ob[1]
        done = (height <= 0.7) or (abs(ang) >= 0.2)
        alive_reward = float(not done)
        reward = reward_run + reward_ctrl + reward_height + alive_reward

        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.model.data.qpos.flat[1:], self.model.data.qvel.flat,]
        )

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 5]
        reward_height = -3.0 * np.square(obs[:, 0] - 1.3)
        height, ang = next_obs[:, 0], next_obs[:, 1]
        done = np.logical_or(height <= 0.7, abs(ang) >= 0.2)
        alive_reward = 1.0 - np.array(done, dtype=np.float)
        reward = reward_run + reward_ctrl + reward_height + alive_reward
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        """
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 0]
        # reward_height = -3.0 * tf.square(next_obs[:, 1] - 1.3)
        reward = reward_run + reward_ctrl
        return -reward
        """
        raise NotImplementedError

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, "action_space", None):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        rewards = -self.cost_np_vec(states, actions, next_states)
        height, ang = next_states[:, 0], next_states[:, 1]
        done = np.logical_or(height <= 0.7, abs(ang) >= 0.2)
        return rewards, done
