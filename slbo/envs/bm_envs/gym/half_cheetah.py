# from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self, xml="", frame_skip=5):
        self.prev_qpos = None
        if xml:
            mujoco_env.MujocoEnv.__init__(self, xml, frame_skip=frame_skip)
            self.xml = xml
        else:
            dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mujoco_env.MujocoEnv.__init__(
                self, f"{dir_path}/assets/half_cheetah.xml", frame_skip=frame_skip
            )
            self.xml = f"{dir_path}/assets/half_cheetah.xml"
        utils.EzPickle.__init__(self)

    def _step(self, action):
        start_ob = self._get_obs()
        reward_run = start_ob[8]

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if getattr(self, "action_space", None):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        reward_ctrl = -0.1 * np.square(action).sum()

        reward = reward_run + reward_ctrl
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.model.data.qpos.flat[1:], self.model.data.qvel.flat,]
        )

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, "action_space", None):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        rewards = -self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def mb_step_tensor(self, states, actions, next_states):
        if getattr(self, "action_space", None):
            actions = tf.clip_by_value(
                actions, self.action_space.low, self.action_space.high
            )
        rewards = -self.cost_tf_vec(states, actions, next_states)
        return rewards

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def cost_np_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 8]
        reward = reward_run + reward_ctrl
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = obs[:, 8]
        reward = reward_run + reward_ctrl
        return -reward

    # def cost_tf_vec(self, obs, acts, next_obs):
    #     raise NotImplementedError
    #     """
    #     reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
    #     reward_run = next_obs[:, 0]
    #     reward = reward_run + reward_ctrl
    #     return -reward
    #     """
