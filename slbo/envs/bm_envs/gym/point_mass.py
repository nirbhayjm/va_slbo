from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv

POINTMASS_REWARD_MULTIPLIER = 10 # To make it easier to learn when there are obstacles

 # Taken from https://github.com/deepmind/dm_control/blob/a243ccf3c93f4e6aa2479e461cf935b879f3bb0b/dm_control/utils/rewards.py#L93
def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.
    Returns:
        A float or numpy array with values between 0.0 and 1.0.
    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError('Lower bound must be <= upper bound.')
    if margin < 0:
        raise ValueError('`margin` must be non-negative.')

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, np.exp(-0.5 * (d*np.sqrt(-2 * np.log(value_at_margin)))**2))

    return float(value) if np.isscalar(x) else value



class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self, frame_skip=5, wind = None, walls=False):
        self.prev_qpos = None
        self.wind = wind
        self.walls = walls
        dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = f"{dir_path}/assets/point_mass_with_walls.xml" if  walls else f"{dir_path}/assets/point_mass.xml"
        print(f"Using xml path: {xml_path}")
        mujoco_env.MujocoEnv.__init__(
            self, xml_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

    def _step(self, action):
        if self.wind:
            action = action + self.wind
        
        if getattr(self, "action_space", None):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        ob = self._get_obs()

        target_pos = self.model.data.qpos.flat[-2:]
        pointmass_pos = self.model.data.qpos.flat[:-2]

        # self.model.geom_names gave "target" as second-last entry
        TARGET_IDX = -2
        target_size_arr = self.model.geom_size[TARGET_IDX]
        target_size = target_size_arr[0]  # First coordinate gives radius

        mass_to_target_dist = np.linalg.norm(target_pos-pointmass_pos)

        near_target = tolerance(mass_to_target_dist, bounds=(0,target_size), margin=target_size*POINTMASS_REWARD_MULTIPLIER if (self.walls or self.wind) else target_size)
        control_reward = tolerance(action, margin=1, value_at_margin=0).mean()
        small_control = (control_reward + 4)/5
        reward = near_target * small_control

        # print(f"""
        # target position: {target_pos}
        # pointmass position: {pointmass_pos}
        # qpos: {self.model.data.qpos.flat[:]}
        # qvel: {self.model.data.qvel.flat[:]}
        # action: {action}
        # reward: {reward}\n\n\n""")
        # import pdb; pdb.set_trace()

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.model.data.qpos.flat[:-2],
                self.model.data.qvel.flat[:-2],
            ]
        )

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, "action_space", None):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        rewards = -self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def mb_step_tensor(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, "action_space", None):
            actions = tf.clip_by_value(
                actions, self.action_space.low, self.action_space.high
            )
        rewards = -self.cost_tf_vec(states, actions, next_states)
        return rewards

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.3, high=0.3
        )
        while np.linalg.norm(qpos[:-2]) < 0.2:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.3, high=0.3
            )
        # If we want to randomize goals per episode, uncomment below and set qpos[-2:] = self.goal
        # while True:
        #     self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #     if np.linalg.norm(self.goal) < 0.2:
        #         break
        
        qpos[-2:] = 0 # The goal is always at the origin, according to the paper
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        # self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def cost_np_vec(self, obs, acts, next_obs):
        target_pos = self.model.data.qpos.flat[-2:]
        pointmass_pos = next_obs[:,:-2]
        target_pos = np.repeat(target_pos[:,np.newaxis], pointmass_pos.shape[0], axis=1).T

        # self.model.geom_names gave "target" as second-last entry
        TARGET_IDX = -2
        target_size_arr = self.model.geom_size[TARGET_IDX]
        target_size = target_size_arr[0]  # First coordinate gives radius

        mass_to_target_dist = np.linalg.norm(target_pos-pointmass_pos, axis=1)

        near_target = tolerance(mass_to_target_dist, bounds=(0,target_size), margin=target_size*POINTMASS_REWARD_MULTIPLIER if (self.walls or self.wind) else target_size)
        control_reward = tolerance(acts, margin=1, value_at_margin=0).mean(axis=1)
        small_control = (control_reward + 4)/5
        reward = near_target * small_control
        return -reward

    def cost_tf_vec(self, obs, acts, next_obs):
        target_pos = self.model.data.qpos.flat[-2:]
        pointmass_pos = next_obs[:,:-2]
        target_pos = np.repeat(target_pos[:,np.newaxis], pointmass_pos.shape[0], axis=1).T

        # self.model.geom_names gave "target" as second-last entry
        TARGET_IDX = -2
        target_size_arr = self.model.geom_size[TARGET_IDX]
        target_size = target_size_arr[0]  # First coordinate gives radius

        mass_to_target_dist = np.linalg.norm(target_pos-pointmass_pos, axis=1)

        near_target = tolerance(mass_to_target_dist, bounds=(0,target_size), margin=target_size*POINTMASS_REWARD_MULTIPLIER if (self.walls or self.wind) else target_size)
        control_reward = tolerance(acts, margin=1, value_at_margin=0).mean(axis=1)
        small_control = (control_reward + 4)/5
        reward = near_target * small_control
        return -reward

    # def cost_tf_vec(self, obs, acts, next_obs):
    #     raise NotImplementedError
    #     """
    #     reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
    #     reward_run = next_obs[:, 0]
    #     # reward_height = -3.0 * tf.square(next_obs[:, 0] - 0.57)
    #     reward = reward_run + reward_ctrl # + reward_height
    #     return -reward
    #     """