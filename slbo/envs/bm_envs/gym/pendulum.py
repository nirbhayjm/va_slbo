from os import path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from lunzi.Logger import logger
from slbo.utils.dataset import Dataset, gen_dtype


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.viewer = None

        high = np.array([1.0, 1.0, self.max_speed])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,)
        )
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        th, thdot = self.state  # th := theta
        """
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
        """

        # for the reward
        y, x, thetadot = np.cos(th), np.sin(th), thdot
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = y + 0.1 * np.abs(x) + 0.1 * (thetadot ** 2) + 0.001 * (u ** 2)
        reward = -costs

        g = 10.0
        m = 1.0
        l = 1.0
        dt = self.dt

        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(
            newthdot, -self.max_speed, self.max_speed
        )  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), reward, False, {}

    def _reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, "action_space", None):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        rewards = -self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def cost_np_vec(self, obs, acts, next_obs):
        """
        dist_vec = obs[:, -3:]
        reward_dist = - np.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - np.sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl

        # for the reward
        y, x, thetadot = np.cos(th), np.sin(th), thdot
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = y + .1 * x + .1 * (thetadot ** 2) + .001 * (u ** 2)
        reward = -costs

        def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

        """
        y, x, thetadot = obs[:, 0], obs[:, 1], obs[:, 2]
        u = np.clip(acts[:, 0], -self.max_torque, self.max_torque)
        costs = y + 0.1 * np.abs(x) + 0.1 * (thetadot ** 2) + 0.001 * (u ** 2)
        return costs

    def verify(self, n=2000, eps=1e-4):
        dataset = Dataset(gen_dtype(self, "state action next_state reward done"), n)
        state = self.reset()
        for _ in range(n):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)
            dataset.append((state, action, next_state, reward, done))

            state = next_state
            if done:
                state = self.reset()

        rewards_, dones_ = self.mb_step(
            dataset.state, dataset.action, dataset.next_state
        )
        diff = dataset.reward - rewards_
        l_inf = np.abs(diff).max()
        logger.info("rewarder difference: %.6f", l_inf)

        assert np.allclose(dones_, dataset.done)
        assert l_inf < eps


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi