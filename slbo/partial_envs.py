# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from slbo.envs.bm_envs.gym import (
    gym_cartpoleO01,
    gym_cartpoleO001,
    gym_cheetahA01,
    gym_cheetahA003,
    gym_cheetahO01,
    gym_cheetahO001,
    gym_fant,
    gym_fhopper,
    gym_fswimmer,
    gym_fwalker2d,
    gym_humanoid,
    gym_nostopslimhumanoid,
    gym_pendulumO01,
    gym_pendulumO001,
    gym_slimhumanoid,
)
from slbo.envs.bm_envs.gym.acrobot import AcrobotEnv
from slbo.envs.bm_envs.gym.ant import AntEnv
from slbo.envs.bm_envs.gym.cartpole import CartPoleEnv
from slbo.envs.bm_envs.gym.half_cheetah import HalfCheetahEnv
from slbo.envs.bm_envs.gym.hopper import HopperEnv
from slbo.envs.bm_envs.gym.inverted_pendulum import InvertedPendulumEnv
from slbo.envs.bm_envs.gym.mountain_car import Continuous_MountainCarEnv
from slbo.envs.bm_envs.gym.pendulum import PendulumEnv

# from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.bm_envs.gym.point_mass import PointMassEnv
from slbo.envs.bm_envs.gym.reacher import ReacherEnv
from slbo.envs.bm_envs.gym.swimmer import SwimmerEnv
from slbo.envs.bm_envs.gym.walker2d import Walker2dEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.mujoco.swimmer_env import SwimmerEnv
from slbo.envs.mujoco.walker2d_env import Walker2DEnv


def make_env(id: str, env_params=None):
    envs = {
        "HalfCheetah": HalfCheetahEnv,
        "Walker2D": Walker2DEnv,
        "Ant": AntEnv,
        "PointMass": PointMassEnv,
        "Hopper": HopperEnv,
        "Swimmer": SwimmerEnv,
        "FixedSwimmer": gym_fswimmer.fixedSwimmerEnv,
        "FixedWalker": gym_fwalker2d.Walker2dEnv,
        "FixedHopper": gym_fhopper.HopperEnv,
        "FixedAnt": gym_fant.AntEnv,
        "Reacher": ReacherEnv,
        "Pendulum": PendulumEnv,
        "InvertedPendulum": InvertedPendulumEnv,
        "Acrobot": AcrobotEnv,
        "CartPole": CartPoleEnv,
        "MountainCar": Continuous_MountainCarEnv,
        "HalfCheetahO01": gym_cheetahO01.HalfCheetahEnv,
        "HalfCheetahO001": gym_cheetahO001.HalfCheetahEnv,
        "HalfCheetahA01": gym_cheetahA01.HalfCheetahEnv,
        "HalfCheetahA003": gym_cheetahA003.HalfCheetahEnv,
        "PendulumO01": gym_pendulumO01.PendulumEnv,
        "PendulumO001": gym_pendulumO001.PendulumEnv,
        "CartPoleO01": gym_cartpoleO01.CartPoleEnv,
        "CartPoleO001": gym_cartpoleO001.CartPoleEnv,
        "gym_humanoid": gym_humanoid.HumanoidEnv,
        "gym_slimhumanoid": gym_slimhumanoid.HumanoidEnv,
        "gym_nostopslimhumanoid": gym_nostopslimhumanoid.HumanoidEnv,
        "HalfCheetah-v2": HalfCheetahEnv,
        "Walker2D-v2": Walker2DEnv,
        "Humanoid-v2": HumanoidEnv,
        "Ant-v2": AntEnv,
        "Hopper-v2": HopperEnv,
        "Swimmer-v2": SwimmerEnv,
    }
    if env_params:
        env = envs[id](**env_params)
    else:
        env = envs[id]()
    if not hasattr(env, "reward_range"):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, "metadata"):
        env.metadata = {}
    env.seed(np.random.randint(2 ** 60))
    return env
