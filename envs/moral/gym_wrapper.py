import envs.moral.randomized_v3
from pycolab import rendering
from typing import Callable
import gym
from gym import spaces
from gym.utils import seeding
import copy
import numpy as np
import time

from stable_baselines3.common.utils import set_random_seed


class GymWrapper(gym.Env):
    """Gym wrapper for pycolab environment"""

    def __init__(self, env_id):
        self.env_id = env_id

        if env_id == 'randomized_v3':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9

        self.game = None
        self.np_random = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height, len(self.layers)),
            dtype=np.int32
        )

        self.renderer = rendering.ObservationToFeatureArray(self.layers)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _obs_to_np_array(self, obs):
        return copy.copy(self.renderer(obs))

    def reset(self):
        if self.env_id == 'randomized_v3':
            self.game = envs.moral.randomized_v3.make_game()
        obs, _, _ = self.game.its_showtime()
        # print(obs.board)
        return self._obs_to_np_array(obs)

    def step(self, action):
        obs, reward, _ = self.game.play(action)
        # print(obs.board)
        # print(obs.board[1:10, 1:10])
        return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

    def step_demo(self, action):
        obs, reward, discount = self.game.play(action)
        return obs, reward, discount, self._obs_to_np_array(obs), self.game.game_over, self.game.the_plot

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = GymWrapper(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init