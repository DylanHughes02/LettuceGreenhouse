from aml_project.environments.lettuce_greenhouse import LettuceGreenhouse
from aml_project.common.helper_functions import DefineParameters, co2dens2ppm, weather2ppmrh, vaporDens2rh, denormalise_array, load_disturbances
from aml_project.common.plot_gh_variables import create_trajectory_figure, plot_trajectory
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AblatedLettuceGreenhouse(LettuceGreenhouse):

    # Simple daytime heuristics for each actuator (in NORMALISED [-1, 1] space)
    HEURISTIC = {
        0: lambda d: 0.5,                              # Inject CO2 at a constant moderate rate during the day, no injection at night
        1: lambda d: 1.0 if d[0] > 10 else -1.0,      # Ventilate if outdoor temp > 10C, else keep closed
        2: lambda d: float(np.clip((10 - d[2]) / 10,  # Heat when outdoor temp < 10C, stronger heating the colder it is, no heating at >=10C
                                -1.0, 1.0)),
    }

    def __init__(self, active_actions: list, **kwargs):
        super().__init__(**kwargs)
        self.active_actions = active_actions
        self.heuristic_actions = [i for i in range(3) if i not in active_actions]

        # Shrink the action space to only the RL-controlled dims
        n = len(active_actions)
        self.action_space = spaces.Box(
            low=-np.ones(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32)
        )

    def initialise_action(self):
        """
        Override to ensure the base environment always tracks the 3 physical actuators
        for internal math (like delta clips), regardless of the RL agent's action space.
        """
        return denormalise_array(-np.ones(3, dtype=np.float32), self.min_action, self.max_action)

    def step(self, action: np.ndarray):
        # Build the full 3-dim action by merging RL action + heuristics
        full_action = np.zeros(3, dtype=np.float32)
        d_now = self.d[self.timestep]  # current weather disturbance

        for i, idx in enumerate(self.active_actions):
            full_action[idx] = action[i]
        for idx in self.heuristic_actions:
            full_action[idx] = self.HEURISTIC[idx](d_now)

        if len(self.active_actions) == 0:
            full_action = np.array([self.HEURISTIC[i](d_now) for i in range(3)],
                           dtype=np.float32)
        return super().step(full_action)