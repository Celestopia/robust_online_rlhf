# file: env.py
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Tuple
from constants import GRID_SIZE, ACTION_DIM
from utils.functions import logistic

class GridWorld:
    """
    A square grid world environment.
    
    The API is inspired by Gymnasium Documentation, https://gymnasium.farama.org/introduction/create_custom_env/.
    """
    def __init__(self, size:int=11, p:float=0.9, trajectory_length_max:int=100):
        """
        Initialize the grid world.

        Attributes:
            size (int): The side length of the grid.
            _state (int): The current state of the agent.
            _p (float): The probability of taking the intended action.
            trajectory_length (int): The number of steps taken in the current episode.
            trajectory_length_max (int): The maximum number of steps in an episode.
        """
        self.size = size
        self._state = self.position_to_state((0,0))
        self.p = p
        self.trajectory_length = 0
        self.trajectory_length_max = trajectory_length_max
        self.reward_model = lambda s, a: 0 # To be implemented later.

    def set_reward_model(self, reward_model:Callable[[int, int], float]):
        """
        Args:
            reward_model (Callable): A Callable that takes a state-action pair (s, a) and returns a scalar reward.
        """
        self.reward_model = reward_model

    def _check_state(self, state):
        """Check the format validity of the state."""
        assert type(state) == int, "State must be an integer; got {}.".format(type(state))
        assert 0 <= state < self.size ** 2, "State must be in range [0, {}^2); got {}.".format(self.size, state)
    
    def _check_position(self, position):
        """Check the format validity of the position."""
        assert len(position)==2, "Position must be an iterable with two integer entries."
        (x, y) = position
        assert type(x)==int and type(y)==int, "Position must be an iterable with integer entries."
        n = self.size // 2
        assert -n <= x <= n and -n <= y <= n, "Position must be within the grid [{}, {}] x [{}, {}]; got ({}, {}).".format(-n, n, -n, n, x, y)

    def _check_action(self, action):
        """Check the format validity of the action."""
        assert type(action) == int, "Action must be an integer; got {}.".format(type(action))
        assert 0 <= action <= 4, "Action must be in range [0, 4]; got {}.".format(action)

    def position_to_state(self, position:Tuple[int, int]) -> int:
        """
        Convert a position (x, y) to a numbered state.
        The left top corner corresponds to state 0. The number increases by 1 for each column and by `self.size` for each row.

        (-2, 2) (-1, 2) (0, 2) (1, 2) (2, 2)      0  1  2  3  4
        (-2, 1) (-1, 1) (0, 1) (1, 1) (2, 1)      5  6  7  8  9
        (-2, 0) (-1, 0) (0, 0) (1, 0) (2, 0)  ->  10 11 12 13 14
        (-2,-1) (-1,-1) (0,-1) (1,-1) (2,-1)      15 16 17 18 19
        (-2,-2) (-1,-2) (0,-2) (1,-2) (2,-2)      20 21 22 23 24
        
        Args:
            position (tuple of int): A tuple with two integer entries, representing the coordinate in the grid world, e.g., (3, 4) or (-1, -3).
        
        Returns:
            state (int): The number representation of the state.
        """
        (x, y) = position
        n = self.size // 2
        return (n + y) * (2 * n + 1) + x + n

    def state_to_position(self, state:int) -> Tuple[int, int]:
        """Convert a numbered state to a position (x, y)."""
        n = self.size // 2
        x = state % (2 * n + 1) - n
        y = (state - x) // (2 * n + 1) - n
        return (x, y)

    def action_to_direction(self, action:int) -> Tuple[int, int]:
        """Convert a numbered action to a direction (delta_x, delta_y)."""
        map = {
            0: (0,0), # Stay
            1: (1,0), # Move right
            2: (0,1), # Move up
            3: (-1,0), # Move left
            4: (0,-1), # Move down
        }
        return map[action]

    def action_to_char(self, action:int) -> str:
        action_to_char_dict = {0: "o", 1: "→", 2: "↑", 3: "←", 4: "↓"}
        return action_to_char_dict[action]

    def get_state(self) -> int:
        """Return a copy of the current state."""
        return copy.deepcopy(self._state)

    def get_position(self) -> Tuple[int, int]:
        """Return the current position as a tuple (x, y)."""
        return self.state_to_position(self._state)

    def reset(self, position:Tuple[int, int]=None) -> int:
        """Reset the environment to a initial state."""
        if position is None:
            self._state = self.position_to_state((0,0))
        else:
            self._check_position(position)
            self._state = self.position_to_state(position)
        self.trajectory_length = 0
        return self.get_state()
    
    def _update_state(self, state:int):
        """Update the current state with the given state."""
        self._state = state

    def step(self, action:int, frozen_state:bool=False) -> Tuple[int, float, bool, bool, dict]:
        self._check_action(action)

        terminated = False # Whether the agent reaches the target or hit the wall
        truncated = False # Whether the episode is truncated due to time limit
        info = {
            "trajectory_length": None,
            "position": None,
            "action_taken": None,
            "messages": [],
        } # Additional information

        if 0 < random.random() < self.p: # With probability p, take the intended action.
            action_taken = action
        else:
            action_taken = random.randint(0, 4) # With probability 1-p, take a random action.
            info["messages"].append("Agent took a random action. The chosen action is: {}.".format(action_taken))
            
        state_prev = self.get_state()
        (x_prev, y_prev) = self.state_to_position(state_prev)
        (x_delta, y_delta) = self.action_to_direction(action_taken)
        (x_new, y_new) = (x_prev + x_delta, y_prev + y_delta)
        state_new = self.position_to_state((x_new, y_new))
        n = self.size // 2

        if not (np.abs(x_new) > n or np.abs(y_new) > n): # Normal case, where there is no hit
            self._update_state(state_new)
            reward = self.reward_model(state_prev, action)
        else: # If the agent moves out of the grid
            reward = -0.5 # Penalty for taking a wrong action
            terminated = True # Terminate if the agent hits the wall
            state_new = state_prev # Keep the previous state
            info["messages"].append("Agent hit the wall! Intended position: ({}, {}); current position: ({}, {}).".format(
                x_new, y_new, x_prev, y_prev))
        
        if frozen_state is False: # Normal case, where the state is updated after the action
            self.trajectory_length += 1
            if self.trajectory_length >= self.trajectory_length_max:
                truncated = True
                info["messages"].append("Episode truncated at step {}.".format(self.trajectory_length))
        elif frozen_state is True: # Do not really update the state, just return signals
            terminated = False
            truncated = False
            state_new = state_prev # Keep the previous state
            self._update_state(state_new)
            info["messages"].append("Took a trial action. The state is not updated.")
        
        info["trajectory_length"] = self.trajectory_length
        info["position"] = self.state_to_position(state_new)
        info["action_taken"] = action_taken
        return self.get_state(), reward, terminated, truncated, info


class GridReward:
    def __init__(self, reward_table:np.ndarray):
        """
        Define a reward function using table values.

        Args:
            reward_table (np.ndarray): A 2-d array of shape (state_dim, action_dim) representing the reward for each state-action pair.
        """
        self.reward_table = reward_table

    def __call__(self, state:int, action:int) -> float:
        return self.reward_table[state, action]


class SimulatedOracle:
    """Simulated (s, a) preference."""
    def __init__(self, reward_model:Callable[[int, int], float], sigma_scale:float=1.0, gamma:float=0.0):
        self.reward_model = reward_model
        self.sigma_scale = sigma_scale
        self.gamma = gamma

    def compare(self, s1:int, a1:int, s0:int, a0:int, simulate:bool=False, n_repeats:int=10000) -> float:
        """Give the preference of a state-action pair over another."""
        r1 = self.reward_model(s1, a1)
        r0 = self.reward_model(s0, a0)
        p = logistic(r1 - r0, scale=self.sigma_scale)
        if not simulate: # If not running Bernoulli experiments
            if self.gamma > 0:
                return p + self.gamma - 2 * p * self.gamma # Return the preference probability with noise
            else:
                return p # Directly return the preference probability
        else: # Simulate multiple human comparisons
            samples = np.random.binomial(1, p, size=(n_repeats,)) # Repeated Bernoulli draws
            if self.gamma > 0:
                flip = np.random.binomial(1, self.gamma, size=(n_repeats,))
                samples = np.logical_xor(samples, flip).astype(int) # Flip the samples with probability gamma
            return samples.mean() # Return average preference probability


