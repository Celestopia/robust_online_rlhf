# file: policy.py
import numpy as np
from constants import GRID_SIZE, STATE_DIM, ACTION_DIM, EPISODE_LENGTH


class GridPolicy:
    """Grid policy which takes a discrete state and returns a discrete action."""
    def __init__(self, policy_array:np.ndarray):
        self.policy_array = policy_array # Shape: (STATE_DIM, ACTION_DIM). Each element is the probability p(a|s).

    def __call__(self, state:int) -> int:
        return np.random.choice(ACTION_DIM, p=self.policy_array[state])

class GridPolicy_h:
    """Grid policy with time hoziron conditioning (h)."""
    def __init__(self, policy_array:np.ndarray):
        assert type(policy_array) == np.ndarray, "Policy array must be a numpy array."
        assert policy_array.ndim == 3, "Policy array must have 3 dimensions (episode_length, state_dim, action_dim)."
        self.policy_array = policy_array # Shape: (EPISODE_LENGTH, STATE_DIM, ACTION_DIM). Each element is the probability p_h(a|s).

    def __call__(self, h:int, state:int) -> int:
        return np.random.choice(ACTION_DIM, p=self.policy_array[h, state])


def generate_random_policy(state_dim=STATE_DIM, action_dim=ACTION_DIM):
    S, A = state_dim, action_dim
    policy_array = np.random.dirichlet(np.ones(A), size=S).reshape(S, A)
    return policy_array
