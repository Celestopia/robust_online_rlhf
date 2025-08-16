# file: policy.py
import numpy as np
from constants import STATE_DIM, ACTION_DIM


class GridPolicy:
    """Grid policy which takes a discrete state and returns a discrete action."""
    def __init__(self, policy_array:np.ndarray):
        """
        Args:
            policy_array (np.ndarray): Shaped (S, A). Each element represents `π(a|s)`,
                the probability of taking action `a` at state `s`.
        """
        assert type(policy_array) == np.ndarray, "Policy array must be a numpy array."
        assert policy_array.ndim == 2, "Policy array must have 2 dimensions (state_dim, action_dim)."
        self.policy_array = policy_array
        self.S, self.A = policy_array.shape

    def __call__(self, state:int) -> int:
        return np.random.choice(self.A, p=self.policy_array[state])

class GridPolicy_h:
    """Grid policy which takes a discrete state and returns a discrete action, with time step conditioning (h)."""
    def __init__(self, policy_array:np.ndarray):
        """
        Args:
            policy_array (np.ndarray): Shaped (H, S, A). Each element represents `π_h(a|s)`,
                the probability of taking action `a` at state `s` and step `h`.
        """
        assert type(policy_array) == np.ndarray, "Policy array must be a numpy array."
        assert policy_array.ndim == 3, "Policy array must have 3 dimensions (trajectory_length, state_dim, action_dim)."
        self.policy_array = policy_array
        self.H, self.S, self.A = policy_array.shape

    def __call__(self, h:int, state:int) -> int:
        return np.random.choice(self.A, p=self.policy_array[h, state])


def generate_random_policy(state_dim=STATE_DIM, action_dim=ACTION_DIM): # not used
    S, A = state_dim, action_dim
    policy_array = np.random.dirichlet(np.ones(A), size=S).reshape(S, A)
    return policy_array
