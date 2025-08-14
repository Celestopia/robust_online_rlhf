# file: ucbvi.py
import numpy as np
import random
import math
import logging
from typing import Callable, Tuple, List
from constants import GRID_SIZE, STATE_DIM, ACTION_DIM, EPISODE_LENGTH
from utils import get_variance



class UCBVI_BF:
    """
    Upper Confidence Bound Value Iteration (UCBVI) with Bernstein-Freedman bonus.
    
    Reference: Minimax Regret Bounds for Reinforcement Learning, https://proceedings.mlr.press/v70/azar17a.
    """
    def __init__(self, state_dim:int, action_dim:int, episode_num:int, episode_length:int, delta:float=0.1):
        assert type(state_dim) == int, "state_dim must be an integer"
        assert type(action_dim) == int, "action_dim must be an integer"
        assert type(episode_num) == int, "episode_num must be an integer"
        assert type(episode_length) == int, "episode_length must be an integer"
        assert type(delta) == float and 0.001 <= delta <= 1.0, "delta must be a float in (0.001, 1.0]"

        self.S = state_dim
        self.A = action_dim
        self.K = episode_num
        self.H = episode_length
        self.T = self.K * self.H
        self.delta = delta
        self.L = np.log(5 * self.S * self.A * self.T / delta)
        
        self.count_sa = np.zeros((self.S, self.A), dtype=int) # Total (s, a) visits
        self.count_sas = np.zeros((self.S, self.A, self.S), dtype=int) # Total (s, a, s') transitions
        self.count_hsa = np.zeros((self.H, self.S, self.A), dtype=int) # Total visits to (s, a) at step h
        self.count_hs = np.zeros((self.H, self.S), dtype=int) # Total visits to state s at step h

        self.Q = np.ones((self.H, self.S, self.A), dtype=float) # Q-values for each (h, s, a), to be implemented

    def update_with_episode(self, episode_tuples:List[Tuple[int, int, int, int, float]]):
        """
        Update the history counts with episode tuples.

        Args:
            episode_tuples (list of tuples): A list of (h, s, a, s', r) for each step
        """
        for (h, s, a, s_, _) in episode_tuples:
            self.count_sa[s, a] += 1
            self.count_sas[s, a, s_] += 1
            self.count_hsa[h, s, a] += 1
            self.count_hs[h, s] += 1

    def get_P_hat(self, s:int, a:int) -> np.ndarray:
        """Empirical estimate of P(y|s,a), the probability of transitioning to state `y` from state `x` given action `a`."""
        n = self.count_sa[s, a]
        if n == 0: # If (s, a) is unseen
            return np.ones(self.S) / self.S # Use uniform prior 
        return self.count_sas[s, a] / n # Shape: (S,)

    def bonus_2(self, h:int, s:int, a:int, V_next:np.ndarray) -> float:
        # V_next: numpy array of length S for V_{h+1}
        N_k = self.count_sa[s, a] # Total visits to (s,a)
        P_hat = self.get_P_hat(s, a) # Shape: (S,)
        
        var = get_variance(V_next, P_hat) # Empirical variance of V_next under probability distribution P_hat
        term1 = math.sqrt((8.0 * self.L * var) / N_k)
        term2 = (14.0 * self.H * self.L) / (3.0 * N_k)
        term3 = 0

        if h + 1 < self.H: # Only for non-terminal next states
            c = (100 ** 2) * (self.H ** 3) * (self.S ** 2) * (self.A) * (self.L ** 2)
            inner_sum = 0.0

            for next_s in range(self.S):
                p = P_hat[next_s]
                N_k_h1 = self.count_hs[h + 1, next_s] if h + 1 < self.H else 0 # Get visits to next state at step h+1
                inner = min(c / N_k_h1, self.H ** 2) if N_k_h1 > 0 else self.H ** 2 # Handle unvisited states (use higher bonus)
                inner_sum += p * inner

            term3 = np.sqrt(8 * inner_sum / N_k) if N_k > 0 else 0

        b = term1 + term2 + term3
        return b

    def ucb_q_values(self, reward_estimator:Callable[[int, int], float]) -> np.ndarray:
        """
        Run backward value-iteration for H steps to get optimistic Q values.
        
        Args:
            reward_estimator: A function that takes (s, a) and returns a scalar reward.
        
        Returns:
            Q (np.ndarray): Shaped (H, S, A), representing the learned Q-values. This is the undiscounted expected cumulative reward.
        """
        H, S, A = self.H, self.S, self.A
        V = np.zeros((H + 1, S), dtype=float)
        Q = np.zeros((H, S, A), dtype=float)
        Q_prev = self.Q.copy() # for computing bonus term
        
        #logging.info("UCBVI-BF: Running backward value iteration to update Q values.")
        for h in reversed(range(H)): # H-1, H-2,..., 0
            for s in range(S):
                for a in range(A):
                    if self.count_sa[s,a] > 0:
                        r_hat = reward_estimator(s, a)
                        P_hat = self.get_P_hat(s, a)
                        expected_v = P_hat.dot(V[h+1]) # expected value of next state
                        b = self.bonus_2(h, s, a, V[h+1]) # bonus term
                        q = r_hat + expected_v + b
                        q = min(Q_prev[h,s,a], H, q) # may replace H with H-h for a tighter bound
                    else:
                        q = self.H
                    Q[h, s, a] = q
                V[h, s] = float(Q[h, s, :].max())
        self.Q = Q
        return Q

    def extract_policy(self, Q_table:np.ndarray, epsilon:float=0.0) -> np.ndarray: # ε-greedy policy from Q-values
        assert 0 <= epsilon < 1
        H, S, A = self.H, self.S, self.A
        policy_table = np.zeros((H, S, A), dtype=float)
        for h in range(H):
            for s in range(S):
                a_opt = np.where(Q_table[h, s] == np.max(Q_table[h, s]))[0]
                for a in range(A):
                    if a in a_opt:
                        policy_table[h, s, a] = (1 - epsilon)/len(a_opt) + epsilon/A
                    else:
                        policy_table[h, s, a] = epsilon/A
        return policy_table
