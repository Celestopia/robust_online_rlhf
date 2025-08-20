# file: p2r.py
import numpy as np
import logging
from utils.functions import logistic, logistic_derivative
from env import SimulatedOracle


class P2R_Interface:
    """
    P2R interface for (s,a) preferences.
    
    Reference: Is RLHF More Difficult than Standard RL? A Theoretical Perspective,
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/efb9629755e598c4f261c44aeb6fde5e-Abstract-Conference.html
    """
    def __init__(self, state_dim:int, action_dim:int, oracle:SimulatedOracle,
                    m_repeats:int=10000,
                    epsilon_0:float=0.01,
                    reward_abs_range:float=10.0,
                    gamma:float=0.0,
                    robust=True,
        ):
        self.S = state_dim
        self.A = action_dim
        self.m = m_repeats # number of Bernoulli draws per query
        self.epsilon_0 = epsilon_0
        self.reward_abs_range = reward_abs_range
        
        self.alpha = logistic_derivative(reward_abs_range)
        self.beta = epsilon_0 ** 2 / 4

        self.gamma = gamma
        self.robust = robust

        assert type(oracle) == SimulatedOracle, "oracle must be an instance of SimulatedOracle."
        self.oracle = oracle

        self.D_hist = [[[] for _ in range(self.A)] for _ in range(self.S)] # Shape: (S, A)
        self.D = [[[] for _ in range(self.A)] for _ in range(self.S)] # Shape: (S, A)
            # Both D_hist and D store the historical episode-reward pairs, classified by (s,a).
            # For any valid state s and action a, D_hist[s][a] is a list of rewards corresponding to episode (s,a).
            # Note that all rewards in D_hist[s][a] belong to a same episode.
        self.B_r = np.array([[-10.0,10.0]]).repeat(self.S * self.A, axis=0).reshape(self.S, self.A, 2) # Shaped (S, A, 2). Parameterized subset of function class.
            # Let (r_min, r_max) = B_r[s][a], then (r_min, r_max) indicates the range of reward estimate for state s and action a.
            # The reward function has S×A parameters in total (since it's a tabular mapping), so the reward function space is S×A dimensional.
            # B_r defines a box in the S×A dimensional function space. All possible reward functions lie in this subset.
        self.D_hist_count = 0 # Total number of episode-reward pairs in D_hist
        self.D_count = 0 # Total number of episode-reward pairs in D
        self.query_count = 0 # Total number of queries made to the oracle

    def set_reference(self, s0:int, a0:int) -> None:
        """Set the reference state-action pair for reward comparison."""
        self.s0 = s0
        self.a0 = a0

    def get_reward_estimate(self, s:int, a:int) -> float:
        if len(self.D_hist[s][a]) > 0: # If (s, a) already appeared, directly use historical estimate.
            r_hat = np.mean(self.D_hist[s][a])
            #logging.info("(x, y, a)=({}, {}, {}). Directly return historical reward estimate: {:.3f}".format(x, y, a, r_hat))
            return r_hat
        
        r_min, r_max = self.B_r[s][a]
        if r_max - r_min < 2 * self.epsilon_0:
            #logging.info("(x, y, a)=({}, {}, {}). Return mid-point estimate: {:.3f}".format(x, y, a, (r_min + r_max) / 2.0))
            r_hat = (r_min + r_max) / 2.0
            self.D_hist[s][a].append(r_hat)
            self.D_hist_count += 1
            return r_hat
        else:
            self.query_count += 1
            #logging.info("(x, y, a)=({}, {}, {}). Query triggered.".format(x, y, a))
            p = self.oracle.compare(s, a, self.s0, self.a0, n_repeats=self.m)
            #logging.info("Probability of (x, y, a)=({}, {}, {}) being preferred: {:.3f}".format(x, y, a, p))
            logit = np.log(p / (1.0 - p)) / max(1e-9, self.oracle.sigma_scale)
            r_hat = np.clip(logit, -self.reward_abs_range, self.reward_abs_range)
            if self.gamma > 0.0 and self.robust: # Robust estimate under label flipping attack
                r_hat = (r_hat - self.gamma) / (1.0 - 2 * self.gamma)
            #logging.info("(x, y, a)=({}, {}, {}). Estimated reward: {:.3f}".format(x, y, a, r_hat))
            self.D[s][a].append(r_hat)
            self.D_hist[s][a].append(r_hat)
            self.D_count += 1
            self.D_hist_count += 1
            
            # Update B_r
            r_diff = np.sqrt(self.beta / self.D_count)
            for s_ in range(self.S):
                for a_ in range(self.A):
                    if len(self.D[s_][a_]) > 0:
                        r_min = np.mean(self.D[s_][a_]) - r_diff
                        r_max = np.mean(self.D[s_][a_]) + r_diff
                        self.B_r[s_][a_] = (r_min, r_max)
            
            return r_hat

    def __call__(self, s:int, a:int) -> float:
        return self.get_reward_estimate(s, a)

