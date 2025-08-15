# file: main.py
import os
import sys
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
import traceback
import argparse
import yaml


from env import GridWorld, SimulatedOracle, GridReward
from policy import GridPolicy, GridPolicy_h
from p2r import P2R_Interface
from ucbvi import UCBVI_BF
from utils.log import get_logger, close_logger
from utils.functions import moving_average_smoothing

from constants import GRID_SIZE, STATE_DIM, ACTION_DIM, TRAJECTORY_LENGTH, REWARD_VEC, EPSILON_0



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--trajectory_num', type=int, default=1000, help='Number of trajectories')
    parser.add_argument('--trajectory_length', type=int, default=TRAJECTORY_LENGTH, help='Length of each trajectory')
    parser.add_argument('--policy_epsilon', type=float, default=0.0, help='Epsilon-greedy exploration parameter') # not used
    parser.add_argument('--policy_epsilon_decay', type=float, default=0.9995, help='Decay rate of epsilon-greedy exploration parameter') # not used
    parser.add_argument('--comparison_gamma', type=float, default=0.0, help='Noise level of comparison oracle')
    parser.add_argument('--ucbvi_delta', type=float, default=0.1, help='UCBVI delta parameter')
    parser.add_argument('--robust', action='store_true', help='Use robust P2R')

    args = parser.parse_args()
    return args



def main(args):

    try:

        start_time = time.time()

        # Set random seed
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Set hyperparameters
        grid_size = GRID_SIZE
        state_dim = STATE_DIM
        action_dim = ACTION_DIM
        trajectory_num = args.trajectory_num
        trajectory_length = args.trajectory_length
        epsilon_0=EPSILON_0

        # Set up the directory to store all experiments
        result_dir = 'results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
            print("Result directory created at {}.".format(result_dir))

        # Set up the directory of the current experiment
        time_string = str(int(time.time())) # Time string to identify the current experiment
        experiment_dir = os.path.join(result_dir,
                                    'experiment_{}_{}_{}_{}_{}'.format(
                                        time_string,
                                        state_dim,
                                        action_dim,
                                        trajectory_num,
                                        trajectory_length,
                                        ))
        if args.comparison_gamma != 0.0:
            experiment_dir += '_gamma={}'.format(args.comparison_gamma)
        if args.robust is True:
            experiment_dir += '_robust'
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print("Experiment directory created at {}.".format(experiment_dir))

        # Set up log file
        log_file_path = os.path.join(experiment_dir, 'log.log') # Set the log path
        logger = get_logger(log_file_path=log_file_path) # Set and get the root logger
    
    except Exception as e:
        print("Error: {}".format(e))
        return


    try: # Now the logger has been successfully set up, and errors can be logged in the log file.

        # Set up experiment objects
        env = GridWorld(size=grid_size, p=0.9, trajectory_length_max=100)
        initial_state = env.reset()

        reward_table = np.array(REWARD_VEC).reshape(1,-1).repeat(state_dim,axis=0) # Shaped (state_dim, action_dim), ground-truth reward table
        reward_model = GridReward(reward_table)

        env.set_reward_model(reward_model)
        oracle = SimulatedOracle(reward_model, gamma=args.comparison_gamma)

        p2r = P2R_Interface(state_dim=state_dim, action_dim=action_dim, oracle=oracle,
                            m_repeats=10000,
                            epsilon_0=epsilon_0,
                            gamma=args.comparison_gamma,
                            robust=args.robust
                            )
        p2r.set_reference(initial_state,0)

        ucbvi_bf = UCBVI_BF(state_dim=state_dim, action_dim=action_dim,trajectory_num=trajectory_num, trajectory_length=trajectory_length, delta=args.ucbvi_delta)

        # --------------------------------------------------------------------------------
        # Main experiment
        logging.info("Experiment {} started.".format(time_string))
        logging.info("Experiment directory: '{}'".format(experiment_dir))
        logging.info("grid_size: {}, state_dim: {}, action_dim: {}, trajectory_num: {}, trajectory_length: {}".format(
                    grid_size, state_dim, action_dim, trajectory_num, trajectory_length))

        trajectory_reward_list = [] # A list to store the reward of each trajectory
        total_queries_list = []
        epsilon=args.policy_epsilon

        for k in range(trajectory_num):

            trajectory_reward = 0.0

            Q_k = ucbvi_bf.ucb_q_values(reward_estimator=p2r)
            policy_table = ucbvi_bf.extract_policy(Q_k, epsilon=epsilon)
            policy = GridPolicy_h(policy_table)

            # Generate new episode
            initial_state = env.reset()
            state = initial_state
            episode_tuples = []
            action_chars = ''
            for h in range(trajectory_length): # Generate new episodes by following the current policy
                action = policy(h, state)
                state_prev = state
                state, reward, terminated, truncated, info = env.step(action)
                trajectory_reward += reward
                episode_tuples.append((h, state_prev, action, state, reward))
                action_chars += env.action_to_char(action)
                if terminated or truncated:
                    break

            ucbvi_bf.update_with_episode(episode_tuples) # Update UCBVI-BF memory with the new episodes

            epsilon = epsilon * args.policy_epsilon_decay
            trajectory_reward_list.append(trajectory_reward) # average reward per trajectory
            total_queries_list.append(p2r.query_count)

            logging.info("Trajectory {:>7}; trajectory reward: {:>6.3f}; total query count: {:>4}; epsilon: {:>6.4f}; actions: [{}].".format(
                k, trajectory_reward, p2r.query_count, epsilon, action_chars))


        # Visualization
        ## Heatmap of final state distribution
        state_distribution_final = ucbvi_bf.count_hs[-1].reshape(grid_size, grid_size)
        plt.figure(figsize=(8,8))
        sns.heatmap(state_distribution_final, annot=True, fmt='d')
        plt.title('Final state distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        fig_save_path = os.path.join(experiment_dir, "state_distribution_final.png")
        plt.savefig(fig_save_path)
        logging.info("State distribution figure saved to {}.".format(fig_save_path))
        plt.close()

        ## Metrics in the learning process
        trajectory_reward_average = np.array(trajectory_reward_list).reshape(-1,20).mean(axis=-1) # average the rewards per 20 episodes
        x_idx_selected = np.linspace(0, len(trajectory_reward_average)-1, 6, dtype=int)
        trajectory_reward_average_smoothed = moving_average_smoothing(trajectory_reward_average, window_size=10)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(trajectory_reward_average, label='Average trajectory reward over each 20 trajectories')
        plt.plot(trajectory_reward_average_smoothed, label='Smoothed average trajectory reward (window_size=10)')
        plt.xticks(x_idx_selected, (x_idx_selected+1)*20)
        plt.xlabel('Trajectory')
        plt.ylabel('Trajectory')
        plt.title('Trajectory Reward Curve')
        plt.subplot(1,2,2)
        plt.plot(total_queries_list)
        plt.xlabel('Trajectory')
        plt.ylabel('Cumulative P2R queries')
        plt.title('Human Preference Query Count')
        plt.tight_layout()
        fig_save_path = os.path.join(experiment_dir, "metrics.png")
        plt.savefig(fig_save_path)
        logging.info("Metrics figure saved to {}.".format(fig_save_path))
        plt.close()

        # Save objects
        object_dict = {}

        Q_final = ucbvi_bf.ucb_q_values(reward_estimator=p2r)
        policy_table_final = ucbvi_bf.extract_policy(Q_final, epsilon=0.0)
        policy_final = GridPolicy_h(policy_table_final)

        object_dict['Q_final'] = Q_final
        object_dict['policy_final'] = policy_final
        object_dict['p2r'] = p2r
        object_dict['ucbvi_bf'] = ucbvi_bf
        object_dict['trajectory_reward_list'] = trajectory_reward_list
        object_dict['total_queries_list'] = total_queries_list

        object_dict_save_path = os.path.join(experiment_dir, "objects.pkl")
        pickle.dump(object_dict, open(object_dict_save_path, 'wb'))
        logging.info("Experiment objects saved to {}.".format(object_dict_save_path))

        logging.info("Action distribution: {}".format(ucbvi_bf.count_sa.sum(axis=0)))

        logging.info("Finished. Time elapsed: {:.2f}s.".format(time.time() - start_time))
        close_logger(logger)
        return 0

    except Exception as e:
        logging.error("Error: {}".format(e))
        logging.error(traceback.format_exc())
        close_logger(logger)

        return e


if __name__ == '__main__':
    sys.exit(main(parse_args()))
