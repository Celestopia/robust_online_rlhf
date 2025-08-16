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
from utils.plot import draw_Q_table, draw_state_action_count, draw_trajectory_reward_curve, draw_query_count_curve, draw_action_count_curve

from constants import GRID_SIZE, STATE_DIM, ACTION_DIM, TRAJECTORY_LENGTH, REWARD_VEC, EPSILON_0



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to store all experiments')
    parser.add_argument('-K', '--trajectory_num', type=int, default=1000, help='Number of trajectories')
    parser.add_argument('-H', '--trajectory_length', type=int, default=TRAJECTORY_LENGTH, help='Length of each trajectory')
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
        result_dir = args.result_dir
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
        query_count_list = [] # A list to store the total number of queries
        
        final_state_count = np.zeros(state_dim, dtype=int) # final_state_count[s] represents the number of times the agent ends up in state s.
        action_count = np.zeros(action_dim, dtype=int) # action_count[a] represents the number of times the agent takes action a.
        action_count_lists = [[] for i in range(action_dim)] # action_count_lists[a][k] represents the number of times the agent takes action a up to the k-th iteration.

        for k in range(trajectory_num):
            Q_k = ucbvi_bf.ucb_q_values(reward_estimator=p2r) # Shaped (H, S, A), the learned Q-value table at k-th iteration.
            policy_table = ucbvi_bf.extract_policy(Q_k, epsilon=0.0) # Shaped (H, S, A), representing π_h(a|s).
            policy = GridPolicy_h(policy_table) # Newest policy

            # Generate new episodes
            initial_state = env.reset()
            state = initial_state
            episode_tuples = []
            action_chars = '' # A string representing the actions taken in the current trajectory, e.g., '→↑o↑→'.
            trajectory_reward = 0.0 # The total reward of the current trajectory, which consists of multiple (s, a) pairs.
            for h in range(trajectory_length): # Generate a new trajectory by following the current policy
                # Updating
                action = policy(h, state)
                state_prev = state
                state, reward, terminated, truncated, info = env.step(action)
                
                # Recording
                episode_tuples.append((h, state_prev, action, state, reward))
                trajectory_reward += reward
                action_count[action] += 1
                for a in range(action_dim):
                    action_count_lists[a].append(action_count[a])
                action_chars += env.action_to_char(action)
                if terminated or truncated:
                    break
            ucbvi_bf.update_with_episode(episode_tuples) # Update UCBVI-BF memory with the new episodes

            # Recording
            final_state_count[state] += 1
            trajectory_reward_list.append(trajectory_reward) # average reward per trajectory
            query_count_list.append(p2r.query_count)

            logging.info("Trajectory {:>7}; trajectory reward: {:>6.3f}; total query count: {:>4}; actions: [{}].".format(
                k, trajectory_reward, p2r.query_count, action_chars))
            
            # Visualization
            if k in [1000, 2500, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 200000, 300000, 500000]:
                Q_table_fig_save_path = os.path.join(experiment_dir, "Q_h_heatmap_{}.png".format(k))
                draw_Q_table(Q_k, Q_table_fig_save_path)
                logging.info("Q-table figure saved to {}.".format(Q_table_fig_save_path))

                state_action_count_fig_save_path = os.path.join(experiment_dir, "state_action_count_heatmap_{}.png".format(k))
                draw_state_action_count(ucbvi_bf.count_sa, state_action_count_fig_save_path)
                logging.info("State-action count figure saved to {}.".format(state_action_count_fig_save_path))

        # Visualization
        trajectory_reward_fig_save_path = os.path.join(experiment_dir, "trajectory_reward_curve.png")
        draw_trajectory_reward_curve(trajectory_reward_list, trajectory_reward_fig_save_path)
        logging.info("Trajectory reward curve figure saved to {}.".format(trajectory_reward_fig_save_path))

        query_count_fig_save_path = os.path.join(experiment_dir, "query_count_curve.png")
        draw_query_count_curve(query_count_list, query_count_fig_save_path)
        logging.info("Query count curve figure saved to {}.".format(query_count_fig_save_path))

        action_count_fig_save_path = os.path.join(experiment_dir, "action_count_curve.png")
        draw_action_count_curve(action_count_lists, action_count_fig_save_path)
        logging.info("Action count curve figure saved to {}.".format(action_count_fig_save_path))

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
        object_dict['query_count_list'] = query_count_list
        object_dict['final_state_count'] = final_state_count
        object_dict['action_count'] = action_count
        object_dict['action_count_lists'] = action_count_lists

        object_dict_save_path = os.path.join(experiment_dir, "objects.pkl")
        pickle.dump(object_dict, open(object_dict_save_path, 'wb'))
        logging.info("Experiment objects saved to {}.".format(object_dict_save_path))

        # Ending matters
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
