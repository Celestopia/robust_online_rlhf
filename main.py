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
import argparse
import yaml

from env import GridWorld, SimulatedOracle, GridReward
from policy import GridPolicy, GridPolicy_h
from p2r import P2R_Interface
from ucbvi import UCBVI_BF
from utils.log import get_logger, close_logger

from constants import GRID_SIZE, STATE_DIM, ACTION_DIM, EPISODE_LENGTH, REWARD_VEC, EPSILON_0



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--episode_num', type=int, default=200, help='Number of episodes')
    parser.add_argument('--episode_length', type=int, default=EPISODE_LENGTH, help='Length of each episode')
    parser.add_argument('--policy_epsilon', type=float, default=0.0, help='Epsilon-greedy exploration parameter')
    parser.add_argument('--policy_epsilon_decay', type=float, default=0.9995, help='Decay rate of epsilon-greedy exploration parameter')
    parser.add_argument('--comparison_gamma', type=float, default=0.2, help='Noise level of comparison oracle')
    parser.add_argument('--ucbvi_delta', type=float, default=0.1, help='UCBVI delta parameter')
    parser.add_argument('--robust', action='store_true', help='Use robust P2R')

    args = parser.parse_args()
    return args



def main(args):
    start_time = time.time()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set hyperparameters
    grid_size = GRID_SIZE
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    episode_num = args.episode_num
    episode_length = args.episode_length
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
                                    episode_num,
                                    episode_length,
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

    # Set up experiment objects
    env = GridWorld(size=grid_size, p=0.9, episode_length_max=100)
    initial_state = env.reset()

    reward_table = np.array(REWARD_VEC).reshape(1,-1).repeat(state_dim,axis=0) # Ground-truth reward table
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

    ucbvi_bf = UCBVI_BF(state_dim=state_dim, action_dim=action_dim,episode_num=episode_num, episode_length=episode_length, delta=args.ucbvi_delta)

    # --------------------------------------------------------------------------------
    # Main experiment
    logging.info("Experiment {} started.".format(time_string))
    logging.info("Experiment directory: '{}'".format(experiment_dir))
    logging.info("grid_size: {}, state_dim: {}, action_dim: {}, episode_num: {}, episode_length: {}".format(
                grid_size, state_dim, action_dim, episode_num, episode_length))

    total_queries = 0
    episode_reward_list = []
    total_queries_list = []
    epsilon=args.policy_epsilon

    for k in range(episode_num):
        
        #logging.info("Episode {}/{} started.".format(k, episode_num))
        episode_reward = 0.0

        #logging.info("Episode {}/{}: Using historical data to update Q value estimate by UCBVI-BF...".format(k, episode_num))
        Q_k = ucbvi_bf.ucb_q_values(reward_estimator=p2r)
        policy_table = ucbvi_bf.extract_policy(Q_k, epsilon=epsilon)
        policy = GridPolicy_h(policy_table)
        
        # Generate new episode
        initial_state = env.reset()
        state = initial_state
        episode_tuples = []
        action_chars = ''
        #logging.info("Episode {}/{}: Generating new episode...".format(k, episode_num))
        for h in range(episode_length):
            action = policy(h, state)
            state_prev = state
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_tuples.append((h, state_prev, action, state, reward))
            action_chars += env.action_to_char(action)
            if terminated or truncated:
                break

        ucbvi_bf.update_with_episode(episode_tuples)

        epsilon = epsilon * args.policy_epsilon_decay
        episode_reward_list.append(episode_reward) # average reward per episode so far
        total_queries = p2r.query_count
        total_queries_list.append(total_queries)

        logging.info("Episode {:>7}; episode reward: {:>6.3f}; total query count: {:>4}; epsilon: {:>6.4f}; actions: [{}].".format(
            k, episode_reward, total_queries, epsilon, action_chars))


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
    episode_reward_average = np.array(episode_reward_list).reshape(-1,20).mean(axis=-1) # average the rewards per 20 episodes
    x_idx_selected = np.linspace(0, len(episode_reward_average)-1, 6, dtype=int)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(episode_reward_average)
    plt.xticks(x_idx_selected, (x_idx_selected+1)*20)
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.title('Episode Reward Curve')
    plt.subplot(1,2,2)
    plt.plot(total_queries_list)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative P2R queries')
    plt.title('Human Preference Query Count')
    plt.tight_layout()
    fig_save_path = os.path.join(experiment_dir, "metrics.png")
    plt.savefig(fig_save_path)
    logging.info("Metrics figure saved to {}.".format(fig_save_path))
    plt.close()

    # Save objects
    Q_final = ucbvi_bf.ucb_q_values(reward_estimator=p2r)
    Q_save_path = os.path.join(experiment_dir, "Q_final.npy")
    np.save(os.path.join(experiment_dir, "Q_final.npy"), Q_final)
    logging.info("Q value table saved to {}.".format(Q_save_path))

    policy_table_final = ucbvi_bf.extract_policy(Q_final, epsilon=0)
    policy_final = GridPolicy_h(policy_table_final)
    policy_save_path = os.path.join(experiment_dir, "policy_final.pkl")
    pickle.dump(policy_final, open(policy_save_path, "wb"))
    logging.info("Policy object saved to {}.".format(policy_save_path))

    p2r_save_path = os.path.join(experiment_dir, "p2r_final.pkl")
    pickle.dump(p2r, open(p2r_save_path, "wb"))
    logging.info("P2R object saved to {}.".format(p2r_save_path))

    ucbvi_bf_save_path = os.path.join(experiment_dir, "ucbvi_bf_final.pkl")
    pickle.dump(ucbvi_bf, open(ucbvi_bf_save_path, "wb"))
    logging.info("UCBVI-BF object saved to {}.".format(ucbvi_bf_save_path))

    episode_reward_list_save_path = os.path.join(experiment_dir, "episode_reward_list.pkl")
    pickle.dump(episode_reward_list, open(episode_reward_list_save_path, "wb"))
    logging.info("Episode reward list saved to {}.".format(episode_reward_list_save_path))

    total_queries_list_save_path = os.path.join(experiment_dir, "total_queries_list.pkl")
    pickle.dump(total_queries_list, open(total_queries_list_save_path, "wb"))
    logging.info("Total queries list saved to {}.".format(total_queries_list_save_path))

    #logging.info("Final Q value table:")
    #logging.info(Q_final.mean(axis=0))

    #logging.info("Final policy table:")
    #logging.info(policy_final.policy_array.mean(axis=0))

    logging.info("Finished. Time elapsed: {:.2f}s.".format(time.time() - start_time))
    close_logger(logger)

    return


if __name__ == '__main__':
    sys.exit(main(parse_args()))
