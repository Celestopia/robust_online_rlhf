import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
from utils.functions import moving_average_smoothing


def draw_Q_table(Q: np.ndarray, fig_save_path: str):
    # Q: Shaped (H, S, A)
    Q_hxya = Q.reshape(5, 11, 11, 5)
    xticks=np.arange(11)+0.5
    yticks=np.arange(11)+0.5
    xticklabels=np.arange(11)-5
    yticklabels=5-np.arange(11)

    fig, axs = plt.subplots(5, 5, figsize=(50, 50))
    for h in range(5):
        for a in range(5):
            axs[h][a].set_title('Q_{}(s, a); action: {}'.format(h, a))
            sns.heatmap(Q_hxya[h, :, :, a], ax=axs[h][a], cmap='rocket', annot=True, fmt='.2f', cbar=False, vmin=-5.0, vmax=5.0)
            axs[h][a].set_xticks(xticks, xticklabels, rotation=0)
            axs[h][a].set_yticks(yticks, yticklabels, rotation=0)
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_state_action_count(count_sa: np.ndarray, fig_save_path: str):
    # count_sa: Shaped (S, A)
    count_xya = count_sa.reshape(11,11,5)
    xticks=np.arange(11)+0.5
    yticks=np.arange(11)+0.5
    xticklabels=np.arange(11)-5
    yticklabels=5-np.arange(11)
    vmax=count_xya.max()

    fig, axs = plt.subplots(1, 5, figsize=(50, 8))
    for i in range(5):
        axs[i].set_title('Action {} Count'.format(i))
        sns.heatmap(count_xya[:,:,i], ax=axs[i], cmap='rocket', fmt='d', annot=True, vmin=0, vmax=vmax)
        axs[i].set_xticks(xticks, xticklabels, rotation=0)
        axs[i].set_yticks(yticks, yticklabels, rotation=0)
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_trajectory_reward_curve(trajectory_reward_list: List[float], fig_save_path: str):
    group_size = 20
    selected_idx_max = len(trajectory_reward_list) // group_size * group_size
    trajectory_reward_average = np.array(trajectory_reward_list[:selected_idx_max]).reshape(-1,group_size).mean(axis=-1) # average the rewards per 20 episodes
    x_idx_selected = np.linspace(0, len(trajectory_reward_average)-1, 8, dtype=int)
    
    window_size = 11
    trajectory_reward_average_smoothed = moving_average_smoothing(trajectory_reward_average, window_size=window_size)
    
    plt.figure(figsize=(8,6))
    plt.plot(trajectory_reward_average, label='Average trajectory reward (pool_size={})'.format(group_size))
    plt.plot(trajectory_reward_average_smoothed, label='Smoothed average trajectory reward (window_size={})'.format(window_size))
    plt.xticks(x_idx_selected, (x_idx_selected+1)*group_size)
    plt.xlabel('Number of Trajectories')
    plt.ylabel('Trajectory Reward')
    plt.title('Trajectory Reward Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_query_count_curve(query_count_list: List[int], fig_save_path: str):
    plt.figure(figsize=(8,6))
    plt.plot(query_count_list)
    plt.xlabel('Number of Trajectories')
    plt.ylabel('Cumulative P2R queries')
    plt.title('Human Preference Query Count')
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_action_count_curve(action_count_lists: List[List[int]], fig_save_path: str):
    
    target_action = 1
    y = action_count_lists[target_action]
    x = np.arange(len(y))
    y_opt = x
    y_ref = x / 5
    plt.figure(figsize=(8,6))
    plt.axis([100, len(y), 10, len(y)*1.2])
    plt.plot(x, y, c='r', label='Optimal Action Count')
    plt.plot(x, y_opt, c='b', ls='--', label=r'$y=x$')
    plt.plot(x, y_ref, c='g', ls='--', label=r'$y=x/5$', alpha=0.3)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Action Count')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Action Count Curve for Action {}'.format(target_action))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()

def draw_final_state_count(final_state_count: np.ndarray, fig_save_path: str):
    count_xy = final_state_count.reshape(11,11)
    xticks=np.arange(11)+0.5
    yticks=np.arange(11)+0.5
    xticklabels=np.arange(11)-5
    yticklabels=5-np.arange(11)
    plt.figure(figsize=(12,10))
    plt.title('Final State Count')
    sns.heatmap(count_xy, cmap='rocket', annot=True, fmt='d')
    plt.xticks(xticks, xticklabels, rotation=0)
    plt.yticks(yticks, yticklabels, rotation=0)
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()

