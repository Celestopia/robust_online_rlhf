import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List
from utils.functions import moving_average_smoothing


def draw_Q_table(Q: np.ndarray, fig_save_path: str):
    # Q: Shaped (H, S, A)
    yticklabels=['{:<3}: ({:>2}, {:>2})'.format(i, i%11-5, 5-i//11) for i in range(121)]
    xticklabels=['{}'.format(i) for i in range(5)]
    fig, axs = plt.subplots(1, 5, figsize=(35, 60))
    plt.suptitle('Q-value Tables')
    for i in range(5):
        axs[i].set_title('Q_{}(s, a)'.format(i))
        sns.heatmap(Q[i], ax=axs[i], cmap='rocket', annot=True, fmt='.2f', cbar=False)
        axs[i].set_xticks(np.arange(5)+0.5, xticklabels, rotation=0)
        axs[i].set_yticks(np.arange(121)+0.5, yticklabels, rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.97)
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_state_action_count(count_sa: np.ndarray, fig_save_path: str):
    # count_sa: Shaped (S, A)
    count_xya = count_sa.reshape(11,11,5)
    xticks=np.arange(11)+0.5
    yticks=np.arange(11)+0.5
    xticklabels=np.arange(11)-5
    yticklabels=5-np.arange(11)

    fig, axs = plt.subplots(1, 5, figsize=(50, 8))
    for i in range(5):
        axs[i].set_title('Action {} Count'.format(i))
        sns.heatmap(count_xya[:,:,i], ax=axs[i], cmap='rocket', fmt='d', annot=True)
        axs[i].set_xticks(xticks, xticklabels, rotation=0)
        axs[i].set_yticks(yticks, yticklabels, rotation=0)
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_trajectory_reward_curve(trajectory_reward_list: List[float], fig_save_path: str):
    trajectory_reward_average = np.array(trajectory_reward_list).reshape(-1,20).mean(axis=-1) # average the rewards per 20 episodes
    x_idx_selected = np.linspace(0, len(trajectory_reward_average)-1, 6, dtype=int)
    trajectory_reward_average_smoothed = moving_average_smoothing(trajectory_reward_average, window_size=10)
    plt.figure(figsize=(8,6))
    plt.plot(trajectory_reward_average, label='Average trajectory reward (pool_size=20)')
    plt.plot(trajectory_reward_average_smoothed, label='Smoothed average trajectory reward (window_size=10)')
    plt.xticks(x_idx_selected, (x_idx_selected+1)*20)
    plt.xlabel('Trajectory')
    plt.ylabel('Trajectory Reward')
    plt.title('Trajectory Reward Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_query_count_curve(query_count_list: List[int], fig_save_path: str):
    plt.figure(figsize=(8,6))
    plt.plot(query_count_list)
    plt.xlabel('Trajectory')
    plt.ylabel('Cumulative P2R queries')
    plt.title('Human Preference Query Count')
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()


def draw_action_count_curve(action_count_lists: List[List[int]], fig_save_path: str):
    plt.figure(figsize=(8,6))
    target_action = 1
    plt.plot(action_count_lists[target_action])
    plt.xlabel('Trajectory')
    plt.ylabel('Action Count')
    plt.title('Action Count Curve for Action {}'.format(target_action))
    plt.tight_layout()
    plt.savefig(fig_save_path, bbox_inches='tight')
    plt.close()
