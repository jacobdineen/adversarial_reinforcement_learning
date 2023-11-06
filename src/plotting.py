# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_and_cumulative(rewards):
    cumulative_rewards = np.cumsum(rewards)

    plt.figure(figsize=(14, 7))

    # Rewards over Episodes
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Rewards")
    # Trendline for Rewards
    z = np.polyfit(range(len(rewards)), rewards, 1)
    p = np.poly1d(z)
    plt.plot(range(len(rewards)), p(range(len(rewards))), "r--", label="Trendline")
    plt.title("Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # Cumulative Rewards over Episodes
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_rewards, label="Cumulative Rewards")
    # Trendline for Cumulative Rewards
    z = np.polyfit(range(len(cumulative_rewards)), cumulative_rewards, 1)
    p = np.poly1d(z)
    plt.plot(range(len(cumulative_rewards)), p(range(len(cumulative_rewards))), "r--", label="Trendline")
    plt.title("Cumulative Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()

    plt.tight_layout()
    plt.show()
