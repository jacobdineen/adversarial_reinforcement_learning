# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_and_cumulative(rewards):
    cumulative_rewards = np.cumsum(rewards)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(cumulative_rewards)
    plt.title("Cumulative Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")

    plt.tight_layout()
    plt.show()
