# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_selected_columns_from_csv(file_path):
    # Define the columns to plot
    columns = ["rollout/ep_rew_mean", "train/loss", "train/value_loss", "train/policy_gradient_loss"]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create subplots
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, column_name in enumerate(columns):
        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            print(f"The specified column '{column_name}' does not exist in the CSV file.")
            continue  # Skip to the next column

        # Plot the specified column
        axs[i].plot(df[column_name], marker="o", linestyle="-", label=column_name.replace("/", " "))
        axs[i].set_title(column_name.replace("/", " "))
        axs[i].set_xlabel("Steps")
        axs[i].set_ylabel(column_name.split("/")[-1])
        axs[i].grid(True)
        axs[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


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
