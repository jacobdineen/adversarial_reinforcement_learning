# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plt_helper(file_path, steps):
    # Define the columns to plot
    columns = ["rollout/ep_rew_mean", "train/loss"]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Accessing the index of the DataFrame
    df["Run"] = (df.index // steps) + 1
    df["Normalized_Steps"] = (df.index % steps) + 1

    # Create subplots
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    # Iterate over the columns to create separate line plots
    for i, column_name in enumerate(columns):
        if column_name not in df.columns:
            print(f"The specified column '{column_name}' does not exist in the CSV file.")
            continue  # Skip to the next column
        sns.lineplot(
            ax=axs[i],
            data=df,
            x="Normalized_Steps",
            y=column_name,
            # hue="Run",
            estimator="mean",
            ci="sd",
        )

        axs[i].set_title(column_name.replace("/", " "))
        axs[i].set_xlabel("Episode")
        axs[i].set_ylabel(column_name.split("/")[-1])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
