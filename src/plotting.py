# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd


def plt_helper(file_path):
    # Define the columns to plot
    columns = ["rollout/ep_rew_mean", "train/loss"]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create subplots
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, column_name in enumerate(columns):
        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            print(
                f"The specified column '{column_name}' does not exist in the CSV file."
            )
            continue  # Skip to the next column

        # Plot the specified column
        axs[i].plot(
            df[column_name],
            marker="o",
            linestyle="-",
            label=column_name.replace("/", " "),
        )
        axs[i].set_title(column_name.replace("/", " "))
        axs[i].set_xlabel("Steps")
        axs[i].set_ylabel(column_name.split("/")[-1])
        axs[i].grid(True)
        axs[i].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
