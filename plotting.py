# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd

episode_rewards_df = pd.read_csv("episode_rewards.csv")
all_actions_taken_df = pd.read_csv("all_actions_taken.csv")

import numpy as np

z = np.polyfit(episode_rewards_df["Episode"], episode_rewards_df["Reward"], 1)
p = np.poly1d(z)

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards_df["Episode"], episode_rewards_df["Reward"], marker="o")
plt.plot(episode_rewards_df["Episode"], p(episode_rewards_df["Episode"]), "r--")
plt.title("Episode Rewards over Time")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()

from collections import Counter

all_actions_taken_df["Actions"] = all_actions_taken_df["Actions"].apply(lambda x: set(map(int, x.split(","))))


import pandas as pd

# Assuming your CSV is loaded into a DataFrame called df
df = pd.read_csv("all_actions_taken.csv")

# Grouping every 20 rows together based on the Episode column
grouped_df = df.groupby(df.index // 20).agg(
    {"Episode": "first", "Actions": lambda x: ",".join(x)}  # or 'last', it doesn't matter if they are the same
)

# Now split the aggregated 'Actions' back into sets of unique actions for each episode
grouped_df["Actions"] = grouped_df["Actions"].apply(lambda x: set(map(int, x.split(","))))


# Expand the DataFrame so each action in an episode gets its own row
expanded_data = []
for i, row in grouped_df.iterrows():
    episode = row["Episode"]
    for action in row["Actions"]:
        expanded_data.append([episode, action])

expanded_df = pd.DataFrame(expanded_data, columns=["Episode", "Action"])
print(expanded_df.tail())
# Create the scatter plot
plt.figure(figsize=(14, 8))
plt.scatter(expanded_df["Episode"], expanded_df["Action"], alpha=0.6)
plt.title("Actions by Episode")
plt.xlabel("Episode")
plt.ylabel("Action")
plt.grid(True)
plt.show()
