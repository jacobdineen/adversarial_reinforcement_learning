# -*- coding: utf-8 -*-

import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO

from env import ImagePerturbEnv
from src.utils import get_cifar_dataloader, load_model

logging.basicConfig(level=logging.INFO)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


if __name__ == "__main__":
    # hyperparameters for stable baselines / env
    reward_lambda = 1
    episodes = 100
    steps_per_episode = 10  # number of images to perturb per episode
    total_timesteps = episodes * steps_per_episode
    # cifar10 dataloader
    dataloader = get_cifar_dataloader()

    # classififer
    model = load_model()
    # env
    env = ImagePerturbEnv(
        dataloader=dataloader, model=model, lambda_=reward_lambda, steps_per_episode=steps_per_episode
    )

    model = PPO("MlpPolicy", env, device=DEVICE, verbose=1, n_steps=steps_per_episode, batch_size=128)
    print("here")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=1)

    print(model.ep_info_buffer)
    ep_info_buffer = model.ep_info_buffer

    rewards = [info["r"] for info in ep_info_buffer]
    lengths = [info["l"] for info in ep_info_buffer]
    times = [info["t"] for info in ep_info_buffer]

    plot_rewards_and_cumulative(rewards)
