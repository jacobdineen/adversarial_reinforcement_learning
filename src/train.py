# -*- coding: utf-8 -*-

import logging
import random

import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    # There is still a weird disconnect here between what we define as an episode
    # and what stable baselines expects as an episode. We need to figure out how to
    # make these two things line up.

    # easiest way to do this is to just make the number of steps in the env
    # equal to the number of steps in the stable baselines model.

    # hyperparameters for stable baselines / env
    attack_budget = 10  # max number of perturbations (len(channel) pixel changes each attack)
    num_times_to_sample = 1  # number of times to sample each image consecutively before sampling new image
    reward_lambda = 1
    episodes = 100
    n_steps = attack_budget * num_times_to_sample * episodes

    # cifar10 dataloader
    dataloader = get_cifar_dataloader()

    # classififer
    model = load_model()
    # env
    env = ImagePerturbEnv(
        dataloader=dataloader,
        model=model,
        attack_budget=attack_budget,
        lambda_=reward_lambda,
        num_times_to_sample=num_times_to_sample,
    )

    print("here")
    model = PPO("MlpPolicy", env, device=DEVICE, verbose=1, n_steps=n_steps, batch_size=128)
    print("here")
    model.learn(total_timesteps=n_steps, progress_bar=True, log_interval=1)
    print(model.ep_info_buffer)
    ep_info_buffer = model.ep_info_buffer

    rewards = [info["r"] for info in ep_info_buffer]
    lengths = [info["l"] for info in ep_info_buffer]
    times = [info["t"] for info in ep_info_buffer]

    # To plot rewards
    plt.figure()
    plt.plot(rewards)
    plt.title("Rewards over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
