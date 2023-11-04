# -*- coding: utf-8 -*-

import logging

import torch
from stable_baselines3 import PPO

from env import ImagePerturbEnv
from src.plotting import plot_rewards_and_cumulative
from src.utils import get_cifar_dataloaders, load_model, set_seed

logging.basicConfig(level=logging.INFO)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
set_seed(42)  # seed for reproducibility, data loaders, model init, etc


if __name__ == "__main__":
    # hyperparameters for stable baselines / env
    reward_lambda = 1
    episodes = 100
    steps_per_episode = 10  # number of images to perturb per episode
    total_timesteps = episodes * steps_per_episode
    # cifar10 dataloader
    train_loader, valid_loader, test_loader = get_cifar_dataloaders(batch_size=64, val_split=0.2, seed=SEED)
    train_loader2, valid_loader, test_loader = get_cifar_dataloaders(batch_size=64, val_split=0.2, seed=SEED)

    print(
        f"""
    size of train_loader: {len(train_loader)}
    size of valid_loader: {len(valid_loader)}
    size of test_loader: {len(test_loader)}
    """
    )

    # classififer
    model = load_model()

    # env
    env = ImagePerturbEnv(
        dataloader=train_loader, model=model, lambda_=reward_lambda, steps_per_episode=steps_per_episode
    )

    # Training here
    model = PPO("MlpPolicy", env, device=DEVICE, verbose=1, n_steps=steps_per_episode, batch_size=128)
    print("here")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=1)

    print(model.ep_info_buffer)
    ep_info_buffer = model.ep_info_buffer

    rewards = [info["r"] for info in ep_info_buffer]
    lengths = [info["l"] for info in ep_info_buffer]
    times = [info["t"] for info in ep_info_buffer]

    plot_rewards_and_cumulative(rewards)
