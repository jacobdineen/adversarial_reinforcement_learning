# -*- coding: utf-8 -*-
import argparse
import logging

import torch
from stable_baselines3 import PPO

from src.env import ImagePerturbEnv
from src.plotting import plot_rewards_and_cumulative
from src.utils import get_cifar_dataloaders, load_model, set_seed

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42

parser = argparse.ArgumentParser(description="Train an agent to perturb images.")
parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to run.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
parser.add_argument("--val_split", type=float, default=0.2, help="Holdout data for validation and testing.")
args = parser.parse_args()

episodes = args.num_episodes
batch_size = args.batch_size
val_split = args.batch_size


if __name__ == "__main__":
    set_seed(SEED)

    train_loader, valid_loader, test_loader = get_cifar_dataloaders(batch_size=1, val_split=val_split, seed=SEED)
    episodes = 3
    steps_per_episode = len(train_loader)  # number of images to perturb per episode
    total_timesteps = episodes * steps_per_episode

    logging.info(
        f"""
    size of train_loader: {len(train_loader)}
    size of valid_loader: {len(valid_loader)}
    size of test_loader: {len(test_loader)}
    """
    )

    # classififer
    model = load_model()

    # env
    env = ImagePerturbEnv(dataloader=train_loader, model=model, steps_per_episode=steps_per_episode, verbose=False)

    # Training here
    model = PPO("MlpPolicy", env, device=DEVICE, verbose=1, n_steps=steps_per_episode, batch_size=batch_size)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    ep_info_buffer = model.ep_info_buffer
    rewards = [info["r"] for info in ep_info_buffer]
    lengths = [info["l"] for info in ep_info_buffer]
    times = [info["t"] for info in ep_info_buffer]

    plot_rewards_and_cumulative(rewards)
