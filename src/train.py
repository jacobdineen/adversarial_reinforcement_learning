# -*- coding: utf-8 -*-

import logging
import random

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
    attack_budget = 1  # max number of perturbations (len(channel) pixel changes each attack)
    num_times_to_sample = 1  # number of times to sample each image consecutively before sampling new image
    reward_lambda = 1

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
    model = PPO("MlpPolicy", env, device=DEVICE, verbose=1)
    print("here")
    model.learn(total_timesteps=1, progress_bar=True)
