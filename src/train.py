# -*- coding: utf-8 -*-
import argparse
import logging

import pandas as pd
import torch
from stable_baselines3 import PPO

from src.env import ImagePerturbEnv
from src.plotting import plot_rewards_and_cumulative
from src.utils import EndlessDataLoader, get_cifar_dataloaders, load_model, set_seed

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42

parser = argparse.ArgumentParser(description="Train an agent to perturb images.")
parser.add_argument("--num_episodes", type=int, default=30, help="Number of episodes to run.")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training.")
parser.add_argument("--val_split", type=float, default=0.2, help="Holdout data for validation and testing.")
parser.add_argument(
    "--train_limit", type=int, default=None, help="Training dataloader limit - useful for debugging shorter runs."
)
parser.add_argument("--verbose", type=bool, default=False, help="If you want environment logging to be verbose.")
parser.add_argument("--prog_bar", type=bool, default=True, help="If you want to use tqdm for train loop.")
parser.add_argument(
    "--model_save_path", type=str, default="src/model_weights/ppo", help="Where to save trained PPO model"
)
parser.add_argument(
    "--model_performance_save_path", type=str, default="src/model_weights/ppo", help="Where to save ep info buff"
)

args = parser.parse_args()

episodes = args.num_episodes
batch_size = args.batch_size
val_split = args.val_split
train_limit = args.train_limit
verbose = args.verbose
prog_bar = args.prog_bar
model_save_path = args.model_save_path
model_performance_save_path = args.model_performance_save_path


# maybe batch size should be

if __name__ == "__main__":
    set_seed(SEED)
    # this needs to be 1 - because each call to iter will return a single image
    # and that's what the env expects
    # This is highly seeded to return the same batches every run
    train_loader, valid_loader, test_loader = get_cifar_dataloaders(
        batch_size=1, val_split=val_split, seed=SEED, train_limit=train_limit
    )

    steps_per_episode = len(train_loader)  # number of images to perturb per episode
    total_timesteps = episodes * steps_per_episode

    # classififer
    model = load_model()

    # env
    # Note the EndlessDataLoader wrapper
    # This is to ensure that when a dataloader has been exhausted, it is refreshed
    # by starting from the beginning
    train_env = ImagePerturbEnv(
        dataloader=EndlessDataLoader(train_loader), model=model, steps_per_episode=steps_per_episode, verbose=verbose
    )
    # eventually use this for validation
    valid_env = ImagePerturbEnv(
        dataloader=EndlessDataLoader(valid_loader), model=model, steps_per_episode=steps_per_episode, verbose=verbose
    )

    # Training here
    model = PPO("MlpPolicy", train_env, device=DEVICE, verbose=1, n_steps=steps_per_episode, batch_size=batch_size)
    logging.info(f"model device: {model.device}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=prog_bar,
    )
    logging.info("Training complete.")
    logging.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)

    ep_info_buffer = model.ep_info_buffer
    print(ep_info_buffer)
    rewards = [info["r"] for info in ep_info_buffer]
    lengths = [info["l"] for info in ep_info_buffer]
    times = [info["t"] for info in ep_info_buffer]

    df = pd.DataFrame(ep_info_buffer)
    df.rename(columns={"r": "rewards", "l": "lengths", "t": "times"}, inplace=True)
    df.to_csv(f"{model_performance_save_path}.csv")
    logging.info(f"Dataframe saved to {model_performance_save_path}")

    plot_rewards_and_cumulative(rewards)

    # Note the below
    # I want to integrate something like this
    # like after we train an for an episode (steps_per_episode), we evaluate the model on the validation env / valid_loader
    """
    from stable_baselines3.common.evaluation import evaluate_policy

    # Determine how often you want to perform validation
    eval_freq = 1000  # perform validation every 1000 timesteps
    n_eval_episodes = 10  # number of episodes to evaluate
    model_save_path = "/path/to/your/model.zip"

    # Create a separate validation environment
    valid_env = ImagePerturbEnv(dataloader=valid_loader, model=load_model(), steps_per_episode=steps_per_episode, verbose=True)

    # Training loop with periodic validation
    for i in range(0, total_timesteps, eval_freq):
        # Train the model for `eval_freq` steps
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)

        # Evaluate the model on the validation environment
        mean_reward, std_reward = evaluate_policy(model, valid_env, n_eval_episodes=n_eval_episodes)
        logging.info(f"Validation results: Mean reward: {mean_reward} +/- {std_reward}")

        # Save the model periodically
        model.save(f"{model_save_path}_step{i}")

    """
