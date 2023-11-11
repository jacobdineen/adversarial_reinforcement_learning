# -*- coding: utf-8 -*-
"""
main train logic for RL agent
"""
import argparse
import gc
import logging

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from src.env import ImagePerturbEnv
from src.plotting import plt_helper

# from src.env import *
from src.rewards import *
from src.utils import EndlessDataLoader, get_dataloaders, load_model, set_seed

# This is to redirect the logging output to a file
LOGGING_OUTPUT_PATH = "./src/logs/sb3_log/"
new_logger = configure(LOGGING_OUTPUT_PATH, ["stdout", "csv"])

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42


class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq):
        super().__init__()
        self.check_freq = check_freq
        self.all_rewards = []
        self.all_lengths = []
        self.all_times = []
        self.policy_gradient_losses = []
        self.value_losses = []

    def _on_step(self) -> bool:
        # Check if it's time to log episode information
        if self.n_calls % self.check_freq == 0:
            # Retrieve the episode information from the buffer
            ep_info = self.model.ep_info_buffer
            if ep_info:
                # Extract the latest episode information
                info = ep_info[-1]
                self.all_rewards.append(info["r"])
                self.all_lengths.append(info["l"])
                self.all_times.append(info["t"])

        return True

    def get_training_info(self):
        """Retrieve the training information."""
        return {
            "rewards": self.all_rewards,
            "lengths": self.all_lengths,
            "times": self.all_times,
        }


reward_functions = {
    # Measures the change in the feature space caused by the perturbation.
    "reward_one": reward_distance,
    # Calculates how much the perturbation reduces the classifier's confidence.
    "reward_two": reward_improvement,
    # Similar to reward_improvement, but with a penalty for taking more steps.
    "reward_three": reward_time_decay,
    # Checks if the perturbation leads to a successful misclassification.
    "reward_four": reward_goal_achievement,
    # A composite reward combining several aspects of the perturbation task.
    "reward_five": reward_composite,
    # Quantifies the alteration of the model's output due to the perturbation.
    "reward_six": reward_output_difference,
    # Rewards the agent for decreasing the model's confidence in the correct class.
    "reward_seven": reward_target_prob_inversion,
}


parser = argparse.ArgumentParser(description="Train an agent to perturb images.")
parser.add_argument("--dataset_name", type=str, default="cifar", help="dataset to use. mnist of cifar")

parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
parser.add_argument(
    "--val_split",
    type=float,
    default=0.2,
    help="Holdout data for validation and testing.",
)
parser.add_argument(
    "--train_limit",
    type=int,
    default=1000,
    help="Training dataloader limit - useful for debugging shorter runs.",
)
parser.add_argument(
    "--verbose",
    type=bool,
    default=False,
    help="If you want environment logging to be verbose.",
)
parser.add_argument(
    "--prog_bar",
    type=bool,
    default=True,
    help="If you want to use tqdm for train loop.",
)
parser.add_argument(
    "--model_save_path",
    type=str,
    default="src/model_weights/ppo",
    help="Where to save trained PPO model",
)
parser.add_argument(
    "--model_performance_save_path",
    type=str,
    default="src/ppo_performance",
    help="Where to save ep info buff",
)
parser.add_argument(
    "--reward_func",
    type=str,
    choices=list(reward_functions.keys()),
    default="reward_seven",
    help="The name of the reward function to use.",
)


args = parser.parse_args()

episodes = args.num_episodes
batch_size = args.batch_size
val_split = args.val_split
train_limit = args.train_limit
verbose = args.verbose
prog_bar = args.prog_bar
model_performance_save_path = args.model_performance_save_path
dataset_name = args.dataset_name
selected_reward_func = reward_functions[args.reward_func]
model_save_path = args.model_save_path + "_" + dataset_name
model_save_path = f"{model_save_path}_{dataset_name}_episodes-{episodes}_trainlim-{train_limit}.zip"

if __name__ == "__main__":
    assert train_limit % 50 == 0, "train_limit must be a multiple of 50"
    logging.info(args)
    torch.cuda.empty_cache()
    gc.collect()
    set_seed(SEED)

    train_loader, valid_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=20 if dataset_name == "cifar10" else 50,
        val_split=val_split,
        seed=SEED,
        train_limit=train_limit,
    )

    steps_per_episode = len(train_loader)  # number of images to perturb per episode
    total_timesteps = episodes * steps_per_episode

    # classififer
    model = load_model(dataset_name=dataset_name)

    #
    env = ImagePerturbEnv(
        dataloader=EndlessDataLoader(train_loader),
        model=model,
        reward_func=selected_reward_func,  # Pass the reward function here
        steps_per_episode=steps_per_episode,
        verbose=verbose,
    )

    # env
    # Note the EndlessDataLoader wrapper
    # This is to ensure that when a dataloader has been exhausted, it is refreshed
    # by starting from the beginning
    train_env = ImagePerturbEnv(
        dataloader=EndlessDataLoader(train_loader),
        model=model,
        reward_func=selected_reward_func,
        steps_per_episode=steps_per_episode,
        verbose=verbose,
    )
    # eventually use this for validation
    valid_env = ImagePerturbEnv(
        dataloader=EndlessDataLoader(valid_loader),
        model=model,
        reward_func=selected_reward_func,
        steps_per_episode=steps_per_episode,
        verbose=verbose,
    )

    # Training here
    # callback necessary because defaults to last 100 episodes
    callback = RewardLoggerCallback(check_freq=steps_per_episode)
    model = PPO(
        "MlpPolicy",
        train_env,
        device=DEVICE,
        verbose=1,
        n_steps=steps_per_episode,
        batch_size=batch_size,
    )
    model.set_logger(new_logger)
    logging.info(f"model device: {model.device}")
    model.learn(total_timesteps=total_timesteps, progress_bar=prog_bar, callback=callback)
    logging.info("Training complete.")
    logging.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)

    #     valid_env = ImagePerturbEnv(
    #     dataloader=EndlessDataLoader(valid_loader), model=model, steps_per_episode=steps_per_episode, verbose=verbose
    # )

    # Evaluate the policy with the validation environment
    mean_reward, std_reward = evaluate_policy(model, valid_env, n_eval_episodes=10)
    logging.info(f"Validation results: Mean reward: {mean_reward} +/- {std_reward}")
    logging.info(f"Rewrd func calculated using{selected_reward_func}")

    episode_info = callback.get_training_info()
    df = pd.DataFrame(episode_info)
    df.rename(
        columns={"rewards": "rewards", "lengths": "lengths", "times": "times"},
        inplace=True,
    )
    df.to_csv(f"{model_performance_save_path}.csv")
    logging.info(f"Dataframe saved to {model_performance_save_path}")

    # If `plot_rewards_and_cumulative` requires just the rewards, you can extract them
    plt_helper(f"{LOGGING_OUTPUT_PATH}/progress.csv")
    # plot_rewards_and_cumulative(df["rewards"])
