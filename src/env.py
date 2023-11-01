# -*- coding: utf-8 -*-
"""
main environment logic
to be used downstream for model trianing
"""
import logging
from typing import Any, Dict, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.utils import load_model

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class ImagePerturbEnv(gym.Env):
    """
    A custom gym environment for perturbing images and evaluating the changes
    against a deep learning model.

    Attributes:
        dataloader (iter): An iterator over a PyTorch DataLoader.
        model (torch.nn.Module): The deep learning model to test against.
        attack_budget (int): The number of actions to perturb the image.
        current_attack_count (int): Counter for perturbation actions.
        action_space (gym.spaces.Discrete): The action space.
        observation_space (gym.spaces.Box): The observation space.
        image (torch.Tensor): The current image from the dataloader.
        target_class (int): The class label for the current image.
        image_shape (Tuple[int, int, int]): The shape of the image tensor.
    """

    def __init__(
        self,
        dataloader: Any,
        model: torch.nn.Module,
        attack_budget: int = 20,
        lambda_: float = 1.0,
        num_times_to_sample: int = 5,
    ):
        """
        Initialize the environment.

        Args:
            dataloader: PyTorch DataLoader iterator.
            model: The deep learning model to evaluate against.
            attack_budget: The number of steps available to perturb the image.
            lambda_: hyperparameter that controls how severely we penalize non-sparse solutions. A higher LAMBDA means a steeper penalty.
            num_times_to_sample: number of times to repeat the same image before resampling the next one
        """
        logging.info("env device: {}".format(DEVICE))
        self.dataloader = iter(dataloader)
        self.model = model.to(DEVICE)
        self.model.eval()
        self.image, self.target_class = next(self.dataloader)
        self.image = self.image.to(DEVICE)
        self.target_class = self.target_class.to(DEVICE)
        self.original_image = self.image.clone()  # Save the original image
        self.image_shape = self.image.shape  # torch.Size([1, 3, 224, 224]) for cifar
        total_actions = self.image_shape[2] * self.image_shape[3]
        self.action_space = spaces.Discrete(total_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.image_shape, dtype=np.float32)
        self.attack_budget = attack_budget
        self.lambda_ = lambda_
        self.current_attack_count = 0
        self.num_times_to_sample = num_times_to_sample
        self.num_samples = 0
        self.image_counter = 0
        self.new_image = False

        logging.info(f"Initialized ImagePerturbEnv with the following parameters:")
        logging.info(f"Action Space Size: {total_actions}")
        logging.info(f"Observation Space Shape: {self.observation_space.shape}")
        logging.info(f"Attack Budget: {self.attack_budget}")
        logging.info(f"Initial Image Shape: {self.image_shape}")

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Take a step using an action.

        Args:
            action: An integer action from the action space.
            Currently - an action corresponds to shutting an (x,y) coordinate across ALL channels
            So one step modifies three separate pixels

        Returns:
            Tuple: A tuple containing:
                - perturbed_image: The new state (perturbed image)
                - reward: The reward for the taken action
                - done: Flag indicating if the episode has ended
                - truncated: Flag indicating if the episode was truncated (currently unused)
                - info: Additional information (empty in this case)
        """
        self.current_attack_count += 1
        perturbed_image = self.image.clone().to(DEVICE)

        channel, temp = divmod(
            action, self.image_shape[2] * self.image_shape[3]
        )  # channel, x*y coordinates in the image
        x, y = divmod(temp, self.image_shape[3])  # x, y coordinates in the image
        # perturbed_image[0, channel, x, y] = 0  # perturb the image by setting the pixel to 0
        for channel in range(self.image_shape[1]):
            perturbed_image[0, channel, x, y] = 0

        reward = self.compute_reward(self.image, perturbed_image)

        done = self.current_attack_count >= self.attack_budget  # continue until attack budget reached

        self.image = perturbed_image

        # "Whether the truncation condition outside the scope of the MDP is satisfied.
        # Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
        # Can be used to end the episode prematurely before a terminal state is reached.
        # If true, the user needs to call :meth:`reset`."
        truncated = False

        info = dict()

        return perturbed_image, reward, done, truncated, info

    def compute_reward(self, original_image: torch.Tensor, perturbed_image: torch.Tensor) -> float:
        """_summary_

        Args:
            original_image (torch.Tensor): og image
            perturbed_image (torch.Tensor): perturbed_image

        Returns:
            float: reward for step
        """
        original_image = original_image.to(DEVICE)
        perturbed_image = perturbed_image.to(DEVICE)
        with torch.no_grad():
            original_output = self.model(original_image)
            original_prob = F.softmax(original_output, dim=1)[0][self.target_class].item()

            perturbed_output = self.model(perturbed_image)
            perturbed_prob = F.softmax(perturbed_output, dim=1)[0][self.target_class].item()

        # sparsity = torch.nonzero(perturbed_image - original_image).size(0)
        reward = (original_prob - perturbed_prob) / original_prob  # * np.exp(-self.lambda_ * sparsity
        return reward

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment state if the num times to sample has been reached.
        Otherwise, reset the image to the original image.and continue sampling.

        Returns:
            The new state (image) after resetting.
        """
        super().reset(seed=seed)

        self.current_attack_count = 0
        self.num_samples += 1
        if self.num_samples >= self.num_times_to_sample:
            self.new_image = True  # Indicate that a new image has been sampled
            self.image, self.target_class = next(self.dataloader)
            self.image = self.image.to(DEVICE)
            self.target_class = self.target_class.to(DEVICE)
            self.original_image = self.image.clone()  # Save the new original image
            self.num_samples = 0  # Reset the counter
            self.image_counter += 1  # Increment the image counter
            logging.info(f"Sampling new image from dataloader. Image Counter: {self.image_counter}")
        else:
            self.new_image = False  # Indicate that it's the same image as before
            self.image = self.original_image.clone()  # Reset to the original image
            logging.info(f"Resetting to original image. Image Counter: {self.image_counter}")
        logging.info("Resetting the environment")

        info = dict()

        return self.image, info


if __name__ == "__main__":
    # This is mainly just for testing
    # but can likely be lifted for other parts of the codebase

    transform_chain = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = load_model()
    env = ImagePerturbEnv(dataloader=dataloader, model=model, attack_budget=100)

    num_steps = env.attack_budget - 1 * env.num_times_to_sample * 2
    for _ in range(num_steps):
        original_image = env.image.clone().detach().cpu().numpy().squeeze()
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)
        perturbed_image = next_state.clone().detach().cpu().numpy().squeeze()
        if done:
            env.reset()

    with torch.no_grad():
        original_output = model(env.original_image)
        original_prob, original_class = F.softmax(original_output, dim=1).max(dim=1)

        perturbed_output = model(env.image)
        perturbed_prob, perturbed_class = F.softmax(perturbed_output, dim=1).max(dim=1)

        # print(f"Original Model Output: {original_output}")
        print(f"Original Model class and Probability: {original_class.item()}, {original_prob.item()}")

        # print(f"Perturbed Model Output: {perturbed_output}")
        print(f"Perturbed Model class and Probability: {perturbed_class.item()}, {perturbed_prob.item()}")

    original_image = env.original_image.clone().detach().cpu().numpy().squeeze()
    perturbed_image = next_state.clone().detach().cpu().numpy().squeeze()

    changed_pixels = np.where(original_image != perturbed_image)
    print(f"Number of pixels changed: {len(changed_pixels[0])}")
    print("Shape of original_image: ", original_image.shape)
    print("Shape of perturbed_image: ", perturbed_image.shape)
    print("Length of changed_pixels tuple: ", len(changed_pixels))

    # Since you only have 3 dimensions [channel, height, width]
    for i in range(len(changed_pixels[0])):
        channel_idx, x_idx, y_idx = changed_pixels[0][i], changed_pixels[1][i], changed_pixels[2][i]
        pixel_value = perturbed_image[channel_idx, x_idx, y_idx].item()
        print(f"Pixel value in perturbed image at ({x_idx}, {y_idx}, channel: {channel_idx}): {pixel_value}")

    original_image_T = np.transpose(original_image, (1, 2, 0))
    highlighted_isolated = np.zeros_like(original_image_T)
    for i in range(len(changed_pixels[0])):
        channel_idx, x_idx, y_idx = changed_pixels[0][i], changed_pixels[1][i], changed_pixels[2][i]
        highlighted_isolated[x_idx, y_idx, :] = [255, 0, 0]  # Red for channel 0

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title(f"Original (Class: {CLASSES[original_class.item()]}, Prob: {original_prob.item():.6f})")
    plt.imshow(np.transpose(original_image, (1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.title(f"Perturbed (Class: {CLASSES[perturbed_class.item()]}, Prob: {perturbed_prob.item():.6f})")
    plt.imshow(np.transpose(perturbed_image, (1, 2, 0)))

    plt.subplot(1, 3, 3)
    plt.title("Highlighted Changes")
    plt.imshow(highlighted_isolated)
    plt.show()
