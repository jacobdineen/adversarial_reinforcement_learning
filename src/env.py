# -*- coding: utf-8 -*-
"""
main environment logic
to be used downstream for model trianing
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

logging.basicConfig(level=logging.INFO)
DEVICE = torch.device("cpu")


class AbstractImagePerturbEnv(gym.Env, ABC):
    """
    An abstract base class for perturbing batches of images with different action spaces and step logic.
    """

    def __init__(
        self,
        dataloader,
        model,
        reward_func,
        steps_per_episode=100,
        verbose=False,
        seed=None,
    ):
        self.dataloader = iter(dataloader)
        self.model = model.to(DEVICE)
        self.model.eval()
        self.batch_size = dataloader.dataloader.batch_size
        self.images, self.target_classes = next(self.dataloader)
        self.images = self.images.to(DEVICE)
        self.target_classes = self.target_classes.to(DEVICE)
        self.original_images = self.images.clone()

        self.image_shape = self.images.shape[1:]  # Assuming shape [batch_size, channels, height, width]
        self.steps_per_episode = steps_per_episode
        self.current_step = 0
        self.episode_count = 0
        self.verbose = verbose
        self.seed = seed
        self.reward_func = reward_func

    @abstractmethod
    def step(self, actions: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Take a step using an action. This method needs to be implemented by subclasses.
        """
        pass

    def compute_reward(self, original_images, perturbed_images, current_step):
        original_images = original_images.to(DEVICE)
        perturbed_images = perturbed_images.to(DEVICE)
        with torch.no_grad():
            original_output = self.model(original_images)
            original_probs = F.softmax(original_output, dim=1)[
                torch.arange(len(self.target_classes)), self.target_classes
            ]

            perturbed_output = self.model(perturbed_images)
            perturbed_probs = F.softmax(perturbed_output, dim=1)[
                torch.arange(len(self.target_classes)), self.target_classes
            ]

        reward_arguments = {
            "original_output": original_output,
            "perturbed_output": perturbed_output,
            "original_probs": original_probs,
            "perturbed_probs": perturbed_probs,
            "current_step": current_step,
            "target_classes": self.target_classes,
        }

        return self.reward_func(self, **reward_arguments)

    def reset(self, seed: Union[int,None] = None) -> Tuple[torch.Tensor, dict]:
        super().reset(seed=seed)  # Call this only if necessary based on your gym.Env implementation.
        self.current_step = 0
        self.episode_count += 1
        self.dataloader = iter(self.dataloader)
        self.images, self.target_classes = next(self.dataloader)
        self.images = self.images.to(DEVICE)
        self.target_classes = self.target_classes.to(DEVICE)
        self.original_images = self.images.clone()

        if self.verbose:
            logging.info("Environment reset.")

        return self.images, {}

    @property
    @abstractmethod
    def action_space(self):
        """
        Define the action space. This property needs to be implemented by subclasses.
        """
        pass


class SinglePixelPerturbEnv(AbstractImagePerturbEnv):
    """
    A concrete implementation of the abstract base class for perturbing batches of images.
    """

    @property
    def action_space(self):
        total_actions = self.image_shape[1] * self.image_shape[2]
        return spaces.MultiDiscrete([total_actions] * self.batch_size)

    @property
    def observation_space(self):
        batched_shape = (self.batch_size,) + self.image_shape
        return spaces.Box(low=0, high=1, shape=batched_shape, dtype=np.float32)

    def step(self, actions: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Take a step using an action. This applies an action of a batch of images
        reward is averaged over a batch to return a scalar

        Args:
            actions (int): The action to take.

        Returns:
            tuple: A tuple containing the next state, the reward, whether the episode is done, and additional info.

        """
        perturbed_images = self.images.clone().to(DEVICE)
        rewards = []

        for i, action in enumerate(actions):
            channel, temp = divmod(action, self.image_shape[1] * self.image_shape[2])
            x, y = divmod(temp, self.image_shape[2])
            for channel in range(self.image_shape[0]):
                perturbed_images[i, channel, x, y] = 0

        rewards = self.compute_reward(self.images, perturbed_images, self.current_step)

        self.current_step += 1
        done = self.current_step >= self.steps_per_episode

        # Load next batch of images
        if done or self.current_step % self.batch_size == 0:
            try:
                self.images, self.target_classes = next(self.dataloader)
                self.images = self.images.to(DEVICE)
                self.target_classes = self.target_classes.to(DEVICE)
            except StopIteration:
                # Handle case where dataloader is exhausted
                pass

        return (
            perturbed_images,
            rewards,
            torch.mean(rewards),
            [done] * self.batch_size,
            False,
            {},
        )


class BlockBasedPerturbEnv(AbstractImagePerturbEnv):
    """
    A concrete implementation of the abstract base class for perturbing batches of images.
    Perturbs a block of pixels rather than a single one
    """

    def __init__(self, dataloader, model, reward_func, steps_per_episode=100, verbose=False, seed=None, num_blocks=20):
        super().__init__(dataloader, model, reward_func, steps_per_episode, verbose, seed)
        self.num_blocks = num_blocks

        # Assuming self.images is a tensor with shape [batch_size, channels, height, width]
        self.full_image_shape = self.images.shape  # This includes the batch dimension
        self.image_shape = self.images.shape[1:]  # Excludes the batch dimension

    @property
    def action_space(self):
        return spaces.MultiDiscrete([self.num_blocks] * self.batch_size)

    @property
    def observation_space(self):
        batched_shape = (self.batch_size,) + self.image_shape
        return spaces.Box(low=0, high=1, shape=batched_shape, dtype=np.float32)

    def step(self, actions: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Take a step using an action. This applies an action of a batch of images
        reward is averaged over a batch to return a scalar

        Args:
            actions (int): The actions to take.

        Returns:
            tuple: A tuple containing the next state, the reward, whether the episode is done, and additional info.

        """
        perturbed_images = self.images.clone().to(DEVICE)
        rewards = []

        block_size_x = self.image_shape[1] // int(self.image_shape[1] / np.sqrt(self.num_blocks))
        block_size_y = self.image_shape[2] // int(self.image_shape[2] / np.sqrt(self.num_blocks))
        blocks_per_row = self.image_shape[2] // block_size_x
        rewards = []
        for i, action in enumerate(actions):
            # Map the action to a unique block for each image
            block_idx_x = action % blocks_per_row
            block_idx_y = action // blocks_per_row

            # Shut off the pixels in the block for each channel
            x_start = block_idx_x * block_size_x
            y_start = block_idx_y * block_size_y

            num_channels = perturbed_images.shape[1]  # Dynamically get the number of channels
            for current_channel in range(num_channels):
                perturbed_images[
                    i, current_channel, x_start : x_start + block_size_x, y_start : y_start + block_size_y
                ] = (
                    1
                    - perturbed_images[
                        i, current_channel, x_start : x_start + block_size_x, y_start : y_start + block_size_y
                    ]
                )

        rewards = self.compute_reward(self.images, perturbed_images, self.current_step)

        self.current_step += 1
        done = self.current_step >= self.steps_per_episode

        # Load next batch of images
        if done or self.current_step % self.batch_size == 0:
            try:
                self.images, self.target_classes = next(self.dataloader)
                self.images = self.images.to(DEVICE)
                self.target_classes = self.target_classes.to(DEVICE)
            except StopIteration:
                # Handle case where dataloader is exhausted
                pass

        return (
            perturbed_images,
            # torch.mean(rewards),
            rewards,
            [done] * self.batch_size,
            False,
            {},
        )


# class ImagePerturbEnv(gym.Env):
#     """
#     A custom gym environment for perturbing batches of images.
#     """

#     def __init__(
#         self,
#         dataloader,
#         model,
#         reward_func,
#         steps_per_episode=100,
#         verbose=False,
#         seed=None,
#     ):
#         self.dataloader = iter(dataloader)
#         self.model = model.to(DEVICE)
#         self.model.eval()
#         self.batch_size = dataloader.dataloader.batch_size
#         self.images, self.target_classes = next(self.dataloader)
#         self.images = self.images.to(DEVICE)
#         self.target_classes = self.target_classes.to(DEVICE)
#         self.original_images = self.images.clone()

#         self.image_shape = self.images.shape[1:]  # Assuming shape [batch_size, channels, height, width]
#         total_actions = self.image_shape[1] * self.image_shape[2]

#         self.action_space = spaces.MultiDiscrete([total_actions] * self.batch_size)
#         batched_shape = (self.batch_size,) + self.image_shape
#         self.observation_space = spaces.Box(low=0, high=1, shape=batched_shape, dtype=np.float32)

#         self.steps_per_episode = steps_per_episode
#         self.current_step = 0
#         self.episode_count = 0
#         self.verbose = verbose
#         self.seed = seed
#         self.reward_func = reward_func

#     def step(self, actions: int) -> tuple[torch.Tensor, float, bool, dict]:
#         """
#         Take a step using an action. This applies an action of a batch of images
#         reward is averaged over a batch to return a scalar

#         Args:
#             actions (int): The action to take.

#         Returns:
#             tuple: A tuple containing the next state, the reward, whether the episode is done, and additional info.

#         """
#         perturbed_images = self.images.clone().to(DEVICE)
#         rewards = []

#         for i, action in enumerate(actions):
#             channel, temp = divmod(action, self.image_shape[1] * self.image_shape[2])
#             x, y = divmod(temp, self.image_shape[2])
#             for channel in range(self.image_shape[0]):
#                 perturbed_images[i, channel, x, y] = 0

#         rewards = self.compute_reward(self.images, perturbed_images, self.current_step)

#         self.current_step += 1
#         done = self.current_step >= self.steps_per_episode

#         # Load next batch of images
#         if done or self.current_step % self.batch_size == 0:
#             try:
#                 self.images, self.target_classes = next(self.dataloader)
#                 self.images = self.images.to(DEVICE)
#                 self.target_classes = self.target_classes.to(DEVICE)
#             except StopIteration:
#                 # Handle case where dataloader is exhausted
#                 pass

#         return (
#             perturbed_images,
#             torch.mean(rewards),
#             [done] * self.batch_size,
#             False,
#             {},
#         )

#     def compute_reward(self, original_images, perturbed_images, current_step):
#         original_images = original_images.to(DEVICE)
#         perturbed_images = perturbed_images.to(DEVICE)
#         with torch.no_grad():
#             original_output = self.model(original_images)
#             original_probs = F.softmax(original_output, dim=1)[
#                 torch.arange(len(self.target_classes)), self.target_classes
#             ]

#             perturbed_output = self.model(perturbed_images)
#             perturbed_probs = F.softmax(perturbed_output, dim=1)[
#                 torch.arange(len(self.target_classes)), self.target_classes
#             ]

#         reward_arguments = {
#             "original_output": original_output,
#             "perturbed_output": perturbed_output,
#             "original_probs": original_probs,
#             "perturbed_probs": perturbed_probs,
#             "current_step": current_step,
#             "target_classes": self.target_classes,
#         }

#         return self.reward_func(self, **reward_arguments)

#     def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
#         """
#         Reset the environment state if the num times to sample has been reached.
#         This is equiv to num_images // batch_size steps
#         Otherwise, reset the image to the original image and continue sampling.

#         Returns:
#             The new state (image) after resetting.
#         """
#         # Note: super().reset(seed=seed) might not be necessary unless the superclass
#         # gym.Env specifically requires it, as gym.Env's reset method does not accept a seed parameter.
#         # If your superclass does not use the seed, you can remove this line.
#         super().reset(seed=seed)  # Only if necessary.
#         self.current_step = 0
#         self.episode_count += 1
#         self.dataloader = iter(self.dataloader)
#         self.images, self.target_classes = next(self.dataloader)
#         self.images = self.images.to(DEVICE)
#         self.target_classes = self.target_classes.to(DEVICE)
#         self.original_images = self.images.clone()

#         if self.verbose:
#             logging.info("Environment reset.")

#         return self.images, {}
