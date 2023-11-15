# -*- coding: utf-8 -*-
"""
main environment logic
to be used downstream for model trianing
"""
import logging
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

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
        steps_per_episode: int = 100,
        verbose: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize the environment.

        Args:
            dataloader: PyTorch DataLoader iterator.
            model: The deep learning model to evaluate against.
            lambda_: hyperparameter that controls how severely we penalize non-sparse solutions. A higher LAMBDA means a steeper penalty.
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
        
        self.num_blocks = 100
        block_size_x = self.image_shape[2] // int(self.image_shape[2] / np.sqrt(self.num_blocks))
        block_size_y = self.image_shape[3] // int(self.image_shape[3] / np.sqrt(self.num_blocks))

    # Update the action space to represent the total number of blocks
        self.action_space = spaces.Discrete(self.num_blocks)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.image_shape, dtype=np.float32)
        self.steps_per_episode = steps_per_episode
        self.current_step = 0
        self.episode_count = 0
        self.verbose = verbose
        self.seed = seed

        logging.info(f"Initialized ImagePerturbEnv with the following parameters:")
        logging.info(f"Action Space Size: {self.num_blocks}")
        logging.info(f"Observation Space Shape: {self.observation_space.shape}")
        logging.info(f"Initial Image Shape: {self.image_shape}")
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Take a step using an action.

        Args:
            action: An integer action from the action space.
            Currently - an action corresponds to shutting off a block of pixels.

        Returns:
            Tuple: A tuple containing:
                - perturbed_image: The new state (perturbed image)
                - reward: The reward for the taken action
                - done: Flag indicating if the episode has ended
                - truncated: Flag indicating if the episode was truncated (currently unused)
                - info: Additional information (empty in this case)
        """
        perturbed_image = self.image.clone().to(DEVICE)

        # Calculate the block size based on the image dimensions
        block_size_x = self.image_shape[2] // int(self.image_shape[2] / np.sqrt(self.num_blocks))
        block_size_y = self.image_shape[3] // int(self.image_shape[3] / np.sqrt(self.num_blocks))

        # Map the action to a unique block
        blocks_per_row = int(self.image_shape[2] / block_size_x)
        block_idx_x = action % blocks_per_row
        block_idx_y = action // blocks_per_row

        # Shut off the pixels in the block
        x_start = block_idx_x * block_size_x
        y_start = block_idx_y * block_size_y

        for current_channel in range(self.image_shape[1]):
            perturbed_image[0, current_channel, x_start:x_start + block_size_x, y_start:y_start + block_size_y] = 1 - perturbed_image[0, current_channel, x_start:x_start + block_size_x, y_start:y_start + block_size_y]
        
        reward = self.compute_reward(self.image, perturbed_image)

        # Increment the step counter and check if the episode should end
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode

        # Grab a new image after each step
        self.image, self.target_class = next(self.dataloader)

        return perturbed_image, reward, done, False, {}

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

        original_prob = max(original_prob, 1e-8)  # for underflow issues
        reward = original_prob - perturbed_prob
        return reward

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment state if the num times to sample has been reached.
        Otherwise, reset the image to the original image and continue sampling.

        Returns:
            The new state (image) after resetting.
        """
        # Note: super().reset(seed=seed) might not be necessary unless the superclass
        # gym.Env specifically requires it, as gym.Env's reset method does not accept a seed parameter.
        # If your superclass does not use the seed, you can remove this line.
        super().reset(seed=seed)  # Only if necessary.

        # Reset the step if we reached the end of an episode
        if self.current_step >= self.steps_per_episode:
            self.current_step = 0
            self.episode_count += 1

            self.dataloader = iter(self.dataloader)
            self.image, self.target_class = next(self.dataloader)

            self.image = self.image.to(DEVICE)
            self.target_class = self.target_class.to(DEVICE)
            self.original_image = self.image.clone()

            if self.verbose:
                logging.info(f"Resetting environment with new refreshed dataloader")
                logging.info(f"episode count: {self.episode_count}")
                logging.info(f"current_step: {self.current_step}")

        info = dict()

        return self.image, info


# if __name__ == "__main__":
#     # This is mainly just for testing
#     # but can likely be lifted for other parts of the codebase

#     transform_chain = transforms.Compose(
#         [
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ]
#     )

#     dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     model = load_model()
#     env = ImagePerturbEnv(dataloader=dataloader, model=model, attack_budget=100)

#     num_steps = env.attack_budget - 1 * env.num_times_to_sample * 2
#     for _ in range(num_steps):
#         original_image = env.image.clone().detach().cpu().numpy().squeeze()
#         action = env.action_space.sample()
#         next_state, reward, done, _, _ = env.step(action)
#         perturbed_image = next_state.clone().detach().cpu().numpy().squeeze()
#         if done:
#             env.reset()

#     with torch.no_grad():
#         original_output = model(env.original_image)
#         original_prob, original_class = F.softmax(original_output, dim=1).max(dim=1)

#         perturbed_output = model(env.image)
#         perturbed_prob, perturbed_class = F.softmax(perturbed_output, dim=1).max(dim=1)

#         # print(f"Original Model Output: {original_output}")
#         print(f"Original Model class and Probability: {original_class.item()}, {original_prob.item()}")

#         # print(f"Perturbed Model Output: {perturbed_output}")
#         print(f"Perturbed Model class and Probability: {perturbed_class.item()}, {perturbed_prob.item()}")

#     original_image = env.original_image.clone().detach().cpu().numpy().squeeze()
#     perturbed_image = next_state.clone().detach().cpu().numpy().squeeze()

#     changed_pixels = np.where(original_image != perturbed_image)
#     print(f"Number of pixels changed: {len(changed_pixels[0])}")
#     print("Shape of original_image: ", original_image.shape)
#     print("Shape of perturbed_image: ", perturbed_image.shape)
#     print("Length of changed_pixels tuple: ", len(changed_pixels))

#     # Since you only have 3 dimensions [channel, height, width]
#     for i in range(len(changed_pixels[0])):
#         channel_idx, x_idx, y_idx = changed_pixels[0][i], changed_pixels[1][i], changed_pixels[2][i]
#         pixel_value = perturbed_image[channel_idx, x_idx, y_idx].item()
#         print(f"Pixel value in perturbed image at ({x_idx}, {y_idx}, channel: {channel_idx}): {pixel_value}")

#     original_image_T = np.transpose(original_image, (1, 2, 0))
#     highlighted_isolated = np.zeros_like(original_image_T)
#     for i in range(len(changed_pixels[0])):
#         channel_idx, x_idx, y_idx = changed_pixels[0][i], changed_pixels[1][i], changed_pixels[2][i]
#         highlighted_isolated[x_idx, y_idx, :] = [255, 0, 0]  # Red for channel 0

#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.title(f"Original (Class: {CLASSES[original_class.item()]}, Prob: {original_prob.item():.6f})")
#     plt.imshow(np.transpose(original_image, (1, 2, 0)))

#     plt.subplot(1, 3, 2)
#     plt.title(f"Perturbed (Class: {CLASSES[perturbed_class.item()]}, Prob: {perturbed_prob.item():.6f})")
#     plt.imshow(np.transpose(perturbed_image, (1, 2, 0)))

#     plt.subplot(1, 3, 3)
#     plt.title("Highlighted Changes")
#     plt.imshow(highlighted_isolated)
#     plt.show()
