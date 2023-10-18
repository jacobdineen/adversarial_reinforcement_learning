# -*- coding: utf-8 -*-
"""
main environment logic
to be used downstream for model trianing
"""
import logging
from typing import Any, Dict, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50

logging.basicConfig(level=logging.INFO)


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

    def __init__(self, dataloader: Any, model: torch.nn.Module, attack_budget: int = 1000):
        """
        Initialize the environment.

        Args:
            dataloader: PyTorch DataLoader iterator.
            model: The deep learning model to evaluate against.
            attack_budget: The number of steps available to perturb the image.
        """
        self.dataloader = iter(dataloader)
        self.model = model
        self.model.eval()

        self.image, self.target_class = next(self.dataloader)
        self.original_image = self.image.clone()  # Save the original image
        self.image_shape = self.image.shape  # torch.Size([1, 3, 224, 224]) for cifar
        self.action_space = spaces.Discrete(self.image_shape[1] * self.image_shape[2])
        self.observation_space = spaces.Box(low=0, high=1, shape=self.image_shape, dtype=np.float32)

        self.attack_budget = attack_budget
        self.current_attack_count = 0

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take a step using an action.

        Args:
            action: An integer action from the action space.

        Returns:
            Tuple: A tuple containing:
                - perturbed_image: The new state (perturbed image)
                - reward: The reward for the taken action
                - done: Flag indicating if the episode has ended
                - info: Additional information (empty in this case)
        """
        self.current_attack_count += 1

        perturbed_image = self.image.clone()
        # This is interesting - this channels a specific x,y coordinate across channels
        # so this reduce the action space quite a bit
        x, y = divmod(action, self.image_shape[2])
        perturbed_image[0, :, x, y] = 0

        with torch.no_grad():
            original_output = self.model(self.image)
            original_prob = F.softmax(original_output, dim=1)[0][self.target_class].item()

            perturbed_output = self.model(perturbed_image)
            perturbed_prob = F.softmax(perturbed_output, dim=1)[0][self.target_class].item()

        reward = original_prob - perturbed_prob  # degradation in the probability of the target class

        done = self.current_attack_count >= self.attack_budget  # continue until attack budget reached
        if done:
            logging.info("attack budget reached. Sampling new image")
            self.reset()

        self.image = perturbed_image

        return perturbed_image, reward, done, {}

    def reset(self) -> torch.Tensor:
        """
        Reset the environment state.

        Returns:
            The new state (image) after resetting.
        """
        self.image, self.target_class = next(self.dataloader)
        self.original_image = self.image.clone()  # Save the original image again
        self.current_attack_count = 0


if __name__ == "__main__":

    def highlight_changes(original_image: np.ndarray, perturbed_image: np.ndarray) -> np.ndarray:
        highlighted_image = np.copy(perturbed_image)

        # Find the coordinates where the original and perturbed images differ
        diff_coords = np.where(np.any(original_image != perturbed_image, axis=-1))

        if diff_coords[0].size == 0:
            return highlighted_image  # No changes, so return the original perturbed_image

        # Color those pixels red in the highlighted image [255, 0, 0] in RGB
        highlighted_image[diff_coords] = [255, 0, 0]

        return highlighted_image

    transform_chain = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )

    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = resnet50(pretrained=True)
    env = ImagePerturbEnv(dataloader=dataloader, model=model)

    num_steps = 500
    for _ in range(num_steps):
        original_image = env.image.clone().detach().cpu().numpy().squeeze()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        perturbed_image = next_state.clone().detach().cpu().numpy().squeeze()

        print(f"Reward: {reward}")
        if done:
            env.reset()

    class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    with torch.no_grad():
        original_output = env.model(env.original_image)
        original_prob = F.softmax(original_output, dim=1)[0][env.target_class].item()

        print(f"Original Model Output: {original_output}")
        print(f"Original Model Probability: {original_prob}")

        perturbed_output = env.model(env.image)
        perturbed_prob = F.softmax(perturbed_output, dim=1)[0][env.target_class].item()

        print(f"Perturbed Model Output: {perturbed_output}")
        print(f"Perturbed Model Probability: {perturbed_prob}")

    target_class_label = class_labels[env.target_class]
    original_image = env.original_image.clone().detach().cpu().numpy().squeeze()
    perturbed_image = next_state.clone().detach().cpu().numpy().squeeze()

    # # Find the coordinates of pixels that have changed
    # changed_pixels = np.where(original_image != perturbed_image)
    # print(f"Number of pixels changed: {changed_pixels[0].size}")
    # print('changed pixels: ', changed_pixels)
    # print('shape', changed_pixels[0][0].shape)
    # highlighted_image = original_image.copy()  # Create a copy of the original image
    # # Set only the changed pixels to red
    # highlighted_image[changed_pixels[0], changed_pixels[1], :] = [255, 0, 0]  # Red

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title(f"Original\nClass: {target_class_label} (Prob: {original_prob:.10f})")
    plt.imshow(np.transpose(original_image, (1, 2, 0)))

    plt.subplot(1, 3, 2)
    plt.title(f"Perturbed\nClass: {target_class_label} (Prob: {perturbed_prob:.10f})")
    plt.imshow(np.transpose(perturbed_image, (1, 2, 0)))

    # plt.subplot(1, 3, 3)
    # plt.title('Highlighted')
    # plt.imshow(np.transpose(highlighted_image, (1, 2, 0)).squeeze())
    plt.show()
