# -*- coding: utf-8 -*-
"""
main environment logic
to be used downstream for model trianing
"""

import logging

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cpu")


class ImagePerturbEnv(gym.Env):
    """
    A custom gym environment for perturbing batches of images.
    """

    def __init__(self, dataloader, model, reward_func, steps_per_episode=100, verbose=False, seed=None):
        self.dataloader = iter(dataloader)
        self.model = model.to(DEVICE)
        self.model.eval()
        self.batch_size = dataloader.dataloader.batch_size
        self.images, self.target_classes = next(self.dataloader)
        self.images = self.images.to(DEVICE)
        self.target_classes = self.target_classes.to(DEVICE)
        self.original_images = self.images.clone()

        self.image_shape = self.images.shape[1:]  # Assuming shape [batch_size, channels, height, width]
        total_actions = self.image_shape[1] * self.image_shape[2]

        self.action_space = spaces.MultiDiscrete([total_actions] * self.batch_size)
        batched_shape = (self.batch_size,) + self.image_shape
        self.observation_space = spaces.Box(low=0, high=1, shape=batched_shape, dtype=np.float32)

        self.steps_per_episode = steps_per_episode
        self.current_step = 0
        self.episode_count = 0
        self.verbose = verbose
        self.seed = seed
        self.reward_func = reward_func

    def step(self, actions: int) -> tuple[torch.Tensor, float, bool, dict]:
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

            reward = self.compute_reward(self.images[i], perturbed_images[i], self.current_step)
            rewards.append(reward)

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

        return perturbed_images, np.mean(rewards), [done] * self.batch_size, False, {}

    def compute_reward(self, original_image, perturbed_image, current_step):
        original_image = original_image.unsqueeze(0).to(DEVICE)
        perturbed_image = perturbed_image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            original_output = self.model(original_image)
            original_prob = F.softmax(original_output, dim=1)[0][self.target_classes[0]].item()

            perturbed_output = self.model(perturbed_image)
            perturbed_prob = F.softmax(perturbed_output, dim=1)[0][self.target_classes[0]].item()

        reward_arguments = {
            "original_output": original_output,
            "perturbed_output": perturbed_output,
            "original_prob": original_prob,
            "perturbed_prob": perturbed_prob,
            "current_step": current_step,
            "target_class": self.target_classes[0].item(),
        }

        return self.reward_func(self, **reward_arguments)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """
        Reset the environment state if the num times to sample has been reached.
        This is equiv to num_images // batch_size steps
        Otherwise, reset the image to the original image and continue sampling.

        Returns:
            The new state (image) after resetting.
        """
        # Note: super().reset(seed=seed) might not be necessary unless the superclass
        # gym.Env specifically requires it, as gym.Env's reset method does not accept a seed parameter.
        # If your superclass does not use the seed, you can remove this line.
        super().reset(seed=seed)  # Only if necessary.
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
