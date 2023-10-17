# -*- coding: utf-8 -*-
import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torchvision import transforms
from torchvision.models import resnet50


class ImagePerturbEnv(gym.Env):
    def __init__(self, image, target_class, budget=10, perturbations_per_step=1):
        super().__init__()

        # Load a pre-trained ResNet50 classifier
        self.model = resnet50(pretrained=True)
        self.model.eval()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=image.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=image.shape, dtype=np.uint8)

        self.image = image
        self.target_class = target_class

        self.budget = budget
        self.perturbations_per_step = perturbations_per_step
        self.perturbed_count = 0

    def step(self, action):
        perturbed_image = self.image
        for _ in range(self.perturbations_per_step):
            # Apply perturbation to the image
            perturbed_image = np.clip(perturbed_image + action, 0, 255)

        # Convert to PyTorch tensor
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(perturbed_image).unsqueeze(0)

        # Get classifier's prediction for the unperturbed image
        with torch.no_grad():
            original_output = self.model(transform(self.image).unsqueeze(0))
            original_prob = F.softmax(original_output, dim=1)[0][self.target_class].item()

        # Get classifier's prediction for the perturbed image
        with torch.no_grad():
            perturbed_output = self.model(input_tensor)
            perturbed_prob = F.softmax(perturbed_output, dim=1)[0][self.target_class].item()

        # Calculate reward as degradation in probability of the correct class
        print("original_prob: ", original_prob)
        print("perturbed_prob: ", perturbed_prob)
        reward = original_prob - perturbed_prob

        # Update perturbation count and check if perturbation limit is reached
        self.perturbed_count += self.perturbations_per_step
        done = self.perturbed_count >= self.budget

        # Update the current state
        self.image = perturbed_image

        info = {}
        return perturbed_image, reward, done, info

    def reset(self):
        self.perturbed_count = 0
        return self.image

    def render(self, mode="human"):
        # For visualizing the perturbed image, can use matplotlib or any other library
        pass

    def close(self):
        pass


if __name__ == "__main__":
    # Create a dummy 224x224x3 image with random values between [0,255]
    dummy_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Initialize the environment with the dummy image, target class, a budget of 5 and 2 perturbations per step
    env = ImagePerturbEnv(dummy_image, target_class=123, budget=5, perturbations_per_step=2)

    # Interact with the environment
    num_steps = 10
    for _ in range(num_steps):
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        print(f"Reward: {reward}")
        if done:
            env.reset()

    env.close()
