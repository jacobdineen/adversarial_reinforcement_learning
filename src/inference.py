import argparse
import gc
import logging
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from src.env import BlockBasedPerturbEnv, SinglePixelPerturbEnv
from src.plotting import plt_helper
from src.rewards import reward_functions
from src.utils import (
    EndlessDataLoader,
    RewardLoggerCallback,
    get_dataloaders,
    load_model,
    set_seed,
)

train_loader, valid_loader, test_loader = get_dataloaders(
    dataset_name='mnist',
    batch_size=50,
    val_split=0.2,
    seed=42,
    train_limit=100,
)
from src.rewards import reward_functions

rew =  "reward_one"
reward_func = reward_functions[rew]

model = load_model('mnist')
test_env = BlockBasedPerturbEnv(dataloader=EndlessDataLoader(test_loader),model = model, reward_func = reward_func, steps_per_episode=len(test_loader))
ppo_model = PPO.load("/home/adv_rl/src/model_weights/ppo_mnist_mnist_episodes-10_trainlim-100")
print("Model loaded successfully")
actions = ppo_model.predict(test_env.images)
perturbed_image,reward, mean_reward,a,b= test_env.step(actions[0])
# print('reward',reward)
top_k_reward, top_k_indices = torch.topk(reward, 10)
# print('top_k_indices',top_k_indices)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Assuming you have the necessary variables set up, including top_k_reward, top_k_indices, test_env.images, and model

# Number of images to display
num_images = 10

# Create a subplot with 5 rows and 4 columns
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(15, 12))

for idx, (reward, index) in enumerate(zip(top_k_reward, top_k_indices)):
    row = idx // 2  # Compute the row index
    col = idx % 2   # Compute the column index

    input_image_with_highest_reward = test_env.images[index].numpy().squeeze(0)
    perturbed_image_with_highest_reward = test_loader.dataset.data[index].numpy()

    # Convert images to uint8 format
    input_image_with_highest_reward = (input_image_with_highest_reward * 255).astype('uint8')
    perturbed_image_with_highest_reward = (perturbed_image_with_highest_reward * 255).astype('uint8')

    # Predictions
    raw_output_input_image = model(torch.from_numpy(input_image_with_highest_reward).unsqueeze(0).unsqueeze(0).float())
    softmax_output_input_image = F.softmax(raw_output_input_image, dim=1)
    predicted_class_input_image = torch.argmax(softmax_output_input_image)
    predicted_confidence_input_image = torch.max(softmax_output_input_image)

    raw_output_perturbed_image = model(torch.from_numpy(perturbed_image_with_highest_reward).unsqueeze(0).unsqueeze(0).float())
    softmax_output_perturbed_image = F.softmax(raw_output_perturbed_image, dim=1)
    predicted_class_perturbed_image = torch.argmax(softmax_output_perturbed_image)
    predicted_confidence_perturbed_image = torch.max(softmax_output_perturbed_image)

    # Display input image, predicted class, and confidence on the left side
    axes[row, col * 2].imshow(input_image_with_highest_reward, cmap='gray')
    axes[row, col * 2].set_title(f'Reward: {reward.item():.4f}\nPredicted Class: {predicted_class_input_image.item()}\nConfidence: {predicted_confidence_input_image.item():.4f}')
    axes[row, col * 2].axis('off')

    # Display perturbed image, predicted class, and confidence on the right side
    axes[row, col * 2 + 1].imshow(perturbed_image_with_highest_reward, cmap='gray')
    axes[row, col * 2 + 1].set_title(f'Reward: {reward.item():.4f}\nPredicted Class: {predicted_class_perturbed_image.item()}\nConfidence: {predicted_confidence_perturbed_image.item():.4f}')
    axes[row, col * 2 + 1].axis('off')

# Save the subplot as an image
plt.tight_layout()
plt.savefig('subplot_image.png')
plt.show()


