# -*- coding: utf-8 -*-

import logging
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from env import ImagePerturbEnv
from src.utils import get_cifar_dataloader, load_model

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# An episode is a sequence of perturbations on the same image until
# attack budget is reached. The reward is the change in the model's
# confidence in the true class of the image.


# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        return self.fc5(x)


# Epsilon-Greedy Policy
def policy(state, epsilon, q_net, env) -> int:
    """
    With probability epsilon, return the index of a random action.
    Otherwise, return the index of the action that maximizes the Q-value.
    """
    action_index: int = None
    if random.random() < epsilon:
        action_index = env.action_space.sample()
    else:
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = q_net(state)
        action_index = q_values.max(1)[1].item()
    return action_index


def train(env: ImagePerturbEnv, q_net: QNetwork, optimizer: optim.Optimizer, num_episodes: int = 100) -> QNetwork:
    """
    Creates an environment from the dataloader and model and trains a Q-Network - returns the trained Q-Network.
    """
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Initialize Q-Network and Optimizer
    q_net = QNetwork(input_shape, n_actions).to(DEVICE)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)

    # Initialize Replay Buffer and Other Training Parameters
    buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 0.1
    update_freq = 1

    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Environment Interaction
            action = policy(state, epsilon, q_net, env)
            next_state, reward, done, _ = env.step(action)

            # Store Experience
            buffer.append((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            # Update Network
            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states, dim=0).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.stack(next_states, dim=0).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                curr_Q = q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_Q = q_net(next_states).max(1)[0]
                target_Q = rewards + (gamma * next_Q) * (1 - dones)

                loss = nn.functional.mse_loss(curr_Q, target_Q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # logging.info(f"Episode {episode}: Total Reward: {episode_reward}")

        episode_rewards.append(episode_reward)

        if episode % update_freq == 0:
            epsilon *= 0.99

    logging.info(f"Completed Training")
    return q_net, episode_rewards


if __name__ == "__main__":
    # cifar10 dataloader
    dataloader = get_cifar_dataloader()
    # classififer
    model = load_model()
    # env
    env = ImagePerturbEnv(
        dataloader=dataloader, model=model, attack_budget=50, lambda_=1
    )  # lambda_ used in reward to penalize more changes

    # Initialize Q-Network and Optimizer here, and pass them to train
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    q_net = QNetwork(input_shape, n_actions).to(DEVICE)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    num_episodes = 500
    trained_qnet, episode_rewards = train(env, q_net, optimizer, num_episodes)
    print(episode_rewards)

    # Plot the episode rewards
    plt.figure()
    plt.plot(range(num_episodes), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")
    plt.show()

    # Save the trained Q-Network
    SAVE_PATH = "src/model_weights/trained_qnet.pth"
    torch.save(trained_qnet.state_dict(), SAVE_PATH)
    logging.info(f"Trained Q-Network saved successfully to {SAVE_PATH}")
