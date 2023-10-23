# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50
from env import ImagePerturbEnv

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

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
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state)
        action_index = q_values.max(1)[1].item()
    return action_index

def train(dataloader: CIFAR10, model: resnet50) -> QNetwork:
    """
    Creates an environment from the dataloader and model and trains a Q-Network - returns the trained Q-Network.
    """
    env = ImagePerturbEnv(dataloader=dataloader, model=model)

    input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Initialize Q-Network and Optimizer
    q_net = QNetwork(input_shape, n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)

    # Initialize Replay Buffer and Other Training Parameters
    buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 0.1
    update_freq = 1

    # Main Training Loop
    n_episodes = 10

    for episode in range(n_episodes):
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
                # states: tuple[torch.Tensor]
                # actions: tuple[int]
                # rewards: tuple[float]
                # next_states: tuple[torch.Tensor]
                # dones: tuple[bool]
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states, dim=0)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.stack(next_states, dim=0)
                dones = torch.FloatTensor(dones)

                curr_Q = q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_Q = q_net(next_states).max(1)[0]
                target_Q = rewards + (gamma * next_Q) * (1 - dones)

                loss = nn.functional.mse_loss(curr_Q, target_Q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode}: Total Reward: {episode_reward}")

        if episode % update_freq == 0:
            epsilon *= 0.99

    print("Completed Training")
    return q_net

if __name__ == "__main__":
    # Code to make the dataloader and model taken from env.py main conditional
    transform_chain = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model = resnet50(pretrained=True)
    trained_qnet = train(dataloader, model)

    # Save the trained Q-Network
    SAVE_PATH = r"./src/model_weights/trained_qnet.pth"
    torch.save(trained_qnet.state_dict(), SAVE_PATH)
