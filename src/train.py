# -*- coding: utf-8 -*-

import logging
import random
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm import tqdm

from env import ImagePerturbEnv
from src.utils import get_cifar_dataloader, load_model

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# An episode is a sequence of perturbations on the same image until
# attack budget is reached. The reward is the change in the model's
# confidence in the true class of the image.


# fine tunes our pretrained resnet as a Q-network
class QNetworkPretrainedResnet(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.model = load_model()
        # Remove the last fully connected layer (classification head)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        # Add your new fully connected layer
        self.final_fc = nn.Linear(512, output_dim).to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)  # Move to device and convert to half precision
        x = x.squeeze(1)  # Remove the extra dimension if needed
        x = self.features(x)  # Pass through pre-trained layers

        # Global average pooling
        x = x.mean([2, 3])

        # Pass through your new fully connected layer
        x = self.final_fc(x)

        return x


# from scratch qnet training with conv layers
class QNetwork(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Note the '3' instead of '1'
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the output shape after convolutions to dynamically set Linear layer sizes

        # Fully Connected Layers
        self.fc1 = nn.Linear(36864, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, output_dim)

    def _forward_conv(self, x):
        x = x.squeeze(1)  # Remove the extra dimension
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        return self.fc5(x)


# Epsilon-Greedy Policy


def policy(state, epsilon, q_net1, q_net2, actions_taken) -> int:
    """
    With probability epsilon, return the index of a random action.
    Otherwise, return the index of the action that maximizes the average Q-value from both networks.

    If an action (pixel) is already selected, don't select it again.
    """
    if torch.rand(1).item() < epsilon:
        available_actions = torch.tensor([a for a in range(n_actions) if a not in actions_taken], dtype=torch.long)
        action_index = available_actions[torch.randint(0, len(available_actions), (1,))].item()
    else:
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values1 = q_net1(state).squeeze()
            q_values2 = q_net2(state).squeeze()
            avg_q_values = (q_values1 + q_values2) / 2  # Average the Q-values

            # Convert actions_taken set to tensor for indexing
            actions_taken_tensor = torch.tensor(list(actions_taken), dtype=torch.long)
            avg_q_values[actions_taken_tensor] = float("-inf")
            action_index = avg_q_values.argmax().item()

    actions_taken.add(action_index)
    return action_index, actions_taken


# double q learning
# one network chooses the action, the other evaluates the action
# the network that chooses the action is updated less frequently
# than the network that evaluates the action
def train(
    env: ImagePerturbEnv,
    q_net1: QNetwork,
    q_net2: QNetwork,
    optimizer1: optim.Optimizer,
    optimizer2: optim.Optimizer,
    num_episodes: int = 100,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    update_freq: int = 1,
    batch_size: int = 256,
    decay: float = 0.99,
) -> Tuple[QNetwork, QNetwork, List[float]]:
    buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=10000)
    episode_scalar_rewards = []
    all_actions_taken = []

    # each episode is a single image
    for episode in tqdm(range(num_episodes)):
        # logging.info(f"Starting episode {episode}")
        aggregated_episode_reward = 0  # Reset aggregated reward counter for the episode
        # that image can be sampled a number of times consecutively
        # with such a large action space, maybe it helps to see image multiple times to learn
        for i in range(env.num_times_to_sample + 1):
            # logging.info(f"Using image {env.image_counter}")
            state = env.reset()
            done = False
            episode_reward = 0
            actions_taken = set()

            while not done:
                action, actions_taken = policy(state, epsilon, q_net1, q_net2, actions_taken)
                # logging.info(actions_taken)
                next_state, reward, done, _ = env.step(action)
                buffer.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward  # Aggregate this sampling's reward

                if len(buffer) >= batch_size:
                    batch = random.sample(buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.stack(states).to(DEVICE)
                    actions = torch.LongTensor(actions).to(DEVICE)
                    rewards = torch.FloatTensor(rewards).to(DEVICE)
                    next_states = torch.stack(next_states).to(DEVICE)
                    dones = torch.FloatTensor(dones).to(DEVICE)

                    curr_Q1 = q_net1(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    curr_Q2 = q_net2(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

                    next_actions = q_net1(next_states).argmax(1)
                    next_Q = q_net2(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)

                    target_Q = rewards + (gamma * next_Q) * (1 - dones)

                    loss1 = nn.functional.smooth_l1_loss(curr_Q1, target_Q.detach())
                    loss2 = nn.functional.smooth_l1_loss(curr_Q2, target_Q.detach())

                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()

                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                if done:
                    # logging.info("attack budget reached. Sampling new image")
                    env.reset()

                # logging.info(f"Episode {episode} image sample {i} reward: {episode_reward}")
                aggregated_episode_reward += episode_reward  # Aggregate the reward for this episode
            # logging.info("reached done state")
        all_actions_taken.append(actions_taken.copy())

        episode_scalar_rewards.append(aggregated_episode_reward)  # Append the aggregated reward for this episode
        # logging.info("rewards", episode_scalar_rewards)
        if episode % update_freq == 0:
            epsilon *= decay

    logging.info(f"Completed Training")
    return q_net1, q_net2, episode_scalar_rewards, all_actions_taken


if __name__ == "__main__":
    num_episodes = 100  # number of episodes to train for
    learning_rate = 10e-3  # learning rate for optimizer
    attack_budget = 50  # max number of perturbations (len(channel) pixel changes each attack)
    num_times_to_sample = 2  # number of times to sample each image consecutively before sampling new image
    reward_lambda = 1
    batch_size = 512  # sample 64 experiences from the replay buffer every time
    gamma = 0.95  # discount factor
    epsilon = 0.99  # start with 50% exploration
    update_freq = 1  # update epsilon every 100 episodes
    decay = 0.99  # decay rate for epsilon

    # cifar10 dataloader
    dataloader = get_cifar_dataloader()
    # classififer
    model = load_model()
    # env
    env = ImagePerturbEnv(
        dataloader=dataloader,
        model=model,
        attack_budget=attack_budget,
        lambda_=reward_lambda,
        num_times_to_sample=num_times_to_sample,
    )

    # Initialize Q-Network and Optimizer here, and pass them to train
    # input_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # QNetwork, QNetworkPretrainedResnet
    q_net1 = QNetwork(n_actions).to(DEVICE)
    q_net2 = QNetwork(n_actions).to(DEVICE)
    optimizer1 = optim.Adam(q_net1.parameters(), lr=learning_rate)
    optimizer2 = optim.Adam(q_net2.parameters(), lr=learning_rate)

    logging.info(
        f"Training Initialization:\n"
        f"  - Hyperparameters:\n"
        f"    • num_episodes: {num_episodes}\n"
        f"    • learning_rate: {learning_rate}\n"
        f"    • attack_budget: {attack_budget}\n"
        f"    • reward_lambda: {reward_lambda}\n"
        f"    • batch_size: {batch_size}\n"
        f"    • gamma: {gamma}\n"
        f"    • initial_epsilon: {epsilon}\n"
        f"    • update_freq: {update_freq}\n"
        f"    • decay: {decay}\n"
        f"  - Environment Details:\n"
        f"    • Action Space: {env.action_space}\n"
        f"    • Observation Space: {env.observation_space}\n"
        f"  - Starting Training with num_episodes: {num_episodes}"
    )

    trained_qnet1, trained_qnet2, episode_rewards, all_actions_taken = train(
        env,
        q_net1,
        q_net2,
        optimizer1,
        optimizer2,
        num_episodes,
        epsilon,
        gamma,
        update_freq,
        batch_size,
    )
    print(episode_rewards)

    # Plot the episode rewards
    plt.figure()
    plt.plot(range(num_episodes), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")
    plt.show()

    # Save the trained Q-Network
    SAVE_PATH = "src/model_weights/trained_qnets.pth"

    torch.save({"q_net1": trained_qnet1.state_dict(), "q_net2": trained_qnet2.state_dict()}, SAVE_PATH)

    logging.info(f"Trained Q-Networks saved successfully to {SAVE_PATH}")

    import csv

    # Save episode_rewards
    with open("episode_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward"])
        for i, reward in enumerate(episode_rewards):
            writer.writerow([i, reward])

    # Save all_actions_taken
    with open("all_actions_taken.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Actions"])
        for i, actions in enumerate(all_actions_taken):
            writer.writerow([i, ",".join(map(str, actions))])
