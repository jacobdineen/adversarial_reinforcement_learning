# -*- coding: utf-8 -*-
# pylint: disable-all
import itertools

import numpy as np
import sklearn
import torch
from sklearn.utils import shuffle
from torch import nn
from torchvision.datasets import MNIST
from tqdm import tqdm

import utils
from src.classifiers import DQNDNN

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cnn_model(input_shape) -> nn.Sequential:
    channel_count = input_shape[0]
    model = nn.Sequential(
        nn.Conv2d(channel_count, 64, kernel_size=(5, 5), padding="same").to(device),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(64, 64, kernel_size=(5, 5), padding="same").to(device),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Flatten(),
        nn.Linear(64 * input_shape[1] * input_shape[2], 128).to(device),
        nn.ReLU(),
    ).to(device)
    return model


def get_dnn_model(number_of_class_labels) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(number_of_class_labels, 128).to(device), nn.ReLU()
    ).to(device)
    return model


class QModel(nn.Module):
    def __init__(self, cnn_model, dnn_model, blocks):
        super().__init__()
        self.cnn_model = cnn_model
        self.dnn_model = dnn_model
        self.blocks = blocks

    def forward(self, image_input, probability_input):
        image_representation = self.cnn_model(image_input)
        probability_representation = self.dnn_model(probability_input)
        x = torch.cat((probability_representation, image_representation), dim=-1)
        x = nn.Linear(256, len(self.blocks)).to(device)(x)
        return x


def update_q_model(experience_replay, q_model, batch_size, discount_factor):
    q_model.train()
    q_model_optimizer = torch.optim.Adam(q_model.parameters(), lr=0.0001)

    train_input_1 = []
    train_input_2 = []
    train_label = []
    for experience in shuffle(experience_replay, random_state=0)[:batch_size]:
        (
            sample_image,
            sample_image_probability,
            action,
            reward,
            sample_image_noise,
            sample_image_noise_probability,
        ) = experience

        # Move tensors to the device
        initial_image_state = sample_image.to(device)
        initial_image_probability_state = sample_image_probability.to(device)
        next_image_state = sample_image_noise.to(device)
        next_image_probability_state = sample_image_noise_probability.to(device)

        action_taken: int = action
        reward_obtained: float = reward

        target = q_model(
            initial_image_state.unsqueeze(0), initial_image_probability_state
        )[0]
        # input(target.shape)

        Q_sa = torch.max(
            q_model(next_image_state.unsqueeze(0), next_image_probability_state)
        )
        # input(Q_sa.shape)

        if reward_obtained == 10 or reward_obtained == -1:
            target[action_taken] = reward_obtained
        else:
            target[action_taken] = reward_obtained + discount_factor * Q_sa

        train_input_1.append(initial_image_state)
        train_input_2.append(initial_image_probability_state)

        train_label.append(target)

    train_input_1 = torch.squeeze(torch.stack(train_input_1))
    train_input_2 = torch.squeeze(torch.stack(train_input_2))
    # print(train_input_1.shape)
    # print(train_input_2.shape)
    # input()

    train_label = torch.squeeze(torch.stack(train_label))
    # print(train_label.shape)
    # input()

    if len(train_input_1.shape) == 3:
        train_input_1 = train_input_1.unsqueeze(-1)
        # print(train_input_1.shape)
        # input()

    train_input_1, train_input_2, train_label = sklearn.utils.shuffle(
        train_input_1, train_input_2, train_label, random_state=0
    )

    # print(f"Q Model Update")
    q_model.train()

    # Train on batch
    t1_batch = train_input_1[0:batch_size].squeeze(-1).unsqueeze(1)
    t2_batch = train_input_2[0:batch_size]
    label_batch = train_label[0:batch_size]

    # print(t1_batch.shape)
    # print(t2_batch.shape)
    # print(label_batch.shape)
    # input()

    q_model_optimizer = torch.optim.Adam(q_model.parameters(), lr=0.0001)
    q_model_optimizer.zero_grad()

    prediction = q_model(t1_batch, t2_batch)
    # print(prediction.shape)
    # print(label_batch.shape)
    # input()

    loss = nn.MSELoss()(prediction, label_batch)
    loss.backward()
    q_model_optimizer.step()

    return q_model


def main():
    mnist_model = DQNDNN()
    # Load the state dictionary from the file
    checkpoint = torch.load("src/model_weights/simple_mnist2.pth")

    # Check if the loaded state dict is a dict and has the key 'state_dict'
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Apply the state dictionary to the model
    mnist_model.load_state_dict(state_dict)
    mnist_model.to(device)
    mnist_model.eval()

    mnist = MNIST("data", download=True, train=True)
    dataset = mnist.data
    labels = mnist.targets

    # Deterministically splitting for now
    (X_train, y_train) = (dataset[:50000], labels[:50000])
    (X_test, y_test) = (dataset[50000:], labels[50000:])

    input_shape = (1, 28, 28)
    class_label_count = 10
    LAMBDA = 1.0

    block_size = 2
    x_span = list(range(0, input_shape[1], block_size))
    blocks = list(itertools.product(x_span, x_span))

    cnn_model = get_cnn_model(input_shape=input_shape)
    dnn_model = get_dnn_model(number_of_class_labels=class_label_count)

    # input(torchsummary.summary(cnn_model, input_shape))
    # input(torchsummary.summary(dnn_model, (class_label_count,)))

    image_input = torch.zeros((1, *input_shape)).to(device)
    probability_input = torch.zeros((1, class_label_count)).to(device)

    # input(probability_input.shape)
    # input(image_input.shape)

    image_representation = cnn_model(image_input)
    probability_representation = dnn_model(probability_input)

    # input(image_representation.shape)
    # input(probability_representation.shape)

    x = torch.cat((probability_representation, image_representation), dim=-1).to(device)

    # input(x.shape)

    x = nn.Linear(256, len(blocks)).to(device)(x)

    # input(x.shape)

    q_model = QModel(cnn_model, dnn_model, blocks).to(device)
    print(q_model)

    epsilon = 0.9
    batch_size = 64
    discount_factor = 0.9
    max_buffer_size = 1000
    max_blocks_attack = 20

    success = []
    success_rate = []
    experience_replay = []

    GAME_COUNT = 5000 * 10

    for game_number in tqdm(range(GAME_COUNT)):
        sample_image = X_train[game_number].to(device)
        sample_image = sample_image.unsqueeze(0)

        sample_image = sample_image.float()

        if game_number > 0 and game_number % 300 == 0:
            epsilon -= 0.1

        if epsilon <= 0.1:
            epsilon = 0.1

        predicted_label_distribution = mnist_model(sample_image.unsqueeze(0))
        original_predicted_label = torch.argmax(predicted_label_distribution, dim=1)

        for iteration in range(0, max_blocks_attack):
            sample_image_probability = mnist_model(sample_image.unsqueeze(0))[0]
            sample_image_probability = sample_image_probability.unsqueeze(0)

            if np.random.rand() < epsilon:
                action = np.random.randint(0, len(blocks))
            else:
                action = torch.argmax(
                    q_model(sample_image.unsqueeze(0), sample_image_probability)
                ).item()

            attack_region = torch.zeros(input_shape)
            attack_coord = blocks[action]
            attack_region[
                0,
                attack_coord[0] : attack_coord[0] + block_size,
                attack_coord[1] : attack_coord[1] + block_size,
            ] = 1

            sample_image_noise = sample_image + (attack_region * LAMBDA).to(device)
            sample_image_noise_probability = mnist_model(
                sample_image_noise.unsqueeze(0)
            )

            modified_predicted_label = torch.argmax(
                mnist_model(sample_image_noise.unsqueeze(0)), dim=1
            )

            if modified_predicted_label != original_predicted_label:
                print("here")
                reward = 10.0
                success.append(1)
                experience = (
                    sample_image,
                    sample_image_probability,
                    action,
                    reward,
                    sample_image_noise,
                    sample_image_noise_probability,
                )
                experience_replay.append(experience)
                break

            else:
                reward = -0.1
                experience = (
                    sample_image,
                    sample_image_probability,
                    action,
                    reward,
                    sample_image_noise,
                    sample_image_noise_probability,
                )
                experience_replay.append(experience)

            sample_image = sample_image_noise
            # input(sample_image.shape)

        if iteration == max_blocks_attack - 1:
            # print("Failure")

            reward = -1.0
            success.append(0)
            experience = (
                sample_image,
                sample_image_probability,
                action,
                reward,
                sample_image_noise,
                sample_image_noise_probability,
            )
            experience_replay.append(experience)

        if len(experience_replay) > max_buffer_size:
            tqdm.write(f"Updating Q Model")
            q_model = update_q_model(
                experience_replay, q_model, batch_size, discount_factor
            )
            experience_replay = []

            tqdm.write(f"Successes: {success}")
            tqdm.write(f"Updated Q Model: Success rate {np.mean(np.array(success))}")
            success_rate.append(np.mean(np.array(success)))
            success = []

        if game_number % 100 == 0:
            tqdm.write(
                f"Episode: {game_number}, Success rate: {np.mean(np.array(success))}"
            )

    print(f"Final success rate: {np.mean(np.array(success))}")
    print(f"Saving model to q_model.pt")
    torch.save(q_model.state_dict(), "q_model.pt")


if __name__ == "__main__":
    main()
