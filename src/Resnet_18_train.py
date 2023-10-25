# -*- coding: utf-8 -*-
# IMPORT LIBRARIES
import gc
import logging

import torch
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from src.classifiers import ResidualBlock, ResNet

logging.basicConfig(level=logging.INFO)

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))


# Data Preparation
def prepare_data(batch_size, image_size):
    """prepare_data

    Args:
        batch_size (_type_): _description_
        image_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    transform_train = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Rotate the image by up to 10 degrees
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Translate by up to 10% of the image size
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


# Train Model
def train_model(model, trainloader, num_epochs, criterion, optimizer, device):
    for epoch in tqdm(range(num_epochs)):
        for _, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Evaluate Model
def evaluate_model(model, testloader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        print(f"Accuracy: {100 * correct / total} %")


# Main Function
def main():
    # Hyperparameters and Initialization
    num_epochs = 5
    batch_size = 256
    image_size = 224
    learning_rate = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Preparation
    trainloader, testloader = prepare_data(batch_size, image_size)

    # Model Initialization
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001, momentum=0.9)

    # Model Training
    train_model(model, trainloader, num_epochs, criterion, optimizer, device)

    # Model Evaluation
    evaluate_model(model, testloader, device)


if __name__ == "__main__":
    main()
