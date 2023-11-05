# -*- coding: utf-8 -*-
"""
Various utility funcs housed here
"""
import logging
import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.classifiers import ResidualBlock, ResNet

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model() -> ResNet:
    """
    Load a pre-trained ResNet model for CIFAR-10 dataset.

    Returns:
        ResNet: The loaded ResNet model.
    """
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(DEVICE)
    model.load_state_dict(torch.load("src/model_weights/resnet_cifar-10.pth", map_location=DEVICE))
    logging.info(f"Resnet model loaded successfully on device: {DEVICE}")
    return model


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators in numpy, random, and torch for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# this is a class that will return an iterator that will loop over the dataloader
# this is necessary because we have a conditiion in the env where
# the dataloader should be refreshed
class EndlessDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data


def get_cifar_dataloaders(
    batch_size: int = 32, val_split: float = 0.1, seed: int = 42, train_limit: int = None  # Add train_limit parameter
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get data loaders for the CIFAR-10 dataset, split into training, validation, and test sets.

    Args:
        batch_size (int): Number of samples per batch to load.
        val_split (float): The fraction of the training data to be used as validation data.
        seed (int): The seed for random operations to ensure reproducibility.
        train_limit (int): The limit for the number of training data samples to use.

    Returns:
        tuple: A tuple containing the training, validation, and test DataLoader objects.
    """
    set_seed(seed)  # Your set_seed function needs to be defined elsewhere

    transform_chain = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    full_train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform_chain)

    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))

    np.random.shuffle(indices)

    # If train_limit is set, reduce the size of train_idx
    if train_limit is not None:
        train_limit = min(train_limit, len(indices) - split)  # Ensure limit is not more than available indices
        train_idx, valid_idx = indices[split : split + train_limit], indices[:split]
    else:
        train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    valid_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info(
        f"""
    size of train_loader: {len(train_loader)}
    size of valid_loader: {len(valid_loader)}
    size of test_loader: {len(test_loader)}
    """
    )
    return train_loader, valid_loader, test_loader


# Image Preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path)
    transform_img = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    img.show()
    input_image_tensor = transform_img(img).unsqueeze(0)
    return input_image_tensor, img
