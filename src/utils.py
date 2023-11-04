# -*- coding: utf-8 -*-
"""
Various utility funcs housed here
"""
import logging
import random

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


# Resnet on Cifar10
def load_model():
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(DEVICE)
    model.load_state_dict(torch.load("src/model_weights/resnet_cifar-10.pth", map_location=DEVICE))
    logging.info(f"Resnet model loaded successfully on device: {DEVICE}")
    return model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar_dataloaders(batch_size: int = 32, val_split: float = 0.1, seed: int = 42) -> tuple:
    """Get Train, Validation and Test Data Loaders for CIFAR10

    Args:
        batch_size (int, optional): _description_. Defaults to 32.
        val_split (float, optional): _description_. Defaults to 0.1.
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: train_loader, valid_loader, test_loader
    """
    set_seed(seed)

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

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Create data loaders with the worker_init_fn set
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    valid_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
