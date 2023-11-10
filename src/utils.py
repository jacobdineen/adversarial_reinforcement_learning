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
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models import resnet18
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(dataset_name) -> resnet18:
    """
    Load a pre-trained ResNet model for CIFAR-10 or MNIST dataset.

    Returns:
        ResNet: The loaded ResNet model.
    """

    model = resnet18(num_classes=10)

    if dataset_name == "cifar":
        model = resnet18(num_classes=10)
        model.load_state_dict(torch.load("src/model_weights/cifar.pth", map_location=DEVICE))
    elif dataset_name == "mnist":
        model = resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.load_state_dict(torch.load("src/model_weights/mnist.pth", map_location=DEVICE))
    logging.info(f"Resnet {dataset_name} model loaded successfully on device: {DEVICE}")
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
        # Note the try-catch here
        # if we try to call next on a depleted iterator, it will raise a StopIteration
        # this iterator will catch that and refresh the iterator
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data


def get_dataloaders(
    dataset_name,
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 42,
    train_limit: int = None,  # Add train_limit parameter
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

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL image or numpy.ndarray to tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize the data
        ]
    )
    if dataset_name == "cifar":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        full_train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    elif dataset_name == "mnist":
        full_train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = MNIST(root="./data", train=False, download=True, transform=transform)

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
    Returning dataloaders for dataset: {dataset_name}
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

#Filter Actions


def add_gaussian_noise(image, mean=0, std=1):
    """
    Add Gaussian noise to the input image.

    Args:
        image (torch.Tensor): Input image tensor.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Image with added Gaussian noise.
    """
    if image.shape[1] == 1:  # Grayscale image
        noise = torch.randn_like(image) * std + mean
    elif image.shape[1] == 3:  # Color image
        noise = torch.randn_like(image[:, :1, :, :]) * std + mean  # Apply to the first channel only
        noise = torch.cat([noise] * 3, dim=1)  # Expand to all channels

    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # Ensure pixel values are in [0, 1] range


def apply_gaussian_blur(image, kernel_size=5):
    """
    Apply Gaussian blur to the input image.

    Args:
        image (torch.Tensor): Input image tensor.
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        torch.Tensor: Image with applied Gaussian blur.
    """
    if image.shape[1] == 1:  # Grayscale image
        blurred_image = F.conv2d(image, torch.ones(1, 1, kernel_size, kernel_size) / kernel_size**2, padding=kernel_size//2)
    elif image.shape[1] == 3:  # Color image
        blurred_image = torch.cat([F.conv2d(image[:, i:i+1, :, :], torch.ones(1, 1, kernel_size, kernel_size) / kernel_size**2, padding=kernel_size//2) for i in range(3)], dim=1)

    return torch.clamp(blurred_image, 0, 1)  # Ensure pixel values are in [0, 1] range


def apply_sharpening(image, alpha=1.5):
    """
    Apply sharpening filter to the input image.

    Args:
        image (torch.Tensor): Input image tensor.
        alpha (float): Strength of the sharpening effect.

    Returns:
        torch.Tensor: Image with applied sharpening.
    """
    if image.shape[1] == 1:  # Grayscale image
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        sharpened_image = F.conv2d(image, kernel.unsqueeze(0).unsqueeze(0), padding=1) * alpha + image * (1 - alpha)
    elif image.shape[1] == 3:  # Color image
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        sharpened_image = torch.cat([F.conv2d(image[:, i:i+1, :, :], kernel.unsqueeze(0).unsqueeze(0), padding=1) * alpha + image[:, i:i+1, :, :] * (1 - alpha) for i in range(3)], dim=1)

    return torch.clamp(sharpened_image, 0, 1)  # Ensure pixel values are in [0, 1] range




def white_out_block(image):
    """
    Apply a filter that whites out a small block in the image.

    Args:
        image (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Image with the white-out block applied.
    """
    # Get image dimensions
    _, _, height, width = image.shape

    # Define block size as 1% of image size
    block_width = max(1, int(0.01 * width))
    block_height = max(1, int(0.01 * height))

    # Randomly choose block position
    x = random.randint(0, width - block_width)
    y = random.randint(0, height - block_height)

    # Create a mask to identify the region of interest (block)
    mask = torch.zeros_like(image)
    mask[:, :, y:y+block_height, x:x+block_width] = 1

    # Set the region identified by the mask to a high intensity white value
    white_block_image = image.clone()
    white_block_image[mask.bool()] = 1.0

    return white_block_image