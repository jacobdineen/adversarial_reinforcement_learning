# -*- coding: utf-8 -*-
"""
Various utility funcs housed here
"""
import logging

import torch
from torch.utils.data import DataLoader
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


def get_cifar_dataloader():
    transform_chain = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # datasets
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader
