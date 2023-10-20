# -*- coding: utf-8 -*-
import unittest

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50

from src.env import ImagePerturbEnv


class TestImagePerturbEnv(unittest.TestCase):
    def setUp(self):
        transform_chain = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform_chain)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        model = resnet50(pretrained=True)
        self.env = ImagePerturbEnv(dataloader=dataloader, model=model)

    def test_init(self):
        self.assertIsNotNone(self.env.dataloader)
        self.assertIsNotNone(self.env.model)
        self.assertEqual(self.env.current_attack_count, 0)

    def test_step(self):
        original_image = self.env.image.clone()
        action = self.env.action_space.sample()
        next_state, reward, done, _ = self.env.step(action)

        # Check if the image is perturbed
        self.assertFalse(torch.equal(original_image, next_state))

        # Check if reward is a float
        self.assertIsInstance(reward, float)

        # Check if done is a boolean
        self.assertIsInstance(done, bool)

    def test_reset(self):
        original_image, original_class = self.env.image, self.env.target_class
        self.env.reset()
        new_image, new_class = self.env.image, self.env.target_class

        # Check if the image and class are different after reset
        self.assertFalse(torch.equal(original_image, new_image))
        self.assertNotEqual(original_class, new_class)

    def test_action_space(self):
        # Test if action space sample is valid
        action = self.env.action_space.sample()
        self.assertTrue(action >= 0)
        self.assertTrue(action < self.env.action_space.n)


if __name__ == "__main__":
    unittest.main()
