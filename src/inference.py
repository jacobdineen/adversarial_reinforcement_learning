# -*- coding: utf-8 -*-
import argparse
import gc
import logging

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from src.env import BlockBasedPerturbEnv, SinglePixelPerturbEnv
from src.rewards import reward_functions
from src.utils import EndlessDataLoader, get_dataloaders, load_model, set_seed

# from stable_baselines3.common.evaluation import evaluate_policy


def plot_best_examples_cifar(model, images, perturbed_images, rewards):
    assert images.dim() == 4, "Expected 'images' to be a 4D tensor"
    assert perturbed_images.dim() == 4, "Expected 'perturbed_images' to be a 4D tensor"
    assert rewards.dim() == 1, "Expected 'rewards' to be a 1D tensor"
    assert images.shape[0] == perturbed_images.shape[0] == rewards.shape[0], "Mismatch in number of examples"

    _, top_k_indices = torch.topk(rewards, 5)
    _, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))

    for idx, index in enumerate(top_k_indices):
        # Process original and perturbed images
        input_image = images[index].numpy()
        perturbed_image = perturbed_images[index].numpy()

        # Normalize if necessary
        if input_image.max() <= 1:
            input_image = (input_image * 255).astype("uint8")
        if perturbed_image.max() <= 1:
            perturbed_image = (perturbed_image * 255).astype("uint8")

        # Prepare for model (only add batch dimension)
        input_tensor = torch.tensor(input_image[None, ...]).float()
        perturbed_tensor = torch.tensor(perturbed_image[None, ...]).float()

        # Model predictions
        with torch.no_grad():
            raw_output_input = model(input_tensor)
            raw_output_perturbed = model(perturbed_tensor)

        softmax_output_input = F.softmax(raw_output_input, dim=1)
        softmax_output_perturbed = F.softmax(raw_output_perturbed, dim=1)

        predicted_class_input = torch.argmax(softmax_output_input)
        predicted_class_perturbed = torch.argmax(softmax_output_perturbed)

        predicted_confidence_input = torch.max(softmax_output_input)
        predicted_confidence_perturbed = torch.max(softmax_output_perturbed)

        # Plotting
        axes[idx, 0].imshow(input_image.transpose(1, 2, 0), cmap="gray")
        axes[idx, 0].set_title(
            f"Original\nClass: {predicted_class_input.item()}\nConf: {predicted_confidence_input.item():.4f}"
        )
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(perturbed_image.transpose(1, 2, 0), cmap="gray")
        axes[idx, 1].set_title(
            f"Perturbed\nClass: {predicted_class_perturbed.item()}\nConf: {predicted_confidence_perturbed.item():.4f}"
        )
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig("subplot_image.png")
    plt.show()


def plot_best_examples_mnist(model, images, perturbed_images, rewards):
    assert images.dim() == 4, "Expected 'images' to be a 4D tensor"
    assert perturbed_images.dim() == 4, "Expected 'perturbed_images' to be a 4D tensor"
    assert rewards.dim() == 1, "Expected 'rewards' to be a 1D tensor"
    assert images.shape[0] == perturbed_images.shape[0] == rewards.shape[0], "Mismatch in number of examples"

    print(images.shape)
    print(perturbed_images.shape)
    _, top_k_indices = torch.topk(rewards, 5)
    _, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))

    for idx, index in enumerate(top_k_indices):
        # Process original and perturbed images
        input_image = images[index].squeeze(0).numpy()
        perturbed_image = perturbed_images[index].squeeze(0).numpy()

        # Normalize if necessary
        if input_image.max() <= 1:
            input_image = (input_image * 255).astype("uint8")
        if perturbed_image.max() <= 1:
            perturbed_image = (perturbed_image * 255).astype("uint8")

        # Model predictions
        with torch.no_grad():
            raw_output_input = model(torch.tensor(input_image[None, None, ...]).float())
            raw_output_perturbed = model(torch.tensor(perturbed_image[None, None, ...]).float())

        softmax_output_input = F.softmax(raw_output_input, dim=1)
        softmax_output_perturbed = F.softmax(raw_output_perturbed, dim=1)

        predicted_class_input = torch.argmax(softmax_output_input)
        predicted_class_perturbed = torch.argmax(softmax_output_perturbed)

        predicted_confidence_input = torch.max(softmax_output_input)
        predicted_confidence_perturbed = torch.max(softmax_output_perturbed)

        # Plotting
        axes[idx, 0].imshow(input_image, cmap="gray")
        axes[idx, 0].set_title(
            f"Original\nClass: {predicted_class_input.item()}\nConf: {predicted_confidence_input.item():.4f}"
        )
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(perturbed_image, cmap="gray")
        axes[idx, 1].set_title(
            f"Perturbed\nClass: {predicted_class_perturbed.item()}\nConf: {predicted_confidence_perturbed.item():.4f}"
        )
        axes[idx, 1].axis("off")

    plt.tight_layout()
    plt.savefig("subplot_image.png")
    plt.show()


def main(env_type, val_split, dataset_name, selected_reward_func, model_save_path):
    if env_type == "single_pixel":
        EnvClass = SinglePixelPerturbEnv
    elif env_type == "block_based":
        EnvClass = BlockBasedPerturbEnv
    else:
        raise ValueError("Invalid environment type")

    logging.info(args)
    torch.cuda.empty_cache()
    gc.collect()

    SEED = 42
    # seed=base_seed+run
    set_seed(SEED)

    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=20 if dataset_name == "cifar10" else 50,
        val_split=val_split,
        seed=SEED,
        train_limit=None,
    )

    steps_per_episode = len(test_loader)  # number of images to perturb per episode

    # classififer
    model = load_model(dataset_name=dataset_name)

    test_env = EnvClass(
        dataloader=EndlessDataLoader(test_loader),
        model=model,
        reward_func=selected_reward_func,
        steps_per_episode=steps_per_episode,
        verbose=False,
    )

    ppo_model = PPO.load(model_save_path)
    print("Model loaded successfully")
    actions = ppo_model.predict(test_env.images)

    images, perturbed_images, rewards, _ = test_env.step(actions[0], testing=True)
    if dataset_name == "mnist":
        plot_best_examples_mnist(model, images, perturbed_images, rewards)
    else:
        plot_best_examples_cifar(model, images, perturbed_images, rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an agent to perturb images.")
    parser.add_argument("--dataset_name", type=str, default="cifar", help="dataset to use. mnist of cifar")
    parser.add_argument(
        "--env_type",
        type=str,
        choices=["single_pixel", "block_based"],
        default="block_based",
        help="Type of environment to use (single_pixel or block_based).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Holdout data for validation and testing.",
    )
    parser.add_argument(
        "--reward_func",
        type=str,
        choices=list(reward_functions.keys()),
        default="reward_one",
        help="The name of the reward function to use.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        # default="src/model_weights/ppo_cifar_cifar_episodes-10_trainlim-100.zip",
        help="Where to load trained PPO model",
    )
    # src/model_weights/ppo_mnist_mnist_episodes-100_trainlim-1000
    # src/model_weights/ppo_cifar_cifar_episodes-10_trainlim-100

    args = parser.parse_args()
    env_type = args.env_type
    val_split = args.val_split
    dataset_name = args.dataset_name
    selected_reward_func = reward_functions[args.reward_func]
    model_save_path = args.model_save_path

    main(env_type, val_split, dataset_name, selected_reward_func, model_save_path)
