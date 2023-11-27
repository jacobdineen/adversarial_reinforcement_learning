# -*- coding: utf-8 -*-
import argparse
import gc
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from src.env import BlockBasedPerturbEnv, SinglePixelPerturbEnv
from src.rewards import reward_functions
from src.utils import EndlessDataLoader, get_dataloaders, load_model, set_seed

# from stable_baselines3.common.evaluation import evaluate_policy
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def get_image_classifier_accuracy(model, test_loader):
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # Move data to the same device as the model
            outputs = model(images)  # Get raw output from the model
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def get_perturbed_classifier_accuracy_and_plot(model, ppomodel, env, max_images_to_plot=10):
    total_correct = 0
    total_samples = 0
    incorrect_images = []

    with torch.no_grad():  # Disable gradient computation
        for images, labels in env.dataloader.dataloader:
            images, labels = images.to("cpu"), labels.to("cpu")
            actions = ppomodel.predict(images)
            print(actions)
            perturbed_images, _, _, _, _ = env.step(actions[0], testing=False)
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs, 1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Identify and store incorrectly classified images
            misclassified_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in misclassified_indices:
                if len(incorrect_images) < max_images_to_plot:
                    incorrect_images.append(
                        (
                            images[idx],
                            perturbed_images[idx],
                            labels[idx].item(),
                            predicted[idx].item(),
                            actions[0][idx].item(),
                        )
                    )
            break

    logging.info(f"Total Correct: {total_correct}")
    logging.info(f"Total Samples: {total_samples}")
    logging.info(f"Total misclassified images: {len(incorrect_images)}")
    accuracy = total_correct / total_samples

    # Plotting the images
    num_images = len(incorrect_images)
    _, axs = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    # Modify the plotting code to display the action taken
    for i, (orig, perturbed, orig_label, perturbed_label, action) in enumerate(incorrect_images):
        axs[i, 0].imshow(orig.permute(1, 2, 0).numpy(), cmap="gray")
        axs[i, 0].set_title(f"Original (Label: {orig_label})")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(perturbed.permute(1, 2, 0).numpy(), cmap="gray")
        axs[i, 1].set_title(f"Perturbed (Label: {perturbed_label}, Action: {action})")
        axs[i, 1].axis("off")

    plt.show()

    return accuracy


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

    SEED = 1
    # seed=base_seed+run
    set_seed(SEED)
    batch_size = 20 if dataset_name == "cifar10" else 50

    _, _, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        val_split=val_split,
        seed=SEED,
        train_limit=10000,
    )

    steps_per_episode = len(test_loader)  # number of images to perturb per episode

    # classififer
    model = load_model(dataset_name=dataset_name)
    model.eval()
    accuracy = get_image_classifier_accuracy(model, test_loader)
    print(f"Classifier Accuracy over the test set: {accuracy * 100:.2f}%")

    test_env = EnvClass(
        dataloader=EndlessDataLoader(test_loader),
        model=model,
        reward_func=selected_reward_func,
        steps_per_episode=steps_per_episode,
        verbose=False,
    )

    ppo_model = PPO.load(model_save_path)
    done = False
    label_flips = []
    num_samples = 0

    while not done:
        # Get predictions for original images
        original_images = test_env.images.clone()
        original_outputs = model(original_images)
        _, original_predictions = torch.max(original_outputs, dim=1)

        # Predict actions for the current images using the PPO model
        actions, _ = ppo_model.predict(original_images)

        # Take a step in the environment using the predicted actions
        # pylint disable=W0632
        perturbed_images, _, dones, _ = test_env.step(actions=actions, testing=False)
        # Get predictions for perturbed images
        perturbed_outputs = model(perturbed_images)
        _, perturbed_predictions = torch.max(perturbed_outputs, dim=1)

        # Compare predictions to find label flips
        flipped_indices = (original_predictions != perturbed_predictions).nonzero(as_tuple=True)[0]
        for idx in flipped_indices:
            label_flips.append(
                {
                    "original_image": original_images[idx].cpu(),
                    "original_label": original_predictions[idx].cpu().item(),
                    "perturbed_image": perturbed_images[idx].cpu(),
                    "perturbed_label": perturbed_predictions[idx].cpu().item(),
                }
            )

        num_samples += batch_size
        print(f"Number of label flips: {len(label_flips)}")
        print(f"Number of samples: {num_samples}")
        print(f"Percentage of label flips: {len(label_flips) / num_samples * 100:.2f}%")
        # if num_samples >= 1000:
        if any(dones):
            num_images_to_plot = min(5, len(label_flips))
            _, axs = plt.subplots(num_images_to_plot, 2, figsize=(10, 2 * num_images_to_plot))

            for i in range(num_images_to_plot):
                info = label_flips[i]

                # Convert the tensors to NumPy arrays
                original_img = info["original_image"].numpy()
                perturbed_img = info["perturbed_image"].numpy()

                # Normalize the images if they're not already in the range [0, 1]
                original_img = np.clip(original_img, 0, 1)
                perturbed_img = np.clip(perturbed_img, 0, 1)

                # Plot the original image
                axs[i, 0].imshow(original_img.transpose(1, 2, 0))  # Transpose the axes from (C, H, W) to (H, W, C)
                axs[i, 0].title.set_text(f'Original (Label: {info["original_label"]})')
                axs[i, 0].axis("off")

                # Plot the perturbed image
                axs[i, 1].imshow(perturbed_img.transpose(1, 2, 0))  # Transpose the axes from (C, H, W) to (H, W, C)
                axs[i, 1].title.set_text(f'Perturbed (Label: {info["perturbed_label"]})')
                axs[i, 1].axis("off")

            plt.tight_layout()
            plt.show()
            break
    # print(rendered)
    # print("Model loaded successfully")
    # # perturbed_images_ = []
    # # real_images = []
    # # for i in range(steps_per_episode / args.batch_size):
    # #     actions = ppo_model.predict(test_env.images)
    # #     images, perturbed_images, rewards, _ = test_env.step(actions[0], testing=True)
    # #     perturbed_images_.append(perturbed_images)
    # #     real_images.append(images)
    # accuracy = get_perturbed_classifier_accuracy_and_plot(model, ppo_model, test_env)
    # print(f"Classifier Accuracy over the test set: {accuracy * 100:.2f}%")

    # if dataset_name == "mnist":
    #     plot_best_examples_mnist(model, images, perturbed_images, rewards)
    # else:
    #     plot_best_examples_cifar(model, images, perturbed_images, rewards)


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
        default="/home/jdineen/Documents/adv_rl/src/model_weights/ppo_mnist_mnist_episodes-20_trainlim-1000.zip",
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
