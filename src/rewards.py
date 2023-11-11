# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def reward_distance(self, **kwargs):
    """
    Calculate the distance between the outputs of a model for the original and perturbed images for each image in a batch.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_outputs (Tensor): The model's outputs for the original images in a batch.
            perturbed_outputs (Tensor): The model's outputs for the perturbed images in a batch.
            norm_type (int): The order of the norm. Default is 2 (Euclidean norm).

    Returns:
        Tensor: A tensor of the calculated distances for each image pair in the batch.
    """
    original_outputs = kwargs.get("original_output")
    perturbed_outputs = kwargs.get("perturbed_output")
    norm_type = kwargs.get("norm_type", 2)  # Default to L2 norm if not provided

    if original_outputs is None or perturbed_outputs is None:
        raise ValueError("original_outputs and perturbed_outputs are required.")

    distances = torch.norm(original_outputs - perturbed_outputs, p=norm_type, dim=1)

    return distances


def reward_improvement(self, **kwargs):
    """
    Calculate the improvement in terms of the probability of the correct class before and after perturbation for each image in a batch.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_probs (Tensor): Probabilities of the correct class for the original images in a batch.
            perturbed_probs (Tensor): Probabilities of the correct class for the perturbed images in a batch.

    Returns:
        Tensor: A tensor of the improvement scores for each image in the batch.
    """
    original_probs = kwargs.get("original_probs")
    perturbed_probs = kwargs.get("perturbed_probs")

    if original_probs is None or perturbed_probs is None:
        raise ValueError("original_probs and perturbed_probs are required.")

    improvement_scores = original_probs - perturbed_probs

    return improvement_scores


def reward_time_decay(self, **kwargs):
    """
    Calculate the time decayed reward for each image in a batch, which decreases over time to encourage faster completion.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_probs (Tensor): Probabilities of the correct class for the original images in a batch.
            perturbed_probs (Tensor): Probabilities of the correct class for the perturbed images in a batch.
            current_step (int): The current step in the episode.
            decay_rate (float): The rate at which the reward decays over time.

    Returns:
        Tensor: A tensor of the time decayed rewards for each image in the batch.
    """
    original_probs = kwargs.get("original_probs")
    perturbed_probs = kwargs.get("perturbed_probs")
    current_step = kwargs.get("current_step")
    decay_rate = kwargs.get("decay_rate", 0.01)

    if original_probs is None or perturbed_probs is None or current_step is None:
        raise ValueError(
            "original_probs, perturbed_probs, and current_step are required."
        )

    time_penalty = decay_rate * current_step
    time_decay_rewards = (original_probs - perturbed_probs) - time_penalty

    return time_decay_rewards


def reward_goal_achievement(self, **kwargs):
    """
    Assess if the perturbation caused the model to misclassify each image in a batch by checking if confidence is below a threshold.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            perturbed_outputs (Tensor): The model's outputs for the perturbed images in a batch.
            target_classes (Tensor): The correct class indices for the images.
            threshold (float): The confidence threshold for considering it as a misclassification.

    Returns:
        Tensor: A tensor of rewards, 1.0 for each misclassified image and 0.0 otherwise.
    """
    perturbed_outputs = kwargs.get("perturbed_output")
    target_classes = kwargs.get("target_classes")
    threshold = kwargs.get("threshold", 0.5)  # Default threshold if not provided

    if perturbed_outputs is None or target_classes is None:
        raise ValueError("perturbed_outputs and target_classes are required.")

    probs = F.softmax(perturbed_outputs, dim=1)
    target_probs = probs[torch.arange(len(target_classes)), target_classes]
    rewards = (target_probs < threshold).float()

    return rewards


def reward_composite(self, **kwargs):
    """
    Calculate a composite reward that combines improvement, time decay, and goal achievement aspects for each image in a batch.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_probs (Tensor): Probabilities of the correct class for the original images.
            perturbed_probs (Tensor): Probabilities of the correct class for the perturbed images.
            current_step (int): The current step in the episode.
            decay_rate (float): The decay rate for the time penalty.
            threshold (float): The confidence threshold for considering it as a misclassification.
            perturbed_outputs (Tensor): The model's outputs for the perturbed images.
            target_classes (Tensor): The correct class indices for the images.

    Returns:
        Tensor: A tensor of the composite rewards for each image in the batch.
    """
    original_probs = kwargs.get("original_probs")
    perturbed_probs = kwargs.get("perturbed_probs")
    current_step = kwargs.get("current_step")
    decay_rate = kwargs.get("decay_rate", 0.01)
    threshold = kwargs.get("threshold", 0.5)
    perturbed_outputs = kwargs.get("perturbed_output")
    target_classes = kwargs.get("target_classes")
    if perturbed_outputs is None:
        raise ValueError("perturbed_outputs is required but was None.")

    improvement_rewards = original_probs - perturbed_probs
    time_penalty = decay_rate * current_step
    target_probs = F.softmax(perturbed_outputs, dim=1)[
        torch.arange(len(target_classes)), target_classes
    ]
    goal_rewards = (target_probs < threshold).float()

    composite_rewards = improvement_rewards + goal_rewards - time_penalty
    return composite_rewards


def reward_output_difference(self, **kwargs):
    """
    Calculate the norm of the difference between the original and perturbed model outputs for each image in a batch.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_output (Tensor): The model's outputs for the original images.
            perturbed_output (Tensor): The model's outputs for the perturbed images.
            norm_type (int): The type of norm to use for calculating the difference. Defaults to 2 (L2 norm).

    Returns:
        Tensor: A tensor of the normalized differences for each image pair in the batch.
    """
    original_output = kwargs.get("original_output")
    perturbed_output = kwargs.get("perturbed_output")
    norm_type = kwargs.get("norm_type", 2)  # Default to L2 norm if not provided

    if original_output is None or perturbed_output is None:
        raise ValueError(
            "reward_output_difference requires 'original_output' and 'perturbed_output'."
        )

    diff = original_output - perturbed_output
    norms = torch.norm(diff, p=norm_type, dim=1)
    normalized_differences = norms / (
        original_output.size(1) ** 0.5
    )  # Normalizing by the sqrt of number of elements in each output vector

    return normalized_differences


def reward_target_prob_inversion(self, **kwargs):
    """
    Calculate the inversion of the probability for the target class after perturbation for each image in a batch.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            perturbed_probs (Tensor): Probabilities of the correct class for the perturbed images in a batch.

    Returns:
        Tensor: A tensor of the inversion of the target class probabilities for each image in the batch.
    """

    perturbed_probs = kwargs.get("perturbed_probs")
    if perturbed_probs is not None and perturbed_probs.nelement() != 0:
        return 1.0 - perturbed_probs
    else:
        # Handle the case where perturbed_probs is None or empty
        return torch.tensor([])


reward_functions = {
    # Measures the change in the feature space caused by the perturbation.
    "reward_one": reward_distance,
    # Calculates how much the perturbation reduces the classifier's confidence.
    "reward_two": reward_improvement,
    # Similar to reward_improvement, but with a penalty for taking more steps.
    "reward_three": reward_time_decay,
    # Checks if the perturbation leads to a successful misclassification.
    "reward_four": reward_goal_achievement,
    # A composite reward combining several aspects of the perturbation task.
    "reward_five": reward_composite,
    # Quantifies the alteration of the model's output due to the perturbation.
    "reward_six": reward_output_difference,
    # Rewards the agent for decreasing the model's confidence in the correct class.
    "reward_seven": reward_target_prob_inversion,
}
