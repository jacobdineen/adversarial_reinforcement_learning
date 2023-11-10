# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def reward_distance(self, **kwargs):
    """Calculate the distance between the outputs of a model for the original and perturbed images.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_output (Tensor): The model's output for the original image.
            perturbed_output (Tensor): The model's output for the perturbed image.
            norm_type (int): The order of the norm. Default is 2 (Euclidean norm).

    Returns:
        float: The calculated distance.
    """
    original_output = kwargs.get("original_output")
    perturbed_output = kwargs.get("perturbed_output")
    norm_type = kwargs.get("norm_type", 2)  # Default to L2 norm if not provided

    distance = torch.norm(original_output - perturbed_output, p=norm_type)
    return distance.item()


def reward_improvement(self, **kwargs):
    """Calculate the improvement in terms of the probability of the correct class before and after perturbation.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_prob (float): Probability of the correct class for the original image.
            perturbed_prob (float): Probability of the correct class for the perturbed image.

    Returns:
        float: The improvement score which is the decrease in probability.
    """
    original_prob = kwargs.get("original_prob")
    perturbed_prob = kwargs.get("perturbed_prob")

    return original_prob - perturbed_prob


def reward_time_decay(self, **kwargs):
    """Calculate the time decayed reward which decreases over time to encourage faster completion.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_prob (float): Probability of the correct class for the original image.
            perturbed_prob (float): Probability of the correct class for the perturbed image.
            current_step (int): The current step in the episode.
            decay_rate (float): The rate at which the reward decays over time.

    Returns:
        float: The time decayed reward.
    """
    # Extract the required arguments using kwargs.get and set defaults if not provided
    original_prob = kwargs.get("original_prob", None)
    perturbed_prob = kwargs.get("perturbed_prob", None)
    current_step = kwargs.get("current_step", None)

    # Decrease reward as time goes by to encourage faster completion
    decay_rate = kwargs.get("decay_rate", 0.01)  # Default decay rate if not provided
    time_penalty = decay_rate * current_step

    return (original_prob - perturbed_prob) - time_penalty


def reward_goal_achievement(self, **kwargs):
    """Assess if the perturbation caused the model to misclassify the image by checking if confidence is below a threshold.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            perturbed_output (Tensor): The model's output for the perturbed image.
            target_class (int): The correct class index for the image.
            threshold (float): The confidence threshold for considering it as a misclassification.

    Returns:
        float: Reward of 1.0 if goal is achieved (misclassification), else 0.0.
    """
    perturbed_output = kwargs.get("perturbed_output")
    target_class = kwargs.get("target_class")
    threshold = kwargs.get("threshold", 0.5)  # Default threshold if not provided

    probs = F.softmax(perturbed_output, dim=1).squeeze()
    target_prob = probs[target_class].item()
    return 1.0 if target_prob < threshold else 0.0


def reward_composite(self, **kwargs):
    """Calculate a composite reward that combines improvement, time decay, and goal achievement aspects.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_prob (float): Probability of the correct class for the original image.
            perturbed_prob (float): Probability of the correct class for the perturbed image.
            current_step (int): The current step in the episode.
            decay_rate (float): The decay rate for the time penalty.
            threshold (float): The confidence threshold for considering it as a misclassification.
            perturbed_output (Tensor): The model's output for the perturbed image.
            target_class (int): The correct class index for the image.

    Returns:
        float: The composite reward.
    """
    # Extract the necessary arguments for the component reward functions
    original_prob = kwargs.get("original_prob")
    perturbed_prob = kwargs.get("perturbed_prob")
    current_step = kwargs.get("current_step")
    decay_rate = kwargs.get("decay_rate", 0.01)  # Default decay rate if not provided
    threshold = kwargs.get("threshold", 0.5)  # Default threshold if not provided
    perturbed_output = kwargs.get("perturbed_output")
    target_class = kwargs.get("target_class")

    # Call the component reward functions with kwargs
    improvement_reward = original_prob - perturbed_prob
    time_penalty = decay_rate * current_step
    probs = F.softmax(perturbed_output, dim=1).squeeze()
    target_prob = probs[target_class].item()
    goal_reward = 1.0 if target_prob < threshold else 0.0
    # Combine the rewards to get the composite reward
    composite_reward = improvement_reward + goal_reward - time_penalty
    return composite_reward


def reward_output_difference(self, **kwargs):
    """Calculate the norm of the difference between the original and perturbed model outputs.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            original_output (Tensor): The model's output for the original image.
            perturbed_output (Tensor): The model's output for the perturbed image.
            norm_type (int): The type of norm to use for calculating the difference. Defaults to 2 (L2 norm).

    Returns:
        float: The normalized difference.
    """
    original_output = kwargs.get("original_output")
    perturbed_output = kwargs.get("perturbed_output")
    norm_type = kwargs.get("norm_type", 2)  # Default to L2 norm if not provided

    if original_output is None or perturbed_output is None:
        raise ValueError(
            "reward_output_difference requires 'original_output' and 'perturbed_output'."
        )

    diff = original_output - perturbed_output
    return torch.norm(diff, p=norm_type) / original_output.numel()


def reward_target_prob_inversion(self, **kwargs):
    """Calculate the inversion of the probability for the target class after perturbation.

    Args:
        **kwargs: Variable length keyword arguments. Expected to contain:
            perturbed_prob (float): Probability of the correct class for the perturbed image.

    Returns:
        float: The inversion of the target class probability.
    """

    perturbed_prob = kwargs.get("perturbed_prob")
    return 1.0 - perturbed_prob
