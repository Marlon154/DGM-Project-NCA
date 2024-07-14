import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim, ms_ssim
from nca import to_rgba, to_rgb


def get_loss_function(loss_name):
    """
    Returns the specified loss function based on the input name.

    Args:
        loss_name (str): Name of the loss function to use.

    Returns:
        function: The corresponding loss function.

    Raises:
        ValueError: If an unknown loss function name is provided.
    """
    if loss_name == "mse":
        return mse_loss
    elif loss_name == "manhattan":
        return manhattan_loss
    elif loss_name == "hinge":
        return hinge_loss
    elif loss_name == "ssim":
        return ssim_loss
    elif loss_name == "combined_ssim_l1":
        return combined_ssim_l1_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def mse_loss(target_batch, cell_states):
    """
    Calculates the Mean Squared Error (MSE) loss between the target batch and cell states.

    Args:
        target_batch (torch.Tensor): The target images batch.
        cell_states (torch.Tensor): The current cell states.

    Returns:
        tuple: A tuple containing the loss per sample and the mean loss.
    """
    mse = nn.MSELoss(reduction='none')
    loss_batch = mse(cell_states[:, :4], target_batch).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def hinge_loss(target_batch, cell_states):
    """
    Calculates the Hinge loss between the target batch and cell states.

    Args:
        target_batch (torch.Tensor): The target images batch.
        cell_states (torch.Tensor): The current cell states.

    Returns:
        tuple: A tuple containing the loss per sample and the mean loss.
    """
    loss_batch = torch.max(torch.abs(target_batch - cell_states[:, :4]) - 0.5, torch.zeros_like(target_batch)).mean(
        dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def manhattan_loss(target_batch, cell_states):
    """
    Calculates the Manhattan distance (L1 loss) between the target batch and cell states.

    Args:
        target_batch (torch.Tensor): The target images batch.
        cell_states (torch.Tensor): The current cell states.

    Returns:
        tuple: A tuple containing the loss per sample and the mean loss.
    """
    loss_batch = (torch.abs(target_batch - cell_states[:, :4])).sum(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def ssim_loss(target_batch, cell_states):
    """
    Calculates the Structural Similarity Index (SSIM) loss between the target batch and cell states.

    Args:
        target_batch (torch.Tensor): The target images batch.
        cell_states (torch.Tensor): The current cell states.

    Returns:
        tuple: A tuple containing the loss per sample and the mean loss.
    """
    ssim_value = ssim(cell_states[:, :4], target_batch, data_range=1.0, size_average=False)
    loss_batch = 1 - ssim_value
    return loss_batch, loss_batch.mean()


def combined_ssim_l1_loss(target_batch, cell_states, alpha=0.5):
    """
    Calculates a combined loss using both SSIM and L1 loss.

    Args:
        target_batch (torch.Tensor): The target images batch.
        cell_states (torch.Tensor): The current cell states.
        alpha (float): Weight for the SSIM loss component (default: 0.5).

    Returns:
        tuple: A tuple containing the combined loss per sample and the mean loss.
    """
    # Calculate SSIM loss
    ssim_value = ssim(cell_states[:, :4], target_batch, data_range=1.0, size_average=False)
    ssim_loss_batch = 1 - ssim_value

    # Calculate L1 loss
    l1_loss = nn.L1Loss(reduction='none')
    l1_loss_batch = l1_loss(cell_states[:, :4], target_batch).mean(dim=[1, 2, 3])

    # Combine SSIM and L1 losses
    combined_loss_batch = alpha * ssim_loss_batch + (1 - alpha) * l1_loss_batch
    return combined_loss_batch, combined_loss_batch.mean()
