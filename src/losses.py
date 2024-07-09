import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim, ms_ssim
from nca import to_rgba, to_rgb    

def get_loss_function(loss_name):
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
    
def loss_prep(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)
    return cell_states_rgb, target_rgb

def mse_loss(target_batch, cell_states):
    cell_states_rgb, target_rgb = loss_prep(target_batch, cell_states)
    mse = nn.MSELoss(reduction='none')
    loss_batch = mse(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()

def hinge_loss(target_batch, cell_states):
    cell_states_rgb, target_rgb = loss_prep(target_batch, cell_states)
    loss_batch = torch.max(torch.abs(target_rgb - cell_states_rgb) - 0.5, torch.zeros_like(target_rgb)).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()

def manhattan_loss(target_batch, cell_states):
    cell_states_rgb, target_rgb = loss_prep(target_batch, cell_states)
    loss_batch = (torch.abs(target_rgb - cell_states_rgb)).sum(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()

def ssim_loss(target_batch, cell_states):
    cell_states_rgb, target_rgb = loss_prep(target_batch, cell_states)
    ssim_value = ssim(cell_states_rgb, target_rgb, data_range=1.0, size_average=False)
    loss_batch = 1 - ssim_value
    return loss_batch, loss_batch.mean()

def combined_ssim_l1_loss(target_batch, cell_states, alpha=0.5):
    cell_states_rgb, target_rgb = loss_prep(target_batch, cell_states)
    ssim_value = ssim(cell_states_rgb, target_rgb, data_range=1.0, size_average=False)
    ssim_loss_batch = 1 - ssim_value
    l1_loss = nn.L1Loss(reduction='none')
    l1_loss_batch = l1_loss(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    combined_loss_batch = alpha * ssim_loss_batch + (1 - alpha) * l1_loss_batch
    return combined_loss_batch, combined_loss_batch.mean()
