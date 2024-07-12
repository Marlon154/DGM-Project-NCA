import torch
import torch.nn.functional as F

def get_filter(filter_name, n_channels, device):
    """
    Returns the specified filter kernel based on the input name.

    Args:
        filter_name (str): Name of the filter to use.
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: The corresponding filter kernel.

    Raises:
        ValueError: If an unknown filter name is provided.
    """
    if filter_name == "sobel_identity":
        return sobel_identity_filter(n_channels, device)
    elif filter_name == "laplacian":
        return laplacian_filter(n_channels, device)
    elif filter_name == "gaussian":
        return gaussian_filter(n_channels, device)
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

def sobel_identity_filter(n_channels, device):
    """
    Creates a combination of an sobel and identity filter kernel.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Sobel Identity filter kernel.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32) / 8
    sobel_y = sobel_x.t()
    identity = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
    identity = torch.outer(identity, identity)
    kernel = torch.stack([identity, sobel_x, sobel_y], dim=0).repeat(n_channels, 1, 1)[:, None, ...].to(device)
    return kernel

def laplacian_filter(n_channels, device):
    """
    Creates a Laplacian filter kernel for edge detection.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Laplacian filter kernel.
    """
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32)
    kernel = laplacian.repeat(n_channels, 1, 1)[:, None, ...].to(device)
    return kernel

def gaussian_filter(n_channels, device, sigma=1.0):
    """
    Creates a Gaussian filter kernel for smoothing.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: Gaussian filter kernel.
    """
    size = 3
    x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32, device=device)
    x = x.repeat(size, 1)
    y = x.t()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    kernel = gaussian.repeat((n_channels, 1, 1))[:, None, ...].to(device)
    return kernel

def apply_filter(x, kernel, n_channels):
    """
    Applies the given filter kernel to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        kernel (torch.Tensor): Filter kernel to apply.

    Returns:
        torch.Tensor: Filtered output tensor.
    """
    return F.conv2d(x, kernel, padding=1, groups=n_channels)