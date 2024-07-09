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
    if filter_name == "identity":
        return identity_filter(n_channels, device)
    elif filter_name == "sobel":
        return sobel_filter(n_channels, device)
    elif filter_name == "laplacian":
        return laplacian_filter(n_channels, device)
    elif filter_name == "gaussian":
        return gaussian_filter(n_channels, device)
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

def identity_filter(n_channels, device):
    """
    Creates an identity filter kernel.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Identity filter kernel.
    """
    identity = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
    identity = torch.outer(identity, identity)
    kernel = identity.repeat((n_channels, 1, 1))[:, None, ...].to(device)
    return kernel

def sobel_filter(n_channels, device):
    """
    Creates Sobel filter kernels for edge detection.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Sobel filter kernels (horizontal and vertical).
    """
    filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32) / 8
    filter_y = filter_x.t()
    kernel = torch.stack([filter_x, filter_y], dim=0)
    kernel = kernel.repeat(n_channels, 1, 1, 1).to(device)
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
    kernel = laplacian.repeat((n_channels, 1, 1))[:, None, ...].to(device)
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

def apply_filter(x, kernel):
    """
    Applies the given filter kernel to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        kernel (torch.Tensor): Filter kernel to apply.

    Returns:
        torch.Tensor: Filtered output tensor.
    """
    if kernel.size(0) == x.size(1) and kernel.size(1) == 2:  # Sobel filter case
        # Apply horizontal and vertical Sobel filters
        gx = F.conv2d(x, kernel[:, 0:1, :, :], padding=1, groups=x.size(1))
        gy = F.conv2d(x, kernel[:, 1:2, :, :], padding=1, groups=x.size(1))
        return torch.cat([gx, gy], dim=1)
    else:
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])