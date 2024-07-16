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
    with_identity = True if "_identity" in filter_name else False
    filter_name = filter_name.split("_")[0]
    if filter_name == "identity":
        return identity_filter_kernel(n_channels, device)
    elif filter_name == "sobel":
        return sobel_filter_kernel(n_channels, device, with_identity)
    elif filter_name == "laplacian":
        return laplacian_filter_kernel(n_channels, device, with_identity)
    elif filter_name == "gaussian":
        return gaussian_filter_kernel(n_channels, device, with_identity)
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

def identity_filter_kernel(n_channels, device):
    """
    Creates an identity filter kernel.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Identity filter kernel.
    """
    identity = identity_filter(device)
    return get_kernel(identity, n_channels, device)

def sobel_filter_kernel(n_channels, device, with_identity=False):
    """
    Creates Sobel filter kernel for edge detection.
    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.
        with_identity (bool, optional): If True, the identity filter kernel is used.
    Returns:
        torch.Tensor: Sobel filter kernel.
    """
    filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32) / 8
    filter_y = filter_x.t()
    filters = [filter_x, filter_y]
    if with_identity:
        filters = [identity_filter(device)] + filters
    filter = filter_concatenation(filters)
    return get_kernel(filter, n_channels, device)

def laplacian_filter_kernel(n_channels,device,with_identity=False):
    """
    Creates a Laplacian filter kernel for edge detection.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.
        with_identity (bool, optional): If True, the identity filter is concatenated.

    Returns:
        torch.Tensor: Laplacian filter kernel.
    """
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32)
    if with_identity:
        laplacian = filter_concatenation([identity_filter(device),laplacian])
    return get_kernel(laplacian, n_channels, device)

def gaussian_filter_kernel(n_channels, device, with_identity=False,sigma=1.0):
    """
    Creates a Gaussian filter kernel for smoothing.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.
        sigma (float): Standard deviation of the Gaussian distribution.
        with_identity (bool, optional): If True, the identity filter is concatenated.
    Returns:
        torch.Tensor: Gaussian filter kernel.
    """
    size = 3
    x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32, device=device)
    x = x.repeat(size, 1)
    y = x.t()
    gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    if with_identity:
        gaussian = filter_concatenation([identity_filter(device), gaussian])
    return get_kernel(gaussian, n_channels, device)

def identity_filter(device):
    """
    Creates an identity filter kernel.

    Args:
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Identity filter.
    """
    identity = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
    identity = torch.outer(identity, identity)
    return identity

def filter_concatenation(list_of_filters):
    """
    Concatenates all filters into a single filter.

    Args:
        list_of_filters (list): List of filters to concatenate.

    Returns:
        torch.Tensor: Concatenated filter.
    """
    kernel = torch.stack(list_of_filters, dim=0)
    return kernel

def get_kernel(filter, n_channels, device):
    """
    Creates the filter kernel.

    Args:
        filter (torch.Tensor): Filter for which to create the kernel.
        n_channels (int): Number of input channels.
        device (torch.device): Device to create the filter on.

    Returns:
        torch.Tensor: Filter kernel.
    """
    return filter.repeat((n_channels, 1, 1))[:, None, ...].to(device)


def apply_filter(x, kernel):
    """
    Applies the given filter kernel to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        kernel (torch.Tensor): Filter kernel to apply.

    Returns:
        torch.Tensor: Filtered output tensor.
    """
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])