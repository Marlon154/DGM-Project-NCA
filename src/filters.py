import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class IdentityFilter(nn.Module):
    def __init__(self, n_channels):
        super(IdentityFilter, self).__init__()
        self.n_channels = n_channels
        identity = torch.tensor([0, 1, 0], dtype=torch.float32)
        identity = torch.outer(identity, identity)
        self.kernel = identity.unsqueeze(0).unsqueeze(0).repeat(n_channels, 1, 1, 1)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=self.n_channels)

class SobelEdgeFilter(nn.Module):
    def __init__(self, n_channels):
        super(SobelEdgeFilter, self).__init__()
        self.n_channels = n_channels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).repeat(n_channels, 1, 1, 1)
        self.sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).repeat(n_channels, 1, 1, 1)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.n_channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.n_channels)
        return torch.sqrt(gx**2 + gy**2)

class LaplacianFilter(nn.Module):
    def __init__(self, n_channels):
        super(LaplacianFilter, self).__init__()
        self.n_channels = n_channels
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.laplacian = laplacian.unsqueeze(0).unsqueeze(0).repeat(n_channels, 1, 1, 1)

    def forward(self, x):
        return F.conv2d(x, self.laplacian, padding=1, groups=self.n_channels)

class GaborFilterBank(nn.Module):
    def __init__(self, n_channels, n_orientations=4, n_scales=2):
        super(GaborFilterBank, self).__init__()
        self.n_channels = n_channels
        self.n_orientations = n_orientations
        self.n_scales = n_scales

        self.gabor_filters = nn.ParameterList([
            nn.Parameter(self._generate_gabor_filter(orientation, scale))
            for orientation in range(n_orientations)
            for scale in range(n_scales)
        ])

    def _generate_gabor_filter(self, orientation, scale):
        sigma = 0.56 * math.pi / (2 ** scale)
        lambda_val = math.pi / (2 ** scale)
        gamma = 0.5
        theta = orientation * math.pi / self.n_orientations

        kernel_size = int(2 * math.ceil(2.5 * sigma) + 1)
        x, y = torch.meshgrid(torch.arange(kernel_size) - kernel_size // 2,
                              torch.arange(kernel_size) - kernel_size // 2)
        
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)

        gb = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * \
             torch.cos(2 * math.pi * x_theta / lambda_val)

        return gb.unsqueeze(0).unsqueeze(0).repeat(self.n_channels, 1, 1, 1)

    def forward(self, x):
        gabor_responses = [F.conv2d(x, gabor_filter, padding=gabor_filter.size(-1)//2, groups=self.n_channels)
                           for gabor_filter in self.gabor_filters]
        return torch.cat(gabor_responses, dim=1)
    

def get_filter(filter_name, n_channels):
    filters = {
        "identity": IdentityFilter,
        "sobel": SobelEdgeFilter,
        "laplacian": LaplacianFilter,
        "gabor": GaborFilterBank,
    }
    
    if filter_name not in filters:
        raise ValueError(f"Unknown filter: {filter_name}. Available filters: {', '.join(filters.keys())}")
    
    return filters[filter_name](n_channels)