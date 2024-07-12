import torch
import torch.nn as nn
from filters import get_filter, apply_filter

class NCA(nn.Module):
    def __init__(self, n_channels=16, num_h_channels=128, fire_rate=0.5, act_fun=nn.ReLU, device="cuda", filter_name="identity"):
        super(NCA, self).__init__()
        self.fire_rate = fire_rate
        self.n_channels = n_channels
        self.device = device
        self.filter_name = filter_name
        
        self.kernel = get_filter(filter_name, n_channels, device)

        input_channels = self.kernel.shape[0]
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, num_h_channels, kernel_size=1),
            act_fun(),
            nn.Conv2d(num_h_channels, n_channels, 1, bias=False)
        ).to(device)
        
        with torch.no_grad():
            self.conv[2].weight.zero_()
        
        self.to(device)

    def perceive(self, x):
        return apply_filter(x, self.kernel)

    def forward(self, x):
        begin_living_cells = get_living_cells(x)
        perceived = self.perceive(x)
        dx = self.conv(perceived)
        update = (torch.rand(x[:, :1, :, :].shape, device=self.device) <= self.fire_rate)
        x = x + dx * update
        return x * (begin_living_cells & get_living_cells(x))


def get_seed(img_size, n_channels, device):
    seed = torch.zeros((1, n_channels, img_size, img_size), dtype=torch.float32, device=device)
    seed[:, 3:, img_size // 2, img_size // 2] = 1.0
    return seed


def get_living_cells(x):
    return nn.functional.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1


def to_rgba(x):
    return x[..., :4]


def to_alpha(x):
    return torch.clip(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb
