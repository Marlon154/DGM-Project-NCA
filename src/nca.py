import torch
import torch.nn as nn


class NCA(nn.Module):
    def __init__(self, n_channels=16, num_h_channels=128, fire_rate=0.5, act_fun=nn.ReLU, device="cuda"):
        super(NCA, self).__init__()
        self.fire_rate = fire_rate
        self.n_channels = n_channels
        self.device = device
        tmp_f = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32) / 8
        filter_x = tmp_f
        filter_y = tmp_f.t()
        identity = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
        identity = torch.outer(identity, identity)
        kernel = torch.stack([identity, filter_x, filter_y], dim=0)
        kernel = kernel.repeat((n_channels, 1, 1))[:, None, ...].to(device)
        self.kernel = kernel
        self.conv = nn.Sequential(
            nn.Conv2d(3 * n_channels, num_h_channels, kernel_size=1),
            act_fun(),
            nn.Conv2d(num_h_channels, n_channels, 1, bias=False)
        ).to(device)

        with torch.no_grad():
            self.conv[2].weight.zero_()

        self.to(device)

    def perceive(self, x):
        return nn.functional.conv2d(x, self.kernel, padding=1, groups=self.n_channels)

    def forward(self, x):
        begin_living_cells = get_living_cells(x)
        dx = self.conv(self.perceive(x))
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
