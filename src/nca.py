import torch
import torch.nn as nn
from filters import get_filter, apply_filter

class NCA(nn.Module):
    def __init__(self, n_channels=16, num_h_channels=128, fire_rate=0.5, act_fun=nn.ReLU, device="cuda", filter_name="identity"):
        super(NCA, self).__init__()
        self.n_channels = n_channels
        self.num_h_channels = num_h_channels
        self.fire_rate = fire_rate
        self.act_fun = act_fun
        self.device = device
        self.filter_name = filter_name

        self.kernel = get_filter(filter_name, n_channels, device)

        self._create_conv_layers()

        self.to(device)

    def _create_conv_layers(self):
        input_channels = self.kernel.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, self.num_h_channels, kernel_size=1),
            self.act_fun(),
            nn.Conv2d(self.num_h_channels, self.n_channels, 1, bias=False)
        ).to(self.device)
        with torch.no_grad():
            self.conv[2].weight.zero_()

    def state_dict(self, **kwargs):
        state_dict = super().state_dict()
        state_dict['fire_rate'] = self.fire_rate
        state_dict['n_channels'] = self.n_channels
        state_dict['filter_name'] = self.filter_name
        state_dict['num_h_channels'] = self.num_h_channels
        state_dict['act_fun'] = self.act_fun
        return state_dict

    def load_state_dict(self, state_dict, device=None, **kwargs):
        self.device = device or self.device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fire_rate = state_dict.pop('fire_rate')
        self.n_channels = state_dict.pop('n_channels')
        self.num_h_channels = state_dict.pop('num_h_channels')
        self.act_fun = state_dict.pop('act_fun')
        self.filter_name = state_dict.pop('filter_name')
        self.kernel = get_filter(self.filter_name, self.n_channels, self.device)
        self._create_conv_layers()
        super().load_state_dict(state_dict)

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
