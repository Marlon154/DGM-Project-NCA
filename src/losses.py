import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from nca import to_rgba, to_rgb


def mse_loss_fun(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)
    mse = nn.MSELoss(reduction='none')
    loss_batch = mse(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ms_ssim(img1, img2, levels=5):
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    mssim = []
    mcs = []
    for i in range(levels):
        sim, cs = ssim(img1, img2, size_average=False)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
    
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    
    return (torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])).mean()

def sobel_filters():
    Gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
    return Gx, Gy

def detect_edges(image):
    Gx, Gy = sobel_filters()
    Gx = Gx.unsqueeze(0).unsqueeze(0).repeat(image.size(1), 1, 1, 1)
    Gy = Gy.unsqueeze(0).unsqueeze(0).repeat(image.size(1), 1, 1, 1)
    
    if image.is_cuda:
        Gx = Gx.cuda()
        Gy = Gy.cuda()
    
    G_x = F.conv2d(image, Gx, padding=1, groups=image.size(1))
    G_y = F.conv2d(image, Gy, padding=1, groups=image.size(1))
    
    return torch.sqrt(G_x**2 + G_y**2)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = [self.slice1(input), self.slice2(input), self.slice3(input)]
        target_features = [self.slice1(target), self.slice2(target), self.slice3(target)]
        loss = 0.0
        for i in range(len(input_features)):
            loss += F.mse_loss(input_features[i], target_features[i])
        return loss

def perceptual_loss_fun(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)

    loss_fn = PerceptualLoss()
    loss = loss_fn(cell_states_rgb.permute(0, 3, 1, 2), target_rgb.permute(0, 3, 1, 2))
    return loss, loss.unsqueeze(0).repeat(cell_states.shape[0])

def ms_ssim_loss_fun(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)
    
    loss = 1 - ms_ssim(cell_states_rgb.permute(0, 3, 1, 2), target_rgb.permute(0, 3, 1, 2))
    return loss, loss.unsqueeze(0).repeat(cell_states.shape[0])

def combined_mse_edge_loss_fun(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)
    
    # MSE Loss
    mse_loss = nn.MSELoss(reduction='none')(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    
    # Edge Detection Loss
    cell_states_edges = detect_edges(cell_states_rgb.permute(0, 3, 1, 2))
    target_edges = detect_edges(target_rgb.permute(0, 3, 1, 2))
    edge_loss = nn.MSELoss(reduction='none')(cell_states_edges, target_edges).mean(dim=[1, 2, 3])
    
    # Combine losses
    combined_loss = mse_loss + 0.5 * edge_loss
    
    return combined_loss, combined_loss


def get_loss_function(loss_name):
    if loss_name == "mse":
        return mse_loss_fun
    elif loss_name == "perceptual":
        return perceptual_loss_fun
    elif loss_name == "ms_ssim":
        return ms_ssim_loss_fun
    elif loss_name == "combined_mse_edge":
        return combined_mse_edge_loss_fun
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
