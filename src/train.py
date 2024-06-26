import torch
import torch.nn as nn
from torchvision.transforms.functional import pad
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from nca import NCA, to_rgb, to_rgba


def get_seed(img_size, n_channels, device):
    seed = torch.zeros((1, n_channels, img_size, img_size), dtype=torch.float32, device=device)
    seed[:, 3:4, img_size // 2, img_size // 2] = 1.0
    return seed


def loss_fun(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)
    mse = nn.MSELoss(reduction='none')
    loss_batch = mse(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def load_image(path, size, device):
    img = Image.open(path).convert('RGBA').resize((size, size))
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)


def make_circle_masks(diameter, device):
    x = torch.linspace(-1, 1, diameter, device=device)
    y = torch.linspace(-1, 1, diameter, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    center = torch.rand(2, device=device) * 0.4 + 0.3
    r = torch.rand(1, device=device) * 0.2 + 0.2
    mask = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 < r ** 2).float()
    return mask.unsqueeze(0).unsqueeze(0)


def main(config):
    wandb.init(project="dgm-nca", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = load_image(config["target_path"], config["img_size"], device)
    target = pad(target, config["padding"])
    target_batch = target.repeat(config["batch_size"], 1, 1, 1)

    model = NCA(n_channels=config["n_channels"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    seed = get_seed(config["img_size"] + 2 * config["padding"], config["n_channels"], device)
    pool = seed.clone().repeat(config["pool_size"], 1, 1, 1)

    loss_values = []

    for iteration in tqdm(range(config["iterations"])):
        batch_indices = torch.randperm(config["pool_size"], device=device)[:config["batch_size"]]

        cell_states = pool[batch_indices]

        for _ in range(torch.randint(64, 96, (1,)).item()):
            cell_states = model(cell_states)

        loss_batch, loss = loss_fun(target_batch, cell_states)

        loss_values.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_loss_index = loss_batch.argmax().item()
        pool_index = batch_indices[max_loss_index]
        remaining_indices = torch.tensor([i for i in range(config["batch_size"]) if i != max_loss_index], device=device)
        pool_remaining_indices = batch_indices[batch_indices != pool_index]

        pool[pool_index] = seed.clone()
        pool[pool_remaining_indices] = cell_states[remaining_indices].detach()

        if config["damage"]:
            best_loss_indices = torch.argsort(loss_batch)[:3]
            best_pool_indices = batch_indices[best_loss_indices]

            for n in range(3):
                damage = 1.0 - make_circle_masks(config["img_size"] + 2 * config["padding"], device)
                pool[best_pool_indices[n]] *= damage

        wandb.log({
            "loss": loss.item(),
            "iteration": iteration,
        })
        if iteration % 100 == 0:
            wandb.log({
                "target_image": wandb.Image(to_rgb(target[0].permute(1, 2, 0)).cpu().numpy()),
                "generated_image": wandb.Image(to_rgb(cell_states[0].permute(1, 2, 0)).detach().cpu().numpy()),
            })

    wandb.save(config["model_path"])
    wandb.finish()
    torch.save(model.state_dict(), config["model_path"])

    return loss_values


if __name__ == "__main__":
    config = {
        "target_path": "./data/pneumonia/image-pneumonia-32.png",
        "img_size": 128,
        "padding": 16,
        "n_channels": 16,
        "batch_size": 4,
        "pool_size": 256,
        "learning_rate": 1e-3,
        "iterations": 1000,
        "damage": False,
        "model_path": "./model/nca.pth",
    }
    loss_history = main(config)
