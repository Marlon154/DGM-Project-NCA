import torch
import torch.nn as nn
from torchvision.transforms.functional import pad
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from nca import NCA, to_rgb, to_rgba


def get_seed(img_size, n_channels):
    seed = torch.zeros((1, n_channels, img_size, img_size), dtype=torch.float32)
    seed[:, 3:4, img_size // 2, img_size // 2] = 1.0
    return seed


def loss_fun(target_batch, cell_states):
    # Convert cell_states to RGBA format
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))

    # Ensure target_batch is in the correct format (NCHW to NHWC)
    target_batch = target_batch.permute(0, 2, 3, 1)

    # Convert both to RGB
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)

    # Calculate MSE loss
    mse = nn.MSELoss(reduction='none')
    loss_batch = mse(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def load_image(path, size):
    img = Image.open(path).convert('RGBA').resize((size, size))
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def make_circle_masks(diameter):
    x = torch.linspace(-1, 1, diameter)
    y = torch.linspace(-1, 1, diameter)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    center = torch.rand(2) * 0.4 + 0.3
    r = torch.rand(1) * 0.2 + 0.2
    mask = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 < r ** 2).float()
    return mask.unsqueeze(0).unsqueeze(0)


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = load_image(config["target_path"], config["img_size"])
    target = pad(target, config["padding"])
    target = target.to(device)
    target_batch = target.repeat(config["batch_size"], 1, 1, 1)

    model = NCA(n_channels=config["n_channels"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    seed = get_seed(config["img_size"] + 2 * config["padding"], config["n_channels"])
    seed = seed.to(device)
    pool = seed.clone().repeat(config["pool_size"], 1, 1, 1)

    loss_values = []

    for iteration in tqdm(range(config["iterations"])):
        batch_indices = torch.randperm(config["pool_size"])[:config["batch_size"]]

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
        remaining_indices = torch.tensor([i for i in range(config["batch_size"]) if i != max_loss_index])
        pool_remaining_indices = batch_indices[batch_indices != pool_index]

        pool[pool_index] = seed.clone()
        pool[pool_remaining_indices] = cell_states[remaining_indices].detach()

        if config["damage"]:
            best_loss_indices = torch.argsort(loss_batch)[:3]
            best_pool_indices = batch_indices[best_loss_indices]

            for n in range(3):
                damage = 1.0 - make_circle_masks(config["img_size"] + 2 * config["padding"]).to(device)
                pool[best_pool_indices[n]] *= damage

    torch.save(model.state_dict(), config["model_path"])

    return loss_values


if __name__ == "__main__":
    def load_emoji(emoji):
        code = hex(ord(emoji))[2:].lower()
        url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
        return load_image(url)

    config = {
        "target_path": "./data/demo/demo-image-pneumonia-32.png",
        "img_size": 128,
        "padding": 16,
        "n_channels": 16,
        "batch_size": 4,
        "pool_size": 256,
        "learning_rate": 1e-3,
        "iterations": 5000,
        "damage": False,
        "model_path": "models/nca.pth",
    }
    loss_history = main(config)

    # Temporary plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('loss_history.png')
    plt.close()
