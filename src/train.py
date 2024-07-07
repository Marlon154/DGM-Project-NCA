import torch
from torch import nn
from torchvision.transforms.functional import pad
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from nca import NCA, to_rgb, to_rgba, get_seed


def load_image(path, size, device):
    img = Image.open(path).convert('RGBA').resize((size, size))
    return torch.tensor(np.array(img) / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)


def make_circle_masks(diameter, n_channels, device):
    x = torch.linspace(-1, 1, diameter, device=device)
    y = torch.linspace(-1, 1, diameter, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    center = torch.rand(2, device=device) * 0.4 + 0.3
    r = torch.rand(1, device=device) * 0.2 + 0.2
    mask = ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 < r ** 2).float()
    return mask.unsqueeze(0).repeat(n_channels, 1, 1)


def loss_fun(target_batch, cell_states):
    cell_states_rgba = to_rgba(cell_states.permute(0, 2, 3, 1))
    target_batch = target_batch.permute(0, 2, 3, 1)
    cell_states_rgb = to_rgb(cell_states_rgba)
    target_rgb = to_rgb(target_batch)
    mse = nn.MSELoss(reduction='none')
    loss_batch = mse(cell_states_rgb, target_rgb).mean(dim=[1, 2, 3])
    return loss_batch, loss_batch.mean()


def main(config):
    wandb.init(project="dgm-nca", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = get_seed(config["img_size"], config["n_channels"], device)
    pool = seed.expand(config["pool_size"], -1, -1, -1).clone()

    target = load_image(config["target_path"], config["img_size"], device)
    target = pad(target, config["padding"])
    batched_target = target.repeat(config["batch_size"], 1, 1, 1)

    model = NCA(n_channels=config["n_channels"], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])


    for iteration in tqdm(range(config["iterations"])):
        batch_indices = torch.randperm(config["pool_size"], device=device)[:config["batch_size"]]

        x = pool[batch_indices]

        steps = torch.randint(config["min_steps"], config["max_steps"], (1,)).item()
        for _ in range(steps):
            x = model.forward(x)

        loss_batch = ((batched_target - x[:, :4, ...])**2).mean(dim=[1, 2, 3])
        loss = loss_batch.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_value"])

        with torch.no_grad():
            # Update pool
            pool[batch_indices] = x.detach()

            # Replace the highest loss sample with seed
            max_loss_idx = loss_batch.argmax()
            pool[batch_indices[max_loss_idx]] = seed

            if config["damage"] and iteration > config["damage_start"]:
                # Apply damage to the best samples
                best_indices = loss_batch.argsort()[:3]
                for idx in best_indices:
                    damage_mask = 1.0 - make_circle_masks(config["img_size"] + 2 * config["padding"],
                                                          config["n_channels"], device)
                    pool[batch_indices[idx]] *= damage_mask

        wandb.log({
            "loss": loss.item(),
            "iteration": iteration,
        })

    wandb.finish()
    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    config = {
        "target_path": "./data/blood/blood-28.png",
        "img_size": 28,
        "padding": 0,
        "min_steps": 64,
        "max_steps": 96,
        "n_channels": 16,
        "batch_size": 8,
        "pool_size": 1024,
        "learning_rate": 0.001,
        "iterations": 2000,
        "damage": True,
        "damage_start": 500,
        "clip_value": 1.0,
        "model_path": "./models/nca_model.pth",
    }
    main(config)
