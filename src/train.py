from torchvision.transforms.v2.functional import pad_image
from tqdm import tqdm


from nca import NCA
from utils import *


def get_seed(param, param1):
    pass


def loss_fun(target_batch, cell_states):
    pass


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target = load_image(config["target_path"], config["img_size"])
    target = pad_image(target, config["padding"])
    target = target.to(device)
    target_batch = target.repeat(config["batch_size"], 1, 1, 1)

    model = NCA(n_channels=16, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    seed = get_seed(config["img_size"], config["n_channels"])
    seed = pad_image(seed, config["padding"])
    seed = seed.to(device)
    pool = seed.clone().repeat(config["pool_size"], 1, 1, 1)

    loss_values = []

    for iteration in tqdm(range(config["iterations"])):
        batch_indices = np.random.choice(
            config["pool_size"], config["batch_size"], replace=False
        ).tolist()

        cell_states = pool[batch_indices]

        for _ in range(np.random.randint(64, 96)):
            cell_states = model(cell_states)

        loss_batch, loss = loss_fun(target_batch, cell_states)

        loss_values.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_loss_index = loss_batch.argmax().item()
        pool_index = batch_indices[max_loss_index]
        remaining_indices = [i for i in range(config["batch_size"]) if i != max_loss_index]
        pool_remaining_indices = [i for i in batch_indices if i != pool_index]

        pool[pool_index] = seed.clone()
        pool[pool_remaining_indices] = cell_states[remaining_indices].detach()

        if config["damage"]:
            best_loss_indices = np.argsort(loss_batch.detach().cpu().numpy())[:3]
            best_pool_indices = [batch_indices[i] for i in best_loss_indices]

            for n in range(3):
                damage = 1.0 - make_circle_masks(config["img_size"]).to(device)
                pool[best_pool_indices[n]] *= damage

    torch.save(model.state_dict(), config["model_path"])


if __name__ == "__main__":
    config = {
        "target_path": "data/emoji.png",
        "img_size": 128,
        "padding": 16,
        "n_channels": 4,
        "batch_size": 4,
        "pool_size": 256,
        "learning_rate": 1e-3,
        "iterations": 5000,
        "damage": False,
        "model_path": "models/nca.pth",
    }
    main(config)
