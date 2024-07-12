import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.animation import FuncAnimation
from nca import NCA, to_rgb, get_seed


def visualize_nca(model_path, config, n_steps, animation_path, interval=50):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    # device = torch.device('mps')
    else:
        device = torch.device('cpu')
    # Load the trained model
    model = NCA(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize the seed
    img_size = config["img_size"]
    seed = get_seed(img_size + 2 * config["padding"], model.n_channels, device)

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size to be square
    ax.axis('off')

    # Initialize an empty image
    img = ax.imshow(np.zeros((img_size + 2 * config["padding"], img_size + 2 * config["padding"], 3)))

    def update(frame):
        nonlocal seed
        with torch.no_grad():
            seed = model.forward(seed)

        # Convert the output to RGB
        rgb_output = to_rgb(seed[0].permute(1, 2, 0)).cpu().numpy()

        # Update the image data
        img.set_array(rgb_output)

        # Remove the previous step number text
        if 'step_text' in update.__dict__:
            update.step_text.remove()

        # Add the current step number text
        text_content = f'Step: {frame}\nFilter: {model.filter_name}'

        update.step_text = ax.text(0.01, 0.99, text_content,
                                   transform=ax.transAxes, color='black',
                                   verticalalignment='top', fontsize=20)

        return [img, update.step_text]

    # Create the animation
    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=True)

    # Save the animation as a gif
    anim.save(animation_path, writer='pillow', dpi=100)

    plt.close(fig)
    print(f"Animation saved as '{animation_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Sweep")

    model_path = "models/nca_model-28.pth"
    config_path = "conf/config.yaml"
    animation_path = "figures/nca_model.gif"

    parser.add_argument("-c", "--config_path", default=config_path, help="Path to config file")
    parser.add_argument("-m", "--model_path", default=model_path, help="Path to model file")
    parser.add_argument("-n", "--n_steps", type=int, default=200, help="Number of steps")
    parser.add_argument("-i", "--interval", type=int, default=50, help="Interval")
    parser.add_argument("-p", "--animation_path", type=str, default=animation_path, help="Path to save animation")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, "r"))
    visualize_nca(args.model_path, config, args.n_steps, args.animation_path, args.interval)
