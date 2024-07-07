import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from nca import NCA, to_rgb, get_seed


def visualize_nca(model_path, img_size, padding, n_channels, n_steps, interval=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = NCA(n_channels=n_channels, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize the seed
    seed = get_seed(img_size + 2 * padding, n_channels, device)

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))  # Set figure size to be square
    ax.axis('off')

    # Initialize an empty image
    img = ax.imshow(np.zeros((img_size + 2 * padding, img_size + 2 * padding, 3)))

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
        update.step_text = ax.text(0.01, 0.99, f'Step: {frame}',
                                   transform=ax.transAxes, color='black',
                                   verticalalignment='top', fontsize=20)

        return [img, update.step_text]
    # Create the animation
    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=True)

    # Save the animation as a gif
    anim.save('nca_growth.gif', writer='pillow', dpi=100)

    plt.close(fig)
    print("Animation saved as 'nca_growth.gif'")


if __name__ == "__main__":
    model_path = "./models/nca_model.pth"
    img_size = 28
    padding = 0
    n_channels = 16
    n_steps = 200

    visualize_nca(model_path, img_size, padding, n_channels, n_steps)
