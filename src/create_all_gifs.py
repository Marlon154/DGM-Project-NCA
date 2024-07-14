import os
import yaml

from visualize import visualize_nca


def create_gifs_for_all_models(model_folder, config_path, n_steps=200, interval=50):
    # Load the configuration once, assuming it's the same for all models
    config = yaml.safe_load(open(config_path, "r"))

    # List all .pth files in the model_folder
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]

    # Loop through each model file and create a GIF
    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        animation_path = os.path.join("figures", model_file.replace('.pth', '.gif'))

        # Call the visualize_nca function for each model
        visualize_nca(model_path, config, n_steps, animation_path, interval)
        print(f"Created GIF for model {model_file} at {animation_path}")


# Example usage
model_folder = "./models"
config_path = "conf/config.yaml"
create_gifs_for_all_models(model_folder, config_path)
