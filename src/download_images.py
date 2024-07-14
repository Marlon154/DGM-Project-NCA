import argparse
import os.path
from medmnist import INFO
import importlib
from PIL import Image
import numpy as np


def get_dataset(dataset_name):
    """Dynamically import the dataset class."""
    module = importlib.import_module('medmnist')
    class_name = dataset_name.capitalize()
    class_name = class_name.replace('mnist', 'MNIST').replace('3d', '3D')
    return getattr(module, class_name)


def save_sample_image(dataset_name, split="test", image_size=28, sample_index=0):
    DataClass = get_dataset(dataset_name)
    dataset = DataClass(split=split, download=True, size=image_size)
    image, label = dataset[sample_index]

    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[0] == 3:  # If it's a 3D image (3 channels)
            image = Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8))
        elif len(image.shape) == 3:  # If it is 3D
            middle_slice = image.shape[0] // 2
            image = Image.fromarray((image[middle_slice] * 255).astype(np.uint8))
        else:
            image = Image.fromarray((image[0] * 255).astype(np.uint8))

    filename = os.path.join("data", dataset_name, f"{image_size}-{split}-{sample_index}.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)
    print(f"Image saved as {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a sample image from a MedMNIST dataset")

    # Use the keys from INFO directly
    dataset_choices = list(INFO.keys())

    parser.add_argument("dataset", type=str, choices=[choice for choice in dataset_choices], help="Name of the MedMNIST dataset")
    parser.add_argument("--size", type=int, choices=[28, 64, 128, 224], default=28, help="Image size (default: 28) options: 28, 64, 128, 224")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split (default: test)")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample image (default: 0)")

    args = parser.parse_args()
    dataset_name = next(name for name in dataset_choices if name.lower() == args.dataset)

    save_sample_image(dataset_name, args.split, args.size, args.index)
