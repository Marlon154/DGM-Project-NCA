from medmnist import BloodMNIST
from PIL import Image
import numpy as np

if __name__ == "__main__":
    # Use a valid size (28, 64, 128, or 224)
    image_size = 28
    dataset = BloodMNIST(split="test", download=True, size=image_size)

    # Get the first image
    image, label = dataset[0]

    # Save the image
    image.save(f"blood-{image_size}.png")
    print(f"Image saved")
