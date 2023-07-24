import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import IMAGE_DIR

image_name = "mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg"
image_path = os.path.join(IMAGE_DIR, image_name)

# Define  augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),  # rotate the image up to 30 degrees clockwise or counter-clockwise
])

# Load an image
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if image is None:
    print(f'Error loading image {image_path}')
else:
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the augmentation pipeline
    data = {"image": image}
    augmented = transform(**data)
    augmented_image = augmented["image"]

    # Show the original and augmented images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(augmented_image)
    ax[1].set_title('Augmented Image')
    plt.show()
