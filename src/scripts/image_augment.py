import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from PIL import Image
import random

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import IMAGE_DIR

image_name = "mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg"
image_path = os.path.join(IMAGE_DIR, image_name)

# Define  augmentation pipeline
""" 
transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=30, p=0.3)
])
"""


def apply_transformations(image):
    transformations = []
    
    if random.random() < 0.3:
        image = A.HorizontalFlip(p=1)(image=image)["image"]
        transformations.append("Horizontal Flip")
    
    if random.random() < 0.3:
        image = A.VerticalFlip(p=1)(image=image)["image"]
        transformations.append("Vertical Flip")

    if random.random() < 0.3:
        image = A.RandomBrightnessContrast(p=1)(image=image)["image"]
        transformations.append("Random Brightness/Contrast")

    if random.random() < 0.3:
        image = A.Rotate(limit=30, p=1)(image=image)["image"]
        transformations.append("Rotation")

    return image, transformations

img = Image.open(image_path)

width, height = img.size

x_center = 828
y_center = 495

scale = 0.2  # Adjust this scale to change the size of the cropped image

half_width = int(width * scale) // 2
half_height = int(height * scale) // 2

x_min = max(0, x_center - half_width)
y_min = max(0, y_center - half_height)
x_max = min(width, x_center + half_width)
y_max = min(height, y_center + half_height)
bbox = (x_min, y_min, x_max, y_max)

cropped_img = img.crop(bbox)

if cropped_img is None:
    print(f"Error loading image {image_path}")
else:
    # Convert the cropped image to a numpy array
    cropped_img_np = np.array(cropped_img)
    augmented_image, transformations = apply_transformations(cropped_img_np)

    # Create a string with the transformations applied
    transformations_str = ", ".join(transformations) if transformations else "No Transformations"

    print(f"Applied transformations: {transformations_str}")


    # Show the original and augmented images side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cropped_img)
    ax[0].set_title("Original Image")
    ax[1].imshow(augmented_image)
    ax[1].set_title(f"Augmented Image ({transformations_str})")
    plt.show()