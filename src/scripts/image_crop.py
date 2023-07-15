import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import IMAGE_DIR

image_name = "mcr_lter1_fringingreef_pole1-2_qu1_20080415.jpg"
input_image = os.path.join(IMAGE_DIR, image_name)

img = Image.open(input_image)

width, height = img.size

x_center = 828
y_center = 495

scale = 0.15  # Adjust this scale to change the size of the cropped image

half_width = int(width * scale) // 2
half_height = int(height * scale) // 2

x_min = max(0, x_center - half_width)
y_min = max(0, y_center - half_height)
x_max = min(width, x_center + half_width)
y_max = min(height, y_center + half_height)
bbox = (x_min, y_min, x_max, y_max)

cropped_img = img.crop(bbox)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Change the figure size here

# Display the original image in the first subplot
axs[0].imshow(img)
axs[0].set_title(f'Original Image - Resolution: {width}x{height}, Coordinates: {x_center, y_center}')

# Draw the center point on the original image
axs[0].scatter(x_center, y_center, c='red')

# Display the cropped image in the second subplot
axs[1].imshow(cropped_img)
cropped_width, cropped_height = cropped_img.size
axs[1].set_title(f'Cropped Image - Resolution: {cropped_width}x{cropped_height}, Scale: {scale}')

# Draw the center point on the cropped image
axs[1].scatter(x_center - x_min, y_center - y_min, c='red')

# Remove axes for a better visual
axs[0].axis('off')
axs[1].axis('off')

plt.show()
