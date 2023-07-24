import matplotlib.pyplot as plt
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

# Define multiple points to be annotated
points = [
    (671,217,"Sand"), 
    (1252,971,"Sand"), 
    (548,1054,"Macro"),
    (1084,211,"CCA"),
    (447,667,"Porit"),
    (1123,1543,"Porit"),
    (1766,1248,"CCA")
]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))  # Change the figure size here

# Display the original image
ax.imshow(img)
ax.set_title(f'{image_name} - Resolution: {width}x{height}')

# Annotate each point on the image
# From the annotaiton file, Rows comes first
for y, x, text in points:
    label = f"({x}, {y}, {text})"
    ax.scatter(x, y, c='red')
    ax.text(x+20, y-20, label, color='red', fontsize=15)

# Remove axes for a better visual
ax.axis('off')

plt.show()
