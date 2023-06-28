from PIL import Image
import os
import sys


ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import IMAGE_DIR

input_folder = IMAGE_DIR
output_folder = input_folder  # Saving output to the same folder

for jpeg_file in os.listdir(input_folder):
    if jpeg_file.endswith(".jpg"):
        print(f"Converting {jpeg_file} to png... ", end=" ")
        img = Image.open(os.path.join(input_folder, jpeg_file))
        png_file = os.path.splitext(jpeg_file)[0] + ".png"
        img.save(os.path.join(output_folder, png_file))
        print("DONE")