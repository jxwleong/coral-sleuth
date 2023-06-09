from PIL import Image, ImageFile
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import IMAGE_DIR

input_folder = IMAGE_DIR
output_folder = input_folder  # Saving output to the same folder

# Tell PIL to accept truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set to True if you want to overwrite existing files
overwrite_existing_files = True

for jpeg_file in os.listdir(input_folder):
    if jpeg_file.endswith(".jpg"):
        png_file = os.path.splitext(jpeg_file)[0] + ".png"
        output_file = os.path.join(output_folder, png_file)
        if os.path.exists(output_file) and not overwrite_existing_files:
            print(f"{output_file} exists! Skipping...")
            continue
        try:
            print(f"Converting {jpeg_file} to png... ", end=" ")
            img = Image.open(os.path.join(input_folder, jpeg_file))
            # Remove ICC profile as it caused this issue:
            # libpng warning: iCCP: known incorrect sRGB profile
            if 'icc_profile' in img.info:
                del img.info['icc_profile']
            img.save(output_file)
            print("DONE")
        except Exception as e:
            print(f"Error converting {jpeg_file} to png: {str(e)}")
