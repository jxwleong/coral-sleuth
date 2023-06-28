""" 
Script name: check_image_existence.py

This script checks the existence of image files specified in a CSV file.

The CSV file should have a column named "Name" which lists the image file names.
The directory containing the image files should be defined by the variable 'image_dir'.

The script goes through each row in the CSV file, extracts the image file name,
and checks whether the image file exists in the specified directory.

If an image file does not exist, the script prints a message to the console
stating that the image file was not found.
"""

import csv
import os

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR, IMAGE_DIR



csv_file = os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k.csv")

not_found_images = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        image_name = row[0]  # "Name" is in the first column
        image_path = os.path.join(IMAGE_DIR, image_name)
        if not os.path.exists(image_path) and image_name not in not_found_images:
            not_found_images.append(image_name)

if not_found_images:
    print(f"The following images were not found: {not_found_images}")
else:
    print("All images were found.")
