from PIL import Image
import os


IMAGE_DIR = r"C:\Users\ad_xleong\Desktop\coral-sleuth\images"


def check_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add/modify the image file extensions you're interested in
            try:
                print(f"Processing image: {filename}")
                img = Image.open(os.path.join(directory, filename))  # Try to open the image file
                img.verify()  # This will check for inconsistencies in the file
                print("DONE!")
            except (IOError, SyntaxError) as e:  # Catch exceptions raised by corrupt files
                print('Bad file:', filename)  # Print out the names of corrupt files

check_images(IMAGE_DIR)
