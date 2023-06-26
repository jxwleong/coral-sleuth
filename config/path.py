import os 

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
print(ROOT_DIR)