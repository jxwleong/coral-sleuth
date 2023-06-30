import os 


ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "data", "annotations")
WEIGHT_DIR = os.path.join(ROOT_DIR, "data", "weights")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "logs")

DEBUG_PATH = False
if DEBUG_PATH is True:
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"ANNOTATION_DIR: {ANNOTATION_DIR}")
    print(f"WEIGHT_DIR: {WEIGHT_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"LOG_DIR: {LOG_DIR}")