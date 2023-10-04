import os
import numpy as np
import csv
import cv2
import sys
import logging
import tensorflow as tf

from tensorflow.python.client import device_lib

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR, WEIGHT_DIR, MODEL_DIR
from src.model import CoralReefClassifier
from src.utils import logging_config, excel
from src.utils.custom_metrics import recall_m, precision_m, f1_m


logger = logging.getLogger(__name__)

annotation_file = "combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample_reduced_1k.csv"
annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)

model_h5 = "coral_reef_classifier_efficientnetv2_full_epoch_200_1_batchsize_32_combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample_reduced_1k_scale_0.2.h5"
model_path = os.path.join(MODEL_DIR, model_h5)

image = "i0201a.png"
image_path = os.path.join(IMAGE_DIR, image)


def create_label_mapping_from_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader)
        labels = [row[3] for row in reader]

    unique_labels = sorted(set(labels))
    label_mapping = {idx: label for idx, label in enumerate(unique_labels)}
    
    return label_mapping


def load_saved_model(model_path):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"recall_m": recall_m, "f1_m": f1_m, "precision_m": precision_m}                
    )


def predict_coral_image(model, image_path, top_k=3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to 224x224
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]

    return [(idx, predictions[idx]) for idx in top_indices]


if __name__ == '__main__':
    label_mapping = create_label_mapping_from_csv(annotation_path)

    # Load the saved model
    model = load_saved_model(model_path)
    top_predictions = predict_coral_image(model, image_path)

    # Print out the predictions using label_mapping
    for idx, prob in top_predictions:
        label = label_mapping[idx]
        print(f"Class: {label}, Probability: {prob:.4f}")