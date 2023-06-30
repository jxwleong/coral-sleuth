""" 
Run in remote SSH
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "model_training.py" 
"""

import os
import numpy as np
import csv
import cv2
import json 
import time 
import logging

from tensorflow.python.client import device_lib

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR, WEIGHT_DIR, MODEL_DIR
from src.model import CoralReefClassifier
from src.utils import logging_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    annotation_filename = "annotations_coralnet_only_trimmed.csv"
    annotation_name = annotation_filename.split(".")[0]
    annotation_filepath = os.path.join(ANNOTATION_DIR, annotation_filename)

    batch_size = 16
    epoch = 1
    
    logger.info(f"Device List: {device_lib.list_local_devices()}")

    # for each classifier
    for model_type in ['efficientnet', 'efficientnetb0','vgg16', 'mobilenetv3', 'custom']:
    #for model_type in ['efficientnetb0']:
        classifier = CoralReefClassifier(ROOT_DIR, DATA_DIR, IMAGE_DIR, annotation_filepath, model_type)
        classifier.create_model()
        logger.info(f"Start model ({model_type}) training...")
        classifier.train(batch_size=batch_size, epochs=epoch)

        logger.info(f"Training model ({model_type}) DONE!")
        model_file = os.path.join(
            MODEL_DIR, 
            f'coral_reef_classifier_{model_type}_full_epoch_{epoch}_1_batchsize_{batch_size}_{annotation_name}.h5'
        )
        classifier.save_model(model_file)

        logger.info(f"{model_file} SAVED!")

        logger.info("Evaluating the model now...")
        # Get metrics
        metrics = classifier.get_evaluation_metrics(batch_size=batch_size)

        # Make sure already run model.train to get this attribute
        metrics["traning_time_in_seconds"] = classifier.training_time

        metrics["annotation_filepath"] = classifier.annotation_file

        # Save metrics to a JSON file
        metrics_file = os.path.join(
            MODEL_DIR,
            f'coral_reef_classifier_{model_type}_epoch_{epoch}_1_batchsize_{batch_size}_metrics_{annotation_name}.json'
        )
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation metrics saved: {metrics_file}")

