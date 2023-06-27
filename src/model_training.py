import os
import numpy as np
import csv
import cv2
import json 
import time 
import logging

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR, WEIGHT_DIR, MODEL_DIR
from src.model import CoralReefClassifier
from src.utils import logging_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    #annotation_file = os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k.csv")
    annotation_file = os.path.join(ANNOTATION_DIR, "combined_annotations_1000.csv")

    batch_size = 16
    epoch = 1
   
    # for each classifier
    for model_type in ['efficientnet', 'vgg16', 'mobilenetv3', 'custom']:
    #for model_type in ['mobilenetv3']:
        classifier = CoralReefClassifier(ROOT_DIR, DATA_DIR, IMAGE_DIR, annotation_file, model_type)
        classifier.create_model()
        logger.info(f"Start model({model_type}) training...")
        classifier.train(batch_size=batch_size, epochs=epoch)

        logger.info(f"Training model({model_type}) DONE!")
        model_file = os.path.join(MODEL_DIR, f'coral_reef_classifier_{model_type}_full_epoch_{epoch}_1_batchsize_{batch_size}.h5')
        classifier.save_model(model_file)

        logger.info(f"{model_file} SAVED!")

        logger.info("Evaluating the model now...")
        # Get metrics
        metrics = classifier.get_evaluation_metrics(batch_size=batch_size)

        # Make sure already run model.train to get this attribute
        metrics["traning_time_in_seconds"] = classifier.training_time

        metrics["annotation_file"] = classifier.annotation_file 

        # Save metrics to a JSON file
        metrics_file = os.path.join(MODEL_DIR,f'coral_reef_classifier_{model_type}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation metrics saved: {metrics_file}")
