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
from src.utils import logging_config, excel

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    annotation_filename = "combined_annotations_about_40k_png_only_remapped.csv"
    annotation_name = annotation_filename.split(".")[0]
    annotation_filepath = os.path.join(ANNOTATION_DIR, annotation_filename)

    batch_size = 16
    epoch = 2
    
    logger.info(f"Device List: {device_lib.list_local_devices()}")

    metrics = {}
    # for each classifier
    for model_type in ['efficientnet', 'efficientnetv2','vgg16', 'mobilenetv3', 'custom']:
    #for model_type in ['efficientnet', 'efficientnetv2']:
    #for model_type in ['efficientnetv2']:
        classifier = CoralReefClassifier(ROOT_DIR, DATA_DIR, IMAGE_DIR, annotation_filepath, model_type)
        classifier.create_model()
        logger.info(f"Start model ({model_type}) training...")
        training_metrics = classifier.train(batch_size=batch_size, epochs=epoch)

        logger.info(f"Training model ({model_type}) DONE!")
        model_file = os.path.join(
            MODEL_DIR, 
            f'coral_reef_classifier_{model_type}_full_epoch_{epoch}_1_batchsize_{batch_size}_{annotation_name}.h5'
        )
        classifier.save_model(model_file)

        logger.info(f"{model_file} SAVED!")

        logger.info("Evaluating the model now...")

        # Get model metrics
        #model_metrics = classifier.get_evaluation_metrics(batch_size=batch_size)
        model_metrics = {}
        
        # training_metrics will contain the training metrics and val metrics as well
        # as it is history object
        metrics[f"{model_type}"] = classifier.normalize_metric_names(training_metrics)
        # Make sure already run model.train to get this attribute
        model_metrics["traning_time_in_seconds"] = classifier.training_time

        model_metrics["batch_size"] = batch_size
        model_metrics["epoch"] = epoch

        # Added some information regarding the dataset/ annotation files
        model_metrics["annotation_file"] = annotation_filename
        model_metrics["images_count"] = classifier.unique_image_count
        model_metrics["annotation_count"] = len(classifier.image_paths)
        model_metrics["annotation_label_count"] = classifier.number_labels_to_train
        model_metrics["annotation_label_skipped_count"] = classifier.label_skipped_count
        
        metrics[f"{model_type}"].update(model_metrics)
        
        # Save metrics to a JSON file
        metrics_file = os.path.join(
            MODEL_DIR,
            f'coral_reef_classifier_epoch_{epoch}_1_batchsize_{batch_size}_metrics_{annotation_name}.json'
        )
        
        logger.info("\n" + json.dumps(metrics, indent=4))
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation metrics saved: {metrics_file}")
        
    excel_file = os.path.join(
        MODEL_DIR,
        f'coral_reef_classifier_epoch_{epoch}_1_batchsize_{batch_size}_metrics_{annotation_name}.xlsx'
    )
    excel.dict_to_excel(metrics, excel_file, "model_name")
    excel.append_label_distribution_to_excel(annotation_filepath, excel_file)
    logger.info(f"Evaluation metrics in excel format saved: {excel_file}")

