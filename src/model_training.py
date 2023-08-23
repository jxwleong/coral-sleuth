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

def train_and_evaluate_models(
    annotation_filepath,
    annotation_name,
    annotation_filename,
    batch_size,
    epoch,
    model_types,
    image_scale=0.2,
    stopping_patience=5,
    use_augmentation=True
):

    logger.info(f"Device List: {device_lib.list_local_devices()}")

    metrics = {}
    
    for model_type in model_types:

        classifier = CoralReefClassifier(ROOT_DIR, DATA_DIR, IMAGE_DIR, annotation_filepath, model_type, image_scale, stopping_patience=stopping_patience, use_augmentation=use_augmentation)
        classifier.create_model()
        logger.info(f"Start model ({model_type}) training...")
        training_metrics = classifier.train(batch_size=batch_size, epochs=epoch)

        logger.info(f"Training model ({model_type}) DONE!")
        model_file = os.path.join(
            MODEL_DIR, 
            f'coral_reef_classifier_{model_type}_full_epoch_{epoch}_1_batchsize_{batch_size}_{annotation_name}_scale_{image_scale}.h5'
        )
        classifier.save_model(model_file)

        logger.info(f"{model_file} SAVED!")

        logger.info("Evaluating the model now...")

        # Get model metrics
        model_metrics = {}
        
        metrics[f"{model_type}"] = classifier.normalize_metric_names(training_metrics)
        model_metrics["traning_time_in_seconds"] = classifier.training_time

        model_metrics["batch_size"] = batch_size
        model_metrics["epoch"] = epoch

        model_metrics["annotation_file"] = annotation_filename
        model_metrics["images_count"] = classifier.unique_image_count
        model_metrics["annotation_count"] = len(classifier.image_paths)
        model_metrics["annotation_label_count"] = classifier.number_labels_to_train
        model_metrics["annotation_label_skipped_count"] = classifier.label_skipped_count
        
        metrics[f"{model_type}"].update(model_metrics)
        
        # Save metrics to a JSON file
        metrics_file = os.path.join(
            MODEL_DIR,
            f'coral_reef_classifier_{model_type}_epoch_{epoch}_batchsize_{batch_size}_metrics_{annotation_name}_scale_{image_scale}.json'
        )
        
        logger.info("\n" + json.dumps(metrics, indent=4))
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation metrics saved: {metrics_file}")
        
    excel_file = os.path.join(
        MODEL_DIR,
        f'coral_reef_classifier_epoch_{epoch}_batchsize_{batch_size}_metrics_{annotation_name}_scale_{image_scale}.xlsx'
    )
    excel.dict_to_excel(metrics, excel_file, "model_name")
    excel.append_label_distribution_to_excel(annotation_filepath, excel_file)
    logger.info(f"Evaluation metrics in excel format saved: {excel_file}")


def continue_training_models(
    h5_model_file,
    annotation_filepath,
    annotation_name,
    annotation_filename,
    batch_size,
    additional_epochs,
    image_scale=0.2
):

    logger.info(f"Device List: {device_lib.list_local_devices()}")

    metrics = {}

    classifier = CoralReefClassifier(ROOT_DIR, DATA_DIR, IMAGE_DIR, annotation_filepath, "pretrained")

    logger.info(f"Loading model from {h5_model_file}...")
    classifier.load_trained_model(h5_model_file)

    logger.info(f"Continuing model training...")
    training_metrics = classifier.train(batch_size=batch_size, epochs=additional_epochs)

    logger.info(f"Training model DONE!")
    model_file = os.path.join(
        MODEL_DIR, 
        f'coral_reef_classifier_continued_epoch_{additional_epochs}_batchsize_{batch_size}_{annotation_name}_scale_{image_scale}.h5'
    )
    classifier.save_model(model_file)

    logger.info(f"{model_file} SAVED!")

    logger.info("Evaluating the model now...")

    model_metrics = {}

    model_metrics = classifier.normalize_metric_names(training_metrics)
    model_metrics["traning_time_in_seconds"] = classifier.training_time

    model_metrics["batch_size"] = batch_size
    model_metrics["epoch"] = additional_epochs

    model_metrics["annotation_file"] = annotation_filename
    model_metrics["images_count"] = classifier.unique_image_count
    model_metrics["annotation_count"] = len(classifier.image_paths)
    model_metrics["annotation_label_count"] = classifier.number_labels_to_train
    model_metrics["annotation_label_skipped_count"] = classifier.label_skipped_count

    metrics.update(model_metrics)

    metrics_file = os.path.join(
        MODEL_DIR,
        f'coral_reef_classifier_continued_epoch_{additional_epochs}_batchsize_{batch_size}_metrics_{annotation_name}_scale_{image_scale}.json'
    )

    logger.info("\n" + json.dumps(metrics, indent=4))
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation metrics saved: {metrics_file}")

    excel_file = os.path.join(
        MODEL_DIR,
        f'coral_reef_classifier_continued_epoch_{additional_epochs}_batchsize_{batch_size}_metrics_{annotation_name}_scale_{image_scale}.xlsx'
    )
    excel.dict_to_excel(metrics, excel_file, "model_name")
    excel.append_label_distribution_to_excel(annotation_filepath, excel_file)
    logger.info(f"Evaluation metrics in excel format saved: {excel_file}")



if __name__ == "__main__":
    annotation_filename = "combined_annotations_remapped_merged_undersample_oversample_5k.csv"
    annotation_name = annotation_filename.split(".")[0]
    annotation_filepath = os.path.join(ANNOTATION_DIR, annotation_filename)
    
    h5_model_filename = 'coral_reef_classifier_efficientnetv2_full_epoch_50_1_batchsize_32_combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample_0p1.h5'
    h5_model_file = os.path.join(
            MODEL_DIR, 
            h5_model_filename
    )
   

    batch_size = 32
    epoch = 50
    additional_epochs = 50
    
    for scale in [0.2]:
        train_and_evaluate_models(
            annotation_filepath,
            annotation_name,
            annotation_filename,
            batch_size,
            epoch,
            model_types=['efficientnetv2'], #'mobilenetv3'], #, 'convnexttiny'],
            #model_types=['efficientnetv2'],
            image_scale=scale,
            stopping_patience=50,
            use_augmentation=False
        )
    
    """
    
    continue_training_models(
        h5_model_file,
        annotation_filepath,
        annotation_name,
        annotation_filename,
        batch_size,
        additional_epochs,
    )
 """