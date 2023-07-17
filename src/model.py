import os
import numpy as np
import csv
import cv2
import json 
import time 
import logging
import re 


from keras.models import Model, load_model
from keras.callbacks import CSVLogger
from keras.metrics import Accuracy, Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, concatenate, Input, MaxPooling2D
from keras.utils import to_categorical
from keras.applications import EfficientNetB0, VGG16, MobileNetV3Large, EfficientNetV2B0, ConvNeXtTiny
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter
from datetime import datetime

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR, WEIGHT_DIR, MODEL_DIR

logger = logging.getLogger(__name__)

class CoralReefClassifier:
    def __init__(self, root_dir, data_dir, image_dir, annotation_file, model_type, image_scale=0.2):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.annotation_filename = os.path.basename(self.annotation_file)
        self.model_type = model_type
        self.image_paths = []
        self.image_scale = image_scale  # Scale used to crop image
        self.labels = []
        self.x_pos = []
        self.y_pos = []
        self.model = None

        self.efficientnet_b0_weight = os.path.join(WEIGHT_DIR, "efficientnetb0_notop.h5")
        self.efficientnet_v2_b0_weight = os.path.join(WEIGHT_DIR, "efficientnetv2-b0_notop.h5")
        self.vgg16_weight = os.path.join(WEIGHT_DIR, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.mobilenet_v3_weight = os.path.join(WEIGHT_DIR, "weights_mobilenet_v3_large_224_1.0_float.h5")
        self.convnext_tiny_weight = os.path.join(WEIGHT_DIR, "convnext_tiny_notop.h5")
        
        self.load_data()

    def load_data(self):
        with open(self.annotation_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                image_name = row[0]
                label = row[3]  
                x = float(row[2])  
                y = float(row[1])
                image_path = os.path.join(self.image_dir, image_name)

                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.labels.append(label)
                    self.x_pos.append(x)
                    self.y_pos.append(y)

        # Convert labels to integers
        unique_labels = np.unique(self.labels)
        self.n_unique_labels = len(unique_labels)  # Store the number of unique labels here
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])

        # Filter out classes with only one sample
        # line 73, in load_data
        # label_counts = np.bincount(self.labels)
        # File "<__array_function__ internals>", line 180, in bincount
        # TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'
        label_counts = np.bincount(self.labels.astype('int64'))
        single_sample_labels = np.where(label_counts == 1)[0]
        multi_sample_labels_count = len(unique_labels) - len(single_sample_labels)  # Number of labels with more than one sample
        
        if list(single_sample_labels) != []:
            logger.warning(f"Skipping label {single_sample_labels} since it only contain one sample")
            
        self.label_skipped_count = len(single_sample_labels)
        indices_to_keep = [i for i, label in enumerate(self.labels) if label not in single_sample_labels]
        self.number_labels_to_train = multi_sample_labels_count
        
        self.image_paths = [self.image_paths[i] for i in indices_to_keep]
        self.unique_image_count = len(set(self.image_paths))
        self.labels = [self.labels[i] for i in indices_to_keep]
        self.x_pos = [self.x_pos[i] for i in indices_to_keep]
        self.y_pos = [self.y_pos[i] for i in indices_to_keep]

        # Convert labels to categorical
        self.labels = to_categorical(self.labels)
        self.x_pos = np.array(self.x_pos)
        self.y_pos = np.array(self.y_pos)

        # Split the data into training and validation sets
        (
            self.image_paths_train, self.image_paths_val, 
            self.labels_train, self.labels_val, 
            self.x_pos_train, self.x_pos_val, 
            self.y_pos_train, self.y_pos_val
        ) = train_test_split(
                self.image_paths, self.labels, self.x_pos, self.y_pos, 
                test_size=0.2, stratify=self.labels, random_state=42
        )

        logger.info(f"Annotation file: {self.annotation_file}")
        logger.info(f"Loaded {self.unique_image_count} images with {len(self.image_paths)} annotations and {self.number_labels_to_train} labels\n")


    def data_generator(self, image_paths, labels, x_pos, y_pos, batch_size):
        while True:
            for i in range(0, len(image_paths), batch_size):
                batch_image_paths = image_paths[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                batch_x_pos = x_pos[i:i+batch_size]
                batch_y_pos = y_pos[i:i+batch_size]
                
                batch_images = []
                for idx, image_path in enumerate(batch_image_paths):
                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        if image is None:
                            logger.error(f'Error processing image {image_path}: could not be read by cv2.imread')
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Extract center of the bounding box
                        x_center = int(batch_x_pos[idx])
                        y_center = int(batch_y_pos[idx])
                        
                        # Create segment from image
                        height, width, _ = image.shape
                        half_width = int(width * self.image_scale) // 2
                        half_height = int(height * self.image_scale) // 2

                        x_min = max(0, x_center - half_width)
                        y_min = max(0, y_center - half_height)
                        x_max = min(width, x_center + half_width)
                        y_max = min(height, y_center + half_height)
                        
                        image = image[y_min:y_max, x_min:x_max]

                        image = cv2.resize(image, (224, 224))  # Make sure all images are resized to (224, 224)
                        image = np.array(image)
                        batch_images.append(image)
                    except Exception as e:
                        logger.error(f'Error processing image {image_path}: {e}')
                        continue

                if len(batch_images) == 0:  # If the batch_images list is empty, logger.info the problematic image paths and skip to the next iteration
                    logger.warning(f'Skipping a batch at index {i}. All image paths in this batch were problematic: {batch_image_paths}')
                    continue
                
                batch_images = np.array(batch_images, dtype=np.float32) / 255.0
                yield batch_images, batch_labels


    def create_model(self):
        image_input = Input(shape=(224, 224, 3))

        if self.model_type == "efficientnet":
            base_model = EfficientNetB0(weights=self.efficientnet_b0_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
        elif self.model_type == "efficientnetv2":
            base_model = EfficientNetV2B0(weights=self.efficientnet_v2_b0_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
        elif self.model_type == "vgg16":
            base_model = VGG16(weights=self.vgg16_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
        elif self.model_type == "mobilenetv3":
            base_model = MobileNetV3Large(weights=self.mobilenet_v3_weight, include_top=True)
            x = base_model(image_input)
        elif self.model_type == "convnexttiny":
            base_model = ConvNeXtTiny(weights=self.convnext_tiny_weight, include_top=False)  # assuming this is the correct class name
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
        elif self.model_type == "custom":
            x = Conv2D(16, (3, 3), activation='relu')(image_input)
            x = MaxPooling2D()(x)
            x = Conv2D(32, (3, 3), activation='relu')(x)
            x = GlobalAveragePooling2D()(x)
        else:
            raise ValueError('Invalid model type')

        # Use the stored number of unique labels here
        output = Dense(self.n_unique_labels, activation='softmax')(x)
        self.model = Model(inputs=[image_input], outputs=output)

        self.model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=[
                Accuracy(), Precision(), Recall(), AUC(), 
                TruePositives(), TrueNegatives(), FalsePositives(), 
                FalseNegatives()
            ]
        )



    def train(self, batch_size, epochs):
        if self.model is None:
            logger.info("No model defined.")
            return

        logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}\n")
        steps_per_epoch = len(self.image_paths_train) // batch_size
        validation_steps = len(self.image_paths_val) // batch_size
        
        csv_logger_filename = f"coral_reef_classifier_{self.model_type}_epoch_{epochs}_batchsize_{batch_size}_metrics_{self.annotation_filename}.csv"
        csv_logger_filepath = os.path.join(MODEL_DIR, csv_logger_filename)
        csv_loggger = CSVLogger(csv_logger_filepath)
        logger.info(f"CSVLogger file will be generated at {csv_logger_filepath}")
        self.model.summary(print_fn=logger.info)
        self.start_time = time.time() 
        start_time = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Start time for {self.model_type} model training: {start_time}")
        model_history = self.model.fit(
            self.data_generator(self.image_paths_train, self.labels_train, self.x_pos_train, self.y_pos_train, batch_size), 
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            validation_data=self.data_generator(self.image_paths_val, self.labels_val, self.x_pos_val, self.y_pos_val, batch_size),
            validation_steps=validation_steps,
            callbacks=[csv_loggger]
        )
        self.end_time = time.time() 
        end_time = datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Finish training at : {end_time}")
        self.training_time = self.end_time - self.start_time  # Compute the training time
        logger.info(f'Training time: {self.training_time} seconds')
        
        training_metrics = model_history.history
        training_metrics = self.convert_to_key_value_pairs(training_metrics)
        return training_metrics


    def save_model(self, model_file):
        if self.model is None:
            logger.info("No model to save.")
            return

        self.model.save(model_file)


    def get_evaluation_metrics(self, batch_size):
        if self.model is None:
            logger.info("No model defined.")
            return {}

        steps = len(self.image_paths_val) // batch_size
        val_data_generator = self.data_generator(
            image_paths=self.image_paths_val,
            labels=self.labels_val,
            x_pos=self.x_pos_val,
            y_pos=self.y_pos_val,
            batch_size=batch_size
        )

        metrics = self.model.evaluate(val_data_generator, steps=steps, verbose=0)
        metrics_dict = {name: value for name, value in zip(self.model.metrics_names, metrics)}
        metrics_dict = self.normalize_metric_names(metrics_dict)
        return metrics_dict
    
    
    def normalize_metric_names(self, metrics):
        """
        Normalize the metric names in the model results.

        Given a dictionary of metrics, this function will normalize the metric
        names by removing trailing digit identifiers.

        For example, for the input:
        {"precision_2": 0.7, "recall_2": 0.8}
        
        the output will be:
        {"precision": 0.6, "recall": 0.9}

        Parameters:
        metrics (dict): The dictionary of metrics.

        Returns:
        dict: The dictionary of metrics with normalized names.
        """
        new_metrics = {}
        for key in list(metrics.keys()):
            # Regular expression to match any key ending with _<number>
            if re.match(".*_\d+$", key):
                new_key = re.sub("_\d+$", "", key)
                # Add the normalized metric to the dictionary
                new_metrics[new_key] = metrics[key]
            else:
                # Handle keys without underscores here, if necessary
                new_metrics[key] = metrics[key]
        return new_metrics
    
    
    def convert_to_key_value_pairs(self, data):
        """
        Converts a dictionary of key-value pairs where values may be lists or nested dictionaries
        into a dictionary of key-value pairs where each value is a single element.
        If a value is a list, the last element of the list is chosen as the new value.
        If a value is a dictionary, the function is called recursively on that dictionary.

        Parameters:
        data (dict): The input dictionary to be converted.

        Returns:
        dict: The converted dictionary with single-element values.
        """
     
        key_value_dict = {}
        for key, values in data.items():
            # If value is list then take the last element
            if isinstance(values, list):
                key_value_dict[key] = values[-1]
            # If value is a dictionary, then recursively call the function
            elif isinstance(values, dict):
                key_value_dict[key] = self.convert_to_key_value_pairs(values)
            else:
                key_value_dict[key] = values
        return key_value_dict
    
    
    def load_trained_model(self, model_file_path):
        """
        Loads a previously trained model from disk.
        
        Parameters:
        model_file_path (str): The path to the model file.
        """
        self.model = load_model(model_file_path)

        # Recompile the model to make sure the metrics are properly loaded
        self.model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=[
                Accuracy(), Precision(), Recall(), AUC(), 
                TruePositives(), TrueNegatives(), FalsePositives(), 
                FalseNegatives()
            ]
        )