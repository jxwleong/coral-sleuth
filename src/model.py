import os
import numpy as np
import csv
import cv2
import json 
import time 
import logging

from keras.models import Model
from keras.metrics import Accuracy, Precision, Recall, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, concatenate, Input, MaxPooling2D
from keras.utils import to_categorical
from keras.applications import EfficientNetB0, VGG16, MobileNetV3Large, EfficientNetV2B0
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import Counter

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR, WEIGHT_DIR, MODEL_DIR

logger = logging.getLogger(__name__)

class CoralReefClassifier:
    def __init__(self, root_dir, data_dir, image_dir, annotation_file, model_type):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.model_type = model_type
        self.image_paths = []
        self.labels = []
        self.x_pos = []
        self.y_pos = []
        self.model = None

        self.efficientnet_b0_weight = os.path.join(WEIGHT_DIR, "efficientnetb0_notop.h5")
        self.efficientnet_v2_b0_weight = os.path.join(WEIGHT_DIR, "efficientnetv2-b0_notop.h5")
        self.vgg16_weight = os.path.join(WEIGHT_DIR, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
        self.mobilenet_v3_weight = os.path.join(WEIGHT_DIR, "weights_mobilenet_v3_large_224_1.0_float.h5")

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
        label_counts = np.bincount(self.labels)
        single_sample_labels = np.where(label_counts == 1)[0]
        indices_to_keep = [i for i, label in enumerate(self.labels) if label not in single_sample_labels]

        self.image_paths = [self.image_paths[i] for i in indices_to_keep]
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



    def data_generator(self, image_paths, labels, x_pos, y_pos, batch_size):
        while True:
            for i in range(0, len(image_paths), batch_size):
                batch_image_paths = image_paths[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                batch_x_pos = x_pos[i:i+batch_size]
                batch_y_pos = y_pos[i:i+batch_size]
                
                batch_images = []
                for image_path in batch_image_paths:
                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        if image is None:
                            logger.error(f'Error processing image {image_path}: could not be read by cv2.imread')
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                yield [batch_images, np.column_stack((batch_x_pos, batch_y_pos))], batch_labels


    def create_model(self):
        image_input = Input(shape=(224, 224, 3))
        pos_input = Input(shape=(2,))

        y = Dense(16, activation='relu')(pos_input)

        if self.model_type == "efficientnet":
            base_model = EfficientNetB0(weights=self.efficientnet_b0_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
            y = Dense(1280, activation='relu')(pos_input)
        elif self.model_type == "efficientnetb0":
            base_model = EfficientNetV2B0(weights=self.efficientnet_v2_b0_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
            y = Dense(1280, activation='relu')(pos_input)
        elif self.model_type == "vgg16":
            base_model = VGG16(weights=self.vgg16_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
            y = Dense(512, activation='relu')(pos_input)
        elif self.model_type == "mobilenetv3":
            base_model = MobileNetV3Large(weights=self.mobilenet_v3_weight, include_top=True)
            x = base_model(image_input)
            y = Dense(1024, activation='relu')(pos_input)  # MobileNetV3Large has 1024 output features
        elif self.model_type == "custom":
            x = Conv2D(16, (3, 3), activation='relu')(image_input)
            x = MaxPooling2D()(x)
            x = Conv2D(32, (3, 3), activation='relu')(x)
            x = GlobalAveragePooling2D()(x)
            y = Dense(32, activation='relu')(pos_input)
        else:
            raise ValueError('Invalid model type')

        combined = concatenate([x, y])

        # Use the stored number of unique labels here
        output = Dense(self.n_unique_labels, activation='softmax')(combined)
        self.model = Model(inputs=[image_input, pos_input], outputs=output)


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

        steps_per_epoch = len(self.image_paths_train) // batch_size
        validation_steps = len(self.image_paths_val) // batch_size
        self.model.summary()
        self.start_time = time.time() 
        self.model.fit(
            self.data_generator(self.image_paths_train, self.labels_train, self.x_pos_train, self.y_pos_train, batch_size), 
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            validation_data=self.data_generator(self.image_paths_val, self.labels_val, self.x_pos_val, self.y_pos_val, batch_size),
            validation_steps=validation_steps
        )
        self.end_time = time.time() 

        self.training_time = self.end_time - self.start_time  # Compute the training time
        logger.info(f'Training time: {self.training_time} seconds')


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
        return metrics_dict