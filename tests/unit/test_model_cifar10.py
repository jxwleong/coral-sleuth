import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2B0, VGG16, MobileNetV3Large
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import keras

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR, WEIGHT_DIR, MODEL_DIR
from config.proxy import proxies

efficientnet_b0_weight = os.path.join(WEIGHT_DIR, "efficientnetb0_notop.h5")
efficientnet_v2_b0_weight = os.path.join(WEIGHT_DIR, "efficientnetv2-b0_notop.h5")
vgg16_weight = os.path.join(WEIGHT_DIR, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
mobilenet_v3_weight = os.path.join(WEIGHT_DIR, "weights_mobilenet_v3_large_224_1.0_float.h5")
convnext_tiny_weight = os.path.join(WEIGHT_DIR, "convnext_tiny_notop.h5")
        
try:
    # Set proxy
    os.environ['HTTP_PROXY'] = proxies["http"]
    os.environ['HTTPS_PROXY'] = proxies["https"]
except:
    # Assuming we dont have the proxies dict then the system don't need it
    pass

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalise the images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to categorical
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Create a data generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(train_images)

# Define the models
def get_model(model_type):
    image_input = Input(shape=(32, 32, 3))

    if model_type == "efficientnetv2":
        base_model = EfficientNetV2B0(weights=efficientnet_v2_b0_weight, include_top=False)
        x = base_model(image_input)
        x = GlobalAveragePooling2D()(x)
    elif model_type == "vgg16":
        base_model = VGG16(weights=vgg16_weight, include_top=False)
        x = base_model(image_input)
        x = GlobalAveragePooling2D()(x)
    elif model_type == "mobilenetv3":
        base_model = MobileNetV3Large(weights=mobilenet_v3_weight, include_top=False)
        x = base_model(image_input)
        x = GlobalAveragePooling2D()(x)
    # Add other models as needed
    else:
        raise ValueError('Invalid model type')

    # Common classifier for all models
    x = Dense(256, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the full model
    model = Model(inputs=image_input, outputs=predictions)

    return model

# Training loop
for model_type in ["efficientnetv2", "vgg16", "mobilenetv3"]:  # add other models to the list
    print(f"Training {model_type} model...")
    model = get_model(model_type)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(train_images, train_labels, batch_size=64), epochs=10, validation_data=(test_images, test_labels))
