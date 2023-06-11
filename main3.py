import os
import cv2
import numpy as np
import csv
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
print(ROOT_DIR)

annotation_file = os.path.join(DATA_DIR, "coralnet_source_2091_annotations.csv")

# initialize the lists to hold the images and labels
images = []
labels = []
x_pos = []
y_pos = []

with open(annotation_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        image_name = row[0]
        label = row[20]  
        x = float(row[19])  
        y = float(row[18])
        image_path = os.path.join(IMAGE_DIR, image_name + ".jpg")

        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(label)
            x_pos.append(x)
            y_pos.append(y)

images = np.array(images) / 255.0
labels = np.array(labels)
x_pos = np.array(x_pos)
y_pos = np.array(y_pos)

unique_labels = np.unique(labels)
label_mapping = {label: i for i, label in enumerate(unique_labels)}
labels = np.array([label_mapping[label] for label in labels])

labels = to_categorical(labels)

split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
test_images = images[split_idx:]
train_labels = labels[:split_idx]
test_labels = labels[split_idx:]
train_x_pos = x_pos[:split_idx]
test_x_pos = x_pos[split_idx:]
train_y_pos = y_pos[:split_idx]
test_y_pos = y_pos[split_idx:]

# Create the model
image_input = Input(shape=(224, 224, 3))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = Flatten()(x)

pos_input = Input(shape=(2,))
y = Dense(32, activation='relu')(pos_input)

combined = concatenate([x, y])
z = Dense(128, activation='relu')(combined)
output = Dense(len(unique_labels), activation='softmax')(z)

model = Model(inputs=[image_input, pos_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([train_images, np.column_stack((train_x_pos, train_y_pos))], train_labels, epochs=100, batch_size=32)

# Evaluate the model
score = model.evaluate([test_images, np.column_stack((test_x_pos, test_y_pos))], test_labels)
print('Test accuracy:', score[1])

# Save the model
model_file = 'coral_reef_classifier_100epoch.h5'
model.save(model_file)
