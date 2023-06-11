import os
import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model

from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

annotation_file = os.path.join(DATA_DIR, "coralnet_source_2091_annotations.csv")

# Load the CSV file
images = []
labels = []
with open(annotation_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)

    # Assuming the label column is the one named 'Label'
    label_idx = headers.index('Label')

    for row in reader:
        image_name = row[0]
        image_path = os.path.join(IMAGE_DIR, image_name + ".jpg")
        if os.path.exists(image_path) is True:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            images.append(image)
            labels.append(row[label_idx]) # Append the corresponding label

""" 
print("Number of images:", len(images))

# Split the data into training and testing sets
train_images = images[:int(len(images) * 0.8)]
test_images = images[int(len(images) * 0.8):]

# Extract the labels for training and testing sets
train_labels = labels[:int(len(labels) * 0.8)]
test_labels = labels[int(len(labels) * 0.8):]

# Convert the images and labels to NumPy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)

# Integer encoding of labels
unique_labels = np.unique(labels)
label_mapping = {label: i for i, label in enumerate(unique_labels)}

num_classes = len(unique_labels) # save the number of unique labels

# Convert the labels to integer-encoded vectors
train_labels = np.array([label_mapping[label] for label in train_labels])
test_labels = np.array([label_mapping[label] for label in test_labels])

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


print('Number of unique classes:', num_classes)
print('Unique labels:', unique_labels)

unique_train_labels = np.unique([np.argmax(label) for label in train_labels])
unique_test_labels = np.unique([np.argmax(label) for label in test_labels])
print('Unique train labels:', unique_train_labels)
print('Unique test labels:', unique_test_labels)

"""

# Create a Counter object for your labels
label_counts = Counter(labels)

# Print out the counts of each label
for label, count in label_counts.items():
    print(f'Label: {label}, Count: {count}')


# Convert the images to NumPy arrays
images = np.array(images)

# Integer encoding of labels
unique_labels = np.unique(labels)
label_mapping = {label: i for i, label in enumerate(unique_labels)}

# Convert the labels to integer-encoded vectors
labels = np.array([label_mapping[label] for label in labels])

# Initialize the stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_index, test_index in sss.split(images, labels):
    train_images, test_images = images[train_index], images[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]

num_classes = len(unique_labels) # save the number of unique labels

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=num_classes)
test_labels = to_categorical(test_labels, num_classes=num_classes)

print('Number of unique classes:', num_classes)
print('Unique labels:', unique_labels)

unique_train_labels = np.unique([np.argmax(label) for label in train_labels])
unique_test_labels = np.unique([np.argmax(label) for label in test_labels])
print('Unique train labels:', unique_train_labels)
print('Unique test labels:', unique_test_labels)


# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Train labels shape:', train_labels.shape)
print('Test labels shape:', test_labels.shape)
print('Model output shape:', model.output_shape)

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model
score = model.evaluate(test_images, test_labels)
print('Test accuracy:', score[1])

# Save the model
model_file = 'coral_reef_classifier.h5'
model.save(model_file)
