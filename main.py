import os
import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.callbacks import ModelCheckpoint



ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

annotation_file = os.path.join(DATA_DIR, "coralnet_source_2091_annotations.csv")
# Load the CSV file
with open(annotation_file, 'r') as csvfile:
    reader = csv.reader(csvfile)

    # Get the labels
    labels = next(reader)

    # Get the images
    images = []
    for row in reader:
        image_name = row[0]
        image_path = os.path.join(IMAGE_DIR, image_name + ".jpg")
        if os.path.exists(image_path) is True:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            images.append(image)

# Print the labels and number of images
print("Labels:", labels)
print("Number of images:", len(images))

# Split the data into training and testing sets
train_images = images[:int(len(images) * 0.8)]
test_images = images[int(len(images) * 0.8):]

# Extract the labels for training and testing sets
labels_train = data['Label'][:len(train_images)]
labels_test = data['Label'][len(train_images):]

# Convert the images to NumPy arrays
train_images = np.array(train_images)
test_images = np.array(test_images)

# Integer encoding of labels
label_mapping = {label: i for i, label in enumerate(labels)}

# Convert the labels to integer-encoded vectors
train_labels = np.array([label_mapping[label] for label in train_images])
test_labels = np.array([label_mapping[label] for label in test_images])

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model
score = model.evaluate(test_images, test_labels)
print('Test accuracy:', score[1])

# Save the model
model_file = 'coral_reef_classifier.h5'
model.save(model_file)

# Load the model
model = load_model(model_file)

# Classify an image
image = cv2.imread('image.png')
image = cv2.resize(image, (224, 224))
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)

# Print the prediction
print('Predicted label:', labels[np.argmax(prediction)])
