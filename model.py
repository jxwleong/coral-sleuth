import os
import numpy as np
import csv
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, concatenate
from keras.utils import to_categorical
from PIL import Image

class CoralReefClassifier:

    def __init__(self, root_dir, data_dir, image_dir, annotation_file):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.image_paths = []
        self.labels = []
        self.x_pos = []
        self.y_pos = []

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

        unique_labels = np.unique(self.labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])
        self.labels = to_categorical(self.labels)
        self.x_pos = np.array(self.x_pos)
        self.y_pos = np.array(self.y_pos)

    def data_generator(self, batch_size):
        while True:
            for i in range(0, len(self.image_paths), batch_size):
                batch_image_paths = self.image_paths[i:i+batch_size]
                batch_labels = self.labels[i:i+batch_size]
                batch_x_pos = self.x_pos[i:i+batch_size]
                batch_y_pos = self.y_pos[i:i+batch_size]
                
                batch_images = []
                for image_path in batch_image_paths:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize((224, 224))
                        image = np.array(image)
                        batch_images.append(image)
                    except Exception as e:
                        print(f'Error processing image {image_path}: {e}')
                        continue
                    
                batch_images = np.array(batch_images, dtype=np.float32) / 255.0
                yield [batch_images, np.column_stack((batch_x_pos, batch_y_pos))], batch_labels

    def create_model(self):
        image_input = Input(shape=(224, 224, 3))
        x = Conv2D(16, (3, 3), activation='relu')(image_input)
        x = Flatten()(x)

        pos_input = Input(shape=(2,))
        y = Dense(16, activation='relu')(pos_input)

        combined = concatenate([x, y])
        output = Dense(len(np.unique(self.labels)), activation='softmax')(combined)

        model = Model(inputs=[image_input, pos_input], outputs=output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, model, batch_size, epochs):
        steps_per_epoch = len(self.image_paths) // batch_size
        model.fit(self.data_generator(batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs)
        return model

    def save_model(self, model, model_file):
        model.save(model_file)


root_dir = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
data_dir = os.path.join(root_dir, "data")
image_dir = os.path.join(root_dir, "images")
annotation_file = os.path.join(data_dir, "combined_annotations.csv")
model_file = 'coral_reef_classifier_1_epoch.h5'

coral_classifier = CoralReefClassifier(root_dir, data_dir, image_dir, annotation_file)
coral_classifier.load_data()
model = coral_classifier.create_model()
model = coral_classifier.train_model(model, batch_size=16, epochs=1)
coral_classifier.save_model(model, model_file)
