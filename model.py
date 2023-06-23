import os
import numpy as np
import csv
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Flatten, concatenate, Input, MaxPooling2D
from keras.utils import to_categorical
from keras.applications import EfficientNetB0, VGG16, ResNet50
from PIL import Image

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

        self.efficientnet_b0_weight = os.path.join(self.data_dir, "efficientnetb0_notop.h5")

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

        unique_labels = np.unique(self.labels)
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        self.labels = np.array([label_mapping[label] for label in self.labels])

        self.labels = to_categorical(self.labels)
        self.x_pos = np.array(self.x_pos)
        self.y_pos = np.array(self.y_pos)

        self.n_unique_labels = len(unique_labels)

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
        pos_input = Input(shape=(2,))

        y = Dense(16, activation='relu')(pos_input)

        if self.model_type == "efficientnet":
            base_model = EfficientNetB0(weights=self.efficientnet_b0_weight, include_top=False)
            x = base_model(image_input)
            x = GlobalAveragePooling2D()(x)
        elif self.model_type == "vgg16":
            base_model = VGG16(weights='imagenet', include_top=False)
            x = base_model(image_input)
        elif self.model_type == "resnet50":
            base_model = ResNet50(weights='imagenet', include_top=False)
            x = base_model(image_input)
        elif self.model_type == "custom":
            x = Conv2D(16, (3, 3), activation='relu')(image_input)
            x = MaxPooling2D()(x)
            x = Conv2D(32, (3, 3), activation='relu')(x)
            x = GlobalAveragePooling2D()(x)
        else:
            raise ValueError('Invalid model type')

        combined = concatenate([x, y])

        # Use the stored number of unique labels here
        output = Dense(self.n_unique_labels, activation='softmax')(combined)
        self.model = Model(inputs=[image_input, pos_input], outputs=output)


        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def train(self, batch_size, epochs):
        if self.model is None:
            print("No model defined.")
            return

        steps_per_epoch = len(self.image_paths) // batch_size
        self.model.summary()
        self.model.fit(self.data_generator(batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs)

    def save_model(self, model_file):
        if self.model is None:
            print("No model to save.")
            return

        self.model.save(model_file)


if __name__ == "__main__":
    ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), ".."))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    annotation_file = os.path.join(DATA_DIR, "combined_annotations_1000k.csv")
    classifier = CoralReefClassifier(ROOT_DIR, DATA_DIR, IMAGE_DIR, annotation_file, model_type='efficientnet')
    classifier.create_model()
    classifier.train(batch_size=32, epochs=50)
    classifier.save_model('coral_reef_classifier_efficientnet_epoch_50_batchsize_32.h5')
