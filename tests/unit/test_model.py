import os
import unittest
import numpy as np
from keras.models import Model
from keras.utils import to_categorical

import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
from src.model_training import CoralReefClassifier


class TestCoralReefClassifier(unittest.TestCase):
    def setUp(self):
        self.root_dir = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", ".."))
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.image_dir = self.data_dir
        self.annotation_file = os.path.join(self.data_dir, 'test_simple_annotations.csv')
        self.model_type = 'custom'  

        self.classifier = CoralReefClassifier(self.root_dir, self.data_dir, self.image_dir, self.annotation_file, self.model_type)


    def test_load_data(self):
        # self.classifier.load_data()
        self.assertEqual(len(self.classifier.image_paths), 2)
        self.assertTrue(all(os.path.exists(path) for path in self.classifier.image_paths))
            
        print("One-hot encoded labels:", self.classifier.labels)  # Added print statement
            
        # Convert one-hot encoded labels back to integer labels
        labels = np.argmax(self.classifier.labels, axis=-1)

        # Mimic the creation of the label mapping in load_data function
        unique_labels = np.unique(labels)
        label_mapping = {i: label for i, label in enumerate(unique_labels)}

        # Convert the integer labels back to their original string labels
        original_labels = [label_mapping[label] for label in labels]

        self.assertEqual(len(original_labels), 2)
        print("Original labels:", original_labels)  # Added print statement
        self.assertTrue(all(label in unique_labels for label in original_labels))


    def test_create_model(self):
        self.classifier.create_model()
        self.assertIsInstance(self.classifier.model, Model)
        self.assertIsNotNone(self.classifier.model)


    def test_train(self):
        self.classifier.create_model()
        initial_weights = self.classifier.model.get_weights()

        # Train on a single batch for a single epoch
        self.classifier.train(batch_size=1, epochs=1)

        # Check that the model's weights have changed
        final_weights = self.classifier.model.get_weights()
        self.assertFalse(np.array_equal(initial_weights, final_weights))

        # Check that the training time has been set
        self.assertIsNotNone(self.classifier.training_time)
        self.assertGreater(self.classifier.training_time, 0)


    def test_evaluate(self):
        self.classifier.create_model()
        self.classifier.train(batch_size=1, epochs=1)
        metrics = self.classifier.get_evaluation_metrics(batch_size=1)

        print(metrics)
      
        self.assertTrue(metrics["loss"] >= 0.0)
        self.assertTrue(metrics["precision_1"] >= 0.0)
        self.assertTrue(metrics["recall_1"] >= 0.0)
        self.assertTrue(metrics["auc_1"] >= 0.0)
        self.assertTrue(metrics["true_positives_1"] >= 0.0)
        self.assertTrue(metrics["true_negatives_1"] >= 0.0)
        self.assertTrue(metrics["false_positives_1"] >= 0.0)
        self.assertTrue(metrics["false_negatives_1"] >= 0.0)


if __name__ == '__main__':
    unittest.main()
