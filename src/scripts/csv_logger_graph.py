import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import MODEL_DIR


filename = "coral_reef_classifier_efficientnetv2_epoch_50_batchsize_32_metrics_combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample_reduced_1k.csv.csv"
filepath = os.path.join(MODEL_DIR, filename)
# Load the data from the CSV file
data = pd.read_csv(filepath)  

# Extract the data for epochs and accuracy
epochs = data['epoch']
accuracy = data['accuracy']

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(epochs, accuracy, label='Training Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()