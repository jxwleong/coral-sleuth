import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import MODEL_DIR


filenames = [
    "coral_reef_classifier_efficientnetv2_epoch_200_batchsize_32_metrics_combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample_reduced_1k.csv.csv"
] 
filepath = MODEL_DIR


# A dictionary mapping filenames to labels
filename_to_label = {
    filenames[0]: "Label 1",
    # Add more as needed
}


# Create the plot
plt.figure(figsize=(10,6))

for filename in filenames:
    file = os.path.join(filepath, filename)
    data = pd.read_csv(file)  

    # Extract the data for epochs and accuracy
    epochs = data['epoch']
    accuracy = data['accuracy']

    # Use the filename to look up the label
    label = filename_to_label[filename]

    # Plot the data
    plt.plot(epochs, accuracy, label=label, linewidth=2)

plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()