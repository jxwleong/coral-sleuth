import pandas as pd
import matplotlib.pyplot as plt

import os
import sys


ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import MODEL_DIR

filename = "coral_reef_classifier_efficientnetv2_epoch_200_batchsize_32_metrics_combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample_reduced_1k_scale_0.2.csv"
input_file = os.path.join(MODEL_DIR, filename)
df = pd.read_csv(input_file)

print("MIN")
# Calculate the mean of each column
means = df.mean()

# Print the mean of each column
print(means)

print("\nMAX")
maxs = df.max()
print(maxs)