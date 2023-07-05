import pandas as pd

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

input_annotation =  os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k_png_only_remapped.csv")
output_annotation = os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample.csv")

# Load CSV file
df = pd.read_csv(input_annotation)

# Define the count thresholds
min_threshold = 3000
max_threshold = 4000  # change this to a specific value if needed

# Get label counts
label_counts = df['Label'].value_counts()

# If max_threshold is set, get labels that have count more than the max_threshold
if max_threshold is not None:
    labels_to_remove_max = label_counts[label_counts > max_threshold].index.tolist()
else:
    labels_to_remove_max = []

# Get labels that have count less than the min_threshold
labels_to_remove_min = label_counts[label_counts < min_threshold].index.tolist()

# Combine the lists of labels to remove
labels_to_remove = labels_to_remove_max + labels_to_remove_min

# Remove rows where Label is in `labels_to_remove`
df = df[~df['Label'].isin(labels_to_remove)]

# Save the resulting dataframe into a new CSV file
df.to_csv(output_annotation, index=False)
