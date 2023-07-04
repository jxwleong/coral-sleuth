import pandas as pd

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

input_annotation =  os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k_png_only_remapped.csv")
output_annotation = os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k_png_only_remapped_majority_class_with_1k_or_more.csv")

# Load CSV file
df = pd.read_csv(input_annotation)

# Define the count threshold
count_threshold = 1000

# Get label counts
label_counts = df['Label'].value_counts()

# Get the labels that have count less than the threshold
labels_to_remove = label_counts[label_counts < count_threshold].index.tolist()

# Remove rows where Label is in `labels_to_remove`
df = df[~df['Label'].isin(labels_to_remove)]

# Save the resulting dataframe into a new CSV file
df.to_csv(output_annotation, index=False)
