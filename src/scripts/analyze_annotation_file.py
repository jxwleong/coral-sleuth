import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

annotation_filename = "combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample.csv"
annotation_filepath = os.path.join(ANNOTATION_DIR, annotation_filename)

df = pd.read_csv(annotation_filepath)

# Print the summary statistics for the data
print(df.describe())

# Count the number of times each label appears
label_counts = df['Label'].value_counts()
print(label_counts)

# Group the data by image name
image_groups = df.groupby('Name')

# Print the mean value for each column in each image group
for name, group in image_groups:
    print(name, group.mean())

# Check for unique values in each column
for column in df.columns:
    if df[column].nunique() == 1:
        print(f'The column {column} only has one unique value.')

# Check for missing values in each column
for column in df.columns:
    if df[column].isna().sum() > 0:
        print(f'The column {column} has {df[column].isna().sum()} missing values.')

# Get information about the data types of each column
print(df.info())

# Create a bar chart of the label counts
label_counts = df['Label'].value_counts()
plt.bar(label_counts.index, label_counts.values)
plt.xlabel('Label')
plt.ylabel('Number of images')
plt.show()