import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

annotation_filename = "combined_annotations.csv"
annotation_filepath = os.path.join(ANNOTATION_DIR, annotation_filename)

df = pd.read_csv(annotation_filepath)


# Create a bar chart of the label counts
label_counts = df['Label'].value_counts()
plt.bar(label_counts.index, label_counts.values, color='#0074D9')
plt.xlabel('Label', rotation=90, ha='right')
plt.xticks(label_counts.index, rotation=90)

plt.ylabel('Number of images')
plt.show()