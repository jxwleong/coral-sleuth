import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import ANNOTATION_DIR

filename = "combined_annotations_remapped_merged.csv"
filepath = os.path.join(ANNOTATION_DIR, filename)

output_filename = filename.replace(".csv", "") + "_undersample.csv"
outfile = os.path.join(ANNOTATION_DIR, output_filename)

threshold = 226000  # threshold for class count
undersample_count = 44000  # desired count for undersampling

data = pd.read_csv(filepath)

# Count the number of instances for each class
class_counts = data['Label'].value_counts().sort_index()

# Get the labels
labels = class_counts.index.tolist()

# Get the initial class distribution
initial_distribution = class_counts.tolist()

# Perform undersampling
undersampled_data = pd.DataFrame()

for label, count in class_counts.iteritems():
    label_data = data[data['Label'] == label]
    
    # If this class count exceeds the threshold, randomly select instances equal to the undersample count
    if count >= threshold:
        label_data = label_data.sample(min(undersample_count, len(label_data)), random_state=123)
        
    undersampled_data = pd.concat([undersampled_data, label_data])

undersampled_data = undersampled_data.sample(frac=1).reset_index(drop=True)  # shuffle the data
undersampled_data.to_csv(outfile, index=False)

# Display the new distribution
print(undersampled_data['Label'].value_counts())
""" 
# Get the new class distribution
new_distribution = undersampled_data['Label'].value_counts().sort_index().tolist()

# Plot the initial and new class distribution
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, initial_distribution, width, label='Before')
rects2 = ax.bar(x + width/2, new_distribution, width, label='After')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Class Distribution before and after Undersampling')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
"""