import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)
from config.path import ANNOTATION_DIR


filename = "combined_annotations_1000_remapped.csv"
filepath = os.path.join(ANNOTATION_DIR, filename)

output_filename = filename.replace(".csv", "") + "_undersample.csv"
outfile = os.path.join(ANNOTATION_DIR, output_filename)

labels_to_undersample = [
    "porites",
    "crustose_coralline_algae",
    "turf"
]

undersample_count = 10  # desired count for undersampling

data = pd.read_csv(filepath)

#  Count the number of instances for each class
class_counts = data['Label'].value_counts().sort_index()

# Get the labels
labels = class_counts.index.tolist()

# Get the initial class distribution
initial_distribution = class_counts.tolist()

# Perform undersampling
undersampled_data = pd.DataFrame()

for label in class_counts.index:
    label_data = data[data['Label'] == label]
    
    # If this is one of the classes to be undersampled, randomly select instances equal to the specified count
    if label in labels_to_undersample:
        if len(label_data) > undersample_count:  # check if the class count is larger than the undersample count
            label_data = label_data.sample(undersample_count)
        
    undersampled_data = pd.concat([undersampled_data, label_data])

undersampled_data = undersampled_data.sample(frac=1).reset_index(drop=True)  # shuffle the data
undersampled_data.to_csv(outfile, index=False)


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