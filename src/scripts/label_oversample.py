import pandas as pd
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

filename = "combined_annotations_1000_remapped.csv"
filepath = os.path.join(ANNOTATION_DIR, filename)

output_filename = filename.replace(".csv", "") + "_oversample.csv"
outfile = os.path.join(ANNOTATION_DIR, output_filename)

threshold = 3  # threshold for minority class count
oversample_count = 10  # desired count for oversampling

data = pd.read_csv(filepath)

#  Count the number of instances for each class
class_counts = data['Label'].value_counts().sort_index()

# Get the labels
labels = class_counts.index.tolist()

# Get the initial class distribution
initial_distribution = class_counts.tolist()

# Perform oversampling
oversampled_data = pd.DataFrame()

for label, count in class_counts.iteritems():
    label_data = data[data['Label'] == label]
    
    # If this class is below the threshold, resample with replacement to the desired count
    if count <= threshold and count < oversample_count:
        label_data = resample(label_data, replace=True, n_samples=oversample_count, random_state=123)
        
    oversampled_data = pd.concat([oversampled_data, label_data])

oversampled_data = oversampled_data.sample(frac=1).reset_index(drop=True)  # shuffle the data
oversampled_data.to_csv(outfile, index=False)

# Get the new class distribution
new_distribution = oversampled_data['Label'].value_counts().sort_index().tolist()

# Plot the initial and new class distribution
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, initial_distribution, width, label='Before')
rects2 = ax.bar(x + width/2, new_distribution, width, label='After')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Class Distribution before and after Oversampling')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
