import pandas as pd
import numpy as np

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

filename = "combined_annotations_1000_remapped.csv"
filepath = os.path.join(ANNOTATION_DIR, filename)

output_filename = filename.replace(".csv", "") + "_merged.csv"
outfile = os.path.join(ANNOTATION_DIR, output_filename)

# Read the data
data = pd.read_csv(filepath)

# Set the threshold
threshold = 500

# Group classes with less than `threshold` instances into "other"
data['Label'] = np.where(data['Label'].map(data['Label'].value_counts()) <= threshold, 'other', data['Label'])

# Save the merged data
data.to_csv(outfile, index=False)

# Display the new distribution
print(data['Label'].value_counts())
