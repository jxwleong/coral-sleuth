import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR


# Show all rows instead of "..." in the middle
pd.set_option('display.max_rows', None)
annotation_file = os.path.join(ANNOTATION_DIR, "combined_annotations.csv")

# Load your CSV file
df = pd.read_csv(annotation_file)

# Check the top 5 rows to see what your data looks like
print(df.head())

# Let's assume the column with the labels is called 'label'
# Count the number of occurrences of each label
label_counts = df['Label'].value_counts()

print("Label distribution:")
print(label_counts)

# You could also visualize this distribution
label_counts.plot(kind='bar')

#plt.show()  # This will show the plot

