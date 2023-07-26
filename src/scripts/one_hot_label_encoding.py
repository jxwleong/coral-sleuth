import pandas as pd
import numpy as np

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR


input_annotation =  os.path.join(ANNOTATION_DIR, "combined_annotations.csv")

# Set the option to display the full list without truncation
pd.set_option('display.max_colwidth', None)
# Set the option to display all rows without truncation
pd.set_option('display.max_rows', None)

# Load the annotations file
df = pd.read_csv(input_annotation)

# Get a list of unique coral species
coral_species = df['Label'].unique()

# Generate one-hot encoded vectors for each species
one_hot_encoded_vectors = np.eye(len(coral_species), dtype=int)

# Create a new DataFrame for the encoding
encoding_df = pd.DataFrame({
    'Coral Species': coral_species,
    'One-hot encoded vector': list(one_hot_encoded_vectors)
})

print(encoding_df)
