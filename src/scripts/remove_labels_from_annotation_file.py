import pandas as pd

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

input_annotation =  os.path.join(ANNOTATION_DIR, "annotations_coralnet_only.csv")
output_annotation = os.path.join(ANNOTATION_DIR, "annotations_coralnet_only_trimmed.csv")

# Load CSV file
df = pd.read_csv(input_annotation)

# Define the labels want to remove
labels_to_remove = [
    'Sand',
    'Acropora',
    'Acr_dig',
    'Pavona',
    'Porites',
    'Acr_tab',
    'BAD',
    'Goniastrea',
    'Monti_encr',
    'Dark',
    'Monti',
    'Algae',
    'Pocill',
    'Millepora',
    'Lepta',
    'D_coral',        
    'SC',             
    'Platy',          
    'Rock',           
    'Echinopora',     
    'GFA',        
    'Favites',        
    'Astreo',          
    'fav'
]


# Remove rows where Label is in `labels_to_remove`
df = df[~df['Label'].isin(labels_to_remove)]

# Save the resulting dataframe into a new CSV file
df.to_csv(output_annotation, index=False)
