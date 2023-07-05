import pandas as pd

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

input_annotation =  os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k_png_only.csv")
output_annotation = os.path.join(ANNOTATION_DIR, "combined_annotations_about_40k_png_CCA.csv")

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
    'fav',
    'Mille', 
    'Lepta', 
    'Millepora', 
    'Monta', 
    'MILLE', 
    'ACROP', 
    'P mass', 
    'Soft', 
    'Fung', 
    'MONTA', 
    'Gardin', 
    'LEPTA', 
    'SOFT', 
    'Herpo', 
    'FUNG', 
    'Lobo', 
    'P. Irr', 
    'GARDIN', 
    'ASTREO', 
    'Favia'
]

additional_labels_to_remove = [
    'Turf',
    'Porit',
    'P. Rus',
    'TURF',
    'HS',
    'Macro',
    'SAND',
    'PORIT',
    'MACRO',
    'C-Rubble',
    'Off',
    'OFF',
    'Pavon',
    'POCILL',
    'MONTI',
    'PAVON',
    'Dark',
    'Algae',
    'Acrop'
]

labels_to_remove.extend(additional_labels_to_remove)

# Remove rows where Label is in `labels_to_remove`
df = df[~df['Label'].isin(labels_to_remove)]

# Save the resulting dataframe into a new CSV file
df.to_csv(output_annotation, index=False)
