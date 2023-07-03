import pandas as pd
import os
import sys

ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.annotation import label_mapping

input_csv = r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\annotations\combined_annotations.csv"
output_csv = r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\annotations\combined_annotations_remapped.csv"

# Load the csv file into a pandas DataFrame
df = pd.read_csv(input_csv)

# Apply the mapping to the specific column
# Assuming the column with the labels is named 'Label'
df['Label'] = df['Label'].replace(label_mapping)

# Save the DataFrame back to csv
df.to_csv(output_csv, index=False)
