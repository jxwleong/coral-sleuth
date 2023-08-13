import csv
import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR


annotation_file = os.path.join(ANNOTATION_DIR, "combined_annotations_remapped_merged_undersample_oversample_5k.csv")

def create_label_mapping_from_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader)
        labels = [row[3] for row in reader]

    unique_labels = sorted(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    return label_mapping

if __name__ == '__main__':
    label_mapping = create_label_mapping_from_csv(annotation_file)
    print(label_mapping)