import csv
from collections import defaultdict
import os
import sys 

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR, DATA_DIR, IMAGE_DIR


def limit_annotations_per_image(input_csv, output_csv, limit=2):
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # defaultdict creates a new list when a new key is encountered
        image_dict = defaultdict(list)

        for row in reader:
            image_dict[row[0]].append(row)

        for image, annotations in image_dict.items():
            if len(annotations) > limit:
                annotations = annotations[:limit]
            for annotation in annotations:
                writer.writerow(annotation)

# Call the function to process the CSV file
limit_annotations_per_image(
    os.path.join(DATA_DIR, 'combined_annotations.csv'), 
    os.path.join(DATA_DIR, 'combined_annotations_limit_16.csv'), 
    16
)
