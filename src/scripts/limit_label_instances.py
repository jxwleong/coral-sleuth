import pandas as pd
import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

input_filename = "combined_annotations_about_40k_png_only_remapped_majority_class_with_3k_to_4k_sample.csv"
output_filename = input_filename.split(".")[0] + "_reduced_1k" + ".csv"
annotation_file = os.path.join(ANNOTATION_DIR, input_filename)
output_file = os.path.join(ANNOTATION_DIR, output_filename)

def limit_label_instances(df, label_column, min_limit, max_limit):
    """
    Limit the number of instances per category in a DataFrame

    Parameters:
    df (pd.DataFrame): DataFrame to modify
    label_column (str): Name of the column containing the labels/categories
    min_limit (int): Minimum number of instances per category
    max_limit (int): Maximum number of instances per category

    Returns:
    pd.DataFrame: Modified DataFrame
    """
    # Remove categories with less than min_limit instances
    df = df.groupby(label_column).filter(lambda x: len(x) >= min_limit)

    # Limit categories with more than max_limit instances
    df = df.groupby(label_column).apply(lambda x: x.sample(min(len(x), max_limit))).reset_index(drop=True)

    return df

df = pd.read_csv(annotation_file)

# Usage
limited_df = limit_label_instances(df, 'Label', 1000, 1000)

# Save to csv
limited_df.to_csv(output_file, index=False)
