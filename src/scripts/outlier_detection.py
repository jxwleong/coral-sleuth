import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import numpy as np

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT_DIR)

from config.path import ANNOTATION_DIR

input_annotation =  os.path.join(ANNOTATION_DIR, "combined_annotations.csv")

def modified_z_score(data):
    median = data.median()
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return modified_z_scores

# Load the data
data = pd.read_csv(input_annotation)

# Drop the columns that you don't want to check for outliers
data = data.drop(['Name', 'Label', 'Unnamed: 4'], axis=1)

# Z-Score Method
data_zscore = data.apply(zscore)
outliers_zscore = data[(np.abs(data_zscore) > 3).any(axis=1)]
print(f"Number of outliers detected by Z-Score: {len(outliers_zscore)}")
print("Outliers:")
print(outliers_zscore)

# Modified Z-Score Method
data_modified_zscore = data.apply(modified_z_score)
outliers_modified_zscore = data[(np.abs(data_modified_zscore) > 3.5).any(axis=1)]
print(f"\nNumber of outliers detected by Modified Z-Score: {len(outliers_modified_zscore)}")
print("Outliers:")
print(outliers_modified_zscore)

# Isolation Forest
iso = IsolationForest(contamination=0.1)
outlier_pred = iso.fit_predict(data)
outliers_isolation_forest = data[outlier_pred == -1]
print(f"\nNumber of outliers detected by Isolation Forest: {len(outliers_isolation_forest)}")
print("Outliers:")
print(outliers_isolation_forest)