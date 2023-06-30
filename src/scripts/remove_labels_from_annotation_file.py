import pandas as pd


input_annotation = r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\annotations\annotations_coralnet_only.csv"
output_annotation = r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\annotations\annotations_coralnet_only_trimmed.csv"

# Load CSV file
df = pd.read_csv(input_annotation)

# Define the labels want to remove
labels_to_remove = [
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
