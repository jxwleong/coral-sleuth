import pandas as pd

# Load your CSV file
df = pd.read_csv(r"C:\Users\ad_xleong\Desktop\coral-sleuth\data\annotations\combined_annotations.csv")

# Check the top 5 rows to see what your data looks like
print(df.head())

# Let's assume the column with the labels is called 'label'
# Count the number of occurrences of each label
label_counts = df['Label'].value_counts()

print("Label distribution:")
print(label_counts)

# You could also visualize this distribution
#label_counts.plot(kind='bar')



