"""# Label Encoding"""

from sklearn.preprocessing import LabelEncoder

# Define label encoding dictionary
label_encoding = {
    'Benign': 0,
    'DoS attacks-GoldenEye': 1,
    'DoS attacks-Slowloris': 1
}

# Apply label encoding to the 'Label' column
df_extracted_features['Label'] = df_extracted_features['Label'].map(label_encoding)

# Print the values of the extracted features with encoded labels
print("Extracted Features with Encoded Labels:")
print(df_extracted_features.head())

# Save the DataFrame with encoded labels to a new CSV file
encoded_features_output_path = r"E:\train\encoded_features_2018data.csv"
df_extracted_features.to_csv(encoded_features_output_path, index=False)

# Print the path where the encoded features are saved
print(f"Encoded features saved to: {encoded_features_output_path}")



import pandas as pd

# Load the dataset
df1 = pd.read_csv(r"E:\train\encoded_features_2018data.csv")

# Get the value counts for the 'Label' column
label_counts = df1['Label'].value_counts()

# Print the value counts
print("Value counts for each class:")
print(label_counts)