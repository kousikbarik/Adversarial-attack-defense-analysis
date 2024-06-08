"""# Feature Extraction"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the normalized data from the CSV file
normalized_data_path = r"E:\train\normalized_data_2017data.csv"
df1_normalized = pd.read_csv(normalized_data_path)

# Separate labels from features
labels = df1_normalized['Label']
features = df1_normalized.drop(columns=['Label'])

# Perform LDA on the feature matrix
n_components = 10  # Number of components to keep
lda = LinearDiscriminantAnalysis(n_components=n_components)
extracted_features = lda.fit_transform(features, labels)

# Create a DataFrame for the extracted features
df1_extracted_features = pd.DataFrame(extracted_features, columns=[f'LDA_Component_{i+1}' for i in range(n_components)])

# Add the labels to the DataFrame
df1_extracted_features['Label'] = labels

# Save the extracted features with labels to a new CSV file
extracted_features_output_path = r"E:\train\extracted_features_lda_2017data.csv"
df1_extracted_features.to_csv(extracted_features_output_path, index=False)

# Print the values of the extracted features with labels
print("Extracted Features with Labels:")
print(df1_extracted_features.head())