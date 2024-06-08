"""# Feature Extraction"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the normalized data from the CSV file
normalized_data_path = r"E:\train\normalized_data_2018data.csv"
df_normalized = pd.read_csv(normalized_data_path)

# Separate labels from features
labels = df_normalized['Label']
features = df_normalized.drop(columns=['Label','Timestamp'])

# Determine the maximum number of components based on the minimum between the number of features and the number of unique classes minus one
max_components = min(features.shape[1], len(labels.unique()) - 1)

# Perform LDA on the feature matrix
n_components = min(10, max_components)  # Number of components to keep
lda = LinearDiscriminantAnalysis(n_components=n_components)
extracted_features = lda.fit_transform(features, labels)

# Create a DataFrame for the extracted features
df_extracted_features = pd.DataFrame(extracted_features, columns=[f'LDA_Component_{i+1}' for i in range(n_components)])

# Add the labels to the DataFrame
df_extracted_features['Label'] = labels

# Save the extracted features with labels to a new CSV file
extracted_features_output_path = r"E:\train\extracted_features_lda_2018data.csv"
df_extracted_features.to_csv(extracted_features_output_path, index=False)

# Print the values of the extracted features with labels
print("Extracted Features with Labels:")
print(df_extracted_features.head())