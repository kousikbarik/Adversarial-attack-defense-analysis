# Import Libraries
"""

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

"""# Load the Dataset"""

df2=pd.read_csv(r"E:\train\samplled_dataset.csv")
df2

# Number of Unique Values
df2.nunique()

# checking the null values
df2.isnull().sum()

import pandas as pd

# Read the CSV file into a DataFrame
df1 = pd.read_csv(r"E:\train\samplled_dataset.csv")

# Extract the label column
labels = df1['Label']

# Drop the label column before handling missing values
df1.drop(columns=['Label'], inplace=True)

# Fill missing values with the mean of their respective columns
df1_cleaned = df1.fillna(df1.mean())

# Add the label column back to the cleaned DataFrame
df1_cleaned['Label'] = labels

# Define the output directory
output_dir = r"E:\train"

# Save the cleaned DataFrame to a new CSV file in the output directory
df1_cleaned.to_csv(output_dir + "\\cleaned_dataset_2017data.csv", index=False)

print("Data cleaned and saved to:", output_dir)

df1=pd.read_csv(r"E:\train\cleaned_dataset_2017data.csv")
df1

import pandas as pd

# Load the dataset
df1 = pd.read_csv(r"E:\train\cleaned_dataset_2017data.csv")

# Get the value counts for the 'Label' column
label_counts = df1['Label'].value_counts()

# Print the value counts
print("Value counts for each class:")
print(label_counts)

"""# Pre-Processing"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Read the CSV file into a DataFrame
df1 = pd.read_csv(r"E:\train\cleaned_dataset_2017data.csv")

# Check for missing values and handle them (replace with mean or drop, depending on your preference)
df1 = df1.dropna()  # Drop rows with missing values

# Check for infinite values and replace them with NaN
df1.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values after handling missing and infinite values
df1 = df1.dropna()

# Select only the numerical columns for normalization
numerical_columns = df1.select_dtypes(include=['float64', 'int64']).columns

# Create a copy of the DataFrame for printing the values before normalization
original_df1 = df1.copy()

# Create a RobustScaler object
scaler = RobustScaler()

# Apply Robust Scaling to the numerical columns
df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])

# Save the normalized DataFrame to a new CSV file
normalized_output_file_path = r"E:\train\normalized_data_2017data.csv"
df1.to_csv(normalized_output_file_path, index=False)

# Index=False is used to prevent pandas from writing row indices to the CSV file

# Print the original and normalized values
print("Original Data:")
print(original_df1.head())  # Print the first few rows of the original DataFrame
print("\nNormalized Data:")
print(df1.head())  # Print the first few rows of the normalized DataFrame