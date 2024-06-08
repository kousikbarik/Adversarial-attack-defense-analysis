-------------------------------------Classification-------------------------------------------


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the extracted features and labels from the CSV file
extracted_features_path = r"E:\train\encoded_features_2017data.csv"
df_extracted_features = pd.read_csv(extracted_features_path)

# Separate features and labels
X = df_extracted_features.drop(columns=['Label'])
y = df_extracted_features['Label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape input data to include timestep dimension
X_train_reshaped = np.expand_dims(X_train, axis=1)
X_test_reshaped = np.expand_dims(X_test, axis=1)

# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.Flatten(),  # Flatten LSTM output for further processing
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Accuracy: {accuracy}")

# Make predictions on the test set
y_pred = (model.predict(X_test_reshaped) > 0.5).astype("int32")
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)