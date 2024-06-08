# CIC-IDS-2017


"""# Layer 1"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from art.estimators.classification import TensorFlowV2Classifier

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

# Reshape the input data to match the expected shape of the LSTM layer
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#Optimized LSTM Recurrent Grey Wolf NeuroNet
# Define the RNN model wit LSTM layers (source model)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(256, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=5, batch_size=32, verbose=1)

# Create an ART classifier for the source model
estimator_source = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(1, X_train.shape[1]))

# Define the function for minimizing (loss function)
def loss_function(X_adv, X, estimator):
    y_pred = estimator.predict(X_adv.reshape(-1, 1, X_train.shape[1]))  # Reshape input data
    return -np.sum(y_pred[:, 0])  # Access index 0 instead of index 1

# Define the Gray Wolf Optimization (GWO) algorithm
class GWO:
    def __init__(self, objective_function, dim, search_space, max_iter=100, pop_size=10, a=2):
        self.objective_function = objective_function
        self.dim = dim
        self.search_space = search_space
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.zeros((pop_size, dim))
        self.initialize_positions()
        self.a = a  # Parameter controlling the randomness of encircling prey

    def initialize_positions(self):
        self.positions = np.random.uniform(low=self.search_space[0], high=self.search_space[1], size=(self.pop_size, self.dim))

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Update the value of 'a'
            for i in range(self.pop_size):
                # Update the position of the alpha wolf
                if self.objective_function(self.positions[i]) < self.alpha_score:
                    self.alpha_score = self.objective_function(self.positions[i])
                    self.alpha_pos = self.positions[i]

                # Update the position of the beta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.beta_score:
                    self.beta_score = self.objective_function(self.positions[i])
                    self.beta_pos = self.positions[i]

                # Update the position of the delta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.delta_score:
                    self.delta_score = self.objective_function(self.positions[i])
                    self.delta_pos = self.positions[i]

                # Update the positions of the wolves
                C1 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C2 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C3 = 2 * np.random.random() - 1  # Random number in [-1, 1]

                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i])

                X1 = self.alpha_pos - a * D_alpha
                X2 = self.beta_pos - a * D_beta
                X3 = self.delta_pos - a * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3

        return self.alpha_pos

# Generate adversarial samples using L-BFGS with GWO
X_adv_gwo = []
gwo = GWO(objective_function=lambda x: loss_function(x, X_test_reshaped[0], estimator_source),
          dim=X_test_reshaped[0].flatten().shape[0],
          search_space=(-1, 1),
          max_iter=100,
          pop_size=10,
          a=2)
for x_test in X_test_reshaped:
    # Run GWO to find the initial solution
    gwo_params = gwo.optimize()
    # Minimize the loss using L-BFGS-B starting from the GWO solution
    result = minimize(loss_function, gwo_params, args=(x_test, estimator_source), method='L-BFGS-B')
    X_adv_gwo.append(result.x.reshape(1, 1, X_train.shape[1]))
X_adv_gwo = np.concatenate(X_adv_gwo)

# Assuming you have another target_model that you want to evaluate transferability on

# Define the target model and its architecture
target_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(256, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the target model
target_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on the original and adversarial samples (source model)
original_accuracy_source = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]
adversarial_accuracy_source_gwo = model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("Source Model:")
print(f"Original Accuracy: {original_accuracy_source}")
print(f"Adversarial Accuracy: {adversarial_accuracy_source_gwo}")

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_accuracy_target_gwo = target_model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("\nTarget Model:")
print(f"Adversarial Accuracy (using adversarial examples from source model): {adversarial_accuracy_target_gwo}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_predictions_target_gwo = target_model.predict(X_adv_gwo)
adversarial_predictions_target_gwo_binary = np.round(adversarial_predictions_target_gwo).flatten()

precision_adversarial_target_gwo = precision_score(y_test, adversarial_predictions_target_gwo_binary)
recall_adversarial_target_gwo = recall_score(y_test, adversarial_predictions_target_gwo_binary)
f1_score_adversarial_target_gwo = f1_score(y_test, adversarial_predictions_target_gwo_binary)

print("\nTarget Model (Using Adversarial Examples from Source Model):")
print(f"Precision: {precision_adversarial_target_gwo}")
print(f"Recall: {recall_adversarial_target_gwo}")
print(f"F1-score: {f1_score_adversarial_target_gwo}")

import pandas as pd

# Create a DataFrame with adversarial predictions and true labels
df_adversarial_predictions_target_gwo = pd.DataFrame({
    'Adversarial_Predictions_Target_GWO': adversarial_predictions_target_gwo.flatten(),
    'True_Labels': y_test  # Include true labels for reference
})

# Save the DataFrame to a CSV file
df_adversarial_predictions_target_gwo.to_csv('E:/train/adversarial_predictions_target_gwo.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the adversarial predictions for the target model
adversarial_predictions_target_gwo = target_model.predict(X_adv_gwo)
adversarial_predictions_target_gwo_binary = np.round(adversarial_predictions_target_gwo).flatten()

# Calculate the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gwo, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute the ROC curve
fpr_gwo, tpr_gwo, thresholds_gwo = roc_curve(y_test, adversarial_predictions_target_gwo.flatten())

# Compute the AUC score
auc_score_gwo = auc(fpr_gwo, tpr_gwo)

# Plot the ROC curve
plt.figure()
plt.plot(fpr_gwo, tpr_gwo, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score_gwo)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve)')
plt.legend(loc='lower right')
plt.show()

# Print the AUC score
print(f"AUC Score): {auc_score_gwo}")

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Compute precision and recall
precision_gwo, recall_gwo, _ = precision_recall_curve(y_test, adversarial_predictions_target_gwo.flatten())

# Compute area under the curve (AUC) for precision-recall curve
pr_auc_gwo = auc(recall_gwo, precision_gwo)

# Plot the precision-recall curve
plt.figure()
plt.plot(recall_gwo, precision_gwo, color='blue', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc_gwo)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve)')
plt.legend(loc='lower left')
plt.show()

# Print the AUC score for precision-recall curve
print(f"AUC Score for Precision-Recall Curve): {pr_auc_gwo}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the False Negative Rate (FNR)
fnr_gwo = fn_gwo / (fn_gwo + tp_gwo)

print(f"False Negative Rate (FNR)): {fnr_gwo}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the False Positive Rate (FPR)
fpr_gwo = fp_gwo / (fp_gwo + tn_gwo)

print(f"False Positive Rate (FPR)): {fpr_gwo}")

import matplotlib.pyplot as plt

# Compute FNR and FPR
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

# Plot FNR and FPR
categories = ['False Negative Rate (FNR)', 'False Positive Rate (FPR)']
values = [fnr, fpr]

# Define colors for FNR and FPR
colors = ['orange', 'magenta']

# Define markers for FNR and FPR
markers = ['o', 's']

# Plot the line graph with specified colors and markers
for i in range(len(categories)):
    plt.plot(categories[i], values[i], marker=markers[i], linestyle='-', color=colors[i], label=categories[i])

plt.ylabel('Rate')
plt.title('False Negative Rate (FNR) and False Positive Rate (FPR)')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the total number of samples
total_gwo = tn_gwo + fp_gwo + fn_gwo + tp_gwo

# Compute the error rate
error_rate_gwo = (fp_gwo + fn_gwo) / total_gwo

print(f"Error Rate): {error_rate_gwo}")

# Calculate the number of adversarial examples misclassified
misclassified_adversarial_gwo = np.sum(adversarial_predictions_target_gwo_binary != y_test)

# Calculate the total number of adversarial examples
total_adversarial_gwo = len(adversarial_predictions_target_gwo_binary)

# Calculate the Attack Success Rate (ASR)
asr_gwo = (misclassified_adversarial_gwo / total_adversarial_gwo) * 100

print(f"Attack Success Rate (ASR)): {asr_gwo:.2f}%")

import numpy as np
import matplotlib.pyplot as plt

# Define a range of epsilon thresholds
epsilon_thresholds = np.linspace(0, 0.5, 20)

def validate_adversarial_examples_gwo(target_model, X_test, X_adv_gwo, epsilon_threshold):
    """
    Function to validate adversarial examples generated by GWO based on epsilon threshold.
    Args:
        target_model: Trained target model.
        X_test: Original test data.
        X_adv_gwo: Adversarial examples generated by GWO.
        epsilon_threshold: Epsilon threshold for validation.
    Returns:
        valid_adversarial_examples: Binary array indicating whether each example is valid or not.
    """
    predictions_original = target_model.predict(X_test)
    predictions_adv_gwo = target_model.predict(X_adv_gwo)

    # Calculate L2 distance between original and adversarial examples
    l2_distance_gwo = np.sqrt(np.sum((X_test - X_adv_gwo) ** 2, axis=(1, 2)))

    # Check if the L2 distance is above the epsilon threshold
    valid_adversarial_examples_gwo = l2_distance_gwo > epsilon_threshold

    return valid_adversarial_examples_gwo

# Initialize lists to store results
adversarial_examples_percentages_gwo = []
attack_success_rates_gwo = []

# Calculate ASR and AE percentage for each epsilon threshold
for epsilon_threshold in epsilon_thresholds:
    valid_adversarial_examples_gwo = validate_adversarial_examples_gwo(target_model, X_test_reshaped, X_adv_gwo, epsilon_threshold)
    adversarial_examples_percentage_gwo = np.sum(valid_adversarial_examples_gwo) / len(X_test) * 100
    attack_success_rate_gwo = (len(X_test) - np.sum(valid_adversarial_examples_gwo)) / len(X_test) * 100

    adversarial_examples_percentages_gwo.append(adversarial_examples_percentage_gwo)
    attack_success_rates_gwo.append(attack_success_rate_gwo)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(adversarial_examples_percentages_gwo, attack_success_rates_gwo, marker='o', linestyle='-')
plt.xlabel('Adversarial Examples Percentage')
plt.ylabel('Attack Success Rate (%)')
plt.title('Adversarial Examples Percentage vs. Attack Success Rate')
plt.grid(True)
plt.show()

"""# Layer 2"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from art.estimators.classification import TensorFlowV2Classifier

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

# Reshape the input data to match the expected shape of the LSTM layer
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#Optimized LSTM Recurrent Grey Wolf NeuroNet
# Define the RNN model wit LSTM layers (source model)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.Flatten(),  # Flatten LSTM output for further processing
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=5, batch_size=32, verbose=1)

# Create an ART classifier for the source model
estimator_source = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(1, X_train.shape[1]))

# Define the function for minimizing (loss function)
def loss_function(X_adv, X, estimator):
    y_pred = estimator.predict(X_adv.reshape(-1, 1, X_train.shape[1]))  # Reshape input data
    return -np.sum(y_pred[:, 0])  # Access index 0 instead of index 1

# Define the Gray Wolf Optimization (GWO) algorithm
class GWO:
    def __init__(self, objective_function, dim, search_space, max_iter=100, pop_size=10, a=2):
        self.objective_function = objective_function
        self.dim = dim
        self.search_space = search_space
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.zeros((pop_size, dim))
        self.initialize_positions()
        self.a = a  # Parameter controlling the randomness of encircling prey

    def initialize_positions(self):
        self.positions = np.random.uniform(low=self.search_space[0], high=self.search_space[1], size=(self.pop_size, self.dim))

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Update the value of 'a'
            for i in range(self.pop_size):
                # Update the position of the alpha wolf
                if self.objective_function(self.positions[i]) < self.alpha_score:
                    self.alpha_score = self.objective_function(self.positions[i])
                    self.alpha_pos = self.positions[i]

                # Update the position of the beta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.beta_score:
                    self.beta_score = self.objective_function(self.positions[i])
                    self.beta_pos = self.positions[i]

                # Update the position of the delta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.delta_score:
                    self.delta_score = self.objective_function(self.positions[i])
                    self.delta_pos = self.positions[i]

                # Update the positions of the wolves
                C1 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C2 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C3 = 2 * np.random.random() - 1  # Random number in [-1, 1]

                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i])

                X1 = self.alpha_pos - a * D_alpha
                X2 = self.beta_pos - a * D_beta
                X3 = self.delta_pos - a * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3

        return self.alpha_pos

# Generate adversarial samples using L-BFGS with GWO
X_adv_gwo = []
gwo = GWO(objective_function=lambda x: loss_function(x, X_test_reshaped[0], estimator_source),
          dim=X_test_reshaped[0].flatten().shape[0],
          search_space=(-1, 1),
          max_iter=100,
          pop_size=10,
          a=2)
for x_test in X_test_reshaped:
    # Run GWO to find the initial solution
    gwo_params = gwo.optimize()
    # Minimize the loss using L-BFGS-B starting from the GWO solution
    result = minimize(loss_function, gwo_params, args=(x_test, estimator_source), method='L-BFGS-B')
    X_adv_gwo.append(result.x.reshape(1, 1, X_train.shape[1]))
X_adv_gwo = np.concatenate(X_adv_gwo)

# Assuming you have another target_model that you want to evaluate transferability on

# Define the target model and its architecture
target_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.Flatten(),  # Flatten LSTM output for further processing
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the target model
target_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on the original and adversarial samples (source model)
original_accuracy_source = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]
adversarial_accuracy_source_gwo = model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("Source Model:")
print(f"Original Accuracy: {original_accuracy_source}")
print(f"Adversarial Accuracy: {adversarial_accuracy_source_gwo}")

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_accuracy_target_gwo = target_model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("\nTarget Model:")
print(f"Adversarial Accuracy (using adversarial examples from source model): {adversarial_accuracy_target_gwo}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_predictions_target_gwo = target_model.predict(X_adv_gwo)
adversarial_predictions_target_gwo_binary = np.round(adversarial_predictions_target_gwo).flatten()

precision_adversarial_target_gwo = precision_score(y_test, adversarial_predictions_target_gwo_binary)
recall_adversarial_target_gwo = recall_score(y_test, adversarial_predictions_target_gwo_binary)
f1_score_adversarial_target_gwo = f1_score(y_test, adversarial_predictions_target_gwo_binary)

print("\nTarget Model (Using Adversarial Examples from Source Model):")
print(f"Precision: {precision_adversarial_target_gwo}")
print(f"Recall: {recall_adversarial_target_gwo}")
print(f"F1-score: {f1_score_adversarial_target_gwo}")

import pandas as pd

# Create a DataFrame with adversarial predictions and true labels
df_adversarial_predictions_target_gwo = pd.DataFrame({
    'Adversarial_Predictions_Target_GWO': adversarial_predictions_target_gwo.flatten(),
    'True_Labels': y_test  # Include true labels for reference
})

# Save the DataFrame to a CSV file
df_adversarial_predictions_target_gwo.to_csv('E:/train/adversarial_predictions_target_gwo1.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gwo, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute the ROC curve
fpr_gwo, tpr_gwo, thresholds_gwo = roc_curve(y_test, adversarial_predictions_target_gwo.flatten())

# Compute the AUC score
auc_score_gwo = auc(fpr_gwo, tpr_gwo)

# Plot the ROC curve
plt.figure()
plt.plot(fpr_gwo, tpr_gwo, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score_gwo)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve)')
plt.legend(loc='lower right')
plt.show()

# Print the AUC score
print(f"AUC Score): {auc_score_gwo}")

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Compute precision and recall
precision_gwo, recall_gwo, _ = precision_recall_curve(y_test, adversarial_predictions_target_gwo.flatten())

# Compute area under the curve (AUC) for precision-recall curve
pr_auc_gwo = auc(recall_gwo, precision_gwo)

# Plot the precision-recall curve
plt.figure()
plt.plot(recall_gwo, precision_gwo, color='blue', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc_gwo)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve)')
plt.legend(loc='lower left')
plt.show()

# Print the AUC score for precision-recall curve
print(f"AUC Score for Precision-Recall Curve): {pr_auc_gwo}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the False Negative Rate (FNR)
fnr_gwo = fn_gwo / (fn_gwo + tp_gwo)

print(f"False Negative Rate (FNR)): {fnr_gwo}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the False Positive Rate (FPR)
fpr_gwo = fp_gwo / (fp_gwo + tn_gwo)

print(f"False Positive Rate (FPR)): {fpr_gwo}")

import matplotlib.pyplot as plt

# Compute FNR and FPR
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

# Plot FNR and FPR
categories = ['False Negative Rate (FNR)', 'False Positive Rate (FPR)']
values = [fnr, fpr]

# Define colors for FNR and FPR
colors = ['orange', 'magenta']

# Define markers for FNR and FPR
markers = ['o', 's']

# Plot the line graph with specified colors and markers
for i in range(len(categories)):
    plt.plot(categories[i], values[i], marker=markers[i], linestyle='-', color=colors[i], label=categories[i])

plt.ylabel('Rate')
plt.title('False Negative Rate (FNR) and False Positive Rate (FPR)')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the total number of samples
total_gwo = tn_gwo + fp_gwo + fn_gwo + tp_gwo

# Compute the error rate
error_rate_gwo = (fp_gwo + fn_gwo) / total_gwo

print(f"Error Rate): {error_rate_gwo}")

# Calculate the number of adversarial examples misclassified
misclassified_adversarial_gwo = np.sum(adversarial_predictions_target_gwo_binary != y_test)

# Calculate the total number of adversarial examples
total_adversarial_gwo = len(adversarial_predictions_target_gwo_binary)

# Calculate the Attack Success Rate (ASR)
asr_gwo = (misclassified_adversarial_gwo / total_adversarial_gwo) * 100

print(f"Attack Success Rate (ASR)): {asr_gwo:.2f}%")

import numpy as np
import matplotlib.pyplot as plt

# Define a range of epsilon thresholds
epsilon_thresholds = np.linspace(0, 0.5, 20)

def validate_adversarial_examples_gwo(target_model, X_test, X_adv_gwo, epsilon_threshold):
    """
    Function to validate adversarial examples generated by GWO based on epsilon threshold.
    Args:
        target_model: Trained target model.
        X_test: Original test data.
        X_adv_gwo: Adversarial examples generated by GWO.
        epsilon_threshold: Epsilon threshold for validation.
    Returns:
        valid_adversarial_examples: Binary array indicating whether each example is valid or not.
    """
    predictions_original = target_model.predict(X_test)
    predictions_adv_gwo = target_model.predict(X_adv_gwo)

    # Calculate L2 distance between original and adversarial examples
    l2_distance_gwo = np.sqrt(np.sum((X_test - X_adv_gwo) ** 2, axis=(1, 2)))

    # Check if the L2 distance is above the epsilon threshold
    valid_adversarial_examples_gwo = l2_distance_gwo > epsilon_threshold

    return valid_adversarial_examples_gwo

# Initialize lists to store results
adversarial_examples_percentages_gwo = []
attack_success_rates_gwo = []

# Calculate ASR and AE percentage for each epsilon threshold
for epsilon_threshold in epsilon_thresholds:
    valid_adversarial_examples_gwo = validate_adversarial_examples_gwo(target_model, X_test_reshaped, X_adv_gwo, epsilon_threshold)
    adversarial_examples_percentage_gwo = np.sum(valid_adversarial_examples_gwo) / len(X_test) * 100
    attack_success_rate_gwo = (len(X_test) - np.sum(valid_adversarial_examples_gwo)) / len(X_test) * 100

    adversarial_examples_percentages_gwo.append(adversarial_examples_percentage_gwo)
    attack_success_rates_gwo.append(attack_success_rate_gwo)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(adversarial_examples_percentages_gwo, attack_success_rates_gwo, marker='o', linestyle='-')
plt.xlabel('Adversarial Examples Percentage')
plt.ylabel('Attack Success Rate (%)')
plt.title('Adversarial Examples Percentage vs. Attack Success Rate')
plt.grid(True)
plt.show()

"""# Layer 3"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from art.estimators.classification import TensorFlowV2Classifier

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

# Reshape the input data to match the expected shape of the LSTM layer
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#Optimized LSTM Recurrent Grey Wolf NeuroNet
# Define the RNN model wit LSTM layers (source model)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=5, batch_size=32, verbose=1)

# Create an ART classifier for the source model
estimator_source = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(1, X_train.shape[1]))

# Define the function for minimizing (loss function)
def loss_function(X_adv, X, estimator):
    y_pred = estimator.predict(X_adv.reshape(-1, 1, X_train.shape[1]))  # Reshape input data
    return -np.sum(y_pred[:, 0])  # Access index 0 instead of index 1

# Define the Gray Wolf Optimization (GWO) algorithm
class GWO:
    def __init__(self, objective_function, dim, search_space, max_iter=100, pop_size=10, a=2):
        self.objective_function = objective_function
        self.dim = dim
        self.search_space = search_space
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.zeros((pop_size, dim))
        self.initialize_positions()
        self.a = a  # Parameter controlling the randomness of encircling prey

    def initialize_positions(self):
        self.positions = np.random.uniform(low=self.search_space[0], high=self.search_space[1], size=(self.pop_size, self.dim))

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Update the value of 'a'
            for i in range(self.pop_size):
                # Update the position of the alpha wolf
                if self.objective_function(self.positions[i]) < self.alpha_score:
                    self.alpha_score = self.objective_function(self.positions[i])
                    self.alpha_pos = self.positions[i]

                # Update the position of the beta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.beta_score:
                    self.beta_score = self.objective_function(self.positions[i])
                    self.beta_pos = self.positions[i]

                # Update the position of the delta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.delta_score:
                    self.delta_score = self.objective_function(self.positions[i])
                    self.delta_pos = self.positions[i]

                # Update the positions of the wolves
                C1 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C2 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C3 = 2 * np.random.random() - 1  # Random number in [-1, 1]

                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i])

                X1 = self.alpha_pos - a * D_alpha
                X2 = self.beta_pos - a * D_beta
                X3 = self.delta_pos - a * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3

        return self.alpha_pos

# Generate adversarial samples using L-BFGS with GWO
X_adv_gwo = []
gwo = GWO(objective_function=lambda x: loss_function(x, X_test_reshaped[0], estimator_source),
          dim=X_test_reshaped[0].flatten().shape[0],
          search_space=(-1, 1),
          max_iter=100,
          pop_size=10,
          a=2)
for x_test in X_test_reshaped:
    # Run GWO to find the initial solution
    gwo_params = gwo.optimize()
    # Minimize the loss using L-BFGS-B starting from the GWO solution
    result = minimize(loss_function, gwo_params, args=(x_test, estimator_source), method='L-BFGS-B')
    X_adv_gwo.append(result.x.reshape(1, 1, X_train.shape[1]))
X_adv_gwo = np.concatenate(X_adv_gwo)

# Assuming you have another target_model that you want to evaluate transferability on

# Define the target model and its architecture
target_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the target model
target_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on the original and adversarial samples (source model)
original_accuracy_source = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]
adversarial_accuracy_source_gwo = model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("Source Model:")
print(f"Original Accuracy: {original_accuracy_source}")
print(f"Adversarial Accuracy: {adversarial_accuracy_source_gwo}")

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_accuracy_target_gwo = target_model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("\nTarget Model:")
print(f"Adversarial Accuracy (using adversarial examples from source model): {adversarial_accuracy_target_gwo}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_predictions_target_gwo = target_model.predict(X_adv_gwo)
adversarial_predictions_target_gwo_binary = np.round(adversarial_predictions_target_gwo).flatten()

precision_adversarial_target_gwo = precision_score(y_test, adversarial_predictions_target_gwo_binary)
recall_adversarial_target_gwo = recall_score(y_test, adversarial_predictions_target_gwo_binary)
f1_score_adversarial_target_gwo = f1_score(y_test, adversarial_predictions_target_gwo_binary)

print("\nTarget Model (Using Adversarial Examples from Source Model):")
print(f"Precision: {precision_adversarial_target_gwo}")
print(f"Recall: {recall_adversarial_target_gwo}")
print(f"F1-score: {f1_score_adversarial_target_gwo}")

import pandas as pd

# Create a DataFrame with adversarial predictions and true labels
df_adversarial_predictions_target_gwo = pd.DataFrame({
    'Adversarial_Predictions_Target_GWO': adversarial_predictions_target_gwo.flatten(),
    'True_Labels': y_test  # Include true labels for reference
})

# Save the DataFrame to a CSV file
df_adversarial_predictions_target_gwo.to_csv('E:/train/adversarial_predictions_target_gwo2.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gwo, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute the ROC curve
fpr_gwo, tpr_gwo, thresholds_gwo = roc_curve(y_test, adversarial_predictions_target_gwo.flatten())

# Compute the AUC score
auc_score_gwo = auc(fpr_gwo, tpr_gwo)

# Plot the ROC curve
plt.figure()
plt.plot(fpr_gwo, tpr_gwo, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score_gwo)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve)')
plt.legend(loc='lower right')
plt.show()

# Print the AUC score
print(f"AUC Score): {auc_score_gwo}")

from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Compute precision and recall
precision_gwo, recall_gwo, _ = precision_recall_curve(y_test, adversarial_predictions_target_gwo.flatten())

# Compute area under the curve (AUC) for precision-recall curve
pr_auc_gwo = auc(recall_gwo, precision_gwo)

# Plot the precision-recall curve
plt.figure()
plt.plot(recall_gwo, precision_gwo, color='blue', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc_gwo)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve)')
plt.legend(loc='lower left')
plt.show()

# Print the AUC score for precision-recall curve
print(f"AUC Score for Precision-Recall Curve): {pr_auc_gwo}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the False Negative Rate (FNR)
fnr_gwo = fn_gwo / (fn_gwo + tp_gwo)

print(f"False Negative Rate (FNR)): {fnr_gwo}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the False Positive Rate (FPR)
fpr_gwo = fp_gwo / (fp_gwo + tn_gwo)

print(f"False Positive Rate (FPR)): {fpr_gwo}")

import matplotlib.pyplot as plt

# Compute FNR and FPR
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

# Plot FNR and FPR
categories = ['False Negative Rate (FNR)', 'False Positive Rate (FPR)']
values = [fnr, fpr]

# Define colors for FNR and FPR
colors = ['orange', 'magenta']

# Define markers for FNR and FPR
markers = ['o', 's']

# Plot the line graph with specified colors and markers
for i in range(len(categories)):
    plt.plot(categories[i], values[i], marker=markers[i], linestyle='-', color=colors[i], label=categories[i])

plt.ylabel('Rate')
plt.title('False Negative Rate (FNR) and False Positive Rate (FPR)')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm_gwo = confusion_matrix(y_test, adversarial_predictions_target_gwo_binary)

# Extract the values from the confusion matrix
tn_gwo, fp_gwo, fn_gwo, tp_gwo = cm_gwo.ravel()

# Compute the total number of samples
total_gwo = tn_gwo + fp_gwo + fn_gwo + tp_gwo

# Compute the error rate
error_rate_gwo = (fp_gwo + fn_gwo) / total_gwo

print(f"Error Rate): {error_rate_gwo}")

# Calculate the number of adversarial examples misclassified
misclassified_adversarial_gwo = np.sum(adversarial_predictions_target_gwo_binary != y_test)

# Calculate the total number of adversarial examples
total_adversarial_gwo = len(adversarial_predictions_target_gwo_binary)

# Calculate the Attack Success Rate (ASR)
asr_gwo = (misclassified_adversarial_gwo / total_adversarial_gwo) * 100

print(f"Attack Success Rate (ASR)): {asr_gwo:.2f}%")

import numpy as np
import matplotlib.pyplot as plt

# Define a range of epsilon thresholds
epsilon_thresholds = np.linspace(0, 0.5, 20)

def validate_adversarial_examples_gwo(target_model, X_test, X_adv_gwo, epsilon_threshold):
    """
    Function to validate adversarial examples generated by GWO based on epsilon threshold.
    Args:
        target_model: Trained target model.
        X_test: Original test data.
        X_adv_gwo: Adversarial examples generated by GWO.
        epsilon_threshold: Epsilon threshold for validation.
    Returns:
        valid_adversarial_examples: Binary array indicating whether each example is valid or not.
    """
    predictions_original = target_model.predict(X_test)
    predictions_adv_gwo = target_model.predict(X_adv_gwo)

    # Calculate L2 distance between original and adversarial examples
    l2_distance_gwo = np.sqrt(np.sum((X_test - X_adv_gwo) ** 2, axis=(1, 2)))

    # Check if the L2 distance is above the epsilon threshold
    valid_adversarial_examples_gwo = l2_distance_gwo > epsilon_threshold

    return valid_adversarial_examples_gwo

# Initialize lists to store results
adversarial_examples_percentages_gwo = []
attack_success_rates_gwo = []

# Calculate ASR and AE percentage for each epsilon threshold
for epsilon_threshold in epsilon_thresholds:
    valid_adversarial_examples_gwo = validate_adversarial_examples_gwo(target_model, X_test_reshaped, X_adv_gwo, epsilon_threshold)
    adversarial_examples_percentage_gwo = np.sum(valid_adversarial_examples_gwo) / len(X_test) * 100
    attack_success_rate_gwo = (len(X_test) - np.sum(valid_adversarial_examples_gwo)) / len(X_test) * 100

    adversarial_examples_percentages_gwo.append(adversarial_examples_percentage_gwo)
    attack_success_rates_gwo.append(attack_success_rate_gwo)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(adversarial_examples_percentages_gwo, attack_success_rates_gwo, marker='o', linestyle='-')
plt.xlabel('Adversarial Examples Percentage')
plt.ylabel('Attack Success Rate (%)')
plt.title('Adversarial Examples Percentage vs. Attack Success Rate')
plt.grid(True)
plt.show()

"""# Layer 4"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from art.estimators.classification import TensorFlowV2Classifier

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

# Reshape the input data to match the expected shape of the LSTM layer
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#Optimized LSTM Recurrent Grey Wolf NeuroNet
# Define the RNN model wit LSTM layers (source model)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(512, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(256, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_reshaped, y_train, epochs=5, batch_size=32, verbose=1)

# Create an ART classifier for the source model
estimator_source = TensorFlowV2Classifier(model=model, nb_classes=2, input_shape=(1, X_train.shape[1]))

# Define the function for minimizing (loss function)
def loss_function(X_adv, X, estimator):
    y_pred = estimator.predict(X_adv.reshape(-1, 1, X_train.shape[1]))  # Reshape input data
    return -np.sum(y_pred[:, 0])  # Access index 0 instead of index 1

# Define the Gray Wolf Optimization (GWO) algorithm
class GWO:
    def __init__(self, objective_function, dim, search_space, max_iter=100, pop_size=10, a=2):
        self.objective_function = objective_function
        self.dim = dim
        self.search_space = search_space
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.positions = np.zeros((pop_size, dim))
        self.initialize_positions()
        self.a = a  # Parameter controlling the randomness of encircling prey

    def initialize_positions(self):
        self.positions = np.random.uniform(low=self.search_space[0], high=self.search_space[1], size=(self.pop_size, self.dim))

    def optimize(self):
        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # Update the value of 'a'
            for i in range(self.pop_size):
                # Update the position of the alpha wolf
                if self.objective_function(self.positions[i]) < self.alpha_score:
                    self.alpha_score = self.objective_function(self.positions[i])
                    self.alpha_pos = self.positions[i]

                # Update the position of the beta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.beta_score:
                    self.beta_score = self.objective_function(self.positions[i])
                    self.beta_pos = self.positions[i]

                # Update the position of the delta wolf
                if self.alpha_score < self.objective_function(self.positions[i]) < self.delta_score:
                    self.delta_score = self.objective_function(self.positions[i])
                    self.delta_pos = self.positions[i]

                # Update the positions of the wolves
                C1 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C2 = 2 * np.random.random() - 1  # Random number in [-1, 1]
                C3 = 2 * np.random.random() - 1  # Random number in [-1, 1]

                D_alpha = np.abs(C1 * self.alpha_pos - self.positions[i])
                D_beta = np.abs(C2 * self.beta_pos - self.positions[i])
                D_delta = np.abs(C3 * self.delta_pos - self.positions[i])

                X1 = self.alpha_pos - a * D_alpha
                X2 = self.beta_pos - a * D_beta
                X3 = self.delta_pos - a * D_delta

                self.positions[i] = (X1 + X2 + X3) / 3

        return self.alpha_pos

# Generate adversarial samples using L-BFGS with GWO
X_adv_gwo = []
gwo = GWO(objective_function=lambda x: loss_function(x, X_test_reshaped[0], estimator_source),
          dim=X_test_reshaped[0].flatten().shape[0],
          search_space=(-1, 1),
          max_iter=100,
          pop_size=10,
          a=2)
for x_test in X_test_reshaped:
    # Run GWO to find the initial solution
    gwo_params = gwo.optimize()
    # Minimize the loss using L-BFGS-B starting from the GWO solution
    result = minimize(loss_function, gwo_params, args=(x_test, estimator_source), method='L-BFGS-B')
    X_adv_gwo.append(result.x.reshape(1, 1, X_train.shape[1]))
X_adv_gwo = np.concatenate(X_adv_gwo)

# Assuming you have another target_model that you want to evaluate transferability on

# Define the target model and its architecture
target_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, activation='relu', return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(512, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.LSTM(256, activation='relu', return_sequences=True),  # Additional LSTM layer
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Adding a dropout layer for regularization
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the target model
target_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model on the original and adversarial samples (source model)
original_accuracy_source = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]
adversarial_accuracy_source_gwo = model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("Source Model:")
print(f"Original Accuracy: {original_accuracy_source}")
print(f"Adversarial Accuracy: {adversarial_accuracy_source_gwo}")

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_accuracy_target_gwo = target_model.evaluate(X_adv_gwo, y_test, verbose=0)[1]

print("\nTarget Model:")
print(f"Adversarial Accuracy (using adversarial examples from source model): {adversarial_accuracy_target_gwo}")

from sklearn.metrics import precision_score, recall_score, f1_score

# Evaluate the adversarial samples generated from the source model on the target model
adversarial_predictions_target = target_model.predict(X_adv)
adversarial_predictions_target_binary = np.round(adversarial_predictions_target).flatten()

precision_adversarial_target = precision_score(y_test, adversarial_predictions_target_binary)
recall_adversarial_target = recall_score(y_test, adversarial_predictions_target_binary)
f1_score_adversarial_target = f1_score(y_test, adversarial_predictions_target_binary)

print("\nTarget Model (Using Adversarial Examples from Source Model):")
print(f"Precision: {precision_adversarial_target}")
print(f"Recall: {recall_adversarial_target}")
print(f"F1-score: {f1_score_adversarial_target}")

# Save the predictions of the target model on adversarial examples into a CSV file
df_adversarial_predictions_target = pd.DataFrame({
    'Adversarial_Predictions_Target': adversarial_predictions_target.flatten(),
    'True_Labels': y_test  # Include true labels for reference
})
df_adversarial_predictions_target.to_csv('E:/train/adversarial_predictions_target_gwo3.csv', index=False)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(y_test, adversarial_predictions_target_binary)

# Display the confusion matrix with values printed on the plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j + 0.5, i + 0.5, str(cm[i, j]), ha='center', va='center', color='black')

plt.title("Confusion Matrix with Values")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.metrics import roc_curve, auc

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, adversarial_predictions_target_binary)

# Compute the AUC score
auc_score = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Print the AUC score
print(f"AUC Score: {auc_score}")

from sklearn.metrics import precision_recall_curve, auc

# Compute precision and recall
precision, recall, _ = precision_recall_curve(y_test, adversarial_predictions_target_binary)

# Compute area under the curve (AUC) for precision-recall curve
pr_auc = auc(recall, precision)

# Plot the precision-recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AUC = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# Print the AUC score for precision-recall curve
print(f"AUC Score for Precision-Recall Curve: {pr_auc}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, adversarial_predictions_target_binary)

# Extract the values from the confusion matrix
tn, fp, fn, tp = cm.ravel()

# Compute the False Negative Rate (FNR)
fnr = fn / (fn + tp)

print(f"False Negative Rate (FNR): {fnr}")

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, adversarial_predictions_target_binary)

# Extract the values from the confusion matrix
tn, fp, fn, tp = cm.ravel()

# Compute the False Positive Rate (FPR)
fpr = fp / (fp + tn)

print(f"False Positive Rate (FPR): {fpr}")

import matplotlib.pyplot as plt

# Compute FNR and FPR
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

# Plot FNR and FPR
categories = ['False Negative Rate (FNR)', 'False Positive Rate (FPR)']
values = [fnr, fpr]

# Define colors for FNR and FPR
colors = ['orange', 'magenta']

# Define markers for FNR and FPR
markers = ['o', 's']

# Plot the line graph with specified colors and markers
for i in range(len(categories)):
    plt.plot(categories[i], values[i], marker=markers[i], linestyle='-', color=colors[i], label=categories[i])

plt.ylabel('Rate')
plt.title('False Negative Rate (FNR) and False Positive Rate (FPR)')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, adversarial_predictions_target_binary)

# Extract the values from the confusion matrix
tn, fp, fn, tp = cm.ravel()

# Compute the total number of samples
total = tn + fp + fn + tp

# Compute the error rate
error_rate = (fp + fn) / total

print(f"Error Rate: {error_rate}")

# Calculate the number of adversarial examples misclassified
misclassified_adversarial = np.sum(adversarial_predictions_target_binary != y_test)

# Calculate the total number of adversarial examples
total_adversarial = len(adversarial_predictions_target_binary)

# Calculate the Attack Success Rate (ASR)
asr = (misclassified_adversarial / total_adversarial) * 100

print(f"Attack Success Rate (ASR): {asr:.2f}%")

import numpy as np
import matplotlib.pyplot as plt

# Define a range of epsilon thresholds
epsilon_thresholds = np.linspace(0, 0.5, 20)

def validate_adversarial_examples(model, X_test, X_adv, epsilon_threshold):
    """
    Function to validate adversarial examples based on epsilon threshold.
    Args:
        model: Trained model.
        X_test: Original test data.
        X_adv: Adversarial examples.
        epsilon_threshold: Epsilon threshold for validation.
    Returns:
        valid_adversarial_examples: Binary array indicating whether each example is valid or not.
    """
    predictions_original = model.predict(X_test)
    predictions_adv = model.predict(X_adv)

    # Calculate L2 distance between original and adversarial examples
    l2_distance = np.sqrt(np.sum((X_test - X_adv) ** 2, axis=(1, 2)))

    # Check if the L2 distance is above the epsilon threshold
    valid_adversarial_examples = l2_distance > epsilon_threshold

    return valid_adversarial_examples

# Initialize lists to store results
adversarial_examples_percentages = []
attack_success_rates = []

# Calculate ASR and AE percentage for each epsilon threshold
for epsilon_threshold in epsilon_thresholds:
    valid_adversarial_examples = validate_adversarial_examples(model, X_test_reshaped, X_adv, epsilon_threshold)
    adversarial_examples_percentage = np.sum(valid_adversarial_examples) / len(X_test) * 100
    attack_success_rate = (len(X_test) - np.sum(valid_adversarial_examples)) / len(X_test) * 100

    adversarial_examples_percentages.append(adversarial_examples_percentage)
    attack_success_rates.append(attack_success_rate)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(adversarial_examples_percentages, attack_success_rates, marker='o', linestyle='-')
plt.xlabel('Adversarial Examples Percentage')
plt.ylabel('Attack Success Rate (%)')
plt.title('Adversarial Examples Percentage vs. Attack Success Rate')
plt.grid(True)
plt.show()