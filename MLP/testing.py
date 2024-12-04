import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Define MLP model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(input_size, hidden_size)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.hidden_layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.output_layer = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.hidden_layer2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

# Load saved label encoder and scaler
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Load test data
# Assuming test data is saved in the labels.parquet file
parquet_data = pd.read_parquet('labels.parquet')

# Encode non-numeric columns
data_x = parquet_data.copy()
for col in data_x.select_dtypes(include=['object']).columns:
    data_x[col] = LabelEncoder().fit_transform(data_x[col])

# Filter out non-numeric columns and keep only numeric feature columns
data_x = data_x.select_dtypes(include=[np.number])

# Check if there are numeric feature columns
if data_x.shape[1] == 0:
    raise ValueError("There are no numeric feature columns in the test data. Please check the data format.")

# Ensure the number of features in the test data matches the training data
expected_features = scaler.n_features_in_
current_features = data_x.shape[1]

if current_features < expected_features:
    missing_features = expected_features - current_features
    # Add missing feature columns at once, filling with 0
    missing_data = pd.DataFrame(0, index=data_x.index, columns=[f'missing_feature_{i}' for i in range(missing_features)])
    data_x = pd.concat([data_x, missing_data], axis=1)
elif current_features > expected_features:
    raise ValueError(f"The number of features in the test data ({current_features}) exceeds the number of features in the training data ({expected_features}). Please check the data format.")

# Convert to NumPy array and get labels
data_x = data_x.values
data_y = parquet_data.iloc[:, -1].values

# Check and unify label format
try:
    data_y = label_encoder.transform([str(label) for label in data_y])
except ValueError as e:
    print(f"Label conversion error: {e}")
    known_labels_mask = np.isin(data_y, label_encoder.classes_)
    if not np.any(known_labels_mask):
        print("Warning: Test data contains unknown labels, which will be ignored.")
    # Filter out samples with unknown labels
    data_y = data_y[known_labels_mask]
    data_x = data_x[known_labels_mask]
    known_labels_mask = np.isin(data_y, label_encoder.classes_)
    data_y = data_y[known_labels_mask]
    data_x = data_x[known_labels_mask]

# Standardize features
# No need to fit again, just transform
if len(data_x) == 0:
    raise ValueError("All samples contain unknown labels. Unable to proceed with testing.")

data_x = scaler.transform(data_x)

# Convert to PyTorch format for testing
data_x = torch.tensor(data_x, dtype=torch.float32)
data_y = torch.tensor(data_y, dtype=torch.long)

# Load the best model
input_size = data_x.shape[1]
num_classes = len(label_encoder.classes_)
model = MLP(input_size=input_size, hidden_size=512, num_classes=num_classes, dropout_rate=0.3)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Test the model
with torch.no_grad():
    outputs = model(data_x)
    _, predicted = torch.max(outputs.data, 1)

# Calculate basic metrics for the test set
accuracy = accuracy_score(data_y.numpy(), predicted.numpy())
precision = precision_score(data_y.numpy(), predicted.numpy(), average='macro', zero_division=0)
recall = recall_score(data_y.numpy(), predicted.numpy(), average='macro', zero_division=0)
f1 = f1_score(data_y.numpy(), predicted.numpy(), average='macro', zero_division=0)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')

# Return confusion matrix
cm = confusion_matrix(data_y.numpy(), predicted.numpy())

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
