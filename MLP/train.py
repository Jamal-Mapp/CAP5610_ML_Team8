import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid
import joblib


random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

data_x = np.load('feature_data.npy')
data_y = np.load('feature_labels.npy')


label_encoder = LabelEncoder()
data_y = label_encoder.fit_transform(data_y)

# Save label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Feature standardization and saving StandardScaler
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)

# Save StandardScaler
joblib.dump(scaler, 'scaler.pkl')

# Convert to PyTorch tensor
data_x = torch.tensor(data_x, dtype=torch.float32)
data_y = torch.tensor(data_y, dtype=torch.long)

#Create dataset
dataset = TensorDataset(data_x, data_y)

# Divide training set and validation set
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, num_classes)

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

# Initialize model, loss function and optimizer
input_size = data_x.shape[1]
num_classes = len(np.unique(data_y))
model = MLP(input_size=input_size, hidden_size=512, num_classes=num_classes, dropout_rate=0.3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Set early stopping policy
early_stopping_patience = 15
best_epoch_accuracy = 0
patience_counter = 0

# Number of training rounds
num_epochs = 1000

# Storage indicators
train_losses = []
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1_scores = []

val_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1_scores = []

for epoch in range(num_epochs):
    #Training phase
    model.train()
    epoch_loss = 0
    all_labels = []
    all_preds = []
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(batch_y.numpy())
        all_preds.extend(predicted.numpy())

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Calculate training set metrics
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1_scores.append(train_f1)

    # Verification phase
    model.eval()
    val_epoch_loss = 0
    val_all_labels = []
    val_all_preds = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_epoch_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_all_labels.extend(batch_y.numpy())
            val_all_preds.extend(predicted.numpy())

    val_avg_loss = val_epoch_loss / len(val_loader)
    val_losses.append(val_avg_loss)

    # Calculate validation set metrics
    val_accuracy = accuracy_score(val_all_labels, val_all_preds)
    val_precision = precision_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
    val_recall = recall_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_all_labels, val_all_preds, average='macro', zero_division=0)

    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1_scores.append(val_f1)

    scheduler.step(val_accuracy)

    # Output results
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training   - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy*100:.2f}%, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}')
        print(f'Validation - Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}')
        print('-' * 80)

    # Early stopping strategy
    if val_accuracy > best_epoch_accuracy:
        best_epoch_accuracy = val_accuracy
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Save model
torch.save(model.state_dict(), 'best_model.pth')

# Draw learning curve
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 10))

# Draw loss curve
plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Draw accuracy curve
plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Draw F1 score curve
plt.subplot(2, 2, 3)
plt.plot(epochs, train_f1_scores, 'b-', label='Training F1 Score')
plt.plot(epochs, val_f1_scores, 'r-', label='Validation F1 Score')
plt.title('F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()
