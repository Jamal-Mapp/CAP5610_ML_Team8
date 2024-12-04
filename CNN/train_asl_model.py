import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

# Paths to dataset files
train_csv_path = 'train.csv'  # Path to the training data CSV file
train_landmark_files = 'train_landmark_files/'  # Folder where landmark files are stored
sign_to_prediction_index_map_path = 'sign_to_prediction_index_map.json'  # Path to the JSON file mapping signs to labels

# Load the sign-to-prediction index mapping
with open(sign_to_prediction_index_map_path, 'r') as f:
    sign_to_prediction_index_map = json.load(f)

# Check that the sign-to-prediction index map is loaded correctly
print("Sign to Prediction Index Map:", sign_to_prediction_index_map)

# Load the training CSV file into a pandas DataFrame
train_df = pd.read_csv(train_csv_path)

# Confirm things are workiung correctly
# Check the first few rows of the DataFrame
print("Training Data:")
print(train_df.head())

# Data Preprocessing
# Function to load landmarks and labels from parquet files
def load_landmarks_and_labels(df, landmark_dir, sign_to_prediction_index_map):
    landmarks = []
    labels = []
    
    for index, row in df.iterrows():
        landmark_file = os.path.join(landmark_dir, row['path'])  # Path to the landmark file
        label = sign_to_prediction_index_map.get(row['sign'], None)  # Label from sign-to-prediction mapping
        
        if label is not None:
            if os.path.exists(landmark_file):
                try:
                    # Read parquet file
                    print(f"Loading {landmark_file}...")
                    table = pq.read_table(landmark_file)
                    df_landmarks = table.to_pandas()  # Convert parquet data to DataFrame
                    
                    # Check for 'left_hand' and 'right_hand' landmarks
                    left_hand_landmarks = df_landmarks[df_landmarks['type'] == 'left_hand']
                    right_hand_landmarks = df_landmarks[df_landmarks['type'] == 'right_hand']
                    
                    if len(left_hand_landmarks) > 0 or len(right_hand_landmarks) > 0:
                        # Extract (x, y, z) coordinates for left and right hand landmarks
                        left_hand_coords = left_hand_landmarks[['x', 'y', 'z']].values.flatten() if len(left_hand_landmarks) > 0 else np.zeros(63)
                        right_hand_coords = right_hand_landmarks[['x', 'y', 'z']].values.flatten() if len(right_hand_landmarks) > 0 else np.zeros(63)
                        
                        # Combine both hands' landmarks (63 + 63 = 126 coordinates)
                        combined_hand_coords = np.concatenate([left_hand_coords, right_hand_coords])  # 126 coordinates
                        
                        # Ensure that all landmarks have the same shape (126)
                        if len(combined_hand_coords) == 126:
                            landmarks.append(combined_hand_coords)
                            labels.append(label)
                        else:
                            print(f"Unexpected landmark size for {landmark_file}: {combined_hand_coords.shape}")
                    else:
                        print(f"Missing hand landmarks for {row['path']}")
                except Exception as e:
                    print(f"Error processing {landmark_file}: {e}")
            else:
                print(f"Landmark file {landmark_file} does not exist.")
        else:
            print(f"Label missing for {row['sign']}")

    print(f"Total landmarks processed: {len(landmarks)}")
    
    # Ensure all landmarks are of the same shape (126)
    if all(len(l) == 126 for l in landmarks):
        landmarks_array = np.array(landmarks)
        labels_array = np.array(labels)
        return landmarks_array, labels_array
    else:
        print("Error: Not all landmarks have the expected shape of 126.")
        return None, None

# Load the training data
train_landmarks, train_labels = load_landmarks_and_labels(train_df, train_landmark_files, sign_to_prediction_index_map)

# Check the shapes of the loaded data
if train_landmarks is not None:
    print("Training Landmarks Shape:", train_landmarks.shape)
    print("Training Labels Shape:", train_labels.shape)

    # Normalize landmarks
    train_landmarks = train_landmarks.astype('float32') / 255.0  # Normalize

    # One-hot encode the labels
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(sign_to_prediction_index_map))

    # Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(train_landmarks, train_labels, test_size=0.2, random_state=42)

    # Check the shapes after splitting
    print(f"X_train Shape: {X_train.shape}")
    print(f"y_train Shape: {y_train.shape}")
    print(f"X_val Shape: {X_val.shape}")
    print(f"y_val Shape: {y_val.shape}")

    # This is where the magic happens!
    # Hyperparameters can be adjusted easily
    # Define the CNN model
    dropout_rate = 0.5
    filter_sizes = [(3, 3), (3, 3), (3, 3)]
    num_filters = [32, 64, 128]
    activation_functions = ['relu', 'relu', 'relu', 'relu', 'relu']

    model = Sequential([
        Conv2D(num_filters[0], filter_sizes[0], activation=activation_functions[0], input_shape=(21, 3, 1)),  # Input shape based on landmarks
        MaxPooling2D((2, 2)),
        Conv2D(num_filters[1], filter_sizes[1], activation=activation_functions[1]),
        MaxPooling2D((2, 2)),
        Conv2D(num_filters[2], filter_sizes[2], activation=activation_functions[2]),
        Flatten(),
        Dense(128, activation=activation_functions[3]),
        Dropout(dropout_rate),
        Dense(len(sign_to_prediction_index_map), activation=activation_functions[4])  # Output layer with number of classes
    ])

    # Compile the model
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',  # Categorical cross-entropy for multi-class classification
                  metrics=['accuracy'])
    
    # Number of Epoches can be adjusted here
    # Train the model
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=64)

    # Evaluate the model on the validation data
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Predictions
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Ground truth labels
    y_true = np.argmax(y_val, axis=1)

    # Calculate precision, recall, F1-score
    precision = precision_score(y_true, y_pred_classes, average='macro')
    recall = recall_score(y_true, y_pred_classes, average='macro')
    f1 = f1_score(y_true, y_pred_classes, average='macro')

    # Print results
    print(f'Learning Rate: {learning_rate}')
    print(f'Batch Size: 64')
    print(f'Number of Layers: {len(model.layers)}')
    print(f'Dropout Rate: {dropout_rate}')
    print(f'Number of Epochs: {len(history.history["accuracy"])}')

    # Print model summary
    model.summary()

    # Print performance metrics
    print()
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
else:
    # Something went wrong, Most likely the directory setup
    print("Error: Could not load landmarks correctly.")
