import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

# Define the path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Create the models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Handle missing values if any
missing_values = df.isnull().sum()
if missing_values.any():
    df = df.fillna(df.mean())

# Prepare data
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure y_train and y_test are numpy arrays
y_train = y_train.values if isinstance(y_train, pd.Series) else np.array(y_train)
y_test = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)

# Build neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Handle class imbalance
class_0_count, class_1_count = np.bincount(y_train)
weight_for_0 = 1.0 / class_0_count
weight_for_1 = 1.0 / class_1_count
weights = {0: weight_for_0, 1: weight_for_1}

# Train model
history = model.fit(X_train_scaled, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    class_weight=weights)

# Save the model and scaler
model.save(os.path.join(MODELS_DIR, 'fraud_detection_model.h5'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

print(f"Model and scaler saved in {MODELS_DIR}")