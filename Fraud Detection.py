import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Set basic style parameters
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True
sns.set_theme(style="whitegrid")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Basic data exploration
print("\n=== Dataset Overview ===")
print("\nFirst few rows:")
print(df.head())

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== Missing Values Check ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.any() else "No missing values found")

# Handle missing values if any
if missing_values.any():
    df = df.fillna(df.mean())

print("\n=== Fraud Distribution ===")
class_dist = df['Class'].value_counts()
print("\nTransaction counts:")
print(class_dist)
print("\nPercentage distribution:")
print(df['Class'].value_counts(normalize=True) * 100)

# Convert Time to hours
df['Time_Hours'] = df['Time'] / 3600


# Create visualizations
def plot_time_amount_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Time distribution
    ax1.hist(df['Time_Hours'], bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Transaction Time Distribution')
    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Number of Transactions')

    # Amount distribution (excluding outliers)
    amount_threshold = df['Amount'].quantile(0.99)
    ax2.hist(df[df['Amount'] < amount_threshold]['Amount'],
             bins=50, color='lightgreen', edgecolor='black')
    ax2.set_title('Transaction Amount Distribution\n(excluding outliers)')
    ax2.set_xlabel('Amount ($)')
    ax2.set_ylabel('Number of Transactions')

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix():
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
                annot=False, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.show()


def plot_amount_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Box plot
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax1)
    ax1.set_title('Transaction Amounts by Class')
    ax1.set_xlabel('Class (0: Normal, 1: Fraud)')
    ax1.set_ylabel('Amount ($)')

    # Violin plot
    sns.violinplot(x='Class', y='Amount', data=df, ax=ax2)
    ax2.set_title('Amount Distribution by Class')
    ax2.set_xlabel('Class (0: Normal, 1: Fraud)')
    ax2.set_ylabel('Amount ($)')

    plt.tight_layout()
    plt.show()


def plot_fraud_patterns():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Time_Hours', y='Amount',
                    hue='Class', alpha=0.5)
    plt.title('Transaction Patterns Over Time')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Amount ($)')
    plt.legend(title='Transaction Type', labels=['Normal', 'Fraud'])
    plt.show()


def plot_feature_distributions():
    v_features = [col for col in df.columns if col.startswith('V')]
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    axes = axes.ravel()

    for idx, feature in enumerate(v_features[:8]):
        sns.kdeplot(data=df, x=feature, hue='Class', ax=axes[idx])
        axes[idx].set_title(f'{feature} Distribution')

    plt.tight_layout()
    plt.show()


# Execute all visualizations
print("\n=== Generating Visualizations ===")
print("1. Time and Amount Distributions")
plot_time_amount_distribution()

print("2. Correlation Matrix")
plot_correlation_matrix()

print("3. Amount Analysis")
plot_amount_analysis()

print("4. Fraud Patterns")
plot_fraud_patterns()

print("5. Feature Distributions")
plot_feature_distributions()

# Additional statistical insights
print("\n=== Additional Insights ===")
print(f"Average transaction amount: ${df['Amount'].mean():.2f}")
print(f"Average fraudulent transaction amount: ${df[df['Class'] == 1]['Amount'].mean():.2f}")
print(f"Average normal transaction amount: ${df[df['Class'] == 0]['Amount'].mean():.2f}")
print(f"Total number of transactions: {len(df)}")
print(f"Total number of fraudulent transactions: {len(df[df['Class'] == 1])}")
print(f"Fraud detection rate: {(len(df[df['Class'] == 1]) / len(df)) * 100:.2f}%")

def plot_statistical_insights():
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Average Transaction Amounts Comparison
    plt.subplot(2, 2, 1)
    avg_amounts = {
        'Overall': df['Amount'].mean(),
        'Normal': df[df['Class'] == 0]['Amount'].mean(),
        'Fraudulent': df[df['Class'] == 1]['Amount'].mean()
    }
    colors = ['skyblue', 'lightgreen', 'salmon']
    plt.bar(avg_amounts.keys(), avg_amounts.values(), color=colors)
    plt.title('Average Transaction Amounts')
    plt.ylabel('Amount ($)')
    for i, v in enumerate(avg_amounts.values()):
        plt.text(i, v, f'${v:.2f}', ha='center', va='bottom')

    # 2. Transaction Volume Pie Chart
    plt.subplot(2, 2, 2)
    fraud_count = len(df[df['Class'] == 1])
    normal_count = len(df[df['Class'] == 0])
    plt.pie([normal_count, fraud_count],
            labels=['Normal', 'Fraudulent'],
            autopct='%1.2f%%',
            colors=['lightgreen', 'salmon'],
            explode=(0, 0.1))
    plt.title('Transaction Distribution')

    # 3. Hourly Transaction Volume
    plt.subplot(2, 2, 3)
    df['Hour'] = df['Time_Hours'] % 24
    hourly_fraud = df[df['Class'] == 1]['Hour'].value_counts().sort_index()
    hourly_normal = df[df['Class'] == 0]['Hour'].value_counts().sort_index()
    plt.plot(hourly_normal.index, hourly_normal.values, label='Normal', color='lightgreen')
    plt.plot(hourly_fraud.index, hourly_fraud.values, label='Fraudulent', color='salmon')
    plt.title('Hourly Transaction Pattern')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Transactions')
    plt.legend()

    # 4. Amount Distribution Density
    plt.subplot(2, 2, 4)
    sns.kdeplot(data=df[df['Class'] == 0]['Amount'], label='Normal', color='lightgreen')
    sns.kdeplot(data=df[df['Class'] == 1]['Amount'], label='Fraudulent', color='salmon')
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount ($)')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Additional Time-based Analysis
    plt.figure(figsize=(12, 6))
    df['Day'] = df['Time_Hours'] // 24
    daily_fraud = df[df['Class'] == 1]['Day'].value_counts().sort_index()
    daily_normal = df[df['Class'] == 0]['Day'].value_counts().sort_index()

    # Normalize the values
    daily_fraud_norm = daily_fraud / daily_fraud.max()
    daily_normal_norm = daily_normal / daily_normal.max()

    plt.plot(daily_normal_norm.index, daily_normal_norm.values,
             label='Normal', color='lightgreen')
    plt.plot(daily_fraud_norm.index, daily_fraud_norm.values,
             label='Fraudulent', color='salmon')
    plt.title('Daily Transaction Pattern (Normalized)')
    plt.xlabel('Day')
    plt.ylabel('Normalized Transaction Volume')
    plt.legend()
    plt.show()


# Add this to your main code execution
print("\n=== Statistical Insights Visualization ===")
plot_statistical_insights()

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

# Build neural network with explicit input layer
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

# Evaluate model
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()  # Flatten to 1D array

print("\n=== Model Performance ===")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix - Neural Network')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print detailed metrics for the fraud class
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
print("\nFraud Class Metrics:")
print(f"Precision: {precision[1]:.2f}")
print(f"Recall: {recall[1]:.2f}")
print(f"F1-Score: {fscore[1]:.2f}")

# Check data shapes before model training
print("X_train_scaled shape:", X_train_scaled.shape)
print("y_train shape:", y_train.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_test shape:", y_test.shape)
