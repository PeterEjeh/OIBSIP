import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from geopy.geocoders import Nominatim
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# File paths
file_path = "transactions.csv"
fraud_alert_path = "fraud_alerts.csv"

# Track processed transactions
processed_ids = set()
fraud_count = 0
last_modified_time = None

# Initialize geolocator for reverse geocoding
geolocator = Nominatim(user_agent="fraud_detection")

# Load and prepare mock data for training
def train_model():
    print("Training the model...")
    num_samples = 1000
    data = {
        "amount": np.random.rand(num_samples) * 20000,
        "time_since_last_transaction": np.random.randint(1, 3600, size=num_samples),
        "num_failed_logins": np.random.randint(0, 5, size=num_samples),
        "location": [
            f"{round(np.random.uniform(-90, 90), 4)},{round(np.random.uniform(-180, 180), 4)}"
            for _ in range(num_samples)
        ],
        "is_fraud": np.random.randint(0, 2, size=num_samples),
    }
    df = pd.DataFrame(data)

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    # Split data into features and target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Model trained successfully!")
    return model, X.columns

# Fraud detection function
def predict_fraud(transaction, model, feature_columns):
    transaction_df = pd.DataFrame([transaction])
    transaction_df = pd.get_dummies(transaction_df, columns=["location"], drop_first=True)

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in transaction_df:
            transaction_df[col] = 0

    transaction_df = transaction_df[feature_columns]
    prediction = model.predict(transaction_df)
    return prediction[0] == 1

# Reverse geocode to get state and country from coordinates
def get_location_info(lat_lon):
    try:
        location = geolocator.reverse(lat_lon, exactly_one=True)
        if location:
            address = location.raw.get("address", {})
            state = address.get("state", "Unknown")
            country = address.get("country", "Unknown")
            return f"{state}, {country}"
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
    return "Unknown Location"

# Monitor the file for new transactions
def monitor_file(file_path, model, feature_columns):
    global processed_ids, fraud_count, last_modified_time

    print("Monitoring file for new transactions...")

    while True:
        try:
            current_modified_time = os.path.getmtime(file_path)
            if last_modified_time is None or current_modified_time > last_modified_time:
                last_modified_time = current_modified_time

                # Read the file
                df = pd.read_csv(file_path, on_bad_lines="warn")
                print(f"File read successfully at {time.ctime(current_modified_time)}.")

                expected_columns = ["transaction_id", "amount", "location", "is_fraud"]
                if not set(expected_columns).issubset(df.columns):
                    print(f"Missing required columns: {set(expected_columns) - set(df.columns)}")
                    time.sleep(5)
                    continue

                new_frauds = []
                for _, row in df.iterrows():
                    transaction_id = row.get("transaction_id", None)
                    if transaction_id is not None and transaction_id not in processed_ids:
                        processed_ids.add(transaction_id)

                        if row.get("is_fraud", 0) == 1:
                            lat_lon = row.get("location", "")
                            location_info = get_location_info(lat_lon)
                            row["location"] = location_info
                            new_frauds.append(row)

                if new_frauds:
                    new_frauds_df = pd.DataFrame(new_frauds)
                    write_mode = "w" if not os.path.exists(fraud_alert_path) else "a"
                    header = write_mode == "w"
                    new_frauds_df.to_csv(fraud_alert_path, mode=write_mode, header=header, index=False)
                    print(f"Fraud alerts updated successfully with {len(new_frauds)} new records.")
        except Exception as e:
            print(f"Error in monitoring file: {e}")

        time.sleep(5)

# Run the file monitoring with a timeout
def run_with_timeout(timeout, func, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            print("Operation timed out.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Main execution
if __name__ == "__main__":
    model, feature_columns = train_model()
    run_with_timeout(120, monitor_file, file_path, model, feature_columns)
