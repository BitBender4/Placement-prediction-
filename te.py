import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to Placements_Dataset.csv
csv_path = os.path.join(BASE_DIR, "Placements_Dataset.csv")

# Load CSV
df = pd.read_csv(csv_path)

print("CSV loaded successfully from:", csv_path)

# Example preprocessing
X = df.drop("Placement Package", axis=1)
y = df["Placement Package"]

# Encode categorical columns
label_encoders = {}
for column in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing completed successfully!")
