import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("Placements_Dataset.csv")
print("Dataset Preview:\n", df.head())
print("\nColumn Names Found:", df.columns.tolist())

# Drop unnecessary columns if they exist
drop_cols = ['Name of Student', 'Roll No.']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Fill missing values for selected columns
fill_cols = ['Knows ML', 'Knows Python', 'Knows JavaScript', 'Knows HTML', 'Knows CSS']
for col in fill_cols:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nNull Values After Filling:\n", df.isnull().sum())

# Create binary target: Placed or Not Placed
if 'Placement Package' in df.columns:
    df['Placed'] = df['Placement Package'].apply(lambda x: 1 if x > 0 else 0)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    print(f"Encoded: {col}")

# Features and target split
X = df.drop('Placement Package', axis=1)
y = df['Placement Package']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "sal_scaler.pkl")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Use linear for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=500, callbacks=[early_stop], verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest Mean Absolute Error: Rs. {mae:.2f}")

# Save model
model.save("salary_model_tf.h5")
print("Model and scaler saved successfully.")
print("Max Package:", df['Placement Package'].max())
print("Min Package:", df['Placement Package'].min())
print("Mean Package:", df['Placement Package'].mean())
