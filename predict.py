import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping



df = pd.read_csv("Placements_Dataset.csv")

print(df.head(3))
print( df.shape)
print(df.isnull().sum())

df['Knows ML'].fillna(df['Knows ML'].mode()[0], inplace=True)
df['Knows Python'].fillna(df['Knows Python'].mode()[0], inplace=True)
df['Knows JavaScript'].fillna(df['Knows JavaScript'].mode()[0], inplace=True)
df['Knows HTML'].fillna(df['Knows HTML'].mode()[0], inplace=True)
df['Knows CSS'].fillna(df['Knows CSS'].mode()[0], inplace=True)
print(df.isnull().sum())

label_cols = ["Name of Student","Roll No.","Knows ML","Knows DSA","Knows Python",	"Knows JavaScript",	"Knows HTML","Knows CSS","Knows Cricket","Knows Dance",	"Participated in College Fest",	"Was in Coding Club","No. of backlogs",	"Interview Room Temperature","Age of Candidate","Branch of Engineering"]  
le = LabelEncoder()
for col in label_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])



plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop(['Placement Package'], axis=1)
y = df['Placement Package']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nProcessed Features Shape:", X_scaled.shape)
print("Target Distribution:\n", pd.Series(y).value_counts())
joblib.dump(scaler, "salary_scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stop], verbose=1)

loss, mae = model.evaluate(X_test, y_test)
print(f"\nTensorFlow Salary Model MAE: INR {mae:.2f}")


model.save("salary_model_tf.h5")