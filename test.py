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

df=pd.read_csv("college_student_placement_dataset.csv")
print(df.head(3))
print( df.shape)

print(df.isnull().sum())

label_cols = ['College_ID',	'IQ' ,'Prev_Sem_Result','CGPA',	'Academic_Performance',	'Internship_Experience','Extra_Curricular_Score','Communication_Skills','Projects_Completed','Placement' ]
le = LabelEncoder()

for col in label_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])


plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


X = df.drop(['Placement'], axis=1)
y = df['Placement']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nProcessed Features Shape:", X_scaled.shape)
print("Target Distribution:\n", pd.Series(y).value_counts())

joblib.dump(scaler,"scalar.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stop], verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n TensorFlow Model Accuracy: {acc:.2f}")

# Save the model
model.save("placement_model_tf.h5")