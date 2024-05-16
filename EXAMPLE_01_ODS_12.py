# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:41:11 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 01 ODS 12
#==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Data simulation
np.random.seed(42)
data = {
    'Budget': np.random.randint(50000, 200000, 1000),
    'Duration': np.random.randint(1, 24, 1000),
    'TeamSize': np.random.randint(3, 15, 1000),
    'Sector': np.random.choice(['Tech', 'Art', 'Design', 'Music', 'Film'], 1000),
    'Success': np.random.randint(0, 2, 1000)
}
df = pd.DataFrame(data)

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Sector'], drop_first=True)

# Separating features and target
X = df.drop('Success', axis=1)
y = df['Success']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model construction
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=0)

# Model evaluation
predictions = (model.predict(X_test_scaled) > 0.5).astype(int)
accuracy = accuracy_score(y_test, predictions)
print(f"The accuracy of the model on the test set is: {accuracy * 100:.2f}%")
