
#Incorporate Domain-Specific Knowledge & Data Preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('/Users/a0975464400/Desktop/Essay/Project/Dataset/archive-2/sensor_raw.csv')

# Domain-specific feature: Safety Margin based on Velocity
data['predicted_safety_margin'] = 0.05 * data['Velocity']**2 + 0.1 * data['Velocity'] + 5

# Normalize the data
features = data[['Velocity', 'NearbyVehicles', 'dry', 'wet', 'icy', 'predicted_safety_margin']]
target = data['Distance']

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

#Neural Network Architecture
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

#Training
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

#Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
