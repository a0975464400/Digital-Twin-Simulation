import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import joblib
import matplotlib.pyplot as plt

# Load the data
sensor_data = pd.read_csv('/Users/a0975464400/Desktop/Essay/Project/Dataset/archive-2/sensor_raw.csv')

sensor_data['VelocityX'] = sensor_data['AccX'].cumsum()

# Create lagged features
sensor_data['Lagged_Velocity'] = sensor_data['VelocityX'].shift(1).fillna(0)
sensor_data['Lagged_Acceleration'] = sensor_data['AccX'].shift(1).fillna(0)

# Compute interaction feature
sensor_data['VelocityX_LagInteraction'] = sensor_data['VelocityX'] * sensor_data['Lagged_Velocity']

features_nn = sensor_data[['VelocityX', 'Lagged_Velocity', 'Lagged_Acceleration', 'VelocityX_LagInteraction']]
target_nn = sensor_data['AccX']

# Normalize the features
scaler_nn = MinMaxScaler()
features_nn_scaled = scaler_nn.fit_transform(features_nn)
joblib.dump(scaler_nn, 'scaler_nn.pkl')

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
features_nn_poly = poly.fit_transform(features_nn_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_nn_poly, target_nn, test_size=0.2)

# Train the Gradient Boosting Model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Visualize Predictions
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Gradient Boosting: Actual vs Predicted')
plt.show()
