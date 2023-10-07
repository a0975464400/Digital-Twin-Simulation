import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the trained neural network model
nn_model = load_model('trained_nn_model.h5')


# Load the scaler
scaler = joblib.load('scaler_nn.pkl')

# Loading sensor data
sensor_data = pd.read_csv('/Users/a0975464400/Desktop/Essay/Project/Dataset/archive-2/sensor_raw.csv')

# Compute the velocity based on the 'AccX' data
sensor_data['VelocityX'] = sensor_data['AccX'].cumsum()

# Data Collection
def collect_data_from_sensor():
    velocities = sensor_data['VelocityX'].values  # Use VelocityX as velocities
    distances = sensor_data['AccX'].values
    road_conditions = np.random.choice(['dry', 'wet', 'icy'], len(sensor_data))
    nearby_vehicles = np.random.randint(0, 5, len(sensor_data))
    return velocities, distances, road_conditions, nearby_vehicles

# Data Preprocessing
def preprocess_data(velocities, distances, road_conditions, nearby_vehicles):
    road_conditions_encoded = pd.get_dummies(road_conditions)
    data = pd.DataFrame({
        'Velocity': velocities,
        'NearbyVehicles': nearby_vehicles
    })
    data = pd.concat([data, road_conditions_encoded], axis=1)
    return data, distances

# Model Training
def train_model(X_train, y_train, params=None):
    model = RandomForestRegressor(**params) if params else RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# Hyperparameter Tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# Neural Network prediction function
def predict_acceleration(velocity, lagged_velocity, lagged_acceleration, velocity_lag_interaction):
    # Create a feature array from the inputs
    features = np.array([[velocity, lagged_velocity, lagged_acceleration, velocity_lag_interaction]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Create polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    features_poly = poly.fit_transform(scaled_features)

    # Use the model to make predictions
    predicted_acceleration = nn_model.predict(features_poly)

    return predicted_acceleration[0][0]

# Execute the workflow
velocities, distances, road_conditions, nearby_vehicles = collect_data_from_sensor()
X, y = preprocess_data(velocities, distances, road_conditions, nearby_vehicles)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Training with Hyperparameter Tuning
best_params = tune_hyperparameters(X_train, y_train)
model = train_model(X_train, y_train, best_params)
predictions = model.predict(X_test)

# Evaluation
mae_optimized = mean_absolute_error(y_test, predictions)
mse_optimized = mean_squared_error(y_test, predictions)
r2_optimized = r2_score(y_test, predictions)

print(f"Mean Absolute Error (MAE): {mae_optimized}")
print(f"Mean Squared Error (MSE): {mse_optimized}")
print(f"R-squared: {r2_optimized}")


#Feature Importance (for tree-based models)
if isinstance(model, RandomForestRegressor):
    importances = model.feature_importances_
    plt.barh(X.columns, importances)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.show()

#Residual Analysis
residuals = y_test - predictions
plt.scatter(y_test, residuals)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Actual vs. Residuals')
plt.show()

#Training vs. Validation Loss (for neural networks)
history = model.fit(...)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()

#Model Architecture (for neural networks)
print(model.summary())

#Data Distribution
sensor_data[['VelocityX', 'AccX']].hist(figsize=(10, 5))
plt.suptitle('Data Distribution')
plt.show()

#Prediction Distribution
plt.hist(predictions, bins=50, label='Predicted', alpha=0.5)
plt.hist(y_test, bins=50, label='Actual', alpha=0.5)
plt.xlabel('Acceleration Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Prediction Distribution vs. Actual Distribution')
plt.show()

#Comparison to Baseline
baseline_preds = [y_train.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_preds)
print(f"Baseline MAE: {baseline_mae}")

#Actual vs. Predicted Plots
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.show()
