import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('/Users/a0975464400/Desktop/Essay/Project/Dataset/sensor_data.csv')


# Features and target variable
X = data[['speed', 'distance_to_next_car', 'road_grade']]
y = data['acc_action']

# Encode the categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf.predict(X_test_scaled)

# Evaluation
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model, scaler, and label encoder for future use
dump(rf, 'random_forest_model.joblib')
dump(scaler, 'rf_scaler.joblib')
dump(label_encoder, 'label_encoder.joblib')
