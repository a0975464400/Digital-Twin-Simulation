import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.exceptions import DataConversionWarning
import warnings
import matplotlib.pyplot as plt

# Suppressing Warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# 1. Dataset Generation
np.random.seed(0)
num_samples = 1000
relative_speeds = np.random.uniform(-10, 10, num_samples)
distances = np.random.uniform(5, 100, num_samples)
vehicle_types = np.random.randint(0, 3, num_samples)
road_conditions = np.random.randint(0, 3, num_samples)

X = pd.DataFrame({
    'relative_speeds': relative_speeds,
    'distances': distances,
    'vehicle_types': vehicle_types,
    'road_conditions': road_conditions
})
y = np.where(relative_speeds < -1, 2, np.where(relative_speeds > 1, 1, 0))

# 2. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train.ravel())

# 3. Model Training
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_encoded)



class ACC_DigitalTwin_Refined:
    def __init__(self, sensor_data, rf_model, scaler, label_encoder):
        self.sensor_data = sensor_data
        self.rf_model = rf_model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.current_index = 0
        
        initial_data = self.get_sensor_data()
        
        self.dt = 0.1 
        self.ego_position = 10
        self.lead_position = self.ego_position + initial_data['distances']
        self.ego_velocity = 30  
        self.lead_velocity = self.ego_velocity + initial_data['relative_speeds']
        
        self.driver_set_speed = 30
        self.kp = 0.5
        vehicle_type_map = {0: 1.5, 1: 2, 2: 1}
        self.t_gap = vehicle_type_map[initial_data['vehicle_types']]
        
        self.D_default = 10
        road_condition_map = {0: (-3, 2), 1: (-2, 1), 2: (-1, 0.5)}
        self.amin_ego, self.amax_ego = road_condition_map[initial_data['road_conditions']]
        self.driver_set_speed = 30
        self.mode = "speed_control"

    def get_sensor_data(self):
        feature_columns = ['relative_speeds', 'distances', 'vehicle_types', 'road_conditions']
        if self.current_index < len(self.sensor_data):
            features = self.sensor_data.iloc[self.current_index][feature_columns]
            self.current_index += 1
            return features
        else:
            raise Exception("No more sensor data available.")
    
    def get_rf_prediction(self, features):
        features_df = features.to_frame().T
        scaled_features = self.scaler.transform(features_df)
        prediction = self.rf_model.predict(scaled_features)[0]
        return self.label_encoder.inverse_transform([prediction])[0]

    def ACC_controller(self):
        features = self.get_sensor_data()
        predicted_action = self.get_rf_prediction(features)

        if predicted_action == 'accelerate':
            acceleration = self.amax_ego
        elif predicted_action == 'decelerate':
            acceleration = self.amin_ego
        else:
            acceleration = 0

        acceleration = max(self.amin_ego, min(acceleration, self.amax_ego))
        return acceleration

    def update_states(self, acceleration):
        self.lead_velocity = self.lead_car_velocity()
        self.lead_position += self.lead_velocity * self.dt
        
        self.ego_velocity += acceleration * self.dt
        self.ego_velocity = max(0, self.ego_velocity)
        self.ego_position += self.ego_velocity * self.dt

    def lead_car_velocity(self):
        return self.ego_velocity + 5 * np.sin(0.1 * self.current_index)

    def compute_safety_distance(self):
        return self.D_default + self.t_gap * self.ego_velocity


# Simulating the ACC system
acc_twin_refined = ACC_DigitalTwin_Refined(X_train, rf, scaler, label_encoder)
duration = len(X_train)

ego_positions_refined, ego_velocities_refined, lead_positions_refined, lead_velocities_refined = [], [], [], []
actual_distances, safety_distances = [], []

for _ in range(duration):
    acceleration = acc_twin_refined.ACC_controller()
    acc_twin_refined.update_states(acceleration)
    
    actual_distance = acc_twin_refined.lead_position - acc_twin_refined.ego_position
    safety_distance = acc_twin_refined.compute_safety_distance()
    
    actual_distances.append(actual_distance)
    safety_distances.append(safety_distance)
    
    ego_positions_refined.append(acc_twin_refined.ego_position)
    ego_velocities_refined.append(acc_twin_refined.ego_velocity)
    lead_positions_refined.append(acc_twin_refined.lead_position)
    lead_velocities_refined.append(acc_twin_refined.lead_velocity)

# Model Evaluation
y_pred = rf.predict(X_test_scaled)
classification_output = classification_report(y_test.ravel(), y_pred, target_names=label_encoder.classes_)
accuracy = accuracy_score(y_test.ravel(), y_pred)

print("y_test.ravel():", y_test.ravel())
print("y_pred:", y_pred)
print("label_encoder.classes_:", label_encoder.classes_)

# Now, generate the classification report
classification_output = classification_report(y_test.ravel(), y_pred, target_names=label_encoder.classes_.astype(str))
print(classification_output)

# Plotting the ego vehicle's and lead vehicle's velocities over time
plt.figure(figsize=(14, 6))
plt.plot(ego_velocities_refined, label='Ego Vehicle Velocity', color='blue')
plt.plot(lead_velocities_refined, label='Lead Vehicle Velocity', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Ego and Lead Vehicle Velocities over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the ego vehicle's and lead vehicle's positions over time
plt.figure(figsize=(14, 6))
plt.plot(ego_positions_refined, label='Ego Vehicle Position', color='blue')
plt.plot(lead_positions_refined, label='Lead Vehicle Position', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Ego and Lead Vehicle Positions over Time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(actual_distances, label='Actual Distance', color='blue')
plt.plot(safety_distances, label='Safety Distance', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Actual and Safety Distances over Time')
plt.legend()
plt.grid(True)
plt.show()




import numpy as np

# Simple Neural Network definition
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)
    
    def predict(self, x):
        hidden_layer = self.relu(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output_layer = self.softmax(np.dot(hidden_layer, self.weights_hidden_output) + self.bias_output)
        return np.argmax(output_layer)

# ACC Digital Twin with Neural Network definition
class ACCDigitalTwinWithNN:
    def __init__(self, neural_network):
        self.vehicle_speed = 20
        self.acceleration = 0
        self.neural_network = neural_network
        self.results = []

    def acc_logic(self, prediction):
        if prediction == 0:
            self.acceleration = 0
        elif prediction == 1:
            self.acceleration = 1
        elif prediction == 2:
            self.acceleration = -1

    def update(self, sensor_data):
        prediction = self.neural_network.predict(sensor_data)
        self.acc_logic(prediction)
        self.vehicle_speed += self.acceleration
        self.results.append(f"Vehicle Speed: {self.vehicle_speed} m/s")

    def run_simulation(self, sensor_data_samples):
        for sensor_data in sensor_data_samples.values:
            self.update(sensor_data)
    
    def print_results(self):
        for idx, result in enumerate(self.results, 1):
            print(f"Step {idx}: {result}")

# Initialize and run the simulation
mock_nn = SimpleNeuralNetwork(input_size=4, hidden_size=5, output_size=3)
digital_twin_nn = ACCDigitalTwinWithNN(mock_nn)
digital_twin_nn.run_simulation(df1)


# Sample sensor data
sample_sensor_data = [
    [2, 30, 0, 0],
    [1, 25, 0, 1],
    [0, 20, 1, 0],
    [-1, 15, 2, 2],
    [-2, 10, 0, 0]
]
sensor_data_test= sensor_data.drop(columns=['acc_action'])
sensor_data_array= sensor_data_test.values

# Initialize and run the simulation
mock_nn = SimpleNeuralNetwork(input_size=4, hidden_size=5, output_size=3)
digital_twin_nn = ACCDigitalTwinWithNN(mock_nn)
digital_twin_nn.run_simulation(df1)
digital_twin_nn.print_results()
pd.set_option('display.max_rows', None)

# Plotting the ego vehicle's and lead vehicle's velocities over time
plt.figure(figsize=(14, 6))
plt.plot(ego_velocities_refined, label='Ego Vehicle Velocity', color='blue')
plt.plot(lead_velocities_refined, label='Lead Vehicle Velocity', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Ego and Lead Vehicle Velocities over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the ego vehicle's and lead vehicle's positions over time
plt.figure(figsize=(14, 6))
plt.plot(ego_positions_refined, label='Ego Vehicle Position', color='blue')
plt.plot(lead_positions_refined, label='Lead Vehicle Position', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Ego and Lead Vehicle Positions over Time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(actual_distances, label='Actual Distance', color='blue')
plt.plot(safety_distances, label='Safety Distance', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Actual and Safety Distances over Time')
plt.legend()
plt.grid(True)
plt.show()