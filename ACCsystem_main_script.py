import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Data Generation
def generate_data():
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
    
    return X, y

# Neural Network Training
def train_nn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nn = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
    nn.fit(X_train, y_train)
    accuracy = nn.score(X_test, y_test)
    print(f"Neural Network Accuracy: {accuracy * 100:.2f}%")
    return nn

# ACC Digital Twin
class ACCDigitalTwin:
    def __init__(self, neural_network, ego_initial_speed=20, lead_initial_speed=25):
        self.nn = neural_network
        self.ego_velocity = ego_initial_speed
        self.lead_velocity = lead_initial_speed
        self.distance = np.random.uniform(20, 40)
        self.ego_speeds = []
        self.lead_speeds = []
        self.distances = []
        self.ego_accelerations = []
        self.lead_accelerations = []
        self.safety_distances = []

    def compute_safety_distance(self):
        D_default = 10
        time_gap = 1.5
        return D_default + time_gap * self.ego_velocity

    def update(self, relative_speed, distance, vehicle_type, road_condition):
        action = self.nn.predict(np.array([[relative_speed, distance, vehicle_type, road_condition]]))[0]
        if action == 0:  
            ego_acceleration = 0
        elif action == 1:  
            ego_acceleration = 1
        else:  
            ego_acceleration = -1
        
        lead_acceleration = np.random.uniform(-1, 1)
        
        self.ego_velocity += ego_acceleration
        self.lead_velocity += lead_acceleration
        self.ego_velocity = max(0, self.ego_velocity)
        self.lead_velocity = max(0, self.lead_velocity)
        
        self.distance += (self.lead_velocity - self.ego_velocity)
        
        self.ego_speeds.append(self.ego_velocity)
        self.lead_speeds.append(self.lead_velocity)
        self.distances.append(self.distance)
        self.ego_accelerations.append(ego_acceleration)
        self.lead_accelerations.append(lead_acceleration)
        self.safety_distances.append(self.compute_safety_distance())

    def run_simulation(self, num_steps=1000):
        for _ in range(num_steps):
            relative_speed = self.lead_velocity - self.ego_velocity
            distance = self.distance
            vehicle_type = np.random.randint(0, 3)
            road_condition = np.random.randint(0, 3)
            self.update(relative_speed, distance, vehicle_type, road_condition)

# Main Execution
X, y = generate_data()
nn = train_nn(X, y)
acc_twin = ACCDigitalTwin(nn, ego_initial_speed=22, lead_initial_speed=28)
acc_twin.run_simulation(num_steps=100)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(acc_twin.ego_speeds, label='Ego Vehicle Speed', color='blue')
plt.plot(acc_twin.lead_speeds, label='Lead Vehicle Speed', color='red')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Speed (m/s)')
plt.title('Vehicle Speeds over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(acc_twin.distances, label='Distance between Vehicles', color='blue')
plt.plot(acc_twin.safety_distances, label='Safety Distance', color='red', linestyle='--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Distance (m)')
plt.title('Distance between Vehicles over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(acc_twin.ego_accelerations, label='Ego Vehicle Acceleration', color='blue', linewidth=2.0)
plt.plot(acc_twin.lead_accelerations, label='Lead Vehicle Acceleration', color='red', linestyle='--', linewidth=2.0)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration Profiles of Ego and Lead Vehicles over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
