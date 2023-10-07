import numpy as np
import pandas as pd

# 1. Synthetic Data Generation
np.random.seed(0)
num_samples = 1000
relative_speeds = np.random.uniform(-10, 10, num_samples)
distances = np.random.uniform(5, 100, num_samples)
vehicle_types = np.random.randint(0, 3, num_samples)
road_conditions = np.random.randint(0, 3, num_samples)
X = np.vstack((relative_speeds, distances, vehicle_types, road_conditions)).T
y = np.where(relative_speeds < -1, 2, np.where(relative_speeds > 1, 1, 0))


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros(output_size)
        self.learning_rate = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        z = x - np.max(x, axis=1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            hidden_layer = self.relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
            output_layer = self.softmax(np.dot(hidden_layer, self.weights_hidden_output) + self.bias_output)
            loss = -np.sum(y * np.log(output_layer))
            output_error = output_layer - y
            hidden_error = output_error.dot(self.weights_hidden_output.T) * self.relu_derivative(hidden_layer)
            d_weights_output = hidden_layer.T.dot(output_error)
            d_bias_output = np.sum(output_error, axis=0, keepdims=True)
            d_weights_input = X.T.dot(hidden_error)
            d_bias_input = np.sum(hidden_error, axis=0, keepdims=True)
            self.weights_hidden_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output.sum(axis=0)
            self.weights_input_hidden -= self.learning_rate * d_weights_input
            self.bias_hidden -= self.learning_rate * d_bias_input.sum(axis=0)

    def predict(self, x):
        hidden_layer = self.relu(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        output_layer = self.softmax(np.dot(hidden_layer, self.weights_hidden_output) + self.bias_output)
        return np.argmax(output_layer, axis=1)


class ModifiedACCDigitalTwinWithNN:
    def __init__(self, neural_network, v_set=30, t_gap=1.4, D_default=10):
        self.vehicle_speed = 20
        self.acceleration = 0
        self.neural_network = neural_network
        self.speed_history = []
        self.v_set = v_set  # driver-set velocity
        self.t_gap = t_gap  # safe time gap
        self.D_default = D_default  # default safe distance

    def calculate_safe_distance(self):
        """Calculate the safe following distance based on current speed."""
        return self.vehicle_speed * self.t_gap + self.D_default

    def acc_logic(self, prediction, relative_distance):
        D_safe = self.calculate_safe_distance()
        if relative_distance >= D_safe:
            if self.vehicle_speed < self.v_set:
                self.acceleration = 1
            else:
                self.acceleration = 0
        else:
            if prediction == 0:
                self.acceleration = 0
            elif prediction == 1:
                self.acceleration = 1
            elif prediction == 2:
                self.acceleration = -1

    def update(self, sensor_data):
        relative_distance = sensor_data[1]
        prediction = self.neural_network.predict(sensor_data.reshape(1, -1))
        self.acc_logic(prediction, relative_distance)
        self.vehicle_speed += self.acceleration
        self.speed_history.append(self.vehicle_speed)

    def run_simulation(self, sensor_data_samples):
        for sensor_data in sensor_data_samples:
            self.update(sensor_data)


# 4. Training and Simulation
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1
mock_nn = SimpleNeuralNetwork(input_size=4, hidden_size=5, output_size=3)
mock_nn.train(X, y_one_hot, epochs=1000)
digital_twin_nn = ModifiedACCDigitalTwinWithNN(mock_nn)
digital_twin_nn.run_simulation(X)

# Display Results
print(digital_twin_nn.speed_history[:100])

# Saving the data
df1 = pd.DataFrame(X, columns=['relative_speeds', 'distances', 'vehicle_types', 'road_conditions'])
df1.to_csv("sensor_data_123.csv", index=False)
