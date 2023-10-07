import pandas as pd
import random

# Number of data points
num_data = 1000

# Generating random data for demonstration
speeds = [random.randint(40, 100) for _ in range(num_data)]
distances = [random.randint(5, 100) for _ in range(num_data)]
road_grades = [round(random.uniform(-0.3, 0.3), 2) for _ in range(num_data)]
actions = [random.choice(['maintain', 'accelerate', 'decelerate']) for _ in range(num_data)]

# Creating a dataframe
df = pd.DataFrame({
    'speed': speeds,
    'distance_to_next_car': distances,
    'road_grade': road_grades,
    'acc_action': actions
})

# Saving to CSV
df.to_csv('sensor_data.csv', index=False)
