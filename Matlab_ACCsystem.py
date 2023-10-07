import numpy as np
import matplotlib.pyplot as plt

class ACC_DigitalTwin_Refined:
    def __init__(self):
        self.current_time = 0
        
        self.ego_position = 10  # Initial position for ego car
        self.ego_velocity = 20  # Initial velocity for ego car
        
        self.lead_position = 50  # Initial position for lead car
        self.lead_velocity = 25  # Initial velocity for lead car
        
        self.driver_set_speed = 30
        self.kp = 0.5
        self.t_gap = 1.5
        self.D_default = 10
        self.amin_ego = -3
        self.amax_ego = 2
        self.dt = 0.1  # Time step

    def lead_car_velocity(self, time):
        # Using a sinusoidal function to vary the lead car's velocity
        return 25 + 5 * np.sin(0.1 * time)

# Inside ACC_DigitalTwin_Refined class

def ACC_controller(self):
    safety_distance = self.D_default + self.ego_velocity * self.t_gap
    actual_distance = self.lead_position - self.ego_position
    distance_error = safety_distance - actual_distance
    
    if actual_distance > safety_distance:
        desired_acceleration = self.kp * (self.driver_set_speed - self.ego_velocity)  # Speed Control
    else:
        desired_acceleration = self.kp * distance_error  # Distance Control

    # Ensure the acceleration is within the limits
    return np.clip(desired_acceleration, self.amin_ego, self.amax_ego)


    def update_states(self, acceleration, time):
        self.lead_velocity = self.lead_car_velocity(time)
        self.lead_position += self.lead_velocity * self.dt
        
        self.ego_velocity += acceleration * self.dt
        self.ego_velocity = max(0, self.ego_velocity)  # Ensure velocity is non-negative
        self.ego_position += self.ego_velocity * self.dt

# Initialize the refined ACC digital twin
acc_twin_refined = ACC_DigitalTwin_Refined()

# Simulate the ACC system over the duration
duration = 80  # seconds
t = np.arange(0, duration, acc_twin_refined.dt)

ego_positions_refined, ego_velocities_refined = [], []
lead_velocities_refined, lead_positions_refined = [], []
actual_distances_refined, safety_distances_refined = [], []

for current_time in t:
    acceleration = acc_twin_refined.ACC_controller()
    acc_twin_refined.update_states(acceleration, current_time)

    ego_positions_refined.append(acc_twin_refined.ego_position)
    ego_velocities_refined.append(acc_twin_refined.ego_velocity)
    
    lead_positions_refined.append(acc_twin_refined.lead_position)
    lead_velocities_refined.append(acc_twin_refined.lead_velocity)
    
    actual_distance = acc_twin_refined.lead_position - acc_twin_refined.ego_position
    safety_distance = acc_twin_refined.D_default + acc_twin_refined.ego_velocity * acc_twin_refined.t_gap
    
    actual_distances_refined.append(actual_distance)
    safety_distances_refined.append(safety_distance)
    
 # Debugging print statements
print(f"Time: {current_time}, Lead Velocity: {acc_twin_refined.lead_velocity}, Ego Velocity: {acc_twin_refined.ego_velocity}, Actual Distance: {actual_distance}")


# Plotting the refined simulation results
plt.figure(figsize=(18, 6))

# ... (rest of the plotting code remains unchanged)


# Velocity-Time Plot
plt.subplot(1, 3, 1)
plt.plot(t, ego_velocities_refined, label='Ego Vehicle Velocity', color='blue')
plt.plot(t, lead_velocities_refined, label='Lead Vehicle Velocity', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity Over Time')
plt.grid(True)
plt.legend()

# Acceleration-Time Plot
ego_accelerations_refined = [0] + [(ego_velocities_refined[i+1] - ego_velocities_refined[i]) / acc_twin_refined.dt for i in range(len(ego_velocities_refined)-1)]
lead_accelerations_refined = [0] + [(lead_velocities_refined[i+1] - lead_velocities_refined[i]) / acc_twin_refined.dt for i in range(len(lead_velocities_refined)-1)]
plt.subplot(1, 3, 2)
plt.plot(t, ego_accelerations_refined, label='Ego Vehicle Acceleration', color='blue')
plt.plot(t, lead_accelerations_refined, label='Lead Vehicle Acceleration', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration Over Time')
plt.grid(True)
plt.legend()

# Distance-Time Plot
plt.subplot(1, 3, 3)
plt.plot(t, actual_distances_refined, label='Actual Distance', color='green')
plt.plot(t, safety_distances_refined, label='Safety Distance', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
