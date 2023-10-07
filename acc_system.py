import numpy as np
import matplotlib.pyplot as plt

# Constants and initial conditions
dt = 0.1
T = 50
time = np.arange(0, T, dt)
ego_pos, ego_vel, ego_acc = [0], [20], [0]
lead_pos, lead_vel, lead_acc = [50], [30], [2]
relative_distance = [lead_pos[0] - ego_pos[0]]
safe_distance = 20  # safe distance in meters
driver_set_speed = 30  # m/s

# Function to calculate acceleration for ego car based on ACC rules
def acc_system_control(ego_velocity, relative_distance, relative_velocity):
    kp, ki, kd = 1.36, 5.42, 0.01
    error = safe_distance - relative_distance
    acc = kp * error - kd * relative_velocity
    if relative_distance > safe_distance:
        acc += (driver_set_speed - ego_velocity) * kp
    return acc

# Simulation loop
for i in range(1, len(time)):
    # Simulating random lead car acceleration and velocity
    if i % 100 == 0:
        lead_acc.append(np.random.normal(0, 2))
    else:
        lead_acc.append(0)
    
    lead_vel.append(lead_vel[-1] + lead_acc[-1]*dt)
    lead_pos.append(lead_pos[-1] + lead_vel[-1]*dt + 0.5*lead_acc[-1]*dt**2)
    
    # ACC decision based on relative distance and relative velocity
    rel_dist = lead_pos[-1] - ego_pos[-1]
    rel_vel = lead_vel[-1] - ego_vel[-1]
    
    ego_acc_cmd = acc_system_control(ego_vel[-1], rel_dist, rel_vel)
    ego_acc.append(ego_acc_cmd)
    
    # Update ego car's velocity and position
    ego_vel.append(ego_vel[-1] + ego_acc[-1]*dt)
    ego_pos.append(ego_pos[-1] + ego_vel[-1]*dt + 0.5*ego_acc[-1]*dt**2)
    
    # Update relative distance list
    relative_distance.append(rel_dist)

# Plotting
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(time, ego_pos, label="Ego Car Position")
plt.plot(time, lead_pos, label="Lead Car Position")
plt.ylabel('Position (m)')
plt.title('Position vs. Time')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(time, ego_vel, label="Ego Car Velocity")
plt.plot(time, lead_vel, label="Lead Car Velocity")
plt.ylabel('Velocity (m/s)')
plt.title('Velocity vs. Time')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(time, ego_acc, label="Ego Car Acceleration")
plt.plot(time, lead_acc, label="Lead Car Acceleration")
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration vs. Time')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(time, relative_distance, label="Relative Distance", color="red")
plt.ylabel('Distance (m)')
plt.xlabel('Time (s)')
plt.title('Distance Between Two Cars vs. Time')
plt.legend()

plt.tight_layout()
plt.show()
