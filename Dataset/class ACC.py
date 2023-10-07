class SensorData:
    """Represents data coming from various sensors."""
    
    def __init__(self, current_speed, distance_to_next_car, road_grade):
        self.current_speed = current_speed
        self.distance_to_next_car = distance_to_next_car
        self.road_grade = road_grade  # Inclination of the road, could affect speed decisions


class ACC:
    """Adaptive Cruise Control system."""
    
    def __init__(self, desired_speed, safe_following_distance):
        self.desired_speed = desired_speed
        self.safe_following_distance = safe_following_distance

    def control_logic(self, sensor_data):
        """Decides the control action based on sensor data."""
        
        # If we're going slower than desired and there's enough space ahead, accelerate
        if (sensor_data.current_speed < self.desired_speed and 
            sensor_data.distance_to_next_car > self.safe_following_distance):
            return "Accelerate"
        
        # If we're too close to the next car, decelerate
        elif sensor_data.distance_to_next_car < self.safe_following_distance:
            return "Decelerate"
        
        # If we're going uphill and the speed drops, we might want to accelerate
        elif sensor_data.road_grade > 5 and sensor_data.current_speed < self.desired_speed:
            return "Accelerate uphill"
        
        # If none of the above conditions are met, maintain the current speed
        else:
            return "Maintain speed"


class DigitalTwin:
    """A digital representation of the real-world ACC system."""

    def __init__(self):
        self.simulated_data = []

    def simulate_scenario(self, scenario, acc_system):
        """Simulate a driving scenario and get ACC decisions."""
        actions = []
        for data_point in scenario:
            sensor_data = SensorData(*data_point)
            action = acc_system.control_logic(sensor_data)
            actions.append(action)
        return actions

    def feedback_to_real_world(self, actions):
        """Here, we can provide feedback from our digital twin's simulations to real-world applications."""
        # This can be metrics, alerts, or any data-driven recommendations.
        return actions


