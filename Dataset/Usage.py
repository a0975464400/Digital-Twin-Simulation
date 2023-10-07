# Initialize our ACC system
acc = ACC(desired_speed=100, safe_following_distance=20)

# A simple scenario where the car starts slow and far from next car, then gets closer to next car.
scenario = [(50, 25, 0),  # Current speed, distance to next car, road grade
            (60, 30, 0),
            (90, 10, 0),
            (95, 5, 3)]

# Initialize and run our digital twin simulation
digital_twin = DigitalTwin()
actions = digital_twin.simulate_scenario(scenario, acc)

# Get feedback
feedback = digital_twin.feedback_to_real_world(actions)
print(feedback)
