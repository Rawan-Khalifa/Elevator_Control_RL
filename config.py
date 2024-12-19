# Parameters for the Elevator RL Simulation

# Environment Parameters
total_floors = 5
floor_height = 6  # in meters
elevator_speed = 3  # m/s
elevator_capacity = 4
stop_time = 2  # seconds
time_step = stop_time / 10  # for Euler integration

# Q-Learning Parameters
learning_rate = 0.38
discount_rate = 0.99
epsilon = 0.8
epsilon_decay = 0.89
tau = 11.8
tau_decay = 0.998
