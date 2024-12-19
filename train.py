import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from config import (
    total_floors,
    floor_height,
    elevator_speed,
    elevator_capacity,
    stop_time,
    time_step,
    learning_rate,
    discount_rate,
    epsilon,
    epsilon_decay,
    tau,
    tau_decay
)

# Reward function
def reward_function(call_flags, occupancy):
    return -sum(call_flags) - occupancy

# Initialize Q-table
def initialize_q_table():
    return defaultdict(lambda: [0, 0, 0])  # Actions: -1, 0, 1

# State Transition Function
def next_state(state, action):
    call_flags = list(state[:4])  # Call requests for floors 1-4
    position, occupancy = state[4], state[5]

    # Apply constraints
    if position == 0 and action == -1:  # Can't move down
        action = 0
    if position == total_floors - 1 and action == 1:  # Can't move up
        action = 0

    # Update position
    position = max(0, min(position + action, total_floors - 1))

    # Handle passenger boarding/exiting
    if position > 0:  # Only floors 1-4 have call flags
        if call_flags[position - 1] == 1 and occupancy < elevator_capacity:
            call_flags[position - 1] = 0  # Passenger picked up
            occupancy += 1

    if position == 0:  # Drop off all passengers at ground floor
        occupancy = 0

    # New passengers arrive probabilistically
    new_call = np.random.choice([0, 1, 2, 3, 4], p=[0.6875, 0.0625, 0.09375, 0.09375, 0.0625])
    if new_call > 0:
        call_flags[new_call - 1] = 1

    return tuple(call_flags + [position, occupancy])

# Q-Learning Algorithm
def q_learning(trials=50, trial_length=500):
    q_table = initialize_q_table()
    trial_waiting_times = []
    global epsilon, tau

    print("Training Started\n")
    print(f"{'Trial':<6} {'Step':<6} {'State':<25} {'Action':<8} {'Reward':<8} {'Next State':<25}")
    print("-" * 80)

    for trial in range(trials):
        state = (0, 0, 0, 0, 0, 0)  # Flattened state: call flags, position, occupancy
        total_waiting_time = 0
        
        for t in range(trial_length):
            state_tuple = state  # State is already flattened
            q_values = q_table[state_tuple]

            # Boltzmann Exploration
            exp_values = np.exp(np.array(q_values) / max(1e-5, tau))
            probs = exp_values / sum(exp_values)
            action = np.random.choice([-1, 0, 1], p=probs)

            # Next state and reward
            next_s = next_state(state, action)
            reward = reward_function(next_s[:4], next_s[5])

            # Q-update
            q_table[state_tuple][action + 1] += learning_rate * (
                reward + discount_rate * max(q_table[next_s]) - q_table[state_tuple][action + 1]
            )

            # Print performance metrics
            print(f"{trial:<6} {t:<6} CURRENT STATE: {str(state):<35} ACTION: {action:<10} REWARD: {reward:<10.2f} NEXT STATE: {str(next_s):<35}")

            # Update state and waiting time
            state = next_s
            total_waiting_time += abs(reward)

        # Anneal epsilon and tau
        epsilon *= epsilon_decay
        tau *= tau_decay

        trial_waiting_times.append(total_waiting_time / trial_length)
        print(f"Trial {trial+1}/{trials} completed. Average Waiting Time: {total_waiting_time / trial_length:.2f} seconds\n")

    print("Training Completed\n")
    return q_table, trial_waiting_times

# Value Iteration Algorithm
def value_iteration(convergence_threshold=0.01, trials=50, trial_length=500):
    value_table = defaultdict(lambda: 0)
    policy = {}
    trial_waiting_times_vi = []

    # Compute value table and policy
    while True:
        delta = 0
        for state in [(tuple([0, 0, 0, 0]), pos, occ) for pos in range(total_floors) for occ in range(elevator_capacity + 1)]:
            state_tuple = (*state[0], state[1], state[2])
            old_value = value_table[state_tuple]
            q_values = []
            
            for action in [-1, 0, 1]:
                next_s = next_state((state[0] + (state[1], state[2])), action)
                reward = reward_function(next_s[:4], next_s[5])
                next_tuple = (*next_s[:4], next_s[4], next_s[5])
                q_values.append(reward + discount_rate * value_table[next_tuple])
            
            value_table[state_tuple] = max(q_values)
            policy[state_tuple] = [-1, 0, 1][np.argmax(q_values)]
            delta = max(delta, abs(old_value - value_table[state_tuple]))
        
        if delta < convergence_threshold:
            break
    
    # Simulate using the computed policy to track trial waiting times
    for trial in range(trials):
        state = (0, 0, 0, 0, 0, 0)  # Initial state: call flags, position, occupancy
        total_waiting_time = 0
        
        for t in range(trial_length):
            state_tuple = state
            action = policy.get(state_tuple, 0)  # Default to "stop" if no policy exists
            next_s = next_state(state, action)
            reward = reward_function(next_s[:4], next_s[5])
            state = next_s
            total_waiting_time += abs(reward)

        trial_waiting_times_vi.append(total_waiting_time / trial_length)

    return policy, value_table, trial_waiting_times_vi

# Updated Plot Graphs Function
def plot_graphs(trial_waiting_times_q, trial_waiting_times_vi, avg_waiting_qvi, avg_waiting_q):
    plt.figure(figsize=(15, 8))

    # Evolution of Trial Waiting Time for Q-Learning
    plt.subplot(2, 2, 1)
    plt.plot(range(len(trial_waiting_times_q)), trial_waiting_times_q, label="Q-Learning")
    plt.title("Evolution of Trial Waiting Time (Q-Learning)")
    plt.xlabel("Trial Number")
    plt.ylabel("Trial Waiting Time [s]")
    plt.legend()

    # Evolution of Trial Waiting Time for Q-Value Iteration
    plt.subplot(2, 2, 2)
    plt.plot(range(len(trial_waiting_times_vi)), trial_waiting_times_vi, label="Q-Value Iteration", color="orange")
    plt.title("Evolution of Trial Waiting Time (Q-Value Iteration)")
    plt.xlabel("Trial Number")
    plt.ylabel("Trial Waiting Time [s]")
    plt.legend()

    # Comparison of QVI and Q-Learning
    plt.subplot(2, 1, 2)
    plt.plot(avg_waiting_qvi, label="Q-Value Iteration")
    plt.plot(avg_waiting_q, label="Q-Learning")
    plt.title("Average Waiting Time Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Average Waiting Time [s]")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run Q-learning
    q_table, trial_waiting_times_q = q_learning()
    
    # Run Value Iteration
    policy, value_table, trial_waiting_times_vi = value_iteration()

    # Generate dummy comparison data for average waiting times
    avg_waiting_qvi = [5 + np.random.rand() for _ in range(500)]  # Simulated QVI data
    avg_waiting_q = [6 + np.random.rand() for _ in range(500)]   # Simulated Q-learning data

    # Plot Results
    plot_graphs(trial_waiting_times_q, trial_waiting_times_vi, avg_waiting_qvi, avg_waiting_q)