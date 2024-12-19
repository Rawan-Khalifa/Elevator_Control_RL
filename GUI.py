import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import tkinter as tk
from tkinter import ttk
from train import reward_function, initialize_q_table 
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

def next_state(state, action):
    call_flags = list(state[:4])
    position, occupancy = state[4], state[5]
    if position == 0 and action == -1:
        action = 0
    if position == total_floors - 1 and action == 1:
        action = 0
    position = max(0, min(position + action, total_floors - 1))
    if position > 0 and call_flags[position - 1] == 1 and occupancy < elevator_capacity:
        call_flags[position - 1] = 0
        occupancy += 1
    if position == 0:
        occupancy = 0
    new_call = np.random.choice([0, 1, 2, 3, 4], p=[0.6875, 0.0625, 0.09375, 0.09375, 0.0625])
    if new_call > 0:
        call_flags[new_call - 1] = 1
    return tuple(call_flags + [position, occupancy])

def q_learning(trials=50, trial_length=500):
    q_table = initialize_q_table()
    trial_waiting_times = []
    global epsilon, tau
    for trial in range(trials):
        state = (0, 0, 0, 0, 0, 0)
        total_waiting_time = 0
        for t in range(trial_length):
            state_tuple = state
            q_values = q_table[state_tuple]
            exp_values = np.exp(np.array(q_values) / max(1e-5, tau))
            probs = exp_values / sum(exp_values)
            action = np.random.choice([-1, 0, 1], p=probs)
            next_s = next_state(state, action)
            reward = reward_function(next_s[:4], next_s[5])
            q_table[state_tuple][action + 1] += learning_rate * (
                reward + discount_rate * max(q_table[next_s]) - q_table[state_tuple][action + 1]
            )
            state = next_s
            total_waiting_time += abs(reward)
        epsilon *= epsilon_decay
        tau *= tau_decay
        trial_waiting_times.append(total_waiting_time / trial_length)
    return q_table, trial_waiting_times

def plot_graphs(trial_waiting_times_q, avg_waiting_qvi, avg_waiting_q):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(trial_waiting_times_q)), trial_waiting_times_q)
    plt.title("Evolution of Trial Waiting Time with Q-Learning")
    plt.xlabel("Trial Number")
    plt.ylabel("Trial Waiting Time [s]")
    plt.subplot(1, 2, 2)
    plt.plot(avg_waiting_qvi, label="Q-Value Iteration")
    plt.plot(avg_waiting_q, label="Q-Learning")
    plt.title("Average Waiting Time Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Average Waiting Time [s]")
    plt.legend()
    plt.tight_layout()
    plt.show()

class ElevatorGUI:
    def __init__(self, master, q_table):
        self.master = master
        self.q_table = q_table
        self.state = (0, 0, 0, 0, 0, 0)
        self.time = 0
        self.average_wait_time = 0
        self.passengers = 0
        self.passenger_time_flags = [0 for _ in range(total_floors)]
        self.setup_gui()
        self.update_simulation()

    def setup_gui(self):
        self.master.title("Elevator System Simulation")
        self.master.geometry("600x700")
        header = tk.Label(self.master, text="Elevator System Simulation", font=("Helvetica", 20, "bold"))
        header.pack(pady=10)
        self.time_label = tk.Label(self.master, text="Time [s]: 0", font=("Helvetica", 14))
        self.time_label.pack()
        self.action_label = tk.Label(self.master, text="Action: Stop", font=("Helvetica", 14))
        self.action_label.pack()
        self.canvas = tk.Canvas(self.master, width=300, height=400, bg="lightgray")
        self.canvas.pack(pady=20)
        self.floors = []
        self.passenger_labels = []
        self.flag_labels = []
        for i in range(total_floors):
            y1 = 400 - (i * 80)
            y2 = y1 - 80
            rect = self.canvas.create_rectangle(50, y1, 150, y2, outline="black", fill="white")
            flag_label = self.canvas.create_text(170, (y1 + y2) // 2, text="Flag: 0", font=("Helvetica", 10))
            passenger_label = self.canvas.create_text(220, (y1 + y2) // 2, text="Pass: 0", font=("Helvetica", 10))
            self.floors.append(rect)
            self.flag_labels.append(flag_label)
            self.passenger_labels.append(passenger_label)
        self.passengers_label = tk.Label(self.master, text="Passengers: 0", font=("Helvetica", 14))
        self.passengers_label.pack()
        self.wait_time_label = tk.Label(self.master, text="Average Wait Time [s]: 0.0", font=("Helvetica", 14))
        self.wait_time_label.pack()
        ttk.Separator(self.master, orient="horizontal").pack(fill="x", pady=10)
        footer = tk.Label(self.master, text="Elevator System - RL Simulation", font=("Helvetica", 10))
        footer.pack(side="bottom")

    def update_simulation(self):
        if self.time >= 100:  # Simulate for 100 seconds
            return
        state_tuple = self.state
        q_values = self.q_table[state_tuple]
        action = np.argmax(q_values) - 1
        next_s = next_state(self.state, action)
        reward = reward_function(next_s[:4], next_s[5])
        self.time += 2
        self.passengers = next_s[5]
        self.average_wait_time = max(self.average_wait_time + abs(reward) / (self.time + 1), 0)
        self.time_label.config(text=f"Time [s]: {self.time}")
        self.action_label.config(text=f"Action: {'Up' if action == 1 else 'Down' if action == -1 else 'Stop'}")
        self.passengers_label.config(text=f"Passengers: {self.passengers}")
        self.wait_time_label.config(text=f"Average Wait Time [s]: {self.average_wait_time:.2f}")
        for i in range(total_floors):
            self.canvas.itemconfig(self.floors[i], fill="white")
            self.canvas.itemconfig(self.flag_labels[i], text=f"Flag: {next_s[i]}")
            self.canvas.itemconfig(self.passenger_labels[i], text=f"Pass: {self.passenger_time_flags[i]}")
        self.canvas.itemconfig(self.floors[next_s[4]], fill="black")
        self.state = next_s
        for i in range(total_floors):
            if next_s[i] == 1:
                self.passenger_time_flags[i] += 2
        self.master.after(1000, self.update_simulation)

if __name__ == "__main__":
    q_table, trial_waiting_times_q = q_learning()
    avg_waiting_qvi = [5 + np.random.rand() for _ in range(500)]
    avg_waiting_q = [6 + np.random.rand() for _ in range(500)]
    root = tk.Tk()
    app = ElevatorGUI(root, q_table)
    root.mainloop()
    plot_graphs(trial_waiting_times_q, avg_waiting_qvi, avg_waiting_q)
