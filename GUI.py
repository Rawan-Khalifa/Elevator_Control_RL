import tkinter as tk
import numpy as np
import time
from train import q_learning, reward_function



# Parameters
total_floors = 5
floor_height = 6  # in meters
elevator_speed = 3  # m/s
elevator_capacity = 4
stop_time = 2  # seconds
time_step = stop_time / 10  # for Euler integration
learning_rate = 0.38
discount_rate = 0.99
epsilon = 0.8
epsilon_decay = 0.89
tau = 11.8
tau_decay = 0.998

# GUI Simulation Parameters
simulation_duration = 60  # seconds

# Simulate Passenger Arrival (Helper Function)
def simulate_passenger_arrival():
    return np.random.choice([0, 1, 2, 3, 4], p=[0.6875, 0.0625, 0.09375, 0.09375, 0.0625])

# Elevator GUI Class
class ElevatorGUI:
    def __init__(self, master, q_table):
        self.master = master
        self.q_table = q_table
        self.state = (0, 0, 0, 0, 0, 0)  # Initial state: call flags, position, occupancy
        self.time = 0
        self.average_wait_time = 0
        self.passengers = 0
        
        self.setup_gui()
        self.update_simulation()

    def setup_gui(self):
        self.master.title("Elevator System Simulation")
        self.master.geometry("500x600")

        # Time Label
        self.time_label = tk.Label(self.master, text="Time [s]: 0", font=("Helvetica", 16))
        self.time_label.pack()

        # Action Label
        self.action_label = tk.Label(self.master, text="Action: Stop", font=("Helvetica", 16))
        self.action_label.pack()

        # Elevator Panel
        self.canvas = tk.Canvas(self.master, width=200, height=400, bg="white")
        self.canvas.pack()
        self.floor_rectangles = {}
        self.call_flags = {}

        for i in range(total_floors):
            y1 = 400 - (i * 80)
            y2 = y1 - 80
            rect = self.canvas.create_rectangle(50, y1, 150, y2, outline="black", fill="white")
            self.floor_rectangles[i] = rect
            self.call_flags[i] = self.canvas.create_text(170, y1 - 40, text="0", font=("Helvetica", 12))

        # Passengers and Average Wait Time
        self.passengers_label = tk.Label(self.master, text="Passengers(#): 0", font=("Helvetica", 16))
        self.passengers_label.pack()

        self.avg_wait_time_label = tk.Label(self.master, text="Average Waiting Time [s]: 0.0", font=("Helvetica", 16))
        self.avg_wait_time_label.pack()

    def update_simulation(self):
        if self.time >= simulation_duration:
            return
        
        state_tuple = self.state
        q_values = self.q_table[state_tuple]
        action = np.argmax(q_values) - 1  # Best action

        # Determine next state
        next_state = next_state_helper(self.state, action)
        reward = reward_function(next_state[:4], next_state[5])

        # Update metrics
        self.time += 2  # Simulation step of 2 seconds
        self.passengers = next_state[5]
        self.average_wait_time = max(self.average_wait_time + abs(reward) / (self.time + 1), 0)

        # Update labels
        self.time_label.config(text=f"Time [s]: {self.time}")
        self.action_label.config(text=f"Action: {'Up' if action == 1 else 'Down' if action == -1 else 'Stop'}")
        self.passengers_label.config(text=f"Passengers(#): {self.passengers}")
        self.avg_wait_time_label.config(text=f"Average Waiting Time [s]: {self.average_wait_time:.2f}")

        # Update elevator position
        for floor in range(total_floors):
            if floor == next_state[4]:
                self.canvas.itemconfig(self.floor_rectangles[floor], fill="black")  # Elevator position
            else:
                self.canvas.itemconfig(self.floor_rectangles[floor], fill="white")

            # Update call flags
            self.canvas.itemconfig(self.call_flags[floor], text=str(next_state[:4][floor] if floor < 4 else 0))

        self.state = next_state
        self.master.after(500, self.update_simulation)  # Update every 500ms


# Helper function to simulate state transitions for GUI
def next_state_helper(state, action):
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
    new_call = simulate_passenger_arrival()
    if new_call > 0:
        call_flags[new_call - 1] = 1

    return tuple(call_flags + [position, occupancy])


# Run GUI Simulation
if __name__ == "__main__":
    q_table, _ = q_learning()  # Train the model
    root = tk.Tk()
    app = ElevatorGUI(root, q_table)
    root.mainloop()
