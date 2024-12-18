# Elevator System Simulation with Reinforcement Learning

This project simulates an **elevator system** using **Reinforcement Learning (RL)** with two main algorithms:  
1. **Q-Learning** (Model-free)  
2. **Value Iteration** (Model-based)

The simulation visualizes the learned control policy in a **Graphical User Interface (GUI)**. The goal is to minimize passenger waiting time in an elevator system.

---

## Key Features 🚀
- **Q-Learning and Value Iteration**:
   - Implemented RL algorithms to optimize elevator movement.
   - Evolution of trial waiting times and average waiting time comparisons.
- **Graphical Visualization**:
   - A Tkinter-based GUI to simulate the elevator system.
   - Real-time display of elevator position, flagged floors, passenger counts, and simulation time.
- **Performance Plots**:
   - Plots showing the evolution of waiting times over trials.
   - Comparison between Q-Learning and Value Iteration policies.
     
---

## Prerequisites ⚙️

Ensure you have the following installed on your system:
1. Python 3.8+  
2. Required libraries:  
   - `numpy`  
   - `matplotlib`  
   - `tkinter` (built-in with Python)  

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Running the Code ▶️

### Step 1: Train the RL Model
Run the training script to train the elevator system:

```bash
python train.py
```

This will output:
1. **Training logs** showing the evolution of states, actions, and rewards.
2. **Performance Metrics** such as trial waiting times.

---

### Step 2: Run the GUI Simulation
Use the trained Q-table to simulate and visualize the elevator system:

```bash
python GUI_2.py
```

This will:
- Display a **real-time GUI** with the elevator position, flagged floors, and waiting passengers.
- Update metrics dynamically, such as simulation time, current action (Up/Down/Stop), and average waiting times.

---

## GUI Features 🖥️

- **Left Panel**:
   - Simulation time.
   - Current elevator action (Up, Down, Stop).
- **Central Visualization**:
   - Floors with:
     - **Black rectangle**: Current elevator position.
     - **Flags**: Call requests for each floor.
     - **Passenger counts**: Number of waiting passengers and accumulated wait times.
- **Right Panel**:
   - Number of passengers in the elevator.
   - Average waiting time updated in real time.

---

## Customization ✏️

You can modify the following parameters in the script:

- **Number of floors**: `total_floors`  
- **Elevator capacity**: `elevator_capacity`  
- **Learning rate**: `learning_rate`  
- **Exploration settings**: `epsilon` and `tau`  
- **Simulation duration**: Adjust `simulation_duration` in the GUI script.
