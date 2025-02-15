# Reinforcement-Learning-Based-Vehicle-Pathfinding-Simulation
## Description
A pathfinding simulation using reinforcement learning, where a car navigates a grid to reach a goal while avoiding obstacles like barriers and cones.

## Technologies Used
- Python 3.9
- Pygame
- Numpy
- Matplotlib
- Pandas

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/graduation-project.git
2. **Install the Required Dependencies:**
   ```bash
   pip install pygame numpy matplotlib pandas

3. **Run the Main Script:**
   ```bash
   python main.py
## How It Works
This project implements a reinforcement learning-based agent that learns to navigate a grid with obstacles. The car uses Q-learning to explore the environment, gradually improving its decision-making by learning from rewards and penalties.

Car: The agent that moves within the grid.
Goal: The target that the agent must reach.
Obstacles: Barriers and cones that the car must avoid.

## Key Features
- Q-learning Algorithm: Used to learn the best path to the goal.
- Pygame Visualization: Real-time graphical representation of the agent's movement.
- Reward System: The agent receives rewards or penalties based on its actions.
## Q-Table Heatmap
Below is the Q-table heatmap, which shows the learned Q-values for each state-action pair and indicates the agent's learned policy.
![Q-Table Heatmap](https://github.com/user-attachments/assets/ea0e6913-8f9d-40de-869f-5abc77180e30)

## Reward Table
The reward table displays the total reward accumulated by the agent over each episode and shows how it improves over time.
![Figure_1](https://github.com/user-attachments/assets/54dfa628-151a-4d9f-863f-7065d6d41b34)

