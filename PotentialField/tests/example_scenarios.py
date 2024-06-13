import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from potential_field_utils.potential_field_perturbation import get_velocity
import matplotlib.pyplot as plt
import numpy as np

# Simulation function
def simulate(robot_pos, goal_pos, obstacle_pos, dt=0.1, max_steps=1000):
    trajectory = [robot_pos.copy()]
    
    for _ in range(max_steps):

        # Calculate velocities
        velocity_x, velocity_y, ang_vel = get_velocity(robot_pos[0], robot_pos[1], goal_pos[0], goal_pos[1], obstacle_pos) 

        # Update robot position
        robot_pos[0] += velocity_x * dt
        robot_pos[1] += velocity_y * dt
        trajectory.append(robot_pos.copy())
        
        # Check if the robot reached the goal
        if np.linalg.norm(robot_pos - goal_pos) < 0.1:
            break
    
    return np.array(trajectory)


# Plotting function
def plot_trajectory(trajectory, goal_pos, obstacle_pos, scenario_name):
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
    plt.plot(goal_pos[0], goal_pos[1], 'ro', label='Goal')
    for obs in obstacle_pos:
        plt.plot(obs[0], obs[1], 'bx', label='Obstacle')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Potential Field Planning - {scenario_name}')
    plt.grid()
    plt.show()


# Test scenarios
scenarios = [
    {
        'name': 'Scenario 1 - Simple Path',
        'robot_pos': np.array([0.0, 0.0]),
        'goal_pos': np.array([5.0, 5.0]),
        'obstacle_pos': [np.array([2.5, 2.5])]
    },
    {
        'name': 'Scenario 2 - Multiple Obstacles',
        'robot_pos': np.array([0.0, 0.0]),
        'goal_pos': np.array([6.0, 6.0]),
        'obstacle_pos': [np.array([2.0, 2.0]), np.array([4.0, 4.0])]
    },
    {
        'name': 'Scenario 3 - Dense Obstacles',
        'robot_pos': np.array([0.0, 0.0]),
        'goal_pos': np.array([5.0, 5.0]),
        'obstacle_pos': [np.array([2.0, 2.0]), np.array([2.5, 2.5]), np.array([3.0, 3.0])]
    },
    {
        'name': 'Scenario 4 - Goal Near Obstacle',
        'robot_pos': np.array([0.0, 0.0]),
        'goal_pos': np.array([5.0, 5.0]),
        'obstacle_pos': [np.array([4.5, 4.5])]
    },
    {
        'name': 'Scenario 5 - Goal Middle Obstacle',
        'robot_pos': np.array([0.0, 10.0]),
        'goal_pos': np.array([0.0, 0.0]),
        'obstacle_pos': [np.array([-5, 5]), np.array([5, 5])]
    },
    {
        'name': 'Scenario 6 - Goal Middle Obstacle',
        'robot_pos': np.array([0.0, 10.0]),
        'goal_pos': np.array([0.0, 0.0]),
        'obstacle_pos': [np.array([-1, 1]), np.array([1, 1]), np.array([-2, 2]), np.array([2, 2]),np.array([-5, 5]), np.array([5, 5])]
    },
    {
        'name': 'Scenario 7 - Hard Setting Obstacle - FAILS',
        'robot_pos': np.array([0.0, 10.0]),
        'goal_pos': np.array([0.0, 0.0]),
        'obstacle_pos': [np.array([-.3, .3]), np.array([0, 3]), np.array([0, 2]), np.array([0, 1]), np.array([-1, 1]), np.array([1, 1]), np.array([-2, 2]), np.array([2, 2]),np.array([-5, 5]), np.array([5, 5])]
    },
        {
        'name': 'Scenario 8 - Different Format Obstacle',
        'robot_pos': np.array([0.0, 10.0]),
        'goal_pos': np.array([0.0, 0.0]),
        'obstacle_pos': [[0, 2],[3, 3], [5, 5]]
    },
]

for scenario in scenarios:
    robot_pos = scenario['robot_pos']
    goal_pos = scenario['goal_pos']
    obstacle_pos = scenario['obstacle_pos']
    trajectory = simulate(robot_pos, goal_pos, obstacle_pos)
    plot_trajectory(trajectory, goal_pos, obstacle_pos, scenario['name'])
    
