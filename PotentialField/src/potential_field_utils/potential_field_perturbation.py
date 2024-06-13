import numpy as np

# Constants
K_ATTRACT = 1.0  # Attractive force gain
K_REPEL = 100.0  # Repulsive force gain
THRESHOLD = 1.0  # Threshold distance for repulsive force
MAX_VELOCITY = 1.0 # Maximum velocity for the robot
RANDOM_PERTURBATION = 0.1  # Random perturbation factor
DT = 0.1  # Time step
MAX_ANGULAR_VELOCITY=0.5  # Maximum angular velocity
ANGLE_GAIN = 1.0  # Gain for angular velocity

# Define functions for attractive and repulsive forces
def attractive_force(robot_pos, goal_pos):
    force = K_ATTRACT * (goal_pos - robot_pos)
    return force

def repulsive_force(robot_pos, obstacle_pos):
    force = np.zeros(2)
    for obs in obstacle_pos:
        distance = np.linalg.norm(robot_pos - obs)
        if distance < THRESHOLD:
            repulsion = K_REPEL * (1.0 / distance - 1.0 / THRESHOLD) * (1.0 / (distance**2)) * (robot_pos - obs) / distance
            force += repulsion
    return force

def compute_total_force(robot_pos, goal_pos, obstacle_pos):
    F_attr = attractive_force(robot_pos, goal_pos)
    F_repl = repulsive_force(robot_pos, obstacle_pos)
    F_total = F_attr + F_repl
    return F_total

def compute_velocity(robot_pos, goal_pos, obstacle_pos, robot_orientation, dt=DT, angle_gain=ANGLE_GAIN):
        # Calculate desired direction
        F_total = compute_total_force(robot_pos, goal_pos, obstacle_pos)
        desired_direction = np.arctan2(F_total[1], F_total[0])
        
        # Calculate angular velocity
        angle_diff = desired_direction - robot_orientation
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize the angle to the range [-pi, pi]
        angular_velocity = angle_gain * angle_diff

        # Add random perturbation to avoid local minima
        perturbation = RANDOM_PERTURBATION * (np.random.rand(2) - 0.5)
        F_total += perturbation

        velocity_x = F_total[0]
        velocity_y = F_total[1]
        
        return velocity_x, velocity_y, angular_velocity

def limit_angular_velocity(angular_velocity, max_ang_vel):
        if abs(angular_velocity) > max_ang_vel:
            angular_velocity = np.sign(angular_velocity) * max_ang_vel
        return angular_velocity

def limit_velocity(velocity_x, velocity_y, max_velocity):
    speed = np.linalg.norm([velocity_x, velocity_y])
    if speed > max_velocity:
        scale = max_velocity / speed
        velocity_x *= scale
        velocity_y *= scale
    return velocity_x, velocity_y

def get_velocity(robot_x, robot_y, goal_x, goal_y, obstacle_pos, robot_orientation=0.0, max_velocity=MAX_VELOCITY, max_ang_vel=MAX_ANGULAR_VELOCITY):
    robot_pos = np.array([robot_x, robot_y])
    goal_pos = np.array([goal_x, goal_y])
    velocity_x, velocity_y, angular_velocity = compute_velocity(robot_pos, goal_pos, obstacle_pos, robot_orientation)
    lim_vx, lim_vy = limit_velocity(velocity_x, velocity_y, max_velocity)
    limit_angular_vel = limit_angular_velocity(angular_velocity, max_ang_vel)
    return lim_vx, lim_vy, limit_angular_vel


