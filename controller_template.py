"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time

import numpy as np
import cv2
import pyrealsense2 as rs
import numpy as np
import random

CONNECT_SERVER = False  # False for local tests, True for deployment


# ----------- DO NOT CHANGE THIS PART -----------

# The deploy.py script runs on the Jetson Nano at IP 192.168.123.14
# and listens on port 9292
# whereas this script runs on one of the two other Go1's Jetson Nano

SERVER_IP = "192.168.123.14"
SERVER_PORT = 9292

# Maximum duration of the task (seconds):
TIMEOUT = 180

# Minimum control loop duration:
MIN_LOOP_DURATION = 0.1


# Use this function to send commands to the robot:
def send(sock, x, y, r):
    """
    Send a command to the robot.

    :param sock: TCP socket
    :param x: forward velocity (between -1 and 1)
    :param y: side velocity (between -1 and 1)
    :param r: yaw rate (between -1 and 1)
    """
    data = struct.pack('<hfff', code, x, y, r)
    if sock is not None:
        sock.sendall(data)


# Fisheye camera (distortion_model: narrow_stereo):

image_width = 640
image_height = 480

# --------- CHANGE THIS PART (optional) ---------

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Could not find a depth camera with color sensor")
    exit(0)

# Depht available FPS: up to 90Hz
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
# RGB available FPS: 30Hz
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
# # Accelerometer available FPS: {63, 250}Hz
# config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
# # Gyroscope available FPS: {200,400}Hz
# config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# Start streaming
profile = pipeline.start(config)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0, 0, 1]])
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

marker_length = 0.147


# ----------- DO NOT CHANGE THIS PART -----------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.markerBorderBits = 1

RECORD = False
history = []

# ----------------- CONTROLLER -----------------

try:
    # We create a TCP socket to talk to the Jetson at IP 192.168.123.14, which runs our walking policy:

    print("Client connecting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        if CONNECT_SERVER:
            s.connect((SERVER_IP, SERVER_PORT))
            print("Connected.")
        else:
            s = None

        code = 1  # 1 for velocity commands

        task_complete = False
        start_time = time.time()
        previous_time_stamp = start_time

        # main control loop:
        while not task_complete and not time.time() - start_time > TIMEOUT:

            # avoid busy loops:
            now = time.time()
            if now - previous_time_stamp < MIN_LOOP_DURATION:
                time.sleep(MIN_LOOP_DURATION - (now - previous_time_stamp))

            # ---------- CHANGE THIS PART (optional) ----------

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Get IMU Frames
            # for frame in frames: 
            #     if frame.is_motion_frame():
            #         imu_data = frame.as_motion_frame().get_motion_data()
            #         if frame.get_profile().stream_type() == rs.stream.accel:
            #             print("Accelerometer data:", imu_data)
            #         elif frame.get_profile().stream_type() == rs.stream.gyro:
            #             print("Gyroscope data:", imu_data)


            if not depth_frame or not color_frame: 
                continue

            if RECORD:
                history.append((depth_frame, color_frame))

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # # Get IMU data
            # accel_data = accel_frame.as_motion_frame().get_motion_data()
            # gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            # print(f"Accelerometer: {accel_frame}")
            # print(f"Gyroscope: {gyro_frame}")

            # --------------- CHANGE THIS PART ---------------

            # --- Detect markers ---

            # Markers detection:
            grey_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(grey_frame, aruco_dict, parameters=arucoParams)

            # print(f"Tags in FOV: {detected_ids}, loc: {detected_corners}")
            

            if detected_ids is not None:
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(detected_corners, marker_length, camera_matrix, dist_coeffs)
                
                detected_april_tags = {key[0]: [val1, val2] for key, val1, val2 in zip(detected_ids, tvecs, rvecs)}

                print(detected_april_tags)

                # for id, rvec, tvec in zip(detected_ids, rvecs, tvecs):
                #     # # Draw the marker
                #     # cv2.aruco.drawDetectedMarkers(color_image, detected_corners)
                #     # cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                #     # Print the pose of the marker
                #     print(f"Detected ID: {id}")
                #     print(f"Translation Vector (tvec): {tvec}")
                #     print(f"Rotation Vector (rvec): {rvec}")




            # --- Compute control ---

            x_velocity = 0.0
            y_velocity = 0.0
            r_velocity = 0.0

            # --- Send control to the walking policy ---

            send(s, x_velocity, y_velocity, r_velocity)

        print(f"End of main loop.")

        if RECORD:
            import pickle as pkl
            with open("frames.pkl", 'wb') as f:
                pkl.dump(frames, f)
finally:
    # Stop streaming
    pipeline.stop()

import numpy as np 
from copy import deepcopy 
from scipy.stats import multivariate_normal
import math 

"""
Methodology: 
1. Initialize particles 
2. Predict step: Motion model step the particles
3. Update step: Use the april tag positions to update the particle locations
4. Resample step: resample the particles. 


NOTE: add theta later? 
"""
class LocalizationFilter_theta:
    def __init__(self, init_robot_position, 
                #  known_tag_ids, init_known_tag_positions, 
                 num_particles): 
        
        self.init_robot_position = init_robot_position 
        self.num_particles = num_particles

        self.particles = None 
        self.particle_weights = None 

        # Parameters to change
        self.grid_size = [[-5, 5], [-5, 5], [0, 2*np.pi]] # [[x start, x end], [y start, y end], [theta start, theta end]]

        self.motion_mean = np.array([0, 0, 0])   # observation noise mean 
        motion_var = 1e-2
        motion_theta_var = 1e-5
        self.motion_cov = np.array([[motion_var, 0, 0],  # observation noise covariance
                                 [0, motion_var, 0], 
                                 [0, 0, motion_theta_var]])
        self.motion_cov_det = np.linalg.det(self.motion_cov)
        self.inv_motion_cov = np.linalg.inv(self.motion_cov)

        self.obs_mean = np.array([0, 0, 0])   # observation noise mean 
        obs_var = 2
        obs_theta_var = 0.01
        self.obs_cov = np.array([[obs_var, 0, 0],  # observation noise covariance
                                 [0, obs_var, 0], 
                                 [0, 0, obs_theta_var]])
        self.obs_cov_det = np.linalg.det(self.obs_cov)
        self.inv_obs_cov = np.linalg.inv(self.obs_cov)
        
        self.init_particles()
        return 

    def reinit_particles(self):
        self.init_particles()
        return

    def init_particles(self): 
        """
        Initialize particles
        """
        initialization_type = "uniform"
        self.particles = []
        for particle_num in range(self.num_particles): 
            if initialization_type == "uniform": 
                x = np.random.uniform(self.grid_size[0][0], self.grid_size[0][1])
                y = np.random.uniform(self.grid_size[1][0], self.grid_size[1][1])
                theta = np.random.uniform(self.grid_size[2][0], self.grid_size[2][1])

                self.particles.append([x, y, theta])
            else: 
                raise NotImplementedError("Initialization type not implemented")

        self.particles = np.array(self.particles)
        self.particle_weights = np.ones(self.num_particles) / self.num_particles
        return 

    def predict_step(self, delta_action): 
        """
        Motion model step the particles
        args: 
            - delta_action: [delta x, delta y, delta theta ?]
        """

        all_noise = np.random.multivariate_normal(self.motion_mean, self.motion_cov, size=self.num_particles)
        self.particles = np.array(self.particles) + delta_action.reshape((1, -1)) + all_noise

        self.particles[:, 2] = self.particles[:, 2] % (2*np.pi)

        # INEFFICIENT: but working 
        # for particle_num in range(len(self.particles)): 
        #     self.particles[particle_num][0] += delta_action[0]
        #     self.particles[particle_num][1] += delta_action[1]
        #     # self.particles[particle_num][2] = (self.particles[particle_num][2] + delta_action[2]) % (2*np.pi) # NOTE: theta

        #     # inject motion noise 
        #     self.particles[particle_num] = np.random.multivariate_normal(np.array(self.particles[particle_num]) + self.motion_mean, self.motion_cov)

        return
    
    def update_step(self, estimated_robot_position): 
        """
        Use the robot position meand from the estimated april tags to update the particle weights 
        and then update the particles via resampling
        """

        # theta wrap around 
        self.particles[:, 2] = self.particles[:, 2] % (2*np.pi)
        estimated_robot_position[2] = estimated_robot_position[2] % (2*np.pi)

        pdf_normalizer = 1/(((2*np.pi)**(3/2)) *np.sqrt(self.obs_cov_det))
        pdf_exp = -0.5 * np.einsum('ij,ij->i', np.einsum('ik,kj->ij', self.particles - estimated_robot_position.reshape((1, -1)) ,  self.inv_obs_cov), (self.particles - estimated_robot_position.reshape((1, -1))))
        obs_weight_update_multipliers = pdf_normalizer * np.exp(pdf_exp)
        updated_particle_weights = self.particle_weights * obs_weight_update_multipliers

        # INEFFICIENT: but working
        # updated_particle_weights = deepcopy(self.particle_weights)
        # for particle_num in range(len(self.particles)): 
            
        #     obs_weight_update_multiplier = get_pdf(observed_position=np.array(estimated_robot_position), 
        #                                                 particle_mean=np.array(self.particles[particle_num]), 
        #                                                 particle_inv_cov=self.inv_obs_cov, 
        #                                                 particle_obs_cov_det=self.obs_cov_det)
        #     updated_particle_weights[particle_num] *= obs_weight_update_multiplier
            

        # normalize the weights 
        updated_particle_weights = np.array(updated_particle_weights)
        updated_particle_weights /= np.sum(updated_particle_weights)

        self.particle_weights = deepcopy(updated_particle_weights)

        # Resampling 
        tmp1=[val**2 for val in self.particle_weights]
        Neff=1/(np.array(tmp1).sum())

        if Neff < self.num_particles/3: # resample 
            # first resampling approach - resampling according to the probabilities stored in the weights
            resampledStateIndex=np.random.choice(np.arange(self.num_particles), self.num_particles, p=self.particle_weights, replace=True)

            # second resampling approach - systematic resampling
            # resampledStateIndex=systematicResampling(self.particle_weights)

            new_particles = self.particles[resampledStateIndex]
            new_particle_weights = self.particle_weights[resampledStateIndex]
            # normalize new particle weights
            new_particle_weights = new_particle_weights/np.sum(new_particle_weights)
        
            self.particles = deepcopy(new_particles)
            self.particle_weights = deepcopy(new_particle_weights)

        return

    ############## Interface ################

    def get_robot_position(self): 
        """
        Get the robot position from the particles
        """
        pos = np.array([0.0, 0.0, 0.0])
        for particle_num in range(len(self.particles)): 
            pos += self.particles[particle_num] * self.particle_weights[particle_num]
        pos[2] = pos[2] % (2*np.pi)
        return pos 
        # return np.mean(self.particles*self.particle_weights.reshape((-1, 1)), axis=0)

    def predict_update_position(self, delta_action, estimated_robot_position):
        """
        Predict and update the robot position
        """
        self.predict_step(delta_action)
        self.update_step(estimated_robot_position)
        return self.get_robot_position()

############### Helper Functions ################
def get_pdf(observed_position, particle_mean, particle_inv_cov, particle_obs_cov_det): 
    """
    Get the probability density function of the particle position given the mean and covariance
    """
    distrib = multivariate_normal(mean=particle_mean, cov=particle_inv_cov)

    # testthing = 1/(2*np.pi*np.sqrt(particle_obs_cov_det)) * np.exp(-0.5 * (observed_position - particle_mean).T @ particle_inv_cov @ (observed_position - particle_mean))
    # return testthing

    output =  distrib.pdf(observed_position)
    return output
    

def systematicResampling(weightArray):
    # N is the total number of samples
    N=len(weightArray)
    # cummulative sum of weights
    cValues=[]
    cValues.append(weightArray[0])
 
    for i in range(N-1):
        cValues.append(cValues[i]+weightArray[i+1])
 
    # starting random point
    startingPoint=np.random.uniform(low=0.0, high=1/(N))
     
    # this list stores indices of resampled states
    resampledIndex=[]
    for j in range(N):
        currentPoint=startingPoint+(1/N)*(j)
        s=0
        while (currentPoint>cValues[s]):
            s=s+1
             
        resampledIndex.append(s)
 
    return resampledIndex

class potentialField:

    def __init__(self):
        # Constants
        self.K_ATTRACT = 1.0 # Attractive force gain
        self.K_REPEL = 100.0  # Repulsive force gain
        self.THRESHOLD = 1.0  # Threshold distance for repulsive force
        self.MAX_VELOCITY = 1.0 # Maximum velocity for the robot
        self.RANDOM_PERTURBATION = 0.1  # Random perturbation factor
        self.DT = 0.1  # Time step
        self.MAX_ANGULAR_VELOCITY=0.5  # Maximum angular velocity
        self.ANGLE_GAIN = 1.0  # Gain for angular velocity

    # Define functions for attractive and repulsive forces
    def attractive_force(self, robot_pos, goal_pos):
        force = self.K_ATTRACT * (goal_pos - robot_pos)
        return force

    def repulsive_force(self, robot_pos, obstacle_pos):
        force = np.zeros(2)
        for obs in obstacle_pos:
            distance = np.linalg.norm(robot_pos - obs)
            if distance < self.THRESHOLD:
                repulsion = self.K_REPEL * (1.0 / distance - 1.0 / self.THRESHOLD) * (1.0 / (distance**2)) * (robot_pos - obs) / distance
                force += repulsion
        return force

    def compute_total_force(self, robot_pos, goal_pos, obstacle_pos):
        F_attr = self.attractive_force(robot_pos, goal_pos)
        F_repl = self.repulsive_force(robot_pos, obstacle_pos)
        F_total = F_attr + F_repl
        return F_total

    def compute_velocity(self, robot_pos, goal_pos, obstacle_pos, robot_orientation):
            # Calculate desired direction
            F_total = self.compute_total_force(robot_pos, goal_pos, obstacle_pos)
            desired_direction = np.arctan2(F_total[1], F_total[0])
            
            # Calculate angular velocity
            angle_diff = desired_direction - robot_orientation
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize the angle to the range [-pi, pi]
            angular_velocity = self.ANGLE_GAIN * angle_diff

            # Add random perturbation to avoid local minima
            perturbation = self.RANDOM_PERTURBATION * (np.random.rand(2) - 0.5)
            F_total += perturbation

            velocity_x = F_total[0]
            velocity_y = F_total[1]
            
            return velocity_x, velocity_y, angular_velocity

    def limit_angular_velocity(self, angular_velocity, max_ang_vel):
            if abs(angular_velocity) > max_ang_vel:
                angular_velocity = np.sign(angular_velocity) * max_ang_vel
            return angular_velocity

    def limit_velocity(self, velocity_x, velocity_y, max_velocity):
        speed = np.linalg.norm([velocity_x, velocity_y])
        if speed > max_velocity:
            scale = max_velocity / speed
            velocity_x *= scale
            velocity_y *= scale
        return velocity_x, velocity_y

    def get_velocity(self, robot_x, robot_y, goal_x, goal_y, obstacle_pos, robot_orientation=0.0):
        robot_pos = np.array([robot_x, robot_y])
        goal_pos = np.array([goal_x, goal_y])
        velocity_x, velocity_y, angular_velocity = self.compute_velocity(robot_pos, goal_pos, obstacle_pos, robot_orientation)
        lim_vx, lim_vy = self.limit_velocity(velocity_x, velocity_y, self.MAX_VELOCITY)
        limit_angular_vel = self.limit_angular_velocity(angular_velocity, self.MAX_ANGULAR_VELOCITY)
        return lim_vx, lim_vy, limit_angular_vel



class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacle_list, rand_area, expand_dis=1.0, goal_sample_rate=5, max_iter=3000):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list

    def planning(self):
        self.node_list = [self.start]
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(nearest_node, new_node):
                continue

            self.node_list.append(new_node)

            if self.calc_dist_to_goal(new_node.x, new_node.y) <= self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if not self.check_collision(new_node, final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Could not find a path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        
        new_node.x += min(extend_length, d) * np.cos(theta)
        new_node.y += min(extend_length, d) * np.sin(theta)
        new_node.parent = from_node

        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.end.x, self.end.y)
        return rnd

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2 for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index

    def check_collision_single_point(self, x, y):
        if x < self.min_rand:
            return True
        elif x > self.max_rand:
            return True
        if y < self.min_rand:
            return True
        elif y > self.max_rand:
            return True

        for (ox, oy, size) in self.obstacle_list:
            dx = ox - x
            dy = oy - y
            d = dx * dx + dy * dy
            if d <= size ** 2:
                return True # in collision
        return False # Safe

    def check_collision(self, start_node, end_node):

        discretization_constant = 0.1
        distance = np.sqrt((start_node.x - end_node.x)**2 + (start_node.y - end_node.y)**2)
        steps = np.arange(0, distance, step=discretization_constant)
        num_steps = len(steps)
        
        for step_num in range(num_steps): 
            x = start_node.x + step_num * (end_node.x - start_node.x) / distance
            y = start_node.y + step_num * (end_node.y - start_node.y) / distance
            if self.check_collision_single_point(x, y):
                return True # in collision
            
        return False  # Safe

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return np.hypot(dx, dy)

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta

def calculate_velocities(path, dt=0.1):
    velocities = []
    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        
        linear_velocity_x = (x1 - x0) / dt
        linear_velocity_y = (y1 - y0) / dt
        
        angle = np.arctan2(y1 - y0, x1 - x0)
        if i == 0:
            prev_angle = angle
        angular_velocity = (angle - prev_angle) / dt
        prev_angle = angle
        
        velocities.append((linear_velocity_x, linear_velocity_y, angular_velocity))
    
    return velocities

def rrt_planning(robot_x, robot_y, goal_x, goal_y, obstacles=None):
    if obstacles is None:
        obstacles = []

    rrt = RRT(start=[robot_x, robot_y], goal=[goal_x, goal_y],
              obstacle_list=obstacles, rand_area=[-2, 15])
    
    path = rrt.planning()
    
    if path is None:
        print("Cannot find path")
        return [], []

    velocities = calculate_velocities(path)

    return velocities, path # Velocities (x, y, angular)


