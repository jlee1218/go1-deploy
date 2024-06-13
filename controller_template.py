"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time

import numpy as np
import cv2
import pyrealsense2 as rs
from tags_poses import TAGS_POSES, update_obstacles_positions, estimate_robot_pose_from_tags, rotation_matrix_to_euler

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

# OUR VARS
obstacles_position_dict = {}

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

            # --- Detect markers ---
            # Markers detection:
            grey_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(grey_frame, aruco_dict, parameters=arucoParams)

            # print(f"Tags in FOV: {detected_ids}, loc: {detected_corners}")

            if detected_ids is not None:
                # Estimate pose of each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(detected_corners, marker_length, camera_matrix, dist_coeffs)

                for id, tvec, rvec in zip(detected_ids, tvecs, rvecs):
                    x = tvec[0][2]
                    y = -tvec[0][0]
                    theta = rvec[0][1]

                    if(id == 4 or id == 8):
                        y = -y
                        theta = -theta

                # detected_april_tags = {key[0]: [tvec, rvec] for key, tvec, rvec in zip(detected_ids, tvecs, rvecs)}
                detected_april_tags = {key[0]: [tvec[0][2], tvec[0][0], -rvec[0][2]] for key, tvec, rvec in zip(detected_ids, tvecs, rvecs)}
                print(detected_april_tags)

                # pose, yaw = estimate_robot_pose_from_tags(detected_april_tags)
                # print(f'Pose: {pose[0]} | {pose[1]} | {yaw}')

                # for id, rvec, tvec in zip(detected_ids, rvecs, tvecs):
                    # # Draw the marker
                    # cv2.aruco.drawDetectedMarkers(color_image, detected_corners)
                    # cv2.aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                    # Print the pose of the marker
                    # print(f"Detected ID: {id}")
                    # print(f"Translation Vector (tvec): {tvec}")
                    # print(f"Rotation Vector (rvec): {rvec}")
                    
                    # print(f'Id: {id} \t {rotation_matrix_to_euler(cv2.Rodrigues(rvec)[0])}')


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
