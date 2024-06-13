
TAGS_POSES = {
    'tag1': [0.0, 0.0],
    'tag2': [0.5, 0.5],
    'tag3': [0.7, 2]
}

TAGS_POSES_COMP = {
    1: (-58, 0),
    2: (32, 117.5),
    3: (203, 117.5),
    4: (293, 0),
    5: (203, -117.5),
    6: (32, -117.5)
}

import math
import numpy as np
import cv2

def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def estimate_robot_pose_from_tags(tags_poses: dict):
    accumulated_position = np.zeros(3)
    accumulated_orientation = np.zeros(3)
    valid_marker_count = 0

    average_position = 0
    average_orientation = 0

    for tag_id, tag_pose in tags_poses:
        if tag_id not in TAGS_POSES.keys():
            continue

        known_tvec, known_rvec = TAGS_POSES[tag_id]
        rvec, tvec = tag_pose

        # Convert the rotation vector to a rotation matrix
        R_cm = cv2.Rodrigues(rvec)[0]  # Rotation matrix from camera to marker
        t_cm = tvec[0]  # Translation vector from camera to marker

        # Invert the transformation (marker to camera)
        R_mc = R_cm.T
        t_mc = -R_mc @ t_cm

        # Camera pose in the world coordinate system
        R_wm = cv2.Rodrigues(known_rvec)[0]  # Rotation matrix from world to marker
        t_wm = known_tvec  # Translation vector from world to marker

        R_wc = R_wm @ R_mc
        t_wc = R_wm @ t_mc + t_wm

        # Convert the rotation matrix to Euler angles
        camera_orientation_world = rotation_matrix_to_euler(R_wc)
        camera_position_world = t_wc

        accumulated_position += camera_position_world
        accumulated_orientation += camera_orientation_world
        valid_marker_count += 1

        print(f"Camera Position (World): {camera_position_world}")
        print(f"Camera Orientation (World): {camera_orientation_world}")

    if valid_marker_count > 0:
        average_position = accumulated_position / valid_marker_count
        average_orientation = accumulated_orientation / valid_marker_count

        print(f"Average Camera Position (World): {average_position}")
        print(f"Average Camera Orientation (World): {average_orientation}")

    return average_position, average_orientation


def update_obstacles_positions(obstacles_poses: dict, tags_poses: dict, robot_pose):
    for tag_id, tag_pose in tags_poses:
        if tag_id in TAGS_POSES.keys():
            continue

        # Convert the rotation vector to a rotation matrix
        R_cm = cv2.Rodrigues(rvec)[0]  # Rotation matrix from camera to marker
        t_cm = tvec[0]  # Translation vector from camera to marker

        # Invert the transformation (marker to camera)
        R_mc = R_cm.T
        t_mc = -R_mc @ t_cm

        # Camera pose in the world coordinate system
        R_wm = cv2.Rodrigues(known_rvec)[0]  # Rotation matrix from world to marker
        t_wm = known_tvec  # Translation vector from world to marker

        R_wc = R_wm @ R_mc
        t_wc = R_wm @ t_mc + t_wm

        # Convert the rotation matrix to Euler angles
        camera_orientation_world = rotation_matrix_to_euler(R_wc)
        camera_position_world = t_wc

        accumulated_position += camera_position_world
        accumulated_orientation += camera_orientation_world
        valid_marker_count += 1

        print(f"Camera Position (World): {camera_position_world}")
        print(f"Camera Orientation (World): {camera_orientation_world}")

    if valid_marker_count > 0:
        average_position = accumulated_position / valid_marker_count
        average_orientation = accumulated_orientation / valid_marker_count

        print(f"Average Camera Position (World): {average_position}")
        print(f"Average Camera Orientation (World): {average_orientation}")

    return average_position, average_orientation