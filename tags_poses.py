import math
import numpy as np
import cv2
# from spatialmath import *
import matplotlib.pyplot as plt
#
# TAGS_POSES = {
#     1: [0.0, 0.0],
#     2: [0.5, 0.5],
#     3: [0.7, 2]
# }

TAGS_POSES = {
    1: (-0.58, 0, 0),
    2: (0.32, 1.175, 3*np.pi/2),
    3: (2.03, 1.175, 3*np.pi/2),
    4: (2.93, 0, np.pi),
    5: (2.03, -1.175, np.pi/2),
    6: (0.32, -1.175, np.pi/2),
    # 9: (1, 1,  3*np.pi/2)
    # 9: (1, -1,  np.pi/2)
    9: (-0.58, 0,  np.pi/2)

}

TO_PLOT = False

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
    accumulated_position = np.zeros(2)
    accumulated_orientation = np.zeros(1)
    valid_marker_count = 0

    average_position = np.zeros(2)
    average_yaw = np.zeros(1)

    # PLOTS
    if TO_PLOT:
        T_w = SE3()
        T_w.plot(frame='w', color='black')

    for tag_id, tag_pose in tags_poses.items():
        if tag_id not in TAGS_POSES.keys():
            continue

        known_pose = TAGS_POSES[tag_id]
        t_wm = np.append(known_pose[0:2], 0)
        y_wm = known_pose[2]
        R_wm = euler_to_rotation_matrix(0,0, y_wm)


        # Convert the rotation vector to a rotation matrix
        # tvec, rvec = tag_pose
        # t_cm = tvec[0]  # Translation vector from camera to marker
        y_cm = tag_pose[1][0] # + np.pi
        t_cm = np.append(tag_pose[0][0:2], 0)
        R_cm = euler_to_rotation_matrix(0, 0 , y_cm)
        # R_cm = cv2.Rodrigues(tag_pose[1])[0]
        # Rotation matrix from camera to marker

        R_mc = R_cm.T

        R_wc = R_wm @ R_mc

        # Marker to camera
        # t_mc = -R_mc @ t_cm
        t_mc = t_cm

        # Camera pose in the world coordinate system
        R_wc = R_wm @ R_mc
        t_wc = R_wm @ t_mc + t_wm

        # PLOTS
        if TO_PLOT:
            T_wm = SE3.Rt(R_wm, t_wm)
            T_wm.plot(frame='T-wm', color='black')

            T_cm = SE3.Rt(R_cm, t_cm)
            T_cm.plot(frame='T-cm')

            T_wc = SE3.Rt(R_wc, t_wc)
            T_wc.plot(frame='T-wc', color='red')

        if TO_PLOT:
            plt.show(block=False)

        # Convert the rotation matrix to Euler angles
        camera_yaw_world = rotation_matrix_to_euler(R_wc)[2]
        camera_position_world = t_wc[0:2]

        accumulated_position += camera_position_world
        accumulated_orientation += camera_yaw_world
        valid_marker_count += 1

        # print(f"Camera Position (World): {camera_position_world}")
        # print(f"Camera Orientation (World): {camera_yaw_world}")

    if valid_marker_count > 0:
        average_position = accumulated_position / valid_marker_count
        average_yaw = accumulated_orientation / valid_marker_count

        # print(f"Average Camera Position (World): {average_position}")
        # print(f"Average Camera Orientation (World): {average_yaw}")

    return average_position, average_yaw[0]


def update_obstacles_positions(obstacles_poses: dict, tags_poses: dict, robot_pose):
    """

    :param obstacles_poses: Memory osbtacles dict
    :param tags_poses: All seen tags 7: {(np.array([[1, 0, 0.0]], dtype=np.float32), np.array([[0, 0, 0]], dtype=np.float32))}
    :param robot_pose: Current estimated pose (np.array([x, y]), np.array([0.0])
    :return: updated obstacles dicts {7: np.array([x, y]), 8: ...}
    """
    updated_obstacles_poses = obstacles_poses

    # Camera pose in the world frame
    t_wc = np.append(robot_pose[0], 0)
    R_wc = euler_to_rotation_matrix(0.0, 0.0, robot_pose[2])

    # PLOTS
    if TO_PLOT:
        T_w = SE3()
        T_w.plot(frame='w', color='black')
        T_wc = SE3.Rt(R_wc, t_wc)
        T_wc.plot(frame='T-wc', color='blue')

    for tag_id, tag_pose in tags_poses.items():
        if tag_id in TAGS_POSES.keys():
            continue

        # Obstacle to camera
        # tvec_obstacle, rvec_obstacle = tag_pose
        # R_co = cv2.Rodrigues(rvec_obstacle)[0]  # Rotation matrix from camera to marker
        # t_co = tvec_obstacle[0]  # Translation vector from camera to marker

        y_co = tag_pose[1][0] # + np.pi
        t_co = np.append(tag_pose[0][0:2], 0)
        R_co = euler_to_rotation_matrix(0, 0, y_co)

        R_wo = R_wc @ R_co
        t_wo = t_wc + R_wc @ t_co

        # PLOTS
        if TO_PLOT:
            T_wo = SE3.Rt(R_wo, t_wo)
            T_wo.plot(frame=f'T-wo_{tag_id}', color='red')

        # Convert the rotation matrix to Euler angles
        obstacle_yaw_world = rotation_matrix_to_euler(R_wo)[2]
        obstacle_position_world = np.array(t_wo[0:2])

        # print(f"Obstacle {tag_id} Position (World): {obstacle_position_world}")
        # print(f"Obstacle {tag_id} Orientation YAW (World): {obstacle_yaw_world}")

        updated_obstacles_poses[tag_id] = obstacle_position_world

    if TO_PLOT:
        plt.show(block=False)

    return updated_obstacles_poses


def estimate_robot_pose_from_tags_3D(tags_poses: dict):
    accumulated_position = np.zeros(2)
    accumulated_orientation = np.zeros(1)
    valid_marker_count = 0

    average_position = np.zeros(2)
    average_yaw = np.zeros(1)

    # PLOTS
    if TO_PLOT:
        T_w = SE3()
        T_w.plot(frame='w', color='black')

    for tag_id, tag_pose in tags_poses.items():
        if tag_id not in TAGS_POSES.keys():
            continue

        known_pose = TAGS_POSES[tag_id]
        t_wm = np.append(known_pose[0:2], 0)
        y_wm = known_pose[2]
        R_wm = euler_to_rotation_matrix(0,0, y_wm)

        # Convert the rotation vector to a rotation matrix
        tvec, rvec = tag_pose
        t_cm = tvec[0]  # Translation vector from camera to marker
        R_cm = cv2.Rodrigues(tag_pose[1])[0]

        # Rotation matrix from camera to marker
        R_mc = R_cm.T

        R_wc = R_wm @ R_mc

        # Marker to camera
        # t_mc = -R_mc @ t_cm
        t_mc = t_cm

        # Camera pose in the world coordinate system
        R_wc = R_wm @ R_mc
        t_wc = R_wm @ t_mc + t_wm

        # PLOTS
        if TO_PLOT:
            T_wm = SE3.Rt(R_wm, t_wm)
            T_wm.plot(frame='T-wm', color='black')

            T_cm = SE3.Rt(R_cm, t_cm)
            T_cm.plot(frame='T-cm')

            T_wc = SE3.Rt(R_wc, t_wc)
            T_wc.plot(frame='T-wc', color='red')

        if TO_PLOT:
            plt.show(block=False)

        # Convert the rotation matrix to Euler angles
        camera_yaw_world = rotation_matrix_to_euler(R_wc)[2]
        camera_position_world = t_wc[0:2]

        accumulated_position += camera_position_world
        accumulated_orientation += camera_yaw_world
        valid_marker_count += 1

        # print(f"Camera Position (World): {camera_position_world}")
        # print(f"Camera Orientation (World): {camera_yaw_world}")

    if valid_marker_count > 0:
        average_position = accumulated_position / valid_marker_count
        average_yaw = accumulated_orientation / valid_marker_count

        print(f"Average Camera Position (World): {average_position}")
        print(f"Average Camera Orientation (World): {average_yaw}")

    return average_position, average_yaw


def test_robot_pose():
    # tags poses measured by camera
    tags_poses = {
        3: (np.array([[1, 0]], dtype=np.float32), np.array([0.0], dtype=np.float32))  # t, r
        # 4: (np.array([[1, 0]], dtype=np.float32), np.array([0.0], dtype=np.float32))  # t, r
        # 4: (np.array([[1, -0.2]], dtype=np.float32), np.array([0.0], dtype=np.float32))  # t, r

    }

    for id, pose in TAGS_POSES.items():
        T = SE3.Trans(pose[0], pose[1], 0)
        # T.plot(frame=f'Frame {id}')
    # plt.show(block=False)
    t, r = estimate_robot_pose_from_tags(tags_poses)

    print('Done')


def test_obs_meas():
    # robot_position = (np.array([1, 1]), np.array([1.5708]))
    robot_position = (np.array([1, 1]), np.array([np.pi]))

    tags_poses= {
        7: (np.array([[1, -0.5]], dtype=np.float32), np.array([[0]], dtype=np.float32)),
        8: (np.array([[1, 0]], dtype=np.float32), np.array([[0.52]], dtype=np.float32)),
    }

    tags_pose_2 = {
        9: (np.array([[0.5, 0.5]], dtype=np.float32), np.array([[0]], dtype=np.float32))
    }
    up_dict = {}
    up_dict = update_obstacles_positions(up_dict, tags_poses, robot_position)
    print(up_dict)

    print('\n\n')
    up_dict = update_obstacles_positions(up_dict, tags_pose_2, robot_position)
    print(up_dict)

    print('Done')



if __name__ == '__main__':
    # test_robot_pose()
    test_obs_meas()
    ...