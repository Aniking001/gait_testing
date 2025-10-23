import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    cos_theta = dot / (mod_v1 * mod_v2)
    theta = math.acos(cos_theta)
    return math.degrees(theta)

def get_length(v):
    """Calculate the length of a vector."""
    return np.dot(v, v) ** 0.5

def get_params(results, all=False):
    """Extract treadmill posture parameters from side view."""
    if results.pose_landmarks is None:
        return np.zeros((1, 5) if not all else (19, 3))  # 5 parameters now

    points = {}
    # Key landmarks for side view
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    points["NOSE"] = np.array([nose.x, nose.y, nose.z])
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    points["LEFT_SHOULDER"] = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    points["RIGHT_SHOULDER"] = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    points["LEFT_HIP"] = np.array([left_hip.x, left_hip.y, left_hip.z])
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    points["RIGHT_HIP"] = np.array([right_hip.x, right_hip.y, right_hip.z])
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    points["LEFT_KNEE"] = np.array([left_knee.x, left_knee.y, left_knee.z])
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    points["RIGHT_KNEE"] = np.array([right_knee.x, right_knee.y, right_knee.z])
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    points["LEFT_ANKLE"] = np.array([left_ankle.x, left_ankle.y, left_ankle.z])
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    points["RIGHT_ANKLE"] = np.array([right_ankle.x, right_ankle.y, right_ankle.z])
    left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
    points["LEFT_HEEL"] = np.array([left_heel.x, left_heel.y, left_heel.z])
    right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
    points["RIGHT_HEEL"] = np.array([right_heel.x, right_heel.y, right_heel.z])
    left_foot_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
    points["LEFT_FOOT_INDEX"] = np.array([left_foot_index.x, left_foot_index.y, left_foot_index.z])

    # Midpoints
    mid_shoulder = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    mid_hip = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    # Torso angle (forward lean)
    torso_vector = points["LEFT_HIP"] - points["LEFT_SHOULDER"]
    vertical_vector = np.array([0, -1, 0])  # Downward vertical
    torso_angle = get_angle(torso_vector, vertical_vector)

    # Knee angle difference
    theta_knee_left = get_angle(points["LEFT_HIP"] - points["LEFT_KNEE"],
                                points["LEFT_ANKLE"] - points["LEFT_KNEE"])
    theta_knee_right = get_angle(points["RIGHT_HIP"] - points["RIGHT_KNEE"],
                                 points["RIGHT_ANKLE"] - points["RIGHT_KNEE"])
    knee_angle_diff = abs(theta_knee_left - theta_knee_right)

    # Pelvic tilt (hip line vs. horizontal)
    hip_vector = points["RIGHT_HIP"] - points["LEFT_HIP"]
    horizontal_vector = np.array([1, 0, 0])  # Rightward horizontal
    pelvic_tilt = get_angle(hip_vector, horizontal_vector)

    # Foot strike angle (simplified, left foot)
    foot_vector = points["LEFT_HEEL"] - points["LEFT_FOOT_INDEX"]
    ground_vector = np.array([1, 0, 0])  # Horizontal ground
    foot_strike_angle = get_angle(foot_vector, ground_vector)

    # Head tilt angle
    neck_vector = points["NOSE"] - mid_shoulder
    vertical_vector = np.array([0, -1, 0])  # Downward vertical
    head_tilt_angle = get_angle(neck_vector, vertical_vector)

    # Parameters
    params = np.array([torso_angle, knee_angle_diff, pelvic_tilt, foot_strike_angle, head_tilt_angle])

    if all:
        params = np.array([[x, y, z] for pos, (x, y, z) in points.items()])

    return np.round(params, 2)