import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = r"D:\SEM 6\gait\data\incorrect_data"
OUTPUT_DIR = r"D:\SEM 6\gait\data"

mp_pose = mp.solutions.pose
POSE_IDX = mp_pose.PoseLandmark

def angle_between_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_2d(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def compute_angles(landmarks, w, h):
    row = {}
    try:
        hip_r = get_2d(landmarks[POSE_IDX.RIGHT_HIP.value], w, h)
        knee_r = get_2d(landmarks[POSE_IDX.RIGHT_KNEE.value], w, h)
        ankle_r = get_2d(landmarks[POSE_IDX.RIGHT_ANKLE.value], w, h)
        heel_r = get_2d(landmarks[POSE_IDX.RIGHT_HEEL.value], w, h)
        foot_r = get_2d(landmarks[POSE_IDX.RIGHT_FOOT_INDEX.value], w, h)
        hip_l = get_2d(landmarks[POSE_IDX.LEFT_HIP.value], w, h)
        knee_l = get_2d(landmarks[POSE_IDX.LEFT_KNEE.value], w, h)
        ankle_l = get_2d(landmarks[POSE_IDX.LEFT_ANKLE.value], w, h)
        heel_l = get_2d(landmarks[POSE_IDX.LEFT_HEEL.value], w, h)
        foot_l = get_2d(landmarks[POSE_IDX.LEFT_FOOT_INDEX.value], w, h)
        shoulder_l = get_2d(landmarks[POSE_IDX.LEFT_SHOULDER.value], w, h)
        shoulder_r = get_2d(landmarks[POSE_IDX.RIGHT_SHOULDER.value], w, h)
        mid_shoulder = (shoulder_l + shoulder_r) / 2
        mid_hip = (hip_l + hip_r) / 2

        row["hip_angle_r"] = angle_between_points(shoulder_r, hip_r, knee_r)
        row["knee_angle_r"] = angle_between_points(hip_r, knee_r, ankle_r)
        row["foot_strike_angle_r"] = angle_between_points(heel_r, ankle_r, foot_r)
        row["hip_angle_l"] = angle_between_points(shoulder_l, hip_l, knee_l)
        row["knee_angle_l"] = angle_between_points(hip_l, knee_l, ankle_l)
        row["foot_strike_angle_l"] = angle_between_points(heel_l, ankle_l, foot_l)
        vertical = np.array([0, -1])
        torso = mid_shoulder - mid_hip
        cos_torso = np.dot(torso, vertical) / (np.linalg.norm(torso) * np.linalg.norm(vertical) + 1e-7)
        angle = np.arccos(np.clip(cos_torso, -1.0, 1.0))
        row["torso_angle"] = np.degrees(angle)
    except Exception:
        row = {k: np.nan for k in [
            "hip_angle_r", "knee_angle_r", "foot_strike_angle_r",
            "hip_angle_l", "knee_angle_l", "foot_strike_angle_l", "torso_angle"
        ]}
    return row

def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False)
    results_list = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        frame_data = {"frame": frame_id}
        if res.pose_landmarks:
            landmarks = res.pose_landmarks.landmark
            frame_data.update(compute_angles(landmarks, w, h))
        else:
            frame_data.update({k: np.nan for k in [
                "hip_angle_r", "knee_angle_r", "foot_strike_angle_r",
                "hip_angle_l", "knee_angle_l", "foot_strike_angle_l", "torso_angle"
            ]})
        results_list.append(frame_data)

    cap.release()
    pd.DataFrame(results_list).to_csv(output_csv, index=False)
    print(f"Extracted: {output_csv}")

if __name__ == "__main__":
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith('.mp4'):
            fpath = os.path.join(DATA_DIR, fname)
            out_csv = os.path.join(OUTPUT_DIR, fname.replace('.MP4', '_angles.csv').replace('.mp4', '_angles.csv'))
            process_video(fpath, out_csv)
