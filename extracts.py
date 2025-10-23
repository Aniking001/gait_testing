import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import os
import time

DATA_DIR = r"D:\SEM 6\gait\data\train"
OUTPUT_DIR = r"D:\SEM 6\gait\data"
VIDEOS = {
    'female': os.path.join(DATA_DIR, 'female', 'crct_female.MP4'),
    'male': os.path.join(DATA_DIR, 'male', 'crct_male.MP4')
}

mp_pose = mp.solutions.pose
POSE_IDX = mp_pose.PoseLandmark

def angle_between_points(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
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
        row["hip_angle_r"] = np.nan
        row["knee_angle_r"] = np.nan
        row["foot_strike_angle_r"] = np.nan
        row["hip_angle_l"] = np.nan
        row["knee_angle_l"] = np.nan
        row["foot_strike_angle_l"] = np.nan
        row["torso_angle"] = np.nan
    return row

def draw_pose(image, landmarks, w, h):
    for idx in POSE_IDX:
        lm = landmarks[idx.value]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
    # Optionally, draw lines (skeleton) between some joints

def process_video_with_vis(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    results_list = []
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
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
            draw_pose(frame, landmarks, w, h)
        else:
            frame_data.update({
                "hip_angle_r": np.nan, "knee_angle_r": np.nan, "foot_strike_angle_r": np.nan,
                "hip_angle_l": np.nan, "knee_angle_l": np.nan, "foot_strike_angle_l": np.nan,
                "torso_angle": np.nan
            })

        # Draw progress bar (bottom of the frame)
        bar_width = int((frame_id/total_frames)*w)
        cv2.rectangle(frame, (0, h-30), (bar_width, h-10), (255,0,0), -1)
        cv2.putText(frame, f"Progress: {frame_id}/{total_frames}", (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        cv2.imshow('Gait Extraction Progress', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit early
            break

        results_list.append(frame_data)

    cap.release()
    cv2.destroyAllWindows()
    pd.DataFrame(results_list).to_csv(output_csv, index=False)
    print(f"Extracted {output_csv}")

if __name__ == "__main__":
    for label, path in VIDEOS.items():
        fname = os.path.join(OUTPUT_DIR, f"{label}_gait_angles.csv")
        process_video_with_vis(path, fname)
