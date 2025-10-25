import mediapipe as mp
import cv2
import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load(r'D:\SEM 6\gait\treadmill_posture_model.pkl')
scaler = joblib.load(r'D:\SEM 6\gait\treadmill_scaler.pkl')

# Reference ranges for detailed feedback
ranges_female = {
    'hip_angle_r': (153.04, 181.99), 'knee_angle_r': (107.08, 196.01),
    'foot_strike_angle_r': (74.91, 123.11), 'hip_angle_l': (159.89, 180.30),
    'knee_angle_l': (113.87, 192.30), 'foot_strike_angle_l': (74.55, 118.66),
    'torso_angle': (0.32, 8.82)
}
ranges_male = {
    'hip_angle_r': (151.69, 178.56), 'knee_angle_r': (84.90, 196.17),
    'foot_strike_angle_r': (59.69, 126.69), 'hip_angle_l': (142.38, 182.66),
    'knee_angle_l': (107.23, 186.00), 'foot_strike_angle_l': (55.25, 123.63),
    'torso_angle': (-0.47, 5.34)
}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE_IDX = mp_pose.PoseLandmark

# ========== CONFIGURATION ==========
# Change these as needed
VIDEO_PATH = r'D:\SEM 6\gait\data\incorrect_data\male_incorrect_1.MP4'
GENDER = 'male'  # Change to 'female' if testing female video
OUTPUT_VIDEO = r'D:\SEM 6\gait\output_analyzed_video.mp4'
OUTPUT_REPORT = r'D:\SEM 6\gait\analysis_report.csv'
WARNING_THRESHOLD = 15  # Warn after X consecutive incorrect frames
# ===================================

ranges = ranges_male if GENDER == 'male' else ranges_female

def angle_between_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_2d(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def compute_angles(landmarks, w, h):
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

        angles = {
            'hip_angle_r': angle_between_points(shoulder_r, hip_r, knee_r),
            'knee_angle_r': angle_between_points(hip_r, knee_r, ankle_r),
            'foot_strike_angle_r': angle_between_points(heel_r, ankle_r, foot_r),
            'hip_angle_l': angle_between_points(shoulder_l, hip_l, knee_l),
            'knee_angle_l': angle_between_points(hip_l, knee_l, ankle_l),
            'foot_strike_angle_l': angle_between_points(heel_l, ankle_l, foot_l)
        }
        vertical = np.array([0, -1])
        torso = mid_shoulder - mid_hip
        cos_torso = np.dot(torso, vertical) / (np.linalg.norm(torso) + 1e-7)
        angles['torso_angle'] = np.degrees(np.arccos(np.clip(cos_torso, -1.0, 1.0)))
        
        return angles
    except:
        return None

def check_individual_angles(angles):
    """Check which specific angles are incorrect"""
    issues = []
    for angle_name, (lo, hi) in ranges.items():
        if angle_name in angles:
            val = angles[angle_name]
            if val < lo:
                issues.append(f"{angle_name.replace('_', ' ').title()}: {val:.1f}° (LOW, ideal {lo:.1f}-{hi:.1f}°)")
            elif val > hi:
                issues.append(f"{angle_name.replace('_', ' ').title()}: {val:.1f}° (HIGH, ideal {lo:.1f}-{hi:.1f}°)")
    return issues

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Tracking variables
frame_count = 0
incorrect_counter = 0
warning_active = False
results_data = []

print(f"Analyzing video: {VIDEO_PATH}")
print(f"Gender: {GENDER}, Total frames: {total_frames}")
print("Processing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    prediction = None
    issues = []
    
    if results.pose_landmarks:
        # Draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract angles
        angles = compute_angles(results.pose_landmarks.landmark, width, height)
        
        if angles:
            # Prepare features
            hip_sym = abs(angles['hip_angle_r'] - angles['hip_angle_l'])
            knee_sym = abs(angles['knee_angle_r'] - angles['knee_angle_l'])
            foot_sym = abs(angles['foot_strike_angle_r'] - angles['foot_strike_angle_l'])
            
            features = np.array([[
                angles['hip_angle_r'], angles['knee_angle_r'], angles['foot_strike_angle_r'],
                angles['hip_angle_l'], angles['knee_angle_l'], angles['foot_strike_angle_l'],
                angles['torso_angle'], hip_sym, knee_sym, foot_sym
            ]])
            
            # Predict
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            confidence = model.predict_proba(features_scaled)[0]
            
            # Update counter
            if prediction == 0:  # Incorrect posture
                incorrect_counter += 1
                issues = check_individual_angles(angles)
            else:  # Correct posture
                incorrect_counter = 0
                warning_active = False
            
            # Check threshold
            if incorrect_counter >= WARNING_THRESHOLD:
                warning_active = True
            
            # Store data
            results_data.append({
                'frame': frame_count,
                'prediction': 'correct' if prediction == 1 else 'incorrect',
                'confidence_correct': confidence[1],
                'confidence_incorrect': confidence[0],
                'incorrect_streak': incorrect_counter,
                'warning': warning_active,
                'issues': '; '.join(issues) if issues else 'None',
                **angles
            })
            
            # Display status
            status_color = (0, 255, 0) if prediction == 1 else (0, 165, 255)
            status_text = "CORRECT" if prediction == 1 else "INCORRECT"
            cv2.putText(frame, f"Frame {frame_count}/{total_frames} - {status_text}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Warning overlay
            if warning_active:
                cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 255), -1)
                cv2.putText(frame, "WARNING: SUSTAINED INCORRECT POSTURE!", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Incorrect for {incorrect_counter} frames", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show specific issues
            y_pos = 80 if not warning_active else 140
            for idx, issue in enumerate(issues[:4]):  # Show top 4 issues
                cv2.putText(frame, issue, (10, y_pos + idx*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Progress bar
    progress = int((frame_count / total_frames) * width)
    cv2.rectangle(frame, (0, height-20), (progress, height-10), (255, 0, 0), -1)
    
    out.write(frame)
    cv2.imshow('Video Analysis', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save detailed report
df_report = pd.DataFrame(results_data)
df_report.to_csv(OUTPUT_REPORT, index=False)

# Print summary
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nTotal frames analyzed: {frame_count}")
print(f"Correct frames: {len(df_report[df_report['prediction']=='correct'])}")
print(f"Incorrect frames: {len(df_report[df_report['prediction']=='incorrect'])}")
print(f"Accuracy: {len(df_report[df_report['prediction']=='correct'])/len(df_report)*100:.2f}%")
print(f"\nWarning episodes: {df_report['warning'].sum()}")
print(f"\nOutput video saved: {OUTPUT_VIDEO}")
print(f"Detailed report saved: {OUTPUT_REPORT}")

# Most common issues
if len(df_report[df_report['prediction']=='incorrect']) > 0:
    print("\n" + "="*60)
    print("MOST COMMON ISSUES:")
    print("="*60)
    all_issues = df_report[df_report['issues'] != 'None']['issues'].str.split('; ').explode()
    issue_counts = all_issues.value_counts().head(5)
    for issue, count in issue_counts.items():
        print(f"  • {issue}: {count} occurrences")

print("\n✓ Analysis complete! Check the output video and CSV report.")
