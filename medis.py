import cv2
import mediapipe as mp

# Initialize mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Path to your video file
video_path = "D:/SEM 6/gait/data/train/DSCN5937.MP4"
   # <-- change this to your video filename

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

# Create display window
cv2.namedWindow('Treadmill Posture Video', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Get pose landmarks
        results = pose.process(frame_rgb)

        # Draw landmarks if available
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

        # Show video with pose overlay
        cv2.imshow('Treadmill Posture Video', frame)

        # Press Esc to exit early
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
