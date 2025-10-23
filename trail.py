# ...existing code...
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

# ...existing code...
with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    # Landmarks to highlight for gait visualization
    highlight_landmarks = [
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        h, w = frame.shape[:2]

        # Convert to RGB for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Get pose landmarks
        results = pose.process(frame_rgb)

        # Make frame writable again and convert back to BGR for OpenCV drawing
        frame_rgb.flags.writeable = True
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw full pose landmarks (optional) and then highlight selected gait points
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=2)
            )

            # Highlight selected gait points with larger colored circles and labels
            for lm_enum in highlight_landmarks:
                lm = results.pose_landmarks.landmark[lm_enum]
                # Skip if visibility is low
                if hasattr(lm, 'visibility') and lm.visibility < 0.35:
                    continue
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Draw a filled circle for emphasis
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                # Put a small label next to the point
                label = lm_enum.name.replace('_', ' ').title()
                cv2.putText(frame, label, (cx + 10, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Show video with pose overlay
        cv2.imshow('Treadmill Posture Video', frame)

        # Press Esc to exit early
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
# ...existing code...