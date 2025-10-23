import sys
import os

print("Starting script...")

try:
    import cv2
    print("Imported cv2 successfully")
except Exception as e:
    print(f"Failed to import cv2: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print("Imported mediapipe successfully")
except Exception as e:
    print(f"Failed to import mediapipe: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("Imported numpy successfully")
except Exception as e:
    print(f"Failed to import numpy: {e}")
    sys.exit(1)

try:
    import pickle
    print("Imported pickle successfully")
except Exception as e:
    print(f"Failed to import pickle: {e}")
    sys.exit(1)

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
print("Added parent directory to sys.path:", parent_dir)

try:
    import TreadmillPosture as tp
    print("Imported TreadmillPosture successfully")
except Exception as e:
    print(f"Failed to import TreadmillPosture: {e}")
    sys.exit(1)

# MediaPipe setup
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe setup completed")
except Exception as e:
    print(f"MediaPipe setup failed: {e}")
    sys.exit(1)

# Load trained posture model
try:
    with open("treadmill_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

if __name__ == '__main__':
    print("Entering main block...")

    # Try multiple camera indices
    cap = None
    for index in range(3):  # Try indices 0, 1, 2
        print(f"Trying camera index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera opened successfully at index {index}")
            break
        else:
            print(f"Camera index {index} failed")
            cap.release()

    if not cap or not cap.isOpened():
        print("Failed to open any camera. Exiting...")
        sys.exit(1)

    # Create the display window
    cv2.namedWindow('Treadmill Posture (Side View)', cv2.WINDOW_NORMAL)

    try:
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            print("Pose model initialized")
            posture_detected = False
            posture_label = "Waiting for posture detection..."

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Camera error: Cannot read frame")
                    break

                print("Processing frame...")

                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = pose.process(frame_rgb)
                frame_rgb.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Check if posture is detected
                if not posture_detected:
                    if results and results.pose_landmarks:
                        posture_detected = True
                        print("Posture detected! Starting analysis...")
                    else:
                        posture_label = "Waiting for posture detection..."
                        print("No person detected yet, waiting...")

                # Analyze posture if detected
                if posture_detected and results and results.pose_landmarks:
                    try:
                        params = tp.get_params(results)
                        print(f"Params received: {params}")
                        input_vector = np.array([params[0], params[1], params[2], params[3], params[4]])

                        # Predict posture
                        prediction = model.predict([input_vector])[0]
                        posture_label = "Correct" if prediction == 1 else "Incorrect"
                        
                        # Add rule-based checks for incorrect posture
                        if params[1] > 30:  # knee_angle_diff
                            posture_label = "Incorrect (Asymmetry)"
                        elif params[0] > 20 or params[0] < 5:  # torso_angle
                            posture_label = "Incorrect (Lean)"
                        elif abs(params[2]) > 10:  # pelvic_tilt
                            posture_label = "Incorrect (Pelvic Tilt)"
                        elif params[3] > 20:  # foot_strike_angle
                            posture_label = "Incorrect (Heel Strike)"
                        elif abs(params[4]) > 15:  # head_tilt_angle
                            posture_label = "Incorrect (Head Tilt)"
                    except Exception as e:
                        print(f"Error processing posture: {e}")
                        posture_label = "Error processing posture"

                # Display feedback
                cv2.putText(frame, f"Posture: {posture_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks if available
                if results and results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display the frame
                cv2.imshow('Treadmill Posture (Side View)', frame)

                # Check for Esc key to exit
                if cv2.waitKey(10) & 0xFF == 27:
                    print("User pressed Esc, exiting...")
                    break

    except Exception as e:
        print(f"Error during main loop: {e}")
        sys.exit(1)

    finally:
        print("Releasing camera...")
        cap.release()
        print("Destroying windows...")
        cv2.destroyAllWindows()
        print("Script finished.")