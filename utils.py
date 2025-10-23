import cv2
import numpy as np


def landmarks_list_to_array(landmark_list, image_shape):
    rows, cols, _ = image_shape

    if landmark_list is None:
        return None

    return np.asarray([(lmk.x * cols, lmk.y * rows)
                       for lmk in landmark_list.landmark])


def label_params(frame, params, coords):
    """
    Label the frame with the three parameters from TreadmillPosture:
    - hip_angle (converted to degrees)
    - stride_length
    - shoulder_diff
    """
    if coords is None:
        return

    # Convert radians to degrees for displaying hip angle
    hip_angle_deg = params[0] * 180 / np.pi

    # Get midpoints for label positioning
    if len(coords) >= 33:  # Make sure we have enough landmarks
        # Shoulder midpoint (for hip angle)
        shoulder_mid = (coords[11] + coords[12]) / 2
        cv2.putText(frame, f"Hip Angle: {hip_angle_deg:.2f}°", 
                    (int(shoulder_mid[0]), int(shoulder_mid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hip midpoint (for shoulder diff)
        hip_mid = (coords[23] + coords[24]) / 2
        cv2.putText(frame, f"Shoulder Diff: {params[2]:.2f}", 
                    (int(hip_mid[0]), int(hip_mid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Ankle midpoint (for stride length)
        ankle_mid = (coords[27] + coords[28]) / 2
        cv2.putText(frame, f"Stride Length: {params[1]:.2f}", 
                    (int(ankle_mid[0]), int(ankle_mid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def label_final_results(image, label):
    """
    Label the final results on the image.
    """
    expanded_labels = {
        "c": "Correct Form",
        "k": "Knee Ahead, push your butt out",
        "h": "Back Wrongly Positioned, keep your chest up",
        "r": "Back Wrongly Positioned, keep your chest up",
        "x": "Correct Depth"
    }

    image_height, image_width, _ = image.shape  # Fixed height/width assignment

    label_list = [character for character in label]
    described_label = list(map(lambda x: expanded_labels.get(x, x), label_list))  # Added fallback

    color = (42, 210, 48) if "c" in label_list else (13, 13, 205)

    cv2.rectangle(image,
        (0, 0), (image_width, 74),  # Fixed to use width rather than height
        color,
        -1
    )

    cv2.putText(
        image, "   "+" + ".join(word for word in described_label),
        (0, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )