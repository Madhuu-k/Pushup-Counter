import numpy as np
import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14    
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15

RIGHT_HIP = 24
LEFT_HIP = 23

rep = 0
stage = None   # Base Case
 
DOWN_ANGLE = 110
UP_ANGLE = 145


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = (np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

# Path to model
MODEL_PATH = "./pose_landmarker_lite.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

# Capture Video
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Camera is not opened")
    exit()
    
last_pose_landmarks = None
pose_visible = False    
down_frames = 0
DOWN_FRAMES_REQUIRED = 4

while True:
    ret, frame = capture.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Detect pose
    results = pose_landmarker.detect(mp_image)

    if results.pose_landmarks:
        last_pose_landmarks = results.pose_landmarks
        pose_visible = True
    else:
        pose_visible = False

    # Check if pose is detected
    if last_pose_landmarks is None:
        cv2.putText(
            frame,
            "Adjust camera for detection",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.imshow("Push Up Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    land_marks = last_pose_landmarks[0]

    right_shoulder = land_marks[RIGHT_SHOULDER]
    right_elbow    = land_marks[RIGHT_ELBOW]
    right_wrist    = land_marks[RIGHT_WRIST]

    left_shoulder  = land_marks[LEFT_SHOULDER]
    left_elbow     = land_marks[LEFT_ELBOW]
    left_wrist     = land_marks[LEFT_WRIST]

    # Convert to pixel coordinates
    right_shoulder_point = (int(right_shoulder.x * w), int(right_shoulder.y * h))
    right_elbow_point = (int(right_elbow.x * w), int(right_elbow.y * h))
    right_wrist_point = (int(right_wrist.x * w), int(right_wrist.y * h))

    left_shoulder_point = (int(left_shoulder.x * w), int(left_shoulder.y * h))
    left_elbow_point = (int(left_elbow.x * w), int(left_elbow.y * h))
    left_wrist_point = (int(left_wrist.x * w), int(left_wrist.y * h))
    
    # Calculate angle (single arm for stability)
    angle = calculate_angle(right_shoulder_point, right_elbow_point, right_wrist_point)

    # =========================
    # REP COUNT LOGIC 
    # =========================
    
    if pose_visible:

    # Get into pushup position
        if stage is None:
            if angle > UP_ANGLE:
                stage = "up"
            elif angle < DOWN_ANGLE:
                stage = "down"
                
    # Detect DOWN position
        if angle < DOWN_ANGLE:
            down_frames += 1
        else:
            down_frames = 0

        if down_frames >= DOWN_FRAMES_REQUIRED and stage == "up":
            stage = "down"

    # When UP -> then rep counts
        if angle > UP_ANGLE and stage == "down":
            stage = "up"
            rep += 1
            down_frames = 0


    # =========================
    # DRAWING 
    # =========================
    cv2.line(frame, right_shoulder_point, right_elbow_point, (0, 255, 0), 3)
    cv2.line(frame, right_elbow_point, right_wrist_point, (0, 255, 0), 3)
    cv2.line(frame, left_shoulder_point, left_elbow_point, (0, 255, 0), 3)
    cv2.line(frame, left_elbow_point, left_wrist_point, (0, 255, 0), 3)

    cv2.circle(frame, right_shoulder_point, 5, (0, 0, 255), -1)
    cv2.circle(frame, right_elbow_point, 5, (0, 0, 255), -1)
    cv2.circle(frame, right_wrist_point, 5, (0, 0, 255), -1)

    cv2.circle(frame, left_shoulder_point, 5, (0, 0, 255), -1)
    cv2.circle(frame, left_elbow_point, 5, (0, 0, 255), -1)
    cv2.circle(frame, left_wrist_point, 5, (0, 0, 255), -1)

    cv2.putText(frame, f"Reps: {rep}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Angle: {int(angle)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Stage: {stage}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Push Up Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
capture.release()
cv2.destroyAllWindows()