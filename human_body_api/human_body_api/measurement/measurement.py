import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def get_measurements(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return {"error": "No person detected"}

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        def dist(p1, p2):
            x1, y1 = landmarks[p1].x * w, landmarks[p1].y * h
            x2, y2 = landmarks[p2].x * w, landmarks[p2].y * h
            return round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 2)

        return {
            "shoulder_width": dist(11, 12),
            "left_arm_length": dist(11, 13) + dist(13, 15),
            "right_arm_length": dist(12, 14) + dist(14, 16),
            "left_leg_length": dist(23, 25) + dist(25, 27),
            "right_leg_length": dist(24, 26) + dist(26, 28),
            "hip_width": dist(23, 24)
        }
