import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

class HandLandmarkExtractor:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        return results

    @staticmethod
    def draw_landmarks(frame_bgr, results):
        if not results or not results.multi_hand_landmarks:
            return frame_bgr
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
        return frame_bgr

    @staticmethod
    def landmarks_to_features(results):
        """Return (features, handedness_label) for the first hand.
        features: np.array shape (64,) => 63 coords + 1 handedness bit
        If no hand: return (None, None)
        """
        if not results or not results.multi_hand_landmarks:
            return None, None
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)  # length 63

        # handedness
        handedness = 0.0
        if results.multi_handedness:
            label = results.multi_handedness[0].classification[0].label
            handedness = 1.0 if label.lower().startswith("right") else 0.0
        features = np.concatenate([coords, np.array([handedness], dtype=np.float32)], axis=0)
        return features, "Right" if handedness == 1.0 else "Left"
