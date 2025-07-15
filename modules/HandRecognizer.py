import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple
from modules.HandLandmarks import HandLandmarks

class HandRecognizer:

    def __init__(
            self, 
            staticImageMode: bool = False, 
            maxNumHands: int = 1, 
            minDetectionConfidence: float = 0.75, 
            minTrackingConfidence: float = 0.5
        ):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = staticImageMode,
            max_num_hands = maxNumHands,
            min_detection_confidence = minDetectionConfidence,
            min_tracking_confidence = minTrackingConfidence
        )

        self.mpDrawing = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
    
    def processFrame(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgbFrame)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            handLandmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            landmarksArray = np.array([
                [landmark.x, landmark.y, landmark.z]
                for landmark in handLandmarks.landmark
            ])
            
            confidence = self._calculateConfidence(landmarksArray)
            
            return HandLandmarks(
                landmarks=landmarksArray,
                confidence=confidence,
                isValid=True,
                handedness=handedness
            )
        
        return None
    
    def _calculateConfidence(self, landmarks: np.ndarray) -> float:
        wrist = landmarks[0]
        middleTip = landmarks[12]
        handSize = np.linalg.norm(middleTip - wrist)
        
        if 0.1 < handSize < 0.5:
            return 0.9
        elif 0.05 < handSize < 0.8:
            return 0.7
        else:
            return 0.3
    
    def cleanup(self):
        self.hands.close()