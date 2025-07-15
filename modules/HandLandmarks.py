import numpy as np

from modules.Point3D import Point3D

class HandLandmarks:
    landmarks: list[Point3D]
    confidence: float
    isValid: bool
    handedness: str

    def __init__(self, landmarks: np.ndarray, confidence: float, isValid: bool, handedness: str):
        self.landmarks = self._create_point3d(landmarks)
        self.confidence = confidence
        self.isValid = isValid
        self.handedness = handedness

    def _create_point3d(self, landmarks: np.ndarray) -> list[Point3D] :
        dst : list[Point3D] = []
        for _, (x, y, z) in enumerate(landmarks):
            dst.append(Point3D(x, y, z))

        return dst

    