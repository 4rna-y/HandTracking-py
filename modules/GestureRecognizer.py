import numpy as np
from math import sqrt, acos, degrees
from modules.Point3D import Point3D
from dataclasses import dataclass

@dataclass
class FingerState:
    isFlexed: bool
    angle: float

@dataclass
class HandState:
    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState

def calculateDistance(p1: Point3D, p2: Point3D) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z

    return sqrt(dx * dx + dy * dy + dz * dz)

def calculateAngle(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    a = calculateDistance(p2, p3)
    b = calculateDistance(p1, p3)
    c = calculateDistance(p1, p2)
    theta = max(-1.0, min(1.0, (a * a + c * c - b * b) / (2 * a * c)))

    return degrees(acos(theta))

def analyzeThumb(landmarks: list[Point3D]) -> FingerState:
    a1 = calculateAngle(landmarks[1], landmarks[2], landmarks[3])
    a2 = calculateAngle(landmarks[2], landmarks[3], landmarks[4])
    angle = (a1 + a2) / 2.0

    return FingerState(angle < 140.0, angle)

def analyzeFinger(landmarks: list[Point3D], base: int) -> FingerState:
    a1 = calculateAngle(landmarks[0], landmarks[base], landmarks[base + 1])
    a2 = calculateAngle(landmarks[base], landmarks[base + 1], landmarks[base + 2])
    a3 = calculateAngle(landmarks[base + 1], landmarks[base + 2], landmarks[base + 3])
    angle = (a1 + a2 + a3) / 3.0

    return FingerState(angle < 160.0, angle)

def analyzeHand(landmarks: list[Point3D]) -> HandState:
    return HandState(
        analyzeThumb(landmarks), 
        analyzeFinger(landmarks, 5), 
        analyzeFinger(landmarks, 9), 
        analyzeFinger(landmarks, 13),
        analyzeFinger(landmarks, 17))




    
