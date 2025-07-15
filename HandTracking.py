import cv2 as cv
import numpy as np
import time

from modules import CameraFrameProcessor, HandRecognizer, HandLandmarks
from modules import GestureRecognizer as gesture

processor = CameraFrameProcessor.CameraFrameProcessor()
recognizer = HandRecognizer.HandRecognizer()

prev = time.time()

if processor.open() == False:
    print("Failed to open camera");
    exit

while True:
    readResult, mat = processor.read()
    if readResult == False:
        break

    mat = cv.flip(mat, 1)
    landmarks = recognizer.processFrame(mat)
    if landmarks:
        state = gesture.analyzeHand(landmarks.landmarks)
        processor.renderLandmarks(mat, landmarks, state)
    
    now = time.time()
    diff = int(1 / (now - prev))
    prev = now

    processor.renderFps(mat, diff)

    cv.imshow("HandTracking", mat)
    key = cv.waitKey(1)
    if key == 27:
        break

processor.release()
recognizer.cleanup()

