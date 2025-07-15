import cv2 as cv
from modules.HandLandmarks import HandLandmarks
from modules.GestureRecognizer import HandState, FingerState

class CameraFrameProcessor:

    cap : cv.VideoCapture
    width: int
    height: int
    fps: int
    frame: cv.Mat
    
    def __init__(self):
        self.cap = cv.VideoCapture()
        self.width = 640
        self.height = 480
        self.fps = 30


    def open(self) -> bool :
        res = self.cap.open(0, cv.CAP_V4L2)

        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv.CAP_PROP_FPS, 30)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        return res
    
    
    def read(self) -> tuple[bool, cv.Mat] :
        return self.cap.read()
    
    def renderLandmarks(self, src: cv.Mat, landmarks: HandLandmarks, handState: HandState):
        for i, p in enumerate(landmarks.landmarks):
            cv.circle(src, (int(p.x * self.width), int(p.y * self.height)), 3, (0, 0, 255))
        
        cv.putText(src, f"Thumb:  {handState.thumb.isFlexed}({handState.thumb.angle:2f})", (0, 10), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv.putText(src, f"Index:  {handState.index.isFlexed}({handState.index.angle:2f})", (0, 25), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv.putText(src, f"Middle: {handState.middle.isFlexed}({handState.middle.angle:2f})", (0, 40), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv.putText(src, f"Ring:   {handState.ring.isFlexed}({handState.ring.angle:2f})", (0, 55), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
        cv.putText(src, f"Pinky:  {handState.pinky.isFlexed}({handState.pinky.angle:2f})", (0, 70), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))

    def renderFps(self, src: cv.Mat, fps: int):
        cv.putText(src, "FPS: " + str(fps), (0, self.height), cv.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0))
    
    def release(self) :
        cv.destroyAllWindows()
        self.cap.release()

