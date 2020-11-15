import cv2
import numpy as np


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture("rtmp://192.168.35.31/live/test")
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
