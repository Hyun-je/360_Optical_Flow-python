import cv2 as cv
import numpy as np


class optical_flow_360:

    def __init__(self, video_source, resize_ratio):
        self.video_source = video_source
        self.resize_ratio = resize_ratio
        self.expand_ratio = 1.2

    def start_capture(self):
        self.cap = cv.VideoCapture(self.video_source)

        self.raw_size = np.array([self.cap.get(cv.CAP_PROP_FRAME_WIDTH), self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)], dtype=np.int32) # 원본 이미지 크기
        self.downsampled_size = (self.raw_size * self.resize_ratio).astype(np.int32)           # 처리속도 향상을 위한 일정 비율로 축소된 이미지 크기
        self.expaned_size = (self.downsampled_size * [self.expand_ratio, 1]).astype(np.int32)  # 360 이미지의 좌우 끝을 연결하기 위해 좌우로 20% 확장된 이미지 크기

    def stop_capture(self):
        self.cap.release()

    def iter_frame(self):

        expaned_width, expaned_height = self.expaned_size

        hsv = np.zeros(shape=[expaned_height, expaned_width, 3], dtype=np.uint8)
        hsv[...,1] = 255

        prev_frame = np.ones(shape=[expaned_height, expaned_width, 1], dtype=np.uint8)
        next_frame = np.ones(shape=[expaned_height, expaned_width, 1], dtype=np.uint8)

        while self.cap.isOpened():

            ret, frame = self.cap.read()

            frame = cv.resize(frame, None, fx=self.resize_ratio, fy=self.resize_ratio)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            #frame = cv.GaussianBlur(frame, (5,5), 0)

            # 좌측 이미지 일부를 복제하여 우측에 이어 붙임
            expand_width = int(self.downsampled_size[0] * (self.expand_ratio - 1))
            frame_left_crop = frame[:, 0:expand_width]
            next_frame = cv.hconcat([frame, frame_left_crop])

            # Dense Optical Flow 수행
            flow = cv.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.0, 1, 20, 1, 5, 1.0, 0)
            prev_frame = next_frame

            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            rgb = cv.bitwise_not(rgb)

            crop_offset = int(expand_width/2)
            rgb_left = rgb[:, crop_offset:crop_offset + int(self.downsampled_size[0]/2)]
            rgb_right = rgb[:, crop_offset + int(self.downsampled_size[0]/2):crop_offset+self.downsampled_size[0]]
            rgb_concat = cv.hconcat([rgb_right, rgb_left])

            yield rgb_concat



optical_flow = optical_flow_360("test2.mp4", 0.2)

optical_flow.start_capture()

for frame in optical_flow.iter_frame():
    
    cv.imshow("video", frame)

    # terminate
    if cv.waitKey(42) == ord('q'):
        break

optical_flow.stop_capture()
cv.destroyAllWindows()
