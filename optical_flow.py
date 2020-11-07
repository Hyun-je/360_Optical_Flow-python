import cv2
import numpy as np


cap = cv2.VideoCapture("360_sample_video.mp4")


# get vcap property 
raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

resize_width = (int)(raw_width * 0.3)
resize_height = (int)(raw_height * 0.3)

expaned_width = (int)(resize_width * 1.2)
expaned_height = (int)(resize_height)

hsv = np.zeros(shape=[expaned_height, expaned_width, 3], dtype=np.uint8)
hsv[...,1] = 255

prev = np.zeros(shape=[expaned_height, expaned_width, 1], dtype=np.uint8)
next = np.zeros(shape=[expaned_height, expaned_width, 1], dtype=np.uint8)


while cap.isOpened():

    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    frame_left_crop = frame[:, 0:expaned_width-resize_width]

    next = cv2.hconcat([frame, frame_left_crop])


    flow = cv2.calcOpticalFlowFarneback(prev,next, None, 0.0, 1, 20, 1, 5, 1.0, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    offset = 50
    rgb_left = rgb[:, offset:offset+(int)(resize_width/2)]
    rgb_right = rgb[:, offset+(int)(resize_width/2):offset+resize_width]
    rgb_concat = cv2.hconcat([rgb_right, rgb_left])

    cv2.imshow("video", cv2.resize(rgb_concat, None, fx=2.0, fy=2.0))

    prev = next


    # terminate
    k = cv2.waitKey(4)
    if k == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break


cap.release()
cv2.destroyAllWindows()
