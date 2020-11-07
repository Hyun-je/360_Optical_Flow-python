import cv2
import numpy as numpy


cap = cv2.VideoCapture("sample.mp4")



while cap.isOpened():

    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_left_crop = frame[:, 0:300]

    frame_expanded = cv2.hconcat([frame, frame_left_crop])

    cv2.imshow("video", frame_expanded)


    # terminate
    k = cv2.waitKey(4)
    if k == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break


cap.release()
cv2.destroyAllWindows()
