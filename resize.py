import cv2
import numpy as np

cap = cv2.VideoCapture("video/video1.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ouput.avi', fourcc, 30, (1280,720))

while True:
    ret, frame = cap.read()
    if ret == True:
         b = cv2.resize(frame, (1280,720), fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
         out.write(b)
         print(b)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
