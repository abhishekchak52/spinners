import numpy as np 
import cv2

cap = cv2.VideoCapture('data/webcam.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()


    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()