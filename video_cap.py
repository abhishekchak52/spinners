import cv2 
import numpy


cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Bitwise AND
        break

cap.release()