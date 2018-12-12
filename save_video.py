import numpy as np
import cv2

cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('webcam.mp4', fourcc, 50.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        out.write(frame)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
out.release()
cv2.destroyAllWindows()