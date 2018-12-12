import numpy as np 
import cv2

cap = cv2.VideoCapture('webcam.avi')

green_sensitivity = 7
pink_sensitivity = 10
lower_green = np.array([60 - green_sensitivity, 100, 50])
upper_green = np.array([60 + green_sensitivity, 255, 255])
lower_pink = np.array([175 - pink_sensitivity, 100, 50])
upper_pink = np.array([175 + pink_sensitivity, 255, 255])


while(cap.isOpened()):
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    centre_mask = cv2.inRange(hsv_frame, lower_green, upper_green) 
    arms_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink) 

    # _, centre_contours, _ = cv2.findContours(centre_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # _, arms_contours, _ = cv2.findContours(arms_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    mask = cv2.add(centre_mask,arms_mask)
    res = cv2.bitwise_and(frame,frame, mask=mask)


    cv2.imshow('Mask', mask)
    cv2.imshow('Detected', res)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()