import numpy as np 
import cv2

img  = cv2.imread('snap.png')
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

sensitivity = 7
lower_pink = np.array([175 - sensitivity, 100, 50])
upper_pink = np.array([175 + sensitivity, 255, 255])

mask = cv2.inRange(hsv_img, lower_pink, upper_pink) 

_, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(mask,contours,-1,color=(255,0,0))

res = cv2.bitwise_and(img, img, mask=mask)

for c in contours:
    M = cv2.moments(c)
    # contour_areas.append(int(M['m00']))

    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(res, (cX, cY), 2, (0, 0, 255), -1)

    


cv2.imshow('Mask', mask)
# cv2.imshow('Snapshot',img)
cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()