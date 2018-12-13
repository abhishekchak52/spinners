import argparse
import numpy as np 
import cv2

# Code for parsing command line arguments

parser = argparse.ArgumentParser(description='Detects centres and arms in an image')
parser.add_argument('filename', metavar='F', type=str, help='Path to the image file for detection')

args = parser.parse_args()

img  = cv2.imread(args.filename)
hsv_frame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

green_sensitivity = 7
pink_sensitivity = 10
lower_green = np.array([60 - green_sensitivity, 100, 50])
upper_green = np.array([60 + green_sensitivity, 255, 255])
lower_pink = np.array([175 - pink_sensitivity, 100, 50])
upper_pink = np.array([175 + pink_sensitivity, 255, 255])

centre_mask = cv2.inRange(hsv_frame, lower_green, upper_green) 
arms_mask = cv2.inRange(hsv_frame, lower_pink, upper_pink)
mask = cv2.add(centre_mask,arms_mask)

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