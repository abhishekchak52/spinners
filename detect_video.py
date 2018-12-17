import argparse
import numpy as np 
import cv2


# Code for parsing command line arguments

parser = argparse.ArgumentParser(description='Detects centres and arms in a video')
parser.add_argument('filename', metavar='F', type=str, help='Path to the video file for detection')

args = parser.parse_args()

# Read in the appropriate file
cap = cv2.VideoCapture(args.filename)

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
    mask = cv2.add(centre_mask,arms_mask)
    res = cv2.bitwise_and(frame,frame, mask=mask)

    # First we find the centres. We look at all contours from filtering the green marker. 
    # The 6 centres will have area ~350 so we select those contours specifically.
    # Next we choose those pink patches with area greater than ~50

    _, green_ctrs, _ = cv2.findContours(centre_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _, pink_ctrs, _ = cv2.findContours(arms_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    centres = []
    for c in green_ctrs:
        M = cv2.moments(c)
        if M["m00"] > 300:
            centres.append(c)

    arms = [] 
    for a in pink_ctrs:
        M = cv2.moments(a)
        if M["m00"] > 50:
            arms.append(a)

    # Now we draw and mark the centres of each green marker
    centre_coords = []
    for c in centres:
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 2, (0, 0, 255), -1)
        cv2.circle(frame, (cX, cY), 80, (0, 0, 255), 1)
        centre_coords.append([cX,cY])

    # Let's find the centres of all the available arms

    arms_coords = []
    for a in arms:
        M = cv2.moments(a)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 2, (255, 255, 255), -1)
        # cv2.circle(frame, (cX, cY), 80, (0, 0, 255), 1)
        arms_coords.append([cX,cY])

    # cv2.drawContours(res,centres,-1,color=(255,0,0))
    # cv2.drawContours(frame,pink_ctrs,-1,color=(255,0,0))  

    window_size = 180
    for index, centre in enumerate(centre_coords):
        cen_x, cen_y = centre
        min_x = max(cen_x - window_size//2, 0)
        max_x = min(cen_x + window_size//2, 639)
        min_y = max(cen_y - window_size//2, 0)
        max_y = min(cen_y + window_size//2, 479)

        # Apparently for ROI the indices work in the reverse order [y,x]
        roi = frame[min_y:max_y,min_x:max_x]
        cv2.imshow("spinner"+str(index), roi)

    # cv2.imshow('Mask', arms_mask)
    # cv2.imshow('Detected', res)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()