import argparse
import numpy as np 
import cv2
import tqdm

# Code for parsing command line arguments

parser = argparse.ArgumentParser(description='Detects centres and arms in a video')
parser.add_argument('filename', metavar='F', type=str, help='Path to the video file for detection')

args = parser.parse_args()

# Read in the appropriate file
cap = cv2.VideoCapture(args.filename)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(args.filename[:-4]+'_cropped.mp4', fourcc, 30.0, (180,180))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        roi = frame[227:407,110:290]
        out.write(roi)
        cv2.imshow('Webcam', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
out.release()
cv2.destroyAllWindows()