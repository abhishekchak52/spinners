import argparse
import numpy as np
import cv2

# Code for parsing command line arguments

parser = argparse.ArgumentParser(description='Captures video from the webcam and saves it to a file')
parser.add_argument('output', type=str, help='name of the output file')

args = parser.parse_args()

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(args.output+'.mp4', fourcc, 50.0, (640,480))

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