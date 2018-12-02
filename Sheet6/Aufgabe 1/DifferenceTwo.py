import numpy as np
import cv2

#read video and define current frame
cap = cv2.VideoCapture('spongebob.mp4')
ret, current_frame = cap.read()

#define previous frame
fgbg = cv2.createBackgroundSubtractorMOG2()
previous_frame = current_frame

#computing the difference between 2 images
def framediff(current_frame_gray, previous_frame_gray):
    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    cv2.imshow('frame diff 2',frame_diff)
    return frame_diff

#updating images while video is running
while(cap.isOpened() and ret is True):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = framediff(current_frame_gray,previous_frame_gray)

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
