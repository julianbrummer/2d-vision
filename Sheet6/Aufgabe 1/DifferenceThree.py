import numpy as np
import cv2

#read video and define current frame
cap = cv2.VideoCapture('spongebob.mp4')
ret, current_frame = cap.read()

#define middle and previous frame
fgbg = cv2.createBackgroundSubtractorMOG2()
middle_frame = current_frame
previous_frame = middle_frame

#computing difference between 3 images
def framediff(current_frame_gray, middle_frame_gray, previous_frame_gray):
    #difference between current and middle image
    frame_diff1 = cv2.absdiff(current_frame_gray, middle_frame_gray)
    #difference between middle and previous image
    frame_diff2 = cv2.absdiff(middle_frame_gray, previous_frame_gray)
    #and operation of the both differences
    frame_diff = cv2.bitwise_and(frame_diff1, frame_diff2)
    cv2.imshow('frame diff 3', frame_diff)
    return frame_diff

#updating images while video is running
while(cap.isOpened() and ret is True):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    middle_frame_gray = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = framediff(current_frame_gray, middle_frame_gray, previous_frame_gray)

    previous_frame = middle_frame.copy()
    middle_frame = current_frame.copy()
    ret, current_frame = cap.read()

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()