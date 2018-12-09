import cv2
import numpy as np

def start():
    ##read webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        #mirroring frame
        frame = cv2.flip(frame, 1)

        #draw square
        width = int(cap.get(3))
        height = int(cap.get(4))
        square = frame[int(height/2) - 50: int(height/2) + 50, width - 100 : width]
        cv2. rectangle(frame, (width - 100, int(height/2) - 50), (width, int(height/2) + 50), (0, 255, 0), 3)

        #show immage
        cv2.imshow('frame', frame)

        #if backspace ist pressed, find brightest and darkest color in square
        k = cv2.waitKey(5) & 0xFF
        if k == 8:
            for w in range(width - 100, width):
                for h in range(int(height/2) - 50, int(height/2) + 50):
                    if(h == int(height/2) - 50 and w == width - 100):
                        lower_bound = frame[h, w]
                        upper_bound = frame[h, w]
                    if(np.all(frame[h, w] is not [0, 255, 0]) and np.all(lower_bound == [0, 255, 0])):
                        lower_bound = frame[h, w]
                    if(np.all(frame[h, w] is not [0, 255, 0]) and np.all(upper_bound == [0, 255, 0])):
                        upper_bound = frame[h, w]
                    if(np.all(frame[h, w] <= lower_bound)):
                        lower_bound = frame[h,w]
                    if(np.all(upper_bound <= frame[h, w])):
                        upper_bound = frame[h, w]
            return lower_bound, upper_bound

    cv2.destroyAllWindows()

def bayesColor(lower_bound, upper_bound):
    #read webcam
    cap = cv2.VideoCapture(0)

    #define lower and upper bound from rgb to hsv
    lower_bound = np.uint8([[lower_bound]])
    upper_bound = np.uint8([[upper_bound]])
    lower_boundhsv = cv2.cvtColor(lower_bound, cv2.COLOR_BGR2HSV)
    upper_boundhsv = cv2.cvtColor(upper_bound, cv2.COLOR_BGR2HSV)

    while True:
        #mirroring
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #proof if pixels are in frame
        mask = cv2.inRange(hsv, lower_boundhsv, upper_boundhsv)

        cv2.imshow('mask', 255 - mask)
        cv2.imshow('frame', frame)

        #stopping with exsc 
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break;

    cv2.destroyAllWindows()

#calls 
lower_bound, upper_bound = start()
bayesColor(lower_bound, upper_bound)