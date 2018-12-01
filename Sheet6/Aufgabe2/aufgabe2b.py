import cv2
import numpy as np

#Fange Videostream von Webcam ein
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#erstelle ein video und per out kann ein frame geschrieben werden
out = cv2.VideoWriter('hand_l2r.avi', fourcc, 20.0, (640, 480))

#solange der stream zur webcam offen ist...
while(cap.isOpened()):
	#...lies ein frame ein und...
    ret, frame = cap.read()
    if ret == True:
		
		#...schreibe das frame.
        out.write(frame)
		
		#zusätzlich zeige es auf dem bildschirm an
        cv2.imshow('frame', frame)
		
		#abbruch bei druck auf q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

#schoen alles schliessen
cap.release()
out.release()
cv2.destroyAllWindows()