import cv2
import numpy as np

#Video einlesen
cap = cv2.VideoCapture("hand_l2r.avi")

#Parameter fuer goodFeaturesToTrack -> Finden von Punkten
feature_params = dict(maxCorners = 100, qualityLevel = 0.1, minDistance = 7, blockSize = 7)

#Parameter des Lucas Kanade Algorithmus
lk_params = dict(winSize = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#erstelle ein paar zufällige Farben
color = np.random.randint(0, 255, (100, 3))

#lese erstes Frame des Videos und finde Punkte/Ecken drin
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params);

mask = np.zeros_like(old_frame)

while(1):
	#lese nächstes Frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#errechne nach Lucas Kanade den Optischen Fluss, Rückgabe: p1 sind die Koordinaten der Punkte p0 im nächsten Frame 
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0,  None, **lk_params)

	#wir nehmen nur "gute" Punkte
    good_new = p1[st==1]
    good_old = p0[st==1]

	#male den optischen Fluss dann in das Bild
    for i,(new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

	#zeige das frame mit eingemalten Fluss
    cv2.imshow('frame', img)
	
	#Abbrechen mit Esc
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

	#neu -> alt und neue Runde in der Schleife
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()