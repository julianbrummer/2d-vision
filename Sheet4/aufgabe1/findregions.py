import cv2
import numpy as np

#Pfad des Bildes und die Genauigkeit des Template Matchings
path = "koreanSigns.png"
genauigkeit = 0.95

img = cv2.imread(path)
region = None

#Hilfsvariablen
point1 = (0, 0)
point2 = (0, 0)
counter = 0

#Funktion, die bei jeder Aktion mit der Maus aufgerufen wird
def mouse_handler(event, x, y, flags, param):
    #nur Mausklicks der linken Maustaste sind interessant
    if event == cv2.EVENT_LBUTTONDOWN:
        global counter
        global point1
        global point2
        #Eingabe des ersten Punktes der ausgewählten Region
        if counter == 0:
            point1 = (x, y)
            counter = counter + 1
        else:
            #Eingabe des zweiten Punktes
            if counter == 1:
                point2 = (x, y)
                counter = counter + 1
                #Nach Eingabe des zweiten Punktes wird das Template angezeigt und das Template mit dem Bild gematched
                showRegion()
                templateMatching()

#Funktion, die die ausgewählte Region im Bild als Template abspeochert und anzeigt
def showRegion():
    global img
    global region
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    region = img[y1:y2, x1:x2]
    cv2.imshow("Region", region)

#Funktion, die das abgespeicherte Template mit dem originalen Bild abgleicht. Es wird das Ergebnis der normalisierten
#Kreuzkorrelation angezeigt und im originalen Bild gefundene Treffer angezeigt.
def templateMatching():
    global genauigkeit
    global region
    result = cv2.matchTemplate(img, region, cv2.TM_CCORR_NORMED)
    cv2.imshow("Result", result)
    loc = np.where(result >= genauigkeit)
    h, w, c = region.shape
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
    cv2.imshow("Original", img)

cv2.namedWindow("Original")
cv2.setMouseCallback("Original", mouse_handler)
cv2.imshow("Original", img)

cv2.waitKey(0)
cv2.destroyAllWindows()