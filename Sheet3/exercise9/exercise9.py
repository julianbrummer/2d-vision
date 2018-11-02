import cv2
import copy
import numpy as np

#Pfad des Bildes
path = "Schwarz.png"

#Konstanten
size_of_points = 5
color_of_old_points = (0, 0, 255)
color_of_new_points = (255, 0, 0)
image_dimensions = (1000, 600)

#Hilfsvariablen
old_points = []
new_points = []
no_points = 0

#Funktion, die bei jeder Aktion mit der Maus aufgerufen wird
def mouse_handler(event, x, y, flags, param):
    global no_points
    global old_points
    global new_points
    #nur Mausklicks sind interessant
    if event == cv2.EVENT_LBUTTONDOWN:
        #in der ersten Phase werden 4 Punkte auf dem Bild ausgewaehlt und eingemahlt
        if no_points <= 3:
            cv2.circle(img, (x, y), size_of_points, color_of_old_points, -1)
            cv2.imshow("Image", img)
            #speichere Punkt ein
            old_points.append([x, y])
            no_points = no_points + 1
        #in der zweiten Phase werden weitere 4 Punkte ausgewaehlt und eingemahlt
        else:
            cv2.circle(img, (x, y), size_of_points, color_of_new_points, -1)
            cv2.imshow("Image", img)
            #speichere neuen Punkt ein
            new_points.append([x, y])
            no_points = no_points + 1
            #wenn man den achten Punkt eingezeichnet hat, wird die Homography errechnet und das Bild entsprechend neu
            #gezeichnet. Die neuen Punkte werden zu den alten Punkten und die Anzahl der Punkte wird entsprechend
            #wieder auf 4 zurueckgesetzt
            if no_points == 8:
                doHomography()
                no_points = 4
                old_points = new_points
                new_points = []

#Funktion, die die Homography errechnet und das Bild entsprechend neu zeichnet
def doHomography():
    global img
    #lade eine Kopie des Bildes
    img = copy.copy(save_img)
    #errechne die Homography
    h, status = cv2.findHomography(np.array(old_points), np.array(new_points))
    #drehe entsprechend der Homography das Bild
    img = cv2.warpPerspective(img, h, image_dimensions)
    #zeichne die neuen Punkte in das neue Bild ein
    for entry in new_points:
        cv2.circle(img, tuple(entry), size_of_points, color_of_old_points, -1)
    cv2.imshow("Image", img)

img = cv2.imread(path)
img = cv2.resize(img, image_dimensions)
save_img = copy.copy(img)
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_handler)
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()