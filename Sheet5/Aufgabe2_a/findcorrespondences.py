import numpy as np
import cv2
import copy

PATH1 = "church_left.png"
PATH2 = "church_right.png"


#Hilfsfunktion zur Errechnung des euklidischen Abstands zwischen zwei Punkten
def entfernung_berechnen(p1, p2):
    x_val = p1[0]-p2[0]
    y_val = p1[1]-p2[1]
    result = x_val*x_val + y_val*y_val
    result = result**0.5
    return result


#Hilfsfunktion, die aus einer Liste aus Punkten den herausfindet, der am nähsten zu dem gegebenen Punkt ist
#Rückgabe ist dieser Punkt und der Index des Punktes in der Liste
def finde_nahen_punkt(punkt, liste):
    naher_punkt = liste[0]
    index = 0
    entfernung = entfernung_berechnen(punkt, naher_punkt)
    for k in range(len(liste)):
        neue_entfernung = entfernung_berechnen(punkt, liste[k])
        if(neue_entfernung < entfernung):
            index = k
            naher_punkt = liste[k]
    return naher_punkt, index


#Hilfsfunktion, die versucht durch Template Matching gefundene Punkte in einem linken Bild mit denen im rechten Bild
#in Verbindung zu bringen. Rückgabe ist eine Liste aus Tupeln (i, j), wobei i und j Indizes der zusammengehörigen Punkte
#ist. j=-1 wenn für das i kein dazugehöriger Punkt gefunden wurde.
def search_for_matches(points_left, points_right, image_left, image_right, useSSD):
    height, width = image_left.shape[:2]
    range = round(min([height, width])/8)

    result = []

    test = 0
    #Fuer jeden Eckpunkt im linken Bild...
    for point in points_left:
        x_val_left = point[0]
        x_val_right = min([point[0]+range, width-1])
        y_val_left = point[1]
        y_val_right = min([point[1]+range, height-1])
        #...schneiden wir eine ROI an ihm aus und...
        template = img_left[y_val_left:y_val_right, x_val_left:x_val_right]

        if useSSD:
            help = cv2.TM_SQDIFF
        else:
            help = cv2.TM_CCORR_NORMED

        #matchen diesen Ausschnitt mit dem rechten Bild. Aus dem Ergebnis suchen wir das Maximum raus (die Stelle mit
        #bestem Matching) und...
        matched = cv2.matchTemplate(image_right, template, help)
        _, _, _, maximum = cv2.minMaxLoc(matched)

        #... suchen die Ecke aus dem rechten Bild die am Maximum am naehsten ist.
        naher_punkt, index = finde_nahen_punkt(maximum, points_right)

        #Sollte jedoch die Entfernung zwischen dem vermeintlich dazugehörigen Punkt zu groß sein, muss ein Fehler
        # vorliegen.
        if entfernung_berechnen(naher_punkt, maximum) > 20:
            index = -1

        result.append((points_left.index(point), index))

    return result


#Hilfsfunktion, die errechnet, ob ein Punkt (x_mitte, y_mitte) das lokale Maximum in einem gegebenen Bild ist
def lokales_max(image, x_mitte, y_mitte):
    weite = 5
    height, width = image.shape[:2]
    wert = image[y_mitte][x_mitte]
    for i in range(-weite, weite+1):
        for j in range(-weite, weite+1):
            if y_mitte+i < 0 or y_mitte+i >= height or x_mitte+j < 0 or x_mitte+j >= width:
                continue
            anderer_wert = image[y_mitte+i][x_mitte+j]
            if(wert < anderer_wert):
                return False
    return True


#Hilfsfunktion, die die Eckenerkennung nach Harris auf einem gegebenen Bild ausfuehrt und die gefundenen Ecke
#einzeichnet. Rückgabe ist das Bild mit eingezeichneten Ecken und eine Liste der Koordinaten der gefundenen Ecken
def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    result = []
    punkte = np.where(dst > 0.01*dst.max())
    for i in range(len(punkte[0])):
        if lokales_max(dst, punkte[1][i], punkte[0][i]):
            result.append((punkte[1][i], punkte[0][i]))
            cv2.circle(image, (punkte[1][i], punkte[0][i]), 2, (0, 0, 255), -1)
    return image, result


#Liest die Bilder ein und macht eine Kopie von beiden
original_left = cv2.imread(PATH1)
original_right = cv2.imread(PATH2)
img_left = copy.copy(original_left)
img_right = copy.copy(original_right)

cv2.imshow("Original", np.hstack((img_left, img_right)))

#Zeige Bilder mit eingezeichneten Ecken ein
corners_left, points_left = harris_corner_detection(img_left)
corners_right, points_right = harris_corner_detection(img_right)
cv2.imshow("Nach Ecken Erkennung", np.hstack((corners_left, corners_right)))

#Fuege die Bilder nebeneinander zusammen
big_image = np.concatenate((corners_left, corners_right), axis=1)
big_image_copy = copy.copy(big_image)

#Suche nach einer passenden Zuordnung der Eckpunkte der beiden Bilder und zeichne zwischen den Punkten Linien ein
matchings_a = search_for_matches(points_left, points_right, original_left, original_right, True)
for match in matchings_a:
    if not match[1] == -1:
        punkt_rechts = ((points_right[match[1]])[0] + corners_left.shape[1], (points_right[match[1]])[1])
        cv2.line(big_image, points_left[match[0]], punkt_rechts, (255, 0, 0), 1)
cv2.imshow("Ecken verbunden nach SSD", big_image)

matchings_b = search_for_matches(points_left, points_right, original_left, original_right, False)
for match in matchings_b:
    if not match[1] == -1:
        punkt_rechts = ((points_right[match[1]])[0] + corners_left.shape[1], (points_right[match[1]])[1])
        cv2.line(big_image_copy, points_left[match[0]], punkt_rechts, (255, 0, 0), 1)
cv2.imshow("Ecken verbunden nach NCC", big_image_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
