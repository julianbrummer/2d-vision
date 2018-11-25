import cv2
import numpy as np

img = cv2.imread('woman.png')
rows, cols, ch = img.shape

#rotate Image
M = cv2.getRotationMatrix2D((cols/2, rows/2), -45, 1)
rotate = cv2.warpAffine(img, M, (cols, rows))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

gray2 = cv2.cvtColor(rotate, cv2.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
rotatedst = cv2.cornerHarris(gray2, 2, 3, 0.04)

#result is dilated for making the corners
dst = cv2.dilate(dst, None)
rotatedst = cv2.dilate(rotatedst, None)

#Threshold for an optimal value, it may vary depending on the image
img[dst > 0.01 * dst.max()] = [0, 0, 255]
rotate[rotatedst > 0.01 * rotatedst.max()] = [0, 0, 255]

cv2.imshow('dst', img)
cv2.imshow('rotatedst', rotate)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()