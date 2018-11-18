import cv2
import numpy as np, sys

horse = cv2.imread('horse.png')
zebra = cv2.imread('zebra.png')

#generate Gaussian Pyramid for horse
Gaussian = horse.copy()
GaussianHorse = [Gaussian]
for i in range(6):
    Gaussian = cv2.pyrDown(Gaussian)
    GaussianHorse.append(Gaussian)

#generate Gaussian Pyramid for zebra
Gaussian = zebra.copy()
GaussianZebra = [Gaussian]
for i in range(6):
    Gaussian = cv2.pyrDown(Gaussian)
    GaussianZebra.append(Gaussian)

#generate Laplacian Pyramid for horse
LaplaceHorse = [GaussianHorse[5]]
for i in range (5, 0, -1):
    GaussianX = cv2.pyrUp(GaussianHorse[i])
    Laplace = cv2.subtract(GaussianHorse[i-1], GaussianX)
    LaplaceHorse.append(Laplace)

#generate Laplacian Pyramid for zebra
LaplaceZebra = [GaussianZebra[5]]
for i in range (5, 0, -1):
    GaussianX = cv2.pyrUp(GaussianZebra[i])
    Laplace = cv2.subtract(GaussianZebra[i-1], GaussianX)
    LaplaceZebra.append(Laplace)

#add left and right halves of the images in each level
LS = []
for lhorse, lzebra in zip(LaplaceHorse, LaplaceZebra):
    rows, cols, dpt = lhorse.shape
    ls = np.hstack((lhorse[:,0:int(cols/2)], lzebra[:,int(cols/2):]))
    LS.append(ls)

#now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

#image with direct connectig each half
real = np.hstack((horse[:,0:int(cols/2)],zebra[:,int(cols/2):]))

cv2.imwrite('Pyramid_blending2.png', ls_)
cv2.imwrite('Direct_blending.jpg', real)
