import cv2
import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import sys


def makeGaussianFilter(numRows, numCols, sigma, highPass):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)
 
   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient
 
   return numpy.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])


# fft image (shift such that low frequencies are closer to image center) and multiply with filter
def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   filteredDFT = shiftedDFT * filterMatrix
   return ifft2(ifftshift(filteredDFT))


def lowPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))


def highPass(imageMatrix, sigma):
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))


def main(argv):
    # parse arguments
    lowImgFile = argv[0] if len(argv) >= 1 else 'Marylin_grey.png'
    highImgFile = argv[1] if len(argv) >= 2 else 'John_grey.png'
    sigma = argv[2] if len(argv) >= 3 else 20

    # load images
    img1 = cv2.imread(lowImgFile, 0)
    img2 = cv2.imread(highImgFile, 0)
    lowPassedImg = lowPass(img1, sigma)
    highPassedImg = highPass(img2, sigma)
    hybrid = lowPassedImg + highPassedImg

    # some debugging
    #fft = fftshift(fft2(img1))
    #n,m = img1.shape
    #low = fft * makeGaussianFilter(n, m, 20, highPass=False)
    #low2 = ifft2(ifftshift(numpy.real(fft)))

    # store images (converts them automatically)
    cv2.imwrite("LowPassed.png", numpy.real(lowPassedImg))
    cv2.imwrite("HighPassed.png", numpy.real(highPassedImg))
    cv2.imwrite("Hybrid.png", numpy.real(hybrid))

    #read back for visualization
    lowImg = cv2.imread("LowPassed.png")
    highImg = cv2.imread("HighPassed.png")
    hybridImg = cv2.imread("Hybrid.png")

    cv2.imshow('Hybrid', numpy.concatenate((lowImg, highImg, hybridImg), axis=1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
