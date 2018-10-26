import numpy as np
import cv2


def nothing(x):
    pass


# load image
img = cv2.imread('woman.png', 0)
cv2.imshow('image', img)
# create windows, trackbars
cv2.namedWindow('Gaussian Blur');
cv2.createTrackbar('width', 'Gaussian Blur', 3, 15, nothing)
cv2.createTrackbar('height', 'Gaussian Blur', 3, 15, nothing)
cv2.createTrackbar('sigma', 'Gaussian Blur', 1, 5, nothing)

cv2.namedWindow('Bilateral Filter');
cv2.createTrackbar('diameter', 'Bilateral Filter', 3, 8, nothing)
cv2.createTrackbar('sigmaColor', 'Bilateral Filter', 10, 150, nothing)
cv2.createTrackbar('sigmaSpace', 'Bilateral Filter', 10, 150, nothing)

cv2.namedWindow('Median Blur');
cv2.createTrackbar('size', 'Median Blur', 3, 15, nothing)
while (1):

    # exit with esc
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    ## Gaussian Blur
    # read trackbar values
    w = cv2.getTrackbarPos('width', 'Gaussian Blur')
    h = cv2.getTrackbarPos('height', 'Gaussian Blur')
    s = cv2.getTrackbarPos('sigma', 'Gaussian Blur')
    # allow odd numbers only
    w = w + (w + 1) % 2
    h = h + (h + 1) % 2
    cv2.setTrackbarPos('width', 'Gaussian Blur', w)
    cv2.setTrackbarPos('height', 'Gaussian Blur', h)
    # do gaussian blur
    blur = cv2.GaussianBlur(img, (w, h), s)
    cv2.imshow('Gaussian Blur', blur)

    ## Bilateral Filter
    # read trackbar values
    d = cv2.getTrackbarPos('diameter', 'Bilateral Filter')
    scolor = cv2.getTrackbarPos('sigmaColor', 'Bilateral Filter')
    sspace = cv2.getTrackbarPos('sigmaSpace', 'Bilateral Filter')
    # apply bilateral filter
    blur = cv2.bilateralFilter(img, d, scolor, sspace)
    cv2.imshow('Bilateral Filter', blur)

    ## Median blur
    # read trackbar values
    s = cv2.getTrackbarPos('size', 'Median Blur')
    # only allow odd number > 1
    s = max(3, s + (s + 1) % 2)
    cv2.setTrackbarPos('size', 'Median Blur', s)
    # do median blur
    blur = cv2.medianBlur(img, s)
    cv2.imshow('Median Blur', blur)

    ## Sobel filter
    dx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    dy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    mag = np.sqrt(dx * dx + dy * dy)
    sobelmag = cv2.normalize(mag, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    sobelx = np.uint8(np.absolute(dx))
    sobely = np.uint8(np.absolute(dy))
    cv2.imshow('AbsSobelX', sobelx)
    cv2.imshow('AbsSobelY', sobely)
    cv2.imshow('SobelMagnitude', sobelmag)

    ## Scharr filter
    dx = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    dy = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    mag = np.sqrt(dx * dx + dy * dy)
    scharrmag = cv2.normalize(mag, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    scharrx = cv2.Scharr(src=img, ddepth=cv2.CV_8U, dx=1, dy=0, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    scharry = cv2.Scharr(src=img, ddepth=cv2.CV_8U, dx=0, dy=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('ScharrX', scharrx)
    cv2.imshow('ScharrY', scharry)
    cv2.imshow('ScharrMagnitude', scharrmag)

    ## Canny Edge Filter
    canny = cv2.Canny(img, 50, 100)
    cv2.imshow('Canny', canny)
cv2.destroyAllWindows()