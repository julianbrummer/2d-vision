import cv2
import numpy as np

#Code Ursprung: Programming Computer Vision with Python, Jan Erik Solem S 40
#Link:  http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf

#load image
img = cv2.imread('woman.png', 0)
#create blur-example
u_init = cv2.GaussianBlur(img, (5, 5), 10)
#constants: weight of tv regularizing term, steplength, tolerance of stop criterion
tv_weight = 100
tau = 0.125
tolerance = 0.1

#size of the image
height, width = img.shape

#initializing
u = u_init
px = img        #x-component to the dual field
py = img        #y-component to the dual field
error = 1

while(error > tolerance):
    uold = u

    #gradient of primal variable
    gradux = np.roll(u, -1, axis = 1) - u   #x-component of u's gradient
    graduy = np.roll(u, -1, axis = 0) - u   #y-component of u's gradient

    #update the dual variable
    pxnew = px + (tau / tv_weight) * gradux
    pynew = py + (tau / tv_weight) * graduy
    normnew = np.maximum(1, np.sqrt(pxnew ** 2 + pynew ** 2))

    px = pxnew / normnew    #update of x-component (dual)
    py = pynew / normnew    #update of y-component (dual)

    #update the primal variable
    rxpx = np.roll(px, 1, axis = 1)         #right x-translation of x-component
    rypy = np.roll(py, 1, axis = 0)         #right y-translation of y-component

    divp = (px - rxpx) + (py - rypy)        #divergence of the dual field 
    u = img + tv_weight * divp              #update of the primal variable

    #update of error
    error = np.linalg.norm(u - uold) / np.sqrt(height * width);

#show images
cv2.imshow('denoised image', u)
cv2.imshow('texture', img - u)
#exit with esc
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()