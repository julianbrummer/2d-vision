import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (25,0,0), (50,0,0) ....,(150,125,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = 25*np.mgrid[0:7,0:6].T.reshape(-1,2)
print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('calibrationImagesCheckerboard/*.JPG')
print(len(images))

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),flags=cv2.CALIB_CB_FAST_CHECK)


    # If found, add object points, image points (after refining them)
    if ret:
        print("found")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        img = cv2.resize(img,(0,0), fx= 0.5, fy= 0.5)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    else:
        print("not found")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

img = cv2.resize(img,(0,0), fx= 0.25, fy= 0.25)
cv2.imshow('distorted',img)

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]

cv2.imshow('undistorted',dst)
cv2.waitKey(0)

cv2.destroyAllWindows()