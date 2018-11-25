import cv2
import numpy
import sys


def main(argv):
    # parse arguments
    imgFile1 = argv[0] if len(argv) >= 1 else 'church_left.png'
    imgFile2 = argv[1] if len(argv) >= 2 else 'church_right.png'

    # load images
    img1 = cv2.imread(imgFile1, 0)
    img2 = cv2.imread(imgFile2, 0)

    # feature_params = dict(maxCorners=100,
    #                       qualityLevel=0.3,
    #                       minDistance=7,
    #                       blockSize=7)
    #
    # kp1 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)
    # kp2 = cv2.goodFeaturesToTrack(img2, mask=None, **feature_params)

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 50 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

    cv2.imshow("Matches", img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
