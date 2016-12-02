import glob
import numpy as np
import cv2

"""
This module is intended to remove background from photos of objects.
Author : Wasim Akram Khan
"""


def sobelEdge(image):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(image, ddepth, 1, 0)
    dy = cv2.Sobel(image, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag



def paperMethod(org):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = org.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Negate the image
    image = cv2.bitwise_not(image)

    # Find sobel edge of the image
    image = sobelEdge(image)

    # Gaussian blurring the result from sobel edge to smooth out outliers
    image = cv2.GaussianBlur(image, (7,7),7)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, image = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)


    # Create a blank image
    blank =  np.zeros_like(image)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask
    cv2.fillPoly(blank, contours, (255,255,255))
    # cv2.drawContours(blank, contours, -1, (255,255,255),1)

    blank = cv2.merge((blank, blank, blank))
    print org.shape
    print blank.shape

    blank = cv2.erode(blank, None, iterations=3)
    add = cv2.bitwise_not(blank)

    # Apply the mask image
    res = cv2.bitwise_and(org, blank)
    res = cv2.add(res, add)

    # cv2.imshow("Image", add)
    cv2.imshow("Mask", org)
    cv2.waitKey(0)

    return res


def exploreBGRemoval(path, SAVE=False):
    folder = glob.glob(path + "/*.jpg")

    if len(folder) == 0:
        folder = glob.glob(path + "/*.png")

    for file in folder:
        image = cv2.imread(file, 1)
        res = paperMethod(image)
        cv2.imshow("BG Removed", res)
        k = cv2.waitKey(0)
        if k == 27 or k == 32:
            break

        if SAVE:
            res = np.hstack((image, res))
            cv2.imwrite(file+"__bg_Removed.jpg",res)


if __name__ == "__main__":
    image = cv2.imread("G:/Filters/BG/p1.jpg",1)

    path = "G:/Filters/BG"
    exploreBGRemoval(path, SAVE=False)
