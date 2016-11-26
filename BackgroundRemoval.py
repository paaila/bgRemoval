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


def GlobablAdaptiveThreshold(image):
    """Removal of background using global adaptive threshold"""

    org = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Doesn't work good
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    cv2.threshold(image,0,255, cv2.THRESH_BINARY |cv2.THRESH_OTSU,image)
    image = cv2.bitwise_not(image)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    res = cv2.bitwise_and(image, org)

    # cv2.imshow("Simple Thresh", res)
    # cv2.waitKey(0)
    return res


def paperMethod(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    org = image.copy()

    # Negate the image
    image = cv2.bitwise_not(image)

    # Find sobel edge of the image
    image = sobelEdge(image)

    # Gaussian blurring the result from sobel edge to smooth out outliers
    image = cv2.GaussianBlur(image, (5,5),5)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.threshold(image, 8, 255, cv2.THRESH_BINARY, image)
    # image = cv2.floodFill(image,None,None,(255,255,255),0,0)

    res = cv2.bitwise_and(org, image)
    image = cv2.bitwise_not(image)
    res = cv2.bitwise_or(image,res)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # cv2.imshow("Negated image", res)
    # cv2.waitKey(0)
    return res


def exploreBGRemoval(path, SAVE=False):
    folder = glob.glob(path + "/*.jpg")

    for file in folder:
        image = cv2.imread(file, 1)
        # GlobablAdaptiveThreshold(image)
        res = paperMethod(image)
        cv2.imshow("BG Removed", res)
        k = cv2.waitKey(0)

        if SAVE:
            res = np.hstack((image, res))
            cv2.imwrite(file+"__bg_Removed.jpg",res)


if __name__ == "__main__":
    image = cv2.imread("G:/Filters/BG/p2.jpg",1)
    # GlobablAdaptiveThreshold(image)
    # print image.shape
    # paperMethod(image)
    # sobelEdge(image)

    path = "G:/Filters/BG"
    exploreBGRemoval(path, SAVE=True)
