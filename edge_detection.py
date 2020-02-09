# Edge detection using OpenCV
import cv2
import numpy as np
import matplotlib.pyplot as plt
import png
import parameters as parm

def detect_edges(in_file, out_file="edges.png", p1=50, p2=200):
    """
    Detect the edges of an image within two tolerance parameters.
    Save as a figure.
    """
    img = cv2.imread(in_file)
    edges = cv2.Canny(img, p1, p2)  # Canny edge detection algorithm

    cv2.imwrite(out_file, edges)  # Black = space, white = edges


def detect_circles(in_file, out_file="circles.png", p1=50, p2=30, minr=50, maxr=80):
    """
    Detect circles in an image. Save as a figure.
    """

    cimg = cv2.imread(in_file)
    cimg = cv2.medianBlur(cimg, 5)  # Blur to remove noise
    img = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)  # Algorithm accepts grey images

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=p1,param2=p2,minRadius=minr,maxRadius=maxr)

    # Will throw complaint here if no circles detected.

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imwrite(out_file, cimg)
