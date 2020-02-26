# Edge detection using OpenCV
import cv2
import numpy as np

def detect_edges(in_file, out_file="edges.png", p1=50, p2=200):
    """
    Detect the edges of an image within two tolerance parameters.
    Save as a PNG.
    """
    img = cv2.imread(in_file)
    edges = cv2.Canny(img, p1, p2)  # Canny edge detection algorithm

    cv2.imwrite(out_file, edges)  # Black = space, white = edges
    print("Saved PNG to {0}".format(out_file))

def remove_duplicate_circles(centres, radii):
    """
    Fix double-counted sample centres.

    arguments:
        centres: Nx2 integer array, coordinates of circle centres detected with
                 the Hough transform
        radii: N integer array, corresponding radii

    returns:
        centres array, with double-counts removed
        corresponding radii
    """

    indices = []
    for i in range(centres.shape[0]):
        for j in range(centres.shape[0]):
            if i != j:

                print(i, j)

                rad1 = radii[i]
                rad2 = radii[j]
                d = np.linalg.norm(centres[j] - centres[i])

                # If the centres are too close, mark for deletion whichever
                # has the smaller corresponding radius
                if d < 2*rad1:
                    # Current
                    if rad2 <= rad1:
                        indices.append(i)
                    # Neighbour
                    else:
                        indices.append(j)

        # TODO: Still needs some work

        print(indices)

def detect_circles(in_file, out_file="circles.png", p1=50, p2=30, minr=50, maxr=80):
    """
    Detect circles in an image read from a slide object. Save as a PNG.
    Return circle centres and radii.

    Default values assume zoom level 3.

    arguments:
        p1, p2: Canny edge detection parameters
        minr, maxr: Integer radii of circles to detect, in pixels.

    returns:
        centres: Nx2 float array, unordered circle centres
        radii: N float array, corresponding circle radii
    """

    cimg = cv2.imread(in_file)
    img = cv2.medianBlur(cimg, ksize=13)  # Blur to remove noise (select ksize carefully)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Algorithm accepts grey images

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
        param1=p1, param2=p2, minRadius=minr, maxRadius=maxr)

    # Will throw complaint here if no circles detected.

    # draw circles on the original unblurred image
    circles = np.array(np.around(circles), dtype='int16')
    centres = []
    radii = []
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        
        centres.append([i[0],i[1]])
        radii.append(i[2])

    cv2.imwrite(out_file, cimg)
    print("Saved PNG to {0}".format(out_file))

    centres = np.array(centres)
    radii = np.array(radii)

    remove_duplicate_circles(centres, radii)

    return centres, radii
