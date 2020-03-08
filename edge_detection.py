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

def detect_background(BGR_img, bg_grey_val, min_area, max_area):
    """
    Detect large regions of off-white background colour in a colour image.

    arguments:
        BGR_img: (B, G, R) image file obtained by reading a PNG file with
                 the cv2.imread() method, or by converting a PIL image
                 from the (R, G, B) format.
        bg_grey_val: Greyscale value of the background colour
        min_area, max_area: pixel range in which contour detection occurs

    returns:
        has_bg: Boolean, returns True if any contours are detected.
    """

    has_bg = False

    # BGR_img = cv2.imread(in_file)

    grey = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2GRAY)
    ret, grey = cv2.threshold(grey, bg_grey_value, 255, cv2.THRESH_BINARY)
    mask = np.zeros(grey.shape, np.uint8)

    contours, hier = cv2.findContours(grey, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if min_area < cv2.contourArea(cnt) < max_area:
            has_bg = True
            # cv2.drawContours(BGR_img, [cnt], 0, (0,255,0), 2)
            # cv2.drawContours(mask, [cnt], 0, 255, -1)
    
    # cv2.imwrite(out_file, grey)

    return has_bg

def remove_duplicate_circles(centres, radii):
    """
    Fix over-counted sample centres by comparing distances from sample i
    to its neighbour j. Neighbours that are within a distance of 2*radius
    are removed.

    NOTE: Made somewhat obsolete by minD parameter in cv2.HoughCircles()

    arguments:
        centres: Nx2 integer numpy array, coordinates of circle centres detected with
                 the Hough transform
        radii: N integer numpy array, corresponding radii

    returns:
        centres array, with extra counts removed
        corresponding radii
    """

    indices = []
    # Loop over centre pairs
    for i in range(0, centres.shape[0] + 1):
        for j in range(i+1, centres.shape[0]):
                d = np.linalg.norm(centres[j] - centres[i])

                # overlap
                if d < 2*radii[i]:
                    # mark for deletion the circle with the
                    # smallest radius
                    if radii[i] < radii[j]:
                        indices.append(i)
                    else:
                        indices.append(j)

    # Account for 3 or more circles near the same sample (NOTE: quick fix)
    indices = np.unique(indices)

    # Delete marked elements
    centres = np.delete(centres, indices, axis=0)
    radii = np.delete(radii, indices)

    return centres[:], radii[:]


def detect_circles(in_file, out_file="circles.png", min_dist=80, p1=50, p2=30,
    minr=50, maxr=80):
    """
    Detect circles in an image read from a slide object. Save as a PNG.
    Return circle centres and radii.

    Default values assume zoom level 3.

    arguments:
        min_dist: integer, minimum distance between detected circles, in pixels.
        p1: integer, gradient value for edge detection
        p2: integer, accumulator threshold for HOUGH_GRADIENT method. A smaller
            value means more circles (including false ones) are detected.
        minr, maxr: integer, radii of circles to detect, in pixels.

    returns:
        centres: Nx2 float array, unordered circle centres
        radii: N float array, corresponding circle radii
    """

    cimg = cv2.imread(in_file)
    img = cv2.medianBlur(cimg, ksize=13)  # Blur to remove noise (k = aperture size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Algorithm accepts grey images

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=min_dist,
        param1=p1, param2=p2, minRadius=minr, maxRadius=maxr)

    # NOTE: Will crash and return error here if no circles detected.

    circles = np.array(np.around(circles), dtype='int16')
    centres = []
    radii = []
    for i in circles[0,:]:
        centres.append([i[0],i[1]])
        radii.append(i[2])

    centres = np.array(centres)
    radii = np.array(radii)

    for i, c in enumerate(centres):
        # draw the outer circle (on unblurred image)
        cv2.circle(cimg,(c[0],c[1]),radii[i],(0,255,0),4)
        # draw the center of the circle
        cv2.circle(cimg,(c[0],c[1]),2,(0,0,255),4)

    cv2.imwrite(out_file, cimg)
    print("Saved PNG to {0}".format(out_file))

    return centres[:], radii[:]
