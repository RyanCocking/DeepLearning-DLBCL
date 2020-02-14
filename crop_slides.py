# Routines that alter the slide contents
# Later: slide pre-processing routines
import cv2
import numpy as np

def create_grid(centres, core_radius, width, height):
    """
    Based on circles detected by cv2.HoughCircles(), find gradient of core 
    centres for a slide and generate a rectangular grid that encompasses 
    every core.

    arguments:
        centres: Nx2 integer array, core centres sorted by y column (pixels)
        core_radius: integer, approximate radius of every core (pixels)
        width: integer, width of slide (no. cores)
        height: integer, height of slide (no.cores)

    returns:
        grid: Mx2 integer array, grid vertices corresponding to core
              centres (pixels)
    """

    # Find gradient of top layer of circles, since circle detection
    # algorithm is imperfect.

    top = []
    # c0 = np.min(centres)
    for c in centres:
        if c

    if top.size < 2 or top.size > width:
        print("ERROR - Incorrect number of points for gradient calculation")
        stop

    # dx = centres[0][-1] - centres[0][0]
    # dy = centres[1][-1] - centres[1][0]

    m = dy / dx


def extract_core():
    """
    Bound a circular core in a square box, then save as a PNG file.
    """
