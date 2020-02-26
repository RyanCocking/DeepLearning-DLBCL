# Routines that alter the slide contents
# Later: slide pre-processing routines
import cv2
import numpy as np

def compute_row_gradient(row):
    """
    arguments:
        row: Nx2 numpy array, coordinates of circle centres lying
             in roughly the same horizontal region

    returns:
        scalar float: gradient of the line passing through the first and last
        coordinates of row
    """

    dx = row[0][0] - row[-1][0]
    dy = row[0][1] - row[-1][1]
                                                                      
    if dx == 0:
        print("ERROR - Division by zero in gradient calculation.")
        quit()

    return float(dy) / float(dx)


def create_grid(centres, sample_radius, width, height):
    """
    Based on circles detected by cv2.HoughCircles(), find gradient of sample 
    centres for a slide and generate a rectangular grid that includes every 
    sample.

    arguments:
        centres: Nx2 integer array, sample centres sorted by y column (pixels)
        sample_radius: integer, approximate radius of every sample (pixels)
        width: integer, width of slide (no. samples)
        height: integer, height of slide (no.samples)

    returns:
        grid: Mx2 integer array, grid vertices corresponding to sample
              centres (pixels)
    """

    # Restructure detected circle centres based on rows
    rows = []
    y = centres[:,1]
    i = 0
    while i < centres.shape[0]:
        # Centres are on the same row if they are within two radii of eachother
        y0 = centres[i][1]
        row = centres[np.where(np.abs(y - y0) < 2*sample_radius)]
        rows.append(row)
        i += row.shape[0]

    # Compute the gradient of each row
    m = []
    for row in rows:
        # Avoid rows with only one sample
        if row.shape[0] < 2:
            continue

        m.append(compute_row_gradient(row))

    # Initial mean gradient
    m = np.array(m, dtype='float64')
    m_mean1 = np.mean(m)
    # Repeat mean calculation, discounting significant outliers
    m_mean2 = np.mean(m[np.where(abs(m) < 1.5*abs(m_mean1))])


def extract_sample():
    """
    Bound a circular sample in a square box, then save as a PNG file.
    """
