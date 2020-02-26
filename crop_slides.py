# Routines that alter the slide contents
# Later: slide pre-processing routines
import cv2
import numpy as np

def create_grid(centres, sample_radius, width, height):
    """
    Based on circles detected by cv2.HoughCircles(), find gradient of sample 
    centres for a slide and generate a rectangular grid that passes through
    every sample.

    arguments:
        centres: Nx2 integer array, sample centres sorted by y column (pixels)
        sample_radius: integer, approximate radius of every sample (pixels)
        width: integer, width of slide (no. samples)
        height: integer, height of slide (no.samples)

    returns:
        grid: Mx2 integer array, grid vertices corresponding to sample
              centres (pixels)
    """

    # TODO: The following can probably be done much more neatly, perhaps
    # with np.reshape()?
    # Structure detected circle centres into rows
    rows = []
    y = centres[:,1]
    j = 0
    for i in range(centres.shape[0]):
        # Centres are on the same row if they are within two radii of eachother
        y0 = centres[i][1]
        row = centres[np.where(np.abs(y - y0) < 2*sample_radius)]
        j += 1

        # Append the row once the last centre in it has been found
        if j == row.shape[0]:
            rows.append(row)
            j = 0

    print(rows)

    # Compute the average row gradient; the slope of the line going through 
    # the centres of the first and last sample on each row.
    m = []
    for row in rows:
        if row.shape[0] < 2:
            continue

        dx = row[0][0] - row[-1][0]
        dy = row[0][1] - row[-1][1]

        if dx == 0:
            print("WARNING - Potential division by zero in gradient "
                "calculation of create_grid function. Gradient ignored.")
            continue

        m.append(float(dy) / float(dx))

    m_mean = np.mean(m)

    print(m)
    print(m_mean)


def extract_sample():
    """
    Bound a circular sample in a square box, then save as a PNG file.
    """
