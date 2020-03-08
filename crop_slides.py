# Routines that alter the slide contents
# Later: slide pre-processing routines
import cv2
import numpy as np
import math
import openslide as opsl

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

    NOTE: unfinished

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


def extract_sample_images(centre, radius, mag_factor, img_size, slide_id,
    sample_ref, dir_path, slide_object, grey_out, bg_func, bg_args):
    """
    Inscribe a square region within a circular sample, then extract square
    images from it.
    """

    centre = np.multiply(centre, mag_factor)
    centre = np.array(centre, dtype='int32')
    radius *= mag_factor
    radius = int(round(radius))

    # Inscribe square within sample
    x_offset = int(-radius*np.cos(0.25*np.pi))
    y_offset = int(radius*np.sin(0.25*np.pi))
    cx = centre[0] + x_offset
    cy = centre[1] - y_offset
    sq_size = int(math.sqrt(2.0)*radius)
    sq_count = int(math.floor((sq_size*sq_size) / (img_size*img_size)))
    dim = int(math.floor(sq_size / img_size))

    # Save square sample image as PNG
    slide_image = slide_object.read_region(location=(cx, cy), level=0,
        size=(sq_size, sq_size))
    filename="id{0}_ref{1}_sample".format(slide_id, sample_ref)
    slide_image.save("{0}/{1}.png".format(dir_path, filename))

    # Extract images from sample
    num_images = 0
    num_discards = 0
    for j in range(dim):
        for i in range(dim):
            if num_images <= sq_count:
                # Window to scan over inscribed square region
                x = i*img_size + cx
                y = j*img_size + cy

                img = slide_object.read_region(location=(x, y), 
                    level=0, size=(img_size, img_size))
                BGR_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                discard = bg_func(BGR_img, *bg_args)

                if discard is False:
                    filename = "id{0}_ref{1}_j{2}_i{3}_output".format(
                        slide_id, sample_ref, j, i)
                    if grey_out is True:
                        img = img.convert("LA")
                    img.save("{0}/{1}.png".format(dir_path, filename))
                    num_images += 1
                else:
                    num_discards += 1
    
    return num_images, num_discards

