# Routines that alter the slide contents
# Later: slide pre-processing routines
import cv2
import numpy as np
import math
import openslide as opsl

def extract_sample_images(centre, radius, mag_factor, img_size, gene, stain,
    slide_id, sample_ref, dir_path, slide_object, grey_out, bg_func, bg_args):
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

    # # Save square sample image as PNG
    # slide_image = slide_object.read_region(location=(cx, cy), level=0,
    #     size=(sq_size, sq_size))
    # filename="id{0}_ref{1}_sample".format(slide_id, sample_ref)
    # slide_image.save("{0}/{1}.png".format(dir_path, filename))

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
                # Convert from PIL to OpenCV image format
                BGR_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                discard = bg_func(BGR_img, *bg_args)

                if discard is False:
                    filename = "{0}_{1}_id{2}_ref{3}_j{4}_i{5}".format(
                        gene, stain, slide_id, sample_ref, j, i)
                    if grey_out is True:
                        img = img.convert("LA")
                    img.save("{0}/{1}.png".format(dir_path, filename))
                    num_images += 1
                else:
                    num_discards += 1
    
    return num_images, num_discards

