import openslide as opsl
import cv2
import numpy as np
import sys
import os

import parameters as parm
from load_slide import get_spreadsheet_info, print_slide_metadata
from edge_detection import detect_circles, detect_background
from crop_slides import extract_sample_images

def setup_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

gene = sys.argv[1]
if gene != "ABC" and gene != "GCB":
    print("ERROR - Invalid input argument")
    quit()

print("Setting up output directories...")
setup_dir(parm.dir_figures)
setup_dir("{0}/{1}".format(parm.dir_image_data, gene))

# Obtain slide IDs
print("Obtaining slide IDs of {0:s} data...".format(gene))
BCL2_slide_id, cMYC_slide_id, sample_refs = get_spreadsheet_info(
    "{0:s}/ALL_REMoDL-B_TMA_{1:s}_RT.xlsx".format(parm.dir_slide_info, gene))

print("BCL2 slide IDs:   ", BCL2_slide_id[:])
print("cMYC slide IDs:   ", cMYC_slide_id[:])

stain = "BCL2"
# Loop over two slides of BCL-2
for slide_id in BCL2_slide_id[:]:

    # Pointer to slide object
    print("Opening whole slide object {0}...".format(slide_id))
    my_slide = opsl.OpenSlide("{0}/{1}.svs".format(parm.dir_slides_raw, slide_id))

    print("Reading slide region...")
    zoom = parm.default_zoom
    slide_px_width  = int(
        my_slide.properties["openslide.level[{0}].width".format(zoom)]) 
    slide_px_height = int(
        my_slide.properties["openslide.level[{0}].height".format(zoom)])

    # Microns per pixel
    mpp = float(my_slide.properties["openslide.mpp-x"])
    print("Microns per pixel = {0:.3f}".format(mpp))

    # Read and save whole slide image
    slide_image = my_slide.read_region(location=(0, 0), level=zoom, 
        size=(slide_px_width, slide_px_height))    
    slide_image.save("{0}/{1}_whole_slide.png".format(
        parm.dir_figures, slide_id))
    print("Saved PNG to {0}/{1}_whole_slide.png".format(
        parm.dir_figures, slide_id))

    # Find circular samples within an image 
    print("Detecting circles in image...")
    centres, radii = detect_circles(
        in_file="{0}/{1}_whole_slide.png".format(parm.dir_figures, slide_id),
        out_file="{0}/{1}_hct_circles.png".format(parm.dir_figures, slide_id),
        min_dist=2*parm.hct_minr, p1=parm.canny_p1, p2=parm.canny_p2,
        minr=parm.hct_minr, maxr=parm.hct_maxr)

    # Magnification factor
    mag = float(
        my_slide.properties["openslide.level[{0}].downsample".format(zoom)])

    # Extract images for deep learning input
    print("Extracting images from samples...")
    num_images = 0
    num_discards = 0
    for i, c in enumerate(centres):
        print("Sample {0} of {1}".format(i+1, centres.shape[0]), end="\r")
        j, k = extract_sample_images(c, radii[i], mag, parm.image_dim, gene,
        stain, slide_id, i, "{0}/{1}".format(parm.dir_image_data, gene), 
        my_slide, parm.gs_output, detect_background,
        (parm.bg_gs_val, parm.min_bg_area, parm.max_bg_area))

        num_images += j
        num_discards += k

    print("Extracted {0} images".format(num_images))
    print("{0} images discarded".format(num_discards))

    print("Closing whole slide object {0}...".format(slide_id))
    print("")
    my_slide.close()

print("Done")
