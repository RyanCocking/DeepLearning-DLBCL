import openslide as opsl
import cv2
import numpy as np

import parameters as parm
from load_slide import get_spreadsheet_info, print_slide_metadata
from edge_detection import detect_circles
from crop_slides import create_grid

# Slide IDs of ABC gene
print("Obtaining slide IDs of ABC data...")
abc_BCL2_slide_id, abc_cMYC_slide_id, abc_sample_refs = get_spreadsheet_info(
    "{0}/ALL_REMoDL-B_TMA_ABC_RT.xlsx".format(parm.dir_slide_data))

print("BCL-2 slide IDs:   ", abc_BCL2_slide_id[:])
print("")

# Loop over two slides of BCL-2
for slide_id in abc_BCL2_slide_id[:2]:

    # Pointer to slide object
    print("Opening slide {0} object...".format(slide_id))
    my_slide = opsl.OpenSlide("{0}/{1}.svs".format(parm.dir_slides_raw, slide_id))

    print("Reading slide region...")
    zoom = 2

    slide_px_width  = int(
        my_slide.properties["openslide.level[{0}].width".format(zoom)]) 
    slide_px_height = int(
        my_slide.properties["openslide.level[{0}].height".format(zoom)])

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
        min_dist=2*130, p1=50, p2=30, minr=130, maxr=150)

    # Sort centres in-place by y pixel (found on StackOverflow)
    centres.view('uint16,uint16').sort(order=['f1'], axis=0)

    # Generate a rectangular grid based on detected sample centres, to account
    # for undetected samples

    create_grid(centres, 75, 7, 7)

    print("Closing slide {0} object...".format(slide_id))
    print("")
    my_slide.close()

print("Done")
