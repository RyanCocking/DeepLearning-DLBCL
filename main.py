import openslide as opsl
import cv2
import numpy as np

import parameters as parm
from load_slide import get_slide_ids, print_slide_metadata
from edge_detection import detect_circles

# Slide IDs of ABC gene
print("Obtaining slide IDs of ABC data...")
abc_BCL2_slide_id, abc_cMYC_slide_id = get_slide_ids("{0}/ALL_REMoDL-B_TMA_ABC_RT.xlsx".format(parm.dir_slide_data))
print("BCL-2 slide IDs: ", abc_BCL2_slide_id)
print("c-MYC slide IDs: ", abc_cMYC_slide_id)

# BCL-2
for slide_id in abc_BCL2_slide_id[0:2]:

    print("Slide ID = {0}".format(slide_id))
    print("==================")

    # Open slide object, read region, save as PNG
    print("Opening slide object...")
    my_slide = opsl.OpenSlide("{0}/{1}.svs".format(parm.dir_slides_raw, slide_id))  # pointer to slide object
    # print_slide_metadata(my_slide)

    # xdim = slide_metadata.level[zoom].width

    print("Reading slide region...")
    zoom = 3
    xcorner = 0
    ycorner = 0
    xdim = 2000
    ydim = 2000
    slide_image = my_slide.read_region(location=(xcorner, ycorner), level=zoom, 
        size=(xdim, ydim))    # (x,y) of top-left corner, zoom level (0 = max zoom), (w,h) pixels

    slide_image.save("{0}/slide_test.png".format(parm.dir_figures))  # save slide object as png
    print("Saved PNG to {0}/{1}_slide_test.png".format(parm.dir_figures, slide_id))

    # Find circular cores within an image 
    print("Detecting circles in image...")
    centres, radii = circle_img = detect_circles(in_file="{0}/slide_test.png".format(parm.dir_figures),
        out_file="{0}/{1}_circles.png".format(parm.dir_figures, slide_id), p1=50, p2=30, minr=50, maxr=80)

    print(centres)
    print(radii)

    print("Closing slide object...")
    my_slide.close()

print("Done")
