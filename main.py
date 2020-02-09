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

# Open slide object, read region, save as PNG
print("Opening slide object...")
my_slide = opsl.OpenSlide("{0}/393932.svs".format(parm.dir_slides_raw))  # pointer to slide object
# print_slide_metadata(my_slide)

# smaller zoom level = more zoomed-in image
print("Reading slide region...")
slide_image = my_slide.read_region(location=(4100,4200), level=3, 
    size=(400,400))    # (x,y) of top-left corner, zoom level, (w,h) pixels

slide_image.save("{0}/slide_test.png".format(parm.dir_figures))  # save slide object as png
print("Saved PNG to {0}/slide_test.png".format(parm.dir_figures))

# Find circular cores within an image 
print("Detecting circles in image...")
circle_img = detect_circles(in_file="{0}/slide_test.png".format(parm.dir_figures),
    out_file="{0}/circles.png".format(parm.dir_figures), p1=50, p2=30, minr=50, maxr=80)

print("Closing slide object...")
my_slide.close()

print("Done")
