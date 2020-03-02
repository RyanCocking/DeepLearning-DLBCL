import openslide as opsl
import cv2
import numpy as np

import parameters as parm
from load_slide import get_spreadsheet_info, print_slide_metadata
from edge_detection import detect_circles
from crop_slides import create_grid, extract_sample_images

# Slide IDs of ABC gene
print("Obtaining slide IDs of ABC data...")
abc_BCL2_slide_id, abc_cMYC_slide_id, abc_sample_refs = get_spreadsheet_info(
    "{0}/ALL_REMoDL-B_TMA_ABC_RT.xlsx".format(parm.dir_slide_data))

print("BCL-2 slide IDs:   ", abc_BCL2_slide_id[:])
print("")

# Loop over two slides of BCL-2
for slide_id in abc_BCL2_slide_id[:2]:

    # Pointer to slide object
    print("Opening whole slide object {0}...".format(slide_id))
    my_slide = opsl.OpenSlide("{0}/{1}.svs".format(parm.dir_slides_raw, slide_id))

    print("Reading slide region...")
    zoom = parm.default_zoom

    slide_px_width  = int(
        my_slide.properties["openslide.level[{0}].width".format(zoom)]) 
    slide_px_height = int(
        my_slide.properties["openslide.level[{0}].height".format(zoom)])

    # microns per pixel
    mpp = float(my_slide.properties["openslide.mpp-x"])
    print("Microns per pixel = {0:.3f}".format(mpp))

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

    mean_radius = np.mean(radii)

    # Magnification factor
    mag = float(
        my_slide.properties["openslide.level[{0}].downsample".format(zoom)])

    # Sort centres by x coordinate (?) --> used in grid creation
    # centres.view('uint16,uint16').sort(order=['f1'], axis=0)

    for i, c in enumerate(centres):
        extract_sample_images(c, radii[i], parm.image_dim, mag,
        parm.dir_slides_cropped, slide_id, i, my_slide)
   
    print("Closing whole slide object {0}...".format(slide_id))
    print("")
    my_slide.close()

    quit()

print("Done")
