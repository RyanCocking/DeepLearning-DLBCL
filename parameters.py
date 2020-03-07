# Directories
dir_wsl_to_edrive = "/mnt/e/OneDrive/Documents/UNIVERSITY/PhD/WellcomeTrust/rotation2"    # for use only when using WSL on my Windows 10 home PC
dir_linux = ""    # main code directory
dir_main = dir_wsl_to_edrive
dir_figures = "{0}/figures".format(dir_main)
dir_slides_raw = "{0}/slides_raw".format(dir_main)    # pathology slides (.svs)
dir_slides_cropped = "{0}/slides_cropped".format(dir_main)    # processed slide images (.png)
dir_slide_data = "{0}/slide_data".format(dir_main)    # slide info spreadsheets

# Constants
default_zoom = 2    # OpenSlide zoom level

canny_p1 = 50    # Circle detection Canny parameters
canny_p2 = 30
hct_minr = 130    # Hough circle transform radii
hct_maxr = 150    # at zoom=2, minr=130 and maxr=150

image_dim = 448    # Size of square image for learning data, pixels
