# Directories
dir_wsl_to_edrive = "/mnt/e/OneDrive/Documents/UNIVERSITY/PhD/WellcomeTrust/rotation2"    # for use only when using WSL on my Windows 10 home PC
dir_linux = ""    # main code directory
dir_main = dir_wsl_to_edrive
dir_figures = "{0}/figures".format(dir_main)
dir_slides_raw = "{0}/slides_raw".format(dir_main)    # pathology slides (.svs)
dir_image_data = "{0}/image_data".format(dir_main)    # processed slide images (.png)
dir_slide_info = "{0}/slide_info".format(dir_main)    # slide score spreadsheets

# Constants
default_zoom = 2    # OpenSlide zoom level
canny_p1 = 50    # Circle detection Canny parameters
canny_p2 = 30
hct_minr = 130    # Hough circle transform radii at default_zoom=2
hct_maxr = 150
image_dim = 224    # Size of square image for learning data, pixels
bg_gs_val = 230    # Greyscale value of the slide background colour
min_bg_area = int(0.01*image_dim**2)    # Pixel area range of background contours
max_bg_area = int(image_dim**2)         # to detect in square images
greyscale_img = True    # Use greyscale for deep learning image data
