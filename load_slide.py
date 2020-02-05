# Load in pathology slides using OpenSlide
import openslide as opsl
import parameters as parm
import pandas as pd

def get_slide_ids(file_name):
    """
    Load Excel spreadsheet and extract slide IDs for BCL-2 and c-MYC
    """
    # Load pathology data from .xlsx file as a dict of sheets
    print("Loading slide data...")
    df = pd.read_excel(io=file_name, sheet_name=None)    # pandas DataFrame

    BCL2_slides=[]
    cMYC_slides=[]

    # df.keys() contains spreadsheet sheet names
    for key in df.keys():
        sheet = df[key]    # can be printed to display full table

        col_d = sheet.columns[3]    # columns D and H
        col_h = sheet.columns[7]

        BCL2_slides.append(sheet[col_d][2])    # slide ID (raw image names)
        cMYC_slides.append(sheet[col_h][2])

    return BCL2_slides, cMYC_slides

BCL2_slides, cMYC_slides = get_slide_ids("{0}/ALL_REMoDL-B_TMA_ABC_RT.xlsx".format(parm.dir_slide_data))

print("BCL-2 slides: ", BCL2_slides)
print("c-MYC slides: ", cMYC_slides)

print("Opening slide object...")
my_slide = opsl.OpenSlide("{0}/393930.svs".format(parm.dir_slides_raw))  # pointer to slide object

print("Reading slide region...")
slide_image = my_slide.read_region(location=(5000,5000), level=1, size=(1000,1000))  # (x,y) of top-left corner, zoom level, (w,h) pixels

print("Saving slide object as PNG...")
slide_image.save("{0}/slide_test.png".format(parm.dir_figures))  # save slide object as png

# now we need to find the (x,y) positions of the cores!

for i in range(len(dir(my_slide))):
    break
    print(dir(my_slide)[i])

print(my_slide.level_dimensions)    # pixel size of image at zoom levels
print(my_slide.level_downsamples)   # x magnification value

props = my_slide.properties

for key, value in props.items():
    break
    print(key," = ", value)

print("Closing slide object...")
my_slide.close()
print("Done")

