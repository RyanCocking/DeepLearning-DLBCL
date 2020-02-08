# Load in pathology slides using OpenSlide
import openslide as opsl
import parameters as parm
import pandas as pd

def get_slide_ids(filename):
    """
    Load Excel spreadsheet and extract slide IDs for BCL-2 and c-MYC
    """
    # Load pathology data from .xlsx file as a dict of sheets
    print("Loading slide data...")
    df = pd.read_excel(filename, None)    # pandas DataFrame

    BCL2_slide_id=[]
    cMYC_slide_id=[]

    for sheetname in df.keys():
        sheet = df[sheetname]    # can be printed to display table contents

        col_d = sheet.columns[3]    # columns D and H
        col_h = sheet.columns[7]

        BCL2_slide_id.append(sheet[col_d][2])    # slide ID (raw image names)
        cMYC_slide_id.append(sheet[col_h][2])

    return BCL2_slide_id, cMYC_slide_id

abc_BCL2_slide_id, abc_cMYC_slide_id = get_slide_ids("{0}/ALL_REMoDL-B_TMA_ABC_RT.xlsx".format(parm.dir_slide_data))

print("BCL-2 slides: ", abc_BCL2_slide_id)
print("c-MYC slides: ", abc_cMYC_slide_id)

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

