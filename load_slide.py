# Load in pathology slides using OpenSlide
import openslide as opsl
import parameters as parm
import pandas as pd

def get_slide_ids(filename):
    """
    Load Excel spreadsheet and extract slide IDs for BCL-2 and c-MYC
    """
    # Load pathology data from .xlsx file as a dict of sheets
    print("Loading slide data from spreadsheet...")
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

def print_slide_metadata(slide_object):
    """
    OpenSlide slide object properties and values. Accessed like
    a dictionary.
    """

    print("Slide metadata:")
    props = slide_object.properties
    for key, value in props.items():
        print("  ", key, " = ", value)

