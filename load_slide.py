# Load in pathology slides using OpenSlide
import openslide as opsl
import parameters as parm
import pandas as pd
import numpy as np

def get_spreadsheet_info(filename):
    """
    Load Excel spreadsheet and extract info such as slide IDs for BCL-2
    and c-MYC, number of columns and rows, pathology data, etc.

    A 'slide' refers to the collection of circular tissue sample 'cores' 
    contained within a single .svs file. 
    
    A single spreadsheet contains several sheets, each containing info on 
    one or more slides.

    arguments:
        filename: .svs file path

    returns:
        BCL2_slide_id: N int array, 7-digit slide IDs for BCL-2 staining
        cMYC_slide_id: N int array, 7-digit slide IDs for c-MYC staining
        core_refs: NxMx2 list, alphanumeric core references for N slides
                   of M cores. Both stains use the same references. Indexed as
                   [slide, core, col/row]

    """
    # Load .xlsx file as a dict of sheets
    print("Loading slide data from spreadsheet...")
    df = pd.read_excel(filename, None)    # pandas DataFrame

    BCL2_slide_id=[]
    cMYC_slide_id=[]
    core_refs = []

    # sheets within a spreadsheet
    for sheetname in df.keys():
        sheet = df[sheetname]    # can be printed to display table contents

        col_d = sheet.columns[3]    # spreadsheet columns D and H
        col_h = sheet.columns[7]

        BCL2_slide_id.append(sheet[col_d][2])    # slide ID (raw image names)
        cMYC_slide_id.append(sheet[col_h][2])

        col_b = sheet.columns[1] 
        col_c = sheet.columns[2]

        core_cols = np.array(sheet[col_c][4:])
        core_rows = np.array(sheet[col_b][4:])

        core_refs.append(np.transpose([core_cols, core_rows]))
        
    return BCL2_slide_id, cMYC_slide_id, core_refs

def print_slide_metadata(slide_object):
    """
    OpenSlide slide object properties and values. Accessed like
    a dictionary.
    """

    print("Slide metadata:")
    props = slide_object.properties
    for key, value in props.items():
        print("  ", key, " = ", value)

