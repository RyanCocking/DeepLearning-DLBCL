# Load in pathology slides using OpenSlide
import openslide as opsl
import parameters as parm
import pandas as pd
import numpy as np

def get_spreadsheet_info(filename):
    """
    Load Excel spreadsheet and extract info such as slide IDs for BCL-2
    and c-MYC, number of columns and rows, pathology data, etc.

    A 'slide' refers to the collection of circular tissue sample 'samples' 
    contained within a single .svs file. 
    
    A single spreadsheet contains several sheets, each containing info on 
    one or more slides.

    arguments:
        filename: .svs file path

    returns:
        BCL2_slide_id: int array, 7-digit slide IDs for BCL-2 staining
        cMYC_slide_id: int array, 7-digit slide IDs for c-MYC staining
        num_BCL2, num_cMYC: scalars, number of samples over all slides

    """
    # Load .xlsx file as a dict of sheets
    print("Loading slide data from spreadsheet...")
    df = pd.read_excel(filename, None)    # pandas DataFrame

    BCL2_slide_id=[]
    cMYC_slide_id=[]
    sample_refs = []

    num_BCL2 = 0    # Total number of samples in all slides
    num_cMYC = 0

    # sheets within a spreadsheet
    for sheetname in df.keys():
        sheet = df[sheetname]    # can be printed to display table contents

        col_d = sheet.columns[3]    # spreadsheet columns D and H
        col_h = sheet.columns[7]

        BCL2_slide_id.append(sheet[col_d][2])    # slide ID (raw image names)
        cMYC_slide_id.append(sheet[col_h][2])

        col_b = sheet.columns[1] 
        col_c = sheet.columns[2]

        sample_cols = np.array(sheet[col_c][4:])    # A, B, C, etc
        BCL2_tumour = np.array(sheet[col_d][4:])    # pathology tumour scores
        cMYC_tumour = np.array(sheet[col_h][4:])

        # check columns d and h (tumour scores) for NA entries
        cols, counts = np.unique(sample_cols, return_counts=True)
        total_samples = np.sum(counts)    # Includes all N/A entries

        entries = np.unique(BCL2_tumour)    # returns nan for non-numerical entry
        nan_mask = np.isnan(entries.astype('float64'))
        num_NA = np.count_nonzero(nan_mask) - 4    # All slides bar one have 4 control samples
        num_BCL2 += total_samples - num_NA

        entries = np.unique(cMYC_tumour)
        nan_mask = np.isnan(entries.astype('float64'))
        num_NA = np.count_nonzero(nan_mask) - 4
        num_cMYC += total_samples - num_NA

    return BCL2_slide_id, cMYC_slide_id, num_BCL2, num_cMYC


def print_slide_metadata(slide_object):
    """
    OpenSlide slide object properties and values. Accessed like
    a dictionary.
    """

    print("Slide metadata:")
    props = slide_object.properties
    for key, value in props.items():
        print("  ", key, " = ", value)

