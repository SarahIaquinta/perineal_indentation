import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pandas as pd
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick

def compute_i(files_zwick, datafile, sheet):
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    max_disp = np.max(disp)
    i = max_force / max_disp
    return i

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230407', '230331', '230411', '230327', '230403']
    types_of_essay = ['C_Indentation_relaxation_500N_force.xlsx']#,'C_Indentation_relaxation_maintienFnulle_500N_trav.xls',  'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    correct_sheets_in_data = files_zwick.find_only_correct_sheets_in_datafile(datafile)