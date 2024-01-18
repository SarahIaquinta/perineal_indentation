import numpy as np
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pickle
from scipy.signal import lfilter, savgol_filter

thickness_dict = {
    "C3PA" : 2.3,
    "C3TA" : 6,
    "C3SA" : 7,
    "C2TA" : 3,
    "C1TA" : 4
}

def thickness(sheet):
    available_thickness_sheets = list(thickness_dict.keys())
    if sheet in available_thickness_sheets :
        thickness = thickness_dict[sheet]
    else:
        region = sheet[-2]
        available_sheets_region = [s for s in available_thickness_sheets if s[-2] == region]
        available_thickness_region = [thickness_dict[s] for s in available_sheets_region]
        thickness = np.mean(available_thickness_region)
    return thickness

width = 40 #mm

def section(sheet):
    thickness = thickness[sheet]
    section = thickness * width
    return section #mm²

def read_sheet_in_datafile(datafile, sheet):
    """
    Extracts the measured time, force and displacement values in a sheet
    
    Parameters:
        ----------
        datafile: string
            name of the datafile to be read
        sheet: string
            name of the sheet to be read

    Returns:
        -------
        time: pandasArray
            list of the time values (in seconds) in the sheet of the datafile
        force: pandasArray
            list of the force values (in Newtons) in the sheet of the datafile
        disp: pandasArray
            list of the displacement values (in mm) in the sheet of the datafile

    """
    date = datafile[0:6]
    path_to_datafile = large_tension_utils.reach_data_path(date) / datafile
    data_in_sheet = pd.read_excel(path_to_datafile, sheet_name=sheet, header=3, names=["s", "elongation", "MPa" ], usecols="A:C", decimal=',') 
    time = data_in_sheet.s
    elongation = data_in_sheet.elongation
    stress = data_in_sheet.MPa
    time, elongation, stress = time.to_numpy(), elongation.to_numpy(), stress.to_numpy() 
    non_negative_or_null_elongations = np.where(elongation > 0.001)[0]
    rescaled_elongation = np.array([e/100 + 1 for e in elongation[non_negative_or_null_elongations]])
    rescaled_elongation = np.array([e - rescaled_elongation[0] +1 for e in rescaled_elongation])
    stress = [s*2.3/thickness(sheet) for s in stress] #correction épaisseur
    stress = savgol_filter(stress, 101, 2)
    rescaled_stress = np.array([s*1000 - stress[0]*1000 for s in stress[non_negative_or_null_elongations]])
    rescaled_time = time[non_negative_or_null_elongations] - time[non_negative_or_null_elongations][0]
    # TODO conclude on the use of  filter or not
    # n=3
    # b= [1.0 / n] * n
    # a=1
    # rescaled_elongation = savgol_filter(rescaled_elongation, 101, n)
    # time = np.array([t - time[1] for t in time])
    return rescaled_time, rescaled_elongation, rescaled_stress


if __name__ == "__main__":
    sheet1 = "C1PA"
    thickness(sheet1)
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '231012'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = sheets_list_with_data[0]
    print('started')
    for sheet in sheets_list_with_data:
        time, elongation, stress = read_sheet_in_datafile(datafile, sheet1)
    print('hello')