import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile, find_peaks
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit, minimize, rosen, rosen_der
from scipy.integrate import quad
from numba import  prange
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle

def store_peaks_information(datafile, sheet):
    """Finds the initial and final indices and elongation values of each step and stores them into a dictionnary

    Args:
        datafile (_type_): _description_
        sheet (_type_): _description_

    Returns:
        _type_: _description_
    """
    times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_peaks(datafile, sheet)
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    index_init_step_dict = {}
    index_final_step_dict = {}
    elongation_init_step_dict = {}
    elongation_final_step_dict = {}
    for p in range(len(elongation_steps)-1):
        elongation_step = elongation_steps[p]
        elongation_init_step = elongation_step
        elongation_final_step = elongation_step+0.1
        index_step = np.where(np.logical_and(elongation<elongation_final_step, elongation>elongation_init_step))[0]
        index_init_step = index_step[0]
        index_final_step = index_step[-1]
        index_init_step_dict[p] = index_init_step
        index_final_step_dict[p] = index_final_step
        elongation_init_step_dict[p] = elongation_init_step
        elongation_final_step_dict[p] = elongation_final_step
    return index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict
 
 
def extract_response_step(datafile, sheet, step_number):
    """Catches the values of time, elongation and stress during the step number step_number

    Args:
        datafile (_type_): _description_
        sheet (_type_): _description_
        step_number (_type_): _description_

    Returns:
        _type_: _description_
    """
    index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict = store_peaks_information(datafile, sheet)
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    step_index_init = index_init_step_dict[step_number] 
    step_index_final = index_final_step_dict[step_number] 
    elongation_during_step = elongation[step_index_init:step_index_final]
    time_during_step = time[step_index_init:step_index_final]
    stress_during_step = stress[step_index_init:step_index_final]
    if step_number == 0 :
        stress_after_offset_indices_during_step = np.where(stress_during_step>1)
        stress_during_step = stress_during_step[stress_after_offset_indices_during_step] - stress_during_step[stress_after_offset_indices_during_step][0]
        time_during_step = time_during_step[stress_after_offset_indices_during_step] - time_during_step[stress_after_offset_indices_during_step][0]
        elongation_during_step = elongation_during_step[stress_after_offset_indices_during_step] - elongation_during_step[stress_after_offset_indices_during_step][0] +1
    return elongation_during_step, time_during_step, stress_during_step

def store_responses_of_steps(datafile, sheet):
    index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict = store_peaks_information(datafile, sheet)
    elongation_steps = index_init_step_dict.keys()
    elongation_list_during_steps_dict = {}
    time_list_during_steps_dict = {}
    stress_list_during_steps_dict = {}
    for p in elongation_steps:
        elongation_during_step, time_during_step, stress_during_step = extract_response_step(datafile, sheet, p)
        elongation_list_during_steps_dict[p] = elongation_during_step
        time_list_during_steps_dict[p] = time_during_step
        stress_list_during_steps_dict[p] = stress_during_step
    return elongation_list_during_steps_dict, time_list_during_steps_dict, stress_list_during_steps_dict

def store_and_export_step_data(datafile, sheet):
    index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict = store_peaks_information(datafile, sheet)
    elongation_list_during_steps_dict, time_list_during_steps_dict, stress_list_during_steps_dict = store_responses_of_steps(datafile, sheet)
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + sheet + "_step_information.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [index_init_step_dict, index_final_step_dict, elongation_init_step_dict,
             elongation_final_step_dict, elongation_list_during_steps_dict,
             time_list_during_steps_dict, stress_list_during_steps_dict
             ],
            f,
        )


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '230727'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = sheets_list_with_data[0]
    # index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict = store_peaks_information(datafile, sheet1)
    # elongation_list_during_steps_dict, time_list_during_steps_dict, stress_list_during_steps_dict = store_responses_of_steps(datafile, sheet1)
    store_and_export_step_data(datafile, sheet1)
    print('hello')