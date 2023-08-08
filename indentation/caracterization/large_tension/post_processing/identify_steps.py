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
    times_end_load_peak, stress_end_load_peak, elongation_end_load_peak, end_load_peak_indices, time_beginning_load_peak, stress_beginning_load_peak, elongation_beginning_load_peak, beginning_load_peak_indices = find_peaks(datafile, sheet)
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    index_init_step_dict_load = {}
    index_final_step_dict_load = {}
    elongation_init_step_dict_load = {}
    elongation_final_step_dict_load = {}
    for p in range(len(elongation_beginning_load_peak)-1):
        # elongation_step = elongation_beginning_load_peak[p]
        elongation_init_step_load = elongation_beginning_load_peak[p]
        elongation_final_step_load = elongation_end_load_peak[p]
        index_step_load = np.arange(beginning_load_peak_indices[p], end_load_peak_indices[p], 1)
        index_init_step_load = index_step_load[0]
        index_final_step_load = index_step_load[-1]
        index_init_step_dict_load[p] = index_init_step_load
        index_final_step_dict_load[p] = index_final_step_load
        elongation_init_step_dict_load[p] = elongation_init_step_load
        elongation_final_step_dict_load[p] = elongation_final_step_load

    index_init_step_dict_relaxation = {}
    index_final_step_dict_relaxation = {}
    elongation_init_step_dict_relaxation = {}
    elongation_final_step_dict_relaxation = {}
    for p in range(len(elongation_end_load_peak)):
        # elongation_step = elongation_beginning_relaxation_peak[p]
        elongation_init_step_relaxation = elongation_end_load_peak[p]
        elongation_final_step_relaxation = elongation_beginning_load_peak[p+1]
        index_step_relaxation = np.arange(end_load_peak_indices[p], beginning_load_peak_indices[p+1], 1)
        index_init_step_relaxation = index_step_relaxation[0]
        index_final_step_relaxation = index_step_relaxation[-1]
        index_init_step_dict_relaxation[p] = index_init_step_relaxation
        index_final_step_dict_relaxation[p] = index_final_step_relaxation
        elongation_init_step_dict_relaxation[p] = elongation_init_step_relaxation
        elongation_final_step_dict_relaxation[p] = elongation_final_step_relaxation

    return index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation
 
 
def extract_response_step(datafile, sheet, step_number):
    """Catches the values of time, elongation and stress during the step number step_number

    Args:
        datafile (_type_): _description_
        sheet (_type_): _description_
        step_number (_type_): _description_

    Returns:
        _type_: _description_
    """
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation = store_peaks_information(datafile, sheet)
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    
    step_index_init_load = index_init_step_dict_load[step_number] 
    step_index_final_load = index_final_step_dict_load[step_number] 
    elongation_during_step_load = elongation[step_index_init_load:step_index_final_load]
    time_during_step_load = time[step_index_init_load:step_index_final_load]
    stress_during_step_load = stress[step_index_init_load:step_index_final_load]
    # if step_number == 0 :
    #     stress_after_offset_indices_during_step_load = np.where(stress_during_step_load>1)
    #     stress_during_step_load = stress_during_step_load[stress_after_offset_indices_during_step_load] - stress_during_step_load[stress_after_offset_indices_during_step_load][0]
    #     time_during_step_load = time_during_step_load[stress_after_offset_indices_during_step_load] - time_during_step_load[stress_after_offset_indices_during_step_load][0]
    #     elongation_during_step_load = elongation_during_step_load[stress_after_offset_indices_during_step_load] - elongation_during_step_load[stress_after_offset_indices_during_step_load][0] +1

    step_index_init_relaxation = index_init_step_dict_relaxation[step_number] 
    step_index_final_relaxation = index_final_step_dict_relaxation[step_number] 
    elongation_during_step_relaxation = elongation[step_index_init_relaxation:step_index_final_relaxation]
    time_during_step_relaxation = time[step_index_init_relaxation:step_index_final_relaxation]
    stress_during_step_relaxation = stress[step_index_init_relaxation:step_index_final_relaxation]

    return elongation_during_step_load, time_during_step_load, stress_during_step_load, elongation_during_step_relaxation, time_during_step_relaxation, stress_during_step_relaxation

def store_responses_of_steps(datafile, sheet):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation = store_peaks_information(datafile, sheet)

    elongation_steps_load = index_init_step_dict_load.keys()
    elongation_list_during_steps_dict_load = {}
    time_list_during_steps_dict_load = {}
    stress_list_during_steps_dict_load = {}
    for p in elongation_steps_load:
        elongation_during_step_load, time_during_step_load, stress_during_step_load, _, _, _ = extract_response_step(datafile, sheet, p)
        elongation_list_during_steps_dict_load[p] = elongation_during_step_load
        time_list_during_steps_dict_load[p] = time_during_step_load
        stress_list_during_steps_dict_load[p] = stress_during_step_load

    elongation_steps_relaxation = index_init_step_dict_relaxation.keys()
    elongation_list_during_steps_dict_relaxation = {}
    time_list_during_steps_dict_relaxation = {}
    stress_list_during_steps_dict_relaxation = {}
    for p in elongation_steps_relaxation:
        _, _, _, elongation_during_step_relaxation, time_during_step_relaxation, stress_during_step_relaxation = extract_response_step(datafile, sheet, p)
        elongation_list_during_steps_dict_relaxation[p] = elongation_during_step_relaxation
        time_list_during_steps_dict_relaxation[p] = time_during_step_relaxation
        stress_list_during_steps_dict_relaxation[p] = stress_during_step_relaxation
    return elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation



def store_and_export_step_data(datafile, sheet):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation = store_peaks_information(datafile, sheet)
    elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = store_responses_of_steps(datafile, sheet)
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + datafile[0:6] + "_" + sheet + "_step_information.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load,
             elongation_final_step_dict_load, elongation_list_during_steps_dict_load,
             time_list_during_steps_dict_load, stress_list_during_steps_dict_load,
             index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation,
             elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation,
             time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation
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
    # index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation = store_peaks_information(datafile, sheet1)
    # elongation_list_during_steps_dict, time_list_during_steps_dict, stress_list_during_steps_dict = store_responses_of_steps(datafile, sheet1)
    store_and_export_step_data(datafile, sheet1)
    print('hello')