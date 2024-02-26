"""
This file contains functions that are used several times in the folder.
They mostly intent to locate the path to save or pick objects from the repository.
Many of the function do not contain documentation, however their name is 
descriptive enough.
"""

import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import pickle


def get_current_path(): 
    """
    Return the current path

    Parameters:
        ----------
        None

    Returns:
        -------
        current_path: str (Path)
            Path format string

    """
    current_path = Path.cwd() / 'indentation\caracterization\large_tension'
    return current_path

def reach_data_path(date):
    """
    Returns the path where the data is stored

    Parameters:
        ----------
        date: string
            date at which the experiments have been performed
            Format :
                YYMMDD

    Returns:
        -------
        path_to_data: str (Path)
            Path format string

    """
    current_path = get_current_path() / 'raw_data'
    path_to_data = current_path / date
    return path_to_data

def get_path_to_processed_data():
    """
    Returns the path where the processed data is stored

    Parameters:
        ----------
        None

    Returns:
        -------
        path_to_processed_data: str (Path)
            Path format string

    """
    current_path = get_current_path() 
    path_to_processed_data = current_path / 'processed_data'
    return path_to_processed_data

def get_path_to_figures():
    """
    Returns the path where the processed data is stored

    Parameters:
        ----------
        None

    Returns:
        -------
        path_to_processed_data: str (Path)
            Path format string

    """
    current_path = get_current_path() 
    path_to_processed_data = current_path / 'figures'
    return path_to_processed_data

def find_nearest(array, value):
    array = np.asarray(array)
    # diff_indices = np.zeros_like(array)
    diff_list = np.zeros_like(array)

    for i in range(len(diff_list)):
        if ~np.isnan(array[i]):
            diff = abs(array[i] - value)
            diff_list[i] = diff
            # diff_index = i
            # diff_indices[i] = diff_index
        else:
            # diff_indices[i] = nan
            diff_list[i] = nan
    idx = np.nanargmin(diff_list)
    return array[idx]


def find_extrema_in_vector(vector):
    """
    Finds the extrema (minima and maxima) values in a vector
    
    Parameters:
        ----------
        vector: vector
            vector of which the extrema values are to be found

    Returns:
        -------
        local_maxima: list
            list of the indices of the maximal values
        local_minima: list
            list of the indices of the minimal values

    """
    local_maxima = []
    local_minima = []

    for i in range(1, len(vector) - 1):
        if vector[i - 1] < vector[i] and vector[i + 1] < vector[i]:
            local_maxima.append(i)
        elif vector[i - 1] > vector[i] and vector[i + 1] > vector[i]:
            local_minima.append(i)

    return local_maxima, local_minima

def export_optimization_params_as_pkl(datafile, sheet, params_opti, minimization_method, suffix):
    pkl_filename = datafile[0:6] + "_" + sheet + "_optimization_results_" + minimization_method + "_" + suffix + ".pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(params_opti, f)

def extract_optimization_params_from_pkl(datafile, sheet, minimization_method, suffix):
    pkl_filename = datafile[0:6] + "_" + sheet + "_optimization_results_" + minimization_method + "_" + suffix + ".pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        params_opti = pickle.load(f)
    return params_opti
        
def extract_data_per_steps(datafile, sheet):
    pkl_filename = datafile[0:6] + "_" + sheet + "step_data.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict] = pickle.load(f)
    return load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict