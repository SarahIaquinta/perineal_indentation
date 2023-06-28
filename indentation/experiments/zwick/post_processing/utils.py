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
    current_path = Path.cwd() / 'indentation\experiments\zwick'
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

# def define_filename(filename, format):
#     path_to_processed_data = get_path_to_processed_data()
#     pkl_name = filename[:-4] + format
#     complete_pkl_filename = path_to_processed_data / pkl_name
#     return complete_pkl_filename

# def export_data_output_as_pkl(filename, mat_Z, vec_time, vec_pos_axis):
#     complete_pkl_filename = define_filename(filename, '.pkl')
#     with open(complete_pkl_filename, "wb") as f:
#         pickle.dump(
#             [mat_Z, vec_time, vec_pos_axis],
#             f,
#         )
        
# def extract_data_from_pkl(filename):
#     complete_pkl_filename = define_filename(filename, '.pkl')
#     with open(complete_pkl_filename, "rb") as f:
#         [mat_Z, vec_time, vec_pos_axis] = pickle.load(f)
#     return mat_Z, vec_time, vec_pos_axis
        
# def export_data_output_as_txt(filename, mat_Z, vec_time, vec_pos_axis):
#     filename_time = filename[:-4] + '_time.txt'
#     filename_axis = filename[:-4] + '_axis.txt'
#     filename_Z = filename[:-4] + '_Z.txt'
#     np.savetxt(get_path_to_processed_data() / filename_time, vec_time)
#     np.savetxt(get_path_to_processed_data() / filename_axis, vec_pos_axis)
#     np.savetxt(get_path_to_processed_data() / filename_Z, mat_Z)

# def get_existing_processed_data():
#     path_to_processed_data = get_path_to_processed_data()
#     all_existing_filenames = {f for f in listdir(path_to_processed_data) if isfile(join(path_to_processed_data, f))}
#     all_existing_pkl = [i[0:-4] for i in all_existing_filenames if i.endswith('.pkl')]
#     return all_existing_pkl

def find_nearest(array, value):
    array = np.asarray(array)
    # diff_indices = np.zeros_like(array)
    diff_list = np.zeros_like(array)
    for i in range(len(diff_list)):
        if ~np.isnan(array[i]):
            diff = np.abs(array[i] - value)
            # diff_index = i
            diff_list[i] = diff
            # diff_indices[i] = diff_index
        else:
            # diff_indices[i] = nan
            diff_list[i] = nan
    idx = np.nanargmin(diff_list)
    return array[idx]