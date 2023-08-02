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
            diff = np.abs(array[i] - value)
            # diff_index = i
            diff_list[i] = diff
            # diff_indices[i] = diff_index
        else:
            # diff_indices[i] = nan
            diff_list[i] = nan
    idx = np.nanargmin(diff_list)
    return array[idx]