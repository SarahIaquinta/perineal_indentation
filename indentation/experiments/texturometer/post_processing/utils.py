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
from os import listdir
from os.path import isfile, join
import pandas as pd

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
    current_path = Path.cwd() / 'indentation/experiments/texturometer'
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

def define_filename(filename, format):
    path_to_processed_data = get_path_to_processed_data()
    pkl_name = filename + format
    complete_pkl_filename = path_to_processed_data / pkl_name
    return complete_pkl_filename

def export_texturometer_data_as_pkl(complete_pkl_filename, ids_list, date_dict, force20_dict, force80_dict, failed_dict):
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [ids_list, date_dict, force20_dict, force80_dict, failed_dict],
            f,
        )
        
def extract_texturometer_data_from_pkl():
    pkl_filename = 'texturometer_forces.pkl'
    complete_pkl_filename = get_path_to_processed_data() / pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [ids_list, date_dict, force20_dict, force80_dict, failed_dict] = pickle.load(f)
    return ids_list, date_dict, force20_dict, force80_dict, failed_dict
        
def extract_data_from_pkl(filename):
    complete_pkl_filename = define_filename(filename, '.pkl')
    with open(complete_pkl_filename, "rb") as f:
        [mat_Z, vec_time, vec_pos_axis] = pickle.load(f)
    return mat_Z, vec_time, vec_pos_axis
        
def export_data_output_as_txt(filename, mat_Z, vec_time, vec_pos_axis):
    filename_time = filename[:-4] + '_time.txt'
    filename_axis = filename[:-4] + '_axis.txt'
    filename_Z = filename[:-4] + '_Z.txt'
    np.savetxt(get_path_to_processed_data() / filename_time, vec_time)
    np.savetxt(get_path_to_processed_data() / filename_axis, vec_pos_axis)
    np.savetxt(get_path_to_processed_data() / filename_Z, mat_Z)

def get_existing_processed_data():
    path_to_processed_data = get_path_to_processed_data()
    all_existing_filenames = {f for f in listdir(path_to_processed_data) if isfile(join(path_to_processed_data, f))}
    all_existing_pkl = [i[0:-4] for i in all_existing_filenames if i.endswith('.pkl')]
    return all_existing_pkl

def extract_recovery_data_from_pkl(filename):
    complete_pkl_filename = get_path_to_processed_data() / filename
    with open(complete_pkl_filename, "rb") as f:
        [filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export] = pickle.load(f)
    return filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export

def transform_csv_input_into_pkl(csv_filename):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_csv_filename = path_to_processed_data + "/" + csv_filename
    ids_list = []
    date_dict = {}
    force20_dict = {}
    force80_dict = {}
    failed_dict = {}
    datas = pd.read_csv(complete_csv_filename, delimiter=';', header=0)
    ids = datas.Id
    ids_list = ids.tolist()
    dates = datas.date
    force20s = datas.F20
    force80s = datas.F80
    faileds = datas.failed
    date_dict = {ids_list[i]: str(dates.tolist()[i]) for i in range(len(ids_list))}
    force20_dict = {ids_list[i]: force20s.tolist()[i] for i in range(len(ids_list))}
    force80_dict = {ids_list[i]: force80s.tolist()[i] for i in range(len(ids_list))}
    failed_dict = {ids_list[i]: faileds.tolist()[i] for i in range(len(ids_list))}
    pkl_filename = csv_filename[:-4] + '.pkl'
    complete_pkl_filename = get_path_to_processed_data() / pkl_filename
    export_texturometer_data_as_pkl(complete_pkl_filename, ids_list, date_dict, force20_dict, force80_dict, failed_dict)


