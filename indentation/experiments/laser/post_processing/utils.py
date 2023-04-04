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
    current_path = Path.cwd() / 'indentation\experiments\laser'
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


def get_path_to_processe_data():
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

def define_filename(filename, format):
    path_to_processed_data = get_path_to_processe_data()
    pkl_name = filename[:-4] + format
    complete_pkl_filename = path_to_processed_data / pkl_name
    return complete_pkl_filename



def export_data_output_as_pkl(filename, mat_Z, vec_time, vec_pos_axis):
    complete_pkl_filename = define_filename(filename, '.pkl')
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [mat_Z, vec_time, vec_pos_axis],
            f,
        )
        
def extract_data_from_pkl(filename):
    complete_pkl_filename = define_filename(filename, '.pkl')
    with open(complete_pkl_filename, "rb") as f:
        [mat_Z, vec_time, vec_pos_axis] = pickle.load(f)
    return mat_Z, vec_time, vec_pos_axis
        
        
def export_data_output_as_txt(filename, mat_Z, vec_time, vec_pos_axis):
    np.savetxt(filename + '_time.txt', vec_time)
    np.savetxt(filename + '_axis.txt', vec_pos_axis)
    np.savetxt(filename + '_Z.txt', mat_Z)
        

    