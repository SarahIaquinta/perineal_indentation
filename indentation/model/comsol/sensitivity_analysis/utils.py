"""
This file contains functions that are used several times in the folder.
They mostly intent to locate the path to save or pick objects from the repository.
Many of the function do not contain documentation, however their name is 
descriptive enough.
"""

import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import indentation.model.comsol.sensitivity_analysis.extract_data_from_raw_files as exd
import pickle
import numpy as np
import openturns as ot

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
    current_path = Path.cwd() / 'indentation\model\comsol\sensitivity_analysis'
    return current_path

def reach_data_path():
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
    path_to_data = get_current_path() / 'raw_data'
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

def export_inputs_as_pkl():
    ids_list, elongation_dict, damage_dict = exd.get_inputs()
    complete_pkl_filename = get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [ids_list, elongation_dict, damage_dict],
            f,
        )
        
def export_stress_as_pkl():
    time_list, stress_dict = exd.get_stress()
    complete_pkl_filename = get_path_to_processed_data() / "stress.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [time_list, stress_dict],
            f,
        )

def export_disp_as_pkl():
    time_list, disp_dict = exd.get_disp()
    complete_pkl_filename = get_path_to_processed_data() / "disp.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [time_list, disp_dict],
            f,
        )

def extract_inputs_from_pkl():
    complete_pkl_filename = get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
    return ids_list, elongation_dict, damage_dict

def extract_stress_from_pkl():
    complete_pkl_filename = get_path_to_processed_data() / "stress.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [time_list, stress_dict] = pickle.load(f)
    return time_list, stress_dict
       
def extract_disp_from_pkl():
    complete_pkl_filename = get_path_to_processed_data() / "disp.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [time_list, disp_dict] = pickle.load(f)
    return time_list, disp_dict 

def export_gridsearch(scaler, grid, complete_filename):
    """
    Exports the objects to the metamodel .pkl

    Parameters:
        ----------
        y_test: array
            test data used to validate the ANN
        y_pred: array
            predictions of the ANN
        Q2: float
            predictivity factor obtained with y_test and y_pred, computed with sklearn

    Returns:
        -------
        None
    """
    with open(get_path_to_processed_data() / complete_filename, "wb") as f:
        pickle.dump(
            [scaler, grid],
            f,
        )

def extract_gridsearch(complete_filename):
    """
    Exports the objects to the metamodel .pkl

    Parameters:
        ----------
        y_test: array
            test data used to validate the ANN
        y_pred: array
            predictions of the ANN
        Q2: float
            predictivity factor obtained with y_test and y_pred, computed with sklearn

    Returns:
        -------
        None
    """
    with open(get_path_to_processed_data() / complete_filename, "rb") as f:
        [scaler, grid] = pickle.load(f)
    return scaler, grid

def export_predictions(y_test, y_pred, Q2, complete_filename):
    """
    Exports the objects to the metamodel .pkl

    Parameters:
        ----------
        y_test: array
            test data used to validate the ANN
        y_pred: array
            predictions of the ANN
        Q2: float
            predictivity factor obtained with y_test and y_pred, computed with sklearn

    Returns:
        -------
        None
    """
    with open(get_path_to_processed_data() / complete_filename, "wb") as f:
        pickle.dump(
            [y_test, y_pred, Q2],
            f,
        )
        
def compute_output_ANN_inverse_model(x):
    complete_filename_grid = "grid_search_inverse.pkl"
    sc_X, grid = extract_gridsearch(complete_filename_grid)
    # sc_X = StandardScaler()
    input = ot.Sample(1, 5)
    output_reshaped = ot.Sample(1, 1)
    for i in range(5):
        input[0, i] = x[i]
    # input[0, 0] = x
    X_testscaled=sc_X.transform(input)
    output = grid.predict(X_testscaled)
    output_reshaped[0, 0] = output[0]
    # print(output_reshaped)
    y = [output[0]]
    # print('hello')
    return y