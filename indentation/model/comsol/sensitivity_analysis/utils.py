import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import pickle
from os import listdir
from os.path import isfile, join
import pandas as pd
import indentation.model.comsol.sensitivity_analysis.extract_data_from_raw_files as exd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
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


def export_data_output_as_txt(filename, mat_Z, vec_time, vec_pos_axis):
    filename_time = filename[:-4] + '_time.txt'
    filename_axis = filename[:-4] + '_axis.txt'
    filename_Z = filename[:-4] + '_Z.txt'
    np.savetxt(get_path_to_processed_data() / filename_time, vec_time)
    np.savetxt(get_path_to_processed_data() / filename_axis, vec_pos_axis)
    np.savetxt(get_path_to_processed_data() / filename_Z, mat_Z)


def extract_recovery_data_from_pkl(filename):
    complete_pkl_filename = get_path_to_processed_data() / filename
    with open(complete_pkl_filename, "rb") as f:
        [filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export] = pickle.load(f)
    return filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export


def export_A_data_as_pkl(complete_pkl_filename, ids_list, date_dict, Umax_dict, def_dict, thickness_dict, deltad_dict, deltadstar_dict, dmin_dict,A_dict, failed_dict ):
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [ids_list, date_dict, Umax_dict, def_dict, thickness_dict, deltad_dict, deltadstar_dict, dmin_dict,A_dict, failed_dict],
            f,
        )
        
def extract_A_data_from_pkl():
    pkl_filename = '0_locations_recoveries_A.pkl'
    complete_pkl_filename = get_path_to_processed_data() / pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [ids_list, date_dict, Umax_dict, def_dict, thickness_dict, deltad_dict, deltadstar_dict, dmin_dict,A_dict, failed_dict] = pickle.load(f)
    return ids_list, date_dict, Umax_dict, def_dict, thickness_dict, deltad_dict, deltadstar_dict, dmin_dict,A_dict, failed_dict



def transform_csv_input_A_into_pkl(csv_filename):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_csv_filename = path_to_processed_data + "/" + csv_filename
    ids_list = []
    date_dict = {}
    Umax_dict = {}
    def_dict = {}
    thickness_dict = {}
    deltad_dict = {}
    deltadstar_dict = {}
    dmin_dict = {}
    A_dict = {}
    failed_dict = {}
    datas = pd.read_csv(complete_csv_filename, delimiter=';', header=0)
    datas.fillna(method='bfill', inplace=True)
    ids = datas.Id
    ids_list = ids.tolist()
    dates = datas.date
    umaxs = datas.Umax
    defs = datas.strain    
    deltads = datas.deltad
    deltadstars = datas.deltadstar
    dmin = datas.dmin
    A = datas.A
    faileds = datas.failed
    date_dict = {ids_list[i]: dates.tolist()[i] for i in range(len(ids_list))}
    Umax_dict = {ids[i]: umaxs.tolist()[i] for i in range(len(ids_list))}
    def_dict = {ids[i]: defs.tolist()[i] for i in range(len(ids_list))}
    thickness_list = [u/d for u,d in zip(Umax_dict.values(), def_dict.values())]
    thickness_dict = {ids[i]:thickness_list[i] for i in range(len(ids_list))}
    deltad_dict = {ids_list[i]: deltads.tolist()[i] for i in range(len(ids_list))}
    deltadstar_dict = {ids_list[i]: deltads.tolist()[i]/umaxs.tolist()[i] for i in range(len(ids_list))}
    dmin_dict = {ids_list[i]: dmin.tolist()[i] for i in range(len(ids_list))}
    A_dict = {ids_list[i]: A.tolist()[i] for i in range(len(ids_list))}
    failed_dict = {ids_list[i]: faileds.tolist()[i] for i in range(len(ids_list))}
    pkl_filename = csv_filename[:-4] + '.pkl'
    complete_pkl_filename = get_path_to_processed_data() / pkl_filename
    export_A_data_as_pkl(complete_pkl_filename, ids_list, date_dict, Umax_dict, def_dict, thickness_dict, deltad_dict, deltadstar_dict, dmin_dict,A_dict, failed_dict )
    
    
    
    

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