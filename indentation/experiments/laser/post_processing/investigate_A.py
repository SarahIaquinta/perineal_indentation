import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.laser.post_processing.read_file as rf
import indentation.experiments.laser.post_processing.display_profiles as dp
from indentation.experiments.laser.post_processing.read_file import Files
import indentation.experiments.texturometer.post_processing.compute_forces as cf
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter
import pickle
import statistics

def remove_failed_A(filenames_from_pkl, dates_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
    """
    Removes the values of A that have not been properly computed or for which the experimental
    testing did not go well
    
    Parameters:
        ----------
        filenames_from_pkl: list
            list of filenames, taken from the pkl file that contains the values of A for each testing
        dates_dict: dict
            dictionnary that associates the date corresponding to each filename 
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        delta_d_star_dict: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename 
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        d_min_dict: dict
            dictionnary that associates the value of d_min corresponding to each filename 
        A_dict: dict
            dictionnary that associates the value of A corresponding to each filename 
        failed_A_acqusitions: list
            list of the filenames of testings for which A could not properly be computed

    Returns:
        -------
        ids_where_not_failed: list
            list of the filenames for which A could be computed properly
        dates_dict_not_failed: dict
            dictionnary that associates the date corresponding to each filename where A could properly be computed
        delta_d_dict_not_failed: dict
            dictionnary that associates the value of delta_d corresponding to each filename where A could properly be computed
        delta_d_star_dict_not_failed: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename where A could properly be computed
        delta_d_dict_not_failed: dict
            dictionnary that associates the value of delta_d corresponding to each filename where A could properly be computed
        d_min_dict_not_failed: dict
            dictionnary that associates the value of d_min corresponding to each filename where A could properly be computed
        A_dict_not_failed: dict
            dictionnary that associates the value of A corresponding to each filename where A could properly be computed

            
    """ 
    ids_where_not_failed = [id for id in filenames_from_pkl if failed_A_acqusitions[id] ==0 and delta_d_dict[id]!='FAILED LASER ACQUISITION']
    date_dict_not_failed = {id: dates_dict[id] for id in ids_where_not_failed}
    delta_d_dict_not_failed = {id: delta_d_dict[id] for id in ids_where_not_failed}
    delta_d_star_dict_not_failed = {id: delta_d_star_dict[id] for id in ids_where_not_failed}
    d_min_dict_not_failed = {id: d_min_dict[id] for id in ids_where_not_failed}
    A_dict_not_failed = {id: A_dict[id] for id in ids_where_not_failed}
    return ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed

def remove_failed_A_and_small_deltad(deltad_threshold, filenames_from_pkl, dates_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
    """
    Removes the values of A that have not been properly computed or for which the experimental
    testing did not go well
    
    Parameters:
        ----------
        deltad_threshold: float
            value of delta_d under which every acquisition is considered failed
        filenames_from_pkl: list
            list of filenames, taken from the pkl file that contains the values of A for each testing
        dates_dict: dict
            dictionnary that associates the date corresponding to each filename 
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        delta_d_star_dict: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename 
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        d_min_dict: dict
            dictionnary that associates the value of d_min corresponding to each filename 
        A_dict: dict
            dictionnary that associates the value of A corresponding to each filename 
        failed_A_acqusitions: list
            list of the filenames of testings for which A could not properly be computed

    Returns:
        -------
        ids_where_not_failed_and_not_small_deltad: list
            list of the filenames for which A could be computed properly and delta_d is above the threshold
        dates_dict_not_failed_and_not_small_deltad: dict
            dictionnary that associates the date corresponding to each filename where A could properly be computed
            and delta_d is above the threshold
        delta_d_dict_not_failed_and_not_small_deltad: dict
            dictionnary that associates the value of delta_d corresponding to each filename where A could properly be computed
            and delta_d is above the threshold
        delta_d_star_dict_not_failed_and_not_small_deltad: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename where A could properly be computed
            and delta_d is above the threshold
        delta_d_dict_not_failed_and_not_small_deltad: dict
            dictionnary that associates the value of delta_d corresponding to each filename where A could properly be computed
            and delta_d is above the threshold
        d_min_dict_not_failed_and_not_small_deltad: dict
            dictionnary that associates the value of d_min corresponding to each filename where A could properly be computed
            and delta_d is above the threshold
        A_dict_not_failed_and_not_small_deltad: dict
            dictionnary that associates the value of A corresponding to each filename where A could properly be computed
            and delta_d is above the threshold

            
    """     
    ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed = remove_failed_A(filenames_from_pkl, dates_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
    ids_where_not_failed_and_not_small_deltad = [id for id in ids_where_not_failed if float(delta_d_dict[id]) > deltad_threshold]
    date_dict_not_failed_and_not_small_deltad = {id: float(date_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    delta_d_dict_not_failed_and_not_small_deltad = {id: float(delta_d_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    delta_d_star_dict_not_failed_and_not_small_deltad = {id: float(delta_d_star_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    d_min_dict_not_failed_and_not_small_deltad = {id: float(d_min_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    A_dict_not_failed_and_not_small_deltad = {id: float(A_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    return ids_where_not_failed_and_not_small_deltad, date_dict_not_failed_and_not_small_deltad, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad

def extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict):
    """
    Extract the data that was measured at a given date and on a given meatpiece
    
    Parameters:
        ----------
        date: str
            date at which the data needs to be extracted
        meatpiece: str
            meatpiece for which the data needs to be extracted
        ids_list: list
            list of all the filenames
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        delta_d_star_dict: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename 
        d_min_dict: dict
            dictionnary that associates the value of d_min corresponding to each filename 
        A_dict: dict
            dictionnary that associates the value of A corresponding to each filename 

    Returns:
        -------
        ids_at_date_and_meatpiece: list
            list of the filenames at the expected date and meatpiece
        delta_d_dict_at_date_and_meatpiece: dict
            dictionnary of the values of delta_d at the expected date and meatpiece
        delta_d_star_dict_at_date_and_meatpiece: dict
            dictionnary of the values of delta_d_star at the expected date and meatpiece
        d_min_dict_at_date_and_meatpiece: dict
            dictionnary of the values of d_min at the expected date and meatpiece
        A_dict_at_date_and_meatpiece: dict
            dictionnary of the values of A at the expected date and meatpiece
            
    """     
    ids_at_date = [id for id in ids_list if date_dict[id] == date]
    ids_at_date_and_meatpiece = [id for id in ids_at_date if id[0:len(str(date)) + 1 + len(meatpiece)] == str(date) + '_' + meatpiece] 
    delta_d_dict_at_date_and_meatpiece = {id: delta_d_dict[id] for id in ids_at_date_and_meatpiece}
    delta_d_star_dict_at_date_and_meatpiece = {id: delta_d_star_dict[id] for id in ids_at_date_and_meatpiece}
    d_min_dict_at_date_and_meatpiece = {id: d_min_dict[id] for id in ids_at_date_and_meatpiece}
    A_dict_at_date_and_meatpiece = {id: A_dict[id] for id in ids_at_date_and_meatpiece}
    return ids_at_date_and_meatpiece, delta_d_dict_at_date_and_meatpiece, delta_d_star_dict_at_date_and_meatpiece, d_min_dict_at_date_and_meatpiece, A_dict_at_date_and_meatpiece

def compute_mean_and_std_at_given_date_and_meatpiece(date, meatpiece, deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
    """
    Compute the mean and the standard deviation of the data measured at a given date and on a given meatpiece
    
    Parameters:
        ----------
        date: str
            date at which the data needs to be extracted
        meatpiece: str
            meatpiece for which the data needs to be extracted
        ids_list: list
            list of all the filenames
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        delta_d_star_dict: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename 
        d_min_dict: dict
            dictionnary that associates the value of d_min corresponding to each filename 
        A_dict: dict
            dictionnary that associates the value of A corresponding to each filename 
        failed_A_acqusitions: list
            list of the filenames of testings for which A could not properly be computed
    Returns:
        -------
        mean_delta_d: float
            mean of delta_d at the given data and meatpiece
        std_delta_d: float
            standard deviation of delta_d at the given data and meatpiece
        mean_delta_d_star: float
            mean of delta_d_star at the given data and meatpiece
        std_delta_d_star: float
            standard deviation of delta_d_star at the given data and meatpiece
        mean_d_min: float
            mean of d_min at the given data and meatpiece
        std_d_min: float
            standard deviation of d_min at the given data and meatpiece
        mean_A: float
            mean of A at the given data and meatpiece
        std_A: float
            standard deviation of A at the given data and meatpiece

            
    """     
    ids_where_not_failed_and_not_small_deltad, _, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad = remove_failed_A_and_small_deltad(deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
    ids_at_date_and_meatpiece, delta_d_dict_at_date_and_meatpiece, delta_d_star_dict_at_date_and_meatpiece, d_min_dict_at_date_and_meatpiece, A_dict_at_date_and_meatpiece = extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_where_not_failed_and_not_small_deltad, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad)
    mean_delta_d, std_delta_d, mean_delta_d_star, std_delta_d_star, mean_d_min, std_d_min, mean_A, std_A = nan, nan, nan, nan, nan, nan, nan, nan
    if len(ids_at_date_and_meatpiece) >0:
        mean_delta_d = statistics.mean(list(delta_d_dict_at_date_and_meatpiece.values()))
        mean_delta_d_star = statistics.mean(list(delta_d_star_dict_at_date_and_meatpiece.values()))
        mean_d_min = statistics.mean(list(d_min_dict_at_date_and_meatpiece.values()))
        mean_A = statistics.mean(list(A_dict_at_date_and_meatpiece.values()))
    if len(ids_at_date_and_meatpiece) >1:
        std_delta_d = statistics.stdev(list(delta_d_dict_at_date_and_meatpiece.values()))
        std_delta_d_star = statistics.stdev(list(delta_d_star_dict_at_date_and_meatpiece.values()))
        std_d_min = statistics.stdev(list(d_min_dict_at_date_and_meatpiece.values()))
        std_A = statistics.stdev(list(A_dict_at_date_and_meatpiece.values()))
    return mean_delta_d, std_delta_d, mean_delta_d_star, std_delta_d_star, mean_d_min, std_d_min, mean_A, std_A

def compute_and_export_indicators_with_maturation_as_pkl(ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions, deltad_threshold):
    """
    Compute the mean and the standard deviation of the data measured during maturation 
    (ie for various dates) for all meatpieces, Exports them as a .pkl file named "indicators_mean_std.pkl"
    
    Parameters:
        ----------
        ids_list: list
            list of all the filenames
        date_dict: dict
            dictionnary that associates the date corresponding to each filename 
        delta_d_dict: dict
            dictionnary that associates the value of delta_d corresponding to each filename 
        delta_d_star_dict: dict
            dictionnary that associates the value of delta_d_star corresponding to each filename 
        d_min_dict: dict
            dictionnary that associates the value of d_min corresponding to each filename 
        A_dict: dict
            dictionnary that associates the value of A corresponding to each filename 
        failed_A_acqusitions: list
            list of the filenames of testings for which A could not properly be computed
    Returns:
        -------
        None

            
    """     
    dates_int = list(set(date_dict.values()))
    dates = [str(d) for d in dates_int]
    mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict = {}, {}, {}, {}
    mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict = {}, {}, {}, {}
    mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict = {}, {}, {}, {}
    mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict = {}, {}, {}, {}
    mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict = {}, {}, {}, {}
    mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict = {}, {}, {}, {}

    mean_d_min_FF1_dict, std_d_min_FF1_dict, mean_A_FF1_dict, std_A_FF1_dict = {}, {}, {}, {}
    mean_d_min_FF2_dict, std_d_min_FF2_dict, mean_A_FF2_dict, std_A_FF2_dict = {}, {}, {}, {}
    mean_d_min_RDG1_dict, std_d_min_RDG1_dict, mean_A_RDG1_dict, std_A_RDG1_dict = {}, {}, {}, {}
    mean_d_min_RDG2_dict, std_d_min_RDG2_dict, mean_A_RDG2_dict, std_A_RDG2_dict = {}, {}, {}, {}
    mean_d_min_FF_dict, std_d_min_FF_dict, mean_A_FF_dict, std_A_FF_dict = {}, {}, {}, {}
    mean_d_min_RDG_dict, std_d_min_RDG_dict, mean_A_RDG_dict, std_A_RDG_dict = {}, {}, {}, {}
    
    for i in range(len(dates)):
        date = dates[i]
        mean_delta_d_FF1_date, std_delta_d_FF1_date, mean_delta_d_star_FF1_date, std_delta_d_star_FF1_date, mean_d_min_FF1_date, std_d_min_FF1_date,  mean_A_FF1_date, std_A_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(int(date), 'FF1', deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF2_date, std_delta_d_FF2_date, mean_delta_d_star_FF2_date, std_delta_d_star_FF2_date, mean_d_min_FF2_date, std_d_min_FF2_date,  mean_A_FF2_date, std_A_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(int(date), 'FF2', deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG1_date, std_delta_d_RDG1_date, mean_delta_d_star_RDG1_date, std_delta_d_star_RDG1_date, mean_d_min_RDG1_date, std_d_min_RDG1_date,  mean_A_RDG1_date, std_A_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(int(date), 'RDG1', deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG2_date, std_delta_d_RDG2_date, mean_delta_d_star_RDG2_date, std_delta_d_star_RDG2_date, mean_d_min_RDG2_date, std_d_min_RDG2_date,  mean_A_RDG2_date, std_A_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(int(date), 'RDG2', deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF_date, std_delta_d_FF_date, mean_delta_d_star_FF_date, std_delta_d_star_FF_date, mean_d_min_FF_date, std_d_min_FF_date,  mean_A_FF_date, std_A_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(int(date), 'FF', deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG_date, std_delta_d_RDG_date, mean_delta_d_star_RDG_date, std_delta_d_star_RDG_date, mean_d_min_RDG_date, std_d_min_RDG_date,  mean_A_RDG_date, std_A_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(int(date), 'RDG', deltad_threshold, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF1_dict[date], std_delta_d_FF1_dict[date], mean_delta_d_star_FF1_dict[date], std_delta_d_star_FF1_dict[date] = mean_delta_d_FF1_date, std_delta_d_FF1_date, mean_delta_d_star_FF1_date, std_delta_d_star_FF1_date
        mean_delta_d_FF2_dict[date], std_delta_d_FF2_dict[date], mean_delta_d_star_FF2_dict[date], std_delta_d_star_FF2_dict[date] = mean_delta_d_FF2_date, std_delta_d_FF2_date, mean_delta_d_star_FF2_date, std_delta_d_star_FF2_date
        mean_delta_d_FF_dict[date], std_delta_d_FF_dict[date], mean_delta_d_star_FF_dict[date], std_delta_d_star_FF_dict[date] = mean_delta_d_FF_date, std_delta_d_FF_date, mean_delta_d_star_FF_date, std_delta_d_star_FF_date
        mean_delta_d_RDG1_dict[date], std_delta_d_RDG1_dict[date], mean_delta_d_star_RDG1_dict[date], std_delta_d_star_RDG1_dict[date] = mean_delta_d_RDG1_date, std_delta_d_RDG1_date, mean_delta_d_star_RDG1_date, std_delta_d_star_RDG1_date
        mean_delta_d_RDG2_dict[date], std_delta_d_RDG2_dict[date], mean_delta_d_star_RDG2_dict[date], std_delta_d_star_RDG2_dict[date] = mean_delta_d_RDG2_date, std_delta_d_RDG2_date, mean_delta_d_star_RDG2_date, std_delta_d_star_RDG2_date
        mean_delta_d_RDG_dict[date], std_delta_d_RDG_dict[date], mean_delta_d_star_RDG_dict[date], std_delta_d_star_RDG_dict[date] = mean_delta_d_RDG_date, std_delta_d_RDG_date, mean_delta_d_star_RDG_date, std_delta_d_star_RDG_date
        mean_d_min_FF_dict[date], std_d_min_FF_dict[date], mean_A_FF_dict[date], std_A_FF_dict[date] = mean_d_min_FF_date, std_d_min_FF_date, mean_A_FF_date, std_A_FF_date
        mean_d_min_RDG_dict[date], std_d_min_RDG_dict[date], mean_A_RDG_dict[date], std_A_RDG_dict[date] = mean_d_min_RDG_date, std_d_min_RDG_date, mean_A_RDG_date, std_A_RDG_date
        mean_d_min_FF1_dict[date], std_d_min_FF1_dict[date], mean_A_FF1_dict[date], std_A_FF1_dict[date] = mean_d_min_FF1_date, std_d_min_FF1_date, mean_A_FF1_date, std_A_FF1_date
        mean_d_min_FF2_dict[date], std_d_min_FF2_dict[date], mean_A_FF2_dict[date], std_A_FF2_dict[date] = mean_d_min_FF2_date, std_d_min_FF2_date, mean_A_FF2_date, std_A_FF2_date
        mean_d_min_RDG1_dict[date], std_d_min_RDG1_dict[date], mean_A_RDG1_dict[date], std_A_RDG1_dict[date] = mean_d_min_RDG1_date, std_d_min_RDG1_date, mean_A_RDG1_date, std_A_RDG1_date
        mean_d_min_RDG2_dict[date], std_d_min_RDG2_dict[date], mean_A_RDG2_dict[date], std_A_RDG2_dict[date] = mean_d_min_RDG2_date, std_d_min_RDG2_date, mean_A_RDG2_date, std_A_RDG2_date
        mean_d_min_FF_dict[date], std_d_min_FF_dict[date], mean_A_FF_dict[date], std_A_FF_dict[date] = mean_d_min_FF_date, std_d_min_FF_date, mean_A_FF_date, std_A_FF_date
        mean_d_min_RDG_dict[date], std_d_min_RDG_dict[date], mean_A_RDG_dict[date], std_A_RDG_dict[date] = mean_d_min_RDG_date, std_d_min_RDG_date, mean_A_RDG_date, std_A_RDG_date
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [dates, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
             mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
             mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
             mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
             mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
             mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict
             ],
            f,
        )
        
def export_indocators_as_txt(deltad_threshold):
    """
    Exports the indicators as a .txt file named "indicators_mean_std_FF1.txt" for meatpiece FF1. The value 'FF1'
    is changed depending on the meatpiece
    
    Parameters:
        ----------
        None
    Returns:
        -------
        None

            
    """     
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates_laser, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
             mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
             mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
             mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
             mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
             mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict
             ] = pickle.load(f)
    
    complete_txt_filename_FF1 = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std_FF1.txt"
    f = open(complete_txt_filename_FF1, "w")
    f.write("INDICATORS FOR FF1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF1_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_FF1_dict[date])
            + "\t"
            + str(std_delta_d_FF1_dict[date])
            + "\t"
            + str(mean_delta_d_star_FF1_dict[date])
            + "\t"
            + str(std_delta_d_star_FF1_dict[date])
            + "\t"
            + str(mean_d_min_FF1_dict[date])
            + "\t"
            + str(std_d_min_FF1_dict[date])
            + "\t"
            + str(mean_A_FF1_dict[date])
            + "\t"
            + str(std_A_FF1_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF2 = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std_FF2.txt"
    f = open(complete_txt_filename_FF2, "w")
    f.write("INDICATORS FOR FF2 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF2_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_FF2_dict[date])
            + "\t"
            + str(std_delta_d_FF2_dict[date])
            + "\t"
            + str(mean_delta_d_star_FF2_dict[date])
            + "\t"
            + str(std_delta_d_star_FF2_dict[date])
            + "\t"
            + str(mean_d_min_FF2_dict[date])
            + "\t"
            + str(std_d_min_FF2_dict[date])
            + "\t"
            + str(mean_A_FF2_dict[date])
            + "\t"
            + str(std_A_FF2_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std_FF.txt"
    f = open(complete_txt_filename_FF, "w")
    f.write("INDICATORS FOR FF \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_FF_dict[date])
            + "\t"
            + str(std_delta_d_FF_dict[date])
            + "\t"
            + str(mean_delta_d_star_FF_dict[date])
            + "\t"
            + str(std_delta_d_star_FF_dict[date])
            + "\t"
            + str(mean_d_min_FF_dict[date])
            + "\t"
            + str(std_d_min_FF_dict[date])
            + "\t"
            + str(mean_A_FF_dict[date])
            + "\t"
            + str(std_A_FF_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG1 = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std_RDG1.txt"
    f = open(complete_txt_filename_RDG1, "w")
    f.write("INDICATORS FOR RDG1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG1_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_RDG1_dict[date])
            + "\t"
            + str(std_delta_d_RDG1_dict[date])
            + "\t"
            + str(mean_delta_d_star_RDG1_dict[date])
            + "\t"
            + str(std_delta_d_star_RDG1_dict[date])
            + "\t"
            + str(mean_d_min_RDG1_dict[date])
            + "\t"
            + str(std_d_min_RDG1_dict[date])
            + "\t"
            + str(mean_A_RDG1_dict[date])
            + "\t"
            + str(std_A_RDG1_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG2 = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std_RDG2.txt"
    f = open(complete_txt_filename_RDG2, "w")
    f.write("INDICATORS FOR RDG2 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG2_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_RDG2_dict[date])
            + "\t"
            + str(std_delta_d_RDG2_dict[date])
            + "\t"
            + str(mean_delta_d_star_RDG2_dict[date])
            + "\t"
            + str(std_delta_d_star_RDG2_dict[date])
            + "\t"
            + str(mean_d_min_RDG2_dict[date])
            + "\t"
            + str(std_d_min_RDG2_dict[date])
            + "\t"
            + str(mean_A_RDG2_dict[date])
            + "\t"
            + str(std_A_RDG2_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std_RDG.txt"
    f = open(complete_txt_filename_RDG, "w")
    f.write("INDICATORS FOR RDG \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_RDG_dict[date])
            + "\t"
            + str(std_delta_d_RDG_dict[date])
            + "\t"
            + str(mean_delta_d_star_RDG_dict[date])
            + "\t"
            + str(std_delta_d_star_RDG_dict[date])
            + "\t"
            + str(mean_d_min_RDG_dict[date])
            + "\t"
            + str(std_d_min_RDG_dict[date])
            + "\t"
            + str(mean_A_RDG_dict[date])
            + "\t"
            + str(std_A_RDG_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_all = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std.txt"
    f = open(complete_txt_filename_all, "w")
    f.write("INDICATORS \n")
    f.write("FF1 \t  FF1 \t  FF1 \t  FF1 \t FF1 \t  FF1 \t  FF1 \t  FF1 \t  FF1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF1_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_FF1_dict[date])
            + "\t"
            + str(std_delta_d_FF1_dict[date])
            + "\t"
            + str(mean_delta_d_star_FF1_dict[date])
            + "\t"
            + str(std_delta_d_star_FF1_dict[date])
            + "\t"
            + str(mean_d_min_FF1_dict[date])
            + "\t"
            + str(std_d_min_FF1_dict[date])
            + "\t"
            + str(mean_A_FF1_dict[date])
            + "\t"
            + str(std_A_FF1_dict[date])
            + "\n"
        )
        
    f.write("FF2 \t  FF2 \t FF2 \t  FF2 \t FF2 \t  FF2 \t  FF2 \t  FF2 \t  FF2 \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF2_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_FF2_dict[date])
            + "\t"
            + str(std_delta_d_FF2_dict[date])
            + "\t"
            + str(mean_delta_d_star_FF2_dict[date])
            + "\t"
            + str(std_delta_d_star_FF2_dict[date])
            + "\t"
            + str(mean_d_min_FF2_dict[date])
            + "\t"
            + str(std_d_min_FF2_dict[date])
            + "\t"
            + str(mean_A_FF2_dict[date])
            + "\t"
            + str(std_A_FF2_dict[date])
            + "\n"
        )
        
    f.write("FF \t  FF \t FF \t  FF \t FF \t  FF \t  FF \t  FF \t  FF \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_FF_dict[date])
            + "\t"
            + str(std_delta_d_FF_dict[date])
            + "\t"
            + str(mean_delta_d_star_FF_dict[date])
            + "\t"
            + str(std_delta_d_star_FF_dict[date])
            + "\t"
            + str(mean_d_min_FF_dict[date])
            + "\t"
            + str(std_d_min_FF_dict[date])
            + "\t"
            + str(mean_A_FF_dict[date])
            + "\t"
            + str(std_A_FF_dict[date])
            + "\n"
        )
        
    f.write("RDG1 \t  RDG1 \t  RDG1 \t RDG1 \t RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG1_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_RDG1_dict[date])
            + "\t"
            + str(std_delta_d_RDG1_dict[date])
            + "\t"
            + str(mean_delta_d_star_RDG1_dict[date])
            + "\t"
            + str(std_delta_d_star_RDG1_dict[date])
            + "\t"
            + str(mean_d_min_RDG1_dict[date])
            + "\t"
            + str(std_d_min_RDG1_dict[date])
            + "\t"
            + str(mean_A_RDG1_dict[date])
            + "\t"
            + str(std_A_RDG1_dict[date])
            + "\n"
        )
        
    f.write("RDG2 \t  RDG2 \t RDG2 \t RDG2 \t RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG2_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_RDG2_dict[date])
            + "\t"
            + str(std_delta_d_RDG2_dict[date])
            + "\t"
            + str(mean_delta_d_star_RDG2_dict[date])
            + "\t"
            + str(std_delta_d_star_RDG2_dict[date])
            + "\t"
            + str(mean_d_min_RDG2_dict[date])
            + "\t"
            + str(std_d_min_RDG2_dict[date])
            + "\t"
            + str(mean_A_RDG2_dict[date])
            + "\t"
            + str(std_A_RDG2_dict[date])
            + "\n"
        )
        
    f.write("RDG \t  RDG \t RDG \t  RDG \t RDG \t  RDG \t  RDG \t  RDG \t  RDG \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG_dict)):
        date = dates_laser[i]
        f.write(
            str(date)
            + "\t"
            + str(mean_delta_d_RDG_dict[date])
            + "\t"
            + str(std_delta_d_RDG_dict[date])
            + "\t"
            + str(mean_delta_d_star_RDG_dict[date])
            + "\t"
            + str(std_delta_d_star_RDG_dict[date])
            + "\t"
            + str(mean_d_min_RDG_dict[date])
            + "\t"
            + str(std_d_min_RDG_dict[date])
            + "\t"
            + str(mean_A_RDG_dict[date])
            + "\t"
            + str(std_A_RDG_dict[date])
            + "\n"
        )
        
        
    f.close()

def plot_recovery_indicators_with_maturation(deltad_threshold):
    """
    Plots the evolution of the recovery indicators with maturation
    
    Parameters:
        ----------
        None
    Returns:
        -------
        None

            
    """         
    dates_to_use = ['230331', '230403', '230407']
    maturation_dict = {'230331':10, '230403':13, '230407':17}
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename = path_to_processed_data + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates_laser, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
             mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
             mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
             mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
             mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
             mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict
             ] = pickle.load(f)
    [dates_laser, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
                mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
                mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
                mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
                mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
                mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict
                ] = [dates_laser, {d:mean_delta_d_FF1_dict[d] for d in dates_to_use}, {d:std_delta_d_FF1_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF1_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF1_dict[d] for d in dates_to_use}, {d:mean_d_min_FF1_dict[d] for d in dates_to_use}, {d:std_d_min_FF1_dict[d] for d in dates_to_use},  {d:mean_A_FF1_dict[d] for d in dates_to_use}, {d:std_A_FF1_dict[d] for d in dates_to_use},
             {d:mean_delta_d_FF2_dict[d] for d in dates_to_use}, {d:std_delta_d_FF2_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF2_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF2_dict[d] for d in dates_to_use}, {d:mean_d_min_FF2_dict[d] for d in dates_to_use}, {d:std_d_min_FF2_dict[d] for d in dates_to_use},  {d:mean_A_FF2_dict[d] for d in dates_to_use}, {d:std_A_FF2_dict[d] for d in dates_to_use},
             {d:mean_delta_d_RDG1_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG1_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG1_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG1_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG1_dict[d] for d in dates_to_use}, {d:std_d_min_RDG1_dict[d] for d in dates_to_use},  {d:mean_A_RDG1_dict[d] for d in dates_to_use}, {d:std_A_RDG1_dict[d] for d in dates_to_use},
             {d:mean_delta_d_RDG2_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG2_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG2_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG2_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG2_dict[d] for d in dates_to_use}, {d:std_d_min_RDG2_dict[d] for d in dates_to_use},  {d:mean_A_RDG2_dict[d] for d in dates_to_use}, {d:std_A_RDG2_dict[d] for d in dates_to_use},
             {d:mean_delta_d_FF_dict[d] for d in dates_to_use}, {d:std_delta_d_FF_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF_dict[d] for d in dates_to_use}, {d:mean_d_min_FF_dict[d] for d in dates_to_use}, {d:std_d_min_FF_dict[d] for d in dates_to_use},  {d:mean_A_FF_dict[d] for d in dates_to_use}, {d:std_A_FF_dict[d] for d in dates_to_use},
             {d:mean_delta_d_RDG_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG_dict[d] for d in dates_to_use}, {d:std_d_min_RDG_dict[d] for d in dates_to_use},  {d:mean_A_RDG_dict[d] for d in dates_to_use}, {d:std_A_RDG_dict[d] for d in dates_to_use}
             ]

    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    
    maturation_FF_dict = {d:maturation_dict[d]-0.1 for d in dates_to_use}
    maturation_RDG_dict = {d:maturation_dict[d]+0.1 for d in dates_to_use}
    fig_delta_d_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_1 = fig_delta_d_1.gca()
    fig_delta_d_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_2 = fig_delta_d_2.gca()
    fig_delta_d = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d = fig_delta_d.gca()
    ax_delta_d.errorbar(list(maturation_FF_dict.values()), list(mean_delta_d_FF_dict.values()), yerr=list(std_delta_d_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_delta_d_1.errorbar(list(maturation_FF_dict.values()), list(mean_delta_d_FF1_dict.values()), yerr=list(std_delta_d_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_2.errorbar(list(maturation_FF_dict.values()), list(mean_delta_d_FF2_dict.values()), yerr=list(std_delta_d_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_1.errorbar(list(maturation_RDG_dict.values()), list(mean_delta_d_RDG1_dict.values()), yerr=list(std_delta_d_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_delta_d.errorbar(list(maturation_RDG_dict.values()), list(mean_delta_d_RDG_dict.values()), yerr=list(std_delta_d_RDG_dict.values()), lw=0,  label='RDG', **kwargs_RDG)
    ax_delta_d_2.errorbar(list(maturation_RDG_dict.values()), list(mean_delta_d_RDG2_dict.values()), yerr=list(std_delta_d_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d.set_title(r'$\Delta d$ vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_1.set_title(r'$\Delta d$ vs maturation 1', font=fonts.serif_rz_legend())
    ax_delta_d_2.set_title(r'$\Delta d$ vs maturation 2', font=fonts.serif_rz_legend())
    ax_delta_d.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    ax_delta_d_1.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    ax_delta_d_2.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_maturation_1+2")
    plt.close(fig_delta_d)        
    savefigure.save_as_png(fig_delta_d_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_maturation_1")
    plt.close(fig_delta_d_1)        
    savefigure.save_as_png(fig_delta_d_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_maturation_2")
    plt.close(fig_delta_d_2)    


    fig_delta_d_star_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star_1 = fig_delta_d_star_1.gca()
    fig_delta_d_star_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star_2 = fig_delta_d_star_2.gca()
    fig_delta_d_star = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star = fig_delta_d_star.gca()
    ax_delta_d_star.errorbar(list(maturation_FF_dict.values()), list(mean_delta_d_star_FF_dict.values()), yerr=list(std_delta_d_star_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_delta_d_star_1.errorbar(list(maturation_FF_dict.values()), list(mean_delta_d_star_FF1_dict.values()), yerr=list(std_delta_d_star_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_star_2.errorbar(list(maturation_FF_dict.values()), list(mean_delta_d_star_FF2_dict.values()), yerr=list(std_delta_d_star_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_star_1.errorbar(list(maturation_RDG_dict.values()), list(mean_delta_d_star_RDG1_dict.values()), yerr=list(std_delta_d_star_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_delta_d_star.errorbar(list(maturation_RDG_dict.values()), list(mean_delta_d_star_RDG_dict.values()), yerr=list(std_delta_d_star_RDG_dict.values()), lw=0,  label='RDG', **kwargs_RDG)
    ax_delta_d_star_2.errorbar(list(maturation_RDG_dict.values()), list(mean_delta_d_star_RDG2_dict.values()), yerr=list(std_delta_d_star_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_star.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star.set_title(r'$\Delta d^*$ vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_title(r'$\Delta d^*$ vs maturation 1', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_title(r'$\Delta d^*$ vs maturation 2', font=fonts.serif_rz_legend())
    ax_delta_d_star.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_maturation_1+2")
    plt.close(fig_delta_d_star)        
    savefigure.save_as_png(fig_delta_d_star_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_maturation_1")
    plt.close(fig_delta_d_star_1)        
    savefigure.save_as_png(fig_delta_d_star_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_maturation_2")
    plt.close(fig_delta_d_star_2)    

    fig_A_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_A_1 = fig_A_1.gca()
    fig_A_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_A_2 = fig_A_2.gca()
    fig_A = createfigure.rectangle_rz_figure(pixels=180)
    ax_A = fig_A.gca()
    ax_A.errorbar(list(maturation_FF_dict.values()), list(mean_A_FF_dict.values()), yerr=list(std_A_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_A_1.errorbar(list(maturation_FF_dict.values()), list(mean_A_FF1_dict.values()), yerr=list(std_A_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_A_2.errorbar(list(maturation_FF_dict.values()), list(mean_A_FF2_dict.values()), yerr=list(std_A_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_A_1.errorbar(list(maturation_RDG_dict.values()), list(mean_A_RDG1_dict.values()), yerr=list(std_A_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_A.errorbar(list(maturation_RDG_dict.values()), list(mean_A_RDG_dict.values()), yerr=list(std_A_RDG_dict.values()), lw=0,  label='RDG', **kwargs_RDG)
    ax_A_2.errorbar(list(maturation_RDG_dict.values()), list(mean_A_RDG2_dict.values()), yerr=list(std_A_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_A.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A.set_title(r'A vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_A_1.set_title(r'A vs maturation 1', font=fonts.serif_rz_legend())
    ax_A_2.set_title(r'A vs maturation 2', font=fonts.serif_rz_legend())
    ax_A.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_A_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_A_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_A.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    ax_A_1.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    ax_A_2.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_maturation_1+2")
    plt.close(fig_A)        
    savefigure.save_as_png(fig_A_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_maturation_1")
    plt.close(fig_A_1)        
    savefigure.save_as_png(fig_A_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_maturation_2")
    plt.close(fig_A_2)    

    fig_d_min_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_d_min_1 = fig_d_min_1.gca()
    fig_d_min_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_d_min_2 = fig_d_min_2.gca()
    fig_d_min = createfigure.rectangle_rz_figure(pixels=180)
    ax_d_min = fig_d_min.gca()
    ax_d_min.errorbar(list(maturation_FF_dict.values()), list(mean_d_min_FF_dict.values()), yerr=list(std_d_min_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_d_min_1.errorbar(list(maturation_FF_dict.values()), list(mean_d_min_FF1_dict.values()), yerr=list(std_d_min_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_d_min_2.errorbar(list(maturation_FF_dict.values()), list(mean_d_min_FF2_dict.values()), yerr=list(std_d_min_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_d_min_1.errorbar(list(maturation_RDG_dict.values()), list(mean_d_min_RDG1_dict.values()), yerr=list(std_d_min_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_d_min.errorbar(list(maturation_RDG_dict.values()), list(mean_d_min_RDG_dict.values()), yerr=list(std_d_min_RDG_dict.values()), lw=0,  label='RDG', **kwargs_RDG)
    ax_d_min_2.errorbar(list(maturation_RDG_dict.values()), list(mean_d_min_RDG2_dict.values()), yerr=list(std_d_min_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_d_min.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_d_min_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_d_min_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_d_min.set_title(r'$d_{min}$ vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_d_min_1.set_title(r'$d_{min}$ vs maturation 1', font=fonts.serif_rz_legend())
    ax_d_min_2.set_title(r'$d_{min}$ vs maturation 2', font=fonts.serif_rz_legend())
    ax_d_min.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_d_min_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_d_min_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_d_min.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    ax_d_min_1.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    ax_d_min_2.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_maturation_1+2")
    plt.close(fig_d_min)        
    savefigure.save_as_png(fig_d_min_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_maturation_1")
    plt.close(fig_d_min_1)        
    savefigure.save_as_png(fig_d_min_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_maturation_2")
    plt.close(fig_d_min_2)    

def plot_laser_indicators_vs_texturometer_forces(deltad_threshold):
    """
    Plots the laser indicators (A, delta_d, delta_d_star, d_min) in terms of the texturometer forces F20 and F80.
    
    Parameters:
        ----------
        None
    Returns:
        -------
        None

            
    """    
    maturation = [10, 13, 17, 21]
    dates_to_use = ['230331', '230403', '230407']
    maturation_dict = {'230331': 10, '230403': 13, '230407': 17}
    maturation_dict_plots = {'230331': 'J+10', '230403': 'J+13', '230407': 'J+17'}
    maturation_FF_dict = {k: v - 0.1 for k, v in maturation_dict.items()}
    maturation_RDG_dict = {k: v + 0.1 for k, v in maturation_dict.items()}
    
    path_to_processed_data_laser = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename_laser = path_to_processed_data_laser + "/0_locations_deltad_threshold" + str(deltad_threshold) + "_indicators_mean_std.pkl"
    with open(complete_pkl_filename_laser, "rb") as f:
        [dates_laser, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
             mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
             mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
             mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
             mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
             mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict
             ] = pickle.load(f)
    [dates_laser, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
                mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
                mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
                mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
                mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
                mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict
                ] = [dates_laser, {d:mean_delta_d_FF1_dict[d] for d in dates_to_use}, {d:std_delta_d_FF1_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF1_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF1_dict[d] for d in dates_to_use}, {d:mean_d_min_FF1_dict[d] for d in dates_to_use}, {d:std_d_min_FF1_dict[d] for d in dates_to_use},  {d:mean_A_FF1_dict[d] for d in dates_to_use}, {d:std_A_FF1_dict[d] for d in dates_to_use},
             {d:mean_delta_d_FF2_dict[d] for d in dates_to_use}, {d:std_delta_d_FF2_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF2_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF2_dict[d] for d in dates_to_use}, {d:mean_d_min_FF2_dict[d] for d in dates_to_use}, {d:std_d_min_FF2_dict[d] for d in dates_to_use},  {d:mean_A_FF2_dict[d] for d in dates_to_use}, {d:std_A_FF2_dict[d] for d in dates_to_use},
             {d:mean_delta_d_RDG1_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG1_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG1_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG1_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG1_dict[d] for d in dates_to_use}, {d:std_d_min_RDG1_dict[d] for d in dates_to_use},  {d:mean_A_RDG1_dict[d] for d in dates_to_use}, {d:std_A_RDG1_dict[d] for d in dates_to_use},
             {d:mean_delta_d_RDG2_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG2_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG2_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG2_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG2_dict[d] for d in dates_to_use}, {d:std_d_min_RDG2_dict[d] for d in dates_to_use},  {d:mean_A_RDG2_dict[d] for d in dates_to_use}, {d:std_A_RDG2_dict[d] for d in dates_to_use},
             {d:mean_delta_d_FF_dict[d] for d in dates_to_use}, {d:std_delta_d_FF_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF_dict[d] for d in dates_to_use}, {d:mean_d_min_FF_dict[d] for d in dates_to_use}, {d:std_d_min_FF_dict[d] for d in dates_to_use},  {d:mean_A_FF_dict[d] for d in dates_to_use}, {d:std_A_FF_dict[d] for d in dates_to_use},
             {d:mean_delta_d_RDG_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG_dict[d] for d in dates_to_use}, {d:std_d_min_RDG_dict[d] for d in dates_to_use},  {d:mean_A_RDG_dict[d] for d in dates_to_use}, {d:std_A_RDG_dict[d] for d in dates_to_use}
             ]

    path_to_processed_data_texturometer = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename_texturometer = path_to_processed_data_texturometer + "/forces_mean_std.pkl"
    with open(complete_pkl_filename_texturometer, "rb") as f:
        [dates_texturometer, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)

    [dates_texturometer, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = [dates_texturometer, {d:mean_force20_FF1_dict[d] for d in dates_to_use}, {d:std_force20_FF1_dict[d] for d in dates_to_use}, {d:mean_force80_FF1_dict[d] for d in dates_to_use}, {d:std_force80_FF1_dict[d] for d in dates_to_use},
             {d:mean_force20_FF2_dict[d] for d in dates_to_use}, {d:std_force20_FF2_dict[d] for d in dates_to_use}, {d:mean_force80_FF2_dict[d] for d in dates_to_use}, {d:std_force80_FF2_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG1_dict[d] for d in dates_to_use}, {d:std_force20_RDG1_dict[d] for d in dates_to_use}, {d:mean_force80_RDG1_dict[d] for d in dates_to_use}, {d:std_force80_RDG1_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG2_dict[d] for d in dates_to_use}, {d:std_force20_RDG2_dict[d] for d in dates_to_use}, {d:mean_force80_RDG2_dict[d] for d in dates_to_use}, {d:std_force80_RDG2_dict[d] for d in dates_to_use},
             {d:mean_force20_FF_dict[d] for d in dates_to_use}, {d:std_force20_FF_dict[d] for d in dates_to_use}, {d:mean_force80_FF_dict[d] for d in dates_to_use}, {d:std_force80_FF_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG_dict[d] for d in dates_to_use}, {d:std_force20_RDG_dict[d] for d in dates_to_use}, {d:mean_force80_RDG_dict[d] for d in dates_to_use}, {d:std_force80_RDG_dict[d] for d in dates_to_use}
             ]
             
    pixels=180
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    labels = {'relaxation_slope' : r"$\alpha_R$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_strain_rate': r"$\i_{25 \%} $ [$Nm^{-1}$]",
                    'i_time_strain_rate': r"$\i_{25 \%} $ [$Ns^{-1}$]",
                    'i_disp_1': r"$\i_{100 \%} $ [$Nm^{-1}$]",
                    'i_time_1': r"$\i_{100 \%} $ [$Ns^{-1}$]"   }    
    
    #A vs force 80    
    
    force_80_1 = np.concatenate((list(mean_force80_FF1_dict.values()), list(mean_force80_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_80_1) 
    A_1 = np.concatenate(( list(mean_A_FF1_dict.values()) , list(mean_A_RDG1_dict.values()) ))
    index_A_1_nan = np.isnan(A_1)
    indices_force_or_A_1_nan = [index_force_1_nan[i] or index_A_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_A_1_nan[i]])
    A_1_without_nan_force = np.array([A_1[i] for i in range(len(indices_force_or_A_1_nan)) if not indices_force_or_A_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, A_1_without_nan_force)
    fitted_response_A_1 = model.predict(force_80_1)
    a_A_1 = reg.coef_
    b_A_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A_1 = reg.score(force_80_1, A_1_without_nan_force)
    
    fig_A_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force80_1 = fig_A_vs_force80_1.gca()
    ax_A_vs_force80_1.errorbar(list(mean_force80_FF1_dict.values()), list(mean_A_FF1_dict.values()), yerr=list(std_A_FF1_dict.values()), xerr=list(std_force80_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_A_vs_force80_1.errorbar(list(mean_force80_RDG1_dict.values()),list(mean_A_RDG1_dict.values()), yerr=list(std_A_RDG1_dict.values()), xerr=list(std_force80_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_A_vs_force80_1.plot(force_80_1, fitted_response_A_1, ':k', alpha=0.8, label=' A = ' + str(np.round(a_A_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_A_1[0], 4)) + '\n R2 = ' + str(np.round(score_A_1, 2)) )
    for i in range(len(mean_force80_FF1_dict)):
        date = dates_to_use[i]
        ax_A_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_FF1_dict[date] +0.04, mean_A_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_RDG1_dict[date]+0.04, mean_A_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_A_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_A_vs_force80_1.set_title(r'A vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_A_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force80_1.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force80_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_force80_1")
    plt.close(fig_A_vs_force80_1)

    force_80_2 = np.concatenate((list(mean_force80_FF2_dict.values()), list(mean_force80_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_80_2) 
    A_2 = np.concatenate(( list(mean_A_FF2_dict.values()) , list(mean_A_RDG2_dict.values()) ))
    index_A_2_nan = np.isnan(A_2)
    indices_force_or_A_2_nan = [index_force_2_nan[i] or index_A_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_A_2_nan[i]])
    A_2_without_nan_force = np.array([A_2[i] for i in range(len(indices_force_or_A_2_nan)) if not indices_force_or_A_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, A_2_without_nan_force)
    fitted_response_A_2 = model.predict(force_80_2)
    a_A_2 = reg.coef_
    b_A_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A_2 = reg.score(force_80_2, A_2_without_nan_force)
    
    fig_A_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force80_2 = fig_A_vs_force80_2.gca()
    ax_A_vs_force80_2.errorbar(list(mean_force80_FF2_dict.values()), list(mean_A_FF2_dict.values()), yerr=list(std_A_FF2_dict.values()), xerr=list(std_force80_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_A_vs_force80_2.errorbar(list(mean_force80_RDG2_dict.values()),list(mean_A_RDG2_dict.values()), yerr=list(std_A_RDG2_dict.values()), xerr=list(std_force80_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_A_vs_force80_2.plot(force_80_2, fitted_response_A_2, ':k', alpha=0.8, label=' A = ' + str(np.round(a_A_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_A_2[0], 4)) + '\n R2 = ' + str(np.round(score_A_2, 2)) )
    for i in range(len(mean_force80_FF2_dict)):
        date = dates_to_use[i]
        ax_A_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_FF2_dict[date] +0.04, mean_A_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_RDG2_dict[date]+0.04, mean_A_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_A_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_A_vs_force80_2.set_title(r'A vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_A_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force80_2.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force80_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_force80_2")
    plt.close(fig_A_vs_force80_2)

    force_80 = np.concatenate((list(mean_force80_FF_dict.values()), list(mean_force80_RDG_dict.values())))
    index_force_nan = np.isnan(force_80) 
    A = np.concatenate(( list(mean_A_FF_dict.values()) , list(mean_A_RDG_dict.values()) ))
    index_A_nan = np.isnan(A)
    indices_force_or_A_nan = [index_force_nan[i] or index_A_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_A_nan[i]])
    A_without_nan_force = np.array([A[i] for i in range(len(indices_force_or_A_nan)) if not indices_force_or_A_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, A_without_nan_force)
    fitted_response_A = model.predict(force_80)
    a_A = reg.coef_
    b_A = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A = reg.score(force_80, A_without_nan_force)
    
    fig_A_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force80 = fig_A_vs_force80.gca()
    ax_A_vs_force80.errorbar(list(mean_force80_FF_dict.values()), list(mean_A_FF_dict.values()), yerr=list(std_A_FF_dict.values()), xerr=list(std_force80_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_A_vs_force80.errorbar(list(mean_force80_RDG_dict.values()),list(mean_A_RDG_dict.values()), yerr=list(std_A_RDG_dict.values()), xerr=list(std_force80_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_A_vs_force80.plot(force_80, fitted_response_A, ':k', alpha=0.8, label=' A = ' + str(np.round(a_A[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_A[0], 4)) + '\n R2 = ' + str(np.round(score_A, 2)) )
    for i in range(len(mean_force80_FF_dict)):
        date = dates_to_use[i]
        ax_A_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_FF_dict[date]+ 0.04, mean_A_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_RDG_dict[date]+0.04, mean_A_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_A_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_A_vs_force80.set_title(r'A vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_A_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force80.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force80, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_force80_1+2")
    plt.close(fig_A_vs_force80)

    #A vs force 20    
    
    force_20_1 = np.concatenate((list(mean_force20_FF1_dict.values()), list(mean_force20_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_20_1) 
    A_1 = np.concatenate(( list(mean_A_FF1_dict.values()) , list(mean_A_RDG1_dict.values()) ))
    index_A_1_nan = np.isnan(A_1)
    indices_force_or_A_1_nan = [index_force_1_nan[i] or index_A_1_nan[i] for i in range(len(index_force_1_nan))]
    force_20_1_without_nan = np.array([force_20_1[i] for i in range(len(force_20_1)) if not indices_force_or_A_1_nan[i]])
    A_1_without_nan_force = np.array([A_1[i] for i in range(len(indices_force_or_A_1_nan)) if not indices_force_or_A_1_nan[i]])
    force_20_1 = force_20_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_1, A_1_without_nan_force)
    fitted_response_A_1 = model.predict(force_20_1)
    a_A_1 = reg.coef_
    b_A_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A_1 = reg.score(force_20_1, A_1_without_nan_force)
    
    fig_A_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force20_1 = fig_A_vs_force20_1.gca()
    ax_A_vs_force20_1.errorbar(list(mean_force20_FF1_dict.values()), list(mean_A_FF1_dict.values()), yerr=list(std_A_FF1_dict.values()), xerr=list(std_force20_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_A_vs_force20_1.errorbar(list(mean_force20_RDG1_dict.values()),list(mean_A_RDG1_dict.values()), yerr=list(std_A_RDG1_dict.values()), xerr=list(std_force20_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_A_vs_force20_1.plot(force_20_1, fitted_response_A_1, ':k', alpha=0.8, label=' A = ' + str(np.round(a_A_1[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_A_1[0], 4)) + '\n R2 = ' + str(np.round(score_A_1, 2)) )
    for i in range(len(mean_force20_FF1_dict)):
        date = dates_to_use[i]
        ax_A_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_FF1_dict[date] +0.04, mean_A_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_RDG1_dict[date]+0.04, mean_A_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_A_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_A_vs_force20_1.set_title(r'A vs Force 20% 1', font=fonts.serif_rz_legend())
    ax_A_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force20_1.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force20_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_force20_1")
    plt.close(fig_A_vs_force20_1)

    force_20_2 = np.concatenate((list(mean_force20_FF2_dict.values()), list(mean_force20_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_20_2) 
    A_2 = np.concatenate(( list(mean_A_FF2_dict.values()) , list(mean_A_RDG2_dict.values()) ))
    index_A_2_nan = np.isnan(A_2)
    indices_force_or_A_2_nan = [index_force_2_nan[i] or index_A_2_nan[i] for i in range(len(index_force_2_nan))]
    force_20_2_without_nan = np.array([force_20_2[i] for i in range(len(force_20_2)) if not indices_force_or_A_2_nan[i]])
    A_2_without_nan_force = np.array([A_2[i] for i in range(len(indices_force_or_A_2_nan)) if not indices_force_or_A_2_nan[i]])
    force_20_2 = force_20_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_2, A_2_without_nan_force)
    fitted_response_A_2 = model.predict(force_20_2)
    a_A_2 = reg.coef_
    b_A_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A_2 = reg.score(force_20_2, A_2_without_nan_force)
    
    fig_A_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force20_2 = fig_A_vs_force20_2.gca()
    ax_A_vs_force20_2.errorbar(list(mean_force20_FF2_dict.values()), list(mean_A_FF2_dict.values()), yerr=list(std_A_FF2_dict.values()), xerr=list(std_force20_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_A_vs_force20_2.errorbar(list(mean_force20_RDG2_dict.values()),list(mean_A_RDG2_dict.values()), yerr=list(std_A_RDG2_dict.values()), xerr=list(std_force20_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_A_vs_force20_2.plot(force_20_2, fitted_response_A_2, ':k', alpha=0.8, label=' A = ' + str(np.round(a_A_2[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_A_2[0], 4)) + '\n R2 = ' + str(np.round(score_A_2, 2)) )
    for i in range(len(mean_force20_FF2_dict)):
        date = dates_to_use[i]
        ax_A_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_FF2_dict[date] +0.04, mean_A_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_RDG2_dict[date]+0.04, mean_A_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_A_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_A_vs_force20_2.set_title(r'A vs Force 20% 2', font=fonts.serif_rz_legend())
    ax_A_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force20_2.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force20_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_force20_2")
    plt.close(fig_A_vs_force20_2)

    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    A = np.concatenate(( list(mean_A_FF_dict.values()) , list(mean_A_RDG_dict.values()) ))
    index_A_nan = np.isnan(A)
    indices_force_or_A_nan = [index_force_nan[i] or index_A_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_A_nan[i]])
    A_without_nan_force = np.array([A[i] for i in range(len(indices_force_or_A_nan)) if not indices_force_or_A_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, A_without_nan_force)
    fitted_response_A = model.predict(force_20)
    a_A = reg.coef_
    b_A = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A = reg.score(force_20, A_without_nan_force)
    
    fig_A_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force20 = fig_A_vs_force20.gca()
    ax_A_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_A_FF_dict.values()), yerr=list(std_A_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_A_vs_force20.errorbar(list(mean_force20_RDG_dict.values()),list(mean_A_RDG_dict.values()), yerr=list(std_A_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_A_vs_force20.plot(force_20, fitted_response_A, ':k', alpha=0.8, label=' A = ' + str(np.round(a_A[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_A[0], 4)) + '\n R2 = ' + str(np.round(score_A, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_A_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date]+ 0.04, mean_A_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_A_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_A_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_A_vs_force20.set_title(r'A vs Force 20% 1+2', font=fonts.serif_rz_legend())
    ax_A_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force20.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force20, "0_locations_deltad_threshold" + str(deltad_threshold) + "_A_vs_force20_1+2")
    plt.close(fig_A_vs_force20)


    #delta_d vs force 80    
    
    force_80_1 = np.concatenate((list(mean_force80_FF1_dict.values()), list(mean_force80_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_80_1) 
    delta_d_1 = np.concatenate(( list(mean_delta_d_FF1_dict.values()) , list(mean_delta_d_RDG1_dict.values()) ))
    index_delta_d_1_nan = np.isnan(delta_d_1)
    indices_force_or_delta_d_1_nan = [index_force_1_nan[i] or index_delta_d_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_delta_d_1_nan[i]])
    delta_d_1_without_nan_force = np.array([delta_d_1[i] for i in range(len(indices_force_or_delta_d_1_nan)) if not indices_force_or_delta_d_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, delta_d_1_without_nan_force)
    fitted_response_delta_d_1 = model.predict(force_80_1)
    a_delta_d_1 = reg.coef_
    b_delta_d_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_1 = reg.score(force_80_1, delta_d_1_without_nan_force)
    
    fig_delta_d_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force80_1 = fig_delta_d_vs_force80_1.gca()
    ax_delta_d_vs_force80_1.errorbar(list(mean_force80_FF1_dict.values()), list(mean_delta_d_FF1_dict.values()), yerr=list(std_delta_d_FF1_dict.values()), xerr=list(std_force80_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_vs_force80_1.errorbar(list(mean_force80_RDG1_dict.values()),list(mean_delta_d_RDG1_dict.values()), yerr=list(std_delta_d_RDG1_dict.values()), xerr=list(std_force80_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_delta_d_vs_force80_1.plot(force_80_1, fitted_response_delta_d_1, ':k', alpha=0.8, label=r'$\Delta d$ = ' + str(np.round(a_delta_d_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_1[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_1, 2)) )
    for i in range(len(mean_force80_FF1_dict)):
        date = dates_to_use[i]
        ax_delta_d_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_FF1_dict[date] +0.04, mean_delta_d_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_RDG1_dict[date]+0.04, mean_delta_d_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_vs_force80_1.set_title(r'$\Delta d$ vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_1.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force80_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_force80_1")
    plt.close(fig_delta_d_vs_force80_1)

    force_80_2 = np.concatenate((list(mean_force80_FF2_dict.values()), list(mean_force80_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_80_2) 
    delta_d_2 = np.concatenate(( list(mean_delta_d_FF2_dict.values()) , list(mean_delta_d_RDG2_dict.values()) ))
    index_delta_d_2_nan = np.isnan(delta_d_2)
    indices_force_or_delta_d_2_nan = [index_force_2_nan[i] or index_delta_d_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_delta_d_2_nan[i]])
    delta_d_2_without_nan_force = np.array([delta_d_2[i] for i in range(len(indices_force_or_delta_d_2_nan)) if not indices_force_or_delta_d_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, delta_d_2_without_nan_force)
    fitted_response_delta_d_2 = model.predict(force_80_2)
    a_delta_d_2 = reg.coef_
    b_delta_d_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_2 = reg.score(force_80_2, delta_d_2_without_nan_force)
    
    fig_delta_d_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force80_2 = fig_delta_d_vs_force80_2.gca()
    ax_delta_d_vs_force80_2.errorbar(list(mean_force80_FF2_dict.values()), list(mean_delta_d_FF2_dict.values()), yerr=list(std_delta_d_FF2_dict.values()), xerr=list(std_force80_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_vs_force80_2.errorbar(list(mean_force80_RDG2_dict.values()),list(mean_delta_d_RDG2_dict.values()), yerr=list(std_delta_d_RDG2_dict.values()), xerr=list(std_force80_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_vs_force80_2.plot(force_80_2, fitted_response_delta_d_2, ':k', alpha=0.8, label=r'$\Delta d$ = ' + str(np.round(a_delta_d_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_2[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_2, 2)) )
    for i in range(len(mean_force80_FF2_dict)):
        date = dates_to_use[i]
        ax_delta_d_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_FF2_dict[date] +0.04, mean_delta_d_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_RDG2_dict[date]+0.04, mean_delta_d_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_delta_d_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_vs_force80_2.set_title(r'$\Delta d$ vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_2.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force80_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_force80_2")
    plt.close(fig_delta_d_vs_force80_2)

    force_80 = np.concatenate((list(mean_force80_FF_dict.values()), list(mean_force80_RDG_dict.values())))
    index_force_nan = np.isnan(force_80) 
    delta_d = np.concatenate(( list(mean_delta_d_FF_dict.values()) , list(mean_delta_d_RDG_dict.values()) ))
    index_delta_d_nan = np.isnan(delta_d)
    indices_force_or_delta_d_nan = [index_force_nan[i] or index_delta_d_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_delta_d_nan[i]])
    delta_d_without_nan_force = np.array([delta_d[i] for i in range(len(indices_force_or_delta_d_nan)) if not indices_force_or_delta_d_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, delta_d_without_nan_force)
    fitted_response_delta_d = model.predict(force_80)
    a_delta_d = reg.coef_
    b_delta_d = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d = reg.score(force_80, delta_d_without_nan_force)
    
    fig_delta_d_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force80 = fig_delta_d_vs_force80.gca()
    ax_delta_d_vs_force80.errorbar(list(mean_force80_FF_dict.values()), list(mean_delta_d_FF_dict.values()), yerr=list(std_delta_d_FF_dict.values()), xerr=list(std_force80_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_delta_d_vs_force80.errorbar(list(mean_force80_RDG_dict.values()),list(mean_delta_d_RDG_dict.values()), yerr=list(std_delta_d_RDG_dict.values()), xerr=list(std_force80_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_delta_d_vs_force80.plot(force_80, fitted_response_delta_d, ':k', alpha=0.8, label=r'$\Delta d$ = ' + str(np.round(a_delta_d[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d, 2)) )
    for i in range(len(mean_force80_FF_dict)):
        date = dates_to_use[i]
        ax_delta_d_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_FF_dict[date]+ 0.04, mean_delta_d_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_RDG_dict[date]+0.04, mean_delta_d_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_vs_force80.set_title(r'$\Delta d$ vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force80, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_force80_1+2")
    plt.close(fig_delta_d_vs_force80)

    #delta_d vs force 20    
    
    force_20_1 = np.concatenate((list(mean_force20_FF1_dict.values()), list(mean_force20_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_20_1) 
    delta_d_1 = np.concatenate(( list(mean_delta_d_FF1_dict.values()) , list(mean_delta_d_RDG1_dict.values()) ))
    index_delta_d_1_nan = np.isnan(delta_d_1)
    indices_force_or_delta_d_1_nan = [index_force_1_nan[i] or index_delta_d_1_nan[i] for i in range(len(index_force_1_nan))]
    force_20_1_without_nan = np.array([force_20_1[i] for i in range(len(force_20_1)) if not indices_force_or_delta_d_1_nan[i]])
    delta_d_1_without_nan_force = np.array([delta_d_1[i] for i in range(len(indices_force_or_delta_d_1_nan)) if not indices_force_or_delta_d_1_nan[i]])
    force_20_1 = force_20_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_1, delta_d_1_without_nan_force)
    fitted_response_delta_d_1 = model.predict(force_20_1)
    a_delta_d_1 = reg.coef_
    b_delta_d_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_1 = reg.score(force_20_1, delta_d_1_without_nan_force)
    
    fig_delta_d_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force20_1 = fig_delta_d_vs_force20_1.gca()
    ax_delta_d_vs_force20_1.errorbar(list(mean_force20_FF1_dict.values()), list(mean_delta_d_FF1_dict.values()), yerr=list(std_delta_d_FF1_dict.values()), xerr=list(std_force20_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_vs_force20_1.errorbar(list(mean_force20_RDG1_dict.values()),list(mean_delta_d_RDG1_dict.values()), yerr=list(std_delta_d_RDG1_dict.values()), xerr=list(std_force20_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_delta_d_vs_force20_1.plot(force_20_1, fitted_response_delta_d_1, ':k', alpha=0.8, label=r'$\Delta d$ = ' + str(np.round(a_delta_d_1[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_delta_d_1[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_1, 2)) )
    for i in range(len(mean_force20_FF1_dict)):
        date = dates_to_use[i]
        ax_delta_d_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_FF1_dict[date] +0.04, mean_delta_d_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_RDG1_dict[date]+0.04, mean_delta_d_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_vs_force20_1.set_title(r'$\Delta d$ vs Force 20% 1', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force20_1.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force20_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_force20_1")
    plt.close(fig_delta_d_vs_force20_1)

    force_20_2 = np.concatenate((list(mean_force20_FF2_dict.values()), list(mean_force20_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_20_2) 
    delta_d_2 = np.concatenate(( list(mean_delta_d_FF2_dict.values()) , list(mean_delta_d_RDG2_dict.values()) ))
    index_delta_d_2_nan = np.isnan(delta_d_2)
    indices_force_or_delta_d_2_nan = [index_force_2_nan[i] or index_delta_d_2_nan[i] for i in range(len(index_force_2_nan))]
    force_20_2_without_nan = np.array([force_20_2[i] for i in range(len(force_20_2)) if not indices_force_or_delta_d_2_nan[i]])
    delta_d_2_without_nan_force = np.array([delta_d_2[i] for i in range(len(indices_force_or_delta_d_2_nan)) if not indices_force_or_delta_d_2_nan[i]])
    force_20_2 = force_20_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_2, delta_d_2_without_nan_force)
    fitted_response_delta_d_2 = model.predict(force_20_2)
    a_delta_d_2 = reg.coef_
    b_delta_d_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_2 = reg.score(force_20_2, delta_d_2_without_nan_force)
    
    fig_delta_d_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force20_2 = fig_delta_d_vs_force20_2.gca()
    ax_delta_d_vs_force20_2.errorbar(list(mean_force20_FF2_dict.values()), list(mean_delta_d_FF2_dict.values()), yerr=list(std_delta_d_FF2_dict.values()), xerr=list(std_force20_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_vs_force20_2.errorbar(list(mean_force20_RDG2_dict.values()),list(mean_delta_d_RDG2_dict.values()), yerr=list(std_delta_d_RDG2_dict.values()), xerr=list(std_force20_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_vs_force20_2.plot(force_20_2, fitted_response_delta_d_2, ':k', alpha=0.8, label=r'$\Delta d$ = ' + str(np.round(a_delta_d_2[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_delta_d_2[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_2, 2)) )
    for i in range(len(mean_force20_FF2_dict)):
        date = dates_to_use[i]
        ax_delta_d_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_FF2_dict[date] +0.04, mean_delta_d_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_RDG2_dict[date]+0.04, mean_delta_d_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_delta_d_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_vs_force20_2.set_title(r'$\Delta d$ vs Force 20% 2', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force20_2.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force20_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_force20_2")
    plt.close(fig_delta_d_vs_force20_2)

    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    delta_d = np.concatenate(( list(mean_delta_d_FF_dict.values()) , list(mean_delta_d_RDG_dict.values()) ))
    index_delta_d_nan = np.isnan(delta_d)
    indices_force_or_delta_d_nan = [index_force_nan[i] or index_delta_d_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_delta_d_nan[i]])
    delta_d_without_nan_force = np.array([delta_d[i] for i in range(len(indices_force_or_delta_d_nan)) if not indices_force_or_delta_d_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, delta_d_without_nan_force)
    fitted_response_delta_d = model.predict(force_20)
    a_delta_d = reg.coef_
    b_delta_d = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d = reg.score(force_20, delta_d_without_nan_force)
    
    fig_delta_d_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force20 = fig_delta_d_vs_force20.gca()
    ax_delta_d_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_delta_d_FF_dict.values()), yerr=list(std_delta_d_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_delta_d_vs_force20.errorbar(list(mean_force20_RDG_dict.values()),list(mean_delta_d_RDG_dict.values()), yerr=list(std_delta_d_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_delta_d_vs_force20.plot(force_20, fitted_response_delta_d, ':k', alpha=0.8, label=r'$\Delta d$ = ' + str(np.round(a_delta_d[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_delta_d[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_delta_d_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date]+ 0.04, mean_delta_d_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_delta_d_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_vs_force20.set_title(r'$\Delta d$ vs Force 20% 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force20.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force20, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_vs_force20_1+2")
    plt.close(fig_delta_d_vs_force20)

    #delta_d_star vs force 80    
    
    force_80_1 = np.concatenate((list(mean_force80_FF1_dict.values()), list(mean_force80_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_80_1) 
    delta_d_star_1 = np.concatenate(( list(mean_delta_d_star_FF1_dict.values()) , list(mean_delta_d_star_RDG1_dict.values()) ))
    index_delta_d_star_1_nan = np.isnan(delta_d_star_1)
    indices_force_or_delta_d_star_1_nan = [index_force_1_nan[i] or index_delta_d_star_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_delta_d_star_1_nan[i]])
    delta_d_star_1_without_nan_force = np.array([delta_d_star_1[i] for i in range(len(indices_force_or_delta_d_star_1_nan)) if not indices_force_or_delta_d_star_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, delta_d_star_1_without_nan_force)
    fitted_response_delta_d_star_1 = model.predict(force_80_1)
    a_delta_d_star_1 = reg.coef_
    b_delta_d_star_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star_1 = reg.score(force_80_1, delta_d_star_1_without_nan_force)
    
    fig_delta_d_star_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force80_1 = fig_delta_d_star_vs_force80_1.gca()
    ax_delta_d_star_vs_force80_1.errorbar(list(mean_force80_FF1_dict.values()), list(mean_delta_d_star_FF1_dict.values()), yerr=list(std_delta_d_star_FF1_dict.values()), xerr=list(std_force80_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_star_vs_force80_1.errorbar(list(mean_force80_RDG1_dict.values()),list(mean_delta_d_star_RDG1_dict.values()), yerr=list(std_delta_d_star_RDG1_dict.values()), xerr=list(std_force80_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_delta_d_star_vs_force80_1.plot(force_80_1, fitted_response_delta_d_star_1, ':k', alpha=0.8, label=r'$\Delta d^*$ = ' + str(np.round(a_delta_d_star_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_star_1[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star_1, 2)) )
    for i in range(len(mean_force80_FF1_dict)):
        date = dates_to_use[i]
        ax_delta_d_star_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_FF1_dict[date] +0.04, mean_delta_d_star_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_star_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_RDG1_dict[date]+0.04, mean_delta_d_star_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_star_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_star_vs_force80_1.set_title(r'$\Delta d^*$ vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_1.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force80_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_force80_1")
    plt.close(fig_delta_d_star_vs_force80_1)

    force_80_2 = np.concatenate((list(mean_force80_FF2_dict.values()), list(mean_force80_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_80_2) 
    delta_d_star_2 = np.concatenate(( list(mean_delta_d_star_FF2_dict.values()) , list(mean_delta_d_star_RDG2_dict.values()) ))
    index_delta_d_star_2_nan = np.isnan(delta_d_star_2)
    indices_force_or_delta_d_star_2_nan = [index_force_2_nan[i] or index_delta_d_star_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_delta_d_star_2_nan[i]])
    delta_d_star_2_without_nan_force = np.array([delta_d_star_2[i] for i in range(len(indices_force_or_delta_d_star_2_nan)) if not indices_force_or_delta_d_star_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, delta_d_star_2_without_nan_force)
    fitted_response_delta_d_star_2 = model.predict(force_80_2)
    a_delta_d_star_2 = reg.coef_
    b_delta_d_star_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star_2 = reg.score(force_80_2, delta_d_star_2_without_nan_force)
    
    fig_delta_d_star_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force80_2 = fig_delta_d_star_vs_force80_2.gca()
    ax_delta_d_star_vs_force80_2.errorbar(list(mean_force80_FF2_dict.values()), list(mean_delta_d_star_FF2_dict.values()), yerr=list(std_delta_d_star_FF2_dict.values()), xerr=list(std_force80_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_star_vs_force80_2.errorbar(list(mean_force80_RDG2_dict.values()),list(mean_delta_d_star_RDG2_dict.values()), yerr=list(std_delta_d_star_RDG2_dict.values()), xerr=list(std_force80_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_star_vs_force80_2.plot(force_80_2, fitted_response_delta_d_star_2, ':k', alpha=0.8, label=r'$\Delta d^*$ = ' + str(np.round(a_delta_d_star_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_star_2[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star_2, 2)) )
    for i in range(len(mean_force80_FF2_dict)):
        date = dates_to_use[i]
        ax_delta_d_star_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_FF2_dict[date] +0.04, mean_delta_d_star_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_star_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_RDG2_dict[date]+0.04, mean_delta_d_star_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_delta_d_star_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_star_vs_force80_2.set_title(r'$\Delta d^*$ vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_2.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force80_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_force80_2")
    plt.close(fig_delta_d_star_vs_force80_2)

    force_80 = np.concatenate((list(mean_force80_FF_dict.values()), list(mean_force80_RDG_dict.values())))
    index_force_nan = np.isnan(force_80) 
    delta_d_star = np.concatenate(( list(mean_delta_d_star_FF_dict.values()) , list(mean_delta_d_star_RDG_dict.values()) ))
    index_delta_d_star_nan = np.isnan(delta_d_star)
    indices_force_or_delta_d_star_nan = [index_force_nan[i] or index_delta_d_star_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_delta_d_star_nan[i]])
    delta_d_star_without_nan_force = np.array([delta_d_star[i] for i in range(len(indices_force_or_delta_d_star_nan)) if not indices_force_or_delta_d_star_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, delta_d_star_without_nan_force)
    fitted_response_delta_d_star = model.predict(force_80)
    a_delta_d_star = reg.coef_
    b_delta_d_star = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star = reg.score(force_80, delta_d_star_without_nan_force)
    
    fig_delta_d_star_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force80 = fig_delta_d_star_vs_force80.gca()
    ax_delta_d_star_vs_force80.errorbar(list(mean_force80_FF_dict.values()), list(mean_delta_d_star_FF_dict.values()), yerr=list(std_delta_d_star_FF_dict.values()), xerr=list(std_force80_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_delta_d_star_vs_force80.errorbar(list(mean_force80_RDG_dict.values()),list(mean_delta_d_star_RDG_dict.values()), yerr=list(std_delta_d_star_RDG_dict.values()), xerr=list(std_force80_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_delta_d_star_vs_force80.plot(force_80, fitted_response_delta_d_star, ':k', alpha=0.8, label=r'$\Delta d^*$ = ' + str(np.round(a_delta_d_star[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_star[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star, 2)) )
    for i in range(len(mean_force80_FF_dict)):
        date = dates_to_use[i]
        ax_delta_d_star_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_FF_dict[date]+ 0.04, mean_delta_d_star_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_star_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_RDG_dict[date]+0.04, mean_delta_d_star_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_star_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_star_vs_force80.set_title(r'$\Delta d^*$ vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force80, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_force80_1+2")
    plt.close(fig_delta_d_star_vs_force80)

    #delta_d_star vs force 20    
    
    force_20_1 = np.concatenate((list(mean_force20_FF1_dict.values()), list(mean_force20_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_20_1) 
    delta_d_star_1 = np.concatenate(( list(mean_delta_d_star_FF1_dict.values()) , list(mean_delta_d_star_RDG1_dict.values()) ))
    index_delta_d_star_1_nan = np.isnan(delta_d_star_1)
    indices_force_or_delta_d_star_1_nan = [index_force_1_nan[i] or index_delta_d_star_1_nan[i] for i in range(len(index_force_1_nan))]
    force_20_1_without_nan = np.array([force_20_1[i] for i in range(len(force_20_1)) if not indices_force_or_delta_d_star_1_nan[i]])
    delta_d_star_1_without_nan_force = np.array([delta_d_star_1[i] for i in range(len(indices_force_or_delta_d_star_1_nan)) if not indices_force_or_delta_d_star_1_nan[i]])
    force_20_1 = force_20_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_1, delta_d_star_1_without_nan_force)
    fitted_response_delta_d_star_1 = model.predict(force_20_1)
    a_delta_d_star_1 = reg.coef_
    b_delta_d_star_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star_1 = reg.score(force_20_1, delta_d_star_1_without_nan_force)
    
    fig_delta_d_star_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force20_1 = fig_delta_d_star_vs_force20_1.gca()
    ax_delta_d_star_vs_force20_1.errorbar(list(mean_force20_FF1_dict.values()), list(mean_delta_d_star_FF1_dict.values()), yerr=list(std_delta_d_star_FF1_dict.values()), xerr=list(std_force20_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_star_vs_force20_1.errorbar(list(mean_force20_RDG1_dict.values()),list(mean_delta_d_star_RDG1_dict.values()), yerr=list(std_delta_d_star_RDG1_dict.values()), xerr=list(std_force20_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_delta_d_star_vs_force20_1.plot(force_20_1, fitted_response_delta_d_star_1, ':k', alpha=0.8, label=r'$\Delta d^*$ = ' + str(np.round(a_delta_d_star_1[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_delta_d_star_1[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star_1, 2)) )
    for i in range(len(mean_force20_FF1_dict)):
        date = dates_to_use[i]
        ax_delta_d_star_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_FF1_dict[date] +0.04, mean_delta_d_star_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_star_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_RDG1_dict[date]+0.04, mean_delta_d_star_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_star_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_star_vs_force20_1.set_title(r'$\Delta d^*$ vs Force 20% 1', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force20_1.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force20_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_force20_1")
    plt.close(fig_delta_d_star_vs_force20_1)

    force_20_2 = np.concatenate((list(mean_force20_FF2_dict.values()), list(mean_force20_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_20_2) 
    delta_d_star_2 = np.concatenate(( list(mean_delta_d_star_FF2_dict.values()) , list(mean_delta_d_star_RDG2_dict.values()) ))
    index_delta_d_star_2_nan = np.isnan(delta_d_star_2)
    indices_force_or_delta_d_star_2_nan = [index_force_2_nan[i] or index_delta_d_star_2_nan[i] for i in range(len(index_force_2_nan))]
    force_20_2_without_nan = np.array([force_20_2[i] for i in range(len(force_20_2)) if not indices_force_or_delta_d_star_2_nan[i]])
    delta_d_star_2_without_nan_force = np.array([delta_d_star_2[i] for i in range(len(indices_force_or_delta_d_star_2_nan)) if not indices_force_or_delta_d_star_2_nan[i]])
    force_20_2 = force_20_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_2, delta_d_star_2_without_nan_force)
    fitted_response_delta_d_star_2 = model.predict(force_20_2)
    a_delta_d_star_2 = reg.coef_
    b_delta_d_star_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star_2 = reg.score(force_20_2, delta_d_star_2_without_nan_force)
    
    fig_delta_d_star_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force20_2 = fig_delta_d_star_vs_force20_2.gca()
    ax_delta_d_star_vs_force20_2.errorbar(list(mean_force20_FF2_dict.values()), list(mean_delta_d_star_FF2_dict.values()), yerr=list(std_delta_d_star_FF2_dict.values()), xerr=list(std_force20_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_star_vs_force20_2.errorbar(list(mean_force20_RDG2_dict.values()),list(mean_delta_d_star_RDG2_dict.values()), yerr=list(std_delta_d_star_RDG2_dict.values()), xerr=list(std_force20_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_star_vs_force20_2.plot(force_20_2, fitted_response_delta_d_star_2, ':k', alpha=0.8, label=r'$\Delta d^*$ = ' + str(np.round(a_delta_d_star_2[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_delta_d_star_2[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star_2, 2)) )
    for i in range(len(mean_force20_FF2_dict)):
        date = dates_to_use[i]
        ax_delta_d_star_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_FF2_dict[date] +0.04, mean_delta_d_star_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_star_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_RDG2_dict[date]+0.04, mean_delta_d_star_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_delta_d_star_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_star_vs_force20_2.set_title(r'$\Delta d^*$ vs Force 20% 2', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force20_2.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force20_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_force20_2")
    plt.close(fig_delta_d_star_vs_force20_2)

    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    delta_d_star = np.concatenate(( list(mean_delta_d_star_FF_dict.values()) , list(mean_delta_d_star_RDG_dict.values()) ))
    index_delta_d_star_nan = np.isnan(delta_d_star)
    indices_force_or_delta_d_star_nan = [index_force_nan[i] or index_delta_d_star_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_delta_d_star_nan[i]])
    delta_d_star_without_nan_force = np.array([delta_d_star[i] for i in range(len(indices_force_or_delta_d_star_nan)) if not indices_force_or_delta_d_star_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, delta_d_star_without_nan_force)
    fitted_response_delta_d_star = model.predict(force_20)
    a_delta_d_star = reg.coef_
    b_delta_d_star = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star = reg.score(force_20, delta_d_star_without_nan_force)
    
    fig_delta_d_star_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force20 = fig_delta_d_star_vs_force20.gca()
    ax_delta_d_star_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_delta_d_star_FF_dict.values()), yerr=list(std_delta_d_star_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_delta_d_star_vs_force20.errorbar(list(mean_force20_RDG_dict.values()),list(mean_delta_d_star_RDG_dict.values()), yerr=list(std_delta_d_star_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_delta_d_star_vs_force20.plot(force_20, fitted_response_delta_d_star, ':k', alpha=0.8, label=r'$\Delta d^*$ = ' + str(np.round(a_delta_d_star[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_delta_d_star[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_delta_d_star_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date]+ 0.04, mean_delta_d_star_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_delta_d_star_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_delta_d_star_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_delta_d_star_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_delta_d_star_vs_force20.set_title(r'$\Delta d^*$ vs Force 20% 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force20.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force20, "0_locations_deltad_threshold" + str(deltad_threshold) + "_delta_d_star_vs_force20_1+2")
    plt.close(fig_delta_d_star_vs_force20)
    
    #d_min vs force 80    
    
    force_80_1 = np.concatenate((list(mean_force80_FF1_dict.values()), list(mean_force80_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_80_1) 
    d_min_1 = np.concatenate(( list(mean_d_min_FF1_dict.values()) , list(mean_d_min_RDG1_dict.values()) ))
    index_d_min_1_nan = np.isnan(d_min_1)
    indices_force_or_d_min_1_nan = [index_force_1_nan[i] or index_d_min_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_d_min_1_nan[i]])
    d_min_1_without_nan_force = np.array([d_min_1[i] for i in range(len(indices_force_or_d_min_1_nan)) if not indices_force_or_d_min_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, d_min_1_without_nan_force)
    fitted_response_d_min_1 = model.predict(force_80_1)
    a_d_min_1 = reg.coef_
    b_d_min_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min_1 = reg.score(force_80_1, d_min_1_without_nan_force)
    
    fig_d_min_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force80_1 = fig_d_min_vs_force80_1.gca()
    ax_d_min_vs_force80_1.errorbar(list(mean_force80_FF1_dict.values()), list(mean_d_min_FF1_dict.values()), yerr=list(std_d_min_FF1_dict.values()), xerr=list(std_force80_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_d_min_vs_force80_1.errorbar(list(mean_force80_RDG1_dict.values()),list(mean_d_min_RDG1_dict.values()), yerr=list(std_d_min_RDG1_dict.values()), xerr=list(std_force80_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_d_min_vs_force80_1.plot(force_80_1, fitted_response_d_min_1, ':k', alpha=0.8, label=r'$d_{min}$ = ' + str(np.round(a_d_min_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_d_min_1[0], 4)) + '\n R2 = ' + str(np.round(score_d_min_1, 2)) )
    for i in range(len(mean_force80_FF1_dict)):
        date = dates_to_use[i]
        ax_d_min_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_FF1_dict[date] +0.04, mean_d_min_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_d_min_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_RDG1_dict[date]+0.04, mean_d_min_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_d_min_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_d_min_vs_force80_1.set_title(r'$d_{min}$ vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_1.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force80_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_force80_1")
    plt.close(fig_d_min_vs_force80_1)

    force_80_2 = np.concatenate((list(mean_force80_FF2_dict.values()), list(mean_force80_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_80_2) 
    d_min_2 = np.concatenate(( list(mean_d_min_FF2_dict.values()) , list(mean_d_min_RDG2_dict.values()) ))
    index_d_min_2_nan = np.isnan(d_min_2)
    indices_force_or_d_min_2_nan = [index_force_2_nan[i] or index_d_min_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_d_min_2_nan[i]])
    d_min_2_without_nan_force = np.array([d_min_2[i] for i in range(len(indices_force_or_d_min_2_nan)) if not indices_force_or_d_min_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, d_min_2_without_nan_force)
    fitted_response_d_min_2 = model.predict(force_80_2)
    a_d_min_2 = reg.coef_
    b_d_min_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min_2 = reg.score(force_80_2, d_min_2_without_nan_force)
    
    fig_d_min_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force80_2 = fig_d_min_vs_force80_2.gca()
    ax_d_min_vs_force80_2.errorbar(list(mean_force80_FF2_dict.values()), list(mean_d_min_FF2_dict.values()), yerr=list(std_d_min_FF2_dict.values()), xerr=list(std_force80_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_d_min_vs_force80_2.errorbar(list(mean_force80_RDG2_dict.values()),list(mean_d_min_RDG2_dict.values()), yerr=list(std_d_min_RDG2_dict.values()), xerr=list(std_force80_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_d_min_vs_force80_2.plot(force_80_2, fitted_response_d_min_2, ':k', alpha=0.8, label=r'$d_{min}$ = ' + str(np.round(a_d_min_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_d_min_2[0], 4)) + '\n R2 = ' + str(np.round(score_d_min_2, 2)) )
    for i in range(len(mean_force80_FF2_dict)):
        date = dates_to_use[i]
        ax_d_min_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_FF2_dict[date] +0.04, mean_d_min_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_d_min_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_RDG2_dict[date]+0.04, mean_d_min_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_d_min_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_d_min_vs_force80_2.set_title(r'$d_{min}$ vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_2.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force80_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_force80_2")
    plt.close(fig_d_min_vs_force80_2)

    force_80 = np.concatenate((list(mean_force80_FF_dict.values()), list(mean_force80_RDG_dict.values())))
    index_force_nan = np.isnan(force_80) 
    d_min = np.concatenate(( list(mean_d_min_FF_dict.values()) , list(mean_d_min_RDG_dict.values()) ))
    index_d_min_nan = np.isnan(d_min)
    indices_force_or_d_min_nan = [index_force_nan[i] or index_d_min_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_d_min_nan[i]])
    d_min_without_nan_force = np.array([d_min[i] for i in range(len(indices_force_or_d_min_nan)) if not indices_force_or_d_min_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, d_min_without_nan_force)
    fitted_response_d_min = model.predict(force_80)
    a_d_min = reg.coef_
    b_d_min = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min = reg.score(force_80, d_min_without_nan_force)
    
    fig_d_min_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force80 = fig_d_min_vs_force80.gca()
    ax_d_min_vs_force80.errorbar(list(mean_force80_FF_dict.values()), list(mean_d_min_FF_dict.values()), yerr=list(std_d_min_FF_dict.values()), xerr=list(std_force80_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_d_min_vs_force80.errorbar(list(mean_force80_RDG_dict.values()),list(mean_d_min_RDG_dict.values()), yerr=list(std_d_min_RDG_dict.values()), xerr=list(std_force80_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_d_min_vs_force80.plot(force_80, fitted_response_d_min, ':k', alpha=0.8, label=r'$d_{min}$ = ' + str(np.round(a_d_min[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_d_min[0], 4)) + '\n R2 = ' + str(np.round(score_d_min, 2)) )
    for i in range(len(mean_force80_FF_dict)):
        date = dates_to_use[i]
        ax_d_min_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_FF_dict[date]+ 0.04, mean_d_min_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_d_min_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_RDG_dict[date]+0.04, mean_d_min_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_d_min_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_d_min_vs_force80.set_title(r'$d_{min}$ vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force80, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_force80_1+2")
    plt.close(fig_d_min_vs_force80)

    #d_min vs force 20    
    
    force_20_1 = np.concatenate((list(mean_force20_FF1_dict.values()), list(mean_force20_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_20_1) 
    d_min_1 = np.concatenate(( list(mean_d_min_FF1_dict.values()) , list(mean_d_min_RDG1_dict.values()) ))
    index_d_min_1_nan = np.isnan(d_min_1)
    indices_force_or_d_min_1_nan = [index_force_1_nan[i] or index_d_min_1_nan[i] for i in range(len(index_force_1_nan))]
    force_20_1_without_nan = np.array([force_20_1[i] for i in range(len(force_20_1)) if not indices_force_or_d_min_1_nan[i]])
    d_min_1_without_nan_force = np.array([d_min_1[i] for i in range(len(indices_force_or_d_min_1_nan)) if not indices_force_or_d_min_1_nan[i]])
    force_20_1 = force_20_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_1, d_min_1_without_nan_force)
    fitted_response_d_min_1 = model.predict(force_20_1)
    a_d_min_1 = reg.coef_
    b_d_min_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min_1 = reg.score(force_20_1, d_min_1_without_nan_force)
    
    fig_d_min_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force20_1 = fig_d_min_vs_force20_1.gca()
    ax_d_min_vs_force20_1.errorbar(list(mean_force20_FF1_dict.values()), list(mean_d_min_FF1_dict.values()), yerr=list(std_d_min_FF1_dict.values()), xerr=list(std_force20_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_d_min_vs_force20_1.errorbar(list(mean_force20_RDG1_dict.values()),list(mean_d_min_RDG1_dict.values()), yerr=list(std_d_min_RDG1_dict.values()), xerr=list(std_force20_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_d_min_vs_force20_1.plot(force_20_1, fitted_response_d_min_1, ':k', alpha=0.8, label=r'$d_{min}$ = ' + str(np.round(a_d_min_1[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_d_min_1[0], 4)) + '\n R2 = ' + str(np.round(score_d_min_1, 2)) )
    for i in range(len(mean_force20_FF1_dict)):
        date = dates_to_use[i]
        ax_d_min_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_FF1_dict[date] +0.04, mean_d_min_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_d_min_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_RDG1_dict[date]+0.04, mean_d_min_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_d_min_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_d_min_vs_force20_1.set_title(r'$d_{min}$ vs Force 20% 1', font=fonts.serif_rz_legend())
    ax_d_min_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force20_1.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force20_1, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_force20_1")
    plt.close(fig_d_min_vs_force20_1)

    force_20_2 = np.concatenate((list(mean_force20_FF2_dict.values()), list(mean_force20_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_20_2) 
    d_min_2 = np.concatenate(( list(mean_d_min_FF2_dict.values()) , list(mean_d_min_RDG2_dict.values()) ))
    index_d_min_2_nan = np.isnan(d_min_2)
    indices_force_or_d_min_2_nan = [index_force_2_nan[i] or index_d_min_2_nan[i] for i in range(len(index_force_2_nan))]
    force_20_2_without_nan = np.array([force_20_2[i] for i in range(len(force_20_2)) if not indices_force_or_d_min_2_nan[i]])
    d_min_2_without_nan_force = np.array([d_min_2[i] for i in range(len(indices_force_or_d_min_2_nan)) if not indices_force_or_d_min_2_nan[i]])
    force_20_2 = force_20_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_2, d_min_2_without_nan_force)
    fitted_response_d_min_2 = model.predict(force_20_2)
    a_d_min_2 = reg.coef_
    b_d_min_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min_2 = reg.score(force_20_2, d_min_2_without_nan_force)
    
    fig_d_min_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force20_2 = fig_d_min_vs_force20_2.gca()
    ax_d_min_vs_force20_2.errorbar(list(mean_force20_FF2_dict.values()), list(mean_d_min_FF2_dict.values()), yerr=list(std_d_min_FF2_dict.values()), xerr=list(std_force20_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_d_min_vs_force20_2.errorbar(list(mean_force20_RDG2_dict.values()),list(mean_d_min_RDG2_dict.values()), yerr=list(std_d_min_RDG2_dict.values()), xerr=list(std_force20_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_d_min_vs_force20_2.plot(force_20_2, fitted_response_d_min_2, ':k', alpha=0.8, label=r'$d_{min}$ = ' + str(np.round(a_d_min_2[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_d_min_2[0], 4)) + '\n R2 = ' + str(np.round(score_d_min_2, 2)) )
    for i in range(len(mean_force20_FF2_dict)):
        date = dates_to_use[i]
        ax_d_min_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_FF2_dict[date] +0.04, mean_d_min_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_d_min_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_RDG2_dict[date]+0.04, mean_d_min_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))    
    ax_d_min_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_d_min_vs_force20_2.set_title(r'$d_{min}$ vs Force 20% 2', font=fonts.serif_rz_legend())
    ax_d_min_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force20_2.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force20_2, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_force20_2")
    plt.close(fig_d_min_vs_force20_2)

    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    d_min = np.concatenate(( list(mean_d_min_FF_dict.values()) , list(mean_d_min_RDG_dict.values()) ))
    index_d_min_nan = np.isnan(d_min)
    indices_force_or_d_min_nan = [index_force_nan[i] or index_d_min_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_d_min_nan[i]])
    d_min_without_nan_force = np.array([d_min[i] for i in range(len(indices_force_or_d_min_nan)) if not indices_force_or_d_min_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, d_min_without_nan_force)
    fitted_response_d_min = model.predict(force_20)
    a_d_min = reg.coef_
    b_d_min = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min = reg.score(force_20, d_min_without_nan_force)
    
    fig_d_min_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force20 = fig_d_min_vs_force20.gca()
    ax_d_min_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_d_min_FF_dict.values()), yerr=list(std_d_min_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_d_min_vs_force20.errorbar(list(mean_force20_RDG_dict.values()),list(mean_d_min_RDG_dict.values()), yerr=list(std_d_min_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_d_min_vs_force20.plot(force_20, fitted_response_d_min, ':k', alpha=0.8, label=r'$d_{min}$ = ' + str(np.round(a_d_min[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_d_min[0], 4)) + '\n R2 = ' + str(np.round(score_d_min, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_d_min_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date]+ 0.04, mean_d_min_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_d_min_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_d_min_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_d_min_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='upper left', framealpha=0.7)
    ax_d_min_vs_force20.set_title(r'$d_{min}$ vs Force 20% 1+2', font=fonts.serif_rz_legend())
    ax_d_min_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force20.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force20, "0_locations_deltad_threshold" + str(deltad_threshold) + "_d_min_vs_force20_1+2")
    plt.close(fig_d_min_vs_force20)





if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    # nb_of_time_increments_to_plot = 10
    # recovery_pkl_file = '0_locations_recovery_meat.pkl'
    deltad_threshold = 0.2
    utils.transform_csv_input_A_into_pkl('0_locations_recoveries_A.csv')
    ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict = utils.extract_A_data_from_pkl()
    # ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed    = remove_failed_A(ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    # ids_where_not_failed_and_not_small_deltad, date_dict_not_failed_and_not_small_deltad, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad = remove_failed_A_and_small_deltad(deltad_threshold, ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    # compute_and_export_indicators_with_maturation_as_pkl(ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict, deltad_threshold)
    # export_indocators_as_txt(deltad_threshold)
    # plot_recovery_indicators_with_maturation(deltad_threshold) 

    plot_laser_indicators_vs_texturometer_forces(deltad_threshold)
    # irr_indicator_list = ['relaxation_slope',
    #             'delta_f',
    #             'delta_f_star',
    #             'i_disp_strain_rate',
    #             'i_time_strain_rate',
    #             'i_disp_1',
    #             'i_time_1'       
    #                 ]
    # for irr_indicator in irr_indicator_list:
    #     plot_indentation_relaxation_indicator_vs_texturometer_forces(irr_indicator)
    print('hello')