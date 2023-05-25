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
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter
import pickle
import csv
import pandas as pd
import statistics

def remove_failed_data(ids_list, date_dict, force20_dict, force80_dict, failed_dict):
    ids_where_not_failed = [id for id in ids_list if failed_dict[id] == 0]
    date_dict_not_failed = {id: date_dict[id] for id in ids_where_not_failed}
    force20_dict_not_failed = {id: force20_dict[id] for id in ids_where_not_failed}
    force80_dict_not_failed = {id: force80_dict[id] for id in ids_where_not_failed}
    return ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed

def extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict):
    ids_at_date = [id for id in ids_list if date_dict[id] == date]
    ids_at_date_and_meatpiece = [id for id in ids_at_date if id[0:len(str(date)) + 1 + len(meatpiece)] == str(date) + '_' + meatpiece] 
    force20_dict_at_date_and_meatpiece = {id: force20_dict[id] for id in ids_at_date_and_meatpiece}
    force80_dict_at_date_and_meatpiece = {id: force80_dict[id] for id in ids_at_date_and_meatpiece}
    return ids_at_date_and_meatpiece, force20_dict_at_date_and_meatpiece, force80_dict_at_date_and_meatpiece

def compute_mean_and_std_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict):
    _, force20_dict_at_date_and_meatpiece, force80_dict_at_date_and_meatpiece = extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict)
    mean_force20 = statistics.mean(list(force20_dict_at_date_and_meatpiece.values()))
    std_force20 = statistics.stdev(list(force20_dict_at_date_and_meatpiece.values()))
    mean_force80 = statistics.mean(list(force80_dict_at_date_and_meatpiece.values()))
    std_force80 = statistics.stdev(list(force80_dict_at_date_and_meatpiece.values()))
    return mean_force20, std_force20, mean_force80, std_force80
# def compute_mean_std(ids_list, date_dict, force20_dict, force80_dict, failed_dict): 
    


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()

    ids_list, date_dict, force20_dict, force80_dict, failed_dict = utils.extract_texturometer_data_from_pkl()
    ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed = remove_failed_data(ids_list, date_dict, force20_dict, force80_dict, failed_dict)
    mean_force20, std_force20, mean_force80, std_force80 = compute_mean_and_std_at_given_date_and_meatpiece(230327, 'FF', ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed)
    print('hello')
    
    
    
