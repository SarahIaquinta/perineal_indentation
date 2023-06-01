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
    ids_where_not_failed = [id for id in filenames_from_pkl if failed_A_acqusitions[id] ==0 and delta_d_dict[id]!='FAILED LASER ACQUISITION']
    date_dict_not_failed = {id: dates_dict[id] for id in ids_where_not_failed}
    delta_d_dict_not_failed = {id: delta_d_dict[id] for id in ids_where_not_failed}
    delta_d_star_dict_not_failed = {id: delta_d_star_dict[id] for id in ids_where_not_failed}
    d_min_dict_not_failed = {id: d_min_dict[id] for id in ids_where_not_failed}
    A_dict_not_failed = {id: A_dict[id] for id in ids_where_not_failed}
    return ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed

def remove_failed_A_and_small_deltad(deltad_threshold, filenames_from_pkl, dates_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
    ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed = remove_failed_A(filenames_from_pkl, dates_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
    ids_where_not_failed_and_not_small_deltad = [id for id in ids_where_not_failed if float(delta_d_dict[id]) > deltad_threshold]
    date_dict_not_failed_and_not_small_deltad = {id: float(date_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    delta_d_dict_not_failed_and_not_small_deltad = {id: float(delta_d_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    delta_d_star_dict_not_failed_and_not_small_deltad = {id: float(delta_d_star_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    d_min_dict_not_failed_and_not_small_deltad = {id: float(d_min_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    A_dict_not_failed_and_not_small_deltad = {id: float(A_dict_not_failed[id]) for id in ids_where_not_failed_and_not_small_deltad}
    return ids_where_not_failed_and_not_small_deltad, date_dict_not_failed_and_not_small_deltad, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad

def extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict):
    ids_at_date = [id for id in ids_list if date_dict[id] == date]
    ids_at_date_and_meatpiece = [id for id in ids_at_date if id[0:len(str(date)) + 1 + len(meatpiece)] == str(date) + '_' + meatpiece] 
    delta_d_dict_at_date_and_meatpiece = {id: delta_d_dict[id] for id in ids_at_date_and_meatpiece}
    delta_d_star_dict_at_date_and_meatpiece = {id: delta_d_star_dict[id] for id in ids_at_date_and_meatpiece}
    d_min_dict_at_date_and_meatpiece = {id: d_min_dict[id] for id in ids_at_date_and_meatpiece}
    A_dict_at_date_and_meatpiece = {id: A_dict[id] for id in ids_at_date_and_meatpiece}
    return ids_at_date_and_meatpiece, delta_d_dict_at_date_and_meatpiece, delta_d_star_dict_at_date_and_meatpiece, d_min_dict_at_date_and_meatpiece, A_dict_at_date_and_meatpiece

def compute_mean_and_std_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
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

def compute_and_export_indicators_with_maturation_as_pkl(ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
    dates = list(set(date_dict.values()))
    mean_delta_d_FF1, std_delta_d_FF1, mean_delta_d_star_FF1, std_delta_d_star_FF1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_delta_d_FF2, std_delta_d_FF2, mean_delta_d_star_FF2, std_delta_d_star_FF2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_delta_d_RDG1, std_delta_d_RDG1, mean_delta_d_star_RDG1, std_delta_d_star_RDG1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_delta_d_RDG2, std_delta_d_RDG2, mean_delta_d_star_RDG2, std_delta_d_star_RDG2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_delta_d_FF, std_delta_d_FF, mean_delta_d_star_FF, std_delta_d_star_FF = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_delta_d_RDG, std_delta_d_RDG, mean_delta_d_star_RDG, std_delta_d_star_RDG = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))

    mean_d_min_FF1, std_d_min_FF1, mean_A_FF1, std_A_FF1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_d_min_FF2, std_d_min_FF2, mean_A_FF2, std_A_FF2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_d_min_RDG1, std_d_min_RDG1, mean_A_RDG1, std_A_RDG1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_d_min_RDG2, std_d_min_RDG2, mean_A_RDG2, std_A_RDG2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_d_min_FF, std_d_min_FF, mean_A_FF, std_A_FF = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_d_min_RDG, std_d_min_RDG, mean_A_RDG, std_A_RDG = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    
    for i in range(len(dates)):
        date = dates[i]
        mean_delta_d_FF1_date, std_delta_d_FF1_date, mean_delta_d_star_FF1_date, std_delta_d_star_FF1_date, mean_d_min_FF1_date, std_d_min_FF1_date,  mean_A_FF1_date, std_A_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF1', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF2_date, std_delta_d_FF2_date, mean_delta_d_star_FF2_date, std_delta_d_star_FF2_date, mean_d_min_FF2_date, std_d_min_FF2_date,  mean_A_FF2_date, std_A_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF2', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG1_date, std_delta_d_RDG1_date, mean_delta_d_star_RDG1_date, std_delta_d_star_RDG1_date, mean_d_min_RDG1_date, std_d_min_RDG1_date,  mean_A_RDG1_date, std_A_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG1', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG2_date, std_delta_d_RDG2_date, mean_delta_d_star_RDG2_date, std_delta_d_star_RDG2_date, mean_d_min_RDG2_date, std_d_min_RDG2_date,  mean_A_RDG2_date, std_A_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG2', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF_date, std_delta_d_FF_date, mean_delta_d_star_FF_date, std_delta_d_star_FF_date, mean_d_min_FF_date, std_d_min_FF_date,  mean_A_FF_date, std_A_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG_date, std_delta_d_RDG_date, mean_delta_d_star_RDG_date, std_delta_d_star_RDG_date, mean_d_min_RDG_date, std_d_min_RDG_date,  mean_A_RDG_date, std_A_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF1[i], std_delta_d_FF1[i], mean_delta_d_star_FF1[i], std_delta_d_star_FF1[i] = mean_delta_d_FF1_date, std_delta_d_FF1_date, mean_delta_d_star_FF1_date, std_delta_d_star_FF1_date
        mean_delta_d_FF2[i], std_delta_d_FF2[i], mean_delta_d_star_FF2[i], std_delta_d_star_FF2[i] = mean_delta_d_FF2_date, std_delta_d_FF2_date, mean_delta_d_star_FF2_date, std_delta_d_star_FF2_date
        mean_delta_d_FF[i], std_delta_d_FF[i], mean_delta_d_star_FF[i], std_delta_d_star_FF[i] = mean_delta_d_FF_date, std_delta_d_FF_date, mean_delta_d_star_FF_date, std_delta_d_star_FF2_date
        mean_delta_d_RDG1[i], std_delta_d_RDG1[i], mean_delta_d_star_RDG1[i], std_delta_d_star_RDG1[i] = mean_delta_d_RDG1_date, std_delta_d_RDG1_date, mean_delta_d_star_RDG1_date, std_delta_d_star_RDG1_date
        mean_delta_d_RDG2[i], std_delta_d_RDG2[i], mean_delta_d_star_RDG2[i], std_delta_d_star_RDG2[i] = mean_delta_d_RDG2_date, std_delta_d_RDG2_date, mean_delta_d_star_RDG2_date, std_delta_d_star_RDG2_date
        mean_delta_d_RDG[i], std_delta_d_RDG[i], mean_delta_d_star_RDG[i], std_delta_d_star_RDG[i] = mean_delta_d_RDG_date, std_delta_d_RDG_date, mean_delta_d_star_RDG_date, std_delta_d_star_RDG_date
        mean_d_min_FF[i], std_d_min_FF[i], mean_A_FF[i], std_A_FF[i] = mean_d_min_FF_date, std_d_min_FF_date, mean_A_FF_date, std_A_FF_date
        mean_d_min_RDG[i], std_d_min_RDG[i], mean_A_RDG[i], std_A_RDG[i] = mean_d_min_RDG_date, std_d_min_RDG_date, mean_A_RDG_date, std_A_RDG_date
        mean_d_min_FF1[i], std_d_min_FF1[i], mean_A_FF1[i], std_A_FF1[i] = mean_d_min_FF1_date, std_d_min_FF1_date, mean_A_FF1_date, std_A_FF1_date
        mean_d_min_FF2[i], std_d_min_FF2[i], mean_A_FF2[i], std_A_FF2[i] = mean_d_min_FF2_date, std_d_min_FF2_date, mean_A_FF2_date, std_A_FF2_date
        mean_d_min_RDG1[i], std_d_min_RDG1[i], mean_A_RDG1[i], std_A_RDG1[i] = mean_d_min_RDG1_date, std_d_min_RDG1_date, mean_A_RDG1_date, std_A_RDG1_date
        mean_d_min_RDG2[i], std_d_min_RDG2[i], mean_A_RDG2[i], std_A_RDG2[i] = mean_d_min_RDG2_date, std_d_min_RDG2_date, mean_A_RDG2_date, std_A_RDG2_date
        mean_d_min_FF[i], std_d_min_FF[i], mean_A_FF[i], std_A_FF[i] = mean_d_min_FF_date, std_d_min_FF_date, mean_A_FF_date, std_A_FF_date
        mean_d_min_RDG[i], std_d_min_RDG[i], mean_A_RDG[i], std_A_RDG[i] = mean_d_min_RDG_date, std_d_min_RDG_date, mean_A_RDG_date, std_A_RDG_date
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indicators_mean_std.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [dates, mean_delta_d_FF1, std_delta_d_FF1, mean_delta_d_star_FF1, std_delta_d_star_FF1, mean_d_min_FF1, std_d_min_FF1,  mean_A_FF1, std_A_FF1,
             mean_delta_d_FF2, std_delta_d_FF2, mean_delta_d_star_FF2, std_delta_d_star_FF2, mean_d_min_FF2, std_d_min_FF2,  mean_A_FF2, std_A_FF2,
             mean_delta_d_RDG1, std_delta_d_RDG1, mean_delta_d_star_RDG1, std_delta_d_star_RDG1, mean_d_min_RDG1, std_d_min_RDG1,  mean_A_RDG1, std_A_RDG1,
             mean_delta_d_RDG2, std_delta_d_RDG2, mean_delta_d_star_RDG2, std_delta_d_star_RDG2, mean_d_min_RDG2, std_d_min_RDG2,  mean_A_RDG2, std_A_RDG2,
             mean_delta_d_FF, std_delta_d_FF, mean_delta_d_star_FF, std_delta_d_star_FF, mean_d_min_FF, std_d_min_FF,  mean_A_FF, std_A_FF,
             mean_delta_d_RDG, std_delta_d_RDG, mean_delta_d_star_RDG, std_delta_d_star_RDG, mean_d_min_RDG, std_d_min_RDG,  mean_A_RDG, std_A_RDG
             ],
            f,
        )
        
def export_indocators_as_txt():
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indicators_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [date, mean_delta_d_FF1, std_delta_d_FF1, mean_delta_d_star_FF1, std_delta_d_star_FF1, mean_d_min_FF1, std_d_min_FF1,  mean_A_FF1, std_A_FF1,
             mean_delta_d_FF2, std_delta_d_FF2, mean_delta_d_star_FF2, std_delta_d_star_FF2, mean_d_min_FF2, std_d_min_FF2,  mean_A_FF2, std_A_FF2,
             mean_delta_d_RDG1, std_delta_d_RDG1, mean_delta_d_star_RDG1, std_delta_d_star_RDG1, mean_d_min_RDG1, std_d_min_RDG1,  mean_A_RDG1, std_A_RDG1,
             mean_delta_d_RDG2, std_delta_d_RDG2, mean_delta_d_star_RDG2, std_delta_d_star_RDG2, mean_d_min_RDG2, std_d_min_RDG2,  mean_A_RDG2, std_A_RDG2,
             mean_delta_d_FF, std_delta_d_FF, mean_delta_d_star_FF, std_delta_d_star_FF, mean_d_min_FF, std_d_min_FF,  mean_A_FF, std_A_FF,
             mean_delta_d_RDG, std_delta_d_RDG, mean_delta_d_star_RDG, std_delta_d_star_RDG, mean_d_min_RDG, std_d_min_RDG,  mean_A_RDG, std_A_RDG
             ] = pickle.load(f)
    
    complete_txt_filename_FF1 = path_to_processed_data + "/indicators_mean_std_FF1.txt"
    f = open(complete_txt_filename_FF1, "w")
    f.write("INDICATORS FOR FF1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_FF1[i])
            + "\t"
            + str(std_delta_d_FF1[i])
            + "\t"
            + str(mean_delta_d_star_FF1[i])
            + "\t"
            + str(std_delta_d_star_FF1[i])
            + "\t"
            + str(mean_d_min_FF1[i])
            + "\t"
            + str(std_d_min_FF1[i])
            + "\t"
            + str(mean_A_FF1[i])
            + "\t"
            + str(std_A_FF1[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF2 = path_to_processed_data + "/indicators_mean_std_FF2.txt"
    f = open(complete_txt_filename_FF2, "w")
    f.write("INDICATORS FOR FF2 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_FF2[i])
            + "\t"
            + str(std_delta_d_FF2[i])
            + "\t"
            + str(mean_delta_d_star_FF2[i])
            + "\t"
            + str(std_delta_d_star_FF2[i])
            + "\t"
            + str(mean_d_min_FF2[i])
            + "\t"
            + str(std_d_min_FF2[i])
            + "\t"
            + str(mean_A_FF2[i])
            + "\t"
            + str(std_A_FF2[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF = path_to_processed_data + "/indicators_mean_std_FF.txt"
    f = open(complete_txt_filename_FF, "w")
    f.write("INDICATORS FOR FF \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_FF[i])
            + "\t"
            + str(std_delta_d_FF[i])
            + "\t"
            + str(mean_delta_d_star_FF[i])
            + "\t"
            + str(std_delta_d_star_FF[i])
            + "\t"
            + str(mean_d_min_FF[i])
            + "\t"
            + str(std_d_min_FF[i])
            + "\t"
            + str(mean_A_FF[i])
            + "\t"
            + str(std_A_FF[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG1 = path_to_processed_data + "/indicators_mean_std_RDG1.txt"
    f = open(complete_txt_filename_RDG1, "w")
    f.write("INDICATORS FOR RDG1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_RDG1[i])
            + "\t"
            + str(std_delta_d_RDG1[i])
            + "\t"
            + str(mean_delta_d_star_RDG1[i])
            + "\t"
            + str(std_delta_d_star_RDG1[i])
            + "\t"
            + str(mean_d_min_RDG1[i])
            + "\t"
            + str(std_d_min_RDG1[i])
            + "\t"
            + str(mean_A_RDG1[i])
            + "\t"
            + str(std_A_RDG1[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG2 = path_to_processed_data + "/indicators_mean_std_RDG2.txt"
    f = open(complete_txt_filename_RDG2, "w")
    f.write("INDICATORS FOR RDG2 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_RDG2[i])
            + "\t"
            + str(std_delta_d_RDG2[i])
            + "\t"
            + str(mean_delta_d_star_RDG2[i])
            + "\t"
            + str(std_delta_d_star_RDG2[i])
            + "\t"
            + str(mean_d_min_RDG2[i])
            + "\t"
            + str(std_d_min_RDG2[i])
            + "\t"
            + str(mean_A_RDG2[i])
            + "\t"
            + str(std_A_RDG2[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG = path_to_processed_data + "/indicators_mean_std_RDG.txt"
    f = open(complete_txt_filename_RDG, "w")
    f.write("INDICATORS FOR RDG \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_RDG[i])
            + "\t"
            + str(std_delta_d_RDG[i])
            + "\t"
            + str(mean_delta_d_star_RDG[i])
            + "\t"
            + str(std_delta_d_star_RDG[i])
            + "\t"
            + str(mean_d_min_RDG[i])
            + "\t"
            + str(std_d_min_RDG[i])
            + "\t"
            + str(mean_A_RDG[i])
            + "\t"
            + str(std_A_RDG[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_all = path_to_processed_data + "/indicators_mean_std.txt"
    f = open(complete_txt_filename_all, "w")
    f.write("INDICATORS \n")
    f.write("FF1 \t  FF1 \t  FF1 \t  FF1 \t FF1 \t  FF1 \t  FF1 \t  FF1 \t  FF1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_FF1[i])
            + "\t"
            + str(std_delta_d_FF1[i])
            + "\t"
            + str(mean_delta_d_star_FF1[i])
            + "\t"
            + str(std_delta_d_star_FF1[i])
            + "\t"
            + str(mean_d_min_FF1[i])
            + "\t"
            + str(std_d_min_FF1[i])
            + "\t"
            + str(mean_A_FF1[i])
            + "\t"
            + str(std_A_FF1[i])
            + "\n"
        )
        
    f.write("FF2 \t  FF2 \t FF2 \t  FF2 \t FF2 \t  FF2 \t  FF2 \t  FF2 \t  FF2 \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_FF2[i])
            + "\t"
            + str(std_delta_d_FF2[i])
            + "\t"
            + str(mean_delta_d_star_FF2[i])
            + "\t"
            + str(std_delta_d_star_FF2[i])
            + "\t"
            + str(mean_d_min_FF2[i])
            + "\t"
            + str(std_d_min_FF2[i])
            + "\t"
            + str(mean_A_FF2[i])
            + "\t"
            + str(std_A_FF2[i])
            + "\n"
        )
        
    f.write("FF \t  FF \t FF \t  FF \t FF \t  FF \t  FF \t  FF \t  FF \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_FF)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_FF[i])
            + "\t"
            + str(std_delta_d_FF[i])
            + "\t"
            + str(mean_delta_d_star_FF[i])
            + "\t"
            + str(std_delta_d_star_FF[i])
            + "\t"
            + str(mean_d_min_FF[i])
            + "\t"
            + str(std_d_min_FF[i])
            + "\t"
            + str(mean_A_FF[i])
            + "\t"
            + str(std_A_FF[i])
            + "\n"
        )
        
    f.write("RDG1 \t  RDG1 \t  RDG1 \t RDG1 \t RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \n")
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_RDG1[i])
            + "\t"
            + str(std_delta_d_RDG1[i])
            + "\t"
            + str(mean_delta_d_star_RDG1[i])
            + "\t"
            + str(std_delta_d_star_RDG1[i])
            + "\t"
            + str(mean_d_min_RDG1[i])
            + "\t"
            + str(std_d_min_RDG1[i])
            + "\t"
            + str(mean_A_RDG1[i])
            + "\t"
            + str(std_A_RDG1[i])
            + "\n"
        )
        
    f.write("RDG2 \t  RDG2 \t RDG2 \t RDG2 \t RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_RDG2[i])
            + "\t"
            + str(std_delta_d_RDG2[i])
            + "\t"
            + str(mean_delta_d_star_RDG2[i])
            + "\t"
            + str(std_delta_d_star_RDG2[i])
            + "\t"
            + str(mean_d_min_RDG2[i])
            + "\t"
            + str(std_d_min_RDG2[i])
            + "\t"
            + str(mean_A_RDG2[i])
            + "\t"
            + str(std_A_RDG2[i])
            + "\n"
        )
        
    f.write("RDG \t  RDG \t RDG \t  RDG \t RDG \t  RDG \t  RDG \t  RDG \t  RDG \n")        
    f.write("date \t mean delta d \t std delta d \t mean delta d star \t std delta d star \t mean d_min \t std d_min \t mean A \t std A \n")
    for i in range(len(mean_delta_d_RDG)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_delta_d_RDG[i])
            + "\t"
            + str(std_delta_d_RDG[i])
            + "\t"
            + str(mean_delta_d_star_RDG[i])
            + "\t"
            + str(std_delta_d_star_RDG[i])
            + "\t"
            + str(mean_d_min_RDG[i])
            + "\t"
            + str(std_d_min_RDG[i])
            + "\t"
            + str(mean_A_RDG[i])
            + "\t"
            + str(std_A_RDG[i])
            + "\n"
        )
        
        
    f.close()

def plot_recovery_indicators_with_maturation():
    maturation = [10, 13, 17, 21]
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indicators_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [_, mean_delta_d_FF1, std_delta_d_FF1, mean_delta_d_star_FF1, std_delta_d_star_FF1, mean_d_min_FF1, std_d_min_FF1,  mean_A_FF1, std_A_FF1,
             mean_delta_d_FF2, std_delta_d_FF2, mean_delta_d_star_FF2, std_delta_d_star_FF2, mean_d_min_FF2, std_d_min_FF2,  mean_A_FF2, std_A_FF2,
             mean_delta_d_RDG1, std_delta_d_RDG1, mean_delta_d_star_RDG1, std_delta_d_star_RDG1, mean_d_min_RDG1, std_d_min_RDG1,  mean_A_RDG1, std_A_RDG1,
             mean_delta_d_RDG2, std_delta_d_RDG2, mean_delta_d_star_RDG2, std_delta_d_star_RDG2, mean_d_min_RDG2, std_d_min_RDG2,  mean_A_RDG2, std_A_RDG2,
             mean_delta_d_FF, std_delta_d_FF, mean_delta_d_star_FF, std_delta_d_star_FF, mean_d_min_FF, std_d_min_FF,  mean_A_FF, std_A_FF,
             mean_delta_d_RDG, std_delta_d_RDG, mean_delta_d_star_RDG, std_delta_d_star_RDG, mean_d_min_RDG, std_d_min_RDG,  mean_A_RDG, std_A_RDG
             ] = pickle.load(f)
    [_, mean_delta_d_FF1, std_delta_d_FF1, mean_delta_d_star_FF1, std_delta_d_star_FF1, mean_d_min_FF1, std_d_min_FF1,  mean_A_FF1, std_A_FF1,
            mean_delta_d_FF2, std_delta_d_FF2, mean_delta_d_star_FF2, std_delta_d_star_FF2, mean_d_min_FF2, std_d_min_FF2,  mean_A_FF2, std_A_FF2,
            mean_delta_d_RDG1, std_delta_d_RDG1, mean_delta_d_star_RDG1, std_delta_d_star_RDG1, mean_d_min_RDG1, std_d_min_RDG1,  mean_A_RDG1, std_A_RDG1,
            mean_delta_d_RDG2, std_delta_d_RDG2, mean_delta_d_star_RDG2, std_delta_d_star_RDG2, mean_d_min_RDG2, std_d_min_RDG2,  mean_A_RDG2, std_A_RDG2,
            mean_delta_d_FF, std_delta_d_FF, mean_delta_d_star_FF, std_delta_d_star_FF, mean_d_min_FF, std_d_min_FF,  mean_A_FF, std_A_FF,
            mean_delta_d_RDG, std_delta_d_RDG, mean_delta_d_star_RDG, std_delta_d_star_RDG, mean_d_min_RDG, std_d_min_RDG,  mean_A_RDG, std_A_RDG
            ] = [_, mean_delta_d_FF1[1:], std_delta_d_FF1[1:], mean_delta_d_star_FF1[1:], std_delta_d_star_FF1[1:], mean_d_min_FF1[1:], std_d_min_FF1[1:],  mean_A_FF1[1:], std_A_FF1[1:],
            mean_delta_d_FF2[1:], std_delta_d_FF2[1:], mean_delta_d_star_FF2[1:], std_delta_d_star_FF2[1:], mean_d_min_FF2[1:], std_d_min_FF2[1:],  mean_A_FF2[1:], std_A_FF2[1:],
            mean_delta_d_RDG1[1:], std_delta_d_RDG1[1:], mean_delta_d_star_RDG1[1:], std_delta_d_star_RDG1[1:], mean_d_min_RDG1[1:], std_d_min_RDG1[1:],  mean_A_RDG1[1:], std_A_RDG1[1:],
            mean_delta_d_RDG2[1:], std_delta_d_RDG2[1:], mean_delta_d_star_RDG2[1:], std_delta_d_star_RDG2[1:], mean_d_min_RDG2[1:], std_d_min_RDG2[1:],  mean_A_RDG2[1:], std_A_RDG2[1:],
            mean_delta_d_FF[1:], std_delta_d_FF[1:], mean_delta_d_star_FF[1:], std_delta_d_star_FF[1:], mean_d_min_FF[1:], std_d_min_FF[1:],  mean_A_FF[1:], std_A_FF[1:],
            mean_delta_d_RDG[1:], std_delta_d_RDG[1:], mean_delta_d_star_RDG[1:], std_delta_d_star_RDG[1:], mean_d_min_RDG[1:], std_d_min_RDG[1:],  mean_A_RDG[1:], std_A_RDG[1:]
            ]    
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    
    maturation_FF = [m - 0.1 for m in maturation]
    maturation_RDG = [m + 0.1 for m in maturation]
    fig_delta_d_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_1 = fig_delta_d_1.gca()
    fig_delta_d_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_2 = fig_delta_d_2.gca()
    fig_delta_d = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d = fig_delta_d.gca()
    ax_delta_d.errorbar(maturation_FF, mean_delta_d_FF, yerr=std_delta_d_FF, lw=0, label='FF', **kwargs_FF)
    ax_delta_d_1.errorbar(maturation_FF, mean_delta_d_FF1, yerr=std_delta_d_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_2.errorbar(maturation_FF, mean_delta_d_FF2, yerr=std_delta_d_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_1.errorbar(maturation_RDG, mean_delta_d_RDG1, yerr=std_delta_d_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_delta_d.errorbar(maturation_RDG, mean_delta_d_RDG, yerr=std_delta_d_RDG, lw=0,  label='RDG', **kwargs_RDG)
    ax_delta_d_2.errorbar(maturation_RDG, mean_delta_d_RDG2, yerr=std_delta_d_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
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
    savefigure.save_as_png(fig_delta_d, "delta_d_vs_maturation_1+2")
    savefigure.save_as_png(fig_delta_d_1, "delta_d_vs_maturation_1")
    savefigure.save_as_png(fig_delta_d_2, "delta_d_vs_maturation_2")

    fig_delta_d_star_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star_1 = fig_delta_d_star_1.gca()
    fig_delta_d_star = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star = fig_delta_d_star.gca()
    fig_delta_d_star_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star_2 = fig_delta_d_star_2.gca()
    ax_delta_d_star_1.errorbar(maturation_FF, mean_delta_d_star_FF1, yerr=std_delta_d_star_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_star.errorbar(maturation_FF, mean_delta_d_star_FF, yerr=std_delta_d_star_FF, lw=0, label='FF', **kwargs_FF)
    ax_delta_d_star_2.errorbar(maturation_FF, mean_delta_d_star_FF2, yerr=std_delta_d_star_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_star_1.errorbar(maturation_RDG, mean_delta_d_star_RDG1, yerr=std_delta_d_star_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_delta_d_star.errorbar(maturation_RDG, mean_delta_d_star_RDG, yerr=std_delta_d_star_RDG, lw=0,  label='RDG', **kwargs_RDG)
    ax_delta_d_star_2.errorbar(maturation_RDG, mean_delta_d_star_RDG2, yerr=std_delta_d_star_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_star.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star.set_title(r'$\Delta d^*$ vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_title(r'$\Delta d^*$ vs maturation 1', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_title(r'$\Delta d^*$ vs maturation 2', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star.set_ylabel(r'$\Delta d^*$ [-]', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_ylabel(r'$\Delta d^*$ [-]', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_ylabel(r'$\Delta d^*$ [-]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star, "delta_d_star_vs_maturation_1+2")
    savefigure.save_as_png(fig_delta_d_star_1, "delta_d_star_vs_maturation_1")
    savefigure.save_as_png(fig_delta_d_star_2, "delta_d_star_vs_maturation_2")

    fig_A_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_A_1 = fig_A_1.gca()
    fig_A = createfigure.rectangle_rz_figure(pixels=180)
    ax_A = fig_A.gca()
    fig_A_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_A_2 = fig_A_2.gca()
    ax_A_1.errorbar(maturation_FF, mean_A_FF1, yerr=std_A_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_A_2.errorbar(maturation_FF, mean_A_FF2, yerr=std_A_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_A.errorbar(maturation_FF, mean_A_FF, yerr=std_A_FF, lw=0, label='FF', **kwargs_FF)
    ax_A_1.errorbar(maturation_RDG, mean_A_RDG1, yerr=std_A_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_A_2.errorbar(maturation_RDG, mean_A_RDG2, yerr=std_A_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_A.errorbar(maturation_RDG, mean_A_RDG, yerr=std_A_RDG, lw=0, label='RDG', **kwargs_RDG)
    ax_A.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A.set_title('A vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_A_1.set_title('A vs maturation 1', font=fonts.serif_rz_legend())
    ax_A_2.set_title('A vs maturation 2', font=fonts.serif_rz_legend())
    ax_A.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_A_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_A_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_A.set_ylabel('A [mm]', font=fonts.serif_rz_legend())
    ax_A_1.set_ylabel('A [mm]', font=fonts.serif_rz_legend())
    ax_A_2.set_ylabel('A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A, "A_vs_maturation_1+2")
    savefigure.save_as_png(fig_A_1, "A_vs_maturation_1")
    savefigure.save_as_png(fig_A_2, "A_vs_maturation_2")

    fig_d_min_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_d_min_1 = fig_d_min_1.gca()
    fig_d_min = createfigure.rectangle_rz_figure(pixels=180)
    ax_d_min = fig_d_min.gca()
    fig_d_min_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_d_min_2 = fig_d_min_2.gca()
    ax_d_min_1.errorbar(maturation_FF, mean_d_min_FF1, yerr=std_d_min_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_d_min_2.errorbar(maturation_FF, mean_d_min_FF2, yerr=std_d_min_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_d_min.errorbar(maturation_FF, mean_d_min_FF, yerr=std_d_min_FF, lw=0, label='FF', **kwargs_FF)
    ax_d_min_1.errorbar(maturation_RDG, mean_d_min_RDG1, yerr=std_d_min_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_d_min_2.errorbar(maturation_RDG, mean_d_min_RDG2, yerr=std_d_min_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_d_min.errorbar(maturation_RDG, mean_d_min_RDG, yerr=std_d_min_RDG, lw=0, label='RDG', **kwargs_RDG)
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
    savefigure.save_as_png(fig_d_min, "d_min_vs_maturation_1+2")
    savefigure.save_as_png(fig_d_min_1, "d_min_vs_maturation_1")
    savefigure.save_as_png(fig_d_min_2, "d_min_vs_maturation_2")

def plot_laser_indicators_vs_texturometer_forces():
    path_to_processed_data_laser = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\laser\processed_data'
    complete_pkl_filename_laser = path_to_processed_data_laser + "/indicators_mean_std.pkl"
    with open(complete_pkl_filename_laser, "rb") as f:
        [_, mean_delta_d_FF1, std_delta_d_FF1, mean_delta_d_star_FF1, std_delta_d_star_FF1, mean_d_min_FF1, std_d_min_FF1,  mean_A_FF1, std_A_FF1,
             mean_delta_d_FF2, std_delta_d_FF2, mean_delta_d_star_FF2, std_delta_d_star_FF2, mean_d_min_FF2, std_d_min_FF2,  mean_A_FF2, std_A_FF2,
             mean_delta_d_RDG1, std_delta_d_RDG1, mean_delta_d_star_RDG1, std_delta_d_star_RDG1, mean_d_min_RDG1, std_d_min_RDG1,  mean_A_RDG1, std_A_RDG1,
             mean_delta_d_RDG2, std_delta_d_RDG2, mean_delta_d_star_RDG2, std_delta_d_star_RDG2, mean_d_min_RDG2, std_d_min_RDG2,  mean_A_RDG2, std_A_RDG2,
             mean_delta_d_FF, std_delta_d_FF, mean_delta_d_star_FF, std_delta_d_star_FF, mean_d_min_FF, std_d_min_FF,  mean_A_FF, std_A_FF,
             mean_delta_d_RDG, std_delta_d_RDG, mean_delta_d_star_RDG, std_delta_d_star_RDG, mean_d_min_RDG, std_d_min_RDG,  mean_A_RDG, std_A_RDG
             ] = pickle.load(f)


    path_to_processed_data_texturometer = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename_texturometer = path_to_processed_data_texturometer + "/forces_mean_std.pkl"
    with open(complete_pkl_filename_texturometer, "rb") as f:
        [_, mean_force20_FF1, std_force20_FF1, mean_force80_FF1, std_force80_FF1,
             mean_force20_FF2, std_force20_FF2, mean_force80_FF2, std_force80_FF2,
             mean_force20_RDG1, std_force20_RDG1, mean_force80_RDG1, std_force80_RDG1,
             mean_force20_RDG2, std_force20_RDG2, mean_force80_RDG2, std_force80_RDG2,
             mean_force20_FF, std_force20_FF, mean_force80_FF, std_force80_FF,
             mean_force20_RDG, std_force20_RDG, mean_force80_RDG, std_force80_RDG
             ] = pickle.load(f)


    pixels=180
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    
    #A vs force 80    
    force_80_1 = np.concatenate((mean_force80_FF1, mean_force80_RDG1))
    index_force_1_nan = np.isnan(force_80_1) 
    A_1 = np.concatenate(( mean_A_FF1 , mean_A_RDG1 ))
    index_A_1_nan = np.isnan(A_1)
    indices_force_or_A_1_nan = [index_force_1_nan[i] or index_A_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_A_1_nan[i]])
    A_1_without_nan_force = np.array([A_1[i] for i in range(len(A_1)) if not indices_force_or_A_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, A_1_without_nan_force)
    fitted_response_A_1 = model.predict(force_80_1)
    a_A_1 = reg.coef_
    b_A_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A_1 = reg.score(force_80_1, A_1_without_nan_force)
    
    fig_A_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force80_1 = fig_A_vs_force80_1.gca()
    ax_A_vs_force80_1.plot([40, 40], [min(mean_A_FF1) - max(std_A_FF1[~np.isnan(std_A_FF1)]), max(mean_A_RDG1) + max(std_A_RDG1[~np.isnan(std_A_RDG1)]) ], '--r', lw=1, alpha=0.2)
    ax_A_vs_force80_1.errorbar(mean_force80_FF1, mean_A_FF1, yerr=std_A_FF1, xerr=std_force80_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    ax_A_vs_force80_1.errorbar(mean_force80_RDG1, mean_A_RDG1, yerr=std_A_RDG1, xerr=std_force80_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_A_vs_force80_1.plot(force_80_1, fitted_response_A_1, ':k', label=' A = ' + str(np.round(a_A_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_A_1[0], 4)) + '\n R2 = ' + str(np.round(score_A_1, 2)) )
    ax_A_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_vs_force80_1.set_title(r'A vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_A_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force80_1.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force80_1, "A_vs_force80_1")

    force_80_2 = np.concatenate((mean_force80_FF2, mean_force80_RDG2))
    index_force_2_nan = np.isnan(force_80_2) 
    A_2 = np.concatenate(( mean_A_FF2 , mean_A_RDG2 ))
    index_A_2_nan = np.isnan(A_2)
    indices_force_or_A_2_nan = [index_force_2_nan[i] or index_A_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_A_2_nan[i]])
    A_2_without_nan_force = np.array([A_2[i] for i in range(len(A_2)) if not indices_force_or_A_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, A_2_without_nan_force)
    fitted_response_A_2 = model.predict(force_80_2)
    a_A_2 = reg.coef_
    b_A_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A_2 = reg.score(force_80_2, A_2_without_nan_force)

    fig_A_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force80_2 = fig_A_vs_force80_2.gca()
    ax_A_vs_force80_2.plot([40, 40], [min(mean_A_FF2[~np.isnan(mean_A_FF2)]) - max(std_A_FF2[~np.isnan(std_A_FF2)]), max(mean_A_RDG2[~np.isnan(mean_A_RDG2)]) + max(std_A_RDG2[~np.isnan(std_A_RDG2)]) ], '--r', lw=1, alpha=0.2)
    ax_A_vs_force80_2.errorbar(mean_force80_FF2, mean_A_FF2, yerr=std_A_FF2, xerr=std_force80_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    ax_A_vs_force80_2.errorbar(mean_force80_RDG2, mean_A_RDG2, yerr=std_A_RDG2, xerr=std_force80_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_A_vs_force80_1.plot(force_80_2, fitted_response_A_2, ':k', label=' A = ' + str(np.round(a_A_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_A_2[0], 4)) + '\n R2 = ' + str(np.round(score_A_2, 2)) )
    ax_A_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_vs_force80_2.set_title(r'A vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_A_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force80_2.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force80_2, "A_vs_force80_2")

    force_80 = np.concatenate((mean_force80_FF, mean_force80_RDG))
    index_force_nan = np.isnan(force_80) 
    A = np.concatenate(( mean_A_FF , mean_A_RDG))
    index_A_nan = np.isnan(A)
    indices_force_or_A_nan = [index_force_nan[i] or index_A_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_A_nan[i]])
    A_without_nan_force = np.array([A[i] for i in range(len(A)) if not indices_force_or_A_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, A_without_nan_force)
    fitted_response_A = model.predict(force_80)
    a_A = reg.coef_
    b_A = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_A = reg.score(force_80, A_without_nan_force)

    fig_A_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_A_vs_force80 = fig_A_vs_force80.gca()
    ax_A_vs_force80.plot([40, 40], [min(mean_A_FF[~np.isnan(mean_A_FF)]) - max(std_A_FF[~np.isnan(std_A_FF)]), max(mean_A_RDG[~np.isnan(mean_A_RDG)]) + max(std_A_RDG[~np.isnan(std_A_RDG)]) ], '--r', lw=1, alpha=0.2)
    ax_A_vs_force80.errorbar(mean_force80_FF, mean_A_FF, yerr=std_A_FF, xerr=std_force80_FF ,lw=0, label='FF', **kwargs_FF)
    ax_A_vs_force80.errorbar(mean_force80_RDG, mean_A_RDG, yerr=std_A_RDG, xerr=std_force80_RDG ,lw=0, label='RDG', **kwargs_RDG)
    ax_A_vs_force80.plot(force_80, fitted_response_A, ':k', label=' A = ' + str(np.round(a_A[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_A[0], 4)) + '\n R2 = ' + str(np.round(score_A, 2)) )
    ax_A_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_A_vs_force80.set_title(r'A vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_A_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_A_vs_force80.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_A_vs_force80, "A_vs_force80_1+2")

    # #A vs force 20    
    # fig_A_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    # ax_A_vs_force20_1 = fig_A_vs_force20_1.gca()
    # ax_A_vs_force20_1.errorbar(mean_force20_FF1, mean_A_FF1, yerr=std_A_FF1, xerr=std_force20_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    # ax_A_vs_force20_1.errorbar(mean_force20_RDG1, mean_A_RDG1, yerr=std_A_RDG1, xerr=std_force20_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    # ax_A_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_A_vs_force20_1.set_title(r'A vs Force 20% 1', font=fonts.serif_rz_legend())
    # ax_A_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_A_vs_force20_1.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_A_vs_force20_1, "A_vs_force20_1")

    # fig_A_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    # ax_A_vs_force20_2 = fig_A_vs_force20_2.gca()
    # ax_A_vs_force20_2.errorbar(mean_force20_FF2, mean_A_FF2, yerr=std_A_FF2, xerr=std_force20_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    # ax_A_vs_force20_2.errorbar(mean_force20_RDG2, mean_A_RDG2, yerr=std_A_RDG2, xerr=std_force20_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    # ax_A_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_A_vs_force20_2.set_title(r'A vs Force 20% 2', font=fonts.serif_rz_legend())
    # ax_A_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_A_vs_force20_2.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_A_vs_force20_2, "A_vs_force20_2")

    # fig_A_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    # ax_A_vs_force20 = fig_A_vs_force20.gca()
    # ax_A_vs_force20.errorbar(mean_force20_FF, mean_A_FF, yerr=std_A_FF, xerr=std_force20_FF ,lw=0, label='FF', **kwargs_FF)
    # ax_A_vs_force20.errorbar(mean_force20_RDG, mean_A_RDG, yerr=std_A_RDG, xerr=std_force20_RDG ,lw=0, label='RDG', **kwargs_RDG)
    # ax_A_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_A_vs_force20.set_title(r'A vs Force 20% 1+2', font=fonts.serif_rz_legend())
    # ax_A_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_A_vs_force20.set_ylabel(r'A [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_A_vs_force20, "A_vs_force20_1+2")




    #delta_d_star vs force 80    
    force_80_1 = np.concatenate((mean_force80_FF1, mean_force80_RDG1))
    index_force_1_nan = np.isnan(force_80_1) 
    delta_d_star_1 = np.concatenate(( mean_delta_d_star_FF1 , mean_delta_d_star_RDG1 ))
    index_delta_d_star_1_nan = np.isnan(delta_d_star_1)
    indices_force_or_delta_d_star_1_nan = [index_force_1_nan[i] or index_delta_d_star_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_delta_d_star_1_nan[i]])
    delta_d_star_1_without_nan_force = np.array([delta_d_star_1[i] for i in range(len(delta_d_star_1)) if not indices_force_or_delta_d_star_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, delta_d_star_1_without_nan_force)
    fitted_response_delta_d_star_1 = model.predict(force_80_1)
    a_delta_d_star_1 = reg.coef_
    b_delta_d_star_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star_1 = reg.score(force_80_1, delta_d_star_1_without_nan_force)
    
    fig_delta_d_star_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force80_1 = fig_delta_d_star_vs_force80_1.gca()
    ax_delta_d_star_vs_force80_1.plot([40, 40], [min(mean_delta_d_star_FF1) - max(std_delta_d_star_FF1[~np.isnan(std_delta_d_star_FF1)]), max(mean_delta_d_star_RDG1) + max(std_delta_d_star_RDG1[~np.isnan(std_delta_d_star_RDG1)]) ], '--r', lw=1, alpha=0.2)
    ax_delta_d_star_vs_force80_1.errorbar(mean_force80_FF1, mean_delta_d_star_FF1, yerr=std_delta_d_star_FF1, xerr=std_force80_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_star_vs_force80_1.errorbar(mean_force80_RDG1, mean_delta_d_star_RDG1, yerr=std_delta_d_star_RDG1, xerr=std_force80_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_delta_d_star_vs_force80_1.plot(force_80_1, fitted_response_delta_d_star_1, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_delta_d_star_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_star_1[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star_1, 2)) )
    ax_delta_d_star_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_vs_force80_1.set_title(r'$\Delta d^*$ vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_1.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force80_1, "delta_d_star_vs_force80_1")

    force_80_2 = np.concatenate((mean_force80_FF2, mean_force80_RDG2))
    index_force_2_nan = np.isnan(force_80_2) 
    delta_d_star_2 = np.concatenate(( mean_delta_d_star_FF2 , mean_delta_d_star_RDG2 ))
    index_delta_d_star_2_nan = np.isnan(delta_d_star_2)
    indices_force_or_delta_d_star_2_nan = [index_force_2_nan[i] or index_delta_d_star_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_delta_d_star_2_nan[i]])
    delta_d_star_2_without_nan_force = np.array([delta_d_star_2[i] for i in range(len(delta_d_star_2)) if not indices_force_or_delta_d_star_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, delta_d_star_2_without_nan_force)
    fitted_response_delta_d_star_2 = model.predict(force_80_2)
    a_delta_d_star_2 = reg.coef_
    b_delta_d_star_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star_2 = reg.score(force_80_2, delta_d_star_2_without_nan_force)
    
    fig_delta_d_star_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force80_2 = fig_delta_d_star_vs_force80_2.gca()
    ax_delta_d_star_vs_force80_2.plot([40, 40], [min(mean_delta_d_star_FF2[~np.isnan(mean_delta_d_star_FF2)]) - max(std_delta_d_star_FF2[~np.isnan(std_delta_d_star_FF2)]), max(mean_delta_d_star_RDG2[~np.isnan(mean_delta_d_star_RDG2)]) + max(std_delta_d_star_RDG2[~np.isnan(std_delta_d_star_RDG2)]) ], '--r', lw=1, alpha=0.2)
    ax_delta_d_star_vs_force80_2.errorbar(mean_force80_FF2, mean_delta_d_star_FF2, yerr=std_delta_d_star_FF2, xerr=std_force80_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_star_vs_force80_2.errorbar(mean_force80_RDG2, mean_delta_d_star_RDG2, yerr=std_delta_d_star_RDG2, xerr=std_force80_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_star_vs_force80_2.plot(force_80_2, fitted_response_delta_d_star_2, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_delta_d_star_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_star_2[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star_2, 2)) )
    ax_delta_d_star_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_vs_force80_2.set_title(r'$\Delta d^*$ vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80_2.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force80_2, "delta_d_star_vs_force80_2")

    force_80 = np.concatenate((mean_force80_FF, mean_force80_RDG))
    index_force_nan = np.isnan(force_80) 
    delta_d_star = np.concatenate(( mean_delta_d_star_FF , mean_delta_d_star_RDG ))
    index_delta_d_star_nan = np.isnan(delta_d_star)
    indices_force_or_delta_d_star_nan = [index_force_nan[i] or index_delta_d_star_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_delta_d_star_nan[i]])
    delta_d_star_without_nan_force = np.array([delta_d_star[i] for i in range(len(delta_d_star)) if not indices_force_or_delta_d_star_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, delta_d_star_without_nan_force)
    fitted_response_delta_d_star = model.predict(force_80)
    a_delta_d_star = reg.coef_
    b_delta_d_star = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_star = reg.score(force_80, delta_d_star_without_nan_force)
    
    fig_delta_d_star_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_star_vs_force80 = fig_delta_d_star_vs_force80.gca()
    ax_delta_d_star_vs_force80.plot([40, 40], [min(mean_delta_d_star_FF[~np.isnan(mean_delta_d_star_FF)]) - max(std_delta_d_star_FF[~np.isnan(std_delta_d_star_FF)]), max(mean_delta_d_star_RDG[~np.isnan(mean_delta_d_star_RDG)]) + max(std_delta_d_star_RDG[~np.isnan(std_delta_d_star_RDG)]) ], '--r', lw=1, alpha=0.2)
    ax_delta_d_star_vs_force80.errorbar(mean_force80_FF, mean_delta_d_star_FF, yerr=std_delta_d_star_FF, xerr=std_force80_FF ,lw=0, label='FF', **kwargs_FF)
    ax_delta_d_star_vs_force80.errorbar(mean_force80_RDG, mean_delta_d_star_RDG, yerr=std_delta_d_star_RDG, xerr=std_force80_RDG ,lw=0, label='RDG', **kwargs_RDG)
    ax_delta_d_star_vs_force80.plot(force_80, fitted_response_delta_d_star, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_delta_d_star[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_star[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_star, 2)) )
    ax_delta_d_star_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_star_vs_force80.set_title(r'$\Delta d^*$ vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_vs_force80.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_vs_force80, "delta_d_star_vs_force80_1+2")

    # #delta_d_star vs force 20    
    # fig_delta_d_star_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    # ax_delta_d_star_vs_force20_1 = fig_delta_d_star_vs_force20_1.gca()
    # ax_delta_d_star_vs_force20_1.errorbar(mean_force20_FF1, mean_delta_d_star_FF1, yerr=std_delta_d_star_FF1, xerr=std_force20_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    # ax_delta_d_star_vs_force20_1.errorbar(mean_force20_RDG1, mean_delta_d_star_RDG1, yerr=std_delta_d_star_RDG1, xerr=std_force20_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    # ax_delta_d_star_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_delta_d_star_vs_force20_1.set_title(r'$\Delta d^*$ vs Force 20% 1', font=fonts.serif_rz_legend())
    # ax_delta_d_star_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_delta_d_star_vs_force20_1.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_delta_d_star_vs_force20_1, "delta_d_star_vs_force20_1")

    # fig_delta_d_star_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    # ax_delta_d_star_vs_force20_2 = fig_delta_d_star_vs_force20_2.gca()
    # ax_delta_d_star_vs_force20_2.errorbar(mean_force20_FF2, mean_delta_d_star_FF2, yerr=std_delta_d_star_FF2, xerr=std_force20_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    # ax_delta_d_star_vs_force20_2.errorbar(mean_force20_RDG2, mean_delta_d_star_RDG2, yerr=std_delta_d_star_RDG2, xerr=std_force20_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    # ax_delta_d_star_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_delta_d_star_vs_force20_2.set_title(r'$\Delta d^*$ vs Force 20% 2', font=fonts.serif_rz_legend())
    # ax_delta_d_star_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_delta_d_star_vs_force20_2.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_delta_d_star_vs_force20_2, "delta_d_star_vs_force20_2")

    # fig_delta_d_star_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    # ax_delta_d_star_vs_force20 = fig_delta_d_star_vs_force20.gca()
    # ax_delta_d_star_vs_force20.errorbar(mean_force20_FF, mean_delta_d_star_FF, yerr=std_delta_d_star_FF, xerr=std_force20_FF ,lw=0, label='FF', **kwargs_FF)
    # ax_delta_d_star_vs_force20.errorbar(mean_force20_RDG, mean_delta_d_star_RDG, yerr=std_delta_d_star_RDG, xerr=std_force20_RDG ,lw=0, label='RDG', **kwargs_RDG)
    # ax_delta_d_star_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_delta_d_star_vs_force20.set_title(r'$\Delta d^*$ vs Force 20% 1+2', font=fonts.serif_rz_legend())
    # ax_delta_d_star_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_delta_d_star_vs_force20.set_ylabel(r'$\Delta d^*$ [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_delta_d_star_vs_force20, "delta_d_star_vs_force20_1+2")

    #d_min vs force 80    
    force_80_1 = np.concatenate((mean_force80_FF1, mean_force80_RDG1))
    index_force_1_nan = np.isnan(force_80_1) 
    d_min_1 = np.concatenate(( mean_d_min_FF1 , mean_d_min_RDG1 ))
    index_d_min_1_nan = np.isnan(d_min_1)
    indices_force_or_d_min_1_nan = [index_force_1_nan[i] or index_d_min_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_d_min_1_nan[i]])
    d_min_1_without_nan_force = np.array([d_min_1[i] for i in range(len(d_min_1)) if not indices_force_or_d_min_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, d_min_1_without_nan_force)
    fitted_response_d_min_1 = model.predict(force_80_1)
    a_d_min_1 = reg.coef_
    b_d_min_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min_1 = reg.score(force_80_1, d_min_1_without_nan_force)
    
    fig_d_min_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force80_1 = fig_d_min_vs_force80_1.gca()
    ax_d_min_vs_force80_1.plot([40, 40], [min(mean_d_min_FF1) - max(std_d_min_FF1[~np.isnan(std_d_min_FF1)]), max(mean_d_min_RDG1) + max(std_d_min_RDG1[~np.isnan(std_d_min_RDG1)]) ], '--r', lw=1, alpha=0.2)
    ax_d_min_vs_force80_1.errorbar(mean_force80_FF1, mean_d_min_FF1, yerr=std_d_min_FF1, xerr=std_force80_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    ax_d_min_vs_force80_1.errorbar(mean_force80_RDG1, mean_d_min_RDG1, yerr=std_d_min_RDG1, xerr=std_force80_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_d_min_vs_force80_1.plot(force_80_1, fitted_response_d_min_1, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_d_min_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_d_min_1[0], 4)) + '\n R2 = ' + str(np.round(score_d_min_1, 2)) )
    ax_d_min_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_d_min_vs_force80_1.set_title(r'$d_{min}$ vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_1.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force80_1, "d_min_vs_force80_1")

    force_80_2 = np.concatenate((mean_force80_FF2, mean_force80_RDG2))
    index_force_2_nan = np.isnan(force_80_2) 
    d_min_2 = np.concatenate(( mean_d_min_FF2 , mean_d_min_RDG2 ))
    index_d_min_2_nan = np.isnan(d_min_2)
    indices_force_or_d_min_2_nan = [index_force_2_nan[i] or index_d_min_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_d_min_2_nan[i]])
    d_min_2_without_nan_force = np.array([d_min_2[i] for i in range(len(d_min_2)) if not indices_force_or_d_min_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, d_min_2_without_nan_force)
    fitted_response_d_min_2 = model.predict(force_80_2)
    a_d_min_2 = reg.coef_
    b_d_min_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min_2 = reg.score(force_80_2, d_min_2_without_nan_force)
    
    fig_d_min_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force80_2 = fig_d_min_vs_force80_2.gca()
    ax_d_min_vs_force80_2.plot([40, 40], [min(mean_d_min_FF2[~np.isnan(mean_d_min_FF2)]) - max(std_d_min_FF2[~np.isnan(std_d_min_FF2)]), max(mean_d_min_RDG2[~np.isnan(mean_d_min_RDG2)]) + max(std_d_min_RDG2[~np.isnan(std_d_min_RDG2)]) ], '--r', lw=1, alpha=0.2)
    ax_d_min_vs_force80_2.errorbar(mean_force80_FF2, mean_d_min_FF2, yerr=std_d_min_FF2, xerr=std_force80_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    ax_d_min_vs_force80_2.errorbar(mean_force80_RDG2, mean_d_min_RDG2, yerr=std_d_min_RDG2, xerr=std_force80_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_d_min_vs_force80_2.plot(force_80_2, fitted_response_d_min_2, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_d_min_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_d_min_2[0], 4)) + '\n R2 = ' + str(np.round(score_d_min_2, 2)) )
    ax_d_min_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_d_min_vs_force80_2.set_title(r'$d_{min}$ vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80_2.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force80_2, "d_min_vs_force80_2")

    force_80 = np.concatenate((mean_force80_FF, mean_force80_RDG))
    index_force_nan = np.isnan(force_80) 
    d_min = np.concatenate(( mean_d_min_FF , mean_d_min_RDG ))
    index_d_min_nan = np.isnan(d_min)
    indices_force_or_d_min_nan = [index_force_nan[i] or index_d_min_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_d_min_nan[i]])
    d_min_without_nan_force = np.array([d_min[i] for i in range(len(d_min)) if not indices_force_or_d_min_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, d_min_without_nan_force)
    fitted_response_d_min = model.predict(force_80)
    a_d_min = reg.coef_
    b_d_min = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_d_min = reg.score(force_80, d_min_without_nan_force)
    
    fig_d_min_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_d_min_vs_force80 = fig_d_min_vs_force80.gca()
    ax_d_min_vs_force80.plot([40, 40], [min(mean_d_min_FF[~np.isnan(mean_d_min_FF)]) - max(std_d_min_FF[~np.isnan(std_d_min_FF)]), max(mean_d_min_RDG[~np.isnan(mean_d_min_RDG)]) + max(std_d_min_RDG[~np.isnan(std_d_min_RDG)]) ], '--r', lw=1, alpha=0.2)
    ax_d_min_vs_force80.errorbar(mean_force80_FF, mean_d_min_FF, yerr=std_d_min_FF, xerr=std_force80_FF ,lw=0, label='FF', **kwargs_FF)
    ax_d_min_vs_force80.errorbar(mean_force80_RDG, mean_d_min_RDG, yerr=std_d_min_RDG, xerr=std_force80_RDG ,lw=0, label='RDG', **kwargs_RDG)
    ax_d_min_vs_force80.plot(force_80, fitted_response_d_min, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_d_min[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_d_min[0], 4)) + '\n R2 = ' + str(np.round(score_d_min, 2)) )
    ax_d_min_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_d_min_vs_force80.set_title(r'$d_{min}$ vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_d_min_vs_force80.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_d_min_vs_force80, "d_min_vs_force80_1+2")

    # #d_min vs force 20    
    # fig_d_min_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    # ax_d_min_vs_force20_1 = fig_d_min_vs_force20_1.gca()
    # ax_d_min_vs_force20_1.errorbar(mean_force20_FF1, mean_d_min_FF1, yerr=std_d_min_FF1, xerr=std_force20_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    # ax_d_min_vs_force20_1.errorbar(mean_force20_RDG1, mean_d_min_RDG1, yerr=std_d_min_RDG1, xerr=std_force20_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    # ax_d_min_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_d_min_vs_force20_1.set_title(r'$d_{min}$ vs Force 20% 1', font=fonts.serif_rz_legend())
    # ax_d_min_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_d_min_vs_force20_1.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_d_min_vs_force20_1, "d_min_vs_force20_1")

    # fig_d_min_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    # ax_d_min_vs_force20_2 = fig_d_min_vs_force20_2.gca()
    # ax_d_min_vs_force20_2.errorbar(mean_force20_FF2, mean_d_min_FF2, yerr=std_d_min_FF2, xerr=std_force20_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    # ax_d_min_vs_force20_2.errorbar(mean_force20_RDG2, mean_d_min_RDG2, yerr=std_d_min_RDG2, xerr=std_force20_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    # ax_d_min_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_d_min_vs_force20_2.set_title(r'$d_{min}$ vs Force 20% 2', font=fonts.serif_rz_legend())
    # ax_d_min_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_d_min_vs_force20_2.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_d_min_vs_force20_2, "d_min_vs_force20_2")

    # fig_d_min_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    # ax_d_min_vs_force20 = fig_d_min_vs_force20.gca()
    # ax_d_min_vs_force20.errorbar(mean_force20_FF, mean_d_min_FF, yerr=std_d_min_FF, xerr=std_force20_FF ,lw=0, label='FF', **kwargs_FF)
    # ax_d_min_vs_force20.errorbar(mean_force20_RDG, mean_d_min_RDG, yerr=std_d_min_RDG, xerr=std_force20_RDG ,lw=0, label='RDG', **kwargs_RDG)
    # ax_d_min_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_d_min_vs_force20.set_title(r'$d_{min}$ vs Force 20% 1+2', font=fonts.serif_rz_legend())
    # ax_d_min_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_d_min_vs_force20.set_ylabel(r'$d_{min}$ [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_d_min_vs_force20, "d_min_vs_force20_1+2")

    #delta_d vs force 80 
    force_80_1 = np.concatenate((mean_force80_FF1, mean_force80_RDG1))
    index_force_1_nan = np.isnan(force_80_1) 
    delta_d_1 = np.concatenate(( mean_delta_d_FF1 , mean_delta_d_RDG1 ))
    index_delta_d_1_nan = np.isnan(delta_d_1)
    indices_force_or_delta_d_1_nan = [index_force_1_nan[i] or index_delta_d_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_delta_d_1_nan[i]])
    delta_d_1_without_nan_force = np.array([delta_d_1[i] for i in range(len(delta_d_1)) if not indices_force_or_delta_d_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, delta_d_1_without_nan_force)
    fitted_response_delta_d_1 = model.predict(force_80_1)
    a_delta_d_1 = reg.coef_
    b_delta_d_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_1 = reg.score(force_80_1, delta_d_1_without_nan_force)
       
    fig_delta_d_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force80_1 = fig_delta_d_vs_force80_1.gca()
    ax_delta_d_vs_force80_1.plot([40, 40], [min(mean_delta_d_FF1) - max(std_delta_d_FF1[~np.isnan(std_delta_d_FF1)]), max(mean_delta_d_RDG1) + max(std_delta_d_RDG1[~np.isnan(std_delta_d_RDG1)]) ], '--r', lw=1, alpha=0.2)
    ax_delta_d_vs_force80_1.errorbar(mean_force80_FF1, mean_delta_d_FF1, yerr=std_delta_d_FF1, xerr=std_force80_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_vs_force80_1.errorbar(mean_force80_RDG1, mean_delta_d_RDG1, yerr=std_delta_d_RDG1, xerr=std_force80_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_delta_d_vs_force80_1.plot(force_80_1, fitted_response_delta_d_1, ':k', label=r' $\Delta d$ = ' + str(np.round(a_delta_d_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_1[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_1, 2)) )
    ax_delta_d_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_vs_force80_1.set_title(r'$\Delta d$ vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_1.set_ylabel(r'$\Delta d$ [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force80_1, "delta_d_vs_force80_1")

    force_80_2 = np.concatenate((mean_force80_FF2, mean_force80_RDG2))
    index_force_2_nan = np.isnan(force_80_2) 
    delta_d_2 = np.concatenate(( mean_delta_d_FF2 , mean_delta_d_RDG2 ))
    index_delta_d_2_nan = np.isnan(delta_d_2)
    indices_force_or_delta_d_2_nan = [index_force_2_nan[i] or index_delta_d_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_delta_d_2_nan[i]])
    delta_d_2_without_nan_force = np.array([delta_d_2[i] for i in range(len(delta_d_2)) if not indices_force_or_delta_d_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, delta_d_2_without_nan_force)
    fitted_response_delta_d_2 = model.predict(force_80_2)
    a_delta_d_2 = reg.coef_
    b_delta_d_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d_2 = reg.score(force_80_2, delta_d_2_without_nan_force)
    
    fig_delta_d_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force80_2 = fig_delta_d_vs_force80_2.gca()
    ax_delta_d_vs_force80_2.plot([40, 40], [min(mean_delta_d_FF2[~np.isnan(mean_delta_d_FF2)]) - max(std_delta_d_FF2[~np.isnan(std_delta_d_FF2)]), max(mean_delta_d_RDG2[~np.isnan(mean_delta_d_RDG2)]) + max(std_delta_d_RDG2[~np.isnan(std_delta_d_RDG2)]) ], '--r', lw=1, alpha=0.2)
    ax_delta_d_vs_force80_2.errorbar(mean_force80_FF2, mean_delta_d_FF2, yerr=std_delta_d_FF2, xerr=std_force80_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_vs_force80_2.errorbar(mean_force80_RDG2, mean_delta_d_RDG2, yerr=std_delta_d_RDG2, xerr=std_force80_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_vs_force80_2.plot(force_80_2, fitted_response_delta_d_2, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_delta_d_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d_2[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d_2, 2)) )
    ax_delta_d_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_vs_force80_2.set_title(r'\Delta d vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80_2.set_ylabel(r'\Delta d [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force80_2, "delta_d_vs_force80_2")

    force_80 = np.concatenate((mean_force80_FF, mean_force80_RDG))
    index_force_nan = np.isnan(force_80) 
    delta_d = np.concatenate(( mean_delta_d_FF , mean_delta_d_RDG ))
    index_delta_d_nan = np.isnan(delta_d)
    indices_force_or_delta_d_nan = [index_force_nan[i] or index_delta_d_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_delta_d_nan[i]])
    delta_d_without_nan_force = np.array([delta_d[i] for i in range(len(delta_d)) if not indices_force_or_delta_d_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, delta_d_without_nan_force)
    fitted_response_delta_d = model.predict(force_80)
    a_delta_d = reg.coef_
    b_delta_d = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_delta_d = reg.score(force_80, delta_d_without_nan_force)
    
    fig_delta_d_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_delta_d_vs_force80 = fig_delta_d_vs_force80.gca()
    ax_delta_d_vs_force80.plot([40, 40], [min(mean_delta_d_FF[~np.isnan(mean_delta_d_FF)]) - max(std_delta_d_FF[~np.isnan(std_delta_d_FF)]), max(mean_delta_d_RDG[~np.isnan(mean_delta_d_RDG)]) + max(std_delta_d_RDG[~np.isnan(std_delta_d_RDG)]) ], '--r', lw=1, alpha=0.2)
    ax_delta_d_vs_force80.errorbar(mean_force80_FF, mean_delta_d_FF, yerr=std_delta_d_FF, xerr=std_force80_FF ,lw=0, label='FF', **kwargs_FF)
    ax_delta_d_vs_force80.errorbar(mean_force80_RDG, mean_delta_d_RDG, yerr=std_delta_d_RDG, xerr=std_force80_RDG ,lw=0, label='RDG', **kwargs_RDG)
    ax_delta_d_vs_force80.plot(force_80, fitted_response_delta_d, ':k', label=r' $\Delta d^*$ = ' + str(np.round(a_delta_d[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_delta_d[0], 4)) + '\n R2 = ' + str(np.round(score_delta_d, 2)) )
    ax_delta_d_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_delta_d_vs_force80.set_title(r'\Delta d vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_vs_force80.set_ylabel(r'\Delta d [mm]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_vs_force80, "delta_d_vs_force80_1+2")

    # #delta_d vs force 20    
    # fig_delta_d_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    # ax_delta_d_vs_force20_1 = fig_delta_d_vs_force20_1.gca()
    # ax_delta_d_vs_force20_1.errorbar(mean_force20_FF1, mean_delta_d_FF1, yerr=std_delta_d_FF1, xerr=std_force20_FF1 ,lw=0, label='FF1', **kwargs_FF1)
    # ax_delta_d_vs_force20_1.errorbar(mean_force20_RDG1, mean_delta_d_RDG1, yerr=std_delta_d_RDG1, xerr=std_force20_RDG1 ,lw=0, label='RDG1', **kwargs_RDG1)
    # ax_delta_d_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_delta_d_vs_force20_1.set_title(r'\Delta d vs Force 20% 1', font=fonts.serif_rz_legend())
    # ax_delta_d_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_delta_d_vs_force20_1.set_ylabel(r'\Delta d [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_delta_d_vs_force20_1, "delta_d_vs_force20_1")

    # fig_delta_d_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    # ax_delta_d_vs_force20_2 = fig_delta_d_vs_force20_2.gca()
    # ax_delta_d_vs_force20_2.errorbar(mean_force20_FF2, mean_delta_d_FF2, yerr=std_delta_d_FF2, xerr=std_force20_FF2 ,lw=0, label='FF2', **kwargs_FF2)
    # ax_delta_d_vs_force20_2.errorbar(mean_force20_RDG2, mean_delta_d_RDG2, yerr=std_delta_d_RDG2, xerr=std_force20_RDG2 ,lw=0, label='RDG2', **kwargs_RDG2)
    # ax_delta_d_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_delta_d_vs_force20_2.set_title(r'\Delta d vs Force 20% 2', font=fonts.serif_rz_legend())
    # ax_delta_d_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_delta_d_vs_force20_2.set_ylabel(r'\Delta d [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_delta_d_vs_force20_2, "delta_d_vs_force20_2")

    # fig_delta_d_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    # ax_delta_d_vs_force20 = fig_delta_d_vs_force20.gca()
    # ax_delta_d_vs_force20.errorbar(mean_force20_FF, mean_delta_d_FF, yerr=std_delta_d_FF, xerr=std_force20_FF ,lw=0, label='FF', **kwargs_FF)
    # ax_delta_d_vs_force20.errorbar(mean_force20_RDG, mean_delta_d_RDG, yerr=std_delta_d_RDG, xerr=std_force20_RDG ,lw=0, label='RDG', **kwargs_RDG)
    # ax_delta_d_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_delta_d_vs_force20.set_title(r'\Delta d vs Force 20% 1+2', font=fonts.serif_rz_legend())
    # ax_delta_d_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    # ax_delta_d_vs_force20.set_ylabel(r'\Delta d [mm]', font=fonts.serif_rz_legend())
    # savefigure.save_as_png(fig_delta_d_vs_force20, "delta_d_vs_force20_1+2")




if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    recovery_pkl_file = 'recoveries_meat.pkl'
    deltad_threshold = 0.99
    # utils.transform_csv_input_A_into_pkl('recoveries_meat_A.csv')
    # ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict = utils.extract_A_data_from_pkl()
    # ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed    = remove_failed_A(ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    # ids_where_not_failed_and_not_small_deltad, date_dict_not_failed_and_not_small_deltad, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad = remove_failed_A_and_small_deltad(deltad_threshold, ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    # compute_and_export_indicators_with_maturation_as_pkl(ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    # plot_recovery_indicators_with_maturation()
    # export_indocators_as_txt()
    # pente à l'origine
    # pente relaxation
    # deltaF*
    
    plot_laser_indicators_vs_texturometer_forces()
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