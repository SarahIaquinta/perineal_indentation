import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pickle
import pandas as pd
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from indentation.experiments.laser.post_processing.investigate_A import extract_data_at_given_date_and_meatpiece
from indentation.experiments.zwick.post_processing.utils import find_nearest
import seaborn as sns
import statistics

def get_data_at_given_strain_rate(files_zwick, datafile, sheet, strain_rate):
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_disp = np.max(disp)
    disp_at_strain_rate = find_nearest(disp, max_disp * strain_rate)
    index_where_disp_is_disp_at_strain_rate = np.where(disp == find_nearest(disp, disp_at_strain_rate))[0]
    force_at_strain_rate = force[index_where_disp_is_disp_at_strain_rate]
    time_at_strain_rate = time[index_where_disp_is_disp_at_strain_rate]
    return time_at_strain_rate, force_at_strain_rate, disp_at_strain_rate

def compute_indicators_indentation_at_strain_rate(files_zwick, datafile, sheet, strain_rate=0.25):
    time_at_strain_rate_0, force_at_strain_rate_0, disp_at_strain_rate_0 = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.05)
    time_at_strain_rate, force_at_strain_rate, disp_at_strain_rate = get_data_at_given_strain_rate(files_zwick, datafile, sheet, strain_rate)
    i_disp = (force_at_strain_rate - force_at_strain_rate_0) / (disp_at_strain_rate - disp_at_strain_rate_0)
    i_time = (force_at_strain_rate - force_at_strain_rate_0) / (time_at_strain_rate - time_at_strain_rate_0)
    return i_disp, i_time

def compute_indicators_relaxation(files_zwick, datafile, sheet):
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    index_where_force_is_max = np.where(force == max_force)[0]
    relaxation_slope = (force[index_where_force_is_max + 3] - force[index_where_force_is_max + 1]) / (time[index_where_force_is_max + 3] - time[index_where_force_is_max + 1])
    time_when_force_is_max = time[index_where_force_is_max]
    relaxation_duration = 20
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation))[0]
    delta_f = max_force - force[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    return relaxation_slope, delta_f, delta_f_star

def compute_indicators_indentation_relaxation(files_zwick, datafile, sheet):
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    index_where_force_is_max = np.where(force == max_force)[0]
    relaxation_slope = (force[index_where_force_is_max + 3] - force[index_where_force_is_max + 1]) / (time[index_where_force_is_max + 3] - time[index_where_force_is_max + 1])
    time_when_force_is_max = time[index_where_force_is_max]
    relaxation_duration = 20
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation))[0]
    delta_f = max_force - force[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    time_at_strain_rate_0, force_at_strain_rate_0, disp_at_strain_rate_0 = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.01)
    time_at_strain_rate_1, force_at_strain_rate_1, disp_at_strain_rate_1 = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.9)
    time_at_strain_rate, force_at_strain_rate, disp_at_strain_rate = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.2)
    i_disp_strain_rate = (force_at_strain_rate - force_at_strain_rate_0) / (disp_at_strain_rate - disp_at_strain_rate_0)
    i_time_strain_rate = (force_at_strain_rate - force_at_strain_rate_0) / (time_at_strain_rate - time_at_strain_rate_0)
    i_disp_1 = (force_at_strain_rate_1 - force_at_strain_rate_0) / (disp_at_strain_rate_1 - disp_at_strain_rate_0)
    i_time_1 = (force_at_strain_rate_1 - force_at_strain_rate_0) / (time_at_strain_rate_1 - time_at_strain_rate_0)
    return relaxation_slope, delta_f, delta_f_star, i_disp_strain_rate, i_time_strain_rate, i_disp_1, i_time_1

def plot_indicators_indentation_relaxation(files_zwick, datafile, sheet):
    date, fig_disp_vs_time, fig_force_vs_disp, fig_force_vs_time, ax_disp_vs_time, ax_force_vs_disp, ax_force_vs_time = files_zwick.build_fig_to_plot_data_from_sheet_meat(datafile, sheet, createfigure, savefigure, fonts)
    colors = sns.color_palette("Paired")
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    index_where_force_is_max = np.where(force == max_force)[0]
    relaxation_slope = (force[index_where_force_is_max + 3] - force[index_where_force_is_max + 1]) / (time[index_where_force_is_max + 3] - time[index_where_force_is_max + 1])
    time_when_force_is_max = time[index_where_force_is_max]
    relaxation_duration = 20
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation))[0]
    delta_f = max_force - force[index_where_time_is_end_relaxation]
    delta_f_star = delta_f[0] / max_force
    ax_force_vs_time.plot([time[index_where_force_is_max + 1], time[index_where_force_is_max + 3]], [force[index_where_force_is_max + 1], force[index_where_force_is_max + 3]], '--', color = colors[5], label = r"$\alpha_R$ = " + str(np.round(relaxation_slope[0], 2)) + r" $N.s^{-1}$")
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]], time[index_where_time_is_end_relaxation[0]]], [force[index_where_time_is_end_relaxation][0], max_force], ':', color = colors[-1], label = r"$\Delta F$  = " + str(np.round(delta_f[0], 2)) + " N \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)))

    time_at_strain_rate_0, force_at_strain_rate_0, disp_at_strain_rate_0 = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.01)
    time_at_strain_rate_1, force_at_strain_rate_1, disp_at_strain_rate_1 = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.9)
    time_at_strain_rate, force_at_strain_rate, disp_at_strain_rate = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.2)
    i_disp_strain_rate = (force_at_strain_rate - force_at_strain_rate_0) / (disp_at_strain_rate - disp_at_strain_rate_0)
    i_time_strain_rate = (force_at_strain_rate - force_at_strain_rate_0) / (time_at_strain_rate - time_at_strain_rate_0)
    i_disp_1 = (force_at_strain_rate_1 - force_at_strain_rate_0) / (disp_at_strain_rate_1 - disp_at_strain_rate_0)
    i_time_1 = (force_at_strain_rate_1 - force_at_strain_rate_0) / (time_at_strain_rate_1 - time_at_strain_rate_0)
    ax_force_vs_time.plot([time_at_strain_rate_0, time_at_strain_rate], [force_at_strain_rate_0, force_at_strain_rate], '-.', color = colors[1], label = r"$i_{25\%}$ = " + str(np.round(i_time_strain_rate[0], 2)) + r' $Ns^{-1}$')
    ax_force_vs_time.plot([time_at_strain_rate_0, time_at_strain_rate_1], [force_at_strain_rate_0, force_at_strain_rate_1], ':', color = colors[7], label = r"$i_{100\%}$ = " + str(np.round(i_time_1[0], 2)) + r' $Ns^{-1}$')

    ax_force_vs_disp.plot([disp_at_strain_rate_0, disp_at_strain_rate], [force_at_strain_rate_0, force_at_strain_rate], '-.', color = colors[1], label = r"$i_{25\%}$ = " + str(np.round(i_disp_strain_rate[0], 2)) + r' $Nmm^{-1}$')
    ax_force_vs_disp.plot([disp_at_strain_rate_0, disp_at_strain_rate_1], [force_at_strain_rate_0, force_at_strain_rate_1], ':', color = colors[7], label = r"$i_{100\%}$ = " + str(np.round(i_disp_1[0], 2)) + r' $Nmm^{-1}$')

    ax_force_vs_time.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    savefigure.save_as_png(fig_force_vs_time, sheet + "_force_vs_time_with_indicators")
    ax_force_vs_disp.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig_force_vs_disp, sheet + "_force_vs_disp_with_indicators")
    return relaxation_slope, delta_f, delta_f_star, i_disp_strain_rate, i_time_strain_rate, i_disp_1, i_time_1

def export_indicators(datafile_list):
    ids_list = []
    date_dict = {}
    meat_piece_dict = {}
    relaxation_slope_dict = {}
    delta_f_dict = {}
    delta_f_star_dict = {}
    i_disp_strain_rate_dict = {}
    i_time_strain_rate_dict = {}
    i_disp_1_dict = {}
    i_time_1_dict = {}
    for datafile in datafile_list:
        date = datafile[0:6]
        correct_sheets_in_data = files_zwick.find_only_correct_sheets_in_datafile(datafile)
        for sheet in correct_sheets_in_data:
            id = date + sheet
            relaxation_slope, delta_f, delta_f_star, i_disp_strain_rate, i_time_strain_rate, i_disp_1, i_time_1 = compute_indicators_indentation_relaxation(files_zwick, datafile, sheet)
            ids_list.append(id)
            relaxation_slope_dict[id], delta_f_dict[id], delta_f_star_dict[id], i_disp_strain_rate_dict[id], i_time_strain_rate_dict[id], i_disp_1_dict[id], i_time_1_dict[id] = relaxation_slope, delta_f, delta_f_star, i_disp_strain_rate, i_time_strain_rate, i_disp_1, i_time_1  
            date_dict[id] = date
            ind = [i for i in range(len(id)) if id[i].isalpha()]
            meat_piece = id[ind[0]:]
            meat_piece_dict[id] = meat_piece
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indicators_indentation_relaxation.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [ids_list, date_dict, meat_piece_dict, relaxation_slope_dict, delta_f_dict, delta_f_star_dict, i_disp_strain_rate_dict, i_time_strain_rate_dict, i_disp_1_dict, i_time_1_dict      ],
            f,
        )
    return ids_list, date_dict, meat_piece_dict, relaxation_slope_dict, delta_f_dict, delta_f_star_dict, i_disp_strain_rate_dict, i_time_strain_rate_dict, i_disp_1_dict, i_time_1_dict      
        
# def extract_data



def extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, meat_piece_dict, date_dict, data_dict):
    ids_at_date = [id for id in ids_list if date_dict[id] == date]
    index_limit = len(meatpiece)
    ids_at_date_and_meatpiece = [id for id in ids_at_date if meat_piece_dict[id][0:index_limit] == meatpiece[0:index_limit]]
    if len(data_dict[ids_at_date[0]]) > 0: 
        data_dict_at_date_and_meatpiece = {id: data_dict[id][0] for id in ids_at_date_and_meatpiece}
    else:
        data_dict_at_date_and_meatpiece = {id: data_dict[id] for id in ids_at_date_and_meatpiece}
    return ids_at_date_and_meatpiece, data_dict_at_date_and_meatpiece      
        

def compute_mean_and_std_at_given_date_and_meatpiece(date, meatpiece, ids_list, meat_piece_dict, date_dict, data_dict):
    ids_at_date_and_meatpiece, data_dict_at_date_and_meatpiece = extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, meat_piece_dict, date_dict, data_dict)
    mean_data, std_data = nan, nan
    if len(ids_at_date_and_meatpiece) >1:
        mean_data = statistics.mean(list(data_dict_at_date_and_meatpiece.values()))
        std_data = statistics.stdev(list(data_dict_at_date_and_meatpiece.values()))
    return mean_data, std_data


def compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, data_dict, indicator):
    dates = list(set(date_dict.values()))
    mean_data_FF1, std_data_FF1 = np.zeros((len(dates))), np.zeros((len(dates)))
    mean_data_FF2, std_data_FF2 = np.zeros((len(dates))), np.zeros((len(dates)))
    mean_data_RDG1, std_data_RDG1 = np.zeros((len(dates))), np.zeros((len(dates)))
    mean_data_RDG2, std_data_RDG2 = np.zeros((len(dates))), np.zeros((len(dates)))
    mean_data_FF, std_data_FF = np.zeros((len(dates))), np.zeros((len(dates)))
    mean_data_RDG, std_data_RDG = np.zeros((len(dates))), np.zeros((len(dates)))
    
    for i in range(len(dates)):
        date = dates[i]
        mean_data_FF1_date, std_data_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF1', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_FF2_date, std_data_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF2', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_RDG1_date, std_data_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG1', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_RDG2_date, std_data_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG2', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_FF_date, std_data_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_RDG_date, std_data_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_FF1[i], std_data_FF1[i] = mean_data_FF1_date, std_data_FF1_date
        mean_data_FF2[i], std_data_FF2[i] = mean_data_FF2_date, std_data_FF2_date
        mean_data_RDG1[i], std_data_RDG1[i] = mean_data_RDG1_date, std_data_RDG1_date
        mean_data_RDG2[i], std_data_RDG2[i] = mean_data_RDG2_date, std_data_RDG2_date
        mean_data_FF[i], std_data_FF[i] = mean_data_FF_date, std_data_FF_date
        mean_data_RDG[i], std_data_RDG[i] = mean_data_RDG_date, std_data_RDG_date
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + indicator + ".pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [dates, mean_data_FF1, std_data_FF1,
             mean_data_FF2, std_data_FF2,
             mean_data_RDG1, std_data_RDG1,
             mean_data_RDG2, std_data_RDG2,
             mean_data_FF, std_data_FF,
             mean_data_RDG, std_data_RDG
             ],
            f,
        )
 
 

def export_data_as_txt(indicator):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [date, mean_data_FF1, std_data_FF1,
             mean_data_FF2, std_data_FF2,
             mean_data_RDG1, std_data_RDG1,
             mean_data_RDG2, std_data_RDG2,
             mean_data_FF, std_data_FF,
             mean_data_RDG, std_data_RDG
             ] = pickle.load(f)
    
    complete_txt_filename_FF1 = path_to_processed_data + "/" + indicator + "_mean_std_FF1.txt"
    f = open(complete_txt_filename_FF1, "w")
    f.write(indicator + "FOR FF1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_FF1[i])
            + "\t"
            + str(std_data_FF1[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF2 = path_to_processed_data + "/" + indicator + "_mean_std_FF2.txt"
    f = open(complete_txt_filename_FF2, "w")
    f.write(indicator +  "FOR FF2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_FF2[i])
            + "\t"
            + str(std_data_FF2[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF = path_to_processed_data + "/" + indicator + "_mean_std_FF.txt"
    f = open(complete_txt_filename_FF, "w")
    f.write(indicator +  "FOR FF \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_FF[i])
            + "\t"
            + str(std_data_FF[i])            + "\n"
        )
    f.close()

    complete_txt_filename_RDG1 = path_to_processed_data + "/" + indicator + "_mean_std_RDG1.txt"
    f = open(complete_txt_filename_RDG1, "w")
    f.write(indicator +  "FOR RDG1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_RDG1[i])
            + "\t"
            + str(std_data_RDG1[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG2 = path_to_processed_data + "/" + indicator + "_mean_std_RDG2.txt"
    f = open(complete_txt_filename_RDG2, "w")
    f.write(indicator +  "FOR RDG2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_RDG2[i])
            + "\t"
            + str(std_data_RDG2[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG = path_to_processed_data + "/" + indicator + "_mean_std_RDG.txt"
    f = open(complete_txt_filename_RDG, "w")
    f.write(indicator +  "FOR RDG \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_RDG[i])
            + "\t"
            + str(std_data_RDG[i])
            + "\n"
        )
    f.close()



    complete_txt_filename_all = path_to_processed_data + "/" + indicator + "_mean_std.txt"
    f = open(complete_txt_filename_all, "w")
    f.write(indicator +  "\n")
    f.write("FF1 \t  FF1 \t  FF1 \t  FF1 \t  FF1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_FF1[i])
            + "\t"
            + str(std_data_FF1[i])
            + "\n"
        )

    f.write("FF2 \t  FF2 \t  FF2 \t  FF2 \t  FF2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_FF2[i])
            + "\t"
            + str(std_data_FF2[i])
            + "\n"
        )

    f.write("FF \t  FF \t  FF \t  FF \t  FF \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_FF[i])
            + "\t"
            + str(std_data_FF[i])            + "\n"
        )

    f.write("RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_RDG1[i])
            + "\t"
            + str(std_data_RDG1[i])
            + "\n"
        )

    f.write("RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_RDG2[i])
            + "\t"
            + str(std_data_RDG2[i])
            + "\n"
        )

    f.write("RDG \t  RDG \t  RDG \t  RDG \t  RDG \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_data_RDG[i])
            + "\t"
            + str(std_data_RDG[i])
            + "\n"
        )
        
        
    f.close()



def plot_data_with_maturation(indicator):
    maturation = [13, 17, 21]
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [date, mean_data_FF1, std_data_FF1,
             mean_data_FF2, std_data_FF2,
             mean_data_RDG1, std_data_RDG1,
             mean_data_RDG2, std_data_RDG2,
             mean_data_FF, std_data_FF,
             mean_data_RDG, std_data_RDG
             ] = pickle.load(f)
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

    labels = {'relaxation_slope' : r"$\alpha_R$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_strain_rate': r"$\i_{25 \%} $ [Nm^{-1}]",
                    'i_time_strain_rate': r"$\i_{25 \%} $ [Ns^{-1}]",
                    'i_disp_1': r"$\i_{100 \%} $ [Nm^{-1}]",
                    'i_time_1': r"$\i_{100 \%} $ [Ns^{-1}]"   }
    fig_data_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_data_1 = fig_data_1.gca()
    fig_data = createfigure.rectangle_rz_figure(pixels=180)
    ax_data = fig_data.gca()
    fig_data_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_data_2 = fig_data_2.gca()
    ax_data_1.errorbar(maturation_FF, mean_data_FF1, yerr=std_data_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_data_2.errorbar(maturation_FF, mean_data_FF2, yerr=std_data_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_data.errorbar(maturation_FF, mean_data_FF, yerr=std_data_FF, lw=0, label='FF', **kwargs_FF)
    ax_data_1.errorbar(maturation_RDG, mean_data_RDG1, yerr=std_data_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_data_2.errorbar(maturation_RDG, mean_data_RDG2, yerr=std_data_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_data.errorbar(maturation_RDG, mean_data_RDG, yerr=std_data_RDG, lw=0, label='RDG', **kwargs_RDG)
    ax_data_1.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_data_2.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_data.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_data.set_title(labels[indicator] + ' vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_data_1.set_title(labels[indicator] + ' vs maturation 1', font=fonts.serif_rz_legend())
    ax_data_2.set_title(labels[indicator] + ' vs maturation 2', font=fonts.serif_rz_legend())
    ax_data.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_data_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_data_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_data.set_ylabel(labels[indicator], font=fonts.serif_rz_legend())
    ax_data_1.set_ylabel(labels[indicator], font=fonts.serif_rz_legend())
    ax_data_2.set_ylabel(labels[indicator], font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_data, indicator + "_vs_maturation_1+2")
    savefigure.save_as_png(fig_data_1, indicator + "_vs_maturation_1")
    savefigure.save_as_png(fig_data_2, indicator + "_vs_maturation_2")



        
        
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230407', '230411', '230403']
    types_of_essay = ['C_Indentation_relaxation_500N_force.xlsx']#,'C_Indentation_relaxation_maintienFnulle_500N_trav.xls',  'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = []
    for i in range(len(experiment_dates)):
        datafile_list += files_zwick.import_files(experiment_dates[i])
    # datafile = datafile_list[0]
    # datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    # correct_sheets_in_data = files_zwick.find_only_correct_sheets_in_datafile(datafile)
    # for sheet in correct_sheets_in_data:
    # # sheet1 = correct_sheets_in_data[0]
    # # i_disp, i_time = compute_indicators_indentation_at_strain_rate(files_zwick, datafile, sheet1, strain_rate=0.25)
    #     relaxation_slope, delta_f, delta_f_star, i_disp_strain_rate, i_time_strain_rate, i_disp_1, i_time_1 = plot_indicators_indentation_relaxation(files_zwick, datafile, sheet)
    #     # pente à l'origine
    # export_indicators(datafile_list)
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indicators_indentation_relaxation.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [ids_list, date_dict, meat_piece_dict, relaxation_slope_dict, delta_f_dict, delta_f_star_dict, i_disp_strain_rate_dict, i_time_strain_rate_dict, i_disp_1_dict, i_time_1_dict] = pickle.load(f)

    indicator_list = ['relaxation_slope',
                    'delta_f',
                    'delta_f_star',
                    'i_disp_strain_rate',
                    'i_time_strain_rate',
                    'i_disp_1',
                    'i_time_1'       
                        ]
    # ids_at_date_and_meatpiece, data_dict_at_date_and_meatpiece  = extract_data_at_given_date_and_meatpiece('230331', 'RDG1', ids_list, meat_piece_dict, date_dict, relaxation_slope_dict)
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, relaxation_slope_dict, 'relaxation_slope')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, delta_f_dict, 'delta_f')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, delta_f_star_dict, 'delta_f_star')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, i_disp_strain_rate_dict, 'i_disp_strain_rate')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, i_time_strain_rate_dict, 'i_time_strain_rate')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, i_disp_1_dict, 'i_disp_1')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, i_time_1_dict, 'i_time_1')
    for indicator in indicator_list:
        # export_data_as_txt(indicator)
        plot_data_with_maturation(indicator)

    print('hello')