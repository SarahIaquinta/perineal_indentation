import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.texturometer.figures.utils import CreateFigure, Fonts, SaveFigure
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
import seaborn as sns


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
    ids_at_date_and_meatpiece, force20_dict_at_date_and_meatpiece, force80_dict_at_date_and_meatpiece = extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict)
    mean_force20, std_force20, mean_force80, std_force80 = nan, nan, nan, nan
    if ids_at_date_and_meatpiece != []:
        mean_force20 = statistics.mean(list(force20_dict_at_date_and_meatpiece.values()))
        std_force20 = statistics.stdev(list(force20_dict_at_date_and_meatpiece.values()))
        mean_force80 = statistics.mean(list(force80_dict_at_date_and_meatpiece.values()))
        std_force80 = statistics.stdev(list(force80_dict_at_date_and_meatpiece.values()))
    return mean_force20, std_force20, mean_force80, std_force80




def compute_and_export_forces_with_maturation_as_pkl(ids_list, date_dict, force20_dict, force80_dict):
    dates = list(set(date_dict.values()))
    mean_force20_FF1, std_force20_FF1, mean_force80_FF1, std_force80_FF1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_FF2, std_force20_FF2, mean_force80_FF2, std_force80_FF2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_RDG1, std_force20_RDG1, mean_force80_RDG1, std_force80_RDG1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_RDG2, std_force20_RDG2, mean_force80_RDG2, std_force80_RDG2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_FF, std_force20_FF, mean_force80_FF, std_force80_FF = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_RDG, std_force20_RDG, mean_force80_RDG, std_force80_RDG = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    
    for i in range(len(dates)):
        date = dates[i]
        mean_force20_FF1_date, std_force20_FF1_date, mean_force80_FF1_date, std_force80_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF1', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_FF2_date, std_force20_FF2_date, mean_force80_FF2_date, std_force80_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF2', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_RDG1_date, std_force20_RDG1_date, mean_force80_RDG1_date, std_force80_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG1', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_RDG2_date, std_force20_RDG2_date, mean_force80_RDG2_date, std_force80_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG2', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_FF_date, std_force20_FF_date, mean_force80_FF_date, std_force80_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_RDG_date, std_force20_RDG_date, mean_force80_RDG_date, std_force80_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_FF1[i], std_force20_FF1[i], mean_force80_FF1[i], std_force80_FF1[i] = mean_force20_FF1_date, std_force20_FF1_date, mean_force80_FF1_date, std_force80_FF1_date
        mean_force20_FF2[i], std_force20_FF2[i], mean_force80_FF2[i], std_force80_FF2[i] = mean_force20_FF2_date, std_force20_FF2_date, mean_force80_FF2_date, std_force80_FF2_date
        mean_force20_RDG1[i], std_force20_RDG1[i], mean_force80_RDG1[i], std_force80_RDG1[i] = mean_force20_RDG1_date, std_force20_RDG1_date, mean_force80_RDG1_date, std_force80_RDG1_date
        mean_force20_RDG2[i], std_force20_RDG2[i], mean_force80_RDG2[i], std_force80_RDG2[i] = mean_force20_RDG2_date, std_force20_RDG2_date, mean_force80_RDG2_date, std_force80_RDG2_date
        mean_force20_FF[i], std_force20_FF[i], mean_force80_FF[i], std_force80_FF[i] = mean_force20_FF_date, std_force20_FF_date, mean_force80_FF_date, std_force80_FF_date
        mean_force20_RDG[i], std_force20_RDG[i], mean_force80_RDG[i], std_force80_RDG[i] = mean_force20_RDG_date, std_force20_RDG_date, mean_force80_RDG_date, std_force80_RDG_date

    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [dates, mean_force20_FF1, std_force20_FF1, mean_force80_FF1, std_force80_FF1,
             mean_force20_FF2, std_force20_FF2, mean_force80_FF2, std_force80_FF2,
             mean_force20_RDG1, std_force20_RDG1, mean_force80_RDG1, std_force80_RDG1,
             mean_force20_RDG2, std_force20_RDG2, mean_force80_RDG2, std_force80_RDG2,
             mean_force20_FF, std_force20_FF, mean_force80_FF, std_force80_FF,
             mean_force20_RDG, std_force20_RDG, mean_force80_RDG, std_force80_RDG
             ],
            f,
        )
   
def export_forces_as_txt():
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [date, mean_force20_FF1, std_force20_FF1, mean_force80_FF1, std_force80_FF1,
             mean_force20_FF2, std_force20_FF2, mean_force80_FF2, std_force80_FF2,
             mean_force20_RDG1, std_force20_RDG1, mean_force80_RDG1, std_force80_RDG1,
             mean_force20_RDG2, std_force20_RDG2, mean_force80_RDG2, std_force80_RDG2,
             mean_force20_FF, std_force20_FF, mean_force80_FF, std_force80_FF,
             mean_force20_RDG, std_force20_RDG, mean_force80_RDG, std_force80_RDG
             ] = pickle.load(f)
    
    complete_txt_filename_FF1 = path_to_processed_data + "/forces_mean_std_FF1.txt"
    f = open(complete_txt_filename_FF1, "w")
    f.write("FORCES FOR FF1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_FF1[i])
            + "\t"
            + str(std_force20_FF1[i])
            + "\t"
            + str(mean_force80_FF1[i])
            + "\t"
            + str(std_force80_FF1[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF2 = path_to_processed_data + "/forces_mean_std_FF2.txt"
    f = open(complete_txt_filename_FF2, "w")
    f.write("FORCES FOR FF2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_FF2[i])
            + "\t"
            + str(std_force20_FF2[i])
            + "\t"
            + str(mean_force80_FF2[i])
            + "\t"
            + str(std_force80_FF2[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF = path_to_processed_data + "/forces_mean_std_FF.txt"
    f = open(complete_txt_filename_FF, "w")
    f.write("FORCES FOR FF \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_FF[i])
            + "\t"
            + str(std_force20_FF[i])
            + "\t"
            + str(mean_force80_FF[i])
            + "\t"
            + str(std_force80_FF[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG1 = path_to_processed_data + "/forces_mean_std_RDG1.txt"
    f = open(complete_txt_filename_RDG1, "w")
    f.write("FORCES FOR RDG1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_RDG1[i])
            + "\t"
            + str(std_force20_RDG1[i])
            + "\t"
            + str(mean_force80_RDG1[i])
            + "\t"
            + str(std_force80_RDG1[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG2 = path_to_processed_data + "/forces_mean_std_RDG2.txt"
    f = open(complete_txt_filename_RDG2, "w")
    f.write("FORCES FOR RDG2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_RDG2[i])
            + "\t"
            + str(std_force20_RDG2[i])
            + "\t"
            + str(mean_force80_RDG2[i])
            + "\t"
            + str(std_force80_RDG2[i])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG = path_to_processed_data + "/forces_mean_std_RDG.txt"
    f = open(complete_txt_filename_RDG, "w")
    f.write("FORCES FOR RDG \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_RDG[i])
            + "\t"
            + str(std_force20_RDG[i])
            + "\t"
            + str(mean_force80_RDG[i])
            + "\t"
            + str(std_force80_RDG[i])
            + "\n"
        )
    f.close()



    complete_txt_filename_all = path_to_processed_data + "/forces_mean_std.txt"
    f = open(complete_txt_filename_all, "w")
    f.write("FORCES \n")
    f.write("FF1 \t  FF1 \t  FF1 \t  FF1 \t  FF1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_FF1[i])
            + "\t"
            + str(std_force20_FF1[i])
            + "\t"
            + str(mean_force80_FF1[i])
            + "\t"
            + str(std_force80_FF1[i])
            + "\n"
        )

    f.write("FF2 \t  FF2 \t  FF2 \t  FF2 \t  FF2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_FF2[i])
            + "\t"
            + str(std_force20_FF2[i])
            + "\t"
            + str(mean_force80_FF2[i])
            + "\t"
            + str(std_force80_FF2[i])
            + "\n"
        )

    f.write("FF \t  FF \t  FF \t  FF \t  FF \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_FF[i])
            + "\t"
            + str(std_force20_FF[i])
            + "\t"
            + str(mean_force80_FF[i])
            + "\t"
            + str(std_force80_FF[i])
            + "\n"
        )

    f.write("RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG1)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_RDG1[i])
            + "\t"
            + str(std_force20_RDG1[i])
            + "\t"
            + str(mean_force80_RDG1[i])
            + "\t"
            + str(std_force80_RDG1[i])
            + "\n"
        )

    f.write("RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG2)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_RDG2[i])
            + "\t"
            + str(std_force20_RDG2[i])
            + "\t"
            + str(mean_force80_RDG2[i])
            + "\t"
            + str(std_force80_RDG2[i])
            + "\n"
        )

    f.write("RDG \t  RDG \t  RDG \t  RDG \t  RDG \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG)):
        f.write(
            str(date[i])
            + "\t"
            + str(mean_force20_RDG[i])
            + "\t"
            + str(std_force20_RDG[i])
            + "\t"
            + str(mean_force80_RDG[i])
            + "\t"
            + str(std_force80_RDG[i])
            + "\n"
        )
        
        
    f.close()


def plot_forces_with_maturation():
    maturation = [6, 10, 13, 17, 21]
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [_, mean_force20_FF1, std_force20_FF1, mean_force80_FF1, std_force80_FF1,
             mean_force20_FF2, std_force20_FF2, mean_force80_FF2, std_force80_FF2,
             mean_force20_RDG1, std_force20_RDG1, mean_force80_RDG1, std_force80_RDG1,
             mean_force20_RDG2, std_force20_RDG2, mean_force80_RDG2, std_force80_RDG2,
             mean_force20_FF, std_force20_FF, mean_force80_FF, std_force80_FF,
             mean_force20_RDG, std_force20_RDG, mean_force80_RDG, std_force80_RDG
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
    fig_force20_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force20_1 = fig_force20_1.gca()
    fig_force20_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force20_2 = fig_force20_2.gca()
    fig_force20 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force20 = fig_force20.gca()
    ax_force20.errorbar(maturation_FF, mean_force20_FF, yerr=std_force20_FF, lw=0, label='FF', **kwargs_FF)
    ax_force20_1.errorbar(maturation_FF, mean_force20_FF1, yerr=std_force20_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_force20_2.errorbar(maturation_FF, mean_force20_FF2, yerr=std_force20_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_force20_1.errorbar(maturation_RDG, mean_force20_RDG1, yerr=std_force20_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force20.errorbar(maturation_RDG, mean_force20_RDG, yerr=std_force20_RDG, lw=0,  label='RDG', **kwargs_RDG)
    ax_force20_2.errorbar(maturation_RDG, mean_force20_RDG2, yerr=std_force20_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_force20.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force20_1.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force20_2.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force20.set_title('Force vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_force20_1.set_title('Force vs maturation 1', font=fonts.serif_rz_legend())
    ax_force20_2.set_title('Force vs maturation 2', font=fonts.serif_rz_legend())
    ax_force20.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force20_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force20_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force20.set_ylabel('Force at 20 % [N]', font=fonts.serif_rz_legend())
    ax_force20_1.set_ylabel('Force at 20 % [N]', font=fonts.serif_rz_legend())
    ax_force20_2.set_ylabel('Force at 20 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_force20, "force20_vs_maturation_1+2")
    savefigure.save_as_png(fig_force20_1, "force20_vs_maturation_1")
    savefigure.save_as_png(fig_force20_2, "force20_vs_maturation_2")

    fig_force80_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force80_1 = fig_force80_1.gca()
    fig_force80 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force80 = fig_force80.gca()
    fig_force80_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force80_2 = fig_force80_2.gca()
    ax_force80_1.errorbar(maturation_FF, mean_force80_FF1, yerr=std_force80_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_force80_2.errorbar(maturation_FF, mean_force80_FF2, yerr=std_force80_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_force80.errorbar(maturation_FF, mean_force80_FF, yerr=std_force80_FF, lw=0, label='FF', **kwargs_FF)
    ax_force80_1.errorbar(maturation_RDG, mean_force80_RDG1, yerr=std_force80_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force80_2.errorbar(maturation_RDG, mean_force80_RDG2, yerr=std_force80_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_force80.errorbar(maturation_RDG, mean_force80_RDG, yerr=std_force80_RDG, lw=0, label='RDG', **kwargs_RDG)
    ax_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force80.set_title('Force 80 % vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_force80_1.set_title('Force 80 % vs maturation 1', font=fonts.serif_rz_legend())
    ax_force80_2.set_title('Force 80 % vs maturation 2', font=fonts.serif_rz_legend())
    ax_force80.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force80_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force80_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force80.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    ax_force80_1.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    ax_force80_2.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_force80, "force80_vs_maturation_1+2")
    savefigure.save_as_png(fig_force80_1, "force80_vs_maturation_1")
    savefigure.save_as_png(fig_force80_2, "force80_vs_maturation_2")




if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()

    # ids_list, date_dict, force20_dict, force80_dict, failed_dict = utils.extract_texturometer_data_from_pkl()
    # ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed = remove_failed_data(ids_list, date_dict, force20_dict, force80_dict, failed_dict)
    # mean_force20, std_force20, mean_force80, std_force80 = compute_mean_and_std_at_given_date_and_meatpiece(230327, 'FF', ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed)
    # compute_and_export_forces_with_maturation_as_pkl(ids_list, date_dict, force20_dict, force80_dict)
    # plot_forces_with_maturation()
    export_forces_as_txt()
    print('hello')
    
    
    
