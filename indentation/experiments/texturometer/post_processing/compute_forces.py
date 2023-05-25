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

def plot_forces_with_maturation(ids_list, date_dict, force20_dict, force80_dict):
    dates = list(set(date_dict.values()))
    maturation = [6, 10, 13, 17, 21]
    mean_force20_FF1, std_force20_FF1, mean_force80_FF1, std_force80_FF1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_FF2, std_force20_FF2, mean_force80_FF2, std_force80_FF2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_RDG1, std_force20_RDG1, mean_force80_RDG1, std_force80_RDG1 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_RDG2, std_force20_RDG2, mean_force80_RDG2, std_force80_RDG2 = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_FF, std_force20_FF, mean_force80_FF, std_force80_FF = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    mean_force20_RDG, std_force20_RDG, mean_force80_RDG, std_force80_RDG = np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates))), np.zeros((len(dates)))
    color = sns.color_palette("Paired")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    
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
    fig_force20_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force20_1 = fig_force20_1.gca()
    fig_force20_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force20_2 = fig_force20_2.gca()
    fig_force20 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force20 = fig_force20.gca()
    ax_force20.errorbar(maturation, mean_force20_FF1, yerr=std_force20_FF1, lw=0, label='FF', **kwargs_FF2)
    ax_force20_1.errorbar(maturation, mean_force20_FF1, yerr=std_force20_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_force20_2.errorbar(maturation, mean_force20_FF2, yerr=std_force20_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_force20_1.errorbar(maturation, mean_force20_RDG1, yerr=std_force20_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force20.errorbar(maturation, mean_force20_RDG1, yerr=std_force20_RDG1, lw=0,  label='RDG', **kwargs_RDG1)
    ax_force20_2.errorbar(maturation, mean_force20_RDG2, yerr=std_force20_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
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
    fig_force80_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force80_2 = fig_force80_2.gca()
    ax_force80_1.errorbar(maturation, mean_force80_FF1, yerr=std_force80_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_force80_2.errorbar(maturation, mean_force80_FF2, yerr=std_force80_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_force80_1.errorbar(maturation, mean_force80_RDG1, yerr=std_force80_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force80_2.errorbar(maturation, mean_force80_RDG2, yerr=std_force80_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force80_1.set_title('Force vs maturation 1', font=fonts.serif_rz_legend())
    ax_force80_2.set_title('Force vs maturation 2', font=fonts.serif_rz_legend())
    ax_force80_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force80_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force80_1.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    ax_force80_2.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_force80_1, "force80_vs_maturation_1")
    savefigure.save_as_png(fig_force80_2, "force80_vs_maturation_2")

    fig_force80 = createfigure.rectangle_rz_figure(pixels=180)
    ax_force80 = fig_force80.gca()
    ax_force80.errorbar(maturation, mean_force80_FF1, yerr=std_force80_FF1, lw=0, label='FF', **kwargs_FF2)
    ax_force80.errorbar(maturation, mean_force80_RDG1, yerr=std_force80_RDG1, lw=0,  label='RDG', **kwargs_RDG2)
    ax_force80.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_force80.set_title('Force vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_force80.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force80.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_force80, "force80_vs_maturation_1+2")



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()

    ids_list, date_dict, force20_dict, force80_dict, failed_dict = utils.extract_texturometer_data_from_pkl()
    ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed = remove_failed_data(ids_list, date_dict, force20_dict, force80_dict, failed_dict)
    mean_force20, std_force20, mean_force80, std_force80 = compute_mean_and_std_at_given_date_and_meatpiece(230327, 'FF', ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed)
    plot_forces_with_maturation(ids_list, date_dict, force20_dict, force80_dict)
    print('hello')
    
    
    
