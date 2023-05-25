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
import statistics



# def transform_recovery_data_in_dictionnaries(recovery_pkl_file):
#     filenames_from_pkl, delta_d_from_pkl, delta_d_stars_from_pkl, d_min_from_pkl, A_from_pkl = utils.extract_data_from_recovery_pkl_file(recovery_pkl_file)
#     dates_dict = dict(zip(filenames_from_pkl, [f[0:6] for f in filenames_from_pkl]))
#     piece_dict = dict(zip(filenames_from_pkl, [f[7:9] for f in filenames_from_pkl]))
#     maturation_dict = {'230327': 6, '230331': 10, '230403': 13, '230407': 17, '230411':21}
#     delta_d_dict = dict(zip(filenames_from_pkl, delta_d_from_pkl))
#     delta_d_star_dict = dict(zip(filenames_from_pkl, delta_d_stars_from_pkl))
#     d_min_dict = dict(zip(filenames_from_pkl, d_min_from_pkl))
#     A_dict = dict(zip(filenames_from_pkl, A_from_pkl))
#     return filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict


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
    if len(ids_at_date_and_meatpiece) >1:
        mean_delta_d = statistics.mean(list(delta_d_dict_at_date_and_meatpiece.values()))
        std_delta_d = statistics.stdev(list(delta_d_dict_at_date_and_meatpiece.values()))
        mean_delta_d_star = statistics.mean(list(delta_d_star_dict_at_date_and_meatpiece.values()))
        std_delta_d_star = statistics.stdev(list(delta_d_star_dict_at_date_and_meatpiece.values()))
        mean_d_min = statistics.mean(list(d_min_dict_at_date_and_meatpiece.values()))
        std_d_min = statistics.stdev(list(d_min_dict_at_date_and_meatpiece.values()))
        mean_A = statistics.mean(list(A_dict_at_date_and_meatpiece.values()))
        std_A = statistics.stdev(list(A_dict_at_date_and_meatpiece.values()))
    return mean_delta_d, std_delta_d, mean_delta_d_star, std_delta_d_star, mean_d_min, std_d_min, mean_A, std_A

def plot_recovery_indicators_with_maturation(ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions):
    dates = list(set(date_dict.values()))
    maturation = [6, 10, 13, 17, 21]
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

    color = sns.color_palette("Paired")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    
    for i in range(len(dates)):
        date = dates[i]
        mean_delta_d_FF1_date, std_delta_d_FF1_date, mean_delta_d_star_FF1_date, std_delta_d_star_FF1_date, mean_d_min_FF1_date, std_d_min_FF1_date,  mean_A_FF1_date, std_A_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF1', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        #TODO add missing outputs in the next lines, following line 101
        mean_delta_d_FF2_date, std_delta_d_FF2_date, mean_delta_d_star_FF2_date, std_delta_d_star_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF2', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG1_date, std_delta_d_RDG1_date, mean_delta_d_star_RDG1_date, std_delta_d_star_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG1', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG2_date, std_delta_d_RDG2_date, mean_delta_d_star_RDG2_date, std_delta_d_star_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG2', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF_date, std_delta_d_FF_date, mean_delta_d_star_FF_date, std_delta_d_star_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_RDG_date, std_delta_d_RDG_date, mean_delta_d_star_RDG_date, std_delta_d_star_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG', ids_list, date_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, failed_A_acqusitions)
        mean_delta_d_FF1[i], std_delta_d_FF1[i], mean_delta_d_star_FF1[i], std_delta_d_star_FF1[i] = mean_delta_d_FF1_date, std_delta_d_FF1_date, mean_delta_d_star_FF1_date, std_delta_d_star_FF1_date
        mean_delta_d_FF2[i], std_delta_d_FF2[i], mean_delta_d_star_FF2[i], std_delta_d_star_FF2[i] = mean_delta_d_FF2_date, std_delta_d_FF2_date, mean_delta_d_star_FF2_date, std_delta_d_star_FF2_date
        mean_delta_d_RDG1[i], std_delta_d_RDG1[i], mean_delta_d_star_RDG1[i], std_delta_d_star_RDG1[i] = mean_delta_d_RDG1_date, std_delta_d_RDG1_date, mean_delta_d_star_RDG1_date, std_delta_d_star_RDG1_date
        mean_delta_d_RDG2[i], std_delta_d_RDG2[i], mean_delta_d_star_RDG2[i], std_delta_d_star_RDG2[i] = mean_delta_d_RDG2_date, std_delta_d_RDG2_date, mean_delta_d_star_RDG2_date, std_delta_d_star_RDG2_date
        mean_delta_d_FF[i], std_delta_d_FF[i], mean_delta_d_star_FF[i], std_delta_d_star_FF[i] = mean_delta_d_FF_date, std_delta_d_FF_date, mean_delta_d_star_FF_date, std_delta_d_star_FF_date
        mean_delta_d_RDG[i], std_delta_d_RDG[i], mean_delta_d_star_RDG[i], std_delta_d_star_RDG[i] = mean_delta_d_RDG_date, std_delta_d_RDG_date, mean_delta_d_star_RDG_date, std_delta_d_star_RDG_date
    fig_delta_d_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_1 = fig_delta_d_1.gca()
    fig_delta_d_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_2 = fig_delta_d_2.gca()
    fig_delta_d = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d = fig_delta_d.gca()
    ax_delta_d.errorbar(maturation, mean_delta_d_FF1, yerr=std_delta_d_FF1, lw=0, label='FF', **kwargs_FF2)
    ax_delta_d_1.errorbar(maturation, mean_delta_d_FF1, yerr=std_delta_d_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_2.errorbar(maturation, mean_delta_d_FF2, yerr=std_delta_d_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_1.errorbar(maturation, mean_delta_d_RDG1, yerr=std_delta_d_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_delta_d.errorbar(maturation, mean_delta_d_RDG1, yerr=std_delta_d_RDG1, lw=0,  label='RDG', **kwargs_RDG1)
    ax_delta_d_2.errorbar(maturation, mean_delta_d_RDG2, yerr=std_delta_d_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_delta_d_1.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_delta_d_2.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_delta_d.set_title('Force vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_delta_d_1.set_title('Force vs maturation 1', font=fonts.serif_rz_legend())
    ax_delta_d_2.set_title('Force vs maturation 2', font=fonts.serif_rz_legend())
    ax_delta_d.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d.set_ylabel('Force at 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_1.set_ylabel('Force at 20 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_2.set_ylabel('Force at 20 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d, "delta_d_vs_maturation_1+2")
    savefigure.save_as_png(fig_delta_d_1, "delta_d_vs_maturation_1")
    savefigure.save_as_png(fig_delta_d_2, "delta_d_vs_maturation_2")

    fig_delta_d_star_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star_1 = fig_delta_d_star_1.gca()
    fig_delta_d_star_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_delta_d_star_2 = fig_delta_d_star_2.gca()
    ax_delta_d_star_1.errorbar(maturation, mean_delta_d_star_FF1, yerr=std_delta_d_star_FF1, lw=0, label='FF1', **kwargs_FF1)
    ax_delta_d_star_2.errorbar(maturation, mean_delta_d_star_FF2, yerr=std_delta_d_star_FF2, lw=0, label='FF2', **kwargs_FF2)
    ax_delta_d_star_1.errorbar(maturation, mean_delta_d_star_RDG1, yerr=std_delta_d_star_RDG1, lw=0,  label='RDG1', **kwargs_RDG1)
    ax_delta_d_star_2.errorbar(maturation, mean_delta_d_star_RDG2, yerr=std_delta_d_star_RDG2, lw=0, label='RDG2', **kwargs_RDG2)
    ax_delta_d_star_1.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_delta_d_star_2.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    ax_delta_d_star_1.set_title('Force vs maturation 1', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_title('Force vs maturation 2', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_delta_d_star_1.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    ax_delta_d_star_2.set_ylabel('Force at 80 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_delta_d_star_1, "delta_d_star_vs_maturation_1")
    savefigure.save_as_png(fig_delta_d_star_2, "delta_d_star_vs_maturation_2")





def plot_variables_with_date(filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, createfigure, savefigure, fonts):
    colors_with_piece = {'FF': 'r', 'RD': 'b'}
    alpha_with_maturation = {6: 0.25, 10: 0.4, 13: 0.6, 17:0.7, 21: 0.85}
    
    fig = createfigure.rectangle_rz_figure(pixels=180)
    ax = fig.gca()
    
    dates = [dates_dict[key] for key in dates_dict.keys()]
    pieces = [piece_dict[key] for key in piece_dict.keys()]

    for date in dates:
        for piece in pieces:
            maturation_list = []
            A_list = []
            id = date + '_' + piece
            for filename in filenames_from_pkl:
                if dates_dict[filename] == date:
                    if piece_dict[filename] == piece:
                        maturation = maturation_dict[date]
                        maturation_list.append(maturation)
                        A_list.append(A_dict[filename])
            ax.plot(maturation_list, A_list, marker="o",  markerfacecolor = colors_with_piece[piece[0:2]], markeredgecolor=None, alpha = alpha_with_maturation[maturation], linewidths=None)
    savefigure.save_as_png(fig, "A_with_maturation_meat")
                    
    
    


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    recovery_pkl_file = 'recoveries_meat.pkl'
    # filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict = transform_recovery_data_in_dictionnaries(recovery_pkl_file)
    # print('hello')
    # plot_variables_with_date(filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, createfigure, savefigure, fonts)
    deltad_threshold = 0.99
    # failed_A_acqusitions = ['230403_FF1B',
    #                         '230403_FF1D',
    #                         '230403_RDG1B',
    #                         '230403_RDG1E',
    #                         '230403_RDG1F',
    #                         '230403_RDG2A',
    #                         '230331_RDG2_1F',
    #                         '230331_RDG1_1E',
    #                         '230411_FF2_ENTIER1',
    #                         '230515_P002-1',
    #                         '230331_FF2_1E',
    #                         '230331_RDG1_ENTIER1'
    #                         '230327_FF1_1A',
    #                         '230327_FF1_2A',
    #                         '230331_FF1_2B_(2)',
    #                         '230331_FF1_2E',
    #                         '230331_FF2_1B_(2)',
    #                         '230331_FF2_1C',
    #                         '230331_FF2_2B_(2)',
    #                         '230331_FF2_2C',
    #                         '230331_RDG1_1A',
    #                         '230331_RDG1_2D',
    #                         '230331_RDG2_1D',
    #                         '230331_RDG2_2A_(2)',
    #                         '230331_RDG2_2B_(2)',
    #                         '230331_RDG2_2C',
    #                         '230331_RDG2_2E',
    #                         '230403_FF1F',
    #                         '230403_FF1G',
    #                         '230403_FF2A',
    #                         '230403_FF2B',
    #                         '230403_FF2C',
    #                         '230403_FF2D',
    #                         '230403_FF2E',
    #                         '230403_FF2F',
    #                         '230403_FF2G',
    #                         '230403_RDG2E',
    #                         '230407_FF1E',
    #                         '230407_FF2A',
    #                         '230407_FF2C',
    #                         '230407_FF2D',
    #                         '230407_FF2E',
    #                         '230407_FF2F',
    #                         '230407_RDG2A',
    #                         '230407_RDG2C',
    #                         '230407_RDG2E',
    #                         '230411_FF1B',
    #                         '230411_FF2D',
    #                         '230411_FF2E',
    #                         '230411_FF2F',
    #                         '230411_RDG1A',
    #                         '230411_RDG2A',
    #                         '230411_RDG2C',
    #                         '230411_RDG2E',
    #                         '230407_FF1A'
    #                         ]
    utils.transform_csv_input_A_into_pkl('recoveries_meat_A.csv')
    ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict = utils.extract_A_data_from_pkl()
    # ids_where_not_failed, date_dict_not_failed, delta_d_dict_not_failed, delta_d_star_dict_not_failed, d_min_dict_not_failed, A_dict_not_failed    = remove_failed_A(ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    # ids_where_not_failed_and_not_small_deltad, date_dict_not_failed_and_not_small_deltad, delta_d_dict_not_failed_and_not_small_deltad, delta_d_star_dict_not_failed_and_not_small_deltad, d_min_dict_not_failed_and_not_small_deltad, A_dict_not_failed_and_not_small_deltad = remove_failed_A_and_small_deltad(deltad_threshold, ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    plot_recovery_indicators_with_maturation(ids_list, date_dict, deltad_dict, deltadstar_dict, dmin_dict, A_dict, failed_dict)
    print('hello')