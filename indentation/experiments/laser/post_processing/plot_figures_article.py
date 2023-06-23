""""
This file is used to generate nice looking figures for illustrating articles.
"""

import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
from indentation.experiments.laser.post_processing.read_file import Files
from indentation.experiments.laser.post_processing.identify_movement import Recovery
from sklearn.linear_model import LinearRegression
import seaborn as sns 
import pickle

def plot_recovery(locations,  createfigure, savefigure, fonts):
    experiment_date = '0_locations_230403'
    meat_piece = '_FF2A.csv'
    # files_meat_piece = Files(meat_piece)
    # list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
    filename =  experiment_date + meat_piece
    recovery_at_date_meat_piece = Recovery(filename, locations)
    # recovery_at_date_meat_piece.plot_recovery(10, createfigure, savefigure, fonts)
    n_smooth = 10
    vec_time_wo_outliers, recovery_positions_wo_outliers_unsmoothed = recovery_at_date_meat_piece.compute_recovery_with_time(n_smooth)
    # recovery_positions = lfilter([1/1]*1, 1, recovery_positions_wo_outliers_unsmoothed)
    nn = 200
    # recovery_positions = recovery_positions_wo_outliers_unsmoothed
    recovery_positions = np.convolve(recovery_positions_wo_outliers_unsmoothed,np.ones(nn)/nn,'same')
    recovery_positions[0:int(nn/2)+1] = recovery_positions[int(nn/2)+2]
    recovery_positions[-int(nn/2)+1:] = recovery_positions[-int(nn/2)+2]
    
    
    
    # A, fitted_response, log_time = recovery_at_date_meat_piece.compute_A(n_smooth)
    fig = createfigure.rectangle_figure(pixels=180)
    fig_log = createfigure.rectangle_figure(pixels=180)
    ax = fig.gca()
    ax_log = fig_log.gca()
    kwargs = {"linewidth": 3}
    index_recovery_position_is_min, min_recovery_position, last_recovery, delta_d, delta_d_star = recovery_at_date_meat_piece.compute_delta_d_star(n_smooth)
    recovery_time_at_beginning = vec_time_wo_outliers[index_recovery_position_is_min] /1e6
    # ax.plot(vec_time_wo_outliers[1:]/1e6, recovery_positions[:-1], '-k', label = recovery_at_date_meat_piece.filename[0:-4], **kwargs)
    vec_time_wo_outliers = vec_time_wo_outliers - vec_time_wo_outliers[0] + 0.1
    
    recovery_position_at_beginning = recovery_positions[index_recovery_position_is_min]
    recovery_time_at_end = vec_time_wo_outliers[-2]/1e6
    recovery_position_at_end = last_recovery
    
    log_time_unshaped = np.array([np.log((t+0.01)/1e6) for t in vec_time_wo_outliers])
    log_time = log_time_unshaped.reshape((-1, 1))
    model = LinearRegression()
    model.fit(log_time[1:], recovery_positions[1:])
    fitted_response = model.predict(log_time)
    A = model.coef_
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket", as_cmap=False)    
    ax.plot(vec_time_wo_outliers[index_recovery_position_is_min+60:-502]/1e6, recovery_positions[index_recovery_position_is_min+60:-502], '-k', **kwargs)
    ax.plot(np.exp(log_time)[+185:-502], fitted_response[+185:-502] ,   '--', color = color_rocket[4], label =r"$z = A \log{t} + z_{t=1}$" + "\nA = " + str(np.round(A[0], 2)) , **kwargs)
    # ax.plot([recovery_time_at_beginning], [recovery_position_at_beginning], label = 'beginning', marker="*", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax.plot([recovery_time_at_end], [recovery_position_at_end], label = 'end', marker="o", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax.text(str(delta_d_star))
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif(), fontsize=26)
    # ax_log.plot(vec_time_wo_outliers[index_recovery_position_is_min+60:-502]/1e6, recovery_positions[index_recovery_position_is_min+60:-502], '-k', **kwargs)
    # ax_log.plot(np.exp(log_time)[+60:-502], fitted_response[+60:-502], ':r', label='A = ' + str(np.round(A[0], 2)), **kwargs)
    # ax_log.plot([recovery_time_at_beginning], [recovery_position_at_beginning], label = 'beginning', marker="*", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax_log.plot([recovery_time_at_end], [recovery_position_at_end], label = 'end', marker="o", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax.text(str(delta_d_star))
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif(), fontsize=26)
    # ax_log.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif(), fontsize=26)
    # ax_log.set_xscale('log')
    # ax_log.set_xlim(1, 31)
    # ax.set_ylim(-5.5, -3.5)
    # ax.set_xticks([1, 10, 100])
    # ax.set_xticklabels([1, 10, 100],  font=fonts.serif(), fontsize=24)
    # ax.set_yticks([-6, -5, -4])
    # ax.set_yticklabels([-6, -5, -4],  font=fonts.serif(), fontsize=24)
    
    # ax.arrow(30.5, -6.3, 0, 1.1, head_width=0.5, head_length=0.05, color='b', length_includes_head=True)
    ax.annotate(r"$\Delta$d", (28, -5.8), font=fonts.serif(), fontsize=26, color = color_rocket[2]  )

    ax.annotate("", xy=(30.5, -5.2), xytext=(30.5, -6.3),
            arrowprops=dict(arrowstyle="<->", linewidth=2, color = color_rocket[2] ))
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels([0, 10, 20, 30], font=fonts.serif(), fontsize=24)
    ax.set_yticks([-6.2, -6, -5.8, -5.6, -5.4, -5.2])
    ax.set_yticklabels([-6.2, -6, -5.8, -5.6, -5.4, -5.2], font=fonts.serif(), fontsize=24)
    ax.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    # ax_log.set_xlabel(r"$log(time) $ [-]", font=fonts.serif(), fontsize=26)
    ax.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
    ax_log.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
    ax.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    # ax_log.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig, "article_recovery_"  + experiment_date + meat_piece[0:5])
    # savefigure.save_as_png(fig_log, "article_recovery_logx_" + experiment_date + meat_piece[0:5])
    plt.close(fig)
    plt.close(fig_log)
        

def plot_A_vs_texturometer_forces(deltad_threshold, strain_threshold, createfigure, savefigure, fonts):
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
    complete_pkl_filename_laser = path_to_processed_data_laser + "/0_locations_deltad_threshold" + str(deltad_threshold) + "strain_treshold" + str(strain_threshold) + "_indicators_mean_std.pkl"
    with open(complete_pkl_filename_laser, "rb") as f:
        [dates_laser, mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
             mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
             mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
             mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
             mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
             mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict,
             mean_Umax_FF1_dict, std_Umax_FF1_dict, mean_strain_FF1_dict, std_strain_FF1_dict, mean_thickness_FF1_dict, std_thickness_FF1_dict,
             mean_Umax_FF_dict,   std_Umax_FF_dict, mean_strain_FF_dict, std_strain_FF_dict, mean_thickness_FF_dict, std_thickness_FF_dict,
             mean_Umax_FF2_dict, std_Umax_FF2_dict, mean_strain_FF2_dict, std_strain_FF2_dict, mean_thickness_FF2_dict, std_thickness_FF2_dict,
             mean_Umax_RDG1_dict, std_Umax_RDG1_dict, mean_strain_RDG1_dict, std_strain_RDG1_dict, mean_thickness_RDG1_dict, std_thickness_RDG1_dict,
             mean_Umax_RDG_dict,   std_Umax_RDG_dict, mean_strain_RDG_dict, std_strain_RDG_dict, mean_thickness_RDG_dict, std_thickness_RDG_dict,
             mean_Umax_RDG2_dict, std_Umax_RDG2_dict, mean_strain_RDG2_dict, std_strain_RDG2_dict, mean_thickness_RDG2_dict, std_thickness_RDG2_dict
             ] = pickle.load(f)
    [dates_laser, 
        mean_delta_d_FF1_dict, std_delta_d_FF1_dict, mean_delta_d_star_FF1_dict, std_delta_d_star_FF1_dict, mean_d_min_FF1_dict, std_d_min_FF1_dict,  mean_A_FF1_dict, std_A_FF1_dict,
        mean_delta_d_FF2_dict, std_delta_d_FF2_dict, mean_delta_d_star_FF2_dict, std_delta_d_star_FF2_dict, mean_d_min_FF2_dict, std_d_min_FF2_dict,  mean_A_FF2_dict, std_A_FF2_dict,
        mean_delta_d_RDG1_dict, std_delta_d_RDG1_dict, mean_delta_d_star_RDG1_dict, std_delta_d_star_RDG1_dict, mean_d_min_RDG1_dict, std_d_min_RDG1_dict,  mean_A_RDG1_dict, std_A_RDG1_dict,
        mean_delta_d_RDG2_dict, std_delta_d_RDG2_dict, mean_delta_d_star_RDG2_dict, std_delta_d_star_RDG2_dict, mean_d_min_RDG2_dict, std_d_min_RDG2_dict,  mean_A_RDG2_dict, std_A_RDG2_dict,
        mean_delta_d_FF_dict, std_delta_d_FF_dict, mean_delta_d_star_FF_dict, std_delta_d_star_FF_dict, mean_d_min_FF_dict, std_d_min_FF_dict,  mean_A_FF_dict, std_A_FF_dict,
        mean_delta_d_RDG_dict, std_delta_d_RDG_dict, mean_delta_d_star_RDG_dict, std_delta_d_star_RDG_dict, mean_d_min_RDG_dict, std_d_min_RDG_dict,  mean_A_RDG_dict, std_A_RDG_dict,
        mean_Umax_FF1_dict, std_Umax_FF1_dict, mean_strain_FF1_dict, std_strain_FF1_dict, mean_thickness_FF1_dict, std_thickness_FF1_dict,
        mean_Umax_FF_dict,   std_Umax_FF_dict, mean_strain_FF_dict, std_strain_FF_dict, mean_thickness_FF_dict, std_thickness_FF_dict,
        mean_Umax_FF2_dict, std_Umax_FF2_dict, mean_strain_FF2_dict, std_strain_FF2_dict, mean_thickness_FF2_dict, std_thickness_FF2_dict,
        mean_Umax_RDG1_dict, std_Umax_RDG1_dict, mean_strain_RDG1_dict, std_strain_RDG1_dict, mean_thickness_RDG1_dict, std_thickness_RDG1_dict,
        mean_Umax_RDG_dict,   std_Umax_RDG_dict, mean_strain_RDG_dict, std_strain_RDG_dict, mean_thickness_RDG_dict, std_thickness_RDG_dict,
        mean_Umax_RDG2_dict, std_Umax_RDG2_dict, mean_strain_RDG2_dict, std_strain_RDG2_dict, mean_thickness_RDG2_dict, std_thickness_RDG2_dict
        ] = [dates_laser, 
        {d:mean_delta_d_FF1_dict[d] for d in dates_to_use}, {d:std_delta_d_FF1_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF1_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF1_dict[d] for d in dates_to_use}, {d:mean_d_min_FF1_dict[d] for d in dates_to_use}, {d:std_d_min_FF1_dict[d] for d in dates_to_use},  {d:mean_A_FF1_dict[d] for d in dates_to_use}, {d:std_A_FF1_dict[d] for d in dates_to_use},
        {d:mean_delta_d_FF2_dict[d] for d in dates_to_use}, {d:std_delta_d_FF2_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF2_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF2_dict[d] for d in dates_to_use}, {d:mean_d_min_FF2_dict[d] for d in dates_to_use}, {d:std_d_min_FF2_dict[d] for d in dates_to_use},  {d:mean_A_FF2_dict[d] for d in dates_to_use}, {d:std_A_FF2_dict[d] for d in dates_to_use},
        {d:mean_delta_d_RDG1_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG1_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG1_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG1_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG1_dict[d] for d in dates_to_use}, {d:std_d_min_RDG1_dict[d] for d in dates_to_use},  {d:mean_A_RDG1_dict[d] for d in dates_to_use}, {d:std_A_RDG1_dict[d] for d in dates_to_use},
        {d:mean_delta_d_RDG2_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG2_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG2_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG2_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG2_dict[d] for d in dates_to_use}, {d:std_d_min_RDG2_dict[d] for d in dates_to_use},  {d:mean_A_RDG2_dict[d] for d in dates_to_use}, {d:std_A_RDG2_dict[d] for d in dates_to_use},
        {d:mean_delta_d_FF_dict[d] for d in dates_to_use}, {d:std_delta_d_FF_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_FF_dict[d] for d in dates_to_use}, {d:std_delta_d_star_FF_dict[d] for d in dates_to_use}, {d:mean_d_min_FF_dict[d] for d in dates_to_use}, {d:std_d_min_FF_dict[d] for d in dates_to_use},  {d:mean_A_FF_dict[d] for d in dates_to_use}, {d:std_A_FF_dict[d] for d in dates_to_use},
        {d:mean_delta_d_RDG_dict[d] for d in dates_to_use}, {d:std_delta_d_RDG_dict[d] for d in dates_to_use}, {d:mean_delta_d_star_RDG_dict[d] for d in dates_to_use}, {d:std_delta_d_star_RDG_dict[d] for d in dates_to_use}, {d:mean_d_min_RDG_dict[d] for d in dates_to_use}, {d:std_d_min_RDG_dict[d] for d in dates_to_use},  {d:mean_A_RDG_dict[d] for d in dates_to_use}, {d:std_A_RDG_dict[d] for d in dates_to_use},
        {d:mean_Umax_FF1_dict[d] for d in dates_to_use}, {d:std_Umax_FF1_dict[d] for d in dates_to_use}, {d:mean_strain_FF1_dict[d] for d in dates_to_use}, {d:std_strain_FF1_dict[d] for d in dates_to_use}, {d:mean_thickness_FF1_dict[d] for d in dates_to_use}, {d:std_thickness_FF1_dict[d] for d in dates_to_use},
        {d:mean_Umax_FF_dict[d] for d in dates_to_use}, {d:  std_Umax_FF_dict[d] for d in dates_to_use}, {d:mean_strain_FF_dict[d] for d in dates_to_use}, {d:std_strain_FF_dict[d] for d in dates_to_use}, {d:mean_thickness_FF_dict[d] for d in dates_to_use}, {d:std_thickness_FF_dict[d] for d in dates_to_use},
        {d:mean_Umax_FF2_dict[d] for d in dates_to_use}, {d:std_Umax_FF2_dict[d] for d in dates_to_use}, {d:mean_strain_FF2_dict[d] for d in dates_to_use}, {d:std_strain_FF2_dict[d] for d in dates_to_use}, {d:mean_thickness_FF2_dict[d] for d in dates_to_use}, {d:std_thickness_FF2_dict[d] for d in dates_to_use},
        {d:mean_Umax_RDG1_dict[d] for d in dates_to_use}, {d:std_Umax_RDG1_dict[d] for d in dates_to_use}, {d:mean_strain_RDG1_dict[d] for d in dates_to_use}, {d:std_strain_RDG1_dict[d] for d in dates_to_use}, {d:mean_thickness_RDG1_dict[d] for d in dates_to_use}, {d:std_thickness_RDG1_dict[d] for d in dates_to_use},
        {d:mean_Umax_RDG_dict[d] for d in dates_to_use}, {d:  std_Umax_RDG_dict[d] for d in dates_to_use}, {d:mean_strain_RDG_dict[d] for d in dates_to_use}, {d:std_strain_RDG_dict[d] for d in dates_to_use}, {d:mean_thickness_RDG_dict[d] for d in dates_to_use}, {d:std_thickness_RDG_dict[d] for d in dates_to_use},
        {d:mean_Umax_RDG2_dict[d] for d in dates_to_use}, {d:std_Umax_RDG2_dict[d] for d in dates_to_use}, {d:mean_strain_RDG2_dict[d] for d in dates_to_use}, {d:std_strain_RDG2_dict[d] for d in dates_to_use}, {d:mean_thickness_RDG2_dict[d] for d in dates_to_use}, {d:std_thickness_RDG2_dict[d] for d in dates_to_use}
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
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'20', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'20', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'20', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[1]}
    labels = {'relaxation_slope' : r"$\alpha_R$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_strain_rate': r"$\i_{25 \%} $ [$Nm^{-1}$]",
                    'i_time_strain_rate': r"$\i_{25 \%} $ [$Ns^{-1}$]",
                    'i_disp_1': r"$\i_{100 \%} $ [$Nm^{-1}$]",
                    'i_time_1': r"$\i_{100 \%} $ [$Ns^{-1}$]"   }    
    

    #A vs force 80    
    
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
    
    fig_A_vs_force80 = createfigure.rectangle_figure(pixels)
    ax_A_vs_force80 = fig_A_vs_force80.gca()
    ax_A_vs_force80.errorbar(list(mean_force80_FF_dict.values()), list(mean_A_FF_dict.values()), yerr=list(std_A_FF_dict.values()), xerr=list(std_force80_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_A_vs_force80.errorbar(list(mean_force80_RDG_dict.values()),list(mean_A_RDG_dict.values()), yerr=list(std_A_RDG_dict.values()), xerr=list(std_force80_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_A_vs_force80.plot(force_80, fitted_response_A, ':k', alpha=0.8, linewidth=3, label=r"$R^2$ = " + str(np.round(score_A, 2)))
    for i in range(len(mean_force80_FF_dict)):
        date = dates_to_use[i]
        ax_A_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_FF_dict[date]+ 0.04, mean_A_FF_dict[date]+0.02), color = color_rocket[3], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_RDG_dict[date]+0.04, mean_A_RDG_dict[date]+0.02), color = color_rocket[1], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_A_vs_force80.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_A_vs_force80.set_xticks([0, 25, 50, 75, 100])
    ax_A_vs_force80.set_xticklabels([0, 25, 50, 75, 100], font=fonts.serif(), fontsize=24)
    ax_A_vs_force80.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_A_vs_force80.set_yticklabels([0, 0.25, 0.5, 0.75, 1], font=fonts.serif(), fontsize=24)
    ax_A_vs_force80.set_xlabel(r'$F_{80 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_A_vs_force80.set_ylabel(r'A [mm]', font=fonts.serif(), fontsize=26)


    savefigure.save_as_png(fig_A_vs_force80, "article_A_vs_F80")
    plt.close(fig_A_vs_force80)
    
    #A vs force 20    
    
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
    
    fig_A_vs_force20 = createfigure.rectangle_figure(pixels)
    ax_A_vs_force20 = fig_A_vs_force20.gca()
    ax_A_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_A_FF_dict.values()), yerr=list(std_A_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_A_vs_force20.errorbar(list(mean_force20_RDG_dict.values()),list(mean_A_RDG_dict.values()), yerr=list(std_A_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_A_vs_force20.plot(force_20, fitted_response_A, ':k', alpha=0.8, linewidth=3, label=r"$R^2$ = " + str(np.round(score_A, 2)))
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_A_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date]+ 0.04, mean_A_FF_dict[date]+0.02), color = color_rocket[3], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_A_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_A_RDG_dict[date]+0.02), color = color_rocket[1], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    ax_A_vs_force20.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_A_vs_force20.set_xlabel(r'$F_{20 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_A_vs_force20.set_ylabel(r'A [mm]', font=fonts.serif(), fontsize=26)
    ax_A_vs_force20.set_xticks([0, 5, 10, 15])
    ax_A_vs_force20.set_xticklabels([0, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_A_vs_force20.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_A_vs_force20.set_yticklabels([0, 0.25, 0.5, 0.75, 1], font=fonts.serif(), fontsize=24)
    savefigure.save_as_png(fig_A_vs_force20, "article_A_vs_F20")
    plt.close(fig_A_vs_force20)

        
        
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    deltad_threshold = 0.22
    strain_threshold = 0.39
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    
    locations = {"230331_FF": [0, 20],
                 "230331_FF2_2C": [0, 20],
                 "230331_RDG": [2, 20],
                 "230331_RDG1_ENTIER": [-18, 0],
                 "230331_FF1_recouvrance_et_relaxation_max" : [-18, 0],
                 "230331_FF1_ENTIER" : [-18, 0],
                 "230327_FF": [-10, 5],
                 "230327_RDG": [-10, 10],
                 "230403_FF": [8, 40],
                 "230403_RDG": [10, 40],
                 "230407_FF": [-30, -10],
                 "230407_RDG": [-30, -10],
                 "230411_FF": [-10, 10],
                 "230411_FF2_ENTIER": [-10, 10],
                 "230411_RDG": [-10, 10],
                 "230411_RDG2D": [-10, 10],
                 "230515_P002": [-12, 0],
                 "230515_P011": [-15, -2]}
    # plot_recovery(locations,  createfigure, savefigure, fonts)
    plot_A_vs_texturometer_forces(deltad_threshold, strain_threshold, createfigure, savefigure, fonts)