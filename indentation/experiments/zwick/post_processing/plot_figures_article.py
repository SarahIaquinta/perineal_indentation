""""
This file is used to generate nice looking figures for illustrating articles.
"""

import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from indentation.experiments.zwick.post_processing.compute_IRR_indicators import get_data_at_given_time
from sklearn.linear_model import LinearRegression
import seaborn as sns 
from indentation.experiments.zwick.post_processing.utils import find_nearest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle 

def compute_alpha(files_zwick, degree):
    datafile = '230403_C_Indentation_relaxation_500N_force.xlsx'
    sheet = '230403-FF2A'
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    index_where_force_is_max = np.where(force == max_force)[0][0]
    force_relaxation = force[index_where_force_is_max:]
    time_relaxation = time[index_where_force_is_max:]
    log_time_unshaped = np.array([-t**degree for t in time_relaxation])
    log_time = log_time_unshaped.reshape((-1, 1))
    model = LinearRegression()
    model.fit(log_time, force_relaxation)
    fitted_response = model.predict(log_time)
    alpha = 0
    score = np.sqrt(r2_score(force_relaxation, fitted_response))
    return alpha, fitted_response, log_time, force_relaxation, score

def compute_gamma(files_zwick, degree):
    datafile = '230403_C_Indentation_relaxation_500N_force.xlsx'
    sheet = '230403-FF2A'
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    index_where_force_is_max = np.where(force == max_force)[0][0]
    force_relaxation = force[:index_where_force_is_max]
    time_relaxation = time[:index_where_force_is_max]
    log_force_unshaped = np.array([np.log(f) for f in force_relaxation])
    log_force = log_force_unshaped.reshape((-1, 1))
    log_time_unshaped = np.array([np.log(t) for t in time_relaxation])
    log_time = log_time_unshaped.reshape((-1, 1))
    # degree=9
    # polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    # polyreg.fit(log_time,force_relaxation)
    model = LinearRegression()
    model.fit(log_time, log_force)
    fitted_response = model.predict(log_time)
    gamma = model.coef_
    score = np.sqrt(r2_score(log_force, fitted_response))
    # plt.figure()
    # plt.plot(log_time_unshaped, force_relaxation, 'b')
    # plt.plot(time_relaxation, force_relaxation, 'k')
    # plt.plot(log_time_unshaped, fitted_response, '--g')
    # plt.plot(1/log_time_unshaped, fitted_response, 'r')
    # plt.show()
    return gamma, fitted_response, log_time, force_relaxation, score

def plot_indicators_indentation_relaxation(files_zwick, createfigure, savefigure, fonts):
    datafile = '230403_C_Indentation_relaxation_500N_force.xlsx'
    sheet = '230403-FF2A'
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    fig_force_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_time = fig_force_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()

    kwargs = {"color":'k', "linewidth": 3}
    time0 = time[0]
    time = time - time0
    force0 = force[0]
    force = force - force0
    disp = disp - disp[0]
    
    max_force = np.nanmax(force)
    index_where_force_is_max = np.where(force == max_force)[0]
    time_when_force_is_max = time[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(time == find_nearest(time, time_when_force_is_max+0.5))[0]
    relaxation_slope = (force[index_where_time_is_end_relaxation_slope] - force[index_where_force_is_max ]) / (time[index_where_time_is_end_relaxation_slope] - time[index_where_force_is_max])
    time_when_force_is_max = time[index_where_force_is_max]
    relaxation_duration = 10
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation))
    delta_f = max_force - force[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    ax_force_vs_time.plot([time[index_where_force_is_max ], time[index_where_time_is_end_relaxation_slope]], [force[index_where_force_is_max ], force[index_where_time_is_end_relaxation_slope]], '-', color = 'r', label = r"$\alpha_R$ = " + str(np.round(relaxation_slope[0], 2)) + r" $N.s^{-1}$", linewidth=3)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation_slope], time[index_where_time_is_end_relaxation_slope]], [force[index_where_force_is_max ], force[index_where_time_is_end_relaxation_slope]], '--', color = 'r', linewidth=2)
    ax_force_vs_time.plot([time[index_where_force_is_max ], time[index_where_time_is_end_relaxation_slope]], [force[index_where_force_is_max ], force[index_where_force_is_max ]], '--', color = 'r', linewidth=2)
    ax_force_vs_time.annotate(r"$\beta$", (time[index_where_force_is_max ] + 0.75, 1.02),  color = 'r', font=fonts.serif(), fontsize=22)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]], time[index_where_time_is_end_relaxation[0]]], [force[index_where_time_is_end_relaxation][0], max_force], '-', color = 'g', label = r"$\Delta F$  = " + str(np.round(delta_f[0], 2)) + " N \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)), linewidth=3)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]]-0.2, time[index_where_time_is_end_relaxation[0]]+0.2], [force[index_where_time_is_end_relaxation][0], force[index_where_time_is_end_relaxation][0]], '--', color = 'g', linewidth=2)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]]-0.2, time[index_where_time_is_end_relaxation[0]]+0.2], [max_force, max_force], '--', color = 'g', linewidth=2)
    ax_force_vs_time.annotate(r"$\Delta F$", (time[index_where_time_is_end_relaxation[0]] - 1, 0.8),  color = 'g', font=fonts.serif(), fontsize=22)
    [time_at_time_0, force_at_time_0] = [0, 0]
    time_at_time, force_at_time, _ = get_data_at_given_time(files_zwick, datafile, sheet, 1) 
    i_time_disp = (force_at_time - force_at_time_0) / (time_at_time - time_at_time_0) #TODO debug 
    ax_force_vs_time.plot([0, time_at_time ], [0, force_at_time[0]], '-', color = 'b', label = r"$i_{25\%}$ = " + str(np.round(i_time_disp[0], 2)) + r' $Ns^{-1}$', linewidth=3)
    ax_force_vs_time.plot([time_at_time, time_at_time ], [0, force_at_time[0]], '--', color = 'b', label = r"$i_{25\%}$ = " + str(np.round(i_time_disp[0], 2)) + r' $Ns^{-1}$', linewidth=2)
    ax_force_vs_time.plot([0, time_at_time ], [0, 0], '--', color = 'b', label = r"$i_{25\%}$ = " + str(np.round(i_time_disp[0], 2)) + r' $Ns^{-1}$', linewidth=2)
    ax_force_vs_time.annotate(r"$\alpha$", (1.25, 0.075),  color = 'b', font=fonts.serif(), fontsize=22)
    ax_force_vs_time.plot(time[:index_where_time_is_end_relaxation[0][0]], force[0:index_where_time_is_end_relaxation[0][0]], linestyle=':',  **kwargs)
    ax_disp_vs_time.plot(time[:index_where_time_is_end_relaxation[0][0]], disp[0:index_where_time_is_end_relaxation[0][0]], linestyle=':',  **kwargs)    
    

    ax_force_vs_time.set_xticks([0, 1, 5, 10, 15])
    ax_force_vs_time.set_xticklabels([0, 1, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_force_vs_time.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax_force_vs_time.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], font=fonts.serif(), fontsize=24)
    ax_force_vs_time.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.set_ylabel(r"force [N]", font=fonts.serif(), fontsize=26)

    ax_disp_vs_time.set_xticks([0, 5, 10])
    ax_disp_vs_time.set_xticklabels([0, 5, 10], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_yticks([0, 1, 2, 3, 4, 5])
    ax_disp_vs_time.set_yticklabels([0, 1, 2, 3, 4, 5], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_force_vs_time, "article_force_vs_time_with_indicators")
    plt.close(fig_force_vs_time)
    savefigure.save_as_png(fig_disp_vs_time, "article_disp_vs_time_with_indicators")
    plt.close(fig_disp_vs_time)

def plot_indentation_relaxation_indicator_vs_texturometer_forces_beta(irr_indicator):
    maturation = [10, 13, 17, 21]
    dates_to_use = ['230331', '230403', '230407']
    maturation_dict = {'230331': 10, '230403': 13, '230407': 17}
    maturation_FF_dict = {k: v - 0.1 for k, v in maturation_dict.items()}
    maturation_RDG_dict = {k: v + 0.1 for k, v in maturation_dict.items()}
    maturation_dict_plots = {'230331': 'J+10', '230403': 'J+13', '230407': 'J+17'}
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + irr_indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = pickle.load(f)
        
    [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = [dates, {d:mean_data_FF1_dict[d] for d in dates_to_use}, {d:std_data_FF1_dict[d] for d in dates_to_use},
             {d:mean_data_FF2_dict[d] for d in dates_to_use}, {d:std_data_FF2_dict[d] for d in dates_to_use},
             {d:mean_data_RDG1_dict[d] for d in dates_to_use}, {d:std_data_RDG1_dict[d] for d in dates_to_use},
             {d:mean_data_RDG2_dict[d] for d in dates_to_use}, {d:std_data_RDG2_dict[d] for d in dates_to_use},
             {d:mean_data_FF_dict[d] for d in dates_to_use}, {d:std_data_FF_dict[d] for d in dates_to_use},
             {d:mean_data_RDG_dict[d] for d in dates_to_use}, {d:std_data_RDG_dict[d] for d in dates_to_use}
             ]
        
    path_to_processed_data_texturometer = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename_texturometer = path_to_processed_data_texturometer + "/forces_mean_std.pkl"
    with open(complete_pkl_filename_texturometer, "rb") as f:
        [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)

    [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = [dates, {d:mean_force20_FF1_dict[d] for d in dates_to_use}, {d:std_force20_FF1_dict[d] for d in dates_to_use}, {d:mean_force80_FF1_dict[d] for d in dates_to_use}, {d:std_force80_FF1_dict[d] for d in dates_to_use},
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
    labels = {'relaxation_slope' : r"$\beta$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_time': r"$\alpha $ [$Nmm^{-1}$]",
                    'i_time_time': r"$\alpha $ [$Ns^{-1}$]"  }    
    
    
    #data vs force 20    
    force_20_1 = np.concatenate((list(mean_force20_FF1_dict.values()), list(mean_force20_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_20_1) 
    data_1 = np.concatenate(( list(mean_data_FF1_dict.values()) , list(mean_data_RDG1_dict.values()) ))
    index_data_1_nan = np.isnan(data_1)
    indices_force_or_data_1_nan = [index_force_1_nan[i] or index_data_1_nan[i] for i in range(len(index_force_1_nan))]
    force_20_1_without_nan = np.array([force_20_1[i] for i in range(len(force_20_1)) if not indices_force_or_data_1_nan[i]])
    data_1_without_nan_force = np.array([data_1[i] for i in range(len(data_1)) if not indices_force_or_data_1_nan[i]])
    force_20_1 = force_20_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_1, data_1_without_nan_force)
    fitted_response_data_1 = model.predict(force_20_1)
    a_data_1 = reg.coef_
    b_data_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data_1 = reg.score(force_20_1, data_1_without_nan_force)
    
    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    data = np.concatenate(( list(mean_data_FF_dict.values()) , list(mean_data_RDG_dict.values()) ))
    index_data_nan = np.isnan(data)
    indices_force_or_data_nan = [index_force_nan[i] or index_data_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_data_nan[i]])
    data_without_nan_force = np.array([data[i] for i in range(len(data)) if not indices_force_or_data_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, data_without_nan_force)
    fitted_response_data = model.predict(force_20)
    a_data = reg.coef_
    b_data = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data = reg.score(force_20, data_without_nan_force)
    
    fig_data_vs_force20 = createfigure.rectangle_figure(pixels)
    ax_data_vs_force20 = fig_data_vs_force20.gca()
    ax_data_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_data_FF_dict.values()), yerr=list(std_data_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_data_vs_force20.errorbar(list(mean_force20_RDG_dict.values()), list(mean_data_RDG_dict.values()), yerr=list(std_data_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_data_vs_force20.plot(force_20, fitted_response_data, ':k', lw=3, label= r"$R^2$ = " + str(np.round(score_data, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_data_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date] +0.04, mean_data_FF_dict[date]+0.02), color = color_rocket[3], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        if date != '230403':
            ax_data_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_data_RDG_dict[date]-0.02), color = color_rocket[1], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force20.annotate(maturation_dict_plots['230403'], (mean_force20_RDG_dict['230403']-1, mean_data_RDG_dict['230403']+0.02), color = color_rocket[1], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force20.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_data_vs_force20.set_ylabel(labels[irr_indicator], font=fonts.serif(), fontsize=26)
    ax_data_vs_force20.set_xlabel(r'$F_{20 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_data_vs_force20.set_xticks([0, 5, 10, 15])
    ax_data_vs_force20.set_xticklabels([0, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_data_vs_force20.set_yticks([-0.6, -0.5, -0.4])
    ax_data_vs_force20.set_yticklabels([-0.6, -0.5, -0.4], font=fonts.serif(), fontsize=24)
    savefigure.save_as_png(fig_data_vs_force20, "article_" + irr_indicator + "_vs_force20_1+2")
    plt.close(fig_data_vs_force20)
        
def plot_indentation_relaxation_indicator_vs_texturometer_forces_alpha(irr_indicator):
    maturation = [10, 13, 17, 21]
    dates_to_use = ['230331', '230403', '230407']
    maturation_dict = {'230331': 10, '230403': 13, '230407': 17}
    maturation_FF_dict = {k: v - 0.1 for k, v in maturation_dict.items()}
    maturation_RDG_dict = {k: v + 0.1 for k, v in maturation_dict.items()}
    maturation_dict_plots = {'230331': 'J+10', '230403': 'J+13', '230407': 'J+17'}
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + irr_indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = pickle.load(f)
        
    [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = [dates, {d:mean_data_FF1_dict[d] for d in dates_to_use}, {d:std_data_FF1_dict[d] for d in dates_to_use},
             {d:mean_data_FF2_dict[d] for d in dates_to_use}, {d:std_data_FF2_dict[d] for d in dates_to_use},
             {d:mean_data_RDG1_dict[d] for d in dates_to_use}, {d:std_data_RDG1_dict[d] for d in dates_to_use},
             {d:mean_data_RDG2_dict[d] for d in dates_to_use}, {d:std_data_RDG2_dict[d] for d in dates_to_use},
             {d:mean_data_FF_dict[d] for d in dates_to_use}, {d:std_data_FF_dict[d] for d in dates_to_use},
             {d:mean_data_RDG_dict[d] for d in dates_to_use}, {d:std_data_RDG_dict[d] for d in dates_to_use}
             ]
        
    path_to_processed_data_texturometer = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename_texturometer = path_to_processed_data_texturometer + "/forces_mean_std.pkl"
    with open(complete_pkl_filename_texturometer, "rb") as f:
        [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)

    [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = [dates, {d:mean_force20_FF1_dict[d] for d in dates_to_use}, {d:std_force20_FF1_dict[d] for d in dates_to_use}, {d:mean_force80_FF1_dict[d] for d in dates_to_use}, {d:std_force80_FF1_dict[d] for d in dates_to_use},
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
    labels = {'relaxation_slope' : r"$\beta$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_time': r"$\alpha $ [$Nmm^{-1}$]",
                    'i_time_time': r"$\alpha $ [$Ns^{-1}$]"  }    
    
    
    
    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    data = np.concatenate(( list(mean_data_FF_dict.values()) , list(mean_data_RDG_dict.values()) ))
    index_data_nan = np.isnan(data)
    indices_force_or_data_nan = [index_force_nan[i] or index_data_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_data_nan[i]])
    data_without_nan_force = np.array([data[i] for i in range(len(data)) if not indices_force_or_data_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, data_without_nan_force)
    fitted_response_data = model.predict(force_20)
    a_data = reg.coef_
    b_data = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data = reg.score(force_20, data_without_nan_force)
    
    fig_data_vs_force20 = createfigure.rectangle_figure(pixels)
    ax_data_vs_force20 = fig_data_vs_force20.gca()
    ax_data_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_data_FF_dict.values()), yerr=list(std_data_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_data_vs_force20.errorbar(list(mean_force20_RDG_dict.values()), list(mean_data_RDG_dict.values()), yerr=list(std_data_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_data_vs_force20.plot(force_20, fitted_response_data, ':k', lw=3, label= r"$R^2$ = " + str(np.round(score_data, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_data_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date] +0.04, mean_data_FF_dict[date]+0.02), color = color_rocket[3], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        if date != '230403':
            ax_data_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_data_RDG_dict[date]-0.02), color = color_rocket[1], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force20.annotate(maturation_dict_plots['230403'], (mean_force20_RDG_dict['230403']-1, mean_data_RDG_dict['230403']+0.02), color = color_rocket[1], fontsize=18, bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force20.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_data_vs_force20.set_ylabel(labels[irr_indicator], font=fonts.serif(), fontsize=26)
    ax_data_vs_force20.set_xlabel(r'$F_{20 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_data_vs_force20.set_xticks([0, 5, 10, 15])
    ax_data_vs_force20.set_xticklabels([0, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_data_vs_force20.set_yticks([0, 0.2, 0.4, 0.6])
    ax_data_vs_force20.set_yticklabels([0, 0.2, 0.4, 0.6], font=fonts.serif(), fontsize=24)
    savefigure.save_as_png(fig_data_vs_force20, "article_" + irr_indicator + "_vs_force20_1+2")
    plt.close(fig_data_vs_force20)
        
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230403']
    types_of_essay = ['C_Indentation_relaxation_500N_force.xlsx']#,'C_Indentation_relaxation_maintienFnulle_500N_trav.xls',  'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    # plot_indicators_indentation_relaxation(files_zwick, createfigure, savefigure, fonts)
    # plot_indentation_relaxation_indicator_vs_texturometer_forces_beta('relaxation_slope')
    plot_indentation_relaxation_indicator_vs_texturometer_forces_alpha('i_time_time')
