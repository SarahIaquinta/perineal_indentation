import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
from indentation.model.comsol.sensitivity_analysis import utils
import os
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pandas as pd
import scipy
import skimage
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression

import multiprocessing as mp
# from indentation.experiments.zwick.post_processing.utils import find_nearest
from functools import partial
from indentation.experiments.zwick.post_processing.utils import find_nearest

def get_inputs():
    path_to_file = utils.reach_data_path() / 'disp_silicone.xlsx'
    input_data = pd.read_excel(path_to_file, sheet_name='input', header=0, names=["Id", "elongation", "damage"]) 
    ids = input_data.Id
    elongations = input_data.elongation
    damages = input_data.damage
    ids_list = ids.tolist()
    elongation_dict = {ids.tolist()[i]: elongations.tolist()[i] for i in range(len(ids.tolist()))}
    damage_dict = {ids.tolist()[i]: damages.tolist()[i] for i in range(len(ids.tolist()))}
    return ids_list, elongation_dict, damage_dict

def get_stress():
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    path_to_file_stress = utils.reach_data_path() / 'stress_silicone.xlsx'
    stress_data = pd.read_excel(path_to_file_stress, sheet_name='output', header=0)
    times = stress_data.time
    time_list = [float(t) for t in times.to_list()]
    stress_dict = {}
    for id in ids_list:
        stress_id = stress_data[id]
        stress_dict[id] = [float(s) for s in stress_id.to_list()]
    return time_list, stress_dict

def get_disp():
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    path_to_file_disp = utils.reach_data_path() / 'disp_silicone.xlsx'
    disp_data = pd.read_excel(path_to_file_disp, sheet_name='output', header=0)
    times = disp_data.time
    time_list = [float(t) for t in times.to_list()]
    disp_dict = {}
    for id in ids_list:
        disp_id = disp_data[id]
        disp_dict[id] = [float(s) for s in disp_id.to_list()]
    return time_list, disp_dict

def get_data(input_id):
    _, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()
    _, disp_dict = utils.extract_disp_from_pkl()
    _, stress_dict = utils.extract_stress_from_pkl()
    disp = disp_dict[input_id]
    stress = stress_dict[input_id]
    elongation = elongation_dict[input_id]
    damage = damage_dict[input_id]
    return elongation, damage, disp, stress

def reshape_disp_to_recovery(input_id):
    _, _, disp, _ = get_data(input_id)
    time_list, _ = utils.extract_disp_from_pkl()
    begginning_recovery_time = 6
    index_where_recovery_beggins = np.where(time_list == find_nearest(time_list, begginning_recovery_time))[0][0]
    recovery_time_list = [t - time_list[index_where_recovery_beggins+1] for t in time_list[index_where_recovery_beggins+1:]]
    recovery_disp_list = [d - disp[index_where_recovery_beggins+1] for d in disp[index_where_recovery_beggins+1:]]
    # plt.figure()
    # plt.plot(time_list, disp)
    # plt.plot(recovery_time_list, recovery_disp_list)
    # plt.show()
    return recovery_time_list, recovery_disp_list

def reshape_stress_to_indentation_relaxation(input_id):
    _, _, _, stress = get_data(input_id)
    time_list, _ = utils.extract_disp_from_pkl()
    begginning_indentation_time = 1.8
    index_where_indentation_beggins = np.where(time_list == find_nearest(time_list, begginning_indentation_time))[0][0]
    end_relaxation_time = 6
    index_where_relaxation_ends = np.where(time_list == find_nearest(time_list, end_relaxation_time))[0][0]
    indentation_relaxation_time_list = [t - time_list[index_where_indentation_beggins] for t in time_list[index_where_indentation_beggins:index_where_relaxation_ends] ]
    indentation_relaxation_stress_list = stress[index_where_indentation_beggins:index_where_relaxation_ends]
    indentation_relaxation_stress_list = [-s for s in indentation_relaxation_stress_list]
    return indentation_relaxation_time_list, indentation_relaxation_stress_list


def get_data_at_given_indentation_relaxation_time(input_id, given_time):
    indentation_relaxation_time_list, indentation_relaxation_stress_list = reshape_stress_to_indentation_relaxation(input_id)
    # recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    time_given_time = find_nearest(indentation_relaxation_time_list,  given_time)
    index_where_time_is_time_given_time = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, time_given_time))[0][0]
    stress_given_time = indentation_relaxation_stress_list[index_where_time_is_time_given_time]
    return time_given_time, stress_given_time

def plot_data(input_id):
    indentation_relaxation_time_list, indentation_relaxation_stress_list = reshape_stress_to_indentation_relaxation(input_id)
    recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    elongation, damage, _, _ = get_data(input_id)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()
    colors = sns.color_palette("Paired")
    
    indentation_relaxation_stress_list_in_kpa = [s/1000 for s in indentation_relaxation_stress_list]
    recovery_disp_list_in_mm = [d*10 for d in recovery_disp_list]
    
    
    max_force = np.nanmax(indentation_relaxation_stress_list_in_kpa)
    index_where_force_is_max = np.where(indentation_relaxation_stress_list_in_kpa == max_force)[0][0]
    time_when_force_is_max = indentation_relaxation_time_list[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, time_when_force_is_max+0.5))[0][0]
    relaxation_slope = (indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation_slope] - indentation_relaxation_stress_list_in_kpa[index_where_force_is_max ]) / (indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope] - indentation_relaxation_time_list[index_where_force_is_max])
    time_when_force_is_max = indentation_relaxation_time_list[index_where_force_is_max]
    relaxation_duration = np.min([10, np.max(np.array(indentation_relaxation_time_list))])
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, end_of_relaxation))[0][0]
    delta_f = max_force - indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    ax_stress_vs_time.plot([indentation_relaxation_time_list[index_where_force_is_max ], indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope]], [indentation_relaxation_stress_list_in_kpa[index_where_force_is_max ], indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation_slope]], '-', color = 'r', label = r"$\beta$ = " + str(np.round(relaxation_slope, 2)) + r" $kPa.s^{-1}$", linewidth=3)
    ax_stress_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope], indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope]], [indentation_relaxation_stress_list_in_kpa[index_where_force_is_max ], indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation_slope]], '--', color = 'r', linewidth=2)
    ax_stress_vs_time.plot([indentation_relaxation_time_list[index_where_force_is_max ], indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope]], [indentation_relaxation_stress_list_in_kpa[index_where_force_is_max ], indentation_relaxation_stress_list_in_kpa[index_where_force_is_max ]], '--', color = 'r', linewidth=2)
    ax_stress_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation], indentation_relaxation_time_list[index_where_time_is_end_relaxation]], [indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation], max_force], '-', color = 'g', label = r"$\Delta F$  = " + str(np.round(delta_f, 2)) + " kPa \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)), linewidth=3)
    ax_stress_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation]-0.2, indentation_relaxation_time_list[index_where_time_is_end_relaxation]+0.2], [indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation], indentation_relaxation_stress_list_in_kpa[index_where_time_is_end_relaxation]], '--', color = 'g', linewidth=2)
    ax_stress_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation]-0.2, indentation_relaxation_time_list[index_where_time_is_end_relaxation]+0.2], [max_force, max_force], '--', color = 'g', linewidth=2)
    [time_at_time_0, stress_at_time_0] = [0, 0]
    time_given_time, stress_given_time = get_data_at_given_indentation_relaxation_time(input_id, 0.1) 
    i_time = (stress_given_time - stress_at_time_0) / (time_given_time - time_at_time_0) /1000
    ax_stress_vs_time.plot([0, time_given_time ], [0, stress_given_time/1000], '-', color = 'b', label = r"$\alpha$ = " + str(np.round(i_time, 2)) + r' $kPa.s^{-1}$', linewidth=3)
    ax_stress_vs_time.plot([time_given_time, time_given_time ], [0, stress_given_time/1000], '--', color = 'b', linewidth=2)
    ax_stress_vs_time.plot([0, time_given_time ], [0, 0], '--', color = 'b', linewidth=2)
    
    
    
    ax_stress_vs_time.plot(indentation_relaxation_time_list, indentation_relaxation_stress_list_in_kpa, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_stress_vs_time.set_xticks([0, 1, 2, 3, 4])
    ax_stress_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_stress_vs_time.set_yticks([100, 200, 300, 400, 500])
    ax_stress_vs_time.set_yticklabels([100, 200, 300, 400, 500], font=fonts.serif(), fontsize=24)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel("stress [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig_stress_vs_time, "stress_vs_time_id" + str(input_id) )
    plt.close(fig_stress_vs_time)


    log_time_unshaped = np.array([np.log(t+0.01) for t in recovery_time_list])
    log_time = log_time_unshaped.reshape((-1, 1))
    model = LinearRegression()
    model.fit(log_time[1:], recovery_disp_list_in_mm[1:])
    fitted_response = model.predict(log_time)
    A = model.coef_
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket", as_cmap=False)    
    ax_disp_vs_time.plot(np.exp(log_time), fitted_response ,   '--', color = color_rocket[4], label =r"$z = A \log{t} + z_{t=1}$" + "\nA = " + str(np.round(A[0], 2)) , lw=3)
 


    ax_disp_vs_time.plot(recovery_time_list, recovery_disp_list_in_mm, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_disp_vs_time.set_xticks([0, 1, 2, 3, 4])
    ax_disp_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_yticks([0, 1, 2, 3, 4])
    ax_disp_vs_time.set_yticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel("U [mm]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.legend(prop=fonts.serif(), loc='lower center', framealpha=0.7)
    savefigure.save_as_png(fig_disp_vs_time, "disp_vs_time_id" + str(input_id) )
    plt.close(fig_disp_vs_time)
    



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    # ids_list, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()
    # time_list, disp_dict = utils.extract_disp_from_pkl()
    # time_list, stress_dict = utils.extract_stress_from_pkl()
    # reshape_disp_to_recovery(1)
    plot_data(1)
    # reshape_disp_to_recovery(input_id)
    print('hello')

