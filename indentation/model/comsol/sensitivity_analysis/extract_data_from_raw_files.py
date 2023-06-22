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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.signal import argrelextrema
import multiprocessing as mp
from functools import partial
from indentation.experiments.zwick.post_processing.utils import find_nearest
from datetime import datetime

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
    time_list, _ = utils.extract_disp_from_pkl()
    # plt.figure()
    # plt.plot(time_list, disp)
    # plt.show()
    return elongation, damage, disp, stress


def compute_dz(input_id):
    elongation, damage, disp, stress = get_data(input_id)
    sample_width = 36 # in mm
    dz = sample_width * (1 - (1 / elongation)**2)
    return dz

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
    _, _, disp, stress = get_data(input_id)
    time_list, _ = utils.extract_disp_from_pkl()
    dz = compute_dz(input_id)
    max_force_index = np.argmin(stress)
    # begginning_indentation_time_index = np.where(np.array(time_list) == find_nearest(np.array(time_list), -dz/10))[0][-1] 
    begginning_indentation_time_index = np.where(disp[0:int(len(disp)/2)] == find_nearest(disp[0:int(len(disp)/2)], -dz/10))[0][-1] 
    # begginning_indentation_time = time_list[begginning_indentation_time_index + 1]
    index_where_indentation_beggins = begginning_indentation_time_index + 0
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


def find_indicators_recovery(input_id):
    recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    recovery_disp_list_in_mm = [d*10 for d in recovery_disp_list]
    log_time = np.array([np.log(t+0.01) for t in recovery_time_list])
    log_disp = np.array([np.log(t+0.01) for t in recovery_disp_list])
    poly = PolynomialFeatures(degree=1, include_bias=False)
    poly_features_log = poly.fit_transform(log_time.reshape(-1, 1))
    model_log = make_pipeline(poly, LinearRegression())
    model_log.fit(poly_features_log, recovery_disp_list_in_mm)
    fitted_response_log = model_log.predict(log_time.reshape(-1, 1))
    A_log = model_log.steps[1][1].coef_[0]
    intercept_log = model_log.steps[1][1].intercept_
    model_power = make_pipeline(poly, LinearRegression())
    model_power.fit(poly_features_log, log_disp)
    fitted_response_power = model_power.predict(log_time.reshape(-1, 1))
    A_power = model_power.steps[1][1].coef_[0]
    intercept_power = model_power.steps[1][1].intercept_
    poly_features_linear = poly.fit_transform(np.array(recovery_time_list).reshape(-1, 1))
    model_linear = make_pipeline(poly, LinearRegression())
    model_linear.fit(poly_features_linear, recovery_disp_list_in_mm)
    fitted_response_linear = model_linear.predict(np.array(recovery_time_list).reshape(-1, 1))
    A_linear = model_linear.steps[1][1].coef_[0]
    intercept_linear = model_linear.steps[1][1].intercept_
    return log_time, fitted_response_log, A_log, intercept_log, fitted_response_power, A_power, intercept_power, fitted_response_linear, A_linear, intercept_linear

    
def find_recovery_slope(input_id):
    recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    recovery_disp_list_in_mm = [d*10 for d in recovery_disp_list]
    recovery_time_list = recovery_time_list[0:100]
    recovery_disp_list_in_mm = recovery_disp_list_in_mm[0:100]
    poly = PolynomialFeatures(degree=1, include_bias=False)
    poly_features_linear = poly.fit_transform(np.array(recovery_time_list).reshape(-1, 1))
    model_linear = make_pipeline(poly, LinearRegression())
    model_linear.fit(poly_features_linear, recovery_disp_list_in_mm)
    fitted_response_linear = model_linear.predict(np.array(recovery_time_list).reshape(-1, 1))
    A_linear = model_linear.steps[1][1].coef_[0]
    intercept_linear = model_linear.steps[1][1].intercept_
    return recovery_time_list, fitted_response_linear, A_linear, intercept_linear

def find_indicators_indentation_relaxation(input_id):
    indentation_relaxation_time_list, indentation_relaxation_stress_list = reshape_stress_to_indentation_relaxation(input_id)
    indentation_relaxation_stress_list_in_kpa = [s/1000 for s in indentation_relaxation_stress_list]
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
    max_stress_index = np.argmax(np.array(indentation_relaxation_stress_list))
    time_given_time, stress_given_time = indentation_relaxation_time_list[max_stress_index-1], indentation_relaxation_stress_list[max_stress_index-1]
    time_at_time_0, stress_at_time_0 = get_data_at_given_indentation_relaxation_time(input_id, time_given_time - 0.1) 
    i_time = (stress_given_time - stress_at_time_0) / (time_given_time - time_at_time_0) /1000    
    return delta_f, delta_f_star, relaxation_slope, i_time




def plot_reshaped_data(input_id):
    indentation_relaxation_time_list, indentation_relaxation_stress_list = reshape_stress_to_indentation_relaxation(input_id)
    recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    elongation, damage, _, _ = get_data(input_id)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()
    colors = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")

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
    max_stress_index = np.argmax(np.array(indentation_relaxation_stress_list))
    time_given_time, stress_given_time = indentation_relaxation_time_list[max_stress_index-1], indentation_relaxation_stress_list[max_stress_index-1]
    # time_at_time_0, stress_at_time_0 = indentation_relaxation_time_list[max_stress_index-2], indentation_relaxation_stress_list[max_stress_index-2]
    time_at_time_0, stress_at_time_0 = get_data_at_given_indentation_relaxation_time(input_id, time_given_time - 0.1) 
    # # time_given_time, stress_given_time = get_data_at_given_indentation_relaxation_time(input_id, 0.2) 
    i_time = (stress_given_time - stress_at_time_0) / (time_given_time - time_at_time_0) /1000
    ax_stress_vs_time.plot([time_at_time_0, time_given_time ], [stress_at_time_0/1000, stress_given_time/1000], '-', color = 'b', label = r"$\alpha'$ = " + str(np.round(i_time, 2)) + r' $kPa.s^{-1}$', linewidth=3)
    ax_stress_vs_time.plot([time_given_time, time_given_time ], [stress_at_time_0/1000, stress_given_time/1000], '--', color = 'b', linewidth=2)
    ax_stress_vs_time.plot([time_at_time_0, time_given_time ], [stress_at_time_0/1000, stress_at_time_0/1000], '--', color = 'b', linewidth=2)
    
    
    
    ax_stress_vs_time.plot(indentation_relaxation_time_list, indentation_relaxation_stress_list_in_kpa, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    # ax_stress_vs_time.set_xticks([0, 1, 2, 3, 4])
    # ax_stress_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    # ax_stress_vs_time.set_yticks([100, 200, 300, 400, 500])
    # ax_stress_vs_time.set_yticklabels([100, 200, 300, 400, 500], font=fonts.serif(), fontsize=24)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel("stress [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig_stress_vs_time, "stress_vs_time_id" + str(input_id) )
    plt.close(fig_stress_vs_time)

    recovery_time_list_linear, fitted_response_linear, A_linear, intercept_linear = find_recovery_slope(input_id)
    ax_disp_vs_time.plot(recovery_time_list_linear, fitted_response_linear ,   '-', color = color_rocket[3], label =r"$z = a t$" + "\na = " + str(np.round((A_linear), 2)), lw=3)
    ax_disp_vs_time.plot([recovery_time_list_linear[0], recovery_time_list_linear[-1]], [fitted_response_linear[0], fitted_response_linear[0]] ,   '--', color = color_rocket[3],  lw=3)
    ax_disp_vs_time.plot([recovery_time_list_linear[-1], recovery_time_list_linear[-1]], [fitted_response_linear[0], fitted_response_linear[-1]] ,   '--', color = color_rocket[3],  lw=3)


    ax_disp_vs_time.plot(recovery_time_list, recovery_disp_list_in_mm, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_disp_vs_time.set_xticks([0, 1, 2, 3, 4])
    ax_disp_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_yticks([0, 1, 2, 3, 4])
    ax_disp_vs_time.set_yticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel("U [mm]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    savefigure.save_as_png(fig_disp_vs_time, "disp_vs_time_id" + str(input_id) )
    plt.close(fig_disp_vs_time)
    

def plot_entire_data(input_id): #TODO plot data entiere
    elongation, damage, disp, stress = get_data(input_id)
    time_list, stress_dict = get_stress()
    # indentation_relaxation_time_list, indentation_relaxation_stress_list = reshape_stress_to_indentation_relaxation(input_id)
    # recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    # elongation, damage, _, _ = get_data(input_id)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()
    colors = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")

    stress_in_kpa = np.array([s/1000 for s in stress])
    disp_in_mm = [d*10 for d in disp]
    
    
    
    
    
    max_force = np.nanmax(stress_in_kpa)
    index_where_force_is_max = np.where(stress_in_kpa == max_force)[0][0]
    time_when_force_is_max = time_list[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(time_list == find_nearest(time_list, time_when_force_is_max+0.5))[0][0]
    relaxation_slope = (stress_in_kpa[index_where_time_is_end_relaxation_slope] - stress_in_kpa[index_where_force_is_max ]) / (time_list[index_where_time_is_end_relaxation_slope] - time_list[index_where_force_is_max])
    time_when_force_is_max = time_list[index_where_force_is_max]
    relaxation_duration = np.min([10, np.max(np.array(time_list))])
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time_list == find_nearest(time_list, end_of_relaxation))[0][0]
    delta_f = max_force - stress_in_kpa[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    # ax_stress_vs_time.plot([time_list[index_where_force_is_max ], time_list[index_where_time_is_end_relaxation_slope]], [stress_in_kpa[index_where_force_is_max ], stress_in_kpa[index_where_time_is_end_relaxation_slope]], '-', color = 'r', label = r"$\beta$ = " + str(np.round(relaxation_slope, 2)) + r" $kPa.s^{-1}$", linewidth=3)
    # ax_stress_vs_time.plot([time_list[index_where_time_is_end_relaxation_slope], time_list[index_where_time_is_end_relaxation_slope]], [stress_in_kpa[index_where_force_is_max ], stress_in_kpa[index_where_time_is_end_relaxation_slope]], '--', color = 'r', linewidth=2)
    # ax_stress_vs_time.plot([time_list[index_where_force_is_max ], time_list[index_where_time_is_end_relaxation_slope]], [stress_in_kpa[index_where_force_is_max ], stress_in_kpa[index_where_force_is_max ]], '--', color = 'r', linewidth=2)
    # ax_stress_vs_time.plot([time_list[index_where_time_is_end_relaxation], time_list[index_where_time_is_end_relaxation]], [stress_in_kpa[index_where_time_is_end_relaxation], max_force], '-', color = 'g', label = r"$\Delta F$  = " + str(np.round(delta_f, 2)) + " kPa \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)), linewidth=3)
    # ax_stress_vs_time.plot([time_list[index_where_time_is_end_relaxation]-0.2, time_list[index_where_time_is_end_relaxation]+0.2], [stress_in_kpa[index_where_time_is_end_relaxation], stress_in_kpa[index_where_time_is_end_relaxation]], '--', color = 'g', linewidth=2)
    # ax_stress_vs_time.plot([time_list[index_where_time_is_end_relaxation]-0.2, time_list[index_where_time_is_end_relaxation]+0.2], [max_force, max_force], '--', color = 'g', linewidth=2)
    # [time_at_time_0, stress_at_time_0] = [0, 0]
    # time_given_time, stress_given_time = get_data_at_given_indentation_relaxation_time(input_id, 0.1) 
    # i_time = (stress_given_time - stress_at_time_0) / (time_given_time - time_at_time_0) /1000
    # ax_stress_vs_time.plot([0, time_given_time ], [0, stress_given_time/1000], '-', color = 'b', label = r"$\alpha$ = " + str(np.round(i_time, 2)) + r' $kPa.s^{-1}$', linewidth=3)
    # ax_stress_vs_time.plot([time_given_time, time_given_time ], [0, stress_given_time/1000], '--', color = 'b', linewidth=2)
    # ax_stress_vs_time.plot([0, time_given_time ], [0, 0], '--', color = 'b', linewidth=2)
    
    
    
    ax_stress_vs_time.plot(time_list, stress_in_kpa, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    # ax_stress_vs_time.set_xticks([0, 1, 2, 3, 4])
    # ax_stress_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    # ax_stress_vs_time.set_yticks([100, 200, 300, 400, 500])
    # ax_stress_vs_time.set_yticklabels([100, 200, 300, 400, 500], font=fonts.serif(), fontsize=24)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel("stress [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig_stress_vs_time, "entire_stress_vs_time_id" + str(input_id) )
    plt.close(fig_stress_vs_time)

    recovery_time_list_linear, fitted_response_linear, A_linear, intercept_linear = find_recovery_slope(input_id)
    # log_time, fitted_response_log, A_log, intercept_log, fitted_response_power, A_power, intercept_power, fitted_response_linear, A_linear, intercept_linear = find_indicators_recovery(input_id)
    # ax_disp_vs_time.plot(np.exp(log_time), fitted_response_log ,   '--', color = color_rocket[4], label =r"$z = A \log{t} + z_{t=1}$" + "\nA = " + str(np.round(A_log, 2)) , lw=3)
    # ax_disp_vs_time.plot(np.exp(log_time), np.exp(fitted_response_power) ,   '--', color = color_rocket[5], label =r"$z = a t^{n}$" + "\na = " + str(np.round(np.exp(intercept_power), 2)) + " ; n = " + str(np.round(A_power, 2)) , lw=3)
    # ax_disp_vs_time.plot(recovery_time_list_linear, fitted_response_linear ,   '-', color = color_rocket[3], label =r"$z = a t$" + "\na = " + str(np.round(np.exp(A_linear), 2)), lw=3)
    # ax_disp_vs_time.plot([recovery_time_list_linear[0], recovery_time_list_linear[-1]], [fitted_response_linear[0], fitted_response_linear[0]] ,   '--', color = color_rocket[3],  lw=3)
    # ax_disp_vs_time.plot([recovery_time_list_linear[-1], recovery_time_list_linear[-1]], [fitted_response_linear[0], fitted_response_linear[-1]] ,   '--', color = color_rocket[3],  lw=3)


    ax_disp_vs_time.plot(time_list, disp_in_mm, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    # ax_disp_vs_time.set_xticks([0, 1, 2, 3, 4])
    # ax_disp_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    # ax_disp_vs_time.set_yticks([0, 1, 2, 3, 4])
    # ax_disp_vs_time.set_yticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel("U [mm]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    savefigure.save_as_png(fig_disp_vs_time, "entire_disp_vs_time_id" + str(input_id) )
    plt.close(fig_disp_vs_time)
    

def compute_and_export_all_indicators():
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    alpha_p_dict = {}
    beta_dict = {}
    delta_f_dict = {}
    delta_f_star_dict = {}
    a_dict = {}
    for input_id in ids_list:
        delta_f, delta_f_star, relaxation_slope, i_time = find_indicators_indentation_relaxation(input_id)
        _, _, A_linear, _ = find_recovery_slope(input_id)
        alpha_p_dict[input_id] = i_time
        beta_dict[input_id] = relaxation_slope
        delta_f_dict[input_id] = delta_f
        delta_f_star_dict[input_id] = delta_f_star
        a_dict[input_id] = A_linear
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict],
            f,
        )

def export_indicators_as_txt():    
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)

    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
        
    complete_txt_filename = utils.get_path_to_processed_data() / "indicators.txt"
    f = open(complete_txt_filename, "w")
    f.write("INDICATORS COMPUTED ON " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("Id \t elongation \t damage \t alphap \t beta \t deltaf \t deltaf* \t a \n")
    for i in range(len(ids_list)):
        id = ids_list[i]
        f.write(
            str(id)
            + "\t"
            + str(elongation_dict[id])
            + "\t"
            + str(damage_dict[id])
            + "\t"
            + str(alpha_p_dict[id])
            + "\t"
            + str(beta_dict[id])
            + "\t"
            + str(delta_f_dict[id])
            + "\t"
            + str(delta_f_star_dict[id])
            + "\t"
            + str(a_dict[id])
            + "\n"
        )
    f.close()    
    


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    ids_list, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()
    # time_list, disp_dict = utils.extract_disp_from_pkl()
    # time_list, stress_dict = utils.extract_stress_from_pkl()
    # reshape_disp_to_recovery(1)
    for id in ids_list:
        plot_reshaped_data(id)
        # reshape_stress_to_indentation_relaxation(id)
    # compute_and_export_all_indicators()
    # export_indicators_as_txt()
    # find_indicators_recovery(1)
    print('hello')

