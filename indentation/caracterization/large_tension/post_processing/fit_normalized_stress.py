import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile, find_peaks
from indentation.caracterization.large_tension.post_processing.identify_steps import store_and_export_step_data
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit, minimize, rosen, rosen_der
from scipy.integrate import quad
from numba import  prange
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle
import itertools


def extract_step_data(datafile, sheet):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + datafile[0:6] + "_" + sheet + "_step_information.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load,
             elongation_final_step_dict_load, elongation_list_during_steps_dict_load,
             time_list_during_steps_dict_load, stress_list_during_steps_dict_load,
             index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation,
             elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation,
             time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation] = pickle.load(f)
    return index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation



def compute_stress_vector_relaxation(tau, datafile, sheet, step):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    normalized_time, _, normalized_elongation =  normalize_step_data(datafile, sheet, step)
    delta_t = np.diff(normalized_time)[0]
    i_list = np.arange(0, len(normalized_time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = 1, 1, 0, 1
    for i in i_list[1:]:
        S_H_i = S_H_list[0]
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1]
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_list = np.multiply(normalized_elongation, S_list)
    return Pi_list


def normalize_step_data(datafile, sheet, step):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    time_relaxation = time_list_during_steps_dict_relaxation[step]
    stress_relaxation = stress_list_during_steps_dict_relaxation[step]
    elongation_relaxation = elongation_list_during_steps_dict_relaxation[step]
    try :
        index_local_max = argrelextrema(stress_relaxation, np.greater)[0][0]
        time_relaxation = time_relaxation[index_local_max:]
        stress_relaxation = stress_relaxation[index_local_max:]
        elongation_relaxation = elongation_relaxation[index_local_max:]
    except:
        None
    normalized_time = time_relaxation - time_relaxation[0]
    normalized_stress = stress_relaxation / stress_relaxation[0]
    normalized_elongation = elongation_relaxation - elongation_relaxation[0] + 1
    return normalized_time, normalized_stress, normalized_elongation






def identify_tau(datafile, sheet, step, minimization_method, previous_optimized_tau):
    _, normalized_experimental_stress, _ = normalize_step_data(datafile, sheet, step)
    def minimization_function(tau):
        Pi_list = compute_stress_vector_relaxation(tau, datafile, sheet, step)
        least_square = mean_squared_error(normalized_experimental_stress, Pi_list)
        return least_square
    
    res = minimize(minimization_function, x0=[previous_optimized_tau], method=minimization_method, bounds=[(1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_tau = parameters[0]
    return optimized_tau

def plot_normalized_data(datafile, sheet, createfigure):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    palette = sns.color_palette("rocket", n_colors=len(index_final_step_dict_load), as_cmap=False)
    for p in step_numbers:
        normalized_time, normalized_stress, normalized_elongation = normalize_step_data(datafile, sheet, p)
        ax_stress_vs_time.plot(normalized_time, normalized_stress, '-', lw=2, color = palette[p], alpha=0.9, label = str(p+1))
        ax_stress_vs_elongation.plot(normalized_elongation, normalized_stress, '-', lw=2, color = palette[p], alpha=0.9, label = str(p+1))
    ax_stress_vs_time.legend(prop=fonts.serif(), loc='lower left', framealpha=0.7)
    ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='lower left', framealpha=0.7)
    savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "normalized_stress_vs_time")
    savefigure.save_as_png(fig_stress_vs_elongation, datafile[0:6] + "normalized_stress_vs_elongation")

    print('hello')
    
def plot_normalized_data_with_fit(datafile, sheet, minimization_method, createfigure):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    fig_tau_vs_step = createfigure.rectangle_figure(pixels=180)
    ax_tau_vs_step = fig_tau_vs_step.gca()
    palette = sns.color_palette("rocket", n_colors=len(index_final_step_dict_load), as_cmap=False)
    tau_dict = {}
    previous_optimized_tau = 30
    for p in step_numbers:
        normalized_time, normalized_stress, normalized_elongation = normalize_step_data(datafile, sheet, p)
        ax_stress_vs_time.plot(normalized_time, normalized_stress, '-', lw=2, color = palette[p], alpha=0.9, label = str(p+1))
        ax_stress_vs_elongation.plot(normalized_elongation, normalized_stress, '-', lw=2, color = palette[p], alpha=0.9, label = str(p+1))
        optimized_tau = identify_tau(datafile, sheet, p, minimization_method, previous_optimized_tau)
        tau_dict[p] = optimized_tau
        previous_optimized_tau = optimized_tau
        fitted_stress = compute_stress_vector_relaxation(optimized_tau, datafile, sheet, p)
        ax_stress_vs_time.plot(normalized_time, fitted_stress, '--', lw=2, color = palette[p], alpha=0.9)
        ax_stress_vs_elongation.plot(normalized_elongation, fitted_stress, '--', lw=2, color = palette[p], alpha=0.9)  
        ax_tau_vs_step.plot([p+1], [optimized_tau], 'o', color = palette[p])
    ax_stress_vs_time.legend(prop=fonts.serif(), loc='lower left', framealpha=0.7)
    ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='lower left', framealpha=0.7)
    savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "normalized_stress_vs_time_" + minimization_method)
    savefigure.save_as_png(fig_stress_vs_elongation, datafile[0:6] + "normalized_stress_vs_elongation_" + minimization_method)
    savefigure.save_as_png(fig_tau_vs_step, datafile[0:6] + "tau_vs_step_" + minimization_method)
    plt.close(fig_stress_vs_time)
    plt.close(fig_stress_vs_elongation)
    plt.close(fig_tau_vs_step)

    print('hello')
    
    
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '230727'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    
    
    
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    
    minimization_method_load_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    minimization_method_relaxation_list =  ['TNC']#['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    sheets_list_with_data_temp = ["C1PA"]
    sheet = sheets_list_with_data_temp[0]
    for minimization_method in minimization_method_load_list:
        try:
            plot_normalized_data_with_fit(datafile, sheet, minimization_method, createfigure)
        except:
            None
    print('hello')
            