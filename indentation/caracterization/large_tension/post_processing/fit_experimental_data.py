import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile, find_peaks
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


def plot_experimental_data_with_steps(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_elongation_vs_time = fig_elongation_vs_time.gca()
    ax_stress_vs_time = fig_stress_vs_time.gca()
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"linestyle": "-","color":'k', "linewidth": 2 ,"alpha":0.7}
    ax_elongation_vs_time.plot(time, elongation, **kwargs)
    ax_stress_vs_time.plot(time, stress, **kwargs)
    ax_stress_vs_elongation.plot(elongation, stress, **kwargs)
    
    
    step_numbers = index_final_step_dict_load.keys()
    for p in step_numbers:
        time_load = time_list_during_steps_dict_load[p]
        stress_load = stress_list_during_steps_dict_load[p]
        elongation_load = elongation_list_during_steps_dict_load[p]
        ax_elongation_vs_time.plot(time_load, elongation_load, '-r', lw=5, alpha=0.4)
        ax_stress_vs_time.plot(time_load, stress_load, '-r', lw=5, alpha=0.4)
        ax_stress_vs_elongation.plot(elongation_load, stress_load, '-r', lw=5, alpha=0.4)

        time_relaxation = time_list_during_steps_dict_relaxation[p]
        stress_relaxation = stress_list_during_steps_dict_relaxation[p]
        elongation_relaxation = elongation_list_during_steps_dict_relaxation[p]
        ax_elongation_vs_time.plot(time_relaxation, elongation_relaxation, '-b', lw=5, alpha=0.4)
        ax_stress_vs_time.plot(time_relaxation, stress_relaxation, '-b', lw=5, alpha=0.4)
        ax_stress_vs_elongation.plot(elongation_relaxation, stress_relaxation, '-b', lw=5, alpha=0.4)
    
    ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.grid(linestyle=':')
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    
    
    savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp_id_steps")
    savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_id_steps")
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_id_steps")
    


def compute_stress_vector_step_0_load(parameters, datafile, sheet, previous_step_values):
    step = 0
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    c1, beta, tau = parameters[0], parameters[1], parameters[2]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros_like(i_list)
    Q_list = np.zeros_like(i_list)
    S_list = np.zeros_like(i_list)
    Pi_list = np.zeros_like(i_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list

def compute_stress_vector_step_variable_c1_load(parameters, datafile, sheet, step, previous_step_values):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    c1, beta, tau = parameters[0], parameters[1], parameters[2]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros_like(i_list)
    Q_list = np.zeros_like(i_list)
    S_list = np.zeros_like(i_list)
    Pi_list = np.zeros_like(i_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list

def compute_stress_vector_step_variable_c1_relaxation(parameters, datafile, sheet, step, previous_step_values, c1):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    beta, tau = parameters[0], parameters[1]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros_like(i_list)
    Q_list = np.zeros_like(i_list)
    S_list = np.zeros_like(i_list)
    Pi_list = np.zeros_like(i_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list

def compute_stress_vector_load(parameters, datafile, sheet, step, previous_step_values, c1):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    beta, tau = parameters[0], parameters[1]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros_like(i_list)
    Q_list = np.zeros_like(i_list)
    S_list = np.zeros_like(i_list)
    Pi_list = np.zeros_like(i_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list

def compute_stress_vector_relaxation(parameters, datafile, sheet, step, load_step_values):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    tau = parameters[0]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros_like(i_list)
    Q_list = np.zeros_like(i_list)
    S_list = np.zeros_like(i_list)
    Pi_list = np.zeros_like(i_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = load_step_values
    for i in i_list[1:]:
        S_H_i = S_H_list[0]
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1]
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list


def find_parameters_step_0_load(datafile, sheet, previous_step_values, minimization_method):
    step = 0
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_0_load(parameters, datafile, sheet, previous_step_values)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, [20, 0, 30], method=minimization_method, bounds=[(1, 50),(0, 50),(1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_c1, optimized_beta, optimized_tau = parameters[0], parameters[1], parameters[2]
    return optimized_c1, optimized_beta, optimized_tau, parameters


def find_parameters_variable_c1_load(datafile, sheet, step, previous_step_values, minimization_method):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_variable_c1_load(parameters, datafile, sheet, step, previous_step_values)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, [20, 0, 30], method=minimization_method, #bounds=[(0, 100), (-10, 100), (0.1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_c1, optimized_beta, optimized_tau = parameters[0], parameters[1], parameters[2]
    return optimized_c1, optimized_beta, optimized_tau, parameters


def find_parameters_variable_c1_relaxation(datafile, sheet, step, previous_step_values, c1, minimization_method):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_variable_c1_relaxation(parameters, datafile, sheet, step, previous_step_values, c1)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, x0=[0, 30], method=minimization_method, #bounds=[(-10, 100), (0.1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_beta, optimized_tau = parameters[0], parameters[1]
    return c1, optimized_beta, optimized_tau, parameters


def find_parameters_load(datafile, sheet, step, previous_step_values, previous_step_optimized_parameters_load, c1, minimization_method):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_load(parameters, datafile, sheet, step, previous_step_values, c1)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, x0=previous_step_optimized_parameters_load, method=minimization_method, bounds=[(0, 50),(1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_beta, optimized_tau = parameters[0], parameters[1]
    return optimized_beta, optimized_tau, parameters


def find_parameters_relaxation(datafile, sheet, step, load_step_values, previous_step_optimized_parameters_relaxation, minimization_method):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_relaxation(parameters, datafile, sheet, step, load_step_values)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, x0=previous_step_optimized_parameters_relaxation, method=minimization_method, bounds=[(1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_tau = parameters[0]
    return optimized_tau, parameters

def find_parameters_all_steps(datafile, sheet, minimization_method_load, minimization_method_relaxation):
    c1_dict_load = {}
    beta_dict_load = {}
    tau_dict_load = {}
    fitted_stress_dict_load = {}
    c1_dict_relaxation = {}
    beta_dict_relaxation = {}
    tau_dict_relaxation = {}
    fitted_stress_dict_relaxation = {}
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    previous_step_values = 0, 0, 0, 0
    
    optimized_c1_step_0, optimized_beta_step_0, optimized_tau_step_0, parameters_step_0 = find_parameters_step_0_load(datafile, sheet, previous_step_values, minimization_method_load)
    c1_dict_load[0] = optimized_c1_step_0
    beta_dict_load[0] = optimized_beta_step_0
    tau_dict_load[0] = optimized_tau_step_0
    Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_0_load(parameters_step_0,  datafile, sheet,previous_step_values)
    fitted_stress_dict_load[0] = Pi_list
    
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
    load_step_values = previous_step_values
    previous_step_optimized_parameters_relaxation = [optimized_tau_step_0]
    
    optimized_tau_relaxation, parameters = find_parameters_relaxation(datafile, sheet, 0, load_step_values,  previous_step_optimized_parameters_relaxation, minimization_method_relaxation)
    c1_dict_relaxation[0] = optimized_c1_step_0
    beta_dict_relaxation[0] = optimized_beta_step_0
    tau_dict_relaxation[0] = optimized_tau_relaxation
    Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_relaxation(parameters,  datafile, sheet, 0, load_step_values)
    fitted_stress_dict_relaxation[0] = Pi_list
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
    previous_step_optimized_parameters_load = [optimized_beta_step_0, optimized_tau_relaxation]
    previous_step_optimized_parameters_relaxation = [optimized_tau_relaxation]
    
    for p in list(step_numbers)[1:]:
        optimized_beta_load, optimized_tau_load, parameters = find_parameters_load(datafile, sheet, p, previous_step_values, previous_step_optimized_parameters_load, optimized_c1_step_0, minimization_method_load)
        c1_dict_load[p] = optimized_c1_step_0
        beta_dict_load[p] = optimized_beta_load
        tau_dict_load[p] = optimized_tau_load
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_load(parameters,  datafile, sheet, p, previous_step_values, optimized_c1_step_0)
        fitted_stress_dict_load[p] = Pi_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
        load_step_values = previous_step_values
        previous_step_optimized_parameters_load = [optimized_beta_load, optimized_tau_load]
        
        optimized_tau_relaxation, parameters = find_parameters_relaxation(datafile, sheet, p, load_step_values, previous_step_optimized_parameters_relaxation, minimization_method_relaxation)
        c1_dict_relaxation[p] = optimized_c1_step_0
        beta_dict_relaxation[p] = optimized_beta_load
        tau_dict_relaxation[p] = optimized_tau_relaxation
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_relaxation(parameters,  datafile, sheet, p, load_step_values)
        fitted_stress_dict_relaxation[p] = Pi_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
        previous_step_optimized_parameters_relaxation = [optimized_tau_relaxation]
        
    return c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation


def find_parameters_all_steps_variable_c1(datafile, sheet, minimization_method):
    c1_dict_load = {}
    beta_dict_load = {}
    tau_dict_load = {}
    fitted_stress_dict_load = {}
    c1_dict_relaxation = {}
    beta_dict_relaxation = {}
    tau_dict_relaxation = {}
    fitted_stress_dict_relaxation = {}
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    previous_step_values = 0, 0, 0, 0
    
    optimized_c1_step_0, optimized_beta_step_0, optimized_tau_step_0, parameters_step_0 = find_parameters_variable_c1_load(datafile, sheet, 0, previous_step_values, minimization_method)
    c1_dict_load[0] = optimized_c1_step_0
    beta_dict_load[0] = optimized_beta_step_0
    tau_dict_load[0] = optimized_tau_step_0
    Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_variable_c1_load(parameters_step_0,  datafile, sheet, 0, previous_step_values)
    fitted_stress_dict_load[0] = Pi_list
    
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
    load_step_values = previous_step_values
    
    _, optimized_beta_relaxation, optimized_tau_relaxation, parameters = find_parameters_variable_c1_relaxation(datafile, sheet, 0, load_step_values, optimized_c1_step_0, minimization_method)
    c1_dict_relaxation[0] = optimized_c1_step_0
    beta_dict_relaxation[0] = optimized_beta_relaxation
    tau_dict_relaxation[0] = optimized_tau_relaxation
    Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_variable_c1_relaxation(parameters,  datafile, sheet, 0, load_step_values, optimized_c1_step_0)
    fitted_stress_dict_relaxation[0] = Pi_list
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
    
    for p in list(step_numbers)[1:]:
        optimized_c1_load, optimized_beta_load, optimized_tau_load, parameters = find_parameters_variable_c1_load(datafile, sheet, p, previous_step_values, minimization_method)
        c1_dict_load[p] = optimized_c1_load
        beta_dict_load[p] = optimized_beta_load
        tau_dict_load[p] = optimized_tau_load
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_variable_c1_load(parameters,  datafile, sheet, p, previous_step_values)
        fitted_stress_dict_load[p] = Pi_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
        load_step_values = previous_step_values

        _, optimized_beta_relaxation, optimized_tau_relaxation, parameters = find_parameters_variable_c1_relaxation(datafile, sheet, p, load_step_values, optimized_c1_load, minimization_method)
        c1_dict_relaxation[p] = optimized_c1_load
        beta_dict_relaxation[p] = optimized_beta_relaxation
        tau_dict_relaxation[p] = optimized_tau_relaxation
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_variable_c1_relaxation(parameters,  datafile, sheet, p, load_step_values, optimized_c1_load)
        fitted_stress_dict_relaxation[p] = Pi_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
        
        
    return c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation

def plot_results_variable_c1(datafile, sheet, minimization_method):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation = find_parameters_all_steps_variable_c1(datafile, sheet, minimization_method)
    
    def plot_experimental_vs_fitted_data(datafile, sheet, minimization_method):
        fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
        ax_elongation_vs_time = fig_elongation_vs_time.gca()
        ax_stress_vs_time = fig_stress_vs_time.gca()
        ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
        date = datafile[0:6]
        kwargs = {"linestyle": "-","color":'k', "linewidth": 2 ,"alpha":0.7}

        
        
        step_numbers = index_final_step_dict_load.keys()
        for p in step_numbers:
            time_load = time_list_during_steps_dict_load[p]
            stress_load = stress_list_during_steps_dict_load[p]
            elongation_load = elongation_list_during_steps_dict_load[p]
            fitted_stress = fitted_stress_dict_load[p]
            ax_elongation_vs_time.plot(time_load, elongation_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_load, stress_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_load, fitted_stress, '-r', lw=2, alpha=0.8)
            ax_stress_vs_elongation.plot(elongation_load, stress_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_elongation.plot(elongation_load, fitted_stress, '-r', lw=2, alpha=0.8)

            time_relaxation = time_list_during_steps_dict_relaxation[p]
            stress_relaxation = stress_list_during_steps_dict_relaxation[p]
            elongation_relaxation = elongation_list_during_steps_dict_relaxation[p]
            fitted_stress = fitted_stress_dict_relaxation[p]
            ax_elongation_vs_time.plot(time_relaxation, elongation_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_relaxation, stress_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_relaxation, fitted_stress, '-b', lw=2, alpha=0.8)
            ax_stress_vs_elongation.plot(elongation_relaxation, stress_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_elongation.plot(elongation_relaxation, fitted_stress, '-b', lw=2, alpha=0.8)

        ax_stress_vs_time.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        
        ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        
        ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
        
        ax_elongation_vs_time.grid(linestyle=':')
        ax_stress_vs_time.grid(linestyle=':')
        ax_stress_vs_elongation.grid(linestyle=':')
        
        savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp_vs_model_load_relaxation_" + minimization_method)
        savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_vs_model_load_relaxation_" + minimization_method)
        savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_vs_model_load_relaxation_" + minimization_method)
     
    def plot_fitted_parameters(datafile, sheet, minimization_method):
        fig_c1_vs_step = createfigure.rectangle_figure(pixels=180)
        fig_beta_vs_step = createfigure.rectangle_figure(pixels=180)
        fig_tau_vs_step = createfigure.rectangle_figure(pixels=180)
        ax_c1_vs_step = fig_c1_vs_step.gca()
        ax_beta_vs_step = fig_beta_vs_step.gca()
        ax_tau_vs_step = fig_tau_vs_step.gca()

        fig_c1_vs_elongation = createfigure.rectangle_figure(pixels=180)
        fig_beta_vs_elongation = createfigure.rectangle_figure(pixels=180)
        fig_tau_vs_elongation = createfigure.rectangle_figure(pixels=180)
        ax_c1_vs_elongation = fig_c1_vs_elongation.gca()
        ax_beta_vs_elongation = fig_beta_vs_elongation.gca()
        ax_tau_vs_elongation = fig_tau_vs_elongation.gca()
        
        step_list = c1_dict_load.keys()
        elongation_list = [elongation_final_step_dict_load[s] for s in step_list]
        
        ax_c1_vs_step.plot(step_list, [c1_dict_load[s] for s in step_list], 'ok')
        ax_beta_vs_step.plot(step_list, [beta_dict_load[s] for s in step_list], 'or')
        ax_tau_vs_step.plot(step_list, [tau_dict_load[s] for s in step_list], 'or')
        ax_beta_vs_step.plot(step_list, [beta_dict_relaxation[s] for s in step_list], 'ob')
        ax_tau_vs_step.plot(step_list, [tau_dict_relaxation[s] for s in step_list], 'ob')
        
        ax_c1_vs_step.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.plot(elongation_list, [c1_dict_load[s] for s in step_list], 'ok')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict_load[s] for s in step_list], 'or')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict_load[s] for s in step_list], 'or')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict_relaxation[s] for s in step_list], 'ob')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict_relaxation[s] for s in step_list], 'ob')
        
        ax_c1_vs_elongation.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)


        ax_c1_vs_step.set_xlabel("load step", font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_xlabel("load step", font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_xlabel("load step", font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)

        ax_c1_vs_step.grid(linestyle=':')
        ax_beta_vs_step.grid(linestyle=':')
        ax_tau_vs_step.grid(linestyle=':')

        ax_c1_vs_elongation.grid(linestyle=':')
        ax_beta_vs_elongation.grid(linestyle=':')
        ax_tau_vs_elongation.grid(linestyle=':')

        ax_c1_vs_step.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_title(minimization_method, font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_title(minimization_method, font=fonts.serif(), fontsize=26)
        

        
        savefigure.save_as_png(fig_c1_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_step_"+ minimization_method)
        savefigure.save_as_png(fig_beta_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_step_"+ minimization_method)
        savefigure.save_as_png(fig_tau_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_step_"+ minimization_method)

        savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_elongation_"+ minimization_method)
        savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_elongation_"+ minimization_method)
        savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_elongation_"+ minimization_method)

    plot_experimental_vs_fitted_data(datafile, sheet, minimization_method)
    plot_fitted_parameters(datafile, sheet, minimization_method)

def plot_results(datafile, sheet, minimization_method_load, minimization_method_relaxation):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation = find_parameters_all_steps(datafile, sheet, minimization_method_load, minimization_method_relaxation)
    
    def plot_experimental_vs_fitted_data(datafile, sheet, minimization_method_load, minimization_method_relaxation):
        fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
        ax_elongation_vs_time = fig_elongation_vs_time.gca()
        ax_stress_vs_time = fig_stress_vs_time.gca()
        ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
        date = datafile[0:6]
        kwargs = {"linestyle": "-","color":'k', "linewidth": 2 ,"alpha":0.7}

        
        
        step_numbers = index_final_step_dict_load.keys()
        for p in step_numbers:
            time_load = time_list_during_steps_dict_load[p]
            stress_load = stress_list_during_steps_dict_load[p]
            elongation_load = elongation_list_during_steps_dict_load[p]
            fitted_stress = fitted_stress_dict_load[p]
            ax_elongation_vs_time.plot(time_load, elongation_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_load, stress_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_load, fitted_stress, '-r', lw=2, alpha=0.8)
            ax_stress_vs_elongation.plot(elongation_load, stress_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_elongation.plot(elongation_load, fitted_stress, '-r', lw=2, alpha=0.8)

            time_relaxation = time_list_during_steps_dict_relaxation[p]
            stress_relaxation = stress_list_during_steps_dict_relaxation[p]
            elongation_relaxation = elongation_list_during_steps_dict_relaxation[p]
            fitted_stress = fitted_stress_dict_relaxation[p]
            ax_elongation_vs_time.plot(time_relaxation, elongation_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_relaxation, stress_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_relaxation, fitted_stress, '-b', lw=2, alpha=0.8)
            ax_stress_vs_elongation.plot(elongation_relaxation, stress_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_elongation.plot(elongation_relaxation, fitted_stress, '-b', lw=2, alpha=0.8)

        ax_stress_vs_time.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        
        ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        
        ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
        
        ax_elongation_vs_time.grid(linestyle=':')
        ax_stress_vs_time.grid(linestyle=':')
        ax_stress_vs_elongation.grid(linestyle=':')
        
        
        fig_elongation_vs_time.close()
        fig_stress_vs_time.close()
        fig_stress_vs_elongation.close()
        
        savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
        savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
        savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
     
    def plot_fitted_parameters(datafile, sheet, minimization_method_load, minimization_method_relaxation):
        fig_c1_vs_step = createfigure.rectangle_figure(pixels=180)
        fig_beta_vs_step = createfigure.rectangle_figure(pixels=180)
        fig_tau_vs_step = createfigure.rectangle_figure(pixels=180)
        ax_c1_vs_step = fig_c1_vs_step.gca()
        ax_beta_vs_step = fig_beta_vs_step.gca()
        ax_tau_vs_step = fig_tau_vs_step.gca()

        fig_c1_vs_elongation = createfigure.rectangle_figure(pixels=180)
        fig_beta_vs_elongation = createfigure.rectangle_figure(pixels=180)
        fig_tau_vs_elongation = createfigure.rectangle_figure(pixels=180)
        ax_c1_vs_elongation = fig_c1_vs_elongation.gca()
        ax_beta_vs_elongation = fig_beta_vs_elongation.gca()
        ax_tau_vs_elongation = fig_tau_vs_elongation.gca()
        
        step_list = c1_dict_load.keys()
        elongation_list = [elongation_final_step_dict_load[s] for s in step_list]
        
        ax_c1_vs_step.plot(step_list, [c1_dict_load[s] for s in step_list], 'ok')
        ax_beta_vs_step.plot(step_list, [beta_dict_load[s] for s in step_list], 'or')
        ax_tau_vs_step.plot(step_list, [tau_dict_load[s] for s in step_list], 'or')
        ax_beta_vs_step.plot(step_list, [beta_dict_relaxation[s] for s in step_list], 'ob')
        ax_tau_vs_step.plot(step_list, [tau_dict_relaxation[s] for s in step_list], 'ob')
        
        ax_c1_vs_step.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.plot(elongation_list, [c1_dict_load[s] for s in step_list], 'ok')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict_load[s] for s in step_list], 'or')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict_load[s] for s in step_list], 'or')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict_relaxation[s] for s in step_list], 'ob')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict_relaxation[s] for s in step_list], 'ob')
        
        ax_c1_vs_elongation.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)


        ax_c1_vs_step.set_xlabel("load step", font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_xlabel("load step", font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_xlabel("load step", font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)

        ax_c1_vs_step.grid(linestyle=':')
        ax_beta_vs_step.grid(linestyle=':')
        ax_tau_vs_step.grid(linestyle=':')

        ax_c1_vs_elongation.grid(linestyle=':')
        ax_beta_vs_elongation.grid(linestyle=':')
        ax_tau_vs_elongation.grid(linestyle=':')

        ax_c1_vs_step.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_title('load : ' + minimization_method_load + ' relax : ' + minimization_method_relaxation, font=fonts.serif(), fontsize=26)
        

        
        savefigure.save_as_png(fig_c1_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_step_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
        savefigure.save_as_png(fig_beta_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_step_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
        savefigure.save_as_png(fig_tau_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_step_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")

        savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
        savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")
        savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + "_fixedc1")

        fig_c1_vs_step.close()
        fig_beta_vs_step.close()
        fig_tau_vs_step.close()


        fig_c1_vs_elongation.close()
        fig_beta_vs_elongation.close()
        fig_tau_vs_elongation.close()

    plot_experimental_vs_fitted_data(datafile, sheet, minimization_method_load, minimization_method_relaxation)
    plot_fitted_parameters(datafile, sheet, minimization_method_load, minimization_method_relaxation)

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '230727'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = sheets_list_with_data[0]
    # plot_experimental_data_with_steps(datafile, sheet1)
    # index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict = store_peaks_information(datafile, sheet1)
    # elongation_list_during_steps_dict, time_list_during_steps_dict, stress_list_during_steps_dict = store_responses_of_steps(datafile, sheet1)
    # index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet1)
    # c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load = find_parameters_all_steps(datafile, sheet1)
    minimization_method_load_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    minimization_method_relaxation_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    for minimization_method_load,minimization_method_relaxation  in list(itertools.product(minimization_method_load_list, minimization_method_relaxation_list)):
        try:
            plot_results(datafile, sheet1, minimization_method_load, minimization_method_relaxation)
            print('load' , minimization_method_load, 'DONE')
            print('relaxation' , minimization_method_relaxation, 'DONE')
        except:
            print('load' , minimization_method_load, 'FAILED')
            print('relaxation' , minimization_method_relaxation, 'FAILED')
    print('hello')