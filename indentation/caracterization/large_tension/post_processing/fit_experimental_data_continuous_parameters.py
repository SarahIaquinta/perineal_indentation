import numpy as np
# import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile, find_peaks
# from indentation.caracterization.large_tension.post_processing.identify_steps import store_and_export_step_data
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
    


def compute_stress_vector_step_0_load(parameters_step_0, datafile, sheet, initial_values):
    step = 0
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    beta_list = np.zeros_like(S_H_list)
    tau_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = initial_values
    c1, a_beta, b_beta, a_tau , b_tau = parameters_step_0[0], parameters_step_0[1], parameters_step_0[2], parameters_step_0[3], parameters_step_0[4]

    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        beta = a_beta*lambda_i + b_beta
        tau = a_tau*lambda_i + b_tau
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        # Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*(S_H_i - S_H_list[i-1])
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
        beta_list[i] = beta
        tau_list[i] = tau
    # Pi_H_list = np.multiply(elongation_list, S_H_list)
    # Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list


def compute_stress_vector_relaxation(datafile, sheet, step, load_step_values):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    tau_list = np.zeros_like(S_H_list)
    beta_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0], beta_list[0] , tau_list[0] = load_step_values
    for i in i_list[1:]:
        tau = tau_list[0]
        beta = beta_list[0]
        S_H_i = S_H_list[0]
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1]
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
        tau_list[i] = tau
        beta_list[i] = beta
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list


def compute_stress_vector_load(parameters, datafile, sheet, step, initial_values, c1):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    beta_list = np.zeros_like(S_H_list)
    tau_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0], beta_list[0], tau_list[0] = initial_values
    a_beta, a_tau = parameters[0], parameters[1]
    b_beta = beta_list[0] - a_beta * elongation_list[0]
    b_tau = tau_list[0] - a_tau * elongation_list[0]
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        beta = a_beta*lambda_i + b_beta
        tau = a_tau*lambda_i + b_tau
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        # Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*(S_H_i - S_H_list[i-1])
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
        beta_list[i] = beta
        tau_list[i] = tau
    # Pi_H_list = np.multiply(elongation_list, S_H_list)
    # Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list




def find_parameters_step_0_load(datafile, sheet, initial_values, minimization_method):
    step = 0
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters_step_0):
        Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list = compute_stress_vector_step_0_load(parameters_step_0, datafile, sheet, initial_values)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, [20, 0, 0, 0, 30], method=minimization_method, bounds=[(1, 50),(-50, 150), (0, 10), (-10, 10), (0, 50)],
               options={'disp': False}) 
    # print(res.message)
    parameters_step_0 = res.x
    optimized_c1, optimized_a_beta, optimized_b_beta, optimized_a_tau , optimized_b_tau = parameters_step_0[0], parameters_step_0[1], parameters_step_0[2], parameters_step_0[3], parameters_step_0[4]
    return optimized_c1, optimized_a_beta, optimized_b_beta, optimized_a_tau , optimized_b_tau, parameters_step_0


def find_parameters_load(datafile, sheet, step, previous_step_values, previous_step_optimized_parameters_load, c1, minimization_method):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list = compute_stress_vector_load(parameters, datafile, sheet, step, previous_step_values, c1)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, x0=previous_step_optimized_parameters_load, method=minimization_method, bounds=[(-1000, 1000),(-1000, 1000)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_a_beta, optimized_a_tau = parameters[0], parameters[1]
    return optimized_a_beta, optimized_a_tau, parameters





def find_parameters_all_steps(datafile, sheet, minimization_method):
    c1_dict_load = {}
    c1_dict_relaxation = {}
    a_beta_dict_load = {}
    a_tau_dict_load = {}
    tau_dict_load = {}
    tau_dict_relaxation = {}
    beta_dict_load = {}
    beta_dict_relaxation = {}
    fitted_stress_dict_load = {}
    fitted_stress_dict_relaxation = {}
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    initial_values = 0, 0, 0, 0
    optimized_c1_step_0, optimized_a_beta_step_0, optimized_b_beta_step_0,  optimized_a_tau_step_0, optimized_b_tau_step_0, parameters_step_0 = find_parameters_step_0_load(datafile, sheet, initial_values, minimization_method)
    c1_dict_load[0] = optimized_c1_step_0
    a_beta_dict_load[0] = optimized_a_beta_step_0
    a_tau_dict_load[0] = optimized_a_tau_step_0
    Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list = compute_stress_vector_step_0_load(parameters_step_0,  datafile, sheet,initial_values)
    fitted_stress_dict_load[0] = Pi_list
    beta_dict_load[0] = beta_list
    tau_dict_load[0] = tau_list
    
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1], beta_list[-1], tau_list[-1]
    load_step_values = previous_step_values
    
    Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list = compute_stress_vector_relaxation( datafile, sheet, 0, load_step_values)
    fitted_stress_dict_relaxation[0] = Pi_list
    beta_dict_relaxation[0] = beta_list
    tau_dict_relaxation[0] = tau_list
    
    
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1], beta_list[-1], tau_list[-1]
    previous_step_optimized_parameters_load = [optimized_a_beta_step_0, optimized_a_tau_step_0]
    
    for p in list(step_numbers)[1:]:
        optimized_a_beta_load, optimized_a_tau_load, parameters = find_parameters_load(datafile, sheet, p, previous_step_values, previous_step_optimized_parameters_load, optimized_c1_step_0, minimization_method)
        c1_dict_load[p] = optimized_c1_step_0
        a_beta_dict_load[p] = optimized_a_beta_load
        a_tau_dict_load[p] = optimized_a_tau_load
        Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list = compute_stress_vector_load(parameters,  datafile, sheet, p, previous_step_values, optimized_c1_step_0)
        fitted_stress_dict_load[p] = Pi_list
        beta_dict_load[p] = beta_list
        tau_dict_load[p] = tau_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1], beta_list[-1], tau_list[-1]
        load_step_values = previous_step_values
        previous_step_optimized_parameters_load = [optimized_a_beta_load, optimized_a_tau_load]
        
        Q_list, S_list, S_H_list, Pi_list, beta_list, tau_list = compute_stress_vector_relaxation(parameters,  datafile, sheet, p, load_step_values)
        c1_dict_relaxation[p] = optimized_c1_step_0
        beta_dict_relaxation[p] = beta_list
        tau_dict_relaxation[p] = tau_list
        fitted_stress_dict_relaxation[p] = Pi_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1], beta_list[-1], tau_list[-1]
        
    return c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation, a_beta_dict_load, a_tau_dict_load






def compute_and_export_results(datafile, sheet, minimization_method):
    c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation, a_beta_dict_load, a_tau_dict_load = find_parameters_all_steps(datafile, sheet, minimization_method)
    pkl_filename = datafile[0:6] + "_" + sheet + "_" +  minimization_method +  "continuous_parameters.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump([c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation, a_beta_dict_load, a_tau_dict_load],f,)
    print(sheet, ' results to pkl DONE')




def plot_results(datafile, sheet, minimization_method):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    pkl_filename = datafile[0:6] + "_" + sheet + "_" +  minimization_method +  "continuous_parameters.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, fitted_stress_dict_relaxation, a_beta_dict_load, a_tau_dict_load] = pickle.load(f)
    
    
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
        
        
        plt.close(fig_elongation_vs_time)
        plt.close(fig_stress_vs_time)
        plt.close(fig_stress_vs_elongation)
        
        savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp_vs_model_" + minimization_method + "_continuous_parameters")
        savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_vs_model_" + minimization_method + "_continuous_parameters")
        savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_vs_model_" + minimization_method + "_continuous_parameters")
     
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
        # ax_beta_vs_step.plot(step_list, [beta_dict_relaxation[s] for s in step_list], 'ob')
        ax_tau_vs_step.plot(list(step_list)[4:], [tau_dict_relaxation[s] for s in list(step_list)[4:]], 'ob')
        
        ax_c1_vs_step.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.plot(elongation_list, [c1_dict_load[s] for s in step_list], 'ok')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict_load[s] for s in step_list], 'or')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict_load[s] for s in step_list], 'or')
        # ax_beta_vs_elongation.plot(elongation_list, [beta_dict_relaxation[s] for s in step_list], 'ob')
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
        

        
        savefigure.save_as_png(fig_c1_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_step_L_" + minimization_method + "_continuous_parameters")
        savefigure.save_as_png(fig_beta_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_step_L_" + minimization_method + "_continuous_parameters")
        savefigure.save_as_png(fig_tau_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_step_L_" + minimization_method + "_continuous_parameters")

        savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_elongation_L_" + minimization_method + "_continuous_parameters")
        savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_elongation_L_" + minimization_method + "_continuous_parameters")
        savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_elongation_L_" + minimization_method + "_continuous_parameters")

        plt.close(fig_c1_vs_step)
        plt.close(fig_beta_vs_step)
        plt.close(fig_tau_vs_step)


        plt.close(fig_c1_vs_elongation)
        plt.close(fig_beta_vs_elongation)
        plt.close(fig_tau_vs_elongation)

    plot_experimental_vs_fitted_data(datafile, sheet, minimization_method)
    plot_fitted_parameters(datafile, sheet, minimization_method)




####
def export_results_as_txt(files, minimization_method_load, minimization_method_relaxation):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_txt_filename = path_to_processed_data + "/large_tension_caracterization.txt"
    f = open(complete_txt_filename, "w")
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    _, sheet_list = files.get_sheets_from_datafile(datafile)
    sheet_list.remove('C2PA')
    sheet_list.remove('C2PB')
    sheet_list.remove('C5TA')
    sheet_list.remove('C15TA')
    sheet_list.remove('C1TB')
    for sheet in sheet_list:
        pkl_filename = datafile[0:6] + "_" + sheet + "_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + ".pkl"
        complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
        f.write("INDICATORS FOR " + sheet + "\n")
        f.write("step \t c1 load \t beta load \t tau load \t c1 relax \t beta relax \t tau relax \n")
        with open(complete_pkl_filename, "rb") as g:
            [c1_dict_load, beta_dict_load, tau_dict_load, _, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, _] = pickle.load(g)
        steps = list(c1_dict_load.keys())
        for i in range(len(c1_dict_load)):
            step = steps[i]
            f.write(
                str(step) 
                + "\t"
                + str(c1_dict_load[step])
                + "\t"
                + str(beta_dict_load[step])
                + "\t"
                + str(tau_dict_load[step])
                + "\t"
                + str(c1_dict_relaxation[step])
                + "\t"
                + str(beta_dict_relaxation[step])
                + "\t"
                + str(tau_dict_relaxation[step])
                + "\n \n")
            
    f.close()

def get_pig_numbers(files, experiment_date):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_txt_filename = path_to_processed_data + "/large_tension_caracterization.txt"
    f = open(complete_txt_filename, "w")
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    _, sheet_list = files.get_sheets_from_datafile(datafile)
    sheet_list.remove('C2PA')
    sheet_list.remove('C2PB')
    sheet_list.remove('C5TA')
    sheet_list.remove('C15TA')
    sheet_list.remove('C1TB')
    corresponding_sheets_int = [int(idx[1:-2]) for idx in sheet_list]
    a = list(set(sorted(corresponding_sheets_int)))
    corresponding_sheets = [str(i) for i in a]
    return corresponding_sheets

def get_sheets_for_given_pig(pig_number, files, experiment_date):
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_txt_filename = path_to_processed_data + "/large_tension_caracterization.txt"
    f = open(complete_txt_filename, "w")
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    _, sheet_list = files.get_sheets_from_datafile(datafile)
    # sheet_list.remove('C2PA')
    # sheet_list.remove('C2PB')
    # sheet_list.remove('C5TA')
    # sheet_list.remove('C15TA')
    # sheet_list.remove('C1TB')
    corresponding_sheets = [idx for idx in sheet_list if idx[1:-2] == str(pig_number)]
    return corresponding_sheets

def get_sheets_for_given_region(region, files, experiment_date):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    _, sheet_list = files.get_sheets_from_datafile(datafile)
    # sheet_list.remove('C2PA')
    # sheet_list.remove('C2PB')
    # sheet_list.remove('C5TA')
    # sheet_list.remove('C15TA')
    # sheet_list.remove('C1TB')
    corresponding_sheets = [idx for idx in sheet_list if idx[-2] == str(region)]
    return corresponding_sheets

       
def plot_indicators_and_stress_per_pig(pig_number, files, minimization_method_load, minimization_method_relaxation):
    palette = sns.color_palette("Set2")
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    corresponding_sheets_pig = get_sheets_for_given_pig(pig_number, files)
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    color_dict_region = {"P":palette[3], "T":palette[2], "D":palette[4], "S":palette[1]}
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    fig_c1_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_c1_vs_elongation = fig_c1_vs_elongation.gca()
    fig_tau_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_tau_vs_elongation = fig_tau_vs_elongation.gca()
    fig_beta_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_beta_vs_elongation = fig_beta_vs_elongation.gca()
    
    ax_stress_vs_time.plot([0], [0], '-', color=palette[3], label="P")
    ax_stress_vs_time.plot([0], [0], '-', color=palette[2], label="T")
    ax_stress_vs_time.plot([0], [0], '-', color=palette[4], label="D")
    ax_stress_vs_time.plot([0], [0], '-', color=palette[1], label="S")
    
    for sheet in corresponding_sheets_pig:
        region = sheet[-2]
        label = sheet[-2:]
        color = color_dict_region[region]
        
        pkl_filename = datafile[0:6] + "_" + sheet + "_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + ".pkl"
        complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
        with open(complete_pkl_filename, "rb") as g:
            [c1_dict_load, beta_dict_load, tau_dict_load, stress_dict_load_model, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, stress_dict_relax_model] = pickle.load(g)
        
        _, _, elongation_init_step_dict_load, _, _, time_list_during_steps_dict_load, stress_dict_load_exp, _, _, _, _, _, time_list_during_steps_dict_relaxation, stress_dict_relax_exp = extract_step_data(datafile, sheet)    
        for i in range(len(time_list_during_steps_dict_load)):
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_load.values())[i][0:len(time_list_during_steps_dict_load):100], list(stress_dict_load_exp.values())[i][0:len(time_list_during_steps_dict_load):100], 'o', markeredgecolor=color, ms=5, markerfacecolor='none')
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_load.values())[i], list(stress_dict_load_model.values())[i], '-', color=color, lw=1)
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_relaxation.values())[i][0:len(time_list_during_steps_dict_load):100], list(stress_dict_relax_exp.values())[i][0:len(time_list_during_steps_dict_load):100], 'o', markeredgecolor=color, ms=5, markerfacecolor='none')
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_relaxation.values())[i], list(stress_dict_relax_model.values())[i], '-', color=color, lw=1)
        
        
        steps = c1_dict_load.keys()
        elongations = list(elongation_init_step_dict_load.values())
        ax_c1_vs_elongation.plot(elongations, list(c1_dict_load.values()), 'o-', color=color, ms=5, lw=1, label=label)
        ax_tau_vs_elongation.plot(elongations, list(tau_dict_load.values()), 'o-', color=color, ms=5, lw=1, label=label)
        ax_beta_vs_elongation.plot(elongations, list(beta_dict_load.values()), 'o-', color=color, ms=5, lw=1, label=label)
        
    
    ax_stress_vs_time.set_xlabel("time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)

    ax_c1_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_tau_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_beta_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_c1_vs_elongation.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_tau_vs_elongation.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)
    ax_beta_vs_elongation.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_c1_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_tau_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_beta_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
       
    savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "_" +  "pig" + str(pig_number) + "_stress_vs_time_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)
    savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" +  "pig" + str(pig_number) + "_c1_vs_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)
    savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" +  "pig" + str(pig_number) + "_tau_vs_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)
    savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" +  "pig" + str(pig_number) + "_beta_vs_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)

    plt.close(fig_stress_vs_time)
    plt.close(fig_c1_vs_elongation)
    plt.close(fig_tau_vs_elongation)
    plt.close(fig_beta_vs_elongation)

def plot_indicators_and_stress_per_region(region, files, minimization_method_load, minimization_method_relaxation):
    palette = sns.color_palette("Set2")
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    corresponding_sheets_region = get_sheets_for_given_region(region, files)
    nb_of_cochons = len(corresponding_sheets_region)
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_time = fig_stress_vs_time.gca()
    fig_c1_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_c1_vs_elongation = fig_c1_vs_elongation.gca()
    fig_tau_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_tau_vs_elongation = fig_tau_vs_elongation.gca()
    fig_beta_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_beta_vs_elongation = fig_beta_vs_elongation.gca()
    color_dict_region={"P": sns.color_palette("RdPu", nb_of_cochons), "S": sns.color_palette("Oranges", nb_of_cochons), "D": sns.color_palette("Greens", nb_of_cochons), "T": sns.color_palette("Blues", nb_of_cochons)}
    
    for k in range(len(corresponding_sheets_region)):
        sheet = corresponding_sheets_region[k]
        region = sheet[-2]
        pig_number = sheet[1:-2] + sheet[-1]
        label = pig_number
        color = color_dict_region[region][k]
        
        pkl_filename = datafile[0:6] + "_" + sheet + "_L_" + minimization_method_load + "_R_" + minimization_method_relaxation + ".pkl"
        complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
        with open(complete_pkl_filename, "rb") as g:
            [c1_dict_load, beta_dict_load, tau_dict_load, stress_dict_load_model, c1_dict_relaxation, beta_dict_relaxation, tau_dict_relaxation, stress_dict_relax_model] = pickle.load(g)
        
        _, _, elongation_init_step_dict_load, _, _, time_list_during_steps_dict_load, stress_dict_load_exp, _, _, _, _, _, time_list_during_steps_dict_relaxation, stress_dict_relax_exp = extract_step_data(datafile, sheet)    
        for i in range(len(time_list_during_steps_dict_load)):
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_load.values())[i][0:len(time_list_during_steps_dict_load):100], list(stress_dict_load_exp.values())[i][0:len(time_list_during_steps_dict_load):100], 'o', markeredgecolor=color, ms=5, markerfacecolor='none')
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_load.values())[i], list(stress_dict_load_model.values())[i], '-', color=color, lw=1)
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_relaxation.values())[i][0:len(time_list_during_steps_dict_load):100], list(stress_dict_relax_exp.values())[i][0:len(time_list_during_steps_dict_load):100], 'o', markeredgecolor=color, ms=5, markerfacecolor='none')
            ax_stress_vs_time.plot(list(time_list_during_steps_dict_relaxation.values())[i], list(stress_dict_relax_model.values())[i], '-', color=color, lw=1)
        ax_stress_vs_time.annotate(str(pig_number), (list(time_list_during_steps_dict_relaxation.values())[-1][-1]*1.01, list(stress_dict_relax_model.values())[-1][-1]*1.01), color=color)
        
        steps = c1_dict_load.keys()
        elongations = list(elongation_init_step_dict_load.values())
        ax_c1_vs_elongation.plot(elongations, list(c1_dict_load.values()), 'o-', color=color, ms=5, lw=1)
        ax_tau_vs_elongation.plot(elongations, list(tau_dict_load.values()), 'o-', color=color, ms=5, lw=1)
        ax_beta_vs_elongation.plot(elongations, list(beta_dict_load.values()), 'o-', color=color, ms=5, lw=1)
        ax_c1_vs_elongation.annotate(str(pig_number), (elongations[-1], list(c1_dict_load.values())[-1]), color=color)
        ax_tau_vs_elongation.annotate(str(pig_number), (elongations[-1], list(tau_dict_load.values())[-1]), color=color)
        ax_beta_vs_elongation.annotate(str(pig_number), (elongations[-1], list(beta_dict_load.values())[-1]), color=color)

    
    ax_stress_vs_time.set_xlabel("time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)

    ax_c1_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_tau_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_beta_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_c1_vs_elongation.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_tau_vs_elongation.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)
    ax_beta_vs_elongation.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
    
    # ax_stress_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    # ax_c1_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    # ax_tau_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    # ax_beta_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
       
    savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "_" +  "region" + str(region) + "_stress_vs_time_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)
    savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" +  "region" + str(region) + "_c1_vs_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)
    savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" +  "region" + str(region) + "_tau_vs_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)
    savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" +  "region" + str(region) + "_beta_vs_elongation_L_" + minimization_method_load + "_R_" + minimization_method_relaxation)

    plt.close(fig_stress_vs_time)
    plt.close(fig_c1_vs_elongation)
    plt.close(fig_tau_vs_elongation)
    plt.close(fig_beta_vs_elongation)



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '230727'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    
    
    
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    
    minimization_method_list = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
    sheets_list_with_data_temp = ["C1PA"]
    # export_results_as_txt(files_zwick, minimization_method_load_list[0], minimization_method_relaxation_list[0])
    # pig_numbers = get_pig_numbers(files_zwick)
    regions = ['P', 'S', 'D', 'T']
    # for pig in pig_numbers:
    #     plot_indicators_and_stress_per_pig(pig, files_zwick, 'CG', 'TNC')
    # for region in regions:
    #     plot_indicators_and_stress_per_region(region, files_zwick, 'CG', 'TNC')
    for sheet in sheets_list_with_data_temp:
    # # plot_experimental_data_with_steps(datafile, sheet1)
    #     # index_init_step_dict, index_final_step_dict, elongation_init_step_dict, elongation_final_step_dict = store_peaks_information(datafile, sheet1)
    #     # elongation_list_during_steps_dict, time_list_during_steps_dict, stress_list_during_steps_dict = store_responses_of_steps(datafile, sheet1)
    #     # print(sheet, "DOING")
    #     # try:
    #     #     store_and_export_step_data(datafile, sheet)
    #     # except:
    #     #     print("FAILED", sheet)
    #     # print(sheet, "DONE")
    #     # index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    # # c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load = find_parameters_all_steps(datafile, sheet1)
        # for minimization_method_load,minimization_method_relaxation  in list(itertools.product(minimization_method_load_list, minimization_method_relaxation_list)):
        for minimization_method in minimization_method_list:
            try:
                compute_and_export_results(datafile, sheet, minimization_method)
            # plot_results_load_and_relaxation(datafile, sheet, minimization_method)
            # try:
                print('load' , minimization_method, 'DONE')
            #     print('relaxation' , minimization_method_relaxation, 'DONE')
            except:
                print('load' , minimization_method, 'FAILED')
            #     print('relaxation' , minimization_method_relaxation, 'FAILED')
    print('hello')