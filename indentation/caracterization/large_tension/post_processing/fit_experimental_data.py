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
    


def compute_stress_vector_step_0(parameters, datafile, sheet, previous_step_values):
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

def compute_stress_vector(parameters, datafile, sheet, step, previous_step_values, c1):
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


def find_parameters_step_0(datafile, sheet, previous_step_values):
    step = 0
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_0(parameters, datafile, sheet, previous_step_values)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, [20, 0, 30], method='Nelder-Mead',
               options={'gtol': 1e-1, 'disp': False}) 
    print(res.message)
    parameters = res.x
    optimized_c1, optimized_beta, optimized_tau = parameters[0], parameters[1], parameters[2]
    return optimized_c1, optimized_beta, optimized_tau, parameters

def find_parameters(datafile, sheet, step, previous_step_values, c1):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]

    def minimization_function(parameters):
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector(parameters, datafile, sheet, step, previous_step_values, c1)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, [0, 30], method='Nelder-Mead',
               options={'gtol': 1e-1, 'disp': False}) 
    print(res.message)
    parameters = res.x
    optimized_beta, optimized_tau = parameters[0], parameters[1]
    return optimized_beta, optimized_tau, parameters

def find_parameters_all_steps(datafile, sheet):
    c1_dict_load = {}
    beta_dict_load = {}
    tau_dict_load = {}
    fitted_stress_dict_load = {}
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    previous_step_values = 0, 0, 0, 0
    
    optimized_c1_step_0, optimized_beta_step_0, optimized_tau_step_0, parameters_step_0 = find_parameters_step_0(datafile, sheet, previous_step_values)
    c1_dict_load[0] = optimized_c1_step_0
    beta_dict_load[0] = optimized_beta_step_0
    tau_dict_load[0] = optimized_tau_step_0
    Q_list, S_list, S_H_list, Pi_list = compute_stress_vector_step_0(parameters_step_0,  datafile, sheet,previous_step_values)
    fitted_stress_dict_load[0] = Pi_list
    
    previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
    
    for p in list(step_numbers)[1:]:
        optimized_beta, optimized_tau, parameters = find_parameters(datafile, sheet, p, previous_step_values, optimized_c1_step_0)
        c1_dict_load[p] = optimized_c1_step_0
        beta_dict_load[p] = optimized_beta
        tau_dict_load[p] = optimized_tau
        Q_list, S_list, S_H_list, Pi_list = compute_stress_vector(parameters,  datafile, sheet, p, previous_step_values, optimized_c1_step_0)
        fitted_stress_dict_load[p] = Pi_list
        previous_step_values = Q_list[-1], S_list[-1], S_H_list[-1], Pi_list[-1]
    return c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load


def plot_experimental_vs_fitted_data(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load = find_parameters_all_steps(datafile, sheet)
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
        ax_elongation_vs_time.plot(time_load, elongation_load, '-k', lw=2, alpha=0.4)
        ax_stress_vs_time.plot(time_load, stress_load, '-k', lw=2, alpha=0.4)
        ax_stress_vs_elongation.plot(elongation_load, stress_load, '-k', lw=2, alpha=0.4)
        fitted_stress = fitted_stress_dict_load[p]
        ax_stress_vs_elongation.plot(elongation_load, fitted_stress, 'or', lw=2, alpha=0.8)


    
    ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.grid(linestyle=':')
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp_vs_model")
    savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_vs_model")
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_vs_model")
  
def plot_fitted_parameters(datafile, sheet):
    c1_dict_load, beta_dict_load, tau_dict_load, fitted_stress_dict_load = find_parameters_all_steps(datafile, sheet)
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
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
    ax_beta_vs_step.plot(step_list, [beta_dict_load[s] for s in step_list], 'ob')
    ax_tau_vs_step.plot(step_list, [tau_dict_load[s] for s in step_list], 'og')
    
    ax_c1_vs_step.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_beta_vs_step.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
    ax_tau_vs_step.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)

    ax_c1_vs_elongation.plot(elongation_list, [c1_dict_load[s] for s in step_list], 'ok')
    ax_beta_vs_elongation.plot(elongation_list, [beta_dict_load[s] for s in step_list], 'ob')
    ax_tau_vs_elongation.plot(elongation_list, [tau_dict_load[s] for s in step_list], 'og')
    
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
    
    savefigure.save_as_png(fig_c1_vs_step, datafile[0:6] + "_" + sheet + "_c1_vs_load_step")
    savefigure.save_as_png(fig_beta_vs_step, datafile[0:6] + "_" + sheet + "_beta_vs_load_step")
    savefigure.save_as_png(fig_tau_vs_step, datafile[0:6] + "_" + sheet + "_tau_vs_load_step")

    savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" + sheet + "_c1_vs_load_elongation")
    savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" + sheet + "_beta_vs_load_elongation")
    savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" + sheet + "_tau_vs_load_elongation")


    

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
    plot_experimental_vs_fitted_data(datafile, sheet1)
    plot_fitted_parameters(datafile, sheet1)
    print('hello')