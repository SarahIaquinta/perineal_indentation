import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile, find_peaks
from indentation.caracterization.large_tension.post_processing.identify_steps import store_and_export_step_data
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from indentation.caracterization.large_tension.post_processing.fit_experimental_data import extract_step_data

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit, minimize, rosen, rosen_der
from scipy.integrate import quad
from numba import  prange
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pickle
import itertools



def compute_stress_vector_step_0_load(parameters, datafile, sheet, previous_step_values):
    step = 0
    _, _, _, _, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, _, _, _, _, _, _, _ = extract_step_data(datafile, sheet)
    elongation_list, time, _ = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    elongation_list = elongation_list[:]
    time = time[:]
    delta_t_list = np.diff(time)
    delta_t = delta_t_list[0]
    c1, beta, tau = parameters[0], parameters[1], parameters[2]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        ## Calcul de Q_i avec la formule de l'article :
        # Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        ## Calcul de Q_i par integration simple :
        Q_i = Q_list[i-1] + beta*(S_H_i - S_H_list[i-1])*np.exp(-delta_t/tau)
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list


def compute_stress_vector_step_fixed_c1_variable_beta_load(c1, parameters, datafile, sheet, step, previous_step_values):
    _, _, _, _, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, _, _, _, _, _, _, _ = extract_step_data(datafile, sheet)
    elongation_list, time, _ = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    elongation_list = elongation_list[:]
    time = time[:]
    delta_t_list = np.diff(time)
    delta_t = delta_t_list[0]
    beta, tau = parameters[0], parameters[1]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        ## Calcul de Q_i avec la formule de l'article :
        # Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        ## Calcul de Q_i par integration simple :
        Q_i = Q_list[i-1] + beta*(S_H_i - S_H_list[i-1])*np.exp(-delta_t/tau)
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_H_list = np.multiply(elongation_list, S_H_list)
    Pi_Q_list = np.multiply(elongation_list, Q_list)
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list



def compute_stress_vector_step_relaxation(tau, datafile, sheet, step, load_step_values):
    _, _, _, _, _, _, _, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, experimental_stress = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]
    # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = load_step_values
    t1 = time[0]
    Q1 = Q_list[0]
    for i in i_list[1:]:
        t = time[i]
        S_H_i = S_H_list[0]
        Q_i = np.exp((t1-t)/tau)*Q1
        # Q_i = np.exp(-delta_t/tau)*Q_list[i-1]
        # Q_i = (S_list[0] -  S_H_list[0])*np.exp((t1-t)/tau)
        S_i = Q_i+ S_H_i
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

    res = minimize(minimization_function, [20, 1, 30], method=minimization_method, bounds=[(1, 100),(1, 50),(0.1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_c1, optimized_beta, optimized_tau = parameters[0], parameters[1], parameters[2]
    return optimized_c1, optimized_beta, optimized_tau, parameters



def find_parameters_load_and_relaxation(c1, datafile, sheet, step, previous_step_values, previous_step_optimized_parameters, minimization_method):
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list_load, time_load, experimental_stress_load = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    elongation_list_relaxation, time_relaxation, experimental_stress_relaxation = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]

    def minimization_function(parameters):
        Q_list_load, S_list_load, S_H_list_load, Pi_list_load = compute_stress_vector_step_fixed_c1_variable_beta_load(c1, parameters, datafile, sheet, step, previous_step_values)
        # least_square_load = mean_squared_error(experimental_stress_load, Pi_list_load)
        load_step_values = Q_list_load[-1], S_list_load[-1], S_H_list_load[-1], Pi_list_load[-1]
        tau = parameters[1]
        Q_list_relaxation, S_list_relaxation, S_H_list_relaxation, Pi_list_relaxation = compute_stress_vector_step_relaxation(tau, datafile, sheet, step, load_step_values)
        experimental_stress_load_and_relaxation = np.concatenate((experimental_stress_load, experimental_stress_relaxation), axis=None)
        Pi_list_load_and_relaxation = np.concatenate((Pi_list_load, Pi_list_relaxation), axis=None)
        # least_square_load_relaxation = mean_squared_error(experimental_stress_load_and_relaxation, Pi_list_load_and_relaxation)
        least_square_load_relaxation = np.linalg.norm((experimental_stress_load_and_relaxation - Pi_list_load_and_relaxation), ord=4)
        least_square = least_square_load_relaxation#least_square_load + least_square_relaxation
        return least_square

    res = minimize(minimization_function, x0=previous_step_optimized_parameters, method=minimization_method, bounds=[(0, 50),(0.1, 100)],
               options={'disp': False}) 
    # print(res.message)
    parameters = res.x
    optimized_beta, optimized_tau = parameters[0], parameters[1]
    return optimized_beta, optimized_tau, parameters



def find_parameters_all_steps_load_and_relaxation(datafile, sheet, minimization_method):
    c1_dict = {}
    beta_dict = {}
    tau_dict = {}
    fitted_stress_dict_load = {}
    fitted_stress_dict_relaxation = {}
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    step_numbers = index_final_step_dict_load.keys()
    previous_step_values_0 = 0, 0, 0, 0
    
    optimized_c1_step_0, optimized_beta_step_0, optimized_tau_step_0, parameters_step_0 = find_parameters_step_0_load(datafile, sheet, previous_step_values_0, minimization_method)
    c1_dict[0] = optimized_c1_step_0
    beta_dict[0] = optimized_beta_step_0
    tau_dict[0] = optimized_tau_step_0
    Q_list_load, S_list_load, S_H_list_load, Pi_list_load = compute_stress_vector_step_0_load(parameters_step_0,  datafile, sheet,previous_step_values_0)
    fitted_stress_dict_load[0] = Pi_list_load
    
    load_step_values = Q_list_load[-1], S_list_load[-1], S_H_list_load[-1], Pi_list_load[-1]
    previous_step_optimized_parameters = [optimized_beta_step_0, optimized_tau_step_0]

    
    # optimized_beta, optimized_tau, parameters = find_parameters_load_and_relaxation(datafile, sheet, 0, previous_step_values_0, previous_step_optimized_parameters, optimized_c1_step_0, minimization_method)
    
    Q_list_relaxation, S_list_relaxation, S_H_list_relaxation, Pi_list_relaxation = compute_stress_vector_step_relaxation(optimized_tau_step_0, datafile, sheet, 0, load_step_values)
    fitted_stress_dict_relaxation[0] = Pi_list_relaxation
    previous_step_values = Q_list_relaxation[-1], S_list_relaxation[-1], S_H_list_relaxation[-1], Pi_list_relaxation[-1]
    previous_step_optimized_parameters = [optimized_beta_step_0, optimized_tau_step_0]
    
    for p in list(step_numbers)[1:]:
        optimized_beta, optimized_tau, parameters = find_parameters_load_and_relaxation(optimized_c1_step_0, datafile, sheet, p, previous_step_values, previous_step_optimized_parameters, minimization_method)
        c1_dict[p] = optimized_c1_step_0
        beta_dict[p] = optimized_beta
        tau_dict[p] = optimized_tau
        Q_list_load, S_list_load, S_H_list_load, Pi_list_load = compute_stress_vector_step_fixed_c1_variable_beta_load(optimized_c1_step_0, parameters,  datafile, sheet, p, previous_step_values)
        fitted_stress_dict_load[p] = Pi_list_load
        load_step_values = Q_list_load[-1], S_list_load[-1], S_H_list_load[-1], Pi_list_load[-1]
        Q_list_relaxation, S_list_relaxation, S_H_list_relaxation, Pi_list_relaxation = compute_stress_vector_step_relaxation(optimized_tau, datafile, sheet, p, load_step_values)
        fitted_stress_dict_relaxation[p] = Pi_list_relaxation
        
        previous_step_values = Q_list_relaxation[-1], S_list_relaxation[-1], S_H_list_relaxation[-1], Pi_list_relaxation[-1]
        previous_step_optimized_parameters = [optimized_beta, optimized_tau]
        
        
    return c1_dict, beta_dict, tau_dict, fitted_stress_dict_load, fitted_stress_dict_relaxation


def compute_and_export_results_variable_c1_variable_beta_load_and_relaxation(datafile, sheet, minimization_method):
    c1_dict, beta_dict, tau_dict, fitted_stress_dict_load, fitted_stress_dict_relaxation = find_parameters_all_steps_load_and_relaxation(datafile, sheet, minimization_method)
    pkl_filename = datafile[0:6] + "_" + sheet + "_" + minimization_method + "_variable_c1_variable_beta_load_and_relaxation.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump([c1_dict, beta_dict, tau_dict, fitted_stress_dict_load, fitted_stress_dict_relaxation],f,)
    print(sheet, ' results to pkl DONE')


def plot_results_variable_c1_variable_beta_load_and_relaxation(datafile, sheet, minimization_method_load):
    _, index_final_step_dict_load, _, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, _, _, _, _, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    c1_dict, beta_dict, tau_dict, fitted_stress_dict_load, fitted_stress_dict_relaxation = find_parameters_all_steps_load_and_relaxation(datafile, sheet, minimization_method_load)
    
    def plot_experimental_vs_fitted_data(datafile, sheet, minimization_method_load):
        # fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
        # ax_elongation_vs_time = fig_elongation_vs_time.gca()
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
            # ax_elongation_vs_time.plot(time_load, elongation_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_load, stress_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_load[:], fitted_stress, '-r', lw=2, alpha=0.8)
            ax_stress_vs_elongation.plot(elongation_load, stress_load, '-k', lw=2, alpha=0.4)
            ax_stress_vs_elongation.plot(elongation_load[:], fitted_stress, '-r', lw=2, alpha=0.8)

            time_relaxation = time_list_during_steps_dict_relaxation[p]
            stress_relaxation = stress_list_during_steps_dict_relaxation[p]
            elongation_relaxation = elongation_list_during_steps_dict_relaxation[p]
            fitted_stress = fitted_stress_dict_relaxation[p]
            # ax_elongation_vs_time.plot(time_relaxation, elongation_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_relaxation, stress_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_time.plot(time_relaxation, fitted_stress, '-b', lw=2, alpha=0.8)
            ax_stress_vs_elongation.plot(elongation_relaxation, stress_relaxation, '-k', lw=2, alpha=0.4)
            ax_stress_vs_elongation.plot(elongation_relaxation, fitted_stress, '-b', lw=2, alpha=0.8)

        ax_stress_vs_time.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        
        # ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        
        # ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
        
        # ax_elongation_vs_time.grid(linestyle=':')
        ax_stress_vs_time.grid(linestyle=':')
        ax_stress_vs_elongation.grid(linestyle=':')
        
        # plt.close(fig_elongation_vs_time)
        plt.close(fig_stress_vs_time)
        plt.close(fig_stress_vs_elongation)
        
        # savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_svg(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_vs_model_load_relaxation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        
     
    def plot_fitted_parameters(datafile, sheet, minimization_method_load):
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
        
        step_list = c1_dict.keys()
        elongation_list = [elongation_final_step_dict_load[s] for s in step_list]
        
        ax_c1_vs_step.plot(step_list, [c1_dict[s] for s in step_list], 'ok')
        ax_beta_vs_step.plot(step_list, [beta_dict[s] for s in step_list], 'or')
        ax_tau_vs_step.plot(step_list, [tau_dict[s] for s in step_list], 'or')
        ax_tau_vs_step.plot(step_list, [tau_dict[s] for s in step_list], 'ob')
        
        ax_c1_vs_step.set_ylabel(r"$c_1$ [kPa]", font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_ylabel(r"$\beta$ [?]", font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_ylabel(r"$\tau$ [s]", font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.plot(elongation_list, [c1_dict[s] for s in step_list], 'ok')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict[s] for s in step_list], 'or')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict[s] for s in step_list], 'or')
        ax_beta_vs_elongation.plot(elongation_list, [beta_dict[s] for s in step_list], 'ob')
        ax_tau_vs_elongation.plot(elongation_list, [tau_dict[s] for s in step_list], 'ob')
        
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

        ax_c1_vs_step.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        ax_beta_vs_step.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        ax_tau_vs_step.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)

        ax_c1_vs_elongation.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        ax_beta_vs_elongation.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        ax_tau_vs_elongation.set_title(minimization_method_load , font=fonts.serif(), fontsize=26)
        

        
        savefigure.save_as_png(fig_c1_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_step_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_png(fig_beta_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_step_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_png(fig_tau_vs_step, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_step_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")

        savefigure.save_as_png(fig_c1_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_c1_vs_load_elongation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_png(fig_beta_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_beta_vs_load_elongation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")
        savefigure.save_as_png(fig_tau_vs_elongation, datafile[0:6] + "_" + sheet + "fitted_params_tau_vs_load_elongation_L_" + minimization_method_load + "_fixedc1_variable_beta_load_and_relaxation")

        plt.close(fig_c1_vs_step)
        plt.close(fig_beta_vs_step)
        plt.close(fig_tau_vs_step)


        plt.close(fig_c1_vs_elongation)
        plt.close(fig_beta_vs_elongation)
        plt.close(fig_tau_vs_elongation)

    plot_experimental_vs_fitted_data(datafile, sheet, minimization_method_load)
    plot_fitted_parameters(datafile, sheet, minimization_method_load)
    

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
    sheets_list_with_data_temp = ["C1PA"]

    for sheet in sheets_list_with_data_temp:
        for minimization_method in minimization_method_load_list:
            plot_results_variable_c1_variable_beta_load_and_relaxation(datafile, sheet, minimization_method)
    print('hello')
