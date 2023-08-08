import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit, minimize, rosen, rosen_der
from scipy.integrate import quad
from numba import  prange
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def extract_hyperelastic_response(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    hyperelastic_elongation_indices = np.where(elongation<1.1)
    hyperelastic_elongation = elongation[hyperelastic_elongation_indices]
    hyperelastic_time = time[hyperelastic_elongation_indices]
    hyperelastic_stress = stress[hyperelastic_elongation_indices]
    
    hyperelastic_stress_after_offset_indices = np.where(hyperelastic_stress>1)
    hyperelastic_stress = hyperelastic_stress[hyperelastic_stress_after_offset_indices] - hyperelastic_stress[hyperelastic_stress_after_offset_indices][0]
    hyperelastic_time = hyperelastic_time[hyperelastic_stress_after_offset_indices] - hyperelastic_time[hyperelastic_stress_after_offset_indices][0]
    hyperelastic_elongation = hyperelastic_elongation[hyperelastic_stress_after_offset_indices] - hyperelastic_elongation[hyperelastic_stress_after_offset_indices][0] +1
    return hyperelastic_elongation, hyperelastic_time, hyperelastic_stress
    

    
def plot_hyperelastic_response(datafile, sheet):
    hyperelastic_elongation, hyperelastic_time, hyperelastic_stress = extract_hyperelastic_response(datafile, sheet)
    fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_elongation_vs_time = fig_elongation_vs_time.gca()
    ax_stress_vs_time = fig_stress_vs_time.gca()
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"color":'k', "linewidth": 2}
    ax_elongation_vs_time.plot(hyperelastic_time, hyperelastic_elongation, **kwargs)
    ax_stress_vs_time.plot(hyperelastic_time, hyperelastic_stress, **kwargs)
    ax_stress_vs_elongation.plot(hyperelastic_elongation, hyperelastic_stress, **kwargs)
    
    
    ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.grid(linestyle=':')
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    
    c1, beta, tau, fitted_response = find_parameters(datafile, sheet)
    ax_stress_vs_elongation.plot(hyperelastic_elongation, fitted_response, lw=2, label=r"$\Pi_x^H$ ; $c_1 = $" + str(np.round(c1, 1)) + " kPa \n" + r"$\beta = $" + str(np.round(beta, 1)) + "\n" + r"$\tau$ = " + str(np.round(tau, 1)) + "[s]")
    
    ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
    
    savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_hyperelastic_exp")
    savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_hyperelastic_exp")
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_hyperelastic_exp")
    



def find_parameters(datafile, sheet):
    elongation_list, time, experimental_stress = extract_hyperelastic_response(datafile, sheet)
    def compute_stress_vector(parameters):
        # time, elongation_list, stress = read_sheet_in_datafile(datafile, sheet)
        c1, beta, tau = parameters[0], parameters[1], parameters[2]
        delta_t = np.diff(time)[0]
        i_list = np.arange(0, len(time), 1, dtype=int)
        S_H_list = np.zeros_like(i_list)
        Q_list = np.zeros_like(i_list)
        S_list = np.zeros_like(i_list)
        Pi_list = np.zeros_like(i_list)
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
        return Pi_H_list, Pi_Q_list, Pi_list

        
    def minimization_function(parameters):
        _, _, Pi_list = compute_stress_vector(parameters)
        least_square = mean_squared_error(experimental_stress, Pi_list)
        return least_square

    res = minimize(minimization_function, [20, 0, 30], method='Nelder-Mead',
               options={'gtol': 1e-1, 'disp': True}) 
    print(res.message)
    parameters = res.x
    optimized_c1, optimized_beta, optimized_tau = parameters[0], parameters[1], parameters[2]
    print('c1 = ', optimized_c1, ' beta = ', optimized_beta, ' tau = ', optimized_tau)
    _, _, fitted_function = compute_stress_vector(parameters)
    plt.figure()
    plt.plot(elongation_list, experimental_stress, 'ok', lw=0, label='exp')
    plt.plot(elongation_list, fitted_function, '-b', lw=2, label='num')
    plt.legend()
    plt.show()
    
    
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
    # plot_hyperelastic_response(datafile, sheet1)
    c1=20
    beta=1
    tau=1
    # Pi_H_list, Pi_Q_list, S_list, Pi_list = compute_stress_vector(datafile, sheet1, c1, beta, tau)
    find_parameters(datafile, sheet1)
    print('hello')