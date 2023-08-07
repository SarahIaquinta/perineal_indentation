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
from scipy.optimize import curve_fit
from scipy.integrate import quad
from numba import  prange
from tqdm import tqdm

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
    def integrand_B(t, tau):
        hyperelastic_elongation, hyperelastic_time, _ = extract_hyperelastic_response(datafile, sheet)
        lambda_dot = (hyperelastic_elongation[-1] - hyperelastic_elongation[0]) / (hyperelastic_time[-1] - hyperelastic_time[0])
        elongation = lambda_dot * t
        integrand = np.exp(t/tau)/(elongation**5)
        return integrand
    
    def stress_first_cycle(time, c1, beta, tau):
        hyperelastic_elongation, hyperelastic_time, _ = extract_hyperelastic_response(datafile, sheet)
        lambda_dot = (hyperelastic_elongation[-1] - hyperelastic_elongation[0]) / (hyperelastic_time[-1] - hyperelastic_time[0])
        elongation = lambda_dot * time
        B_vec = []
        time_list = [hyperelastic_time[i] for i in range(5, len(hyperelastic_time), 100)]
        for i in tqdm(prange(len(time_list))):
            t = time_list[i]
            B = quad(integrand_B, hyperelastic_time[5], t, args=(tau))[0]
            B_vec.append(B)
        return elongation*2*c1*(1 - 1/(elongation**4)) + elongation*8*c1*beta*lambda_dot*np.array(B_vec)*np.exp(-time/tau)

    hyperelastic_elongation, hyperelastic_time, hyperelastic_stress = extract_hyperelastic_response(datafile, sheet)
    time_list = [hyperelastic_time[i] for i in range(5, len(hyperelastic_time), 100)]
    stress_list = [hyperelastic_stress[i] for i in range(5, len(hyperelastic_stress), 100)]
    popt, pcov = curve_fit(stress_first_cycle, time_list, stress_list, p0=np.array([20, 0, 30]), bounds=([0, 0 , 10], [100, 60, 100]))
    fitted_response = stress_first_cycle(time_list, *popt)
    (c1, beta, tau) = tuple(popt)
    return c1, beta, tau, fitted_response
        
        
def integrand_B(t, tau,  datafile, sheet):
    hyperelastic_elongation, hyperelastic_time, _ = extract_hyperelastic_response(datafile, sheet)
    lambda_dot = (hyperelastic_elongation[-1] - hyperelastic_elongation[0]) / (hyperelastic_time[-1] - hyperelastic_time[0])
    elongation = lambda_dot * t
    integrand = np.exp(t/tau)/(elongation**5)
    return integrand
    
def stress_first_cycle_test(t, c1, beta, tau, datafile, sheet):
    hyperelastic_elongation, hyperelastic_time, _ = extract_hyperelastic_response(datafile, sheet)
    lambda_dot = (hyperelastic_elongation[-1] - hyperelastic_elongation[0]) / (hyperelastic_time[-1] - hyperelastic_time[0])
    elongation = lambda_dot * t
    B = quad(integrand_B, hyperelastic_time[0]+1, t, args=(tau))[0]
    return elongation*2*c1*(1 - 1/(elongation**4)) + elongation*8*c1*beta*lambda_dot*B*np.exp(-t/tau)  


    
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
    plot_hyperelastic_response(datafile, sheet1)
    # find_parameters(datafile, sheet1)
    print('hello')