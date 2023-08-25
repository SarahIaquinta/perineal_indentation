import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.caracterization.large_tension.post_processing.read_file import read_sheet_in_datafile, find_peaks
from indentation.caracterization.large_tension.post_processing.identify_steps import store_and_export_step_data
from indentation.caracterization.large_tension.post_processing.fit_experimental_data import extract_step_data
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
import plotly.graph_objects as go
from matplotlib.widgets import Slider, Button
# Create figure

# Add traces, one for each slider step


beta_init = -0.1
tau1_init = 100
tau2_init = 100
c1_init = 20
damage_init = 0


def adaptive_plot(datafile, sheet, step):
    # time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    index_init_step_dict_load, index_final_step_dict_load, elongation_init_step_dict_load, elongation_final_step_dict_load, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, index_init_step_dict_relaxation, index_final_step_dict_relaxation, elongation_init_step_dict_relaxation, elongation_final_step_dict_relaxation, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)    
    
    elongation_load = elongation_list_during_steps_dict_load[step]
    elongation_relaxation = elongation_list_during_steps_dict_relaxation[step]
    exp_stress_load = stress_list_during_steps_dict_load[step]
    exp_stress_relaxation = stress_list_during_steps_dict_relaxation[step]
    elongation = np.concatenate((elongation_load, elongation_relaxation), axis=None)
    
    time_load = time_list_during_steps_dict_load[step]
    time_relaxation = time_list_during_steps_dict_relaxation[step]
    exp_stress_load = stress_list_during_steps_dict_load[step]
    exp_stress_relaxation = stress_list_during_steps_dict_relaxation[step]
    time = np.concatenate((time_load, time_relaxation), axis=None)
    
    
    def compute_pi_vector_load_relaxation_adaptive(c1, beta, tau1, tau2, damage):
        initial_elongation = elongation_init_step_dict_load[step]
        initial_stress_load = exp_stress_load[0]
        s_h_load_init = 2*c1*(1-(initial_elongation**(-4)))
        q_init = initial_stress_load/initial_elongation - s_h_load_init
        initial_stress_values_load = q_init, initial_stress_load/initial_elongation, s_h_load_init, initial_stress_load
        # exp_stress_relaxation = stress_list_during_steps_dict_relaxation[step]
        q_list_load, s_list_load, s_h_list_load, model_stress_load = compute_stress_vector_load(c1, beta, tau1, damage, datafile, sheet, step, initial_stress_values_load)
        initial_stress_values_relaxation = q_list_load[-1], s_list_load[-1], s_h_list_load[-1], model_stress_load[-1]
        q_list_relaxation, s_list_relaxation, s_h_list_relaxation, model_stress_relaxation = compute_stress_vector_relaxation_constant_tau(tau2, damage, datafile, sheet, step, initial_stress_values_relaxation)
        model_stress_load_relaxation = np.concatenate((model_stress_load, model_stress_relaxation), axis=None)
        return model_stress_load_relaxation

    # def compute_pi_vector_load_adaptive(c1, beta, tau):
    #     initial_elongation = elongation_init_step_dict_load[step]
    #     initial_stress_load = exp_stress_load[0]
    #     s_h_load_init = 2*c1*(1-(initial_elongation**(-4)))
    #     q_init = initial_stress_load/initial_elongation - s_h_load_init
    #     initial_stress_values_load = q_init, initial_stress_load/initial_elongation, s_h_load_init, initial_stress_load
    #     # exp_stress_relaxation = stress_list_during_steps_dict_relaxation[step]
    #     q_list_load, s_list_load, s_h_list_load, model_stress_load = compute_stress_vector_load(c1, beta, tau, datafile, sheet, step, initial_stress_values_load)
    #     # initial_stress_values_relaxation = q_list_load[-1], s_list_load[-1], s_h_list_load[-1], model_stress_load[-1]
    #     # q_list_relaxation, s_list_relaxation, s_h_list_relaxation, model_stress_relaxation = compute_stress_vector_relaxation_constant_tau(tau, datafile, sheet, step, initial_stress_values_relaxation)
    #     # model_stress_load_relaxation = np.concatenate((model_stress_load, model_stress_relaxation), axis=None)
    #     return model_stress_load
    
    fig, ax = plt.subplots()
    line, = ax.plot(time, compute_pi_vector_load_relaxation_adaptive(c1_init, beta_init, tau1_init, tau2_init, damage_init), '-b', lw=2)
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.4, bottom=0.25)
    
    ax.plot(time_load, exp_stress_load, ':k', alpha=0.8)
    ax.plot(time_relaxation, exp_stress_relaxation, ':k', alpha=0.8)
    
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Pi stress [kPa]')
    

    # Make a horizontal slider to control the frequency.
    axc1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    c1_slider = Slider(
        ax=axc1,
        label='c1 [kPa]',
        valmin=0,
        valmax=50,
        valinit=c1_init,
        color='y'
    )
    
    # Make a vertically oriented slider to control the amplitude
    axbeta = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    beta_slider = Slider(
        ax=axbeta,
        label="beta",
        valmin=-10,
        valmax=500,
        valinit=beta_init,
        orientation="vertical",
        color='m'
    )

    # Make a vertically oriented slider to control the amplitude
    axtau1 = fig.add_axes([0.175, 0.25, 0.0225, 0.63])
    tau1_slider = Slider(
        ax=axtau1,
        label="tauL",
        valmin=1,
        valmax=100,
        valinit=tau1_init,
        orientation="vertical",
        color='r'
    )

    # Make a vertically oriented slider to control the amplitude
    axtau2 = fig.add_axes([0.25, 0.25, 0.0225, 0.63])
    tau2_slider = Slider(
        ax=axtau2,
        label="tauR",
        valmin=1,
        valmax=100,
        valinit=tau2_init,
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the amplitude
    axdamage = fig.add_axes([0.325, 0.25, 0.0225, 0.63])
    damage_slider = Slider(
        ax=axdamage,
        label="damage",
        valmin=0,
        valmax=1,
        valinit=damage_init,
        orientation="vertical",
        color='g'
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        line.set_ydata(compute_pi_vector_load_relaxation_adaptive(c1_slider.val, beta_slider.val, tau1_slider.val, tau2_slider.val, damage_slider.val))
        fig.canvas.draw_idle()
        
    # register the update function with each slider
    c1_slider.on_changed(update)
    beta_slider.on_changed(update)
    tau1_slider.on_changed(update)
    tau2_slider.on_changed(update)
    damage_slider.on_changed(update)
    
    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    
    def reset(event):
        c1_slider.reset()
        beta_slider.reset()
        tau1_slider.reset()
        tau2_slider.reset()
        damage_slider.reset()
    button.on_clicked(reset)
    
    plt.show()



def compute_stress_vector_load(c1, beta, tau, damage, datafile, sheet, step, previous_step_values):
    _, _, _, _, elongation_list_during_steps_dict_load, time_list_during_steps_dict_load, stress_list_during_steps_dict_load, _, _, _, _, _, _, _ = extract_step_data(datafile, sheet)
    elongation_list, time, _ = elongation_list_during_steps_dict_load[step], time_list_during_steps_dict_load[step], stress_list_during_steps_dict_load[step]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = previous_step_values
    for i in i_list[1:]:
        lambda_i = elongation_list[i]
        time_i = time[i]
        S_H_i = 2*c1*(1-(lambda_i**(-4)))
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1] + beta*tau/delta_t*(1 - np.exp(-delta_t/tau))*(S_H_i - S_H_list[i-1])
        # Q_i = beta*(S_H_i - S_H_list[i-1]) + np.exp(-delta_t/tau)*Q_list[i-1]
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list


def compute_stress_vector_relaxation_constant_tau(tau, damage, datafile, sheet, step, load_step_values):
    _, _, _, _, _, _, _, _, _, _, _, elongation_list_during_steps_dict_relaxation, time_list_during_steps_dict_relaxation, stress_list_during_steps_dict_relaxation = extract_step_data(datafile, sheet)
    elongation_list, time, _ = elongation_list_during_steps_dict_relaxation[step], time_list_during_steps_dict_relaxation[step], stress_list_during_steps_dict_relaxation[step]
    delta_t = np.diff(time)[0]
    i_list = np.arange(0, len(time), 1, dtype=int)
    S_H_list = np.zeros(len(i_list))
    Q_list = np.zeros_like(S_H_list)
    S_list = np.zeros_like(S_H_list)
    Pi_list = np.zeros_like(S_H_list)
    Q_list[0], S_list[0], S_H_list[0], Pi_list[0] = load_step_values
    for i in i_list[1:]:
        elongation = elongation_list[i]
        time_i = time[i]
        S_H_i = S_H_list[0]
        Q_i = np.exp(-delta_t/tau)*Q_list[i-1]
        S_i = Q_i + S_H_i
        S_H_list[i] = S_H_i 
        Q_list[i] = Q_i 
        S_list[i] = S_i
    Pi_list = np.multiply(elongation_list, S_list)
    return Q_list, S_list, S_H_list, Pi_list


def compute_pi_vector_load_relaxation(c1, beta, tau, datafile, sheet, step, previous_step_values):
    Q_list_load, S_list_load, S_H_list_load, Pi_list_load = compute_stress_vector_load(c1, beta, tau, datafile, sheet, step, previous_step_values)
    previous_step_values = Q_list_load, S_list_load, S_H_list_load, Pi_list_load
    Q_list_relaxation, S_list_relaxation, S_H_list_relaxation, Pi_list_relaxation = compute_stress_vector_relaxation_constant_tau(tau, datafile, sheet, step, previous_step_values)
    pi = np.concatenate((Pi_list_load, Pi_list_relaxation), axis=None)
    return pi, Q_list_relaxation, S_list_relaxation, S_H_list_relaxation, Pi_list_relaxation

if __name__ == "__main__":
    
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '230727'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    
    sheets_list_with_data_temp = ["C1PA"]
    
    for sheet in sheets_list_with_data_temp:
        adaptive_plot(datafile, sheet, 5)
    print('done')