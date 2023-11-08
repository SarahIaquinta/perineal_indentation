import numpy as np
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
from indentation.caracterization.large_tension.post_processing.load_relaxation_discharge.identify_phases import find_peaks_handmade, gather_data_per_steps

import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import lfilter, savgol_filter

def remove_relaxation_stress(datafile, sheet):
    _, _, _, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, _, _, _ = large_tension_utils.extract_data_per_steps(datafile, sheet)
    load_after_sliding_dict = {}
    relaxation_after_sliding_dict = {}
    discharge_after_sliding_dict = {}
    loss_stress_during_relaxation = 0
    for i in range(len(load_phase_stress_dict)-1):
        step = int(i+1)
        load_phase_stress_step = load_phase_stress_dict[step]
        discharge_phase_stress_step = discharge_phase_stress_dict[step]
        load_after_sliding = [stress + loss_stress_during_relaxation for stress in load_phase_stress_step]
        relaxation_phase_stress_step = relaxation_phase_stress_dict[step]
        stress_beginning_relaxation = relaxation_phase_stress_step[0]
        relaxation_after_sliding_dict[step] = [stress_beginning_relaxation + loss_stress_during_relaxation]
        stress_end_relaxation = relaxation_phase_stress_step[-1]
        loss_stress_during_relaxation_step = stress_beginning_relaxation - stress_end_relaxation
        loss_stress_during_relaxation += loss_stress_during_relaxation_step
        discharge_after_sliding = [stress + loss_stress_during_relaxation for stress in discharge_phase_stress_step]
        load_after_sliding_dict[step] = load_after_sliding
        discharge_after_sliding_dict[step] = discharge_after_sliding    
    return load_after_sliding_dict, relaxation_after_sliding_dict, discharge_after_sliding_dict

def plot_stress_without_relaxation(datafile, sheet):
    load_after_sliding_dict, relaxation_after_sliding_dict, discharge_after_sliding_dict = remove_relaxation_stress(datafile, sheet)
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    for i in range(len(load_after_sliding_dict)-1):
        step = int(i+1)
        elongation_during_discharge_step_1 = discharge_phase_elongation_dict[step]
        last_elongation_during_charge_step_2 = elongation_during_discharge_step_1[0]
        last_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[-1]
        stress_during_load_step_1 = load_after_sliding_dict[step]
        elongation_during_load_step_1 = load_phase_elongation_dict[step]
        stress_during_discharge_step_1 = discharge_after_sliding_dict[step]
        elongation_during_discharge_step_1 = discharge_phase_elongation_dict[step]
        stress_of_interest_during_load_step_1_index = np.where(elongation_during_load_step_1 == large_tension_utils.find_nearest(elongation_during_load_step_1, 1.01*last_elongation_during_discharge_step_1))[0][0]
        stress_of_interest_during_load_step_1 = stress_during_load_step_1[stress_of_interest_during_load_step_1_index:]
        elongation_of_interest_during_load_step_1 = elongation_during_load_step_1[stress_of_interest_during_load_step_1_index:]
        stress_during_charge_step_2 = load_after_sliding_dict[step+1]
        elongation_during_charge_step_2 = load_phase_elongation_dict[step+1]
        stress_of_interest_during_charge_step_2_index = np.where(elongation_during_charge_step_2 == large_tension_utils.find_nearest(elongation_during_charge_step_2, 0.999*last_elongation_during_charge_step_2))[0][0]
        stress_of_interest_during_load_step_2 = stress_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
        elongation_of_interest_during_load_step_2 = elongation_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
        
        ax_stress_vs_elongation.plot(elongation_during_load_step_1, stress_during_load_step_1, '-r', lw=1)
        ax_stress_vs_elongation.plot(elongation_during_discharge_step_1, stress_during_discharge_step_1, '-g', lw=1)

        
        ax_stress_vs_elongation.fill_between(
        elongation_of_interest_during_load_step_1, 
        stress_of_interest_during_load_step_1, 
        color= "r",
        alpha= 0.2)
        ax_stress_vs_elongation.fill_between(
        elongation_of_interest_during_load_step_2, 
        stress_of_interest_during_load_step_2, 
        color= "g",
        alpha= 0.4)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    

    ax_stress_vs_elongation.grid(linestyle=':')
    

    plt.close(fig_stress_vs_elongation)
    
    
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_without_relaxation_exp_with_areas")

    

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '231012'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = sheets_list_with_data[0]
    for sheet in sheets_list_with_data:
        plot_stress_without_relaxation(datafile, sheet)

    print('hello')
    


