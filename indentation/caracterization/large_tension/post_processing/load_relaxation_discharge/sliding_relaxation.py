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
from scipy import integrate
from indentation.caracterization.large_tension.post_processing.fit_experimental_data_continuous_parameters import  get_sheets_for_given_pig, get_pig_numbers, get_sheets_for_given_region


def remove_relaxation_stress(datafile, sheet):
    _, _, _, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = large_tension_utils.extract_data_per_steps(datafile, sheet)
    load_after_sliding_stress_dict = {}
    relaxation_after_sliding_stress_dict = {}
    discharge_after_sliding_stress_dict = {}
    load_after_sliding_elongation_dict = {}
    relaxation_after_sliding_elongation_dict = {}
    discharge_after_sliding_elongation_dict = {}
    loss_stress_during_relaxation = 0
    loss_elongation_during_relaxation = 0
    for i in range(len(load_phase_stress_dict)-1):
        step = int(i+1)
        load_phase_stress_step = load_phase_stress_dict[step]
        load_phase_elongation_step = load_phase_elongation_dict[step]
        
        load_after_sliding_stress = [stress + loss_stress_during_relaxation for stress in load_phase_stress_step]
        load_after_sliding_elongation = [elongation - loss_elongation_during_relaxation for elongation in load_phase_elongation_step]
        
        relaxation_phase_stress_step = relaxation_phase_stress_dict[step]
        relaxation_phase_elongation_step = relaxation_phase_elongation_dict[step]
        discharge_phase_stress_step = discharge_phase_stress_dict[step]
        discharge_phase_elongation_step = discharge_phase_elongation_dict[step]
        stress_beginning_relaxation = relaxation_phase_stress_step[0]
        relaxation_after_sliding_stress_dict[step] = [stress_beginning_relaxation + loss_stress_during_relaxation]
        stress_end_relaxation = relaxation_phase_stress_step[-1]
        loss_stress_during_relaxation_step = stress_beginning_relaxation - stress_end_relaxation
        loss_stress_during_relaxation += loss_stress_during_relaxation_step
        
        elongation_beginning_relaxation = load_phase_elongation_step[-1]
        elongation_end_relaxation = relaxation_phase_elongation_step[-1]
        loss_elongation_during_relaxation_step = elongation_end_relaxation - elongation_beginning_relaxation
        loss_elongation_during_relaxation += loss_elongation_during_relaxation_step
        relaxation_after_sliding_elongation_dict[step] = [elongation_beginning_relaxation - loss_elongation_during_relaxation]
        
        discharge_after_sliding_stress = [stress + loss_stress_during_relaxation for stress in discharge_phase_stress_step]
        load_after_sliding_stress_dict[step] = load_after_sliding_stress
        discharge_after_sliding_stress_dict[step] = discharge_after_sliding_stress    

        discharge_after_sliding_elongation = [elongation - loss_elongation_during_relaxation for elongation in discharge_phase_elongation_step]
        
        load_after_sliding_elongation_dict[step] = load_after_sliding_elongation
        discharge_after_sliding_elongation_dict[step] = discharge_after_sliding_elongation    


    return load_after_sliding_stress_dict, relaxation_after_sliding_stress_dict, discharge_after_sliding_stress_dict, load_after_sliding_elongation_dict, relaxation_after_sliding_elongation_dict, discharge_after_sliding_elongation_dict

def plot_stress_without_relaxation(datafile, sheet):
    load_after_sliding_stress_dict, relaxation_after_sliding_stress_dict, discharge_after_sliding_stress_dict, load_after_sliding_elongation_dict, relaxation_after_sliding_elongation_dict, discharge_after_sliding_elongation_dict = remove_relaxation_stress(datafile, sheet)
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    for i in range(len(load_after_sliding_stress_dict)-1):
        step = int(i+1)
        elongation_during_discharge_step_1 = discharge_after_sliding_elongation_dict[step]
        last_elongation_during_charge_step_2 = elongation_during_discharge_step_1[0]
        last_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[-1]
        stress_during_load_step_1 = load_after_sliding_stress_dict[step]
        elongation_during_load_step_1 = load_after_sliding_elongation_dict[step]
        stress_during_discharge_step_1 = discharge_after_sliding_stress_dict[step]
        stress_of_interest_during_load_step_1_index = np.where(elongation_during_load_step_1 == large_tension_utils.find_nearest(elongation_during_load_step_1, 0.999*last_elongation_during_discharge_step_1))[0][0]
        stress_of_interest_during_load_step_1 = stress_during_load_step_1[stress_of_interest_during_load_step_1_index:]
        elongation_of_interest_during_load_step_1 = elongation_during_load_step_1[stress_of_interest_during_load_step_1_index:]
        stress_during_charge_step_2 = load_after_sliding_stress_dict[step+1]
        elongation_during_charge_step_2 = load_after_sliding_elongation_dict[step+1]
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

 
def compute_integrals_load_mullins_without_relaxation(datafile, sheet):
    load_after_sliding_stress_dict, relaxation_after_sliding_stress_dict, discharge_after_sliding_stress_dict, load_after_sliding_elongation_dict, relaxation_after_sliding_elongation_dict, discharge_after_sliding_elongation_dict = remove_relaxation_stress(datafile, sheet)
    integral_charge_step_1_dict = {}
    integral_charge_step_2_dict = {}
    ratio_integral_charge_step_1_and_2_dict = {}
    
    for i in range(len(load_after_sliding_stress_dict)-1):
        step = int(i+1)
        try:
            elongation_during_discharge_step_1 = discharge_after_sliding_elongation_dict[step]
            last_elongation_during_charge_step_2 = elongation_during_discharge_step_1[0]
            last_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[-1]
            stress_during_load_step_1 = load_after_sliding_stress_dict[step]
            elongation_during_load_step_1 = load_after_sliding_elongation_dict[step]
            stress_of_interest_during_load_step_1_index = np.where(elongation_during_load_step_1 == large_tension_utils.find_nearest(elongation_during_load_step_1, 0.999*last_elongation_during_discharge_step_1))[0][0]
            stress_of_interest_during_load_step_1 = stress_during_load_step_1[stress_of_interest_during_load_step_1_index:]
            elongation_of_interest_during_load_step_1 = elongation_during_load_step_1[stress_of_interest_during_load_step_1_index:]
            integral_charge_step_1 = integrate.trapezoid(stress_of_interest_during_load_step_1, elongation_of_interest_during_load_step_1)
            stress_during_charge_step_2 = load_after_sliding_stress_dict[step+1]
            elongation_during_charge_step_2 = load_after_sliding_elongation_dict[step+1]
            stress_of_interest_during_charge_step_2_index = np.where(elongation_during_charge_step_2 == large_tension_utils.find_nearest(elongation_during_charge_step_2, 0.999*last_elongation_during_charge_step_2))[0][0]
            stress_of_interest_during_load_step_2 = stress_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
            elongation_of_interest_during_load_step_2 = elongation_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
            integral_charge_step_2 = integrate.trapezoid(stress_of_interest_during_load_step_2, elongation_of_interest_during_load_step_2)
            integral_charge_step_1_dict[step] = integral_charge_step_1
            integral_charge_step_2_dict[step] = integral_charge_step_2
            ratio_integral_charge_step_1_and_2_dict[step] = 1 - integral_charge_step_2 / integral_charge_step_1
        except:
            None
    return integral_charge_step_1_dict, integral_charge_step_2_dict, ratio_integral_charge_step_1_and_2_dict

def plot_integral_ratios_comparison_between_tissue(pig_number, files):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    fig_integral_vs_step = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    ax_integral_vs_step = fig_integral_vs_step.gca()
    date = datafile[0:6]
    corresponding_sheets_pig = get_sheets_for_given_pig(pig_number, files, experiment_date)

    for sheet in corresponding_sheets_pig:
        region = sheet[-2]
        color_region = color_dict_region[region]
        label_region = labels_dict[region]
        # try:
        _, _, ratio_integral_charge_step_1_and_2_dict = compute_integrals_load_mullins_without_relaxation(datafile, sheet)
        number_of_steps = len(ratio_integral_charge_step_1_and_2_dict)
        step_list = [int(k) for k in range(1, number_of_steps+1)]
        elongation_list = [1.2 + 0.1*k for k in range(number_of_steps)]
        # step_list = ratio_integral_charge_step_1_and_2_dict.keys()
        ratio_integral_charge_step_1_and_2_list = [ratio_integral_charge_step_1_and_2_dict[step] for step in step_list]
        ax_integral_vs_step.plot(elongation_list, ratio_integral_charge_step_1_and_2_list, '-*', color=color_region, label = label_region)
        # except:
        #     None
    
    ax_integral_vs_step.set_ylabel(r"ratio of integrals (loss of stiffness) [-]", font=fonts.serif(), fontsize=16)
    

    ax_integral_vs_step.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.grid(linestyle=':')
    ax_integral_vs_step.set_ylim((0,0.4))
    ax_integral_vs_step.set_xticks(elongation_list)
    ax_integral_vs_step.set_xticklabels(elongation_list)
    
    ax_integral_vs_step.legend(prop=fonts.serif_1(), loc='upper right', framealpha=0.7,frameon=False)
    plt.close(fig_integral_vs_step)
    savefigure.save_as_png(fig_integral_vs_step, date + "_integral_stress_vs_step_wo_relaxation_comparison_tissues_pig" + pig_number)
        
def plot_integral_ratios_comparison_between_pigs(region, files):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    fig_integral_vs_step = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    # color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    ax_integral_vs_step = fig_integral_vs_step.gca()
    date = datafile[0:6]
    # corresponding_sheets_pig = get_sheets_for_given_pig(pig_number, files, experiment_date)
    corresponding_sheets_region = get_sheets_for_given_region(region, files, experiment_date)

    for sheet in corresponding_sheets_region:
        region = sheet[-2]
        pig_number = sheet[1:-2] + sheet[-1]
        label_pig = "pig " + pig_number
        # try:
        _, _, ratio_integral_charge_step_1_and_2_dict = compute_integrals_load_mullins_without_relaxation(datafile, sheet)
        number_of_steps = len(ratio_integral_charge_step_1_and_2_dict)
        step_list = [int(k) for k in range(1, number_of_steps+1)]
        elongation_list = [1.2 + 0.1*k for k in range(number_of_steps)]
        # step_list = ratio_integral_charge_step_1_and_2_dict.keys()
        ratio_integral_charge_step_1_and_2_list = [ratio_integral_charge_step_1_and_2_dict[step] for step in step_list]
        ax_integral_vs_step.plot(elongation_list, ratio_integral_charge_step_1_and_2_list, '-*', label = label_pig)
        # except:
        #     None
    
    ax_integral_vs_step.set_ylabel(r"ratio of integrals (loss of stiffness) [-]", font=fonts.serif(), fontsize=16)
    

    ax_integral_vs_step.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.grid(linestyle=':')
    
    ax_integral_vs_step.set_ylim((0,0.4))
    ax_integral_vs_step.set_xticks(elongation_list)
    ax_integral_vs_step.set_xticklabels(elongation_list)
    
    ax_integral_vs_step.legend(prop=fonts.serif_1(), loc='upper right', framealpha=0.7,frameon=False)
    plt.close(fig_integral_vs_step)
    savefigure.save_as_png(fig_integral_vs_step, date + "_integral_stress_vs_step_wo_relaxation_comparison_pigs_tissue_" + region)
        


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
    pig_numbers = ['1', '2', '3']
    regions = ['P', 'S', 'T']
    print('started')
    for pig_number in pig_numbers:
        plot_integral_ratios_comparison_between_tissue(pig_number, files_zwick)
    for region  in regions:
        plot_integral_ratios_comparison_between_pigs(region, files_zwick)
        
    # for sheet in sheets_list_with_data:
    #     plot_stress_without_relaxation(datafile, sheet)
    #     print(sheet, ' done')

    print('hello')
    


