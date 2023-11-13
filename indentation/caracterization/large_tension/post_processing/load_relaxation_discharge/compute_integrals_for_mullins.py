import numpy as np
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
from indentation.caracterization.large_tension.post_processing.load_relaxation_discharge.read_file_load_relaxation_discharge import read_sheet_in_datafile
from indentation.caracterization.large_tension.post_processing.load_relaxation_discharge.identify_phases import find_peaks_handmade, gather_data_per_steps
from indentation.caracterization.large_tension.post_processing.fit_experimental_data_continuous_parameters import  get_sheets_for_given_pig, get_pig_numbers, get_sheets_for_given_region
import pandas as pd
import seaborn as sns
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import lfilter, savgol_filter
import pickle 
from scipy import integrate
  
def compute_integrals_load_mullins(datafile, sheet):
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = large_tension_utils.extract_data_per_steps(datafile, sheet)
    integral_charge_step_1_dict = {}
    integral_charge_step_2_dict = {}
    difference_integral_charge_step_1_and_2_dict = {}
    
    for i in range(len(load_phase_time_dict)-1):
        step = int(i+1)
        try:
            elongation_during_discharge_step_1 = discharge_phase_elongation_dict[step]
            last_elongation_during_charge_step_2 = elongation_during_discharge_step_1[0]
            last_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[-1]
            stress_during_load_step_1 = load_phase_stress_dict[step]
            elongation_during_load_step_1 = load_phase_elongation_dict[step]
            stress_of_interest_during_load_step_1_index = np.where(elongation_during_load_step_1 == large_tension_utils.find_nearest(elongation_during_load_step_1, 0.999*last_elongation_during_discharge_step_1))[0][0]
            stress_of_interest_during_load_step_1 = stress_during_load_step_1[stress_of_interest_during_load_step_1_index:]
            elongation_of_interest_during_load_step_1 = elongation_during_load_step_1[stress_of_interest_during_load_step_1_index:]
            integral_charge_step_1 = integrate.trapezoid(stress_of_interest_during_load_step_1, elongation_of_interest_during_load_step_1)
            stress_during_charge_step_2 = load_phase_stress_dict[step+1]
            elongation_during_charge_step_2 = load_phase_elongation_dict[step+1]
            stress_of_interest_during_charge_step_2_index = np.where(elongation_during_charge_step_2 == large_tension_utils.find_nearest(elongation_during_charge_step_2, 0.999*last_elongation_during_charge_step_2))[0][0]
            stress_of_interest_during_load_step_2 = stress_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
            elongation_of_interest_during_load_step_2 = elongation_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
            integral_charge_step_2 = integrate.trapezoid(stress_of_interest_during_load_step_2, elongation_of_interest_during_load_step_2)
            integral_charge_step_1_dict[step] = integral_charge_step_1
            integral_charge_step_2_dict[step] = integral_charge_step_2
            difference_integral_charge_step_1_and_2_dict[step] = integral_charge_step_1 - integral_charge_step_2
        except:
            None
    return integral_charge_step_1_dict, integral_charge_step_2_dict, difference_integral_charge_step_1_and_2_dict

def plot_experimental_data_with_integrals(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    beginning_load_phase_indices_list, end_load_phase_indices_list, beginning_relaxation_phase_indices_list, end_relaxation_phase_indices_list, beginning_discharge_phase_indices_list, end_discharge_phase_indices_list = find_peaks_handmade(datafile, sheet)
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    # number_of_steps = len(load_phase_time_dict)

    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)

    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}

    ax_stress_vs_elongation.plot(elongation, stress, **kwargs)
    # number_of_steps = 9
    for i in range(len(load_phase_time_dict)-1):
        step = int(i+1)
        elongation_during_discharge_step_1 = discharge_phase_elongation_dict[step]
        last_elongation_during_charge_step_2 = elongation_during_discharge_step_1[0]
        last_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[-1]
        stress_during_load_step_1 = load_phase_stress_dict[step]
        elongation_during_load_step_1 = load_phase_elongation_dict[step]
        stress_of_interest_during_load_step_1_index = np.where(elongation_during_load_step_1 == large_tension_utils.find_nearest(elongation_during_load_step_1, 0.999*last_elongation_during_discharge_step_1))[0][0]
        stress_of_interest_during_load_step_1 = stress_during_load_step_1[stress_of_interest_during_load_step_1_index:]
        elongation_of_interest_during_load_step_1 = elongation_during_load_step_1[stress_of_interest_during_load_step_1_index:]
        stress_during_charge_step_2 = load_phase_stress_dict[step+1]
        elongation_during_charge_step_2 = load_phase_elongation_dict[step+1]
        stress_of_interest_during_charge_step_2_index = np.where(elongation_during_charge_step_2 == large_tension_utils.find_nearest(elongation_during_charge_step_2, 0.999*last_elongation_during_charge_step_2))[0][0]
        stress_of_interest_during_load_step_2 = stress_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
        elongation_of_interest_during_load_step_2 = elongation_during_charge_step_2[:stress_of_interest_during_charge_step_2_index]
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
    
    
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp_with_areas")
        
def plot_integrals_values(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    beginning_load_phase_indices_list, end_load_phase_indices_list, beginning_relaxation_phase_indices_list, end_relaxation_phase_indices_list, beginning_discharge_phase_indices_list, end_discharge_phase_indices_list = find_peaks_handmade(datafile, sheet)
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    integral_charge_step_1_dict, integral_charge_step_2_dict, difference_integral_charge_step_1_and_2_dict = compute_integrals_load_mullins(datafile, sheet)
    fig_integral_vs_step = createfigure.rectangle_figure(pixels=180)

    ax_integral_vs_step = fig_integral_vs_step.gca()
    date = datafile[0:6]

    step_list = integral_charge_step_1_dict.keys()
    integral_charge_step_1_list = [integral_charge_step_1_dict[step] for step in step_list]
    integral_charge_step_2_list = [integral_charge_step_2_dict[step] for step in step_list]
    difference_integral_charge_step_1_and_2_list = [difference_integral_charge_step_1_and_2_dict[step] for step in step_list]
    ax_integral_vs_step.plot(step_list, integral_charge_step_1_list, '-or', label="aire charge")
    ax_integral_vs_step.plot(step_list, integral_charge_step_2_list, '-^g', label= "aire recharge")
    ax_integral_vs_step.plot(step_list, difference_integral_charge_step_1_and_2_list, '-*k', label = "différence")
        
    ax_integral_vs_step.set_ylabel(r"$\int(\Pi(\lambda)d\lambda$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.set_xlabel(r"$\lambda$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.grid(linestyle=':')
    
    ax_integral_vs_step.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)

    plt.close(fig_integral_vs_step)
    
    
    savefigure.save_as_png(fig_integral_vs_step, date + "_" + sheet + "_integral_stress_vs_step")

def plot_integral_differences_comparison_between_tissue(pig_number, files):
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
        _, _, difference_integral_charge_step_1_and_2_dict = compute_integrals_load_mullins(datafile, sheet)
        number_of_steps = min(9, len(difference_integral_charge_step_1_and_2_dict))
        step_list = [int(k) for k in range(1, number_of_steps+1)]
        # step_list = difference_integral_charge_step_1_and_2_dict.keys()
        difference_integral_charge_step_1_and_2_list = [difference_integral_charge_step_1_and_2_dict[step] for step in step_list]
        ax_integral_vs_step.plot(step_list, difference_integral_charge_step_1_and_2_list, '-*', color=color_region, label = label_region)
        # except:
        #     None
    
    ax_integral_vs_step.set_ylabel(r"$\int(\Pi(\lambda)d\lambda$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.set_xlabel(r"step", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.grid(linestyle=':')
    
    ax_integral_vs_step.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    plt.close(fig_integral_vs_step)
    savefigure.save_as_png(fig_integral_vs_step, date + "_integral_stress_vs_step_comparison_tissues_pig" + pig_number)
        
def plot_integral_differences_comparison_between_pigs(region, files):
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
        _, _, difference_integral_charge_step_1_and_2_dict = compute_integrals_load_mullins(datafile, sheet)
        number_of_steps = min(9, len(difference_integral_charge_step_1_and_2_dict))
        step_list = [int(k) for k in range(1, number_of_steps+1)]
        # step_list = difference_integral_charge_step_1_and_2_dict.keys()
        difference_integral_charge_step_1_and_2_list = [difference_integral_charge_step_1_and_2_dict[step] for step in step_list]
        ax_integral_vs_step.plot(step_list, difference_integral_charge_step_1_and_2_list, '-*', label = label_pig)
        # except:
        #     None
    
    ax_integral_vs_step.set_ylabel(r"$\int(\Pi(\lambda)d\lambda$ [-]", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.set_xlabel(r"step", font=fonts.serif(), fontsize=26)
    

    ax_integral_vs_step.grid(linestyle=':')
    
    ax_integral_vs_step.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    plt.close(fig_integral_vs_step)
    savefigure.save_as_png(fig_integral_vs_step, date + "_integral_stress_vs_step_comparison_pigs_tissue_" + region)
        
        
def compute_stress_loss_during_relaxation(datafile, sheet):
    _, _, _, _, relaxation_phase_stress_dict, _, _, _, _ = large_tension_utils.extract_data_per_steps(datafile, sheet)
    stress_loss_relaxation_dict = {}
    normalized_stress_loss_relaxation_dict = {}
    
    for i in range(len(relaxation_phase_stress_dict)-1):
        step = int(i+1)
        beginning_stress = relaxation_phase_stress_dict[step][0]
        end_stress = relaxation_phase_stress_dict[step][-1]
        stress_loss = beginning_stress - end_stress
        stress_loss_relaxation_dict[step] = stress_loss
        normalized_stress_loss_relaxation_dict[step] = stress_loss/beginning_stress

    return stress_loss_relaxation_dict, normalized_stress_loss_relaxation_dict

def plot_stress_loss_during_relaxation(datafile, sheet):
    stress_loss_relaxation_dict, normalized_stress_loss_relaxation_dict = compute_stress_loss_during_relaxation(datafile, sheet)
    fig = createfigure.rectangle_figure(pixels=180)
    fig_normalized = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    ax = fig.gca()
    ax_normalized = fig_normalized.gca()
    date = datafile[0:6]
    number_of_steps = min(9, len(stress_loss_relaxation_dict))

    step_list = [int(k) for k in range(1, number_of_steps+1)]
    stress_loss_list = [stress_loss_relaxation_dict[step] for step in step_list]
    normalized_stress_loss_list = [normalized_stress_loss_relaxation_dict[step] for step in step_list]
    region = sheet[-2]
    ax.plot(step_list, stress_loss_list, '-*', color=color_dict_region[region], label = sheet)
    ax_normalized.plot(step_list, normalized_stress_loss_list, '-*', color=color_dict_region[region], label = sheet)
    ax.set_ylabel(r"relaxation stress loss [kPa]", font=fonts.serif(), fontsize=18)
    ax_normalized.set_ylabel(r"normalized relaxation stress loss [-]", font=fonts.serif(), fontsize=18)
    

    ax.set_xlabel(r"step", font=fonts.serif(), fontsize=26)
    ax_normalized.set_xlabel(r"step", font=fonts.serif(), fontsize=26)
    

    ax.grid(linestyle=':')
    ax.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)

    ax_normalized.grid(linestyle=':')
    ax_normalized.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
     
    plt.close(fig)
    savefigure.save_as_png(fig, date + "stress_loss_" + sheet) 
    plt.close(fig_normalized)
    savefigure.save_as_png(fig_normalized, date + "normalized_stress_loss_" + sheet) 


def plot_stress_loss_relaxation_comparison_between_tissue(pig_number, files):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    fig_normalized = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    date = datafile[0:6]
    corresponding_sheets_pig = get_sheets_for_given_pig(pig_number, files, experiment_date)
    ax_normalized_stress_relaxation = fig_normalized.gca()
    
    for sheet in corresponding_sheets_pig:
        region = sheet[-2]
        color_region = color_dict_region[region]
        label_region = labels_dict[region]
        # try:
        stress_loss_relaxation_dict, normalized_stress_loss_relaxation_dict = compute_stress_loss_during_relaxation(datafile, sheet)
        number_of_steps = min(9, len(stress_loss_relaxation_dict))
        step_list = [int(k) for k in range(1, number_of_steps+1)]
        # step_list = difference_integral_charge_step_1_and_2_dict.keys()
        normalized_stress_loss_list = [normalized_stress_loss_relaxation_dict[step] for step in step_list]
        ax_normalized_stress_relaxation.plot(step_list, normalized_stress_loss_list, '-*', color=color_region, label = label_region)
        # except:
        #     None
    
    ax_normalized_stress_relaxation.set_ylabel(r"relaxation stress loss [kPa]", font=fonts.serif(), fontsize=18)
    

    ax_normalized_stress_relaxation.set_xlabel(r"step", font=fonts.serif(), fontsize=26)
    

    ax_normalized_stress_relaxation.grid(linestyle=':')
    
    ax_normalized_stress_relaxation.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    plt.close(fig_normalized)
    savefigure.save_as_png(fig_normalized, date + "_loss_relaxation_stress_vs_step_comparison_tissues_pig" + pig_number)
 
 

def plot_stress_loss_relaxation_comparison_between_pigs(region, files):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    fig_normalized = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    date = datafile[0:6]
    ax_normalized_stress_relaxation = fig_normalized.gca()
    corresponding_sheets_region = get_sheets_for_given_region(region, files, experiment_date)

    for sheet in corresponding_sheets_region:
        region = sheet[-2]
        pig_number = sheet[1:-2] + sheet[-1]
        label_pig = "pig " + pig_number
        
        stress_loss_relaxation_dict, normalized_stress_loss_relaxation_dict = compute_stress_loss_during_relaxation(datafile, sheet)
        number_of_steps = min(9, len(stress_loss_relaxation_dict))
        step_list = [int(k) for k in range(1, number_of_steps+1)]
        normalized_stress_loss_list = [normalized_stress_loss_relaxation_dict[step] for step in step_list]
        ax_normalized_stress_relaxation.plot(step_list, normalized_stress_loss_list, '-*', label = label_pig)
    
    ax_normalized_stress_relaxation.set_ylabel(r"relaxation stress loss [kPa]", font=fonts.serif(), fontsize=18)  

    ax_normalized_stress_relaxation.set_xlabel(r"step", font=fonts.serif(), fontsize=26)

    ax_normalized_stress_relaxation.grid(linestyle=':')
    
    ax_normalized_stress_relaxation.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    plt.close(fig_normalized)
    savefigure.save_as_png(fig_normalized, date + "_loss_relaxation_stress_vs_step_comparison_pig_region" + region)
 
 
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
    # for pig_number in pig_numbers:
    #     plot_integral_differences_comparison_between_tissue(pig_number, files_zwick)
        # plot_stress_loss_relaxation_comparison_between_tissue(pig_number, files_zwick)
    for region  in regions:
    #     plot_integral_differences_comparison_between_pigs(region, files_zwick)
        plot_stress_loss_relaxation_comparison_between_pigs(region, files_zwick)
    # time, elongation, stress = read_sheet_in_datafile(datafile, sheet1)
    # plot_experimental_data(datafile, sheet1)
    # for sheet in sheets_list_with_data:
    #     plot_stress_loss_during_relaxation(datafile, sheet)
    # #     try:
        # plot_experimental_data_with_integrals(datafile, sheet)
        # print(sheet)
            # plot_integrals_values(datafile, sheet)
    #     except:
    #  