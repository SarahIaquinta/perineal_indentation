import numpy as np
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from read_file_load_relaxation_discharge import read_sheet_in_datafile
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pickle
from scipy.signal import lfilter, savgol_filter
from indentation.caracterization.large_tension.post_processing.fit_experimental_data_continuous_parameters import  get_sheets_for_given_pig, get_pig_numbers, get_sheets_for_given_region


def plot_experimental_data_3D(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    beginning_load_phase_indices_list, end_load_phase_indices_list, beginning_relaxation_phase_indices_list, end_relaxation_phase_indices_list, beginning_discharge_phase_indices_list, end_discharge_phase_indices_list = find_peaks_handmade(datafile, sheet)
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    number_of_steps = len(load_phase_time_dict)
    fig = createfigure.rectangle_figure(pixels=180)
    date = datafile[0:6]

    ax = fig.gca()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(time, elongation, stress, c = sns.color_palette("flare", len(elongation)),
                    lw=0, antialiased=False, s = 1 + np.zeros_like(np.array(stress)), alpha = 1)
    ax.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax.set_zlabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    plt.close(fig)
    savefigure.save_as_png(fig, date + "_" + sheet + "_3D_exp")

def plot_stress_vs_elongation_comparison_between_tissue(pig_number, files):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    ax_stress_vs_time = fig_stress_vs_time.gca()
    date = datafile[0:6]
    corresponding_sheets_pig = get_sheets_for_given_pig(pig_number, files, experiment_date)
    for sheet in corresponding_sheets_pig:
        region = sheet[-2]
        color_region = color_dict_region[region]
        label_region = labels_dict[region]
        time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
        ax_stress_vs_elongation.plot(elongation, stress, '-', color=color_region, label = label_region)
        ax_stress_vs_time.plot(time, stress, '-', color=color_region, label = label_region)
        
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    ax_stress_vs_time.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    ax_stress_vs_elongation.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)

    plt.close(fig_stress_vs_time)
    plt.close(fig_stress_vs_elongation)
    
    savefigure.save_as_png(fig_stress_vs_time, date + "_stress_vs_time_comparison_tissues_pig" + pig_number)
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_stress_vs_elongation_comparison_tissues_pig" + pig_number)
    print(sheet, 'done')
    
def plot_stress_vs_elongation_comparison_between_pigs(region, files):
    datafile_list = files.import_files(experiment_date)
    datafile = datafile_list[0]
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    palette_dict = {"P": sns.color_palette("RdPu_d", as_cmap=False, n_colors=10),
                    "S": sns.color_palette("Reds_d", as_cmap=False, n_colors=10),
                    "T": sns.color_palette("Greens_d", as_cmap=False, n_colors=10)}

    ax_stress_vs_time = fig_stress_vs_time.gca()
    date = datafile[0:6]
    corresponding_sheets_region = get_sheets_for_given_region(region, files, experiment_date)
    i=0
    for sheet in corresponding_sheets_region:
        region = sheet[-2]
        palette = sns.color_palette(palette='Dark2')
        # palette = palette_dict[region]
        pig_number = sheet[1:-2] + sheet[-1]
        color_region = color_dict_region[region]
        color_pig = palette[i]
        label_region = labels_dict[region]
        label_pig = "pig " + pig_number
        time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
        ax_stress_vs_elongation.plot(elongation, stress, '-',  color=color_pig, label = label_pig)
        ax_stress_vs_time.plot(time, stress, '-', color=color_pig, label = label_pig)
        i += 1
        
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    ax_stress_vs_time.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    ax_stress_vs_elongation.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)

    plt.close(fig_stress_vs_time)
    plt.close(fig_stress_vs_elongation)
    
    savefigure.save_as_png(fig_stress_vs_time, date + "_stress_vs_time_comparison_pigs_tissue_" + region)
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_stress_vs_elongation_comparison_pigs_tissue_" + region)
    savefigure.save_as_svg(fig_stress_vs_time, date + "_stress_vs_time_comparison_pigs_tissue_" + region)
    savefigure.save_as_svg(fig_stress_vs_elongation, date + "_stress_vs_elongation_comparison_pigs_tissue_" + region)
    print(sheet, 'done')
    
 
def plot_stress_vs_elongation(sheet):
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    kwargs = {"color":'k', "linewidth": 1, "alpha":1}
    color_dict_region={"P": 'm', "S": 'r', "D": 'b', "T": 'g'}
    labels_dict={"P": "peau", "S": "muscle", "D":"d-muscle", "T": "connective tissue"}
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    palette_dict = {"P": sns.color_palette("RdPu_d", as_cmap=False, n_colors=10),
                    "S": sns.color_palette("Reds_d", as_cmap=False, n_colors=10),
                    "T": sns.color_palette("Greens_d", as_cmap=False, n_colors=10)}

    ax_stress_vs_time = fig_stress_vs_time.gca()
    date = datafile[0:6]
    region = sheet[-2]
    palette = sns.color_palette(palette='Dark2')
    # palette = palette_dict[region]
    pig_number = sheet[1:-2] + sheet[-1]
    color_region = color_dict_region[region]
    label_pig = "pig " + pig_number
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    ax_stress_vs_elongation.plot(elongation, stress, '-',  color=color_region, label = label_pig)
    ax_stress_vs_time.plot(time, stress, '-', color=color_region, label = label_pig)
        
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    # ax_stress_vs_time.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    # ax_stress_vs_elongation.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)

    plt.close(fig_stress_vs_time)
    plt.close(fig_stress_vs_elongation)
    
    savefigure.save_as_png(fig_stress_vs_time, date + "_stress_vs_time_raw_" + sheet)
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_stress_vs_elongation_raw_" + sheet)
    print(sheet, 'done')
  
    
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '231012'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    _, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    print('started')
    # for sheet in sheets_list_with_data:
    #     plot_stress_vs_elongation(sheet)
        # plot_experimental_data_3D(datafile, sheet)
    
    

    pig_numbers = ['1', '2', '3']
    regions = ['P']#, 'S', 'T']
    # for pig_number in pig_numbers:
    #     plot_stress_vs_elongation_comparison_between_tissue(pig_number, files_zwick)

    for region  in regions:
        plot_stress_vs_elongation_comparison_between_pigs(region, files_zwick)
    #     plot_integral_differences_comparison_between_pigs(region, files_zwick)
    #     plot_stress_loss_relaxation_comparison_between_pigs(region, files_zwick)
    #     plot_slope_discharge_comparison_between_pigs(region, files_zwick)
    #     plot_hysteresis_comparison_between_pigs(region, files_zwick)
    # time, elongation, stress = read_sheet_in_datafile(datafile, sheet1)
    print('hello')
