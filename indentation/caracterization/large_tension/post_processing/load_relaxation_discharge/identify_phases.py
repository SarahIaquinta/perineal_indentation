import numpy as np
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
from indentation.caracterization.large_tension.post_processing.load_relaxation_discharge.read_file_load_relaxation_discharge import read_sheet_in_datafile
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import pickle
from scipy.signal import lfilter, savgol_filter

time_beginning_cycle_values_dict = {
    "C1PA" : [0, 23.294, 45.529, 67.853, 90.353],
    "C1SA" : [0, 23.185, 45.4, 67.5, 90, 112.4, 134.4, 156.7, 179, 201.4, 223.9, 246.1, 268.3, 291],
    "C1TA" : [0, 23.6, 46.3, 68.2, 90.5, 113, 135, 157.5, 180, 201.835, 224.354, 247.1, 269],
    "C2SA" : [0, 23.672, 45.7, 68, 90, 112.5, 135, 157, 180, 202, 224.1, 246.3, 268.6, 291.3],
    "C2TA" : [0, 23.185, 45.4, 67.5, 90, 112.4, 134.4, 156.7, 179, 201.4, 223.9, 246.1, 268.3, 291],
    "C3TA" : [0, 23.4, 45.6, 67.8, 90, 112, 134, 156.8, 179, 202, 223, 245.6, 268, 290],
    "C1PB" : [0, 23.4, 45.6, 67.8, 90, 112, 134, 156.8, 179, 202, 223, 245.6, 268, 290],
    "C2PA" : [0, 23.4, 45.6, 67.8, 90, 112, 134, 156.8, 179, 202, 223, 245.6, 268, 290],
    "C3PA" : [0, 23.6, 45.6, 67.8, 90.2, 112.5, 134.8, 157, 179.4, 201.8, 224],
    "C3SA" : [0, 23.4, 45.6, 67.8, 90, 112, 134, 156.8, 179, 202, 223, 245.6, 268, 290]
}



def find_peaks_handmade(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    extrema_indices_list = []
    beginning_load_phase_indices_list = []
    end_load_phase_indices_list = []
    beginning_relaxation_phase_indices_list = []
    end_relaxation_phase_indices_list = []
    beginning_discharge_phase_indices_list = []
    end_discharge_phase_indices_list = []

    time_beginning_cycle_values = time_beginning_cycle_values_dict[sheet]
    i=0
    plt.figure()
    plt.plot(time, elongation, '-k')
    
    while i < len(time_beginning_cycle_values)-1:
    # for i in range(len(time_beginning_cycle_values)-1):
        # time_beginning_cycle_value = time_beginning_cycle_values[i-1]
        # cycle_duration = time_end_cycle_value - time_beginning_cycle_value
        
        # isolate cycle data
        time_beginning_cycle_value = time_beginning_cycle_values[i]
        time_end_cycle_value = time_beginning_cycle_values[i+1]
        index_beginning_cycle = np.where(time == large_tension_utils.find_nearest(time, 0.999*time_beginning_cycle_value))[0][0]
        index_end_cycle = np.where(time == large_tension_utils.find_nearest(time,0.999*time_end_cycle_value))[0][0]
        time_values_during_cycle = time[index_beginning_cycle:index_end_cycle+1]
        stress_values_during_cycle = stress[index_beginning_cycle:index_end_cycle+1]
        
        plt.plot(time_values_during_cycle, stress_values_during_cycle, '-k', lw=2, alpha=0.4)
        # isolate load phase data only
        duration_load_step = 6.7
        if i != 0:
            duration_load_step = 1.5*6.7/2
        time_end_load_phase = time_beginning_cycle_value + duration_load_step
        if i == 0:
            if sheet == "C2TA":
                time_end_load_phase = 6.388
        index_end_load_phase = np.where(time == large_tension_utils.find_nearest(time, 0.999*time_end_load_phase))[0][0]
        time_values_during_load_phase = time[index_beginning_cycle:index_end_load_phase+1]
        stress_values_during_load_phase = stress[index_beginning_cycle:index_end_load_phase+1]
        plt.plot(time_values_during_load_phase, stress_values_during_load_phase, '-r', lw=3, alpha=0.8)

        # isolate relaxation phase data only
        time_end_relaxation_phase = time_end_load_phase + 15 
        index_end_relaxation_phase = np.where(time == large_tension_utils.find_nearest(time, 0.999*time_end_relaxation_phase))[0][0]
        time_values_during_relaxation_phase = time[index_end_load_phase:index_end_relaxation_phase+1]
        stress_values_during_relaxation_phase = stress[index_end_load_phase:index_end_relaxation_phase+1] 
        plt.plot(time_values_during_relaxation_phase, stress_values_during_relaxation_phase, '-b', lw=3, alpha=0.8)

        # isolate discharge phase data only    
        time_end_discharge_phase   = time_end_relaxation_phase + 1.7
        index_end_discharge_phase = np.where(time == large_tension_utils.find_nearest(time, 0.999*time_end_discharge_phase))[0][0]
        time_values_during_discharge_phase = time[index_end_relaxation_phase:index_end_discharge_phase+1]
        stress_values_during_discharge_phase = stress[index_end_relaxation_phase:index_end_discharge_phase+1] 
        plt.plot(time_values_during_discharge_phase, stress_values_during_discharge_phase, '-g', lw=3, alpha=0.8)


        # find local extrema in each phase
        local_maxima_indices_load_phase, local_minima_indices_load_phase = large_tension_utils.find_extrema_in_vector(stress_values_during_load_phase)
        local_maxima_indices_relaxation_phase, local_minima_indices_relaxation_phase = large_tension_utils.find_extrema_in_vector(stress_values_during_relaxation_phase)
        local_maxima_indices_discharge_phase, local_minima_indices_discharge_phase = large_tension_utils.find_extrema_in_vector(stress_values_during_discharge_phase)
        

        # take only extrema of interest for each phase
        
        try:
            beginning_load_phase_index_temp = local_minima_indices_load_phase[0] + index_beginning_cycle
        except:
            beginning_load_phase_index_temp = 0 
        try:
            end_load_phase_index_temp = local_maxima_indices_load_phase[-1] + index_beginning_cycle
        except:
            end_load_phase_index_temp = index_end_load_phase
        try:
            beginning_relaxation_phase_index_temp = local_maxima_indices_relaxation_phase[0] + index_end_load_phase
        except:
            beginning_relaxation_phase_index_temp = index_end_load_phase
        try:
            end_relaxation_phase_index_temp = local_maxima_indices_relaxation_phase[-1] + index_end_load_phase
        except:
            end_relaxation_phase_index_temp = index_end_relaxation_phase
        try:
            beginning_discharge_phase_index_temp = local_maxima_indices_discharge_phase[0] + index_end_relaxation_phase
        except:
            beginning_discharge_phase_index_temp = index_end_relaxation_phase
        try:
            end_discharge_phase_index_temp = local_minima_indices_discharge_phase[-1] + index_end_relaxation_phase
        except:
            end_discharge_phase_index_temp = index_end_discharge_phase
        
        beginning_load_phase_index = max(beginning_load_phase_index_temp, index_beginning_cycle)
        
        end_load_phase_index = max(beginning_relaxation_phase_index_temp, end_load_phase_index_temp)
        beginning_relaxation_phase_index = beginning_relaxation_phase_index_temp
        end_relaxation_phase_index = max(beginning_discharge_phase_index_temp, end_relaxation_phase_index_temp)
        beginning_discharge_phase_index = end_relaxation_phase_index
        
        
        
        end_discharge_phase_index = max(end_discharge_phase_index_temp, index_end_cycle)
        # plt.figure()
        # plt.plot(time_values_during_cycle, elongation_values_during_cycle, '-k')
        plt.plot(time[beginning_load_phase_index], stress[beginning_load_phase_index], 'or')
        plt.plot(time[end_load_phase_index], stress[end_load_phase_index], 'or')
        plt.plot(time[beginning_relaxation_phase_index], stress[beginning_relaxation_phase_index], 'ob')
        plt.plot(time[end_relaxation_phase_index], stress[end_relaxation_phase_index], 'ob')
        plt.plot(time[beginning_discharge_phase_index], stress[beginning_discharge_phase_index], 'og')
        plt.plot(time[end_discharge_phase_index], stress[end_discharge_phase_index], 'og')
        # plt.show()
        # indices, _ = find_peaks(stress_values_during_cycle, threshold=0.01)
        beginning_load_phase_indices_list += [beginning_load_phase_index]
        end_load_phase_indices_list += [end_load_phase_index]
        beginning_relaxation_phase_indices_list += [beginning_relaxation_phase_index]
        end_relaxation_phase_indices_list += [end_relaxation_phase_index]
        beginning_discharge_phase_indices_list += [beginning_discharge_phase_index]
        end_discharge_phase_indices_list += [end_discharge_phase_index]
        
        # extrema_indices_list += [beginning_load_phase_index, end_load_phase_index, beginning_relaxation_phase_index, end_relaxation_phase_index, beginning_discharge_phase_index, end_discharge_phase_index]
        # extrema_indices_list += list(indices + index_beginning_cycle)
        # time_end_cycle_value = time_end_discharge_phase
        time_beginning_cycle_value = time_end_discharge_phase
        time_end_cycle_value = time_beginning_cycle_value + 22.35
        i+=1
        
        # print('end time', time_end_discharge_phase)
    # local_maxima_indices, local_minima_indices = large_tension_utils.find_extrema_in_vector(elongation)

    for k in range(len(beginning_load_phase_indices_list)-1):
        end_discharge_phase_indices_list[k] = beginning_load_phase_indices_list[k+1]
        
    # extrema_indices_list = beginning_load_phase_indices_list + end_load_phase_indices_list + beginning_relaxation_phase_indices_list + end_relaxation_phase_indices_list + beginning_discharge_phase_indices_list + end_discharge_phase_indices_list
    # for k in range(len(beginning_load_phase_indices_list)):
    #     load_phase_indices_list.append(beginning_load_phase_indices_list[k]) 
    #     load_phase_indices_list.append(end_load_phase_indices_list[k]) 
    #     relaxation_phase_indices_list.append(beginning_relaxation_phase_indices_list[k]) 
    #     relaxation_phase_indices_list.append(end_relaxation_phase_indices_list[k]) 
    #     discharge_phase_indices_list.append(beginning_discharge_phase_indices_list[k]) 
    #     discharge_phase_indices_list.append(end_discharge_phase_indices_list[k]) 
        
    return beginning_load_phase_indices_list, end_load_phase_indices_list, beginning_relaxation_phase_indices_list, end_relaxation_phase_indices_list, beginning_discharge_phase_indices_list, end_discharge_phase_indices_list







def gather_data_per_steps(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    beginning_load_phase_indices_list, end_load_phase_indices_list, beginning_relaxation_phase_indices_list, end_relaxation_phase_indices_list, beginning_discharge_phase_indices_list, end_discharge_phase_indices_list = find_peaks_handmade(datafile, sheet)
    load_phase_time_dict = {}
    relaxation_phase_time_dict = {}
    discharge_phase_time_dict = {}
    load_phase_stress_dict = {}
    relaxation_phase_stress_dict = {}
    discharge_phase_stress_dict = {}
    load_phase_elongation_dict = {}
    relaxation_phase_elongation_dict = {}
    discharge_phase_elongation_dict = {}

    number_of_steps = int(len(beginning_load_phase_indices_list))
    for i in range(number_of_steps):
        step = int(i+1)
        # load phase
        beginning_load_phase_step_i_index = beginning_load_phase_indices_list[i]
        end_load_phase_step_i_index = end_load_phase_indices_list[i]
        time_during_load_phase_step_i = time[beginning_load_phase_step_i_index:end_load_phase_step_i_index+1]
        elongation_during_load_phase_step_i = elongation[beginning_load_phase_step_i_index:end_load_phase_step_i_index+1]
        stress_during_load_phase_step_i = stress[beginning_load_phase_step_i_index:end_load_phase_step_i_index+1]
        load_phase_time_dict[step] = time_during_load_phase_step_i
        load_phase_elongation_dict[step] = elongation_during_load_phase_step_i
        load_phase_stress_dict[step] = stress_during_load_phase_step_i
        # relaxation phase
        beginning_relaxation_phase_step_i_index = beginning_relaxation_phase_indices_list[i]
        end_relaxation_phase_step_i_index = end_relaxation_phase_indices_list[i]
        time_during_relaxation_phase_step_i = time[beginning_relaxation_phase_step_i_index:end_relaxation_phase_step_i_index+1]
        elongation_during_relaxation_phase_step_i = elongation[beginning_relaxation_phase_step_i_index:end_relaxation_phase_step_i_index+1]
        stress_during_relaxation_phase_step_i = stress[beginning_relaxation_phase_step_i_index:end_relaxation_phase_step_i_index+1]
        relaxation_phase_time_dict[step] = time_during_relaxation_phase_step_i
        relaxation_phase_elongation_dict[step] = elongation_during_relaxation_phase_step_i
        relaxation_phase_stress_dict[step] = stress_during_relaxation_phase_step_i
        # discharge phase
        beginning_discharge_phase_step_i_index = beginning_discharge_phase_indices_list[i]
        end_discharge_phase_step_i_index = end_discharge_phase_indices_list[i]
        time_during_discharge_phase_step_i = time[beginning_discharge_phase_step_i_index:end_discharge_phase_step_i_index+1]
        elongation_during_discharge_phase_step_i = elongation[beginning_discharge_phase_step_i_index:end_discharge_phase_step_i_index+1]
        stress_during_discharge_phase_step_i = stress[beginning_discharge_phase_step_i_index:end_discharge_phase_step_i_index+1]
        discharge_phase_time_dict[step] = time_during_discharge_phase_step_i
        discharge_phase_elongation_dict[step] = elongation_during_discharge_phase_step_i
        discharge_phase_stress_dict[step] = stress_during_discharge_phase_step_i
    return load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict

def export_data_per_steps(datafile, sheet):
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    pkl_filename = datafile[0:6] + "_" + sheet + "step_data.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump([load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict],f,)
    print(sheet, ' results to pkl DONE')

    
    
    
def plot_experimental_data(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    beginning_load_phase_indices_list, end_load_phase_indices_list, beginning_relaxation_phase_indices_list, end_relaxation_phase_indices_list, beginning_discharge_phase_indices_list, end_discharge_phase_indices_list = find_peaks_handmade(datafile, sheet)
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = gather_data_per_steps(datafile, sheet)
    number_of_steps = len(load_phase_time_dict)
    fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_elongation_vs_time = fig_elongation_vs_time.gca()
    ax_stress_vs_time = fig_stress_vs_time.gca()
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"color":'k', "linewidth": 2, "alpha":0.5}
    ax_elongation_vs_time.plot(time, elongation, **kwargs)
    ax_stress_vs_time.plot(time, stress, **kwargs)
    ax_stress_vs_elongation.plot(elongation, stress, **kwargs)
    
    
    
    
    
    for i in range(number_of_steps):
        beginning_load_phase_step_i_index = beginning_load_phase_indices_list[i]
        end_load_phase_step_i_index = end_load_phase_indices_list[i]
        ax_stress_vs_time.plot([time[beginning_load_phase_step_i_index], time[end_load_phase_step_i_index]] , [stress[beginning_load_phase_step_i_index], stress[end_load_phase_step_i_index]], 'or')
        ax_stress_vs_elongation.plot([elongation[beginning_load_phase_step_i_index], elongation[end_load_phase_step_i_index]] , [stress[beginning_load_phase_step_i_index], stress[end_load_phase_step_i_index]], 'or')
        ax_elongation_vs_time.plot([time[beginning_load_phase_step_i_index], time[end_load_phase_step_i_index]] , [elongation[beginning_load_phase_step_i_index], elongation[end_load_phase_step_i_index]], 'or')
        time_during_load_phase_step_i = load_phase_time_dict[i+1]
        elongation_during_load_phase_step_i = load_phase_elongation_dict[i+1]
        stress_during_load_phase_step_i = load_phase_stress_dict[i+1]

        ax_elongation_vs_time.plot(time_during_load_phase_step_i, elongation_during_load_phase_step_i, '-r', alpha=0.7)
        ax_stress_vs_time.plot(time_during_load_phase_step_i, stress_during_load_phase_step_i, '-r', alpha=0.7)
        ax_stress_vs_elongation.plot(elongation_during_load_phase_step_i, stress_during_load_phase_step_i, '-r', alpha=0.7)

        beginning_relaxation_phase_step_i_index = beginning_relaxation_phase_indices_list[i]
        end_relaxation_phase_step_i_index = end_relaxation_phase_indices_list[i]
        
        ax_stress_vs_time.plot([time[beginning_relaxation_phase_step_i_index], time[end_relaxation_phase_step_i_index]] , [stress[beginning_relaxation_phase_step_i_index], stress[end_relaxation_phase_step_i_index]], 'ob')
        ax_stress_vs_elongation.plot([elongation[beginning_relaxation_phase_step_i_index], elongation[end_relaxation_phase_step_i_index]] , [stress[beginning_relaxation_phase_step_i_index], stress[end_relaxation_phase_step_i_index]], 'ob')
        ax_elongation_vs_time.plot([time[beginning_relaxation_phase_step_i_index], time[end_relaxation_phase_step_i_index]] , [elongation[beginning_relaxation_phase_step_i_index], elongation[end_relaxation_phase_step_i_index]], 'ob')        
        
        time_during_relaxation_phase_step_i = relaxation_phase_time_dict[i+1]
        elongation_during_relaxation_phase_step_i = relaxation_phase_elongation_dict[i+1]
        stress_during_relaxation_phase_step_i = relaxation_phase_stress_dict[i+1]
            
        ax_elongation_vs_time.plot(time_during_relaxation_phase_step_i, elongation_during_relaxation_phase_step_i, '-b', alpha=0.7)
        ax_stress_vs_time.plot(time_during_relaxation_phase_step_i, stress_during_relaxation_phase_step_i, '-b', alpha=0.7)
        ax_stress_vs_elongation.plot(elongation_during_relaxation_phase_step_i, stress_during_relaxation_phase_step_i, '-b', alpha=0.7)
    
    
        beginning_discharge_phase_step_i_index = beginning_discharge_phase_indices_list[i]
        end_discharge_phase_step_i_index = end_discharge_phase_indices_list[i]
        
        ax_stress_vs_time.plot([time[beginning_discharge_phase_step_i_index], time[end_discharge_phase_step_i_index]] , [stress[beginning_discharge_phase_step_i_index], stress[end_discharge_phase_step_i_index]], 'og')
        ax_stress_vs_elongation.plot([elongation[beginning_discharge_phase_step_i_index], elongation[end_discharge_phase_step_i_index]] , [stress[beginning_discharge_phase_step_i_index], stress[end_discharge_phase_step_i_index]], 'og')
        ax_elongation_vs_time.plot([time[beginning_discharge_phase_step_i_index], time[end_discharge_phase_step_i_index]] , [elongation[beginning_discharge_phase_step_i_index], elongation[end_discharge_phase_step_i_index]], 'og')
        
        time_during_discharge_phase_step_i = discharge_phase_time_dict[i+1]
        elongation_during_discharge_phase_step_i = discharge_phase_elongation_dict[i+1]
        stress_during_discharge_phase_step_i = discharge_phase_stress_dict[i+1]
            
        ax_elongation_vs_time.plot(time_during_discharge_phase_step_i, elongation_during_discharge_phase_step_i, '-g', alpha=0.7)
        ax_stress_vs_time.plot(time_during_discharge_phase_step_i, stress_during_discharge_phase_step_i, '-g', alpha=0.7)
        ax_stress_vs_elongation.plot(elongation_during_discharge_phase_step_i, stress_during_discharge_phase_step_i, '-g', alpha=0.7)
    
        
    
    
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
    
    savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp")
    savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp")
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp")
    print(sheet, 'done')
    

    
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_date = '231012'
    files_zwick = Files_Zwick('large_tension_data.xlsx')
    datafile_list = files_zwick.import_files(experiment_date)
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = 'C2TA'#sheets_list_with_data[0]
    print('started')
    # time, elongation, stress = read_sheet_in_datafile(datafile, sheet1)
    # plot_experimental_data(datafile, sheet1)
    for sheet in sheets_list_with_data:
        export_data_per_steps(datafile, sheet)
        plot_experimental_data(datafile, sheet)
    # find_peaks(datafile, sheet1)
    print('hello')