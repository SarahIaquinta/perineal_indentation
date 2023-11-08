import numpy as np
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def read_sheet_in_datafile(datafile, sheet):
    """
    Extracts the measured time, force and displacement values in a sheet
    
    Parameters:
        ----------
        datafile: string
            name of the datafile to be read
        sheet: string
            name of the sheet to be read

    Returns:
        -------
        time: pandasArray
            list of the time values (in seconds) in the sheet of the datafile
        force: pandasArray
            list of the force values (in Newtons) in the sheet of the datafile
        disp: pandasArray
            list of the displacement values (in mm) in the sheet of the datafile

    """
    date = datafile[0:6]
    path_to_datafile = large_tension_utils.reach_data_path(date) / datafile
    data_in_sheet = pd.read_excel(path_to_datafile, sheet_name=sheet, header=3, names=["s", "elongation", "MPa" ], usecols="A:C", decimal=',') 
    time = data_in_sheet.s
    elongation = data_in_sheet.elongation
    stress = data_in_sheet.MPa
    time, elongation, stress = time.to_numpy(), elongation.to_numpy(), stress.to_numpy() 
    rescaled_elongation = np.array([e/100 + 1 for e in elongation[np.nonzero(time)]])
    rescaled_elongation = np.array([e - rescaled_elongation[0] +1 for e in rescaled_elongation])
    rescaled_stress = np.array([s*1000 - stress[0]*1000 for s in stress[np.nonzero(time)]])
    rescaled_time = time[np.nonzero(time)] - time[np.nonzero(time)][0]
    # time = np.array([t - time[1] for t in time])
    return rescaled_time, rescaled_elongation, rescaled_stress

def find_peaks(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    elongation_step_values = np.arange(1.2, 2.2, 0.1)
    end_load_peak_indices = []
    elongation_end_load_peak = []
    times_end_load_peak = []
    stress_end_load_peak = []

    beginning_load_peak_indices = [0]
    elongation_beginning_load_peak = [1]
    time_beginning_load_peak = [0]
    stress_beginning_load_peak = [0]
    
    for elongation_step in elongation_step_values:
        first_elongation_step_index = np.where(elongation == large_tension_utils.find_nearest(elongation, 0.99*elongation_step))[0][0]
        time_at_first_elongation_step = time[first_elongation_step_index]
        stress_at_first_elongation_step = stress[first_elongation_step_index]
        end_load_peak_indices.append(first_elongation_step_index)
        elongation_end_load_peak.append(elongation[first_elongation_step_index])
        times_end_load_peak.append(time_at_first_elongation_step)
        stress_end_load_peak.append(stress_at_first_elongation_step)
        
    duration_step = np.diff(times_end_load_peak)[1]
    for i in range(len(end_load_peak_indices)):
        time_end_peak = times_end_load_peak[i]
        time_at_beginning_load_peak = time_end_peak + duration_step/2
        beginning_load_peak_index = np.where(time == large_tension_utils.find_nearest(time, 0.999*time_at_beginning_load_peak))[0][0]
        stress_beginning = stress[beginning_load_peak_index]
        time_beginning = time[beginning_load_peak_index]
        elongation_beginning = elongation[beginning_load_peak_index]
        beginning_load_peak_indices.append(beginning_load_peak_index)
        elongation_beginning_load_peak.append(elongation_beginning)
        time_beginning_load_peak.append(time_beginning)
        stress_beginning_load_peak.append(stress_beginning)  
        
    return times_end_load_peak, stress_end_load_peak, elongation_end_load_peak, end_load_peak_indices, time_beginning_load_peak, stress_beginning_load_peak, elongation_beginning_load_peak, beginning_load_peak_indices





def plot_experimental_data(datafile, sheet):
    time, elongation, stress = read_sheet_in_datafile(datafile, sheet)
    # times_at_elongation_steps, stress_at_elongation_steps, elongation_steps = find_end_load_peaks(datafile, sheet)
    times_end_load_peak, stress_end_load_peak, elongation_end_load_peak, _, time_beginning_load_peak, stress_beginning_load_peak, elongation_beginning_load_peak, beginning_load_peak_indices = find_peaks(datafile, sheet)
    fig_elongation_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_elongation_vs_time = fig_elongation_vs_time.gca()
    ax_stress_vs_time = fig_stress_vs_time.gca()
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    date = datafile[0:6]
    kwargs = {"color":'k', "linewidth": 2}
    ax_elongation_vs_time.plot(time, elongation, **kwargs)
    ax_stress_vs_time.plot(time, stress, **kwargs)
    ax_stress_vs_elongation.plot(elongation, stress, **kwargs)
    
    ax_elongation_vs_time.plot(times_end_load_peak, elongation_end_load_peak, 'or')
    ax_stress_vs_time.plot(times_end_load_peak, stress_end_load_peak, 'or')
    ax_stress_vs_elongation.plot(elongation_end_load_peak, stress_end_load_peak, 'or')

    ax_elongation_vs_time.plot(time_beginning_load_peak, elongation_beginning_load_peak, 'ob')
    ax_stress_vs_time.plot(time_beginning_load_peak, stress_beginning_load_peak, 'ob')
    ax_stress_vs_elongation.plot(elongation_beginning_load_peak, stress_beginning_load_peak, 'ob')
    
    ax_elongation_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.set_ylabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_time.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    
    ax_elongation_vs_time.grid(linestyle=':')
    ax_stress_vs_time.grid(linestyle=':')
    ax_stress_vs_elongation.grid(linestyle=':')
    
    
    savefigure.save_as_png(fig_elongation_vs_time, date + "_" + sheet + "_elongation_vs_time_exp")
    savefigure.save_as_png(fig_stress_vs_time, date + "_" + sheet + "_stress_vs_time_exp")
    savefigure.save_as_png(fig_stress_vs_elongation, date + "_" + sheet + "_stress_vs_elongation_exp")
    

     
    
    
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
    # time, elongation, stress = read_sheet_in_datafile(datafile, sheet1)
    # plot_experimental_data(datafile, sheet1)
    for sheet in sheets_list_with_data:
        plot_experimental_data(datafile, sheet)
    # find_peaks(datafile, sheet1)
    print('hello')