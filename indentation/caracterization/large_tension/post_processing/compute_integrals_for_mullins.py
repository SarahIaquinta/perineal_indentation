import numpy as np
import utils
import os
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
# from indentation.caracterization.large_tension.post_processing.read_file_load_relaxation_discharge import .
import pandas as pd
import seaborn as sns
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.signal import lfilter, savgol_filter
import pickle 

def extract_data_per_steps(datafile, sheet):
    pkl_filename = datafile[0:6] + "_" + sheet + "step_data.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\large_tension\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict] = pickle.load(f)
    return load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict
    
def identify_stress_values_of_interest(datafile, sheet):
    load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = extract_data_per_steps(datafile, sheet)
    for i in range(len(load_phase_time_dict)):
        step = int(i+1)
        elongation_during_discharge_step_1 = discharge_phase_elongation_dict[step]
        first_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[0]
        # TODO continue 
        last_elongation_during_discharge_step_1 = elongation_during_discharge_step_1[-1]
        stress_during_load_step_1 = load_phase_stress_dict[step]
        elongation_during_load_step_1 = load_phase_elongation_dict[step]
        stress_of_interest_during_load_step_1_index = np.where(stress_during_load_step_1 == utils.find_nearest(elongation_during_load_step_1, 0.999*last_elongation_during_discharge_step_1))[0][0]
        stress_of_interest_during_load_step_1 = stress_during_load_step_1[stress_of_interest_during_load_step_1_index:]

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
    print('started')
    # time, elongation, stress = read_sheet_in_datafile(datafile, sheet1)
    # plot_experimental_data(datafile, sheet1)
    for sheet in sheets_list_with_data:
        export_data_per_steps(datafile, sheet)
    # find_peaks(datafile, sheet1)
    print('hello')