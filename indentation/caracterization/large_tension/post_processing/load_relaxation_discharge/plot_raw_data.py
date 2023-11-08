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
    for sheet in sheets_list_with_data:
        plot_experimental_data_3D(datafile, sheet)
    print('hello')