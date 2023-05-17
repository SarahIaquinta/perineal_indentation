import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.zwick.post_processing.read_file as zrf
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from tqdm import tqdm
import pandas as pd
from scipy.signal import argrelextrema


def identify_beginning_recovery(files_zwick, datafile, sheet):
    """
    Identifies the index for which the recovery starts in a sheet of the datafile
    
    Parameters:
        ----------
        datafile: string
            name of the datafile to be read
        sheet: string
            name of the sheet to be read

    Returns:
        -------
        last_max_disp_index: int
            last index before recovery beggins

    """
    _, _, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    metadatafile = files_zwick.import_metadatafile(datafile[0:6])
    imposed_disp_dict, _ = files_zwick.read_metadatas(metadatafile)
    imposed_disp = imposed_disp_dict[sheet]
    disp_as_array = disp.to_numpy()
    disp_as_array[disp_as_array < imposed_disp-0.5] = 0
    max_disp = argrelextrema(np.array(disp_as_array), np.greater)
    last_max_disp_index = max_disp[0][-1]
    return last_max_disp_index
    
def extract_data_during_recovery(files_zwick, datafile, sheet):
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    last_max_disp_index = identify_beginning_recovery(files_zwick, datafile, sheet)
    time_during_recovery = time[last_max_disp_index +1 :]
    force_during_recovery = force[last_max_disp_index +1 :]
    disp_during_recovery = disp[last_max_disp_index +1 :]
    return time_during_recovery, force_during_recovery, disp_during_recovery

def remove_error_pid_from_data(files_zwick, datafile, sheet):
    time_during_recovery, force_during_recovery, disp_during_recovery = extract_data_during_recovery(files_zwick, datafile, sheet)
    last_force = force_during_recovery.to_numpy()[-1]
    correct_pid_indices = np.where(force_during_recovery>0.9*last_force)
    force_during_recovery_correct_pid = [force_during_recovery.to_numpy()[k] for k in correct_pid_indices[0]]
    time_during_recovery_correct_pid = [time_during_recovery.to_numpy()[k] for k in correct_pid_indices[0]]
    disp_during_recovery_correct_pid = [disp_during_recovery.to_numpy()[k] for k in correct_pid_indices[0]]
    return force_during_recovery_correct_pid, time_during_recovery_correct_pid, disp_during_recovery_correct_pid
    
def plot_recovery_data_from_sheet(files_zwick, datafile, sheet, createfigure, savefigure, fonts):
    time_during_recovery, force_during_recovery, disp_during_recovery = extract_data_during_recovery(files_zwick, datafile, sheet)
    fig_force_vs_time_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    fig_disp_vs_time_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    fig_force_vs_disp_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    ax_force_vs_time_during_recovery = fig_force_vs_time_during_recovery.gca()
    ax_disp_vs_time_during_recovery = fig_disp_vs_time_during_recovery.gca()
    ax_force_vs_disp_during_recovery = fig_force_vs_disp_during_recovery.gca()
    date = datafile[0:6]
    metadatafile = files_zwick.import_metadatafile(date)
    imposed_disp_dict, speed_dict = files_zwick.read_metadatas(metadatafile)
    imposed_disp = imposed_disp_dict[sheet]
    imposed_speed = speed_dict[sheet]
    force_during_recovery_correct_pid, time_during_recovery_correct_pid, disp_during_recovery_correct_pid = remove_error_pid_from_data(files_zwick, datafile, sheet)
    kwargs = {"linewidth": 2}
    ax_force_vs_time_during_recovery.plot(time_during_recovery, force_during_recovery, '-', color='k', alpha=0.8, label='RECOVERY \n Umax = ' + str(imposed_disp) + ' mm \n vitesse retour = ' + str(imposed_speed) + 'mm/min', **kwargs)
    ax_disp_vs_time_during_recovery.plot(time_during_recovery, disp_during_recovery, '-', color='k', alpha=0.8, label='RECOVERY \n Umax = ' + str(imposed_disp) + ' mm \n vitesse retour = ' + str(imposed_speed) + 'mm/min', **kwargs)
    ax_force_vs_disp_during_recovery.plot(force_during_recovery, disp_during_recovery, '-', color='k', alpha=0.8, label='RECOVERY \n Umax = ' + str(imposed_disp) + ' mm \n vitesse retour = ' + str(imposed_speed) + 'mm/min', **kwargs)
    ax_force_vs_time_during_recovery.plot(time_during_recovery_correct_pid, force_during_recovery_correct_pid, 'or', markersize=2)
    ax_disp_vs_time_during_recovery.plot(time_during_recovery_correct_pid, disp_during_recovery_correct_pid, 'or', markersize=2)
    ax_force_vs_disp_during_recovery.plot(force_during_recovery_correct_pid, disp_during_recovery_correct_pid, 'or', markersize=2)
    ax_force_vs_time_during_recovery.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time_during_recovery.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp_during_recovery.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time_during_recovery.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time_during_recovery.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp_during_recovery.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time_during_recovery.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
    ax_disp_vs_time_during_recovery.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
    ax_force_vs_disp_during_recovery.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
    savefigure.save_as_png(fig_force_vs_time_during_recovery, sheet + "_force_vs_time_during_recovery")
    savefigure.save_as_png(fig_disp_vs_time_during_recovery, sheet + "_disp_vs_time_during_recovery")
    savefigure.save_as_png(fig_force_vs_disp_during_recovery, sheet + "_force_vs_disp_during_recovery")    
    
    
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230515']#, '230411']#'230331', '230327', '230403']
    types_of_essay = ['C_Indentation_relaxation_maintienFnulle_500N_trav.xls']#, 'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet2 = sheets_list_with_data[1]
    # identify_beginning_recovery(files_zwick, datafile, sheet1)
    for sheet in sheets_list_with_data:
        plot_recovery_data_from_sheet(files_zwick, datafile, sheet, createfigure, savefigure, fonts)
    # error_pid_indices = remove_error_pid_from_data(files_zwick, datafile, sheet2)
    print('hello')
    