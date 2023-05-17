import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.zwick.post_processing.read_file as zrf
import indentation.experiments.laser.post_processing.read_file as lrf
import indentation.experiments.zwick.post_processing.identify_force_controlled_recovery as z_id_recovery
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from indentation.experiments.laser.post_processing.read_file import Files
from tqdm import tqdm
import pandas as pd
from scipy.signal import argrelextrema
from indentation.experiments.laser.post_processing.identify_movement import Recovery

def find_ids_with_same_imposed_displacement_laser(files_zwick, datafile, imposed_displacement):
    metadatafile = files_zwick.import_metadatafile(datafile[0:6])
    imposed_disp_dict = lrf.read_metadatas_laser(metadatafile)
    ids_where_disp_is_imposed_disp_laser = [id for id, disp in imposed_disp_dict.items() if disp == int(imposed_displacement)]
    return ids_where_disp_is_imposed_disp_laser

def plot_recovery_data_for_imposed_displacement_laser(files_zwick, datafile, imposed_displacement, createfigure, savefigure, fonts, locations_laser, failed_laser_acqusitions):
    metadatafile = files_zwick.import_metadatafile(datafile[0:6])
    ids_where_disp_is_imposed_disp_laser = find_ids_with_same_imposed_displacement_laser(files_zwick, datafile, imposed_displacement)
    file_list = [i + '.csv' for i in ids_where_disp_is_imposed_disp_laser]
    location_keys = [key for key in locations_laser]
    imposed_disp_dict = lrf.read_metadatas_laser(metadatafile)
    fig_disp_vs_time_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    ax_disp_vs_time_during_recovery = fig_disp_vs_time_during_recovery.gca()
    for file_id in file_list:
        print(file_id)
        location_key = file_id[0:11]
        if location_key in location_keys :
            location_at_date_pig = locations_laser[location_key]
            if file_id[:-4] not in failed_laser_acqusitions:
                recovery_at_date_pig = Recovery(file_id, location_at_date_pig)
                time_vector_at_datepig_laser = recovery_at_date_pig.vec_time
                recovery_positions_laser = recovery_at_date_pig.compute_recovery_with_time(10)
                imposed_disp = imposed_disp_dict[file_id[:-4]]
                ax_disp_vs_time_during_recovery.plot(time_vector_at_datepig_laser/1e6, recovery_positions_laser - min(recovery_positions_laser), '-', alpha=0.6, label= file_id[:-4] + ' ; ' + str(imposed_disp) + ' mm')
                ax_disp_vs_time_during_recovery.legend(prop=fonts.serif_rz_legend(), loc='center right', framealpha=0.7)
                ax_disp_vs_time_during_recovery.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
                ax_disp_vs_time_during_recovery.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        savefigure.save_as_png(fig_disp_vs_time_during_recovery, location_key + "_repetability_recovery_imposed_disp_" + str(imposed_displacement))



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230515']
    pig = 'P002'
    types_of_essay = ['C_Indentation_relaxation_maintienFnulle_500N_trav.xls']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    files_pig_laser = Files(pig)
    imposed_displacement = 5
    failed_laser_acqusitions = ['230515_P002-1']
    locations_laser = {
                "230515_P002": [-12, 0],
                "230515_P011": [-15, -2]}
    plot_recovery_data_for_imposed_displacement_laser(files_zwick, datafile, imposed_displacement, createfigure, savefigure, fonts, locations_laser, failed_laser_acqusitions)