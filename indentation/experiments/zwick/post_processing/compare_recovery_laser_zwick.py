import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils 
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.zwick.post_processing.read_file as zrf
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import indentation.experiments.zwick.post_processing as zpp
from tqdm import tqdm
import pandas as pd
from scipy.signal import argrelextrema
from indentation.experiments.laser.post_processing.identify_movement import Recovery
from indentation.experiments.laser.post_processing.read_file import Files



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    locations_laser = {
                 "230515_P002": [-12, 0],
                 "230515_P011": [-15, -2]}
    experiment_dates = ['230515']
    pigs = ["P002"]
    failed_laser_acqusitions = ['230515_P002-1']
    n_smooth = 10
    types_of_essay = ['C_Indentation_relaxation_maintienFnulle_500N_trav.xls']#, 'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    ids_where_disp_is_imposed_disp_zwick = zpp.evaluate_repeatability.find_ids_with_same_imposed_displacement_zwick(files_zwick, datafile, imposed_displacement)
    force_during_recovery_correct_pid, time_during_recovery_correct_pid, disp_during_recovery_correct_pid = zpp.identify_force_controlled_recovery.remove_error_pid_from_data(files_zwick, datafile, sheet)
    
    location_keys = [key for key in locations_laser]
    for experiment_date in experiment_dates:
        metadatafile = files_zwick.import_metadatafile(experiment_date)
        for pig in pigs:
            files_pig_laser = Files(pig)
            imposed_disp_dict_pig = files_pig_laser.read_metadatas_laser(metadatafile)
            location_key = experiment_date + '_' + pig
            if location_key in location_keys :
                list_of_pig_files = files_pig_laser.import_files(experiment_date)
                location_at_date_pig = locations_laser[experiment_date + '_' + pig]
                for filename in list_of_pig_files:
                    if filename[:-4] not in failed_laser_acqusitions:
                        recovery_at_date_pig = Recovery(filename, location_at_date_pig)
                        time_vector_at_datepig_laser = recovery_at_date_pig.vec_time
                        recovery_positions_laser = recovery_at_date_pig.compute_recovery_with_time(n_smooth)
            

               
               

