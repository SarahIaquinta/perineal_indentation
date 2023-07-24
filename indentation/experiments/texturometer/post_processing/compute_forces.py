"""
This file contains routine aimed at extracting the indicators from the
texturometer testing (the force measured for a displacement equivalent
to 20 % and 80 % of the sample thickness). These indicators are then
plotted in terms of the maturation of the sample on which they have 
been measured.
"""

from matplotlib import pyplot as plt
from math import nan
from indentation.experiments.texturometer.figures.utils import CreateFigure, Fonts, SaveFigure
import pickle
import statistics
import seaborn as sns
import utils

def remove_failed_data(ids_list, date_dict, force20_dict, force80_dict, failed_dict):
    """Removes the data where the experimental acquisition did not go well. 
    The failed acquisition are referenced in a dictionnary.

    Args:
        ids_list (list): list of the testing ids
        date_dict (dict): dictionnary associating the experiment date to each id
        force20_dict (dict): dictionnary associating the force at 20 % to each id
        force80_dict (dict): dictionnary associating the force at 80 % to each id
        failed_dict (dict): dictionnary describing for each id if the acquisition 
                            failed (value is 1) or not (value is 0)

    Returns:
        ids_where_not_failed (list): list of the ids of the testing that did not fail
        date_dict_not_failed (dict): dictionnary associating the experiment date to each unfailed testing id
        force20_dict_not_failed (dict): dictionnary associating the force at 20 % to each unfailed testing id
        force80_dict_not_failed (dict): dictionnary associating the force at 80 % to each unfailed testing id
    """
    ids_where_not_failed = [id for id in ids_list if failed_dict[id] == 0]
    date_dict_not_failed = {id: date_dict[id] for id in ids_where_not_failed}
    force20_dict_not_failed = {id: force20_dict[id] for id in ids_where_not_failed}
    force80_dict_not_failed = {id: force80_dict[id] for id in ids_where_not_failed}
    return ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed

def extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict):
    """Extracts the ids of the testings that have been performed on a given date 
    and on a specific meatpiece (FF or RDG), along with the corresponding values 
    of the indicators.

    Args:
        date (str): experiment date, in the YYMMDD format
        meatpiece (str): meatpiece on which the esperiment has been performed
        ids_list (list): list of the testing ids
        date_dict (dict): dictionnary associating the experiment date to each id
        force20_dict (dict): dictionnary associating the force at 20 % to each id
        force80_dict (dict): dictionnary associating the force at 80 % to each id

    Returns:
        ids_where_not_failed (list): list of the ids of the testing that did not fail
        date_dict_not_failed (dict): dictionnary associating the experiment date to each unfailed testing id
        force20_dict_not_failed (dict): dictionnary associating the force at 20 % to each unfailed testing id
        force80_dict_not_failed (dict): dictionnary associating the force at 80 % to each unfailed testing id
    """
    ids_at_date = [id for id in ids_list if date_dict[id] == date]
    ids_at_date_and_meatpiece = [id for id in ids_at_date if id[0:len(str(date)) + 1 + len(meatpiece)] == str(date) + '_' + meatpiece] 
    force20_dict_at_date_and_meatpiece = {id: force20_dict[id] for id in ids_at_date_and_meatpiece}
    force80_dict_at_date_and_meatpiece = {id: force80_dict[id] for id in ids_at_date_and_meatpiece}
    return ids_at_date_and_meatpiece, force20_dict_at_date_and_meatpiece, force80_dict_at_date_and_meatpiece

def compute_mean_and_std_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict):
    """Computes the mean and the standard deviation of the indicators
    that have been extracted from measurements conducted on the same day (date)
    and on the same meatpiece (FF or RDG)

    Args:
        date (str): experiment date, in the YYMMDD format
        meatpiece (str): meatpiece on which the esperiment has been performed
        ids_list (list): list of the testing ids
        date_dict (dict): dictionnary associating the experiment date to each id
        force20_dict (dict): dictionnary associating the force at 20 % to each id
        force80_dict (dict): dictionnary associating the force at 80 % to each id

    Returns:
        mean_force20 (float): mean value of the force at 20 % indicator
        std_force20 (float): standard deviation of the force at 20 % indicator
        mean_force80 (float): mean value of the force at 80 % indicator
        std_force80 (float): standard deviation of the force at 80 % indicator
    """
    ids_at_date_and_meatpiece, force20_dict_at_date_and_meatpiece, force80_dict_at_date_and_meatpiece = extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, date_dict, force20_dict, force80_dict)
    mean_force20, std_force20, mean_force80, std_force80 = nan, nan, nan, nan
    if ids_at_date_and_meatpiece != []:
        mean_force20 = statistics.mean(list(force20_dict_at_date_and_meatpiece.values()))
        std_force20 = statistics.stdev(list(force20_dict_at_date_and_meatpiece.values()))
        mean_force80 = statistics.mean(list(force80_dict_at_date_and_meatpiece.values()))
        std_force80 = statistics.stdev(list(force80_dict_at_date_and_meatpiece.values()))
    return mean_force20, std_force20, mean_force80, std_force80

def compute_and_export_forces_with_maturation_as_pkl(ids_list, date_dict, force20_dict, force80_dict):
    """Computes the mean and the standard deviation of the indicators
    that have been extracted from measurements conducted on the same day (date)
    and on the same meatpiece (FF or RDG). This operation is repeated for all the 
    dates at which experiments have been conducted in order to 
    investigate the evolution of these indicators with the maturation of the meatpiece.
    Dictionnaries associating each date to the mean and standard deviation of each indicator
    are finally created and exported in a pkl file.

    Args:
        ids_list (list): list of the testing ids
        date_dict (dict): dictionnary associating the experiment date to each id
        force20_dict (dict): dictionnary associating the force at 20 % to each id
        force80_dict (dict): dictionnary associating the force at 80 % to each id

    Returns:
        None
    """
    dates = list(set(date_dict.values()))
    mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict = {}, {}, {}, {}
    mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict = {}, {}, {}, {}
    mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict = {}, {}, {}, {}
    mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict = {}, {}, {}, {}
    mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict = {}, {}, {}, {}
    mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict = {}, {}, {}, {}
    
    for i in range(len(dates)):
        date = dates[i]
        mean_force20_FF1_date, std_force20_FF1_date, mean_force80_FF1_date, std_force80_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF1', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_FF2_date, std_force20_FF2_date, mean_force80_FF2_date, std_force80_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF2', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_RDG1_date, std_force20_RDG1_date, mean_force80_RDG1_date, std_force80_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG1', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_RDG2_date, std_force20_RDG2_date, mean_force80_RDG2_date, std_force80_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG2', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_FF_date, std_force20_FF_date, mean_force80_FF_date, std_force80_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_RDG_date, std_force20_RDG_date, mean_force80_RDG_date, std_force80_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG', ids_list, date_dict, force20_dict, force80_dict)
        mean_force20_FF1_dict[date], std_force20_FF1_dict[date], mean_force80_FF1_dict[date], std_force80_FF1_dict[date] = mean_force20_FF1_date, std_force20_FF1_date, mean_force80_FF1_date, std_force80_FF1_date
        mean_force20_FF2_dict[date], std_force20_FF2_dict[date], mean_force80_FF2_dict[date], std_force80_FF2_dict[date] = mean_force20_FF2_date, std_force20_FF2_date, mean_force80_FF2_date, std_force80_FF2_date
        mean_force20_RDG1_dict[date], std_force20_RDG1_dict[date], mean_force80_RDG1_dict[date], std_force80_RDG1_dict[date] = mean_force20_RDG1_date, std_force20_RDG1_date, mean_force80_RDG1_date, std_force80_RDG1_date
        mean_force20_RDG2_dict[date], std_force20_RDG2_dict[date], mean_force80_RDG2_dict[date], std_force80_RDG2_dict[date] = mean_force20_RDG2_date, std_force20_RDG2_date, mean_force80_RDG2_date, std_force80_RDG2_date
        mean_force20_FF_dict[date], std_force20_FF_dict[date], mean_force80_FF_dict[date], std_force80_FF_dict[date] = mean_force20_FF_date, std_force20_FF_date, mean_force80_FF_date, std_force80_FF_date
        mean_force20_RDG_dict[date], std_force20_RDG_dict[date], mean_force80_RDG_dict[date], std_force80_RDG_dict[date] = mean_force20_RDG_date, std_force20_RDG_date, mean_force80_RDG_date, std_force80_RDG_date

    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ],
            f,
        )
   
def export_forces_as_txt():
    """Creation of a textfile that contains the mean and standard deviation
    of both indicators (force at 20 % and 80 %) at every date. 
    """
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)
    
    complete_txt_filename_FF1 = path_to_processed_data + "/forces_mean_std_FF1.txt"
    f = open(complete_txt_filename_FF1, "w")
    f.write("FORCES FOR FF1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_FF1_dict[date])
            + "\t"
            + str(std_force20_FF1_dict[date])
            + "\t"
            + str(mean_force80_FF1_dict[date])
            + "\t"
            + str(std_force80_FF1_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF2 = path_to_processed_data + "/forces_mean_std_FF2.txt"
    f = open(complete_txt_filename_FF2, "w")
    f.write("FORCES FOR FF2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_FF2_dict[date])
            + "\t"
            + str(std_force20_FF2_dict[date])
            + "\t"
            + str(mean_force80_FF2_dict[date])
            + "\t"
            + str(std_force80_FF2_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF = path_to_processed_data + "/forces_mean_std_FF.txt"
    f = open(complete_txt_filename_FF, "w")
    f.write("FORCES FOR FF \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_FF_dict[date])
            + "\t"
            + str(std_force20_FF_dict[date])
            + "\t"
            + str(mean_force80_FF_dict[date])
            + "\t"
            + str(std_force80_FF_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG1 = path_to_processed_data + "/forces_mean_std_RDG1.txt"
    f = open(complete_txt_filename_RDG1, "w")
    f.write("FORCES FOR RDG1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_RDG1_dict[date])
            + "\t"
            + str(std_force20_RDG1_dict[date])
            + "\t"
            + str(mean_force80_RDG1_dict[date])
            + "\t"
            + str(std_force80_RDG1_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG2 = path_to_processed_data + "/forces_mean_std_RDG2.txt"
    f = open(complete_txt_filename_RDG2, "w")
    f.write("FORCES FOR RDG2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_RDG2_dict[date])
            + "\t"
            + str(std_force20_RDG2_dict[date])
            + "\t"
            + str(mean_force80_RDG2_dict[date])
            + "\t"
            + str(std_force80_RDG2_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG = path_to_processed_data + "/forces_mean_std_RDG.txt"
    f = open(complete_txt_filename_RDG, "w")
    f.write("FORCES FOR RDG \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_RDG_dict[date])
            + "\t"
            + str(std_force20_RDG_dict[date])
            + "\t"
            + str(mean_force80_RDG_dict[date])
            + "\t"
            + str(std_force80_RDG_dict[date])
            + "\n"
        )
    f.close()



    complete_txt_filename_all = path_to_processed_data + "/forces_mean_std.txt"
    f = open(complete_txt_filename_all, "w")
    f.write("FORCES \n")
    f.write("FF1 \t  FF1 \t  FF1 \t  FF1 \t  FF1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_FF1_dict[date])
            + "\t"
            + str(std_force20_FF1_dict[date])
            + "\t"
            + str(mean_force80_FF1_dict[date])
            + "\t"
            + str(std_force80_FF1_dict[date])
            + "\n"
        )

    f.write("FF2 \t  FF2 \t  FF2 \t  FF2 \t  FF2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_FF2_dict[date])
            + "\t"
            + str(std_force20_FF2_dict[date])
            + "\t"
            + str(mean_force80_FF2_dict[date])
            + "\t"
            + str(std_force80_FF2_dict[date])
            + "\n"
        )

    f.write("FF \t  FF \t  FF \t  FF \t  FF \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_FF_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_FF_dict[date])
            + "\t"
            + str(std_force20_FF_dict[date])
            + "\t"
            + str(mean_force80_FF_dict[date])
            + "\t"
            + str(std_force80_FF_dict[date])
            + "\n"
        )

    f.write("RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_RDG1_dict[date])
            + "\t"
            + str(std_force20_RDG1_dict[date])
            + "\t"
            + str(mean_force80_RDG1_dict[date])
            + "\t"
            + str(std_force80_RDG1_dict[date])
            + "\n"
        )

    f.write("RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_RDG2_dict[date])
            + "\t"
            + str(std_force20_RDG2_dict[date])
            + "\t"
            + str(mean_force80_RDG2_dict[date])
            + "\t"
            + str(std_force80_RDG2_dict[date])
            + "\n"
        )

    f.write("RDG \t  RDG \t  RDG \t  RDG \t  RDG \n")
    f.write("date \t mean force20 \t std force20 \t mean force80 \t std force80 \n")
    for i in range(len(mean_force20_RDG_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_force20_RDG_dict[date])
            + "\t"
            + str(std_force20_RDG_dict[date])
            + "\t"
            + str(mean_force80_RDG_dict[date])
            + "\t"
            + str(std_force80_RDG_dict[date])
            + "\n"
        )
        
        
    f.close()

def plot_forces_with_maturation():
    """Plots the evolution of the mean of the force at 20 % and 80 %, along
    with their standard deviation, in terms of the maturation of the meatpiece
    (ie the date on which the experiment has been performed)
    """
    maturation = [7, 10, 13, 17, 21]
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)
    
    dates_to_use = ['230718', '230331', '230403', '230407']
    maturation_dict = {'230718':7, '230331': 10, '230403': 13, '230407': 17}
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    maturation_FF_dict = {k: v - 0.1 for k, v in maturation_dict.items()}
    maturation_RDG_dict = {k: v + 0.1 for k, v in maturation_dict.items()}
    fig_force20_1 = createfigure.rectangle_figure(pixels=180)
    ax_force20_1 = fig_force20_1.gca()
    fig_force20_2 = createfigure.rectangle_figure(pixels=180)
    ax_force20_2 = fig_force20_2.gca()
    fig_force20 = createfigure.rectangle_figure(pixels=180)
    ax_force20 = fig_force20.gca()

    [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = [dates, {d:mean_force20_FF1_dict[d] for d in dates_to_use}, {d:std_force20_FF1_dict[d] for d in dates_to_use}, {d:mean_force80_FF1_dict[d] for d in dates_to_use}, {d:std_force80_FF1_dict[d] for d in dates_to_use},
             {d:mean_force20_FF2_dict[d] for d in dates_to_use}, {d:std_force20_FF2_dict[d] for d in dates_to_use}, {d:mean_force80_FF2_dict[d] for d in dates_to_use}, {d:std_force80_FF2_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG1_dict[d] for d in dates_to_use}, {d:std_force20_RDG1_dict[d] for d in dates_to_use}, {d:mean_force80_RDG1_dict[d] for d in dates_to_use}, {d:std_force80_RDG1_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG2_dict[d] for d in dates_to_use}, {d:std_force20_RDG2_dict[d] for d in dates_to_use}, {d:mean_force80_RDG2_dict[d] for d in dates_to_use}, {d:std_force80_RDG2_dict[d] for d in dates_to_use},
             {d:mean_force20_FF_dict[d] for d in dates_to_use}, {d:std_force20_FF_dict[d] for d in dates_to_use}, {d:mean_force80_FF_dict[d] for d in dates_to_use}, {d:std_force80_FF_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG_dict[d] for d in dates_to_use}, {d:std_force20_RDG_dict[d] for d in dates_to_use}, {d:mean_force80_RDG_dict[d] for d in dates_to_use}, {d:std_force80_RDG_dict[d] for d in dates_to_use}
             ]
    ax_force20.errorbar(list(maturation_FF_dict.values()), list(mean_force20_FF_dict.values()), yerr=list(std_force20_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_force20_1.errorbar(list(maturation_FF_dict.values()), list(mean_force20_FF1_dict.values()), yerr=list(std_force20_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_force20_2.errorbar(list(maturation_FF_dict.values()), list(mean_force20_FF2_dict.values()), yerr=list(std_force20_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_force20_1.errorbar(list(maturation_RDG_dict.values()), list(mean_force20_RDG1_dict.values()), yerr=list(std_force20_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force20.errorbar(list(maturation_RDG_dict.values()), list(mean_force20_RDG_dict.values()), yerr=list(std_force20_RDG_dict.values()), lw=0,  label='RDG', **kwargs_RDG)
    ax_force20_2.errorbar(list(maturation_RDG_dict.values()), list(mean_force20_RDG2_dict.values()), yerr=list(std_force20_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_force20.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force20_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force20_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_force20.set_title('Force vs maturation 1+2', font=fonts.serif_rz_legend())
    # ax_force20_1.set_title('Force vs maturation 1', font=fonts.serif_rz_legend())
    # ax_force20_2.set_title('Force vs maturation 2', font=fonts.serif_rz_legend())
    ax_force20.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_force20_1.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_force20_2.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_force20.set_ylabel(r'$F_{20\%}$ [N]', font=fonts.serif_rz_legend())
    ax_force20_1.set_ylabel(r'$F_{20\%}$ [N]', font=fonts.serif_rz_legend())
    ax_force20_2.set_ylabel(r'$F_{20\%}$ [N]', font=fonts.serif_rz_legend())
    ax_force20.set_xticks([7, 10, 13, 17])
    ax_force20.set_xticklabels(['7?', '10', '13', '17'])
    ax_force20_1.set_xticks([7, 10, 13, 17])
    ax_force20_1.set_xticklabels(['7?', '10', '13', '17'])
    ax_force20_2.set_xticks([7, 10, 13, 17])
    ax_force20_2.set_xticklabels(['7?', '10', '13', '17'])    ###
    savefigure.save_as_png(fig_force20, "force20_vs_maturation_1+2")
    plt.close(fig_force20)
    savefigure.save_as_png(fig_force20_1, "force20_vs_maturation_1")
    plt.close(fig_force20_1)
    savefigure.save_as_png(fig_force20_2, "force20_vs_maturation_2")
    plt.close(fig_force20_2)

    fig_force80_1 = createfigure.rectangle_figure(pixels=180)
    ax_force80_1 = fig_force80_1.gca()
    fig_force80 = createfigure.rectangle_figure(pixels=180)
    ax_force80 = fig_force80.gca()
    fig_force80_2 = createfigure.rectangle_figure(pixels=180)
    ax_force80_2 = fig_force80_2.gca()
    ax_force80_1.plot([maturation_FF_dict[dates_to_use[0]], maturation_FF_dict[dates_to_use[-1]]], [40, 40], '--r', lw=1)
    ax_force80_2.plot([maturation_FF_dict[dates_to_use[0]], maturation_FF_dict[dates_to_use[-1]]], [40, 40], '--r', lw=1)
    # ax_force80.plot([maturation_FF_dict[dates_to_use[0]], maturation_FF_dict[dates_to_use[-1]]], [40, 40], '--r', lw=1)
    ax_force80_1.errorbar(list(maturation_FF_dict.values()), list(mean_force80_FF1_dict.values()), yerr=list(std_force80_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_force80_2.errorbar(list(maturation_FF_dict.values()), list(mean_force80_FF2_dict.values()), yerr=list(std_force80_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_force80.errorbar(list(maturation_FF_dict.values()), list(mean_force80_FF_dict.values()), yerr=list(std_force80_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_force80_1.errorbar(list(maturation_RDG_dict.values()), list(mean_force80_RDG1_dict.values()), yerr=list(std_force80_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force80_2.errorbar(list(maturation_RDG_dict.values()), list(mean_force80_RDG2_dict.values()), yerr=list(std_force80_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_force80.errorbar(list(maturation_RDG_dict.values()), list(mean_force80_RDG_dict.values()), yerr=list(std_force80_RDG_dict.values()), lw=0, label='RDG', **kwargs_RDG)
    ax_force80_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force80_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force80.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    # ax_force80.set_title('Force 80 % vs maturation 1+2', font=fonts.serif_rz_legend())
    # ax_force80_1.set_title('Force 80 % vs maturation 1', font=fonts.serif_rz_legend())
    # ax_force80_2.set_title('Force 80 % vs maturation 2', font=fonts.serif_rz_legend())
    ax_force80.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_force80_1.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_force80_2.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_force80.set_ylabel(r'$F_{80 \%}$ [N]', font=fonts.serif_rz_legend())
    ax_force80_1.set_ylabel(r'$F_{80 \%}$ [N]', font=fonts.serif_rz_legend())
    ax_force80_2.set_ylabel(r'$F_{80 \%}$ [N]', font=fonts.serif_rz_legend())
    ax_force80.set_xticks([7, 10, 13, 17])
    ax_force80.set_xticklabels(['7?', '10', '13', '17'])
    ax_force80_1.set_xticks([7, 10, 13, 17])
    ax_force80_1.set_xticklabels(['7?', '10', '13', '17'])
    ax_force80_2.set_xticks([7, 10, 13, 17])
    ax_force80_2.set_xticklabels(['7?', '10', '13', '17'])    ###
    savefigure.save_as_png(fig_force80, "force80_vs_maturation_1+2")
    plt.close(fig_force80)
    savefigure.save_as_png(fig_force80_1, "force80_vs_maturation_1")
    plt.close(fig_force80_1)
    savefigure.save_as_png(fig_force80_2, "force80_vs_maturation_2")
    plt.close(fig_force80_2)

    
    mean_force8020_FF1_dict = {d:mean_force80_FF1_dict[d] - mean_force20_FF1_dict[d] for d in dates_to_use}
    mean_force8020_FF2_dict = {d:mean_force80_FF2_dict[d] - mean_force20_FF2_dict[d] for d in dates_to_use}
    mean_force8020_FF_dict = {d:mean_force80_FF_dict[d] - mean_force20_FF_dict[d] for d in dates_to_use}
    mean_force8020_RDG1_dict = {d:mean_force80_RDG1_dict[d] - mean_force20_RDG1_dict[d] for d in dates_to_use}
    mean_force8020_RDG2_dict = {d:mean_force80_RDG2_dict[d] - mean_force20_RDG2_dict[d] for d in dates_to_use}
    mean_force8020_RDG_dict = {d:mean_force80_RDG_dict[d] - mean_force20_RDG_dict[d] for d in dates_to_use}

    std_force8020_FF1_dict = {d: max([std_force80_FF1_dict[d], std_force20_FF1_dict[d]]) for d in dates_to_use}
    std_force8020_FF2_dict = {d: max([std_force80_FF2_dict[d], std_force20_FF2_dict[d]]) for d in dates_to_use}
    std_force8020_FF_dict = {d: max([std_force80_FF_dict[d], std_force20_FF_dict[d]]) for d in dates_to_use}
    std_force8020_RDG1_dict = {d: max([std_force80_RDG1_dict[d], std_force20_RDG1_dict[d]]) for d in dates_to_use}
    std_force8020_RDG2_dict = {d: max([std_force80_RDG2_dict[d], std_force20_RDG2_dict[d]]) for d in dates_to_use}
    std_force8020_RDG_dict = {d: max([std_force80_RDG_dict[d], std_force20_RDG_dict[d]]) for d in dates_to_use}
    
    fig_force8020_1 = createfigure.rectangle_figure(pixels=180)
    ax_force8020_1 = fig_force8020_1.gca()
    fig_force8020 = createfigure.rectangle_figure(pixels=180)
    ax_force8020 = fig_force8020.gca()
    fig_force8020_2 = createfigure.rectangle_figure(pixels=180)
    ax_force8020_2 = fig_force8020_2.gca()
    ax_force8020_1.plot([maturation_FF_dict[dates_to_use[0]], maturation_FF_dict[dates_to_use[-1]]], [40, 40], '--r', lw=1)
    ax_force8020_2.plot([maturation_FF_dict[dates_to_use[0]], maturation_FF_dict[dates_to_use[-1]]], [40, 40], '--r', lw=1)
    ax_force8020.plot([maturation_FF_dict[dates_to_use[0]], maturation_FF_dict[dates_to_use[-1]]], [40, 40], '--r', lw=1)
    ax_force8020_1.errorbar(list(maturation_FF_dict.values()), list(mean_force8020_FF1_dict.values()), yerr=list(std_force8020_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_force8020_2.errorbar(list(maturation_FF_dict.values()), list(mean_force8020_FF2_dict.values()), yerr=list(std_force8020_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_force8020.errorbar(list(maturation_FF_dict.values()), list(mean_force8020_FF_dict.values()), yerr=list(std_force8020_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_force8020_1.errorbar(list(maturation_RDG_dict.values()), list(mean_force8020_RDG1_dict.values()), yerr=list(std_force8020_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_force8020_2.errorbar(list(maturation_RDG_dict.values()), list(mean_force8020_RDG2_dict.values()), yerr=list(std_force8020_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_force8020.errorbar(list(maturation_RDG_dict.values()), list(mean_force8020_RDG_dict.values()), yerr=list(std_force8020_RDG_dict.values()), lw=0, label='RDG', **kwargs_RDG)
    ax_force8020_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force8020_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force8020.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_force8020.set_title(r"$F_{80\%} - F_{20\%}$ [N]  vs maturation 1+2", font=fonts.serif_rz_legend())
    ax_force8020_1.set_title(r"$F_{80\%} - F_{20\%}$ [N] vs maturation 1", font=fonts.serif_rz_legend())
    ax_force8020_2.set_title(r"$F_{80\%} - F_{20\%}$ [N] vs maturation 2", font=fonts.serif_rz_legend())
    ax_force8020.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force8020_1.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force8020_2.set_xlabel('Maturation [days]', font=fonts.serif_rz_legend())
    ax_force8020.set_ylabel(r"$F_{80\%} - F_{20\%}$ [N]", font=fonts.serif_rz_legend())
    ax_force8020_1.set_ylabel(r"$F_{80\%} - F_{20\%}$ [N]", font=fonts.serif_rz_legend())
    ax_force8020_2.set_ylabel(r"$F_{80\%} - F_{20\%}$ [N]", font=fonts.serif_rz_legend())
    ax_force8020.set_xticks([7, 10, 13, 17])
    ax_force8020.set_xticklabels(['7?', '10', '13', '17'])
    ax_force8020_1.set_xticks([7, 10, 13, 17])
    ax_force8020_1.set_xticklabels(['7?', '10', '13', '17'])
    ax_force8020_2.set_xticks([7, 10, 13, 17])
    ax_force8020_2.set_xticklabels(['7?', '10', '13', '17'])    ###
    savefigure.save_as_png(fig_force8020, "force8020_vs_maturation_1+2")
    plt.close(fig_force8020)
    savefigure.save_as_png(fig_force8020_1, "force8020_vs_maturation_1")
    plt.close(fig_force8020_1)
    savefigure.save_as_png(fig_force8020_2, "force8020_vs_maturation_2")
    plt.close(fig_force8020_2)

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    # current_path = utils.get_current_path()
    # utils.transform_csv_input_into_pkl('texturometer_forces.csv')
    # ids_list, date_dict, force20_dict, force80_dict, failed_dict = utils.extract_texturometer_data_from_pkl()
    # ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed = remove_failed_data(ids_list, date_dict, force20_dict, force80_dict, failed_dict)
    # mean_force20, std_force20, mean_force80, std_force80 = compute_mean_and_std_at_given_date_and_meatpiece(230327, 'FF', ids_where_not_failed, date_dict_not_failed, force20_dict_not_failed, force80_dict_not_failed)
    # compute_and_export_forces_with_maturation_as_pkl(ids_list, date_dict, force20_dict, force80_dict)
    plot_forces_with_maturation()
    # export_forces_as_txt()
    print('hello')
    
