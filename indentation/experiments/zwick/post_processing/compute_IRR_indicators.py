""" 
The objective of this file is to extract and post-process the indicators from
the IRR testing (indentation and relaxation part). Note that the recovery is 
investigated in the "laser" folder.
"""

import numpy as np
from matplotlib import pyplot as plt
from math import nan
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
import pickle
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from indentation.experiments.laser.post_processing.investigate_A import extract_data_at_given_date_and_meatpiece
from indentation.experiments.zwick.post_processing.utils import find_nearest
import seaborn as sns
import statistics
from sklearn.linear_model import LinearRegression

def get_data_at_given_strain_rate(files_zwick, datafile, sheet, strain_rate):
    """Extracts the value of time, disp and force measured when the strain
    reaches a given strain rate.

    Args:
        files_zwick (class object): class that contains the tools to extract and read datafiles
        datafile (str): name of the excel file that contains the testing data
        sheet (str): name of the sxcel sheet to be red. The sheets often have the 
            following format: date_meatpeace_1. example: 200403_FF1_1
        strain_rate (float): amount of the total applied disp 

    Returns:
        time_at_strain_rate (float): time at which the disp equals the disp at strain_rate
        force_at_strain_rate (float): force at which the disp equals the disp at strain_rate
        disp_at_strain_rate (float): disp at strain_rate
    """
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_disp = np.nanmax(disp)
    disp_at_strain_rate = find_nearest(disp, max_disp * strain_rate)
    index_where_disp_is_disp_at_strain_rate = np.where(disp == find_nearest(disp, disp_at_strain_rate))[0]
    force_at_strain_rate = force[index_where_disp_is_disp_at_strain_rate]
    time_at_strain_rate = time[index_where_disp_is_disp_at_strain_rate]
    return time_at_strain_rate, force_at_strain_rate, disp_at_strain_rate

def get_data_at_given_time(files_zwick, datafile, sheet, given_time):
    """Extracts the value of time, disp and force measured when the strain
    reaches a given strain rate.

    Args:
        files_zwick (class object): class that contains the tools to extract and read datafiles
        datafile (str): name of the excel file that contains the testing data
        sheet (str): name of the sxcel sheet to be red. The sheets often have the 
            following format: date_meatpeace_1. example: 200403_FF1_1
        given time (float): experiment time at which the velues of disp and force are to
            be extracted

    Returns:
        time_at_given time (float): given time (used to test the function and for a consistency 
            in the output format with the function get_data_at_given_strain_rate)
        force_at_given time (float): force at time equals given time
        disp_at_given time (float): disp at time equals given time
    """
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    time_given_time = find_nearest(time,  given_time)
    index_where_time_is_time_given_time = np.where(time == find_nearest(time, time_given_time))[0]
    force_given_time = force[index_where_time_is_time_given_time]
    disp_given_time = disp[index_where_time_is_time_given_time]
    return time_given_time, force_given_time, disp_given_time

def compute_indicators_indentation_relaxation(files_zwick, datafile, sheet):
    """Computes the indicators measured during the indentation and relaxation 
    phases, ie the indentation and relaxation slopes and the variation
    of force during relaxation.

    Args:
        files_zwick (class object): class that contains the tools to extract and read datafiles
        datafile (str): name of the excel file that contains the testing data
        sheet (str): name of the sxcel sheet to be red. The sheets often have the 
            following format: date_meatpeace_1. example: 200403_FF1_1

    Returns:
        relaxation_slope (float): slope of the force in terms of time at the very beginning of 
            the relaxation phase (first 0.5 second). The beginning of relaxation is designated by
            the instant where the force reaches the peak force.
        delta_f (float): variation of force during the first 10 seconds of relaxation.
        delta_f_star (float): delta_f, normalized by the peak force
        i_disp_time (float): slope of the force in terms of disp at the very beginning of the indentation
            phase (first second).
        i_time_time (float): slope of the force in terms of time at the very beginning of the indentation
            phase (first second).
    """
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.nanmax(force)
    index_where_force_is_max = np.where(force == max_force)[0]
    time_when_force_is_max = time[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(time == find_nearest(time, time_when_force_is_max+0.5))[0]
    relaxation_slope = (force[index_where_time_is_end_relaxation_slope] - force[index_where_force_is_max ]) / (time[index_where_time_is_end_relaxation_slope] - time[index_where_force_is_max])
    time_when_force_is_max = time[index_where_force_is_max]
    relaxation_duration = 10
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation))[0]
    delta_f = max_force - force[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    time_at_time_0, force_at_time_0, disp_at_time_0 = get_data_at_given_time(files_zwick, datafile, sheet, 0.01)
    time_at_time, force_at_time, disp_at_time = get_data_at_given_time(files_zwick, datafile, sheet, 1)
    i_disp_time = (force_at_time - force_at_time_0) / (disp_at_time - disp_at_time_0)
    i_time_time = (force_at_time - force_at_time_0) / (time_at_time - time_at_time_0)
    return relaxation_slope, delta_f, delta_f_star, i_disp_time, i_time_time

def plot_indicators_indentation_relaxation(files_zwick, datafile, sheet):
    # Create figures
    kwargs = {"color":'k', "linewidth": 3}
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    fig_force_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_time = fig_force_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()
    fig_force_vs_disp = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_disp = fig_force_vs_disp.gca()
    
    # Import data to plot
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    
    # Compute the indicators
    max_force = np.nanmax(force)
    index_where_force_is_max = np.where(force == max_force)[0]
    time_when_force_is_max = time[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(time == find_nearest(time, time_when_force_is_max+0.5))[0]
    relaxation_slope = (force[index_where_time_is_end_relaxation_slope] - force[index_where_force_is_max ]) / (time[index_where_time_is_end_relaxation_slope] - time[index_where_force_is_max])
    time_when_force_is_max = time[index_where_force_is_max]
    relaxation_duration = 10
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation))
    delta_f = max_force - force[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    [time_at_time_0, force_at_time_0] = [0, 0]
    time_at_time, force_at_time, _ = get_data_at_given_time(files_zwick, datafile, sheet, 1) 
    i_time_disp = (force_at_time - force_at_time_0) / (time_at_time - time_at_time_0) 
    
    # Plot construction lines for indicators
    ax_force_vs_time.plot([time[index_where_force_is_max ], time[index_where_time_is_end_relaxation_slope]], [force[index_where_force_is_max ], force[index_where_time_is_end_relaxation_slope]], '-', color = 'r', label = r"$\beta$ = " + str(np.round(relaxation_slope[0], 2)) + r" $N.s^{-1}$", linewidth=3)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation_slope], time[index_where_time_is_end_relaxation_slope]], [force[index_where_force_is_max ], force[index_where_time_is_end_relaxation_slope]], '--', color = 'r', linewidth=2)
    ax_force_vs_time.plot([time[index_where_force_is_max ], time[index_where_time_is_end_relaxation_slope]], [force[index_where_force_is_max ], force[index_where_force_is_max ]], '--', color = 'r', linewidth=2)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]], time[index_where_time_is_end_relaxation[0]]], [force[index_where_time_is_end_relaxation][0], max_force], '-', color = 'g', label = r"$\Delta F$  = " + str(np.round(delta_f[0], 2)) + " N \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)), linewidth=3)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]]-0.2, time[index_where_time_is_end_relaxation[0]]+0.2], [force[index_where_time_is_end_relaxation][0], force[index_where_time_is_end_relaxation][0]], '--', color = 'g', linewidth=2)
    ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]]-0.2, time[index_where_time_is_end_relaxation[0]]+0.2], [max_force, max_force], '--', color = 'g', linewidth=2)
    ax_force_vs_time.plot([0, time_at_time ], [0, force_at_time[0]], '-', color = 'b', label = r"$\alpha$ = " + str(np.round(i_time_disp[0], 2)) + r' $Ns^{-1}$', linewidth=3)
    ax_force_vs_time.plot([time_at_time, time_at_time ], [0, force_at_time[0]], '--', color = 'b', linewidth=2)
    ax_force_vs_time.plot([0, time_at_time ], [0, 0], '--', color = 'b', linewidth=2)
    
    # Plot data
    ax_force_vs_time.plot(time[:index_where_time_is_end_relaxation[0][0]], force[0:index_where_time_is_end_relaxation[0][0]], linestyle=':',  **kwargs)
    ax_disp_vs_time.plot(time[:index_where_time_is_end_relaxation[0][0]], disp[0:index_where_time_is_end_relaxation[0][0]], linestyle=':',  **kwargs)    
    ax_force_vs_disp.plot(disp[0:index_where_time_is_end_relaxation[0][0]], force[:index_where_time_is_end_relaxation[0][0]], linestyle=':',  **kwargs)    
    
    # Specify figure settings (labels, fonts, legend, etc)
    ax_force_vs_time.set_xticks([0, 1, 5, 10, 15])
    ax_force_vs_time.set_xticklabels([0, 1, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_force_vs_time.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax_force_vs_time.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], font=fonts.serif(), fontsize=24)
    ax_force_vs_time.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.set_ylabel(r"force [N]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_xticks([0, 5, 10])
    ax_disp_vs_time.set_xticklabels([0, 5, 10], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_yticks([0, 1, 2, 3, 4, 5])
    ax_disp_vs_time.set_yticklabels([0, 1, 2, 3, 4, 5], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp.set_xticks([0, 1, 2, 3, 4, 5])
    ax_force_vs_disp.set_xticklabels([0, 1, 2, 3, 4, 5], font=fonts.serif(), fontsize=24)
    ax_force_vs_disp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax_force_vs_disp.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], font=fonts.serif(), fontsize=24)
    ax_force_vs_disp.set_xlabel(r"force [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    
    #Save and close figures
    savefigure.save_as_png(fig_force_vs_time, sheet + "_force_vs_time_with_indicators")
    plt.close(fig_force_vs_time)
    savefigure.save_as_png(fig_force_vs_disp, sheet + "_force_vs_disp_with_indicators")
    plt.close(fig_force_vs_disp)
    savefigure.save_as_png(fig_disp_vs_time, sheet + "_disp_vs_time_with_indicators")
    plt.close(fig_disp_vs_time)

def export_indicators(datafile_list):
    """Exports all the indicators of the indentation and relaxation phases
        in a .pkl file named indicators_indentation_relaxation.pkl"

    Args:
        datafile_list (list): list of the files from which the indicators
            are extracted and computed

    Returns:
        None
    """
    ids_list = []
    date_dict = {}
    meat_piece_dict = {}
    relaxation_slope_dict = {}
    delta_f_dict = {}
    delta_f_star_dict = {}
    i_disp_time_dict = {}
    i_time_time_dict = {}
    for datafile in datafile_list:
        date = datafile[0:6]
        correct_sheets_in_data = files_zwick.find_only_correct_sheets_in_datafile(datafile)
        for sheet in correct_sheets_in_data:
            id = date + sheet
            relaxation_slope, delta_f, delta_f_star, i_disp_time, i_time_time = compute_indicators_indentation_relaxation(files_zwick, datafile, sheet)
            ids_list.append(id)
            relaxation_slope_dict[id], delta_f_dict[id], delta_f_star_dict[id], i_disp_time_dict[id], i_time_time_dict[id] = relaxation_slope, delta_f, delta_f_star, i_disp_time, i_time_time
            date_dict[id] = date
            ind = [i for i in range(len(id)) if id[i].isalpha()]
            meat_piece = id[ind[0]:]
            meat_piece_dict[id] = meat_piece
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indicators_indentation_relaxation.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [ids_list, date_dict, meat_piece_dict, relaxation_slope_dict, delta_f_dict, delta_f_star_dict, i_disp_time_dict, i_time_time_dict],
            f,
        )
    return ids_list, date_dict, meat_piece_dict, relaxation_slope_dict, delta_f_dict, delta_f_star_dict, i_disp_time_dict, i_time_time_dict     

def extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, meat_piece_dict, date_dict, data_dict):
    """Finds the value of an indicator for a testing performed on a given date and on
        a given meatpiece

    Args:
        date (str): experiment date (format YYMMDD)
        meatpiece (str): identification of the meat piece (FF or RDG)
        ids_list (list): list of the id of all testings
        meat_piece_dict (dict): dictionnary that associates each testing to the meatpiece
            on which it was performed
        date_dict (dict): dictionnary that associates each testing to the date
            at which it was performed
        data_dict (dict): dictionnary that associates each testing to the valeu of the
            indicator that was computed in this code

    Returns:
        ids_at_date_and_meatpiece (list): list of the ids corresponding to testings performed at a given
            date on a given meatpiece
        data_dict_at_date_and_meatpiece (list): dictonnary that associates the indicators measured for
            the testing ids referenced in the ids_at_date_and_meatpiece list
    """
    ids_at_date = [id for id in ids_list if date_dict[id] == date]
    index_limit = len(meatpiece)
    ids_at_date_and_meatpiece = [id for id in ids_at_date if meat_piece_dict[id][0:index_limit] == meatpiece[0:index_limit]]
    data_list = []
    for i in range(len(ids_at_date_and_meatpiece)):
        data = data_dict[ids_at_date_and_meatpiece[i]]
        if len(data) >1 : #test based on the format of the data object
            data_list.append(data[0][0])
        elif len(data) >0 :
            data_list.append(data[0])
        else:
            data_list.append(data)
    data_dict_at_date_and_meatpiece = {ids_at_date_and_meatpiece[i]: data_list[i] for i in range(len(ids_at_date_and_meatpiece))}
    return ids_at_date_and_meatpiece, data_dict_at_date_and_meatpiece      
        
def compute_mean_and_std_at_given_date_and_meatpiece(date, meatpiece, ids_list, meat_piece_dict, date_dict, data_dict):
    """Computes the mean and the standard deviation of the indicator, extracted from testings 
        performed at a given date and on a given meatpiece 

    Args:
        date (str): experiment date (format YYMMDD)
        meatpiece (str): identification of the meat piece (FF or RDG)
        ids_list (list): list of the id of all testings
        meat_piece_dict (dict): dictionnary that associates each testing to the meatpiece
            on which it was performed
        date_dict (dict): dictionnary that associates each testing to the date
            at which it was performed
        data_dict (dict): dictionnary that associates each testing to the valeu of the
            indicator that was computed in this code

    Returns:
        mean_data (float): mean of the indicator
        std_data (float): standard deviation of the indicator
    """
    ids_at_date_and_meatpiece, data_dict_at_date_and_meatpiece = extract_data_at_given_date_and_meatpiece(date, meatpiece, ids_list, meat_piece_dict, date_dict, data_dict)
    mean_data, std_data = nan, nan
    if len(ids_at_date_and_meatpiece) >1:
        try: #it is not sometimes not possible to evaluate the standard deviation of a list.
             # if it is the case, the std is not computed and a NaN value is returned.
            mean_data = statistics.mean(list(data_dict_at_date_and_meatpiece.values()))
            std_data = statistics.stdev(list(data_dict_at_date_and_meatpiece.values()))
        except:
            pass
    return mean_data, std_data

def compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, data_dict, indicator):
    """Computes the mean and stardard deviation of an indicator for each meatpiece, 
    being only from one side (FF1 or FF2) or gathering both sides (FF = FF1 + FF2), for experiments
    conducted at the same date. This is performed for every experiment date in order to study the
    evolution of these indicators with maturation. The mean and std values are then stored in a dictionnary for 
    each meatpiece, whose keys are the experiment date. These dictionnaries are exported in a .pkl file.
    
    Args:
        ids_list (list): list of the id of all testings
        date_dict (dict): dictionnary that associates each testing to the date
            at which it was performed
        data_dict (dict): dictionnary that associates each testing to the valeu of the
            indicator that was computed in this code
        indicator (str): name of the indicator to be investigated.
            Possible values:
                - relaxation_slope
                - delta_f
                - delta_f_star
                - i_disp_time
                - i_time_time
    """
    dates = list(set(date_dict.values()))
    dates.sort()
    mean_data_FF1_dict, std_data_FF1_dict = {}, {}
    mean_data_FF2_dict, std_data_FF2_dict = {}, {}
    mean_data_RDG1_dict, std_data_RDG1_dict = {}, {}
    mean_data_RDG2_dict, std_data_RDG2_dict = {}, {}
    mean_data_FF_dict, std_data_FF_dict = {}, {}
    mean_data_RDG_dict, std_data_RDG_dict = {}, {}
    
    for i in range(len(dates)):
        date = dates[i]
        mean_data_FF1_date, std_data_FF1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF1', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_FF2_date, std_data_FF2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF2', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_RDG1_date, std_data_RDG1_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG1', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_RDG2_date, std_data_RDG2_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG2', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_FF_date, std_data_FF_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'FF', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_RDG_date, std_data_RDG_date = compute_mean_and_std_at_given_date_and_meatpiece(date, 'RDG', ids_list, meat_piece_dict, date_dict, data_dict)
        mean_data_FF1_dict[date], std_data_FF1_dict[date] = mean_data_FF1_date, std_data_FF1_date
        mean_data_FF2_dict[date], std_data_FF2_dict[date] = mean_data_FF2_date, std_data_FF2_date
        mean_data_RDG1_dict[date], std_data_RDG1_dict[date] = mean_data_RDG1_date, std_data_RDG1_date
        mean_data_RDG2_dict[date], std_data_RDG2_dict[date] = mean_data_RDG2_date, std_data_RDG2_date
        mean_data_FF_dict[date], std_data_FF_dict[date] = mean_data_FF_date, std_data_FF_date
        mean_data_RDG_dict[date], std_data_RDG_dict[date] = mean_data_RDG_date, std_data_RDG_date
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + indicator + ".pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ],
            f,
        )
 
def export_data_as_txt(indicator):
    """Exports the mean and standard deviation of the indicator, for each meatpiece and 
    experiment date, in a .txt file. These values are not computed in this function, 
    they are solely extracted from the .pkl file generated in the function
    compute_and_export_mean_std_data_with_maturation_as_pkl. The creation
    of a txt file is operated in order to have an easilly readable format.

    Args:
        indicator (str): name of the indicator to be investigated.
            Possible values:
                - relaxation_slope
                - delta_f
                - delta_f_star
                - i_disp_time
                - i_time_time
    """
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = pickle.load(f)
    
    complete_txt_filename_FF1 = path_to_processed_data + "/" + indicator + "_mean_std_FF1.txt"
    f = open(complete_txt_filename_FF1, "w")
    f.write(indicator + "FOR FF1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_FF1_dict[date])
            + "\t"
            + str(std_data_FF1_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF2 = path_to_processed_data + "/" + indicator + "_mean_std_FF2.txt"
    f = open(complete_txt_filename_FF2, "w")
    f.write(indicator +  "FOR FF2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_FF2_dict[date])
            + "\t"
            + str(std_data_FF2_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_FF = path_to_processed_data + "/" + indicator + "_mean_std_FF.txt"
    f = open(complete_txt_filename_FF, "w")
    f.write(indicator +  "FOR FF \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_FF_dict[date])
            + "\t"
            + str(std_data_FF_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG1 = path_to_processed_data + "/" + indicator + "_mean_std_RDG1.txt"
    f = open(complete_txt_filename_RDG1, "w")
    f.write(indicator +  "FOR RDG1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_RDG1_dict[date])
            + "\t"
            + str(std_data_RDG1_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG2 = path_to_processed_data + "/" + indicator + "_mean_std_RDG2.txt"
    f = open(complete_txt_filename_RDG2, "w")
    f.write(indicator +  "FOR RDG2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_RDG2_dict[date])
            + "\t"
            + str(std_data_RDG2_dict[date])
            + "\n"
        )
    f.close()

    complete_txt_filename_RDG = path_to_processed_data + "/" + indicator + "_mean_std_RDG.txt"
    f = open(complete_txt_filename_RDG, "w")
    f.write(indicator +  "FOR RDG \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_RDG_dict[date])
            + "\t"
            + str(std_data_RDG_dict[date])
            + "\n"
        )
    f.close()



    complete_txt_filename_all = path_to_processed_data + "/" + indicator + "_mean_std.txt"
    f = open(complete_txt_filename_all, "w")
    f.write(indicator +  "\n")
    f.write("FF1 \t  FF1 \t  FF1 \t  FF1 \t  FF1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_FF1_dict[date])
            + "\t"
            + str(std_data_FF1_dict[date])
            + "\n"
        )

    f.write("FF2 \t  FF2 \t  FF2 \t  FF2 \t  FF2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_FF2_dict[date])
            + "\t"
            + str(std_data_FF2_dict[date])
            + "\n"
        )

    f.write("FF \t  FF \t  FF \t  FF \t  FF \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_FF_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_FF_dict[date])
            + "\t"
            + str(std_data_FF_dict[date])
            + "\n"
        )

    f.write("RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \t  RDG1 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG1_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_RDG1_dict[date])
            + "\t"
            + str(std_data_RDG1_dict[date])
            + "\n"
        )

    f.write("RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \t  RDG2 \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG2_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_RDG2_dict[date])
            + "\t"
            + str(std_data_RDG2_dict[date])
            + "\n"
        )

    f.write("RDG \t  RDG \t  RDG \t  RDG \t  RDG \n")
    f.write("date \t mean \t std \n")
    for i in range(len(mean_data_RDG_dict)):
        date = dates[i]
        f.write(
            str(dates[i])
            + "\t"
            + str(mean_data_RDG_dict[date])
            + "\t"
            + str(std_data_RDG_dict[date])
            + "\n"
        )
        
        
    f.close()

def plot_data_with_maturation(indicator):
    """Plots the evolution of the indicator in terms of the maturation
    (i.e. experiment dates)

    Args:
        indicator (str): name of the indicator to be investigated.
            Possible values:
                - relaxation_slope
                - delta_f
                - delta_f_star
                - i_disp_time
                - i_time_time
    """
    
    # Load data to be plotted
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [date, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = pickle.load(f)
        
    # Defines plot settings
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    labels = {'relaxation_slope' : r"$\beta$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_time': r"$\alpha $ [$Nmm^{-1}$]",
                    'i_time_time': r"$\alpha $ [$Ns^{-1}$]"   }
    
    # Created the figure objects
    fig_data_1 = createfigure.rectangle_rz_figure(pixels=180)
    ax_data_1 = fig_data_1.gca()
    fig_data = createfigure.rectangle_rz_figure(pixels=180)
    ax_data = fig_data.gca()
    fig_data_2 = createfigure.rectangle_rz_figure(pixels=180)
    ax_data_2 = fig_data_2.gca()
    
    maturation_dict = {'230331': 10, '230403': 13, '230407': 17}
    maturation_FF_dict = {k: v - 0.1 for k, v in maturation_dict.items()} #use a delta -0.1 or + 0.1 to have a clearer plot
    maturation_RDG_dict = {k: v + 0.1 for k, v in maturation_dict.items()}

    # Selects only the data to appear on plot, based on the dates given in 
    # the list dates_to_use
    dates_to_use = ['230331', '230403', '230407']
    [date, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = [date, {d:mean_data_FF1_dict[d] for d in dates_to_use}, {d:std_data_FF1_dict[d] for d in dates_to_use},
             {d:mean_data_FF2_dict[d] for d in dates_to_use}, {d:std_data_FF2_dict[d] for d in dates_to_use},
             {d:mean_data_RDG1_dict[d] for d in dates_to_use}, {d:std_data_RDG1_dict[d] for d in dates_to_use},
             {d:mean_data_RDG2_dict[d] for d in dates_to_use}, {d:std_data_RDG2_dict[d] for d in dates_to_use},
             {d:mean_data_FF_dict[d] for d in dates_to_use}, {d:std_data_FF_dict[d] for d in dates_to_use},
             {d:mean_data_RDG_dict[d] for d in dates_to_use}, {d:std_data_RDG_dict[d] for d in dates_to_use}
             ]
    
    # Plots data
    ax_data_1.errorbar(list(maturation_FF_dict.values()), list(mean_data_FF1_dict.values()), yerr=list(std_data_FF1_dict.values()), lw=0, label='FF1', **kwargs_FF1)
    ax_data_2.errorbar(list(maturation_FF_dict.values()), list(mean_data_FF2_dict.values()), yerr=list(std_data_FF2_dict.values()), lw=0, label='FF2', **kwargs_FF2)
    ax_data.errorbar(list(maturation_FF_dict.values()), list(mean_data_FF_dict.values()), yerr=list(std_data_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_data_1.errorbar(list(maturation_RDG_dict.values()), list(mean_data_RDG1_dict.values()), yerr=list(std_data_RDG1_dict.values()), lw=0,  label='RDG1', **kwargs_RDG1)
    ax_data_2.errorbar(list(maturation_RDG_dict.values()), list(mean_data_RDG2_dict.values()), yerr=list(std_data_RDG2_dict.values()), lw=0, label='RDG2', **kwargs_RDG2)
    ax_data.errorbar(list(maturation_RDG_dict.values()), list(mean_data_RDG_dict.values()), yerr=list(std_data_RDG_dict.values()), lw=0, label='RDG', **kwargs_RDG)
    
    # Define plot settings
    ax_data_1.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_data_2.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_data.legend(prop=fonts.serif_rz_legend(), loc='lower center', framealpha=0.7)
    ax_data.set_title(labels[indicator] + ' vs maturation 1+2', font=fonts.serif_rz_legend())
    ax_data_1.set_title(labels[indicator] + ' vs maturation 1', font=fonts.serif_rz_legend())
    ax_data_2.set_title(labels[indicator] + ' vs maturation 2', font=fonts.serif_rz_legend())
    ax_data.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_data_1.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_data_2.set_xlabel('Durée de stockage [jours]', font=fonts.serif_rz_legend())
    ax_data.set_ylabel(labels[indicator], font=fonts.serif_rz_legend())
    ax_data_1.set_ylabel(labels[indicator], font=fonts.serif_rz_legend())
    ax_data_2.set_ylabel(labels[indicator], font=fonts.serif_rz_legend())
    
    # Save and close figures
    savefigure.save_as_png(fig_data, indicator + "_vs_maturation_1+2")
    savefigure.save_as_png(fig_data_1, indicator + "_vs_maturation_1")
    savefigure.save_as_png(fig_data_2, indicator + "_vs_maturation_2")
    plt.close(fig_data)
    plt.close(fig_data_1)
    plt.close(fig_data_2)

def plot_indentation_relaxation_indicator_vs_texturometer_forces(irr_indicator):
    """Plots the indicator in terms of the texturometer forces,
    measured on the same meatpiece, the same day

    Args:
        indicator (str): name of the indicator to be investigated.
            Possible values:
                - relaxation_slope
                - delta_f
                - delta_f_star
                - i_disp_time
                - i_time_time
    """
    dates_to_use = ['230331', '230403', '230407']
    maturation_dict_plots = {'230331': 'J+10', '230403': 'J+13', '230407': 'J+17'}
    
    # Load indicator data
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    complete_pkl_filename = path_to_processed_data + "/indentation_relaxation_mean_std_" + irr_indicator + ".pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = pickle.load(f)
        
    [dates, mean_data_FF1_dict, std_data_FF1_dict,
             mean_data_FF2_dict, std_data_FF2_dict,
             mean_data_RDG1_dict, std_data_RDG1_dict,
             mean_data_RDG2_dict, std_data_RDG2_dict,
             mean_data_FF_dict, std_data_FF_dict,
             mean_data_RDG_dict, std_data_RDG_dict
             ] = [dates, {d:mean_data_FF1_dict[d] for d in dates_to_use}, {d:std_data_FF1_dict[d] for d in dates_to_use},
             {d:mean_data_FF2_dict[d] for d in dates_to_use}, {d:std_data_FF2_dict[d] for d in dates_to_use},
             {d:mean_data_RDG1_dict[d] for d in dates_to_use}, {d:std_data_RDG1_dict[d] for d in dates_to_use},
             {d:mean_data_RDG2_dict[d] for d in dates_to_use}, {d:std_data_RDG2_dict[d] for d in dates_to_use},
             {d:mean_data_FF_dict[d] for d in dates_to_use}, {d:std_data_FF_dict[d] for d in dates_to_use},
             {d:mean_data_RDG_dict[d] for d in dates_to_use}, {d:std_data_RDG_dict[d] for d in dates_to_use}
             ]
        
    path_to_processed_data_texturometer = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename_texturometer = path_to_processed_data_texturometer + "/forces_mean_std.pkl"
    with open(complete_pkl_filename_texturometer, "rb") as f:
        [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)

    # Selects only the data to appear on plot, based on the dates given in 
    # the list dates_to_use
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

    # Definition of the plot settings
    pixels=180
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF1 = {'marker':'o', 'mfc':color[6], 'elinewidth':3, 'ecolor':color[6], 'alpha':0.8, 'ms':'10', 'mec':color[6]}
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[3]}
    kwargs_FF2 = {'marker':'o', 'mfc':color[7], 'elinewidth':3, 'ecolor':color[7], 'alpha':0.8, 'ms':'10', 'mec':color[7]}
    kwargs_RDG1 = {'marker':'^', 'mfc':color[0], 'elinewidth':3, 'ecolor':color[0], 'alpha':0.8, 'ms':10, 'mec':color[0]}
    kwargs_RDG2 = {'marker':'^', 'mfc':color[1], 'elinewidth':3, 'ecolor':color[1], 'alpha':0.8, 'ms':'10', 'mec':color[1]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'10', 'mec':color_rocket[1]}
    labels = {'relaxation_slope' : r"$\beta$ [$Ns^{-1}$]",
                    'delta_f' : r"$\Delta F$ [$N$]",
                    'delta_f_star' : r"$\Delta F^*$ [-]",
                    'i_disp_time': r"$\alpha $ [$Nmm^{-1}$]",
                    'i_time_time': r"$\alpha $ [$Ns^{-1}$]"  }    
    
    ### Plot indicator in meatpiece 1 vs 80% texturometer force
    
    # Create figure object    
    fig_data_vs_force80_1 = createfigure.rectangle_rz_figure(pixels)
    ax_data_vs_force80_1 = fig_data_vs_force80_1.gca()
    
    # Make a linear regression of the indicator and force values
    force_80_1 = np.concatenate((list(mean_force80_FF1_dict.values()), list(mean_force80_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_80_1) 
    data_1 = np.concatenate(( list(mean_data_FF1_dict.values()) , list(mean_data_RDG1_dict.values()) ))
    index_data_1_nan = np.isnan(data_1)
    indices_force_or_data_1_nan = [index_force_1_nan[i] or index_data_1_nan[i] for i in range(len(index_force_1_nan))]
    force_80_1_without_nan = np.array([force_80_1[i] for i in range(len(force_80_1)) if not indices_force_or_data_1_nan[i]])
    data_1_without_nan_force = np.array([data_1[i] for i in range(len(data_1)) if not indices_force_or_data_1_nan[i]])
    force_80_1 = force_80_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_1, data_1_without_nan_force)
    fitted_response_data_1 = model.predict(force_80_1)
    a_data_1 = reg.coef_
    b_data_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data_1 = reg.score(force_80_1, data_1_without_nan_force)
    
    # Plot data
    ax_data_vs_force80_1.errorbar(list(mean_force80_FF1_dict.values()), list(mean_data_FF1_dict.values()), yerr=list(std_data_FF1_dict.values()), xerr=list(std_force80_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_data_vs_force80_1.errorbar(list(mean_force80_RDG1_dict.values()), list(mean_data_RDG1_dict.values()), yerr=list(std_data_RDG1_dict.values()), xerr=list(std_force80_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_data_vs_force80_1.plot(force_80_1, fitted_response_data_1, ':k', label=labels[irr_indicator] + ' = ' + str(np.round(a_data_1[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_data_1[0], 4)) + '\n R2 = ' + str(np.round(score_data_1, 2)) )
    
    # Add annotations with the maturation to plot
    for i in range(len(mean_force80_FF1_dict)):
        date = dates_to_use[i]
        ax_data_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_FF1_dict[date] +0.04, mean_data_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_RDG1_dict[date]+0.04, mean_data_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
    
    # Add plot settings 
    ax_data_vs_force80_1.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
    ax_data_vs_force80_1.set_title(labels[irr_indicator] + ' vs Force 80% 1', font=fonts.serif_rz_legend())
    ax_data_vs_force80_1.set_ylabel(labels[irr_indicator], font=fonts.serif_rz_legend())
    ax_data_vs_force80_1.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    
    # Save and close figures
    savefigure.save_as_png(fig_data_vs_force80_1, irr_indicator + "_vs_force80_1")
    plt.close(fig_data_vs_force80_1)
    
    ### Plot indicator in meatpiece 2 vs 80% texturometer force
    # The steps are the same as for plotting meatpiece 1, but are not ordered
    
    fig_data_vs_force80_2 = createfigure.rectangle_rz_figure(pixels)
    ax_data_vs_force80_2 = fig_data_vs_force80_2.gca()

    force_80_2 = np.concatenate((list(mean_force80_FF2_dict.values()), list(mean_force80_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_80_2) 
    data_2 = np.concatenate(( list(mean_data_FF2_dict.values()) , list(mean_data_RDG2_dict.values()) ))
    index_data_2_nan = np.isnan(data_2)
    indices_force_or_data_2_nan = [index_force_2_nan[i] or index_data_2_nan[i] for i in range(len(index_force_2_nan))]
    force_80_2_without_nan = np.array([force_80_2[i] for i in range(len(force_80_2)) if not indices_force_or_data_2_nan[i]])
    data_2_without_nan_force = np.array([data_2[i] for i in range(len(data_2)) if not indices_force_or_data_2_nan[i]])
    force_80_2 = force_80_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80_2, data_2_without_nan_force)
    fitted_response_data_2 = model.predict(force_80_2)
    a_data_2 = reg.coef_
    b_data_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data_2 = reg.score(force_80_2, data_2_without_nan_force)
    
    ax_data_vs_force80_2.errorbar(list(mean_force80_FF2_dict.values()), list(mean_data_FF2_dict.values()), yerr=list(std_data_FF2_dict.values()), xerr=list(std_force80_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_data_vs_force80_2.errorbar(list(mean_force80_RDG2_dict.values()), list(mean_data_RDG2_dict.values()), yerr=list(std_data_RDG2_dict.values()), xerr=list(std_force80_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_data_vs_force80_2.plot(force_80_2, fitted_response_data_2, ':k', label=labels[irr_indicator] + ' = ' + str(np.round(a_data_2[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_data_2[0], 4)) + '\n R2 = ' + str(np.round(score_data_2, 2)) )
    for i in range(len(mean_force80_FF2_dict)):
        date = dates_to_use[i]
        ax_data_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_FF2_dict[date] +0.04, mean_data_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force80_2.annotate(maturation_dict_plots[date], (mean_force80_RDG2_dict[date]+0.04, mean_data_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force80_2.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
    ax_data_vs_force80_2.set_title(labels[irr_indicator] + ' vs Force 80% 2', font=fonts.serif_rz_legend())
    ax_data_vs_force80_2.set_ylabel(labels[irr_indicator], font=fonts.serif_rz_legend())
    ax_data_vs_force80_2.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_data_vs_force80_2, irr_indicator + "_vs_force80_2")
    plt.close(fig_data_vs_force80_2)

    ### Plot indicator in meatpiece 1+2 (without distinction) vs 80% texturometer force
    # The steps are the same as for plotting meatpiece 1, but are not ordered
    
    force_80 = np.concatenate((list(mean_force80_FF_dict.values()), list(mean_force80_RDG_dict.values())))
    index_force_nan = np.isnan(force_80) 
    data = np.concatenate(( list(mean_data_FF_dict.values()) , list(mean_data_RDG_dict.values()) ))
    index_data_nan = np.isnan(data)
    indices_force_or_data_nan = [index_force_nan[i] or index_data_nan[i] for i in range(len(index_force_nan))]
    force_80_without_nan = np.array([force_80[i] for i in range(len(force_80)) if not indices_force_or_data_nan[i]])
    data_without_nan_force = np.array([data[i] for i in range(len(data)) if not indices_force_or_data_nan[i]])
    force_80 = force_80_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_80, data_without_nan_force)
    fitted_response_data = model.predict(force_80)
    a_data = reg.coef_
    b_data = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data = reg.score(force_80, data_without_nan_force)
    
    fig_data_vs_force80 = createfigure.rectangle_rz_figure(pixels)
    ax_data_vs_force80 = fig_data_vs_force80.gca()
    ax_data_vs_force80.errorbar(list(mean_force80_FF_dict.values()), list(mean_data_FF_dict.values()), yerr=list(std_data_FF_dict.values()), xerr=list(std_force80_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_data_vs_force80.errorbar(list(mean_force80_RDG_dict.values()), list(mean_data_RDG_dict.values()), yerr=list(std_data_RDG_dict.values()), xerr=list(std_force80_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_data_vs_force80.plot(force_80, fitted_response_data, ':k', label=labels[irr_indicator] + ' = ' + str(np.round(a_data[0], 4)) + r'$F_{80 \%}$ + '+  str(np.round(b_data[0], 4)) + '\n R2 = ' + str(np.round(score_data, 2)) )
    for i in range(len(mean_force80_FF_dict)):
        date = dates_to_use[i]
        ax_data_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_FF_dict[date] +0.04, mean_data_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force80.annotate(maturation_dict_plots[date], (mean_force80_RDG_dict[date]+0.04, mean_data_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force80.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
    ax_data_vs_force80.set_title(labels[irr_indicator] + ' vs Force 80% 1+2', font=fonts.serif_rz_legend())
    ax_data_vs_force80.set_ylabel(labels[irr_indicator], font=fonts.serif_rz_legend())
    ax_data_vs_force80.set_xlabel('Force 80 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_data_vs_force80, irr_indicator + "_vs_force80_1+2")
    plt.close(fig_data_vs_force80)

    ### Plot indicator in meatpiece 1 vs 20% texturometer force
    # The steps are the same as for plotting meatpiece 1, but are not ordered
       
    force_20_1 = np.concatenate((list(mean_force20_FF1_dict.values()), list(mean_force20_RDG1_dict.values())))
    index_force_1_nan = np.isnan(force_20_1) 
    data_1 = np.concatenate(( list(mean_data_FF1_dict.values()) , list(mean_data_RDG1_dict.values()) ))
    index_data_1_nan = np.isnan(data_1)
    indices_force_or_data_1_nan = [index_force_1_nan[i] or index_data_1_nan[i] for i in range(len(index_force_1_nan))]
    force_20_1_without_nan = np.array([force_20_1[i] for i in range(len(force_20_1)) if not indices_force_or_data_1_nan[i]])
    data_1_without_nan_force = np.array([data_1[i] for i in range(len(data_1)) if not indices_force_or_data_1_nan[i]])
    force_20_1 = force_20_1_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_1, data_1_without_nan_force)
    fitted_response_data_1 = model.predict(force_20_1)
    a_data_1 = reg.coef_
    b_data_1 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data_1 = reg.score(force_20_1, data_1_without_nan_force)
    
    fig_data_vs_force20_1 = createfigure.rectangle_rz_figure(pixels)
    ax_data_vs_force20_1 = fig_data_vs_force20_1.gca()
    ax_data_vs_force20_1.errorbar(list(mean_force20_FF1_dict.values()), list(mean_data_FF1_dict.values()), yerr=list(std_data_FF1_dict.values()), xerr=list(std_force20_FF1_dict.values()) ,lw=0, label='FF1', **kwargs_FF1)
    ax_data_vs_force20_1.errorbar(list(mean_force20_RDG1_dict.values()), list(mean_data_RDG1_dict.values()), yerr=list(std_data_RDG1_dict.values()), xerr=list(std_force20_RDG1_dict.values()) ,lw=0, label='RDG1', **kwargs_RDG1)
    ax_data_vs_force20_1.plot(force_20_1, fitted_response_data_1, ':k', label=labels[irr_indicator] + ' = ' + str(np.round(a_data_1[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_data_1[0], 4)) + '\n R2 = ' + str(np.round(score_data_1, 2)) )
    for i in range(len(mean_force20_FF1_dict)):
        date = dates_to_use[i]
        ax_data_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_FF1_dict[date] +0.04, mean_data_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force20_1.annotate(maturation_dict_plots[date], (mean_force20_RDG1_dict[date]+0.04, mean_data_RDG1_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force20_1.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
    ax_data_vs_force20_1.set_title(labels[irr_indicator] + ' vs Force 20% 1', font=fonts.serif_rz_legend())
    ax_data_vs_force20_1.set_ylabel(labels[irr_indicator], font=fonts.serif_rz_legend())
    ax_data_vs_force20_1.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_data_vs_force20_1, irr_indicator + "_vs_force20_1")
    plt.close(fig_data_vs_force20_1)
    force_20_2 = np.concatenate((list(mean_force20_FF2_dict.values()), list(mean_force20_RDG2_dict.values())))
    index_force_2_nan = np.isnan(force_20_2) 
    data_2 = np.concatenate(( list(mean_data_FF2_dict.values()) , list(mean_data_RDG2_dict.values()) ))
    index_data_2_nan = np.isnan(data_2)
    indices_force_or_data_2_nan = [index_force_2_nan[i] or index_data_2_nan[i] for i in range(len(index_force_2_nan))]
    force_20_2_without_nan = np.array([force_20_2[i] for i in range(len(force_20_2)) if not indices_force_or_data_2_nan[i]])
    data_2_without_nan_force = np.array([data_2[i] for i in range(len(data_2)) if not indices_force_or_data_2_nan[i]])
    force_20_2 = force_20_2_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20_2, data_2_without_nan_force)
    fitted_response_data_2 = model.predict(force_20_2)
    a_data_2 = reg.coef_
    b_data_2 = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data_2 = reg.score(force_20_2, data_2_without_nan_force)
    
    ### Plot indicator in meatpiece 2 vs 20% texturometer force
    # The steps are the same as for plotting meatpiece 1, but are not ordered  
      
    fig_data_vs_force20_2 = createfigure.rectangle_rz_figure(pixels)
    ax_data_vs_force20_2 = fig_data_vs_force20_2.gca()
    ax_data_vs_force20_2.errorbar(list(mean_force20_FF2_dict.values()), list(mean_data_FF2_dict.values()), yerr=list(std_data_FF2_dict.values()), xerr=list(std_force20_FF2_dict.values()) ,lw=0, label='FF2', **kwargs_FF2)
    ax_data_vs_force20_2.errorbar(list(mean_force20_RDG2_dict.values()), list(mean_data_RDG2_dict.values()), yerr=list(std_data_RDG2_dict.values()), xerr=list(std_force20_RDG2_dict.values()) ,lw=0, label='RDG2', **kwargs_RDG2)
    ax_data_vs_force20_2.plot(force_20_2, fitted_response_data_2, ':k', label=labels[irr_indicator] + ' = ' + str(np.round(a_data_2[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_data_2[0], 4)) + '\n R2 = ' + str(np.round(score_data_2, 2)) )
    for i in range(len(mean_force20_FF2_dict)):
        date = dates_to_use[i]
        ax_data_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_FF2_dict[date] +0.04, mean_data_FF2_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force20_2.annotate(maturation_dict_plots[date], (mean_force20_RDG2_dict[date]+0.04, mean_data_RDG2_dict[date]+0.02), color = color[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force20_2.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
    ax_data_vs_force20_2.set_title(labels[irr_indicator] + ' vs Force 20% 2', font=fonts.serif_rz_legend())
    ax_data_vs_force20_2.set_ylabel(labels[irr_indicator], font=fonts.serif_rz_legend())
    ax_data_vs_force20_2.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_data_vs_force20_2, irr_indicator + "_vs_force20_2")
    plt.close(fig_data_vs_force20_2)

    ### Plot indicator in meatpiece 1+2 (without distinction) vs 20% texturometer force
    # The steps are the same as for plotting meatpiece 1, but are not ordered
    
    force_20 = np.concatenate((list(mean_force20_FF_dict.values()), list(mean_force20_RDG_dict.values())))
    index_force_nan = np.isnan(force_20) 
    data = np.concatenate(( list(mean_data_FF_dict.values()) , list(mean_data_RDG_dict.values()) ))
    index_data_nan = np.isnan(data)
    indices_force_or_data_nan = [index_force_nan[i] or index_data_nan[i] for i in range(len(index_force_nan))]
    force_20_without_nan = np.array([force_20[i] for i in range(len(force_20)) if not indices_force_or_data_nan[i]])
    data_without_nan_force = np.array([data[i] for i in range(len(data)) if not indices_force_or_data_nan[i]])
    force_20 = force_20_without_nan.reshape((-1, 1))
    model = LinearRegression()
    reg = model.fit(force_20, data_without_nan_force)
    fitted_response_data = model.predict(force_20)
    a_data = reg.coef_
    b_data = model.predict(np.array([0, 0, 0, 0]).reshape(-1, 1))
    score_data = reg.score(force_20, data_without_nan_force)
    
    fig_data_vs_force20 = createfigure.rectangle_rz_figure(pixels)
    ax_data_vs_force20 = fig_data_vs_force20.gca()
    ax_data_vs_force20.errorbar(list(mean_force20_FF_dict.values()), list(mean_data_FF_dict.values()), yerr=list(std_data_FF_dict.values()), xerr=list(std_force20_FF_dict.values()) ,lw=0, label='FF', **kwargs_FF)
    ax_data_vs_force20.errorbar(list(mean_force20_RDG_dict.values()), list(mean_data_RDG_dict.values()), yerr=list(std_data_RDG_dict.values()), xerr=list(std_force20_RDG_dict.values()) ,lw=0, label='RDG', **kwargs_RDG)
    ax_data_vs_force20.plot(force_20, fitted_response_data, ':k', label=labels[irr_indicator] + ' = ' + str(np.round(a_data[0], 4)) + r'$F_{20 \%}$ + '+  str(np.round(b_data[0], 4)) + '\n R2 = ' + str(np.round(score_data, 2)) )
    for i in range(len(mean_force20_FF_dict)):
        date = dates_to_use[i]
        ax_data_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_FF_dict[date] +0.04, mean_data_FF_dict[date]+0.02), color = color_rocket[3], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))
        ax_data_vs_force20.annotate(maturation_dict_plots[date], (mean_force20_RDG_dict[date]+0.04, mean_data_RDG_dict[date]+0.02), color = color_rocket[1], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    ax_data_vs_force20.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
    ax_data_vs_force20.set_title(labels[irr_indicator] + ' vs Force 20% 1+2', font=fonts.serif_rz_legend())
    ax_data_vs_force20.set_ylabel(labels[irr_indicator], font=fonts.serif_rz_legend())
    ax_data_vs_force20.set_xlabel('Force 20 % [N]', font=fonts.serif_rz_legend())
    savefigure.save_as_png(fig_data_vs_force20, irr_indicator + "_vs_force20_1+2")
    plt.close(fig_data_vs_force20)
        
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230331', '230407', '230411', '230403']
    types_of_essay = ['C_Indentation_relaxation_500N_force.xlsx']#,'C_Indentation_relaxation_maintienFnulle_500N_trav.xls',  'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = []
    for i in range(len(experiment_dates)):
        datafile_list += files_zwick.import_files(experiment_dates[i])


    # export_indicators(datafile_list)
    # path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\zwick\processed_data'
    # complete_pkl_filename = path_to_processed_data + "/indicators_indentation_relaxation.pkl"
    # with open(complete_pkl_filename, "rb") as f:
    #     [ids_list, date_dict, meat_piece_dict, relaxation_slope_dict, delta_f_dict, delta_f_star_dict, i_disp_time_dict, i_time_time_dict] = pickle.load(f)

    indicator_list = ['relaxation_slope',
                    'delta_f',
                    'delta_f_star',
                    'i_disp_time',
                    'i_time_time' ]
    # ids_at_date_and_meatpiece, data_dict_at_date_and_meatpiece  = extract_data_at_given_date_and_meatpiece('230331', 'RDG1', ids_list, meat_piece_dict, date_dict, relaxation_slope_dict)
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, relaxation_slope_dict, 'relaxation_slope')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, delta_f_dict, 'delta_f')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, delta_f_star_dict, 'delta_f_star')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, i_disp_time_dict, 'i_disp_time')
    # compute_and_export_mean_std_data_with_maturation_as_pkl(ids_list, date_dict, i_time_time_dict, 'i_time_time')
    for indicator in indicator_list:
        # export_data_as_txt(indicator)
        plot_data_with_maturation(indicator)
        # plot_indentation_relaxation_indicator_vs_texturometer_forces(indicator)
        
    for datafile in datafile_list:
        sheet_list = files_zwick.find_only_correct_sheets_in_datafile(datafile)
        for sheet in sheet_list:
            plot_indicators_indentation_relaxation(files_zwick, datafile, sheet)

    print('hello')
    
    