"""
The objective of this file is to extract the force and displacement values, generated
via the COMSOL model, from the corresponding txt excel file.
The indicatrs are then computed based on this data.
"""

import numpy as np
from matplotlib import pyplot as plt
from indentation.model.comsol.sensitivity_analysis import utils
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from indentation.experiments.zwick.post_processing.utils import find_nearest
from datetime import datetime

def get_inputs():
    """Extract the input dataset from the "disp_silicone.xlsx" file. 
    Note that it could also have been extracted from the "force_indent.xlsx" file,
    since both contain an "input" tab.

    Returns:
        ids_list (list):list containing the id of each input sample
        elongation_dict (dictionnary): dictionnary that assocites to each id the corresponding elongation
            used as an input parameter. The elongation is also called "lambda" in the excel files.
        damage_dict (dictionnary): dictionnary that assocites to each id the corresponding damage
            used as an input parameter
    """
    path_to_file = utils.reach_data_path() / 'disp_silicone.xlsx'
    input_data = pd.read_excel(path_to_file, sheet_name='input', header=0, names=["Id", "elongation", "damage"]) 
    ids = input_data.Id
    elongations = input_data.elongation
    damages = input_data.damage
    ids_list = ids.tolist()
    elongation_dict = {ids.tolist()[i]: elongations.tolist()[i] for i in range(len(ids.tolist()))}
    damage_dict = {ids.tolist()[i]: damages.tolist()[i] for i in range(len(ids.tolist()))}
    return ids_list, elongation_dict, damage_dict

def get_force():
    """Extract the force in terms of time from the COMSOL model, for each input sample.

    Returns:
        time_list (list): temporal discretisation used to compute the COMSOL model. 
        force_dict (dictionnary): dictionnary that associatesto each id the corresponding temporal
            evolution of force
    """
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    path_to_file_force = utils.reach_data_path() / 'force_indent.xlsx'
    force_data = pd.read_excel(path_to_file_force, sheet_name='output', header=0)
    times = force_data.time
    time_list = [float(t) for t in times.to_list()]
    force_dict = {}
    for id in ids_list:
        force_id = force_data[id]
        force_dict[id] = [-float(s) for s in force_id.to_list()]
    return time_list, force_dict

def get_disp():
    """Extract the disp in terms of time from the COMSOL model, for each input sample.

    Returns:
        time_list (list): temporal discretisation used to compute the COMSOL model. 
        disp_dict (dictionnary): dictionnary that associatesto each id the corresponding temporal
            evolution of disp
    """
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    path_to_file_disp = utils.reach_data_path() / 'disp_silicone.xlsx'
    disp_data = pd.read_excel(path_to_file_disp, sheet_name='output', header=0)
    times = disp_data.time
    time_list = [float(t) for t in times.to_list()]
    disp_dict = {}
    for id in ids_list:
        disp_id = disp_data[id]
        disp_dict[id] = [float(s) for s in disp_id.to_list()]
    return time_list, disp_dict

def get_data(input_id):
    """Give, for a given input sample id, the corresponding input parameters and COMSOL model outputs

    Args:
        input_id (float): id of the input sample

    Returns:
        elongation (float): value of elongation (lambda) used for this sample
        damage (float): value of damage used for this sample
        disp (array): temporal evolution of displacement, computed with the COMSOL model, 
            for the input parameters corresponding to the id input sample
        force (array): temporal evolution of force, computed with the COMSOL model, 
            for the input parameters corresponding to the id input sample
    """
    _, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()
    _, disp_dict = utils.extract_disp_from_pkl()
    _, force_dict = utils.extract_force_from_pkl()
    disp = disp_dict[input_id]
    force = force_dict[input_id]
    elongation = elongation_dict[input_id]
    damage = damage_dict[input_id]
    return elongation, damage, disp, force

# def compute_dz(input_id):
#     """Compute the variation in height of the silicon piece during elongation

#     Args:
#         input_id (float): id of the input sample

#     Returns:
#         dz (float): variation in height of the silicon piece during elongation
#     """
#     elongation, _, _, _ = get_data(input_id)
#     sample_width = 36
#     dz = sample_width * (1 - (1 / elongation)**2)
#     return dz

def reshape_disp_to_recovery(input_id):
    """Shorten the displacement vector to the recovery phase, which is defined, based on the COMSOL model,
    to be started after 6 seconds.

    Args:
        input_id (float): id of the input sample

    Returns:
        recovery_time_list (list): temporal discretisation during the recovery phase, after removing the offset.
        recovery_disp_list (list): variation of displacement during the recovery phase, after removing the offset.
    """
    _, _, disp, _ = get_data(input_id)
    time_list, _ = utils.extract_disp_from_pkl()
    begginning_recovery_time = 6
    index_where_recovery_beggins = np.where(np.array(time_list) == begginning_recovery_time)[-1][0]
    recovery_time_list = [t - time_list[index_where_recovery_beggins+1] for t in time_list[index_where_recovery_beggins+1:]]
    recovery_disp_list = [-d + disp[index_where_recovery_beggins+1] for d in disp[index_where_recovery_beggins+1:]]
    return recovery_time_list, recovery_disp_list

def reshape_force_to_indentation_relaxation(input_id):
    """Shorten the force vector to the indentation-relaxation phases, which is defined, based on the COMSOL model,
    to be started at the end of elongation, ie when the displacement equals that computed with the compute_dz function.

    Args:
        input_id (float): id of the input sample

    Returns:
        indentation_relaxation_time_list (list): temporal discretisation during the indentation-relaxation phases, 
            after removing the offset.
        indentation_relaxation_force_list (list): variation of force during the indentation-relaxation phases, 
            after removing the offset.
    """
    _, _, disp, force = get_data(input_id)
    time_list, _ = utils.extract_disp_from_pkl()
    begginning_indentation_time_index = np.where(np.array(force[0:int(len(force)/2)])>0)[0][-1] 
    index_where_indentation_beggins = begginning_indentation_time_index + 0
    end_relaxation_time = 6
    index_where_relaxation_ends = np.where(time_list == find_nearest(time_list, end_relaxation_time))[0][0]
    indentation_relaxation_time_list = [t - time_list[index_where_indentation_beggins] for t in time_list[index_where_indentation_beggins:index_where_relaxation_ends] ]
    indentation_relaxation_force_list = force[index_where_indentation_beggins:index_where_relaxation_ends]
    indentation_relaxation_force_list = [-s for s in indentation_relaxation_force_list]
    return indentation_relaxation_time_list, indentation_relaxation_force_list

def get_data_at_given_indentation_relaxation_time(input_id, given_time):
    """ Provides the force value measured for a given id input sample at a given time

    Args:
        input_id (float): id of the input sample
        given_time (float): time (in second) at which the force is measured

    Returns:
        time_given_time (float): time at the given time. Generated only for internal validation, to make sure that
            the value that is returned acutally corresponds to the given_time
        force_given_time (float): force measured at a given_time
    """
    indentation_relaxation_time_list, indentation_relaxation_force_list = reshape_force_to_indentation_relaxation(input_id)
    time_given_time = find_nearest(indentation_relaxation_time_list,  given_time)
    index_where_time_is_time_given_time = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, time_given_time))[0][0]
    force_given_time = indentation_relaxation_force_list[index_where_time_is_time_given_time]
    return time_given_time, force_given_time

def find_recovery_slope(input_id):
    """Computes the initial slope of the displcaement with respect to time 
    during the recovery phase. The slope is measured during the 100 first time instants

    Args:
        input_id (float): id of the input sample

    Returns:
        recovery_time_list (list): time values used for the fit
        fitted_response_linear (list): fitted values of displacement
        A_linear (float): value of the slope of the fitted linear function (y = ax+b)
        intercept_linear (float): value of the intercept of the fitted linear function (y = ax+b)
    """
    recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    recovery_disp_list_in_mm = [d*10 for d in recovery_disp_list]
    recovery_time_list = recovery_time_list[0:100]
    recovery_disp_list_in_mm = recovery_disp_list_in_mm[0:100]
    poly = PolynomialFeatures(degree=1, include_bias=False)
    poly_features_linear = poly.fit_transform(np.array(recovery_time_list).reshape(-1, 1))
    model_linear = make_pipeline(poly, LinearRegression())
    model_linear.fit(poly_features_linear, recovery_disp_list_in_mm)
    fitted_response_linear = model_linear.predict(np.array(recovery_time_list).reshape(-1, 1))
    A_linear = model_linear.steps[1][1].coef_[0]
    intercept_linear = model_linear.steps[1][1].intercept_
    return recovery_time_list, fitted_response_linear, A_linear, intercept_linear

def find_indicators_indentation_relaxation(input_id):
    """Compute the indicators for the indentation-relaxation phases

    Args:
        input_id (float): id of the input sample

    Returns:
        delta_f (float): variation of force during the first 10 seconds of relaxation.
        delta_f_star (float): delta_f, normalized by the peak force
        relaxation_slope (float): slope of the force in terms of time at the very beginning of 
            the relaxation phase (first 0.5 second). The beginning of relaxation is designated by
            the instant where the force reaches the peak force.
        i_time (float): slope of the force in terms of time at the very beginning of the indentation
            phase (first second).    
    """
    indentation_relaxation_time_list, indentation_relaxation_force_list = reshape_force_to_indentation_relaxation(input_id)
    indentation_relaxation_force_list_in_N = [s for s in indentation_relaxation_force_list]
    max_force = np.nanmax(indentation_relaxation_force_list_in_N)
    index_where_force_is_max = np.where(indentation_relaxation_force_list_in_N == max_force)[0][0]
    time_when_force_is_max = indentation_relaxation_time_list[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, time_when_force_is_max+0.5))[0][0]
    relaxation_slope = (indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation_slope] - indentation_relaxation_force_list_in_N[index_where_force_is_max ]) / (indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope] - indentation_relaxation_time_list[index_where_force_is_max])
    time_when_force_is_max = indentation_relaxation_time_list[index_where_force_is_max]
    relaxation_duration = np.min([10, np.max(np.array(indentation_relaxation_time_list))])
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, end_of_relaxation))[0][0]
    delta_f = max_force - indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    max_force_index = np.argmax(np.array(indentation_relaxation_force_list))
    time_given_time, force_given_time = indentation_relaxation_time_list[max_force_index-1], indentation_relaxation_force_list[max_force_index-1]
    time_at_time_0, force_at_time_0 = get_data_at_given_indentation_relaxation_time(input_id, time_given_time - 0.1) 
    i_time = (force_given_time - force_at_time_0) / (time_given_time - time_at_time_0)     
    return delta_f, delta_f_star, relaxation_slope, i_time

def plot_reshaped_data(input_id):
    """Plot the force and displacement curves after removing the offsets, with indicators.

    Args:
        input_id (float): id of the input sample
    """
    indentation_relaxation_time_list, indentation_relaxation_force_list = reshape_force_to_indentation_relaxation(input_id)
    recovery_time_list, recovery_disp_list = reshape_disp_to_recovery(input_id)
    elongation, damage, _, _ = get_data(input_id)
    fig_force_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_time = fig_force_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()
    colors = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    indentation_relaxation_force_list_in_N = [s for s in indentation_relaxation_force_list]
    recovery_disp_list_in_mm = [d*10 for d in recovery_disp_list]
    max_force = np.nanmax(indentation_relaxation_force_list_in_N)
    index_where_force_is_max = np.where(indentation_relaxation_force_list_in_N == max_force)[0][0]
    time_when_force_is_max = indentation_relaxation_time_list[index_where_force_is_max]
    index_where_time_is_end_relaxation_slope = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, time_when_force_is_max+0.5))[0][0]
    relaxation_slope = (indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation_slope] - indentation_relaxation_force_list_in_N[index_where_force_is_max ]) / (indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope] - indentation_relaxation_time_list[index_where_force_is_max])
    time_when_force_is_max = indentation_relaxation_time_list[index_where_force_is_max]
    relaxation_duration = np.min([10, np.max(np.array(indentation_relaxation_time_list))])
    end_of_relaxation = time_when_force_is_max + relaxation_duration
    index_where_time_is_end_relaxation = np.where(indentation_relaxation_time_list == find_nearest(indentation_relaxation_time_list, end_of_relaxation))[0][0]
    delta_f = max_force - indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation]
    delta_f_star = delta_f / max_force
    ax_force_vs_time.plot([indentation_relaxation_time_list[index_where_force_is_max ], indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope]], [indentation_relaxation_force_list_in_N[index_where_force_is_max ], indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation_slope]], '-', color = 'r', label = r"$\beta$ = " + str(np.round(relaxation_slope, 2)) + r" $N.s^{-1}$", linewidth=3)
    ax_force_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope], indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope]], [indentation_relaxation_force_list_in_N[index_where_force_is_max ], indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation_slope]], '--', color = 'r', linewidth=2)
    ax_force_vs_time.plot([indentation_relaxation_time_list[index_where_force_is_max ], indentation_relaxation_time_list[index_where_time_is_end_relaxation_slope]], [indentation_relaxation_force_list_in_N[index_where_force_is_max ], indentation_relaxation_force_list_in_N[index_where_force_is_max ]], '--', color = 'r', linewidth=2)
    ax_force_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation], indentation_relaxation_time_list[index_where_time_is_end_relaxation]], [indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation], max_force], '-', color = 'g', label = r"$\Delta F$  = " + str(np.round(delta_f, 2)) + " N \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)), linewidth=3)
    ax_force_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation]-0.2, indentation_relaxation_time_list[index_where_time_is_end_relaxation]+0.2], [indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation], indentation_relaxation_force_list_in_N[index_where_time_is_end_relaxation]], '--', color = 'g', linewidth=2)
    ax_force_vs_time.plot([indentation_relaxation_time_list[index_where_time_is_end_relaxation]-0.2, indentation_relaxation_time_list[index_where_time_is_end_relaxation]+0.2], [max_force, max_force], '--', color = 'g', linewidth=2)
    max_force_index = np.argmax(np.array(indentation_relaxation_force_list))
    time_given_time, force_given_time = indentation_relaxation_time_list[max_force_index-1], indentation_relaxation_force_list[max_force_index-1]
    time_at_time_0, force_at_time_0 = get_data_at_given_indentation_relaxation_time(input_id, time_given_time - 0.1) 
    i_time = (force_given_time - force_at_time_0) / (time_given_time - time_at_time_0) 
    ax_force_vs_time.plot([time_at_time_0, time_given_time ], [force_at_time_0, force_given_time], '-', color = 'b', label = r"$\alpha'$ = " + str(np.round(i_time, 2)) + r' $N.s^{-1}$', linewidth=3)
    ax_force_vs_time.plot([time_given_time, time_given_time ], [force_at_time_0, force_given_time], '--', color = 'b', linewidth=2)
    ax_force_vs_time.plot([time_at_time_0, time_given_time ], [force_at_time_0, force_at_time_0], '--', color = 'b', linewidth=2)
    ax_force_vs_time.plot(indentation_relaxation_time_list, indentation_relaxation_force_list_in_N, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_force_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.set_ylabel("force [N]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig_force_vs_time, "force_vs_time_id" + str(input_id) )
    plt.close(fig_force_vs_time)
    recovery_time_list_linear, fitted_response_linear, A_linear, intercept_linear = find_recovery_slope(input_id)
    ax_disp_vs_time.plot(recovery_time_list_linear, fitted_response_linear ,   '-', color = color_rocket[3], label =r"$z = a t$" + "\na = " + str(np.round((A_linear), 2)), lw=3)
    ax_disp_vs_time.plot([recovery_time_list_linear[0], recovery_time_list_linear[-1]], [fitted_response_linear[0], fitted_response_linear[0]] ,   '--', color = color_rocket[3],  lw=3)
    ax_disp_vs_time.plot([recovery_time_list_linear[-1], recovery_time_list_linear[-1]], [fitted_response_linear[0], fitted_response_linear[-1]] ,   '--', color = color_rocket[3],  lw=3)
    ax_disp_vs_time.plot(recovery_time_list, recovery_disp_list_in_mm, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_disp_vs_time.set_xticks([0, 1, 2, 3, 4])
    ax_disp_vs_time.set_xticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_yticks([0, 1, 2, 3, 4])
    ax_disp_vs_time.set_yticklabels([0, 1, 2, 3, 4], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel("U [mm]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    savefigure.save_as_png(fig_disp_vs_time, "disp_vs_time_id" + str(input_id) )
    plt.close(fig_disp_vs_time)
    
def plot_entire_data(input_id):
    """Plot the force and displacement curves, before any kind of post processing

    Args:
        input_id (float): id of the input sample
    """
    elongation, damage, disp, force = get_data(input_id)
    time_list, _ = get_force()
    fig_force_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_time = fig_force_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()
    force_in_N = np.array([s for s in force])
    disp_in_mm = [d*10 for d in disp]
    ax_force_vs_time.plot(time_list, force_in_N, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_force_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.set_ylabel("force [N]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig_force_vs_time, "entire_force_vs_time_id" + str(input_id) )
    plt.close(fig_force_vs_time)
    ax_disp_vs_time.plot(time_list, disp_in_mm, ':k', label = r"$\lambda$ = " + str(np.round(elongation, 3)) + r" ; $D$ = " + str(np.round(damage, 3)), linewidth=3)
    ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel("U [mm]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    savefigure.save_as_png(fig_disp_vs_time, "entire_disp_vs_time_id" + str(input_id) )
    plt.close(fig_disp_vs_time)
    
def compute_and_export_all_indicators():
    """Computes the indicators (both related to indentation-relaxation and recovery phases)
    and exports them in a pkl file named "indicators.pkl"
    """
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    alpha_p_dict = {}
    beta_dict = {}
    delta_f_dict = {}
    delta_f_star_dict = {}
    a_dict = {}
    for input_id in ids_list:
        delta_f, delta_f_star, relaxation_slope, i_time = find_indicators_indentation_relaxation(input_id)
        _, _, A_linear, _ = find_recovery_slope(input_id)
        alpha_p_dict[input_id] = i_time
        beta_dict[input_id] = relaxation_slope
        delta_f_dict[input_id] = delta_f
        delta_f_star_dict[input_id] = delta_f_star
        a_dict[input_id] = A_linear
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict],
            f,
        )

def export_indicators_as_txt():    
    """Takes the indicators from the .pkl file generated via the compute_and_export_all_indicators()
    function and writes them into a .txt file, named "indicators.txt"
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)

    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
        
    complete_txt_filename = utils.get_path_to_processed_data() / "indicators.txt"
    f = open(complete_txt_filename, "w")
    f.write("INDICATORS COMPUTED ON " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("Id \t elongation \t damage \t alphap \t beta \t deltaf \t deltaf* \t a \n")
    for i in range(len(ids_list)):
        id = ids_list[i]
        f.write(
            str(id)
            + "\t"
            + str(elongation_dict[id])
            + "\t"
            + str(damage_dict[id])
            + "\t"
            + str(alpha_p_dict[id])
            + "\t"
            + str(beta_dict[id])
            + "\t"
            + str(delta_f_dict[id])
            + "\t"
            + str(delta_f_star_dict[id])
            + "\t"
            + str(a_dict[id])
            + "\n"
        )
    f.close()    
    

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    utils.export_inputs_as_pkl()
    ids_list, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()
    utils.export_force_as_pkl()
    utils.export_disp_as_pkl()
    # time_list, disp_dict = utils.extract_disp_from_pkl()
    # time_list, force_dict = utils.extract_force_from_pkl()
    # reshape_disp_to_recovery(1)
    for id in ids_list:
        plot_reshaped_data(id)
        reshape_force_to_indentation_relaxation(id)
    compute_and_export_all_indicators()
    export_indicators_as_txt()
    print('hello')

