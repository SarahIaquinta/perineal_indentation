"""
This file contains functions to evaluate the representativeness of a dataset 
based on its size. A dataset is large enough to be representative if its behavior 
(represented by its mean and standard deviation) is similar to that of a very large dataset 
(converged behavior). 
The dataset is divided in smaller subdatasets, of which the mean and
standard deviation are calculated. Then, they are compared for subdatasets of different size,
in order to identifiy for which size they converge (ie having a large dataset would not 
improve the representativeness.)
"""

import numpy as np
from indentation.model.comsol.sensitivity_analysis import utils
import numpy as np
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure
import pickle
import seaborn as sns


def compute_cumulative_mean_std(vector):
    """Compute the mean and standard deviation of each subdivision of a vector. 
    Each subdivision is made of the previous one + one element. The first subdivision
    is the first vector element.

    Args:
        vector (list or array): vector whose cumulative means and standard deviations
            are to be computed

    Returns:
        cumulative_mean (array): array of the same shape as the input vector, whose 
        i_th element is the mean of the 0_th to i_th elements of the input vector
        cumulative_std (array): array of the same shape as the input vector, whose 
        i_th element is the standard deviation of the 0_th to i_th elements of the input vector
    """
    cumulative_mean = np.zeros(len(vector))
    cumulative_std = np.zeros_like(cumulative_mean)
    cumulative_mean[0] = vector[0]
    cumulative_std[0] = 0
    for i in range(1, len(vector)):
        cumulative_mean[i] = np.mean(vector[0:i])
        cumulative_std[i] = np.std(vector[0:i])
    return cumulative_mean, cumulative_std

def plot_cumulative_mean_vs_sample_size_indicators(createfigure, savefigure, fonts):
    """Generate and save a figure depicting the variation of the cumulative mean
    and standard deviation of the dataset in terms of the size. 

    Args:
        createfigure (class): class used to provide consistent settings for the creation of figures
        savefigure (class):class used to provide consistent settings for saving figures
        fonts (class): class used to provide consistent fonts for the figures
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict]
    labels = ['alpha', 'beta', 'deltaf', 'deltafstar', 'a']
    label_dict = {'alpha': r"$\alpha' [kPa.s^{-1}]$", 'beta': r"$\beta [kPa.s^{-1}]$", 'deltaf':r"$\Delta F$ [kPa]", 'deltafstar':r"$\Delta F^*$ [-]", 'a': 'a [-]'}
    for i in range(len(indicators_dicts)):
        indicator_dict = indicators_dicts[i]
        label_title = labels[i]
        label_legend = label_dict[label_title]
        indicator_vector = list(indicator_dict.values())
        cumulative_mean, cumulative_std = compute_cumulative_mean_std(indicator_vector)
        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig_mean = createfigure.rectangle_figure(pixels=180)
        ax_mean = fig_mean.gca()
        fig_std = createfigure.rectangle_figure(pixels=180)
        ax_std = fig_std.gca()
        ax_mean.plot(
            sample_size,
            cumulative_mean, '-k')
        ax_std.plot(
            sample_size,
            cumulative_std, '-k')
        ax_mean.set_ylabel("Mean of " + label_legend, font=fonts.serif(), fontsize=24)
        ax_mean.grid(linestyle='--')
        ax_std.set_ylabel("Std of " + label_legend, font=fonts.serif(), fontsize=24)
        ax_std.grid(linestyle='--')
        ax_mean.set_xlabel("Number of samples [-]")
        ax_std.set_xlabel("Number of samples [-]")
        
        savefigure.save_as_png(fig_mean, "cumulative_mean_vs_sample_size_" + label_title)
        savefigure.save_as_png(fig_std, "cumulative_std_vs_sample_size_" + label_title)


def plot_cumulative_mean_vs_sample_size_indicators_retour_force_nulle(createfigure, savefigure, fonts):
    """Generate and save a figure depicting the variation of the cumulative mean
    and standard deviation of the dataset in terms of the size. 

    Args:
        createfigure (class): class used to provide consistent settings for the creation of figures
        savefigure (class):class used to provide consistent settings for saving figures
        fonts (class): class used to provide consistent fonts for the figures
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators_retour_force.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict]
    labels = ['alpha', 'beta', 'deltaf', 'deltafstar', 'a']
    label_dict = {'alpha': r"$\alpha' [kPa.s^{-1}]$", 'beta': r"$\beta [kPa.s^{-1}]$", 'deltaf':r"$\Delta F$ [kPa]", 'deltafstar':r"$\Delta F^*$ [-]", 'a': 'a [-]'}
    for i in range(len(indicators_dicts)):
        indicator_dict = indicators_dicts[i]
        label_title = labels[i]
        label_legend = label_dict[label_title]
        indicator_vector = list(indicator_dict.values())
        cumulative_mean, cumulative_std = compute_cumulative_mean_std(indicator_vector)
        sample_size = np.arange(1, len(cumulative_mean) + 1)
        fig_mean = createfigure.rectangle_figure(pixels=180)
        ax_mean = fig_mean.gca()
        fig_std = createfigure.rectangle_figure(pixels=180)
        ax_std = fig_std.gca()
        ax_mean.plot(
            sample_size,
            cumulative_mean, '-k')
        ax_std.plot(
            sample_size,
            cumulative_std, '-k')
        ax_mean.set_ylabel("Mean of " + label_legend, font=fonts.serif(), fontsize=24)
        ax_mean.grid(linestyle='--')
        ax_std.set_ylabel("Std of " + label_legend, font=fonts.serif(), fontsize=24)
        ax_std.grid(linestyle='--')
        ax_mean.set_xlabel("Number of samples [-]")
        ax_std.set_xlabel("Number of samples [-]")
        
        savefigure.save_as_png(fig_mean, "cumulative_mean_vs_sample_size_" + label_title + "retour_force_nulle")
        savefigure.save_as_png(fig_std, "cumulative_std_vs_sample_size_" + label_title + "retour_force_nulle")

def plot_outputs():
    """Create and save figure showing the temporal evolution of stress and displacement
    computed via the COMSOL model for every set of input parameters. The corresponding indicators are 
    also displayed.
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    # kwargs = {'marker':'o', 's':'10'}
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict]
    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
    labels = ['alpha', 'beta', 'deltaf', 'deltafstar', 'a']
    label_dict = {'alpha': r"$\alpha' [kPa.s^{-1}]$", 'beta': r"$\beta [kPa.s^{-1}]$", 'deltaf':r"$\Delta F$ [kPa]", 'deltafstar':r"$\Delta F^*$ [-]", 'a': 'a [-]'}
    for i in range(len(labels)):
        label = labels[i]
        indicator_dict = indicators_dicts[i]
        fig3D = createfigure.square_figure_10(pixels=180)
        ax3D = fig3D.add_subplot(111, projection='3d')
        fig_indicator_vs_elongation = createfigure.square_figure_10(pixels=180)
        ax_indicator_vs_elongation = fig_indicator_vs_elongation.gca()
        fig_indicator_vs_damage = createfigure.square_figure_10(pixels=180)
        ax_indicator_vs_damage = fig_indicator_vs_damage.gca()
        x = list(elongation_dict.values())
        y = list(damage_dict.values())
        z = list(indicator_dict.values())
        alpha_list = [1] * len(z)
        sorted_elongation = [x for _,x in sorted(zip(z,x))]
        sorted_damage = [y for _,y in sorted(zip(z,y))]
        sorted_indicator = sorted(z)
        ax3D.scatter(sorted_elongation, sorted_damage, sorted_indicator, c = sns.color_palette("flare", len(sorted_elongation)),
                       lw=0, antialiased=False, s = 100 + np.zeros_like(np.array(z)), alpha = 1)
        ax3D.set_xlabel(r"$\lambda$ [-]", font=fonts.serif(), fontsize=26)
        ax3D.set_ylabel(r"D [-]", font=fonts.serif(), fontsize=26)
        ax3D.set_zlabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax3D.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax3D.set_xticklabels([1, 1.1, 1.2, 1.3, 1.4, 1.5], font=fonts.serif(), fontsize=16)
        ax3D.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax3D.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], font=fonts.serif(), fontsize=16)
        ax3D.xaxis.pane.fill = False
        ax3D.yaxis.pane.fill = False
        ax3D.zaxis.pane.fill = False
        ax_indicator_vs_elongation.scatter(sorted_elongation, sorted_indicator, c = sns.color_palette("rocket", len(sorted_elongation)), s = 100 + np.zeros_like(np.array(z)))
        ax_indicator_vs_elongation.set_xlabel(r"$\lambda$ [-]", font=fonts.serif(), fontsize=26)
        ax_indicator_vs_elongation.set_ylabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax_indicator_vs_elongation.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax_indicator_vs_elongation.set_xticklabels([1, 1.1, 1.2, 1.3, 1.4, 1.5], font=fonts.serif(), fontsize=16)

        ax_indicator_vs_damage.scatter(sorted_damage, sorted_indicator, c = sns.color_palette("viridis", len(sorted_elongation)), s = 100 + np.zeros_like(np.array(z)))
        ax_indicator_vs_damage.set_xlabel(r"D [-]", font=fonts.serif(), fontsize=26)
        ax_indicator_vs_damage.set_ylabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax_indicator_vs_damage.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax_indicator_vs_damage.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8], font=fonts.serif(), fontsize=16)
        savefigure.save_as_png(fig3D, "3Dplot_v2_" + label)
        savefigure.save_as_png(fig_indicator_vs_damage, "2Dplot_v2_" + label + "_vs_damage")
        savefigure.save_as_png(fig_indicator_vs_elongation, "2Dplot_v2_" + label + "_vs_elongation")

def plot_outputs_force_nulle():
    """Create and save figure showing the temporal evolution of stress and displacement
    computed via the COMSOL model for every set of input parameters. The corresponding indicators are 
    also displayed.
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators_retour_force.pkl"
    # kwargs = {'marker':'o', 's':'10'}
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, delta_d] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, delta_d]
    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs_rf.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
    labels = ['alpha', 'beta', 'deltaf', 'deltafstar', 'deltad']
    label_dict = {'alpha': r"$\alpha' [kPa.s^{-1}]$", 'beta': r"$\beta [kPa.s^{-1}]$", 'deltaf':r"$\Delta F$ [kPa]", 'deltafstar':r"$\Delta F^*$ [-]", 'a': 'a [-]', 'deltad':r"$\Delta d$"}
    for i in range(len(labels)):
        label = labels[i]
        indicator_dict = indicators_dicts[i]
        fig3D = createfigure.square_figure_10(pixels=180)
        ax3D = fig3D.add_subplot(111, projection='3d')
        fig_indicator_vs_elongation = createfigure.square_figure_10(pixels=180)
        ax_indicator_vs_elongation = fig_indicator_vs_elongation.gca()
        fig_indicator_vs_damage = createfigure.square_figure_10(pixels=180)
        ax_indicator_vs_damage = fig_indicator_vs_damage.gca()
        x = list(elongation_dict.values())
        y = list(damage_dict.values())
        z = list(indicator_dict.values())
        alpha_list = [1] * len(z)
        sorted_elongation = [x for _,x in sorted(zip(z,x))]
        sorted_damage = [y for _,y in sorted(zip(z,y))]
        sorted_indicator = sorted(z)
        ax3D.scatter(sorted_elongation, sorted_damage, sorted_indicator, c = sns.color_palette("flare", len(sorted_elongation)),
                       lw=0, antialiased=False, s = 100 + np.zeros_like(np.array(z)), alpha = 1)
        ax3D.set_xlabel(r"$\lambda$ [-]", font=fonts.serif(), fontsize=26)
        ax3D.set_ylabel(r"D [-]", font=fonts.serif(), fontsize=26)
        ax3D.set_zlabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax3D.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax3D.set_xticklabels([1, 1.1, 1.2, 1.3, 1.4, 1.5], font=fonts.serif(), fontsize=16)
        ax3D.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax3D.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8], font=fonts.serif(), fontsize=16)
        ax3D.xaxis.pane.fill = False
        ax3D.yaxis.pane.fill = False
        ax3D.zaxis.pane.fill = False
        ax_indicator_vs_elongation.scatter(sorted_elongation, sorted_indicator, c = sns.color_palette("rocket", len(sorted_elongation)), s = 100 + np.zeros_like(np.array(z)))
        ax_indicator_vs_elongation.set_xlabel(r"$\lambda$ [-]", font=fonts.serif(), fontsize=26)
        ax_indicator_vs_elongation.set_ylabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax_indicator_vs_elongation.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax_indicator_vs_elongation.set_xticklabels([1, 1.1, 1.2, 1.3, 1.4, 1.5], font=fonts.serif(), fontsize=16)

        ax_indicator_vs_damage.scatter(sorted_damage, sorted_indicator, c = sns.color_palette("viridis", len(sorted_elongation)), s = 100 + np.zeros_like(np.array(z)))
        ax_indicator_vs_damage.set_xlabel(r"D [-]", font=fonts.serif(), fontsize=26)
        ax_indicator_vs_damage.set_ylabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax_indicator_vs_damage.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax_indicator_vs_damage.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8], font=fonts.serif(), fontsize=16)
        savefigure.save_as_png(fig3D, "3Dplot_v2_" + label + "_retour_force_nulle")
        savefigure.save_as_png(fig_indicator_vs_damage, "2Dplot_v2_" + label + "_vs_damage_retour_force_nulle")
        savefigure.save_as_png(fig_indicator_vs_elongation, "2Dplot_v2_" + label + "_vs_elongation_retour_force_nulle")


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    # plot_outputs()
    # plot_cumulative_mean_vs_sample_size_indicators(createfigure, savefigure, fonts)
    # plot_cumulative_mean_vs_sample_size_indicators_retour_force_nulle(createfigure, savefigure, fonts)
    plot_outputs_force_nulle()
