import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
from indentation.model.comsol.sensitivity_analysis import utils
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pandas as pd
import scipy
import skimage
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.signal import argrelextrema
import multiprocessing as mp
from functools import partial
from indentation.experiments.zwick.post_processing.utils import find_nearest
from datetime import datetime




def compute_cumulative_mean_std(vector):
    cumulative_mean = np.zeros(len(vector))
    cumulative_std = np.zeros_like(cumulative_mean)
    cumulative_mean[0] = vector[0]
    cumulative_std[0] = 0
    for i in range(1, len(vector)):
        cumulative_mean[i] = np.mean(vector[0:i])
        cumulative_std[i] = np.std(vector[0:i])
    return cumulative_mean, cumulative_std

def plot_cumulative_mean_vs_sample_size_indicators(createfigure, savefigure, fonts):
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

def plot_outputs():
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
        sorted_elongation = [x for _,x in sorted(zip(z,x))]
        sorted_damage = [y for _,y in sorted(zip(z,y))]
        sorted_indicator = sorted(z)
        ax3D.scatter(sorted_elongation, sorted_damage, sorted_indicator, c = sns.color_palette("flare", len(z)),
                       lw=0, antialiased=False, s = 100 + np.zeros_like(np.array(z)), alpha = 1 + np.zeros_like(np.array(z)))
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
        ax_indicator_vs_elongation.scatter(sorted_elongation, sorted_indicator, c = sns.color_palette("rocket", len(z)), s = 100 + np.zeros_like(np.array(z)))
        ax_indicator_vs_elongation.set_xlabel(r"$\lambda$ [-]", font=fonts.serif(), fontsize=26)
        ax_indicator_vs_elongation.set_ylabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax_indicator_vs_elongation.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax_indicator_vs_elongation.set_xticklabels([1, 1.1, 1.2, 1.3, 1.4, 1.5], font=fonts.serif(), fontsize=16)

        ax_indicator_vs_damage.scatter(sorted_damage, sorted_indicator, c = sns.color_palette("viridis", len(z)), s = 100 + np.zeros_like(np.array(z)))
        ax_indicator_vs_damage.set_xlabel(r"D [-]", font=fonts.serif(), fontsize=26)
        ax_indicator_vs_damage.set_ylabel(label_dict[label], font=fonts.serif(), fontsize=26)
        ax_indicator_vs_damage.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax_indicator_vs_damage.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8], font=fonts.serif(), fontsize=16)
        savefigure.save_as_png(fig3D, "3Dplot_" + label)
        savefigure.save_as_png(fig_indicator_vs_damage, "2Dplot_" + label + "_vs_damage")
        savefigure.save_as_png(fig_indicator_vs_elongation, "2Dplot_" + label + "_vs_elongation")


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    plot_outputs()
    # plot_cumulative_mean_vs_sample_size_indicators(createfigure, savefigure, fonts)