import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
from indentation.model.comsol.sensitivity_analysis import utils
import os
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

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    plot_cumulative_mean_vs_sample_size_indicators(createfigure, savefigure, fonts)