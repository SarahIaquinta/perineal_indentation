import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.laser.post_processing.read_file as rf
import indentation.experiments.laser.post_processing.display_profiles as dp
from indentation.experiments.laser.post_processing.read_file import Files
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter
import pickle

def extract_data_from_recovery_pkl_file(recovery_pkl_file):
    complete_pkl_filename = utils.get_path_to_processed_data() / recovery_pkl_file
    with open(complete_pkl_filename, "rb") as f:
        [filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export] = pickle.load(f)
    return filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export

def transform_recovery_data_in_dictionnaries(recovery_pkl_file):
    filenames_from_pkl, delta_d_from_pkl, delta_d_stars_from_pkl, d_min_from_pkl, A_from_pkl = extract_data_from_recovery_pkl_file(recovery_pkl_file)
    dates_dict = dict(zip(filenames_from_pkl, [f[0:6] for f in filenames_from_pkl]))
    piece_dict = dict(zip(filenames_from_pkl, [f[7:9] for f in filenames_from_pkl]))
    maturation_dict = {'230327': 6, '230331': 10, '230403': 13, '230407': 17, '230411':21}
    delta_d_dict = dict(zip(filenames_from_pkl, delta_d_from_pkl))
    delta_d_star_dict = dict(zip(filenames_from_pkl, delta_d_stars_from_pkl))
    d_min_dict = dict(zip(filenames_from_pkl, d_min_from_pkl))
    A_dict = dict(zip(filenames_from_pkl, A_from_pkl))
    return filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict

def plot_variables_with_date(filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, createfigure, savefigure, fonts):
    colors_with_piece = {'FF': 'r', 'RD': 'b'}
    alpha_with_maturation = {6: 0.25, 10: 0.4, 13: 0.6, 17:0.7, 21: 0.85}
    
    fig = createfigure.rectangle_rz_figure(pixels=180)
    ax = fig.gca()
    
    dates = [dates_dict[key] for key in dates_dict.keys()]
    pieces = [piece_dict[key] for key in piece_dict.keys()]

    for date in dates:
        for piece in pieces:
            maturation_list = []
            A_list = []
            id = date + '_' + piece
            for filename in filenames_from_pkl:
                if dates_dict[filename] == date:
                    if piece_dict[filename] == piece:
                        maturation = maturation_dict[date]
                        maturation_list.append(maturation)
                        A_list.append(A_dict[filename])
            ax.plot(maturation_list, A_list, marker="o",  markerfacecolor = colors_with_piece[piece[0:2]], markeredgecolor=None, alpha = alpha_with_maturation[maturation], linewidths=None)
    savefigure.save_as_png(fig, "A_with_maturation_meat")
                    
    
    


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    recovery_pkl_file = 'recoveries_meat.pkl'
    filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict = transform_recovery_data_in_dictionnaries(recovery_pkl_file)
    print('hello')
    plot_variables_with_date(filenames_from_pkl, dates_dict, piece_dict, maturation_dict, delta_d_dict, delta_d_star_dict, d_min_dict, A_dict, createfigure, savefigure, fonts)