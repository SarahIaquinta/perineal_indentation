import glob
from indentation import make_data_folder
import os

def make_folder_for_figures():
    figure_folder = 'caracterization\\tiguida\\tensile_testings_porks\\figures' 
    path_to_figure_folder = make_data_folder(figure_folder)
    return path_to_figure_folder