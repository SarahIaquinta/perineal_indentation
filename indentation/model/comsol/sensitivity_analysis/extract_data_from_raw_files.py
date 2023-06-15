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
import multiprocessing as mp
# from indentation.experiments.zwick.post_processing.utils import find_nearest
from functools import partial

def get_inputs():
    path_to_file = utils.reach_data_path() / 'disp_silicone.xlsx'
    input_data = pd.read_excel(path_to_file, sheet_name='input', header=0, names=["Id", "elongation", "damage"]) 
    ids = input_data.Id
    elongations = input_data.elongation
    damages = input_data.damage
    ids_list = ids.tolist()
    elongation_dict = {ids.tolist()[i]: elongations.tolist()[i] for i in range(len(ids.tolist()))}
    damage_dict = {ids.tolist()[i]: damages.tolist()[i] for i in range(len(ids.tolist()))}
    return ids_list, elongation_dict, damage_dict

def get_output2():
    path_to_file_stress = utils.reach_data_path() / 'stress_silicone.xlsx'
    stress_data = pd.read_excel(path_to_file_stress, sheet_name='output', header=0)
    times = stress_data.time
    ids = stress_data.Id
    elongations = stress_data.elongation
    damages = stress_data.damage
    elongation_dict = {ids.tolist()[i]: elongations.tolist()[i] for i in range(len(ids.tolist()))}
    damage_dict = {ids.tolist()[i]: damages.tolist()[i] for i in range(len(ids.tolist()))}
    return elongation_dict, damage_dict



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    utils.export_inputs_as_pkl()
    # ids_list, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()

    print('hello')

