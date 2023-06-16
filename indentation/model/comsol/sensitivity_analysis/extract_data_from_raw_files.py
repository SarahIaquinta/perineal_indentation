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

def get_stress():
    ids_list, _, _ = utils.extract_inputs_from_pkl()
    path_to_file_stress = utils.reach_data_path() / 'stress_silicone.xlsx'
    stress_data = pd.read_excel(path_to_file_stress, sheet_name='output', header=0)
    times = stress_data.time
    time_list = [float(t) for t in times.to_list()]
    stress_dict = {}
    for id in ids_list:
        stress_id = stress_data[id]
        stress_dict[id] = [float(s) for s in stress_id.to_list()]
    return time_list, stress_dict

def get_disp():
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

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    ids_list, elongation_dict, damage_dict = utils.extract_inputs_from_pkl()
    time_list, disp_dict = utils.extract_disp_from_pkl()
    time_list, stress_dict = utils.extract_stress_from_pkl()
    print('hello')

