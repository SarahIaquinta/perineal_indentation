import glob
from indentation import make_data_folder
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from indentation.caracterization.tiguida.tensile_testings_porks.post_processing.collect_and_export_exp_data import collect_data
import pickle



def export_exp_data_as_pkl():
    dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch = collect_data()
    pkl_filename = "processed_data_from_tiguida_uniaxial_tensile_testings.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\tiguida\tensile_testings_porks\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump([dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch], f)


def extract_exp_data_as_pkl():
    pkl_filename = "processed_data_from_tiguida_uniaxial_tensile_testings.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\tiguida\tensile_testings_porks\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch] = pickle.load(f)
    return dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch