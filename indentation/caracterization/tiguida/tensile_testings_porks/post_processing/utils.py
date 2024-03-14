import glob
from indentation import make_data_folder
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from indentation.caracterization.tiguida.tensile_testings_porks.post_processing.collect_and_export_exp_data import collect_data
import pickle



def export_exp_data_as_pkl():
    dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch = collect_data()
    pkl_filename = "processed_data_from_tiguida_uniaxial_tensile_testings.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\tiguida\tensile_testings_porks\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump([dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch], f)


def extract_exp_data_as_pkl():
    pkl_filename = "processed_data_from_tiguida_uniaxial_tensile_testings.pkl"
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\caracterization\tiguida\tensile_testings_porks\processed_data'
    complete_pkl_filename = path_to_processed_data + "/" + pkl_filename
    with open(complete_pkl_filename, "rb") as f:
        [dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch] = pickle.load(f)
    return dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch

def get_data_from_datafile(datafile):
    dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch =  extract_exp_data_as_pkl()
    pig_number = dict_pigs[datafile]
    tissue = dict_tissue[datafile]
    total_time = dict_datas_total_time[datafile]
    total_stress = dict_datas_total_stress[datafile]
    total_stretch = dict_datas_total_stretch[datafile]
    undamaged_time = dict_datas_undamaged_time[datafile]
    undamaged_stress = dict_datas_undamaged_stress[datafile]
    undamaged_stretch = dict_datas_undamaged_stretch[datafile]
    return pig_number, tissue, total_time, total_stress, total_stretch, undamaged_time, undamaged_stress, undamaged_stretch
    
def get_exp_datafile_list():
    dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch =  extract_exp_data_as_pkl()
    filelist = [f for f in dict_pigs.keys()]
    return filelist
