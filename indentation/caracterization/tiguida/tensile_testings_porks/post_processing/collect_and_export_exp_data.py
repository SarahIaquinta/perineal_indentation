import glob
from indentation import make_data_folder
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import indentation.caracterization.tiguida.tensile_testings_porks.post_processing.utils as tiguida_postpro_utils

# DIRECTORY OF THE FILES
def load_data():
    path_to_data = make_data_folder('caracterization\\tiguida\\tensile_testings_porks\\raw_data')
    entries = os.listdir(path_to_data)
    path_to_data_as_string = str(path_to_data)
    datas_cochon4 = sorted(glob.glob(path_to_data_as_string + r'\%04*-1%**.txt',recursive = True))
    datas_cochon5 = sorted(glob.glob(path_to_data_as_string + r'\%05*-1%**.txt',recursive = True))
    datas_cochon7 = sorted(glob.glob(path_to_data_as_string + r'\%07*-1%**.txt',recursive = True))
    datas_cochon8 = sorted(glob.glob(path_to_data_as_string + r'\%08*-1%**.txt',recursive = True))
    datas_cochon9 = sorted(glob.glob(path_to_data_as_string + r'\%09*-1%**.txt',recursive = True))
    all_datas = [datas_cochon4, datas_cochon5, datas_cochon7, datas_cochon8, datas_cochon9]
    cochon_ids = [4, 5, 7, 8, 9]
    nb_of_cochons = len(cochon_ids)
    return all_datas, cochon_ids, nb_of_cochons

def collect_data():
    all_datas, cochon_ids, nb_of_cochons =  load_data()
    dict_datas_total_time = {}
    dict_datas_total_stress = {}
    dict_datas_total_stretch = {}
    dict_datas_undamaged_time = {}
    dict_datas_undamaged_stress = {}
    dict_datas_undamaged_stretch = {}
    dict_pigs = {}
    dict_tissue = {}
    for c in range(len(all_datas)):
        datas = all_datas[c]
        for i, element in enumerate(datas):
            data = np.loadtxt(element, delimiter = '\t', skiprows = 20)

            lo = float(element.split("%")[2])
            w = float(element.split("%")[3])
            t = float(element.split("%")[4])
            section = w * t

            time = data[:,0]
            force_gf = -data[:,2]
            gf_N = 0.00980665
            force_N = force_gf * gf_N
            stress = force_N*1000 / section

            disp = -data[:,1]
            strain = (disp / lo) * 100
            stretch = (strain/100) + 1

            prefix = element.split("%")[1] # for the filename
            pig_number = prefix[0:2]
            tissue = prefix[3] 

            peaks, _ = find_peaks(stress, prominence=10) # to find drops in stress that indicate damage or fiber rupture

            if len(peaks) == 0:
                end = np.argmax(stress)
            else:
                end = peaks[0]

            beg = np.argmin(stretch)
            
            stress_total = stress[:] - stress[beg]
            time_total = time[:] - time[beg]
            stretch_total = stretch[:]/stretch[beg]
            # strain_total = (stretch_total-1)*100
            
            stress_undamaged = stress[beg:end] - stress[beg]
            time_undamaged = time[beg:end] - time[beg]
            stretch_undamaged = stretch[beg:end]/stretch[beg]
            # strain_undamaged = (stretch_undamaged-1)*100
            
            dict_pigs[element] = pig_number
            dict_tissue[element] = tissue
            
            
            dict_datas_total_time[element] = time_total
            dict_datas_total_stress[element] = stress_total
            dict_datas_total_stretch[element] = stretch_total
            dict_datas_undamaged_time[element] = time_undamaged
            dict_datas_undamaged_stress[element] = stress_undamaged
            dict_datas_undamaged_stretch[element] = stretch_undamaged
    return dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch

def plot_data(file_test):
    dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch = collect_data()
    undamaged_time = dict_datas_undamaged_time[file_test]
    undamaged_stress = dict_datas_undamaged_stress[file_test]
    undamaged_stretch = dict_datas_undamaged_stretch[file_test]

    total_time = dict_datas_total_time[file_test]
    total_stress = dict_datas_total_stress[file_test]
    total_stretch = dict_datas_total_stretch[file_test]
    
    plt.figure()
    plt.plot(total_time, total_stress)
    plt.plot(undamaged_time, undamaged_stress, ':r')
    plt.show()


if __name__ == "__main__":
    dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch =  tiguida_postpro_utils.extract_exp_data_as_pkl()

    print('hello')
    
            