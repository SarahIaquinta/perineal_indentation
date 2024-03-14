import glob
from indentation import make_data_folder
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from indentation.caracterization.tiguida.tensile_testings_porks.figures.utils import SaveFigure, CreateFigure, Fonts
from scipy.ndimage import gaussian_filter1d

import indentation.caracterization.tiguida.tensile_testings_porks.post_processing.utils as tiguida_postpro_utils
import seaborn as sns
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

def detect_inflexion(vector):
    smooth = gaussian_filter1d(vector, 100)
    grad = np.gradient(np.gradient(smooth))
    infls = np.where(grad<-0.0001)[0]
    return infls

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

            peaks = detect_inflexion(stress)
            # peaks, _ = find_peaks(stress, prominence=3, plateau_size=1) # to find drops in stress that indicate damage or fiber rupture

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
            
            
            dict_datas_total_time[element] = np.array(time_total)
            dict_datas_total_stress[element] = np.array(stress_total)
            dict_datas_total_stretch[element] = np.array(stretch_total)
            dict_datas_undamaged_time[element] = np.array(time_undamaged)
            dict_datas_undamaged_stress[element] = np.array(stress_undamaged)
            dict_datas_undamaged_stretch[element] = np.array(stretch_undamaged)
    return dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch

def plot_data(datafile_list, given_tissue):
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    k=0
    for datafile in datafile_list:
        pig_number, tissue, total_time, total_stress, total_stretch, undamaged_time, undamaged_stress, undamaged_stretch = tiguida_postpro_utils.get_data_from_datafile(datafile)
        if tissue == given_tissue:
            k += 1
    color_black = sns.color_palette("Greys", as_cmap=False, n_colors=k )
    color_red = sns.color_palette("Reds", as_cmap=False, n_colors=k )
    k=0
    for datafile in datafile_list:
        pig_number, tissue, total_time, total_stress, total_stretch, undamaged_time, undamaged_stress, undamaged_stretch = tiguida_postpro_utils.get_data_from_datafile(datafile)
        if tissue == given_tissue:
            if pig_number != "04" :
                pig_number_legend = int(float(pig_number)) - 4
                ax_stress_vs_elongation.plot(total_stretch, total_stress, '-', color=color_black[k], label=pig_number_legend)
                ax_stress_vs_elongation.plot(undamaged_stretch, undamaged_stress, '-', color=color_red[k])
                k += 1
            ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
            ax_stress_vs_elongation.set_ylabel(r"$\sigma_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
            ax_stress_vs_elongation.legend(prop=fonts.serif_rz_legend(), loc='upper right', framealpha=0.7)
            fig_name = "stress_vs_elong_exp_undamaged_vs_total_allpigs_" + tissue
            savefigure.save_as_svg(fig_stress_vs_elongation, fig_name)


def plot_all_data(datafile, tissue):
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    pig_number, tissue, total_time, total_stress, total_stretch, undamaged_time, undamaged_stress, undamaged_stretch = tiguida_postpro_utils.get_data_from_datafile(datafile)
   

    ax_stress_vs_elongation.plot(total_stretch, total_stress, '-k')
    ax_stress_vs_elongation.plot(undamaged_stretch, undamaged_stress, ':r')
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\sigma_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    fig_name = "stress_vs_elong_exp_undamaged_vs_total_" + pig_number + "_" + tissue
    savefigure.save_as_svg(fig_stress_vs_elongation, fig_name)    

if __name__ == "__main__":
    tiguida_postpro_utils.export_exp_data_as_pkl()

    datafile_list = tiguida_postpro_utils.get_exp_datafile_list()
    plot_data(datafile_list, 'p')
    # for datafile in datafile_list:
    #     plot_data(datafile)
    print('hello')
    
            