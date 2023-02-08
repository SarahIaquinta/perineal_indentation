import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import glob
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import seaborn as sns

plt.ion()


# DIRECTORY OF THE FILES
datas_cochon7 = sorted(glob.glob(r'C:\Users\siaquinta\Documents\Projet Périnée\02_Caracterisation\Tiguida\Traction_porc\data_traction\%07*-1%**.txt',recursive = True))
datas_cochon4 = sorted(glob.glob(r'C:\Users\siaquinta\Documents\Projet Périnée\02_Caracterisation\Tiguida\Traction_porc\data_traction\%04*-1%**.txt',recursive = True))
datas_cochon5 = sorted(glob.glob(r'C:\Users\siaquinta\Documents\Projet Périnée\02_Caracterisation\Tiguida\Traction_porc\data_traction\%05*-1%**.txt',recursive = True))
datas_cochon8 = sorted(glob.glob(r'C:\Users\siaquinta\Documents\Projet Périnée\02_Caracterisation\Tiguida\Traction_porc\data_traction\%08*-1%**.txt',recursive = True))
datas_cochon9 = sorted(glob.glob(r'C:\Users\siaquinta\Documents\Projet Périnée\02_Caracterisation\Tiguida\Traction_porc\data_traction\%09*-1%**.txt',recursive = True))
all_datas = [datas_cochon4, datas_cochon5, datas_cochon7, datas_cochon8, datas_cochon9]
cochon_ids = [4, 5, 7, 8, 9]
nb_of_cochons = len(cochon_ids)
col = ["red", "blue", "brown", "green", "purple", "gold", "turquoise", "orange", "magenta", "gray"]

colors_reds = sns.color_palette("Reds", nb_of_cochons)
colors_blues = sns.color_palette("Blues", nb_of_cochons)
colors_brown = sns.color_palette("YlOrBr", nb_of_cochons)
colors_greens = sns.color_palette("Greens", nb_of_cochons)
colors_purple = sns.color_palette("RdPu", nb_of_cochons)


for i in range(len(all_datas)):
    fig, ax = plt.subplots()
    datas = all_datas[i]
    cochon_id = int(cochon_ids[i])
    col_cochon = [colors_reds[i], colors_blues[i], colors_brown[i], colors_greens[i], colors_purple[i]]
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
        n = prefix[0:2]

        peaks, _ = find_peaks(stress, prominence=10) # to find drops in stress that indicate damage or fiber rupture

        if len(peaks) == 0:
            end = np.argmax(stress)
        else:
            end = peaks[0]

        beg = np.argmin(stretch)
        
        stressu = stress[:] - stress[beg]
        stretchu = stretch[:]/stretch[beg]
        strainu = (stretchu-1)*100

        # TO PLOT ONLY EVERY 1000 POINTS
        ax.plot(strainu[0:(len(strainu)-1):1000], stressu[0:(len(strainu)-1):1000], '-', mfc='none',color=col[i], label= prefix)

        # TO PLOT ALL MODELS FOR THE SAME EXPERIMENTAL CURVE

        # ax.plot(strainu, yeoh(stretchu, *popt1), ':', color=col[1], label='Yeoh')
        # ax.plot(strainu, mooney(stretchu, *popt2), ':', color=col[2], label='Mooney-Rivlin')
        # ax.plot(strainu, ogden(stretchu, *popt3), '-.', color=col[3], label='Ogden')
        # ax.plot(strainu, humphrey(stretchu, *popt4), linestyle=(0, (1, 10)), color=col[4], label='Humphrey')
        # ax.plot(strainu, veronda(stretchu, *popt5), linestyle=(0, (3, 10, 1, 10)), color=col[5], label='Veronda-Westmann')
        # ax.plot(strainu, martins(stretchu, *popt6), linestyle=(0, (5, 10)), color=col[6], label='Martins')


        ax.grid(True)
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Stress (kPa)')
        ax.legend()

        # TO SAVE THE FIGURE

        save_results_to = r'C:\Users\siaquinta\Documents\Projet Périnée\02_Caracterisation\Tiguida\Traction_porc'
        fig.savefig(save_results_to + '/tensile_testing_cochon-' + str(cochon_id) + '.jpg', dpi=300)




