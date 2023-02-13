import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import glob
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import utils
import indentation.caracterization.tiguida.tensile_testings_porks as sk_ttp
import indentation.caracterization.tiguida.tensile_testings_porks.figures.utils as ttp_fu

plt.ion()

# Constitutive laws
def yeoh_simple(x, a, b):
    return 2*(x - 1/x**2)*(a + 2*b*(x**2 + 2/x - 3))

def yeoh(x, c1, c2, c3):
    return 2*(x - 1/x**2)*(c1 + (x**2 + 2/x - 3)*(2*c2 + 3*c3*(x**2 + 2/x - 3)))

def mooney(x, c1, c2):
    return 2*(x - 1/x**2)*(c1 + c2/x)

def ogden(x, c1, c2, c3, c4, c5, c6):
    return (c1 * (x**(c2-1) - x**(-(c2/2)-1))) + (c3 * (x**(c4-1) - x**(-(c4/2)-1))) + (c5 * (x**(c6-1) - x**(-(c6/2)-1)))

def humphrey(x, c1, c2):
    return 2*(x - 1/x**2)*c1*c2*np.exp(c2*(x**2 + 2/x - 3))

def veronda(x, c1, c2):
    return 2*(x - 1/x**2)*c1*c2*(np.exp(c2*(x**2 + 2/x - 3))-(1/2*x))

def martins(x,c1,c2,c3,c4):
    return 2*(x - 1/x**2)*c1*c2*np.exp(c2*(x**2 + 2/x - 3)) + 2*(x-1)*c3*c4*np.exp(c3*(x-1)**2)



# DIRECTORY OF THE FILES

all_datas, cochon_ids, _ = utils.load_data()


col = ["red", "blue", "brown", "green", "purple", "gold", "turquoise", "orange", "magenta", "gray"]
save_results_to = ttp_fu.make_folder_for_figures()


for i in range(len(all_datas)):
    fig, ax = plt.subplots()
    datas = all_datas[i]
    cochon_id = int(cochon_ids[i])
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
        
        stressu = stress[beg:end] - stress[beg]
        stretchu = stretch[beg:end]/stretch[beg]
        strainu = (stretchu-1)*100


        popt1, pcov1 = opt.curve_fit(yeoh, stretchu, stressu, maxfev= 100000)
        # print(prefix, popt1) # to print the coefficients
        # popt2, pcov2 = opt.curve_fit(mooney, stretchu, stressu, maxfev=100000)
        # popt3, pcov3 = opt.curve_fit(ogden, stretchu, stressu, maxfev=100000)
        # popt4, pcov4 = opt.curve_fit(humphrey, stretchu, stressu, maxfev=100000)
        # popt5, pcov5 = opt.curve_fit(veronda, stretchu, stressu, maxfev=100000)
        # popt6, pcov6 = opt.curve_fit(martins, stretchu, stressu, maxfev=100000)

        # TO PLOT ONLY EVERY 1000 POINTS

        ax.plot(strainu[0:(len(strainu)-1):1000], stressu[0:(len(strainu)-1):1000], 'o', mfc='none',color=col[i], label= prefix)
        ax.plot(strainu[0:(len(strainu)-1):1000], yeoh(stretchu[0:(len(strainu)-1):1000], *popt1), color=col[i])

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



    filename = 'identification_yeoh_cochon-' + str(cochon_id) + '.jpg'
    fig.savefig(save_results_to /filename, dpi=300)

