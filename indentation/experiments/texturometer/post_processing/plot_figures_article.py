import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.texturometer.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.laser.post_processing.read_file as rf
import indentation.experiments.laser.post_processing.display_profiles as dp
from indentation.experiments.laser.post_processing.read_file import Files
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter
import pickle
import csv
import pandas as pd
import statistics
import seaborn as sns
import pandas as pd
from indentation.experiments.zwick.post_processing.utils import find_nearest

def get_data_from_file():
    path_to_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\raw_data'
    complete_excel_filename = path_to_data + "/" + '230331_FF2_7.xlsx'
    # pd.read_excel(path_to_metadatafile, sheet_name='laser', header=1, names=["Id", "imposed_disp"], usecols="A:B", decimal=',')
    datas = pd.read_excel(complete_excel_filename, header=0, decimal=',')
    disp = datas.U
    force = datas.F
    max_disp = np.max(disp)
    disp_at_80 = find_nearest(disp, max_disp * 0.8)
    index_where_disp_is_disp_at_80 = np.where(disp == find_nearest(disp, disp_at_80))[0][0]
    force_at_80 = force.to_list()[index_where_disp_is_disp_at_80]
    disp_at_20 = find_nearest(disp, max_disp * 0.2)
    index_where_disp_is_disp_at_20 = np.where(disp == find_nearest(disp, disp_at_20))[0][0]
    force_at_20 = force.to_list()[index_where_disp_is_disp_at_20]
    return disp, force, disp_at_20, force_at_20, disp_at_80, force_at_80



def plot_force_vs_disp(createfigure, savefigure, fonts):
    disp, force, disp_at_20, force_at_20, disp_at_80, force_at_80 = get_data_from_file()
    fig_force_vs_disp = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_disp = fig_force_vs_disp.gca()
    ax_force_vs_disp.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp.set_ylim([0,90])
    ax_force_vs_disp.set_xlim([0,10.5])
    ax_force_vs_disp.set_xticks([0, 2, 4, 6, 8, 10])
    ax_force_vs_disp.set_xticklabels([0, 2, 4, 6, 8, 10], font=fonts.serif(), fontsize=20)
    ax_force_vs_disp.set_yticks([0, 20, 40, 60, 80])
    ax_force_vs_disp.set_yticklabels([0, 20, 40, 60, 80], font=fonts.serif(), fontsize=20)
    color_rocket = sns.color_palette("rocket")
    color_F20 = color_rocket[3]
    color_F80 = color_rocket[1]  
    ax_force_vs_disp.plot(disp, force, '-', color='k')
    ax_force_vs_disp.plot([disp_at_20, disp_at_20], [force.to_list()[0], force_at_20], '--', color=color_F20)
    ax_force_vs_disp.plot([disp.to_list()[0], disp_at_20], [force_at_20, force_at_20], '--', color=color_F20)
    ax_force_vs_disp.plot([disp_at_80, disp_at_80], [force.to_list()[0], force_at_80], '--', color=color_F80)
    ax_force_vs_disp.plot([disp.to_list()[0], disp_at_80], [force_at_80, force_at_80], '--', color=color_F80)
    ax_force_vs_disp.annotate(r"$F_{20\%}$", ((disp.to_list()[0]+ disp_at_20)/2 -0.5 , force_at_20 + 2), color=color_F20, font=fonts.serif(), fontsize=22)
    ax_force_vs_disp.annotate(r"$F_{80\%}$", ((disp.to_list()[0]+ disp_at_80)/2 -0.5 , force_at_80 + 2), color=color_F80, font=fonts.serif(), fontsize=22)
# ax_A_vs_force80_1.annotate(maturation_dict_plots[date], (mean_force80_FF1_dict[date] +0.04, mean_A_FF1_dict[date]+0.02), color = color[7], bbox=dict(boxstyle="round", fc="0.8", alpha=0.4))

    savefigure.save_as_png(fig_force_vs_disp, 'texturometer_article')

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    plot_force_vs_disp(createfigure, savefigure, fonts)