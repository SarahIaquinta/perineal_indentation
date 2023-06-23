""""
This file is used to generate nice looking figures for illustrating articles.
"""

import numpy as np
from matplotlib import pyplot as plt
from indentation.experiments.texturometer.figures.utils import CreateFigure, Fonts, SaveFigure
import seaborn as sns
import pickle
import pandas as pd
import seaborn as sns
import pandas as pd
from indentation.experiments.zwick.post_processing.utils import find_nearest

def get_data_from_file():
    path_to_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\raw_data'
    complete_excel_filename = path_to_data + "\\" + '230331_FF2_7.xlsx'
    # pd.read_excel(path_to_metadatafile, sheet_name='laser', header=1, names=["Id", "imposed_disp"], usecols="A:B", decimal=',')
    datas = pd.read_excel(complete_excel_filename, header=0, decimal=',')
    disp = datas.U
    force = datas.F
    max_disp = np.max(disp)
    disp_at_80 = find_nearest(disp,(max_disp+2) * 0.8)
    index_where_disp_is_disp_at_80 = np.where(disp == find_nearest(disp, disp_at_80))[0][0]
    force_at_80 = force.to_list()[index_where_disp_is_disp_at_80]
    disp_at_20 = find_nearest(disp,(max_disp+2) * 0.2)
    index_where_disp_is_disp_at_20 = np.where(disp == find_nearest(disp, disp_at_20))[0][0]
    force_at_20 = force.to_list()[index_where_disp_is_disp_at_20]
    return disp, force, disp_at_20, force_at_20, disp_at_80, force_at_80

def plot_force_vs_disp(createfigure, savefigure, fonts):
    disp, force, disp_at_20, force_at_20, disp_at_80, force_at_80 = get_data_from_file()
    fig_force_vs_disp = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_disp = fig_force_vs_disp.gca()
    ax_force_vs_disp.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp.set_ylabel(r"force [N]", font=fonts.serif(), fontsize=26)
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

def plot_forces_with_maturation():
    path_to_processed_data = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\experiments\texturometer\processed_data'
    complete_pkl_filename = path_to_processed_data + "/forces_mean_std.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = pickle.load(f)
    dates_to_use = ['230331', '230403', '230407']
    maturation_dict = {'230331': 10, '230403': 13, '230407': 17}
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[3]}
    kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[1]}
    maturation_FF_dict = {k: v - 0.1 for k, v in maturation_dict.items()}
    maturation_RDG_dict = {k: v + 0.1 for k, v in maturation_dict.items()}
    fig_force20 = createfigure.rectangle_figure(pixels=180)
    ax_force20 = fig_force20.gca()
    [dates, mean_force20_FF1_dict, std_force20_FF1_dict, mean_force80_FF1_dict, std_force80_FF1_dict,
             mean_force20_FF2_dict, std_force20_FF2_dict, mean_force80_FF2_dict, std_force80_FF2_dict,
             mean_force20_RDG1_dict, std_force20_RDG1_dict, mean_force80_RDG1_dict, std_force80_RDG1_dict,
             mean_force20_RDG2_dict, std_force20_RDG2_dict, mean_force80_RDG2_dict, std_force80_RDG2_dict,
             mean_force20_FF_dict, std_force20_FF_dict, mean_force80_FF_dict, std_force80_FF_dict,
             mean_force20_RDG_dict, std_force20_RDG_dict, mean_force80_RDG_dict, std_force80_RDG_dict
             ] = [dates, {d:mean_force20_FF1_dict[d] for d in dates_to_use}, {d:std_force20_FF1_dict[d] for d in dates_to_use}, {d:mean_force80_FF1_dict[d] for d in dates_to_use}, {d:std_force80_FF1_dict[d] for d in dates_to_use},
             {d:mean_force20_FF2_dict[d] for d in dates_to_use}, {d:std_force20_FF2_dict[d] for d in dates_to_use}, {d:mean_force80_FF2_dict[d] for d in dates_to_use}, {d:std_force80_FF2_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG1_dict[d] for d in dates_to_use}, {d:std_force20_RDG1_dict[d] for d in dates_to_use}, {d:mean_force80_RDG1_dict[d] for d in dates_to_use}, {d:std_force80_RDG1_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG2_dict[d] for d in dates_to_use}, {d:std_force20_RDG2_dict[d] for d in dates_to_use}, {d:mean_force80_RDG2_dict[d] for d in dates_to_use}, {d:std_force80_RDG2_dict[d] for d in dates_to_use},
             {d:mean_force20_FF_dict[d] for d in dates_to_use}, {d:std_force20_FF_dict[d] for d in dates_to_use}, {d:mean_force80_FF_dict[d] for d in dates_to_use}, {d:std_force80_FF_dict[d] for d in dates_to_use},
             {d:mean_force20_RDG_dict[d] for d in dates_to_use}, {d:std_force20_RDG_dict[d] for d in dates_to_use}, {d:mean_force80_RDG_dict[d] for d in dates_to_use}, {d:std_force80_RDG_dict[d] for d in dates_to_use}
             ]
    ax_force20.errorbar(list(maturation_FF_dict.values()), list(mean_force20_FF_dict.values()), yerr=list(std_force20_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_force20.errorbar(list(maturation_RDG_dict.values()), list(mean_force20_RDG_dict.values()), yerr=list(std_force20_RDG_dict.values()), lw=0,  label='RDG', **kwargs_RDG)
    ax_force20.legend(prop=fonts.serif(), loc='lower left', framealpha=0.7)
    ax_force20.set_xlabel('Durée de stockage [jours]', font=fonts.serif(), fontsize=26)
    ax_force20.set_ylabel(r'$F_{20\%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_force20.set_xticks([10, 13, 17])
    ax_force20.set_xticklabels([10, 13, 17], font=fonts.serif(), fontsize=24)
    ax_force20.set_yticks([0, 5, 10, 15])
    ax_force20.set_yticklabels([0, 5, 10, 15], font=fonts.serif(), fontsize=24)
    savefigure.save_as_png(fig_force20, "article_force20_vs_maturation_1+2")
    plt.close(fig_force20)
    fig_force80 = createfigure.rectangle_figure(pixels=180)
    ax_force80 = fig_force80.gca()
    ax_force80.errorbar(list(maturation_FF_dict.values()), list(mean_force80_FF_dict.values()), yerr=list(std_force80_FF_dict.values()), lw=0, label='FF', **kwargs_FF)
    ax_force80.errorbar(list(maturation_RDG_dict.values()), list(mean_force80_RDG_dict.values()), yerr=list(std_force80_RDG_dict.values()), lw=0, label='RDG', **kwargs_RDG)
    ax_force80.legend(prop=fonts.serif(), loc='lower left', framealpha=0.7)
    ax_force80.set_xlabel('Durée de stockage [jours]', font=fonts.serif(), fontsize=26)
    ax_force80.set_ylabel(r'$F_{80 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_force80.set_xticks([10, 13, 17])
    ax_force80.set_xticklabels([10, 13, 17], font=fonts.serif(), fontsize=24)
    ax_force80.set_yticks([0, 25, 50, 75, 100])
    ax_force80.set_yticklabels([0, 25, 50, 75, 100], font=fonts.serif(), fontsize=24)
    savefigure.save_as_png(fig_force80, "article_force80_vs_maturation_1+2")
    plt.close(fig_force80)

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    # plot_force_vs_disp(createfigure, savefigure, fonts)
    plot_forces_with_maturation()