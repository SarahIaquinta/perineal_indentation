import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
from indentation.experiments.zwick.post_processing.compute_IRR_indicators import get_data_at_given_strain_rate
from sklearn.linear_model import LinearRegression
import seaborn as sns 
from indentation.experiments.zwick.post_processing.utils import find_nearest
# from matplotlib import AngleAnnotation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def compute_alpha(files_zwick, degree):
    datafile = '230403_C_Indentation_relaxation_500N_force.xlsx'
    sheet = '230403-FF2A'
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    max_force = np.max(force)
    index_where_force_is_max = np.where(force == max_force)[0][0]
    force_relaxation = force[index_where_force_is_max:]
    time_relaxation = time[index_where_force_is_max:]
    log_time_unshaped = np.array([-t**degree for t in time_relaxation])
    log_time = log_time_unshaped.reshape((-1, 1))
    # degree=9
    # polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    # polyreg.fit(log_time,force_relaxation)
    model = LinearRegression()
    model.fit(log_time, force_relaxation)
    fitted_response = model.predict(log_time)
    alpha = 0#model.coef_
    score = np.sqrt(r2_score(force_relaxation, fitted_response))
    # plt.figure()
    # plt.plot(log_time_unshaped, force_relaxation, 'b')
    # plt.plot(time_relaxation, force_relaxation, 'k')
    # plt.plot(log_time_unshaped, fitted_response, '--g')
    # plt.plot(1/log_time_unshaped, fitted_response, 'r')
    # plt.show()
    return alpha, fitted_response, log_time, force_relaxation, score




def plot_indicators_indentation_relaxation(files_zwick, createfigure, savefigure, fonts):
    datafile = '230403_C_Indentation_relaxation_500N_force.xlsx'
    sheet = '230403-FF2A'
    time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet)
    fig_force_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_force_vs_time = fig_force_vs_time.gca()
    fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
    ax_disp_vs_time = fig_disp_vs_time.gca()

    date = datafile[0:6]
    colors = sns.color_palette("Paired")
    kwargs = {"color":'k', "linewidth": 3}
    palette = sns.color_palette("YlOrBr", 10)
    time0 = time[0]
    time = time - time0
    ax_force_vs_time.plot(time, force, linestyle=':',  **kwargs)    
    ax_disp_vs_time.plot(time, disp, linestyle=':',  **kwargs)    

    for degree in range(1, 5):
        degree = int(degree)
        alpha, fitted_response, log_time_relaxation, force_relaxation, score = compute_alpha(files_zwick, degree)
        ax_force_vs_time.plot((-log_time_relaxation)**(1/degree)- time0, fitted_response ,   '-', color = palette[3+degree], label = 'd = ' + str(degree) + " R2 = " + str(np.round(score,3)))
    ax_force_vs_time.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    # ax_force_vs_time.set_title('fit en t')
    # indicators force
    # max_force = np.max(force)
    # index_where_force_is_max = np.where(force == max_force)[0]
    # relaxation_slope = (force[index_where_force_is_max + 3] - force[index_where_force_is_max + 1]) / (time[index_where_force_is_max + 3] - time[index_where_force_is_max + 1])
    # time_when_force_is_max = time[index_where_force_is_max]
    # relaxation_duration = 20
    # end_of_relaxation = time_when_force_is_max + relaxation_duration
    # index_where_time_is_end_relaxation = np.where(time == find_nearest(time, end_of_relaxation[0]))[0]
    # delta_f = max_force - force[index_where_time_is_end_relaxation]
    # delta_f_star = delta_f[0] / max_force
    # ax_force_vs_time.plot([time[index_where_force_is_max ], time[index_where_force_is_max + 2]], [force[index_where_force_is_max ], force[index_where_force_is_max + 2]], '-', color = 'r', label = r"$\alpha_R$ = " + str(np.round(relaxation_slope[0], 2)) + r" $N.s^{-1}$", linewidth=3)
    # ax_force_vs_time.plot([time[index_where_time_is_end_relaxation[0]], time[index_where_time_is_end_relaxation[0]]], [force[index_where_time_is_end_relaxation][0], max_force], '-', color = 'g', label = r"$\Delta F$  = " + str(np.round(delta_f[0], 2)) + " N \n" + r"$\Delta F^*$ = " + str(np.round(delta_f_star, 2)), linewidth=3)
    # time_at_strain_rate_0, force_at_strain_rate_0, _ = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.01)
    # time_at_strain_rate, force_at_strain_rate, _ = get_data_at_given_strain_rate(files_zwick, datafile, sheet, 0.25)
    # i_time_strain_rate = (force_at_strain_rate - force_at_strain_rate_0) / (time_at_strain_rate - time_at_strain_rate_0)
    # ax_force_vs_time.plot([time_at_strain_rate_0 - time0, time_at_strain_rate - time0], [force_at_strain_rate_0, force_at_strain_rate], '-', color = 'b', label = r"$i_{25\%}$ = " + str(np.round(i_time_strain_rate[0], 2)) + r' $Ns^{-1}$', linewidth=3)
    

    ax_force_vs_time.set_xticks([0, 5, 10, 15])
    ax_force_vs_time.set_xticklabels([0, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_force_vs_time.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax_force_vs_time.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], font=fonts.serif(), fontsize=24)
    ax_force_vs_time.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time.set_ylabel(r"force [N]", font=fonts.serif(), fontsize=26)

    ax_disp_vs_time.set_xticks([0, 5, 10, 15])
    ax_disp_vs_time.set_xticklabels([0, 5, 10, 15], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    ax_disp_vs_time.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2], font=fonts.serif(), fontsize=24)
    ax_disp_vs_time.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_force_vs_time, "article_force_vs_time_with_indicators")
    plt.close(fig_force_vs_time)
    savefigure.save_as_png(fig_disp_vs_time, "article_disp_vs_time_with_indicators")
    plt.close(fig_disp_vs_time)

    




if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230403']
    types_of_essay = ['C_Indentation_relaxation_500N_force.xlsx']#,'C_Indentation_relaxation_maintienFnulle_500N_trav.xls',  'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    plot_indicators_indentation_relaxation(files_zwick, createfigure, savefigure, fonts)
