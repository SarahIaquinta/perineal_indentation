import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
from indentation.experiments.laser.post_processing.read_file import Files
from indentation.experiments.laser.post_processing.identify_movement import Recovery
from sklearn.linear_model import LinearRegression
import seaborn as sns 

def plot_recovery(locations,  createfigure, savefigure, fonts):
    experiment_date = '0_locations_230403'
    meat_piece = '_FF2A.csv'
    # files_meat_piece = Files(meat_piece)
    # list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
    filename =  experiment_date + meat_piece
    recovery_at_date_meat_piece = Recovery(filename, locations)
    # recovery_at_date_meat_piece.plot_recovery(10, createfigure, savefigure, fonts)
    n_smooth = 10
    vec_time_wo_outliers, recovery_positions_wo_outliers_unsmoothed = recovery_at_date_meat_piece.compute_recovery_with_time(n_smooth)
    # recovery_positions = lfilter([1/1]*1, 1, recovery_positions_wo_outliers_unsmoothed)
    nn = 200
    # recovery_positions = recovery_positions_wo_outliers_unsmoothed
    recovery_positions = np.convolve(recovery_positions_wo_outliers_unsmoothed,np.ones(nn)/nn,'same')
    recovery_positions[0:int(nn/2)+1] = recovery_positions[int(nn/2)+2]
    recovery_positions[-int(nn/2)+1:] = recovery_positions[-int(nn/2)+2]
    
    
    
    # A, fitted_response, log_time = recovery_at_date_meat_piece.compute_A(n_smooth)
    fig = createfigure.rectangle_figure(pixels=180)
    fig_log = createfigure.rectangle_figure(pixels=180)
    ax = fig.gca()
    ax_log = fig_log.gca()
    kwargs = {"linewidth": 3}
    index_recovery_position_is_min, min_recovery_position, last_recovery, delta_d, delta_d_star = recovery_at_date_meat_piece.compute_delta_d_star(n_smooth)
    recovery_time_at_beginning = vec_time_wo_outliers[index_recovery_position_is_min] /1e6
    # ax.plot(vec_time_wo_outliers[1:]/1e6, recovery_positions[:-1], '-k', label = recovery_at_date_meat_piece.filename[0:-4], **kwargs)
    vec_time_wo_outliers = vec_time_wo_outliers - vec_time_wo_outliers[0] + 0.1
    
    recovery_position_at_beginning = recovery_positions[index_recovery_position_is_min]
    recovery_time_at_end = vec_time_wo_outliers[-2]/1e6
    recovery_position_at_end = last_recovery
    
    log_time_unshaped = np.array([np.log((t+0.01)/1e6) for t in vec_time_wo_outliers])
    log_time = log_time_unshaped.reshape((-1, 1))
    model = LinearRegression()
    model.fit(log_time[1:], recovery_positions[1:])
    fitted_response = model.predict(log_time)
    A = model.coef_
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket", as_cmap=False)    
    ax.plot(vec_time_wo_outliers[index_recovery_position_is_min+60:-502]/1e6, recovery_positions[index_recovery_position_is_min+60:-502], '-k', **kwargs)
    ax.plot(np.exp(log_time)[+185:-502], fitted_response[+185:-502] ,   '--', color = color_rocket[4], label =r"$z = A \log{t} + z_{t=1}$" + "\nA = " + str(np.round(A[0], 2)) , **kwargs)
    # ax.plot([recovery_time_at_beginning], [recovery_position_at_beginning], label = 'beginning', marker="*", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax.plot([recovery_time_at_end], [recovery_position_at_end], label = 'end', marker="o", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax.text(str(delta_d_star))
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif_rz_legend())
    # ax_log.plot(vec_time_wo_outliers[index_recovery_position_is_min+60:-502]/1e6, recovery_positions[index_recovery_position_is_min+60:-502], '-k', **kwargs)
    # ax_log.plot(np.exp(log_time)[+60:-502], fitted_response[+60:-502], ':r', label='A = ' + str(np.round(A[0], 2)), **kwargs)
    # ax_log.plot([recovery_time_at_beginning], [recovery_position_at_beginning], label = 'beginning', marker="*", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax_log.plot([recovery_time_at_end], [recovery_position_at_end], label = 'end', marker="o", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
    # ax.text(str(delta_d_star))
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif_rz_legend())
    # ax_log.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif_rz_legend())
    # ax_log.set_xscale('log')
    # ax_log.set_xlim(1, 31)
    # ax.set_ylim(-5.5, -3.5)
    # ax.set_xticks([1, 10, 100])
    # ax.set_xticklabels([1, 10, 100],  font=fonts.serif(), fontsize=24)
    # ax.set_yticks([-6, -5, -4])
    # ax.set_yticklabels([-6, -5, -4],  font=fonts.serif(), fontsize=24)
    
    # ax.arrow(30.5, -6.3, 0, 1.1, head_width=0.5, head_length=0.05, color='b', length_includes_head=True)
    ax.annotate(r"$\Delta$d", (28, -5.8), font=fonts.serif(), fontsize=26, color = color_rocket[2]  )

    ax.annotate("", xy=(30.5, -5.2), xytext=(30.5, -6.3),
            arrowprops=dict(arrowstyle="<->", linewidth=2, color = color_rocket[2] ))
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels([0, 10, 20, 30], font=fonts.serif(), fontsize=24)
    ax.set_yticks([-6.2, -6, -5.8, -5.6, -5.4, -5.2])
    ax.set_yticklabels([-6.2, -6, -5.8, -5.6, -5.4, -5.2], font=fonts.serif(), fontsize=24)
    ax.set_xlabel(r"temps [s]", font=fonts.serif(), fontsize=26)
    # ax_log.set_xlabel(r"$log(time) $ [-]", font=fonts.serif(), fontsize=26)
    ax.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
    ax_log.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
    ax.legend(prop=fonts.serif_1(), loc='upper left', framealpha=0.7,frameon=False)
    # ax_log.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
    savefigure.save_as_png(fig, "article_recovery_"  + experiment_date + meat_piece[0:5])
    # savefigure.save_as_png(fig_log, "article_recovery_logx_" + experiment_date + meat_piece[0:5])
    plt.close(fig)
    plt.close(fig_log)
        
        
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    
    locations = {"230331_FF": [0, 20],
                 "230331_FF2_2C": [0, 20],
                 "230331_RDG": [2, 20],
                 "230331_RDG1_ENTIER": [-18, 0],
                 "230331_FF1_recouvrance_et_relaxation_max" : [-18, 0],
                 "230331_FF1_ENTIER" : [-18, 0],
                 "230327_FF": [-10, 5],
                 "230327_RDG": [-10, 10],
                 "230403_FF": [8, 40],
                 "230403_RDG": [10, 40],
                 "230407_FF": [-30, -10],
                 "230407_RDG": [-30, -10],
                 "230411_FF": [-10, 10],
                 "230411_FF2_ENTIER": [-10, 10],
                 "230411_RDG": [-10, 10],
                 "230411_RDG2D": [-10, 10],
                 "230515_P002": [-12, 0],
                 "230515_P011": [-15, -2]}
    plot_recovery(locations,  createfigure, savefigure, fonts)