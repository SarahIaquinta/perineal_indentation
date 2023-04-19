import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
from indentation.experiments.laser.post_processing.read_file import Files

def plot_Z_profile(filename, createfigure, savefigure, fonts):
    mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
    fig = createfigure.rectangle_vertical_rz_figure(pixels=180)
    ax = fig.gca()
    im = ax.imshow(mat_Z)
    cbar = fig.colorbar(im, ticks = [-15, -10, -5, 0, 5])
    cbar.ax.set_yticklabels([-15, -10, -5, 0, 5], font=fonts.serif())
    cbar.set_label(r'$z [mm]$', fontsize=22)
    x_ticks = np.linspace(0, len(vec_pos_axis), 4)
    x_ticklabels_values = np.linspace(vec_pos_axis[0], vec_pos_axis[-1], 4)
    x_ticklabels = [int(x) for x in x_ticklabels_values] 
    y_ticks = np.linspace(0, len(vec_time), 6)
    y_ticklabels_values = np.linspace(vec_time[0], vec_time[-1], 6)
    y_ticklabels = [np.round(y/1e6, 1) for y in y_ticklabels_values]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels,  font=fonts.serif(), fontsize=18)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels,  font=fonts.serif(), fontsize=18)
    ax.set_xlabel(r"$x$ [mm]", font=fonts.serif(), fontsize=24)
    ax.set_ylabel('time ' + r"[s]", font=fonts.serif(), fontsize=22)
    savefigure.save_as_png(fig, "matZ_" + filename[0:-4] )
    savefigure.save_as_svg(fig, "matZ_" + filename[0:-4] )
    
def create_fig_profile_at_time(mat_Z, vec_time, vec_pos_axis, time, createfigure, savefigure, fonts):
    time_index = int(time)
    experiment_time_in_microsecond = vec_time[time_index]
    experiment_time_in_second = experiment_time_in_microsecond / 1e6
    experiment_time_in_millisecond = experiment_time_in_microsecond / 1e3
    Z_at_time = mat_Z[time_index, :]
    Z_not_nan_indices = np.where([np.isnan(Z_at_time) == False])[1]
    Z_at_time_not_nan = [Z_at_time[i] for i in Z_not_nan_indices[:-1]]
    vec_pos_axis_not_nan = [vec_pos_axis[i] for i in Z_not_nan_indices[:-1]]
    fig = createfigure.rectangle_rz_figure(pixels=180)
    ax = fig.gca()
    kwargs = {"linewidth": 2}
    ax.plot(vec_pos_axis_not_nan, Z_at_time_not_nan, '-k', alpha=1, label = 't = ' + str(np.round(experiment_time_in_second, 4)) + ' s', **kwargs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ [mm]", font=fonts.serif(), fontsize=26)
    ax.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
    # ax.legend(prop=fonts.serif(), loc='lower right', framealpha=0.7)
    return fig, ax, experiment_time_in_millisecond

def plot_profile_at_time(filename, time, createfigure, savefigure, fonts):
    mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
    fig, ax, experiment_time_in_millisecond = create_fig_profile_at_time(mat_Z, vec_time, vec_pos_axis, time, createfigure, savefigure, fonts)
    savefigure.save_as_png(fig, "zx_profile_" + filename[0:-4] + '_t' + str(int(experiment_time_in_millisecond)) + 'ms')
    savefigure.save_as_svg(fig, "zx_profile_" + filename[0:-4] + '_t' + str(int(experiment_time_in_millisecond)) + 'ms')


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    experiment_date = '230403'
    path_to_data = utils.reach_data_path(experiment_date)
    print(path_to_data)
    files = Files('FF')
    list_of_FF_files = files.import_files(experiment_date)
    filename_0 =list_of_FF_files[1]
    # read_datafile(filename_0)
    # plot_Z_profile(filename_0, createfigure, savefigure, fonts)
    print('hello')
    plot_profile_at_time(filename_0, 1000, createfigure, savefigure, fonts)