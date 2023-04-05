import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.laser.post_processing.read_file as rf
import indentation.experiments.laser.post_processing.display_profiles as dp
from indentation.experiments.laser.post_processing.read_file import Files
import seaborn as sns
    
class Recovery:
    """
    A class define the .csv files to import.

    Attributes:
        ----------
        location: list
            list of 2 elements containing the lower and upper limits of the x values in which 
            the indentor is located

    Methods:
        -------
        import_files(self):
            Returns a list of the imported the .csv files

    """

    def __init__(self, location):
        """
        Constructs all the necessary attributes for the files object.

        Parameters:
            ----------
            location: list
                list of 2 elements containing the lower and upper limits of the x values in which 
                the indentor is located

        Returns:
            -------
            None
        """

        self.location = location

    def find_location_indices(self, vec_pos_axis):
        [x_min, x_max] = self.location
        x_is_x_min = np.abs(vec_pos_axis - x_min) 
        x_min_index = np.argmin(x_is_x_min)
        x_is_x_max = np.abs(vec_pos_axis - x_max) 
        x_max_index = np.argmin(x_is_x_max)
        return [x_min_index, x_max_index]
            
    def region_identification(self, mat_Z, vec_time, vec_pos_axis):
        """
        Returns the data points focused within the region of interest, 
        defined by self.location

        Parameters:
            ----------
            matZ: array
                values of Z in terms of X and time
            vec_time: array
                values of the time discretization
            vec_pos_axis: array
                values of the X position axis

        Returns:
            -------
            datafile_list: list
                list of .csv files starting by "beginning"

        """
        [x_min_index, x_max_index] = self.find_location_indices(vec_pos_axis)
        len_region = int(x_max_index - x_min_index) + 1
        mat_Z_indent = np.zeros((len(vec_time), len_region))
        vec_pos_axis_indent = np.zeros((len_region))
        for i in range(len_region):
            vec_pos_axis_indent[i] = vec_pos_axis[x_min_index + i]
            mat_Z_at_position = mat_Z[:, x_min_index + i]
            mat_Z_indent[:, i] = mat_Z_at_position
        return vec_pos_axis_indent, mat_Z_indent

    def plot_profile_region(self, filename, time, createfigure, savefigure, fonts):
        mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(mat_Z, vec_time, vec_pos_axis)
        fig, ax, experiment_time_in_millisecond = dp.create_fig_profile_at_time(mat_Z_indent, vec_time, vec_pos_axis_indent, time, createfigure, savefigure, fonts)
        savefigure.save_as_png(fig, "zx_profile_indent_" + filename[0:-4] + '_t' + str(int(experiment_time_in_millisecond)) + 'ms')
        savefigure.save_as_svg(fig, "zx_profile_indent_" + filename[0:-4] + '_t' + str(int(experiment_time_in_millisecond)) + 'ms')


    def combine_profile_timelapse(self, filename, nb_of_time_increments_to_plot, createfigure, savefigure, fonts):
        mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(mat_Z, vec_time, vec_pos_axis)
        nb_of_time_steps = len(vec_time)
        if abs(nb_of_time_increments_to_plot - nb_of_time_steps) < 1.1:
            nb_of_time_increments_to_plot = nb_of_time_steps 
            print('ok')
        time_steps_to_plot = np.linspace(0, nb_of_time_steps - 1, nb_of_time_increments_to_plot + 1, dtype=int)
        blues = sns.color_palette("Blues", nb_of_time_increments_to_plot + 1 )
        fig, ax, _ = dp.create_fig_profile_at_time(mat_Z_indent, vec_time, vec_pos_axis_indent, 0, createfigure, savefigure, fonts)    
        kwargs = {"linewidth": 2}
        for i in range(1, len(time_steps_to_plot)):
            t = time_steps_to_plot[i]
            experiment_time_in_second = vec_time[t] / 1e6
            ax.plot(vec_pos_axis_indent, mat_Z_indent[t, :], '-', color = blues[i], label = 't = ' + str(np.round(experiment_time_in_second, 1)) + ' s',  **kwargs)
        ax.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
        savefigure.save_as_png(fig, "zx_profile_indent_timelapse_" + filename[0:-4])
        savefigure.save_as_svg(fig, "zx_profile_indent_timelapse_" + filename[0:-4])



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    experiment_dates = ['230331', '230327']

    recovery = Recovery([0, 30])
    
    for experiment_date in experiment_dates:
        path_to_data = utils.reach_data_path(experiment_date)
        print(path_to_data)
        files = Files('FF')
        list_of_FF_files = files.import_files(experiment_date)
        nb_of_time_increments_to_plot = 10
        for filename in list_of_FF_files :
            rf.read_datafile(filename)
            mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
            recovery.combine_profile_timelapse(filename, nb_of_time_increments_to_plot, createfigure, savefigure, fonts)
        files = Files('RDG')
        list_of_FF_files = files.import_files(experiment_date)
        nb_of_time_increments_to_plot = 10
        for filename in list_of_FF_files :
            rf.read_datafile(filename)
            mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
            recovery.combine_profile_timelapse(filename, nb_of_time_increments_to_plot, createfigure, savefigure, fonts)

    # vec_pos_axis_indent, mat_Z_indent = recovery.region_identification(mat_Z, vec_time, vec_pos_axis)
    # recovery.plot_profile_region(filename_0, 0, createfigure, savefigure, fonts)