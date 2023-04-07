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

    def make_polyfit_profile(self, filename, time_index, degree):
        mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(mat_Z, vec_time, vec_pos_axis)
        vec_Z_at_time_indent = mat_Z_indent[time_index, :]
        fitted_z_vec = utils.make_polyfit(vec_pos_axis_indent, vec_Z_at_time_indent, degree)
        return fitted_z_vec

    def find_x_index_where_z_indent_is_max(self, filename):
        mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
        _, mat_Z_indent = self.region_identification(mat_Z, vec_time, vec_pos_axis)
        vec_Z_at_time_indent = mat_Z_indent[0, :]
        index_where_z_is_max = np.nanargmax(vec_Z_at_time_indent)
        return index_where_z_is_max
        
                
    def combine_profile_timelapse(self, filename, nb_of_time_increments_to_plot, createfigure, savefigure, fonts):
        mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(mat_Z, vec_time, vec_pos_axis)
        nb_of_time_steps = len(vec_time)
        if abs(nb_of_time_increments_to_plot - nb_of_time_steps) < 1.1:
            nb_of_time_increments_to_plot = nb_of_time_steps 
            print('ok')
        time_steps_to_plot = np.linspace(0, nb_of_time_steps - 1, nb_of_time_increments_to_plot + 1, dtype=int)
        blues = sns.color_palette("Blues", nb_of_time_increments_to_plot + 1 )
        reds = sns.color_palette("Reds", nb_of_time_increments_to_plot + 1 )
        fig, ax, _ = dp.create_fig_profile_at_time(mat_Z_indent, vec_time, vec_pos_axis_indent, 0, createfigure, savefigure, fonts)    
        kwargs = {"linewidth": 2}
        index_where_z_is_max = self.find_x_index_where_z_indent_is_max(filename)
        for i in range(1, len(time_steps_to_plot)):
            t = time_steps_to_plot[i]
            experiment_time_in_second = vec_time[t] / 1e6
            vec_Z_at_time_indent = mat_Z_indent[t, :]
            ax.plot(vec_pos_axis_indent, vec_Z_at_time_indent, '-', color = blues[i], label = 't = ' + str(np.round(experiment_time_in_second, 1)) + ' s',  **kwargs)
            ax.plot([vec_pos_axis_indent[index_where_z_is_max]], [vec_Z_at_time_indent[index_where_z_is_max]], marker="o", markersize=8, markeredgecolor="k", markerfacecolor = reds[i], alpha=0.8)
        ax.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
        savefigure.save_as_png(fig, "zx_profile_indent_timelapse_" + filename[0:-4])
        savefigure.save_as_svg(fig, "zx_profile_indent_timelapse_" + filename[0:-4])

    def compute_recovery_with_time(self, filename):
        mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(mat_Z, vec_time, vec_pos_axis)
        index_where_z_is_max = self.find_x_index_where_z_indent_is_max(filename)
        recovery_positions = np.zeros_like(vec_time)
        for t in range(1, len(vec_time)):
            vec_Z_at_time_indent = mat_Z_indent[t, :]
            recovery_positions[t] = vec_Z_at_time_indent[index_where_z_is_max]
        return recovery_positions

    def compute_delta_d_star(self, filename):
        recovery_positions = self.compute_recovery_with_time(filename)
        delta_d = recovery_positions[-1] - np.min(recovery_positions)
        delta_d_star = delta_d / np.min(recovery_positions)
        return delta_d_star
    
    def plot_recovery(self, filename, createfigure, savefigure, fonts):
        recovery_positions = self.compute_recovery_with_time(filename)
        _, vec_time, _ = utils.extract_data_from_pkl(filename)
        fig = createfigure.rectangle_rz_figure(pixels=180)
        ax = fig.gca()
        kwargs = {"linewidth": 2}
        delta_d_star = self.compute_delta_d_star(filename)
        ax.plot(vec_time[1:], recovery_positions[:-1], '-k', label = filename[0:-4], **kwargs)
        # ax.text(str(delta_d_star))
        # ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$time$ [s]", font=fonts.serif(), fontsize=24)
        ax.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=22)
        ax.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
        savefigure.save_as_png(fig, "recovery_" + filename[0:-4])
        savefigure.save_as_svg(fig, "recovery_" + filename[0:-4])



def plot_all_combined_profiles(experiment_dates, meat_pieces, locations, nb_of_time_increments_to_plot, createfigure, savefigure, fonts):
    location_keys = [key for key in locations]
    for experiment_date in experiment_dates:
        for meat_piece in meat_pieces:
            files_meat_piece = Files(meat_piece)
            location_key = experiment_date + '_' + meat_piece
            if location_key in location_keys :
                list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
                location_at_date_meat_piece = locations[experiment_date + '_' + meat_piece]
                recovery_at_date_meat_piece = Recovery(location_at_date_meat_piece)
                for filename in list_of_meat_piece_files:
                    recovery_at_date_meat_piece.combine_profile_timelapse(filename, nb_of_time_increments_to_plot, createfigure, savefigure, fonts)
           
def plot_all_recoveries(experiment_dates, meat_pieces, locations, failed_laser_acqusitions, createfigure, savefigure, fonts):
    location_keys = [key for key in locations]
    for experiment_date in experiment_dates:
        for meat_piece in meat_pieces:
            files_meat_piece = Files(meat_piece)
            location_key = experiment_date + '_' + meat_piece
            if location_key in location_keys :
                list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
                location_at_date_meat_piece = locations[experiment_date + '_' + meat_piece]
                recovery_at_date_meat_piece = Recovery(location_at_date_meat_piece)
                for filename in list_of_meat_piece_files:
                    if filename[:-4] not in failed_laser_acqusitions:
                        recovery_at_date_meat_piece.plot_recovery(filename, createfigure, savefigure, fonts)
               
    
def export_delta_d_star(experiment_dates, meat_pieces, locations, failed_laser_acqusitions):
    location_keys = [key for key in locations]
    delta_d_stars_to_export = []
    filenames_to_export = [] 
    for experiment_date in experiment_dates:
        for meat_piece in meat_pieces:
            files_meat_piece = Files(meat_piece)
            location_key = experiment_date + '_' + meat_piece
            if location_key in location_keys :
                list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
                location_at_date_meat_piece = locations[experiment_date + '_' + meat_piece]
                recovery_at_date_meat_piece = Recovery(location_at_date_meat_piece)
                for filename in list_of_meat_piece_files:
                    if filename[:-4] not in failed_laser_acqusitions:
                        delta_d_star = recovery_at_date_meat_piece.compute_delta_d_star(filename)
                        delta_d_stars_to_export.append(delta_d_star)
                        filenames_to_export.append(filename[:-4])
    filename_export_all_delta_d_star = "recoveries.txt"
    path_to_processed_data = utils.get_path_to_processed_data()
    path_to_export_file = path_to_processed_data / filename_export_all_delta_d_star
    f = open(path_to_export_file, "w")
    f.write("test Id \t delta d star \n")
    mat_to_export = np.zeros((len(filenames_to_export), 2))
    for i in range(len(filenames_to_export)):
        f.write(
            filenames_to_export[i]
            + "\t"
            + str(delta_d_stars_to_export[i])
            + "\n"
        )
        # mat_to_export[i, 0] = filenames_to_export[i]
        # mat_to_export[i, 1] = delta_d_stars_to_export
    for k in range(len(failed_laser_acqusitions)):
            f.write(
            failed_laser_acqusitions[k]
            + "\t"
            + 'FAILED LASER ACQUISITION'
            + "\n"
        )
    f.close()
    return delta_d_stars_to_export, filenames_to_export



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    
    current_path = utils.get_current_path()
    nb_of_time_increments_to_plot = 10
    locations = {"230331_FF": [0, 20],
                 "230331_RDG": [0, 20],
                 "230331_RDG1_ENTIER": [-18, 0],
                 "230331_FF1_recouvrance_et_relaxation_max" : [-18, 0],
                 "230331_FF1_ENTIER" : [-18, 0],
                 "230327_FF": [-10, 5],
                 "230327_RDG": [-10, 10],
                 "230403_FF": [8, 40],
                 "230403_RDG": [10, 40]}
    fit_degree = 5
    
    experiment_dates = ['230331', '230327', '230404']
    # experiment_dates = ['230403']
    meat_pieces = ['RDG', 'FF', 'RDG1_ENTIER', "FF1_recouvrance_et_relaxation_max", "FF1_ENTIER"]
    # meat_pieces = ["RDG"]
    # plot_all_combined_profiles(experiment_dates, meat_pieces, locations, nb_of_time_increments_to_plot, createfigure, savefigure, fonts)
    failed_laser_acqusitions = ['230403_FF1B',
                                '230403_FF1D',
                                '230403_RDG1B',
                                '230403_RDG1E',
                                '230403_RDG1F',
                                '230403_RDG2A'
                                ]
    
    # plot_all_recoveries(experiment_dates, meat_pieces, locations, failed_laser_acqusitions, createfigure, savefigure, fonts)
    delta_d_stars_to_export, filenames_to_export = export_delta_d_star(experiment_dates, meat_pieces, locations, failed_laser_acqusitions)