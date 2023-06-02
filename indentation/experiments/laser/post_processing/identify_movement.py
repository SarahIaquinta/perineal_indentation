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
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter
import pickle


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

    def __init__(self, filename, location):
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
        self.filename = filename
        self.mat_Z, self.vec_time, self.vec_pos_axis = utils.extract_data_from_pkl(self.filename)

    def find_location_indices(self):
        [x_min, x_max] = self.location
        x_is_x_min = np.abs(self.vec_pos_axis - x_min) 
        x_min_index = np.argmin(x_is_x_min)
        x_is_x_max = np.abs(self.vec_pos_axis - x_max) 
        x_max_index = np.argmin(x_is_x_max)
        return [x_min_index, x_max_index]
            
    def region_identification(self, n_smooth):
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
        [x_min_index, x_max_index] = self.find_location_indices()
        len_region = int(x_max_index - x_min_index) + 1
        smoothed_Z = self.mat_Z
        smoothed_Z = self.smooth_Z_profile(n_smooth)
        mat_Z_indent = np.zeros((len(self.vec_time), len_region))
        vec_pos_axis_indent = np.zeros((len_region))
        for i in range(len_region):
            vec_pos_axis_indent[i] = self.vec_pos_axis[x_min_index + i]
            mat_Z_at_position = smoothed_Z[:, x_min_index + i]
            mat_Z_indent[:, i] = mat_Z_at_position
        return vec_pos_axis_indent, mat_Z_indent

    def smooth_Z_profile(self, n_smooth):
        """
        Applies a convolution to the z-profile of the top surface 

        Parameters:
            ----------
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            smoothed_Z: array
                array containing the values of z after convolution

        """
        smoothed_Z = np.zeros_like(self.mat_Z)
        for i in range(len(self.vec_time)):
            vec_Z_at_time = self.mat_Z[i, :]
            smoothed_Z_at_time = np.convolve(vec_Z_at_time,1/n_smooth*np.ones(n_smooth),'same')
            smoothed_Z[i, :] = smoothed_Z_at_time
        return smoothed_Z

    def plot_profile_region(self, time, n_smooth, createfigure, savefigure, fonts):
        """
        Plots the profile restricted in a given window of x-values, at a given time

        Parameters:
            ----------
            time: float
                index number of time at which the z profile of the top surface is to be plotted
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            None

        """
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(n_smooth)
        fig, ax, experiment_time_in_millisecond = dp.create_fig_profile_at_time(mat_Z_indent, self.vec_time, vec_pos_axis_indent, time, createfigure, savefigure, fonts)
        savefigure.save_as_png(fig, "zx_smoothed_n" + str(n_smooth) + "profile_indent_" + self.filename[0:-4] + '_t' + str(int(experiment_time_in_millisecond)) + 'ms')
        savefigure.save_as_svg(fig, "zx_smoothed_n" + str(n_smooth) + "profile_indent_" + self.filename[0:-4] + '_t' + str(int(experiment_time_in_millisecond)) + 'ms')

    def find_x_index_where_z_indent_is_max(self, n_smooth):
        """
        Plots the profile restricted in a given window of x-values, at a given time

        Parameters:
            ----------
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            index_where_z_is_max: int
                x-index where z at time zero is max. This criterion is used to roughly identify 
                the deepest point of the top surface.

        """
        _, mat_Z_indent = self.region_identification(n_smooth)
        vec_Z_at_time_indent = mat_Z_indent[0, :]
        index_where_z_is_max = np.nanargmax(vec_Z_at_time_indent)
        return index_where_z_is_max
                        
    def combine_profile_timelapse(self, nb_of_time_increments_to_plot, n_smooth, createfigure, savefigure, fonts):
        """
        Combines all the z-profiles during a given timelapse using nb_of_time_increments_to_plot

        Parameters:
            ----------
            nb_of_time_increments_to_plot: int
                number of time increments to be plotted in the figure
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            None

        """
        vec_pos_axis_indent, mat_Z_indent = self.region_identification(n_smooth)
        nb_of_time_steps = len(self.vec_time)
        if abs(nb_of_time_increments_to_plot - nb_of_time_steps) < 1.1:
            nb_of_time_increments_to_plot = nb_of_time_steps 
            print('ok')
        time_steps_to_plot = np.linspace(0, nb_of_time_steps - 1, nb_of_time_increments_to_plot + 1, dtype=int)
        blues = sns.color_palette("Blues", nb_of_time_increments_to_plot + 1 )
        reds = sns.color_palette("Reds", nb_of_time_increments_to_plot + 1 )
        fig, ax, _ = dp.create_fig_profile_at_time(mat_Z_indent, self.vec_time, vec_pos_axis_indent, 0, createfigure, savefigure, fonts)    
        kwargs = {"linewidth": 3}
        index_where_z_is_max = self.find_x_index_where_z_indent_is_max(n_smooth)
        for i in range(1, len(time_steps_to_plot)):
            t = time_steps_to_plot[i]
            experiment_time_in_second = self.vec_time[t] / 1e6
            vec_Z_at_time_indent = mat_Z_indent[t, :]
            ax.plot(vec_pos_axis_indent, vec_Z_at_time_indent, '-', color = blues[i], label = 't = ' + str(np.round(experiment_time_in_second, 1)) + ' s',  **kwargs)
            ax.plot([vec_pos_axis_indent[index_where_z_is_max]], [vec_Z_at_time_indent[index_where_z_is_max]], marker="o", markersize=12, markeredgecolor="k", markerfacecolor = reds[i], alpha=0.8)
        # ax.set_xticks([-10, -5, 0, 5, 10])
        # ax.set_xticklabels(['-10', '-5', '0', '5', '10'], font=fonts.serif(), fontsize=24)
        # ax.set_yticks([-5, 0, 5])
        # ax.set_yticklabels([-5, 0, 5], font=fonts.serif(), fontsize=24)
        # ax.set_ylim(-6, 2)
        # ax.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
        savefigure.save_as_png(fig, "zx_smoothed_n" + str(n_smooth) + "_profile_indent_timelapse_" + self.filename[0:-4])
        savefigure.save_as_svg(fig, "zx_smoothed_n" + str(n_smooth) + "_profile_indent_timelapse_" + self.filename[0:-4])

    def compute_recovery_with_time(self, n_smooth):
        """
        Computes the evolution of the z-position of the deepest position with time

        Parameters:
            ----------
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            recovery_positions: array
                vector containing the evolution of the z position in terms of the elapsed time        

        """ 
        _, mat_Z_indent = self.region_identification(n_smooth)
        index_where_z_is_max = self.find_x_index_where_z_indent_is_max(n_smooth)
        recovery_positions = np.zeros_like(self.vec_time)
        for t in range(1, len(self.vec_time)):
            vec_Z_at_time_indent = mat_Z_indent[t, :]
            recovery_positions[t] = vec_Z_at_time_indent[index_where_z_is_max]
        return recovery_positions

    def compute_delta_d_star(self, n_smooth):
        """
        Computes the value of delta_d_star, which is defined as the variation in z of the deepest
        point of the top surface, normalized by the depth of the deepest point

        Parameters:
            ----------
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            index_recovery_position_is_min: int
                index at which the top surface is the deepest
            min_recovery_position: float
                depth of the deepest point of the top surface
            last_recovery: float
                position of the top surface at the end of acquisition
            delta_d: float
                difference between last_recovery and min_recovery_position
            delta_d_star: float
                delta_d normalized by min_recovery_position
                
        """ 
        recovery_positions = self.compute_recovery_with_time(n_smooth)
        last_recovery_index = (~np.isnan(np.array(recovery_positions))).cumsum().argmax()
        last_recovery = recovery_positions[last_recovery_index]
        index_recovery_position_is_min = np.nanargmin(recovery_positions)
        # if index_recovery_position_is_min == 0:
        #     recovery_positions = recovery_positions[1:]
        #     index_recovery_position_is_min = np.nanargmin(recovery_positions)
        min_recovery_position = recovery_positions[index_recovery_position_is_min]
        delta_d = last_recovery - min_recovery_position
        delta_d_star = np.abs(delta_d / min_recovery_position)
        return index_recovery_position_is_min, min_recovery_position, last_recovery, delta_d, delta_d_star
    
    def compute_A(self, n_smooth):
        """
        Computes the value of A, which corresponds to the slope of the evolution of the z position of 
        the deepest point of the top surface in terms of log(time) during recovery
        A is obtained after performing a linear regression to fit the curve
        
        Parameters:
            ----------
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            A: float
                value fo the slope
            fitted_response: array
                fittes values from the linear regression
            log_time: array
                array containign the logarithmic values of the time vector
                
        """ 
        index_recovery_position_is_min, min_recovery_position, last_recovery, delta_d, delta_d_star = self.compute_delta_d_star(n_smooth)
        
        recovery_positions_with_NaN_and_inf = self.compute_recovery_with_time(n_smooth)[index_recovery_position_is_min:-2]
        log_time_unshaped_wth_NaN_and_inf_recovery = np.array([np.log((t+0.01)/1e6) for t in self.vec_time[index_recovery_position_is_min:-2]]) #TODO check for NaN
        recovery_positions_with_inf = recovery_positions_with_NaN_and_inf[~np.isnan(recovery_positions_with_NaN_and_inf)]
        log_time_unshaped_wth_inf_recovery = log_time_unshaped_wth_NaN_and_inf_recovery[~np.isnan(recovery_positions_with_NaN_and_inf)]
        recovery_positions_unsmoothed = recovery_positions_with_inf[np.isfinite(recovery_positions_with_inf)]
        recovery_positions = lfilter([1/1]*1, 1, recovery_positions_unsmoothed)
        log_time_unshaped = log_time_unshaped_wth_inf_recovery[np.isfinite(recovery_positions_with_inf)]
        
        # recovery_time_at_beginning = self.vec_time[index_recovery_position_is_min] /1e6
        # recovery_position_at_beginning = recovery_positions[index_recovery_position_is_min]
        # recovery_time_at_end = self.vec_time[-2]/1e6
        # recovery_position_at_end = last_recovery
        log_time = log_time_unshaped.reshape((-1, 1))
        model = LinearRegression()
        model.fit(log_time[1:], recovery_positions[1:])
        fitted_response = model.predict(log_time)
        A = model.coef_
        return A, fitted_response, log_time
        
            
    def plot_recovery(self, n_smooth, createfigure, savefigure, fonts):
        """
        Plots the recovery of the top surface (z position of the deepest point)
        
        Parameters:
            ----------
            n_smooth: int
                coefficient for the convolution
        Returns:
            -------
            None
                
        """ 
        recovery_positions_unsmoothed = self.compute_recovery_with_time(n_smooth)
        recovery_positions = lfilter([1/1]*1, 1, recovery_positions_unsmoothed)
        A, fitted_response, log_time = self.compute_A(n_smooth)
        fig = createfigure.rectangle_rz_figure(pixels=180)
        fig_log = createfigure.rectangle_rz_figure(pixels=180)
        ax = fig.gca()
        ax_log = fig_log.gca()
        kwargs = {"linewidth": 3}
        index_recovery_position_is_min, min_recovery_position, last_recovery, delta_d, delta_d_star = self.compute_delta_d_star(n_smooth)
        recovery_time_at_beginning = self.vec_time[index_recovery_position_is_min] /1e6
        recovery_position_at_beginning = recovery_positions[index_recovery_position_is_min]
        recovery_time_at_end = self.vec_time[-2]/1e6
        recovery_position_at_end = last_recovery
        # ax.plot(self.vec_time[1:]/1e6, recovery_positions[:-1], '-k', label = self.filename[0:-4], **kwargs)
        ax.plot(self.vec_time[index_recovery_position_is_min:-2]/1e6, recovery_positions[index_recovery_position_is_min:-2], '-k', **kwargs)
        ax.plot([recovery_time_at_beginning], [recovery_position_at_beginning], label = 'beginning', marker="*", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
        ax.plot([recovery_time_at_end], [recovery_position_at_end], label = 'end', marker="o", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
        # ax.text(str(delta_d_star))
        # ax.set_aspect("equal", adjustable="box")
        ax.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif_rz_legend())
        ax_log.plot(self.vec_time[index_recovery_position_is_min:-2]/1e6, recovery_positions[index_recovery_position_is_min:-2], '-k', **kwargs)
        ax_log.plot(np.exp(log_time), fitted_response, ':r', label='A = ' + str(np.round(A[0], 2)), **kwargs)
        ax_log.plot([recovery_time_at_beginning], [recovery_position_at_beginning], label = 'beginning', marker="*", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
        ax_log.plot([recovery_time_at_end], [recovery_position_at_end], label = 'end', marker="o", markersize=12, markeredgecolor="k", markerfacecolor = 'r', linestyle = 'None', alpha=0.8)
        # ax.text(str(delta_d_star))
        # ax.set_aspect("equal", adjustable="box")
        ax.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif_rz_legend())
        ax_log.set_title(r'$\Delta d$ = ' + str(np.round(delta_d,2)) +  r'  $\Delta d^*$ = ' + str(np.round(delta_d_star, 2)), font=fonts.serif_rz_legend())
        ax_log.set_xscale('log')
        # ax.set_xlim(1, 40.5)
        # ax.set_ylim(-5.5, -3.5)
        # ax.set_xticks([1, 10, 100])
        # ax.set_xticklabels([1, 10, 100],  font=fonts.serif(), fontsize=24)
        # ax.set_yticks([-6, -5, -4])
        # ax.set_yticklabels([-6, -5, -4],  font=fonts.serif(), fontsize=24)
        
        ax.set_xlabel(r"$time $ [s]", font=fonts.serif(), fontsize=26)
        ax_log.set_xlabel(r"$log(time) $ [-]", font=fonts.serif(), fontsize=26)
        ax.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
        ax_log.set_ylabel(r"$z$ [mm]", font=fonts.serif(), fontsize=26)
        # ax.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
        ax_log.legend(prop=fonts.serif_rz_legend(), loc='lower right', framealpha=0.7)
        savefigure.save_as_png(fig, "recovery_smoothed_n" + str(n_smooth) +self.filename[0:-4])
        savefigure.save_as_svg(fig, "recovery_smoothed_n" + str(n_smooth) +self.filename[0:-4])
        savefigure.save_as_png(fig_log, "recovery_logx_smoothed_n" + str(n_smooth) +self.filename[0:-4])
        savefigure.save_as_svg(fig_log, "recovery_logx_smoothed_n" + str(n_smooth) +self.filename[0:-4])

def plot_all_combined_profiles(experiment_dates, meat_pieces, locations, n_smooth, nb_of_time_increments_to_plot, createfigure, savefigure, fonts):
    """
    Combines all the profiles obtained at given dates and for given pieces
    
    Parameters:
        ----------
        experiment_dates: list
            list of the dates at which the testing have been performed. 
            Possible values for the date:
                - '230337'
                - '230331'
                - '230403'
                - '230407'
                - '230411'
        meatpieces: list
            list of the meatpieces to plot
            Possible values for the meatpiece:
                - 'RDG'
                - 'FF'
                - It is also possible to be more specific by detailing the number of the meatpiece
                using 'RDG1', 'RDG2', 'FF1', 'FF2'
        locations: dict
            dictionnary containing the location of the x-axis windows to be used to focus on the
            recovery, for all testings subcases
        n_smooth: int
            coefficient for the convolution
        nb_of_time_increments_to_plot: int
            number of time increments to be plotted in the figure

    Returns:
        -------
        None
            
    """ 
    location_keys = [key for key in locations]
    for experiment_date in experiment_dates:
        for meat_piece in meat_pieces:
            files_meat_piece = Files(meat_piece)
            location_key = experiment_date + '_' + meat_piece
            if location_key in location_keys :
                list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
                location_at_date_meat_piece = locations[experiment_date + '_' + meat_piece]
                for filename in list_of_meat_piece_files:
                    recovery_at_date_meat_piece = Recovery(filename, location_at_date_meat_piece)
                    recovery_at_date_meat_piece.combine_profile_timelapse(nb_of_time_increments_to_plot, n_smooth, createfigure, savefigure, fonts)
           
def plot_all_recoveries(experiment_dates, meat_pieces, locations, n_smooth, failed_laser_acqusitions, createfigure, savefigure, fonts):
    """
    Combines all the profiles obtained at given dates and for given pieces
    
    Parameters:
        ----------
        experiment_dates: list
            list of the dates at which the testing have been performed. 
            Possible values for the date:
                - '230337'
                - '230331'
                - '230403'
                - '230407'
                - '230411'
        meatpieces: list
            list of the meatpieces to plot
            Possible values for the meatpiece:
                - 'RDG'
                - 'FF'
                - It is also possible to be more specific by detailing the number of the meatpiece
                using 'RDG1', 'RDG2', 'FF1', 'FF2'
        locations: dict
            dictionnary containing the location of the x-axis windows to be used to focus on the
            recovery, for all testings subcases
        n_smooth: int
            coefficient for the convolution
        failed_laser_acqusitions: list
            list of the ids of acquisitions that have been failed

    Returns:
        -------
        None
            
    """ 
    location_keys = [key for key in locations]
    for experiment_date in experiment_dates:
        for meat_piece in meat_pieces:
            files_meat_piece = Files(meat_piece)
            location_key = experiment_date + '_' + meat_piece
            if location_key in location_keys :
                list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
                location_at_date_meat_piece = locations[experiment_date + '_' + meat_piece]
                for filename in list_of_meat_piece_files:
                    if filename[:-4] not in failed_laser_acqusitions:
                        recovery_at_date_meat_piece = Recovery(filename, location_at_date_meat_piece)
                        recovery_at_date_meat_piece.plot_recovery(n_smooth, createfigure, savefigure, fonts)
            
def export_delta_d_star_and_A(experiment_dates, n_smooth, meat_pieces, locations, failed_laser_acqusitions):
    """
    Exports the indicators into a txt file
    
    Parameters:
        ----------
        experiment_dates: list
            list of the dates at which the testing have been performed. 
            Possible values for the date:
                - '230337'
                - '230331'
                - '230403'
                - '230407'
                - '230411'
        meatpieces: list
            list of the meatpieces to plot
            Possible values for the meatpiece:
                - 'RDG'
                - 'FF'
                - It is also possible to be more specific by detailing the number of the meatpiece
                using 'RDG1', 'RDG2', 'FF1', 'FF2'
        locations: dict
            dictionnary containing the location of the x-axis windows to be used to focus on the
            recovery, for all testings subcases
        n_smooth: int
            coefficient for the convolution
        failed_laser_acqusitions: list
            list of the ids of acquisitions that have been failed

    Returns:
        -------
        delta_d_stars_to_export: list
            list of the values of delta_d_stars to be exported
        filenames_to_export: list
            list of the ids of the filenames that have been exported
            
    """ 
    location_keys = [key for key in locations]
    delta_d_to_export = []
    delta_d_stars_to_export = []
    d_min_to_export = [] 
    filenames_to_export = [] 
    A_to_export = []
    for experiment_date in experiment_dates:
        for meat_piece in meat_pieces:
            files_meat_piece = Files(meat_piece)
            location_key = experiment_date + '_' + meat_piece
            if location_key in location_keys :
                list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
                location_at_date_meat_piece = locations[experiment_date + '_' + meat_piece]
                for filename in list_of_meat_piece_files:
                    if filename[:-4] not in failed_laser_acqusitions:
                        print(filename[:-4])
                        recovery_at_date_meat_piece = Recovery(filename, location_at_date_meat_piece)
                        _, min_recovery_position, _, delta_d, delta_d_star = recovery_at_date_meat_piece.compute_delta_d_star(n_smooth)
                        A, _, _ = recovery_at_date_meat_piece.compute_A(n_smooth)
                        delta_d_stars_to_export.append(delta_d_star)
                        delta_d_to_export.append(delta_d)
                        d_min_to_export.append(min_recovery_position)
                        filenames_to_export.append(filename[:-4])
                        A_to_export.append(A[0])
    filename_export_all_delta_d_star = "recoveries.txt"
    path_to_processed_data = utils.get_path_to_processed_data()
    path_to_export_file = path_to_processed_data / filename_export_all_delta_d_star
    f = open(path_to_export_file, "w")
    f.write("test Id \t delta d , \t delta d star \t d_min \t A \n")
    # mat_to_export = np.zeros((len(filenames_to_export), 2))
    for i in range(len(filenames_to_export)):
        f.write(
            filenames_to_export[i]
            + "\t"
            + str(delta_d_to_export[i])
            + "\t"
            + str(delta_d_stars_to_export[i])
            + "\t"
            + str(d_min_to_export[i])
            + "\t"
            + str(A_to_export[i])
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
    complete_pkl_filename = utils.get_path_to_processed_data() / 'recovery_meat.pkl'
    with open(complete_pkl_filename, "wb") as f:
        pickle.dump(
            [filenames_to_export, delta_d_to_export, delta_d_stars_to_export, d_min_to_export, A_to_export],
            f,
        )
    return delta_d_stars_to_export, filenames_to_export



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
    experiment_dates = ['230327', '230331', '230403', '230407', '230411']#, '230331', '230327']
    meat_pieces = ['FF', "FF2_ENTIER", 'RDG', 'RDG1_ENTIER', "FF1_recouvrance_et_relaxation_max", "FF1_ENTIER"]
    failed_laser_acqusitions = ['230403_FF1B',
                                '230403_FF1D',
                                '230403_RDG1B',
                                '230403_RDG1E',
                                '230403_RDG1F',
                                '230403_RDG2A',
                                '230331_RDG2_1F',
                                '230331_RDG1_1E',
                                '230411_FF2_ENTIER1',
                                '230515_P002-1',
                                '230331_FF2_1E',
                                '230331_RDG1_ENTIER1'
                                ]
    
    for n_smooth in [10]:
        delta_d_stars_to_export, filenames_to_export = export_delta_d_star_and_A(experiment_dates, n_smooth, meat_pieces, locations, failed_laser_acqusitions)
        # # plot_all_combined_profiles(experiment_dates,  meat_pieces, locations, n_smooth, nb_of_time_increments_to_plot, createfigure, savefigure, fonts)
        # plot_all_recoveries(experiment_dates, meat_pieces, locations, n_smooth, failed_laser_acqusitions, createfigure, savefigure, fonts)