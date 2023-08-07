"""
This file contains tools to extract data from the zwick raw datafiles, 
in order to compute the evolution of force and displacement of the 
indentor during the experiment.
The script removes the experiments that failed, based on information given in 
a .csv file, and then plots the force vs disp, force vs time and disp vs time curves.
"""

import numpy as np
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
import pandas as pd
import seaborn as sns

class Files_Zwick:
    """
    A class that defines the .xls files to import.

    Attributes:
        ----------
        type_of_essay: string
            name of the essay of which the csv file needs to be extracted
            Possible values : 
            C_Indentation_relaxation_500N_force
            C_Indentation_relaxation_maintienFnulle_500N_trav

    Methods:
        -------
        import_files(self):
            Returns a list of the imported the .xls files

    """

    def __init__(self, type_of_essay):
        """
        Constructs all the necessary attributes for the files object.

        Parameters:
            ----------
            type_of_essay: string
                name of the essay of which the csv file needs to be extracted
                Possible values : 
                C_Indentation_relaxation_500N_force
                C_Indentation_relaxation_maintienFnulle_500N_trav

        Returns:
            -------
            None
        """

        self.type_of_essay = type_of_essay

    def import_files(self, date):
        """
        Imports the .xls files

        Parameters:
            ----------
            date: string
                date at which the experiments have been performed
                Format : YYMMDD

        Returns:
            -------
            datafile_list: list
                list of .xls files starting by "type_of_essay"

        """
        path_to_data_folder = utils.reach_data_path(date)
        entries = os.listdir(path_to_data_folder)
        type_of_essay_to_test = date + '_' + self.type_of_essay
        len_type_of_essay = len(type_of_essay_to_test)
        datafile_list = []
        for i in range(len(entries)):
            outfile = entries[i]
            if outfile[0:len_type_of_essay] == type_of_essay_to_test:
                datafile_list.append(outfile)
        return datafile_list
    
    def import_metadatafile(self, date):
        path_to_data_folder = utils.reach_data_path(date)
        entries = os.listdir(path_to_data_folder)
        type_of_essay_to_test = date + '_recap_' + self.type_of_essay
        len_type_of_essay = len(type_of_essay_to_test)
        datafile_list = []
        for i in range(len(entries)):
            outfile = entries[i]
            if outfile[0:len_type_of_essay] == type_of_essay_to_test:
                datafile_list.append(outfile)
        return datafile_list[0]        

    def get_sheets_from_datafile(self, datafile):
        """
        Lists the sheets in the xls file that contain data

        Parameters:
            ----------
            datafile: string
                name of the datafile to be read

        Returns:
            -------
            sheets_list_with_data: list
                list of the sheets in the datafile

        """
        date = datafile[0:6]
        path_to_datafile = utils.reach_data_path(date)
        datafile_as_pds = pd.ExcelFile(path_to_datafile/datafile)
        sheets_list = datafile_as_pds.sheet_names
        sheets_list_with_data = [i for i in sheets_list if i.startswith(date) or i.startswith('RDG') or i.startswith('FF') or i.startswith('C')]
        return datafile_as_pds, sheets_list_with_data
        
    def read_sheet_in_datafile(self, datafile, sheet):
        """
        Extracts the measured time, force and displacement values in a sheet
        
        Parameters:
            ----------
            datafile: string
                name of the datafile to be read
            sheet: string
                name of the sheet to be read

        Returns:
            -------
            time: pandasArray
                list of the time values (in seconds) in the sheet of the datafile
            force: pandasArray
                list of the force values (in Newtons) in the sheet of the datafile
            disp: pandasArray
                list of the displacement values (in mm) in the sheet of the datafile

        """
        date = datafile[0:6]
        path_to_datafile = utils.reach_data_path(date) / datafile
        data_in_sheet = pd.read_excel(path_to_datafile, sheet_name=sheet, header=2, names=["s", "N", "mm" ], usecols="A:C", decimal=',') 
        time_with_offset = data_in_sheet.s
        force_with_offset = data_in_sheet.N
        disp_with_offset = data_in_sheet.mm
        time_with_offset, force_with_offset, disp_with_offset = time_with_offset.to_numpy(), force_with_offset.to_numpy(), disp_with_offset.to_numpy() 
        nb_points = len(time_with_offset)
        time = np.array([time_with_offset[i] for i in range(nb_points) if force_with_offset[i]>0.05])
        force = np.array([force_with_offset[i] for i in range(nb_points) if force_with_offset[i]>0.05])
        disp = np.array([disp_with_offset[i] for i in range(nb_points) if force_with_offset[i]>0.05])
        time0 = time[0]
        time = time - time0
        force0 = force[0]
        force = force - force0
        disp0 = disp[0]
        disp = disp - disp[0]
        return time, force, disp

    def get_umax_and_strain(self, datafile, sheet):
        date = datafile[0:6]
        path_to_datafile = utils.reach_data_path(date) / datafile
        data_in_sheet = pd.read_excel(path_to_datafile, sheet_name=sheet, header=2, names=["s", "N", "mm" ], usecols="A:D", decimal=',') 
        time_with_offset = data_in_sheet.s
        force_with_offset = data_in_sheet.N
        disp_with_offset = data_in_sheet.mm
        time_with_offset, force_with_offset, disp_with_offset = time_with_offset.to_numpy(), force_with_offset.to_numpy(), disp_with_offset.to_numpy() 
        nb_points = len(time_with_offset)
        time = np.array([time_with_offset[i] for i in range(nb_points) if force_with_offset[i]>0.05])
        force = np.array([force_with_offset[i] for i in range(nb_points) if force_with_offset[i]>0.05])
        disp = np.array([disp_with_offset[i] for i in range(nb_points) if force_with_offset[i]>0.05])

        

    def read_metadatas_zwick(self, metadatafile):
        date = metadatafile[0:6]
        path_to_metadatafile = utils.reach_data_path(date) / metadatafile
        metadatas = pd.read_excel(path_to_metadatafile, sheet_name='zwick', header=1, names=["Id", "imposed_disp", "speed_mm_min" ], usecols="A:C", decimal=',') 
        ids = metadatas.Id
        imposed_disp = metadatas.imposed_disp
        speed = metadatas.speed_mm_min
        imposed_disp_dict = {ids.tolist()[i]: imposed_disp.tolist()[i] for i in range(len(ids.tolist()))}
        speed_dict = {ids.tolist()[i]: speed.tolist()[i] for i in range(len(ids.tolist()))}
        return imposed_disp_dict, speed_dict
        
    def is_testing_failed(self, datafile, sheet):
        _, force, _ = self.read_sheet_in_datafile(datafile, sheet)
        bool = False
        max_force = np.max(force)
        if max_force > 1.2 or max_force < 1:
            bool = True
        return bool
    
    def find_only_correct_sheets_in_datafile(self, datafile):
        _, sheets_list_with_data = self.get_sheets_from_datafile(datafile)
        list_of_correct_sheets = [sheet for sheet in sheets_list_with_data if not self.is_testing_failed(datafile, sheet)]
        return list_of_correct_sheets
            
    def plot_data_from_sheet(self, datafile, sheet, createfigure, savefigure, fonts):
        time, force, disp = self.read_sheet_in_datafile(datafile, sheet)
        fig_force_vs_time = createfigure.rectangle_rz_figure(pixels=180)
        fig_disp_vs_time = createfigure.rectangle_rz_figure(pixels=180)
        fig_force_vs_disp = createfigure.rectangle_rz_figure(pixels=180)
        ax_force_vs_time = fig_force_vs_time.gca()
        ax_disp_vs_time = fig_disp_vs_time.gca()
        ax_force_vs_disp = fig_force_vs_disp.gca()
        date = datafile[0:6]
        metadatafile = self.import_metadatafile(date)
        imposed_disp_dict, speed_dict = self.read_metadatas_zwick(metadatafile)
        imposed_disp = imposed_disp_dict[sheet]
        imposed_speed = speed_dict[sheet]
        kwargs = {"color":'k', "linewidth": 2}
        ax_force_vs_time.plot(time, force, label='Umax = ' + str(imposed_disp) + ' mm \n vitesse retour = ' + str(imposed_speed) + 'mm/min', **kwargs)
        ax_disp_vs_time.plot(time, disp, label='Umax = ' + str(imposed_disp) + ' mm \n vitesse retour = ' + str(imposed_speed) + 'mm/min', **kwargs)
        ax_force_vs_disp.plot(force, disp, label='Umax = ' + str(imposed_disp) + ' mm \n vitesse retour = ' + str(imposed_speed) + 'mm/min', **kwargs)
        ax_force_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        ax_force_vs_time.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
        ax_disp_vs_time.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
        ax_force_vs_time.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
        ax_disp_vs_time.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
        ax_force_vs_disp.legend(prop=fonts.serif(), loc='center right', framealpha=0.7)
        savefigure.save_as_png(fig_force_vs_time, sheet + "_force_vs_time")
        savefigure.save_as_png(fig_disp_vs_time, sheet + "_disp_vs_time")
        savefigure.save_as_png(fig_force_vs_disp, sheet + "_force_vs_disp")
        
    def build_fig_to_plot_data_from_sheet_meat(self, datafile, sheet, createfigure, savefigure, fonts):
        time, force, disp = self.read_sheet_in_datafile(datafile, sheet)
        fig_force_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_disp_vs_time = createfigure.rectangle_figure(pixels=180)
        fig_force_vs_disp = createfigure.rectangle_figure(pixels=180)
        ax_force_vs_time = fig_force_vs_time.gca()
        ax_disp_vs_time = fig_disp_vs_time.gca()
        ax_force_vs_disp = fig_force_vs_disp.gca()
        date = datafile[0:6]
        colors = sns.color_palette("tab10")
        kwargs = {"color":colors[7], "linewidth": 2}
        ax_force_vs_time.plot(time, force, alpha = 0.75, **kwargs)
        ax_disp_vs_time.plot(time, disp, alpha = 0.75, **kwargs)
        ax_force_vs_disp.plot(disp, force, alpha = 0.75, **kwargs)
        ax_force_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        ax_force_vs_time.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
        ax_disp_vs_time.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_ylim([0,None])
        ax_force_vs_time.set_ylim([0,None])
        ax_disp_vs_time.set_ylim([0,None])
        ax_force_vs_disp.set_xlim([0,None])
        ax_force_vs_time.set_xlim([0,None])
        ax_disp_vs_time.set_xlim([0,None])
        return date, fig_disp_vs_time, fig_force_vs_disp, fig_force_vs_time, ax_disp_vs_time, ax_force_vs_disp, ax_force_vs_time
        
    def plot_data_from_sheet_meat(self, datafile, sheet, createfigure, savefigure, fonts):
        date, fig_disp_vs_time, fig_force_vs_disp, fig_force_vs_time, ax_disp_vs_time, ax_force_vs_disp, ax_force_vs_time = self.build_fig_to_plot_data_from_sheet_meat(datafile, sheet, createfigure, savefigure, fonts)
        savefigure.save_as_png(fig_force_vs_time, date + '_' + sheet + "_force_vs_time")
        savefigure.save_as_png(fig_disp_vs_time, date + '_' + sheet + "_disp_vs_time")
        savefigure.save_as_png(fig_force_vs_disp, date + '_' + sheet + "_force_vs_disp")
        
    def plot_all_data(self, date, createfigure, savefigure, fonts):
        fig_force_vs_time = createfigure.rectangle_rz_figure(pixels=180)
        fig_disp_vs_time = createfigure.rectangle_rz_figure(pixels=180)
        fig_force_vs_disp = createfigure.rectangle_rz_figure(pixels=180)
        ax_force_vs_time = fig_force_vs_time.gca()
        ax_disp_vs_time = fig_disp_vs_time.gca()
        ax_force_vs_disp = fig_force_vs_disp.gca()
        colors = sns.color_palette("tab10")
        kwargs = {"color":colors[7], "linewidth": 2}
        ax_force_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_disp_vs_time.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        ax_force_vs_time.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
        ax_disp_vs_time.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
        ax_force_vs_disp.set_ylim([0,1.2])
        ax_force_vs_time.set_ylim([0,1.2])
        # ax_disp_vs_time.set_ylim([0,None])
        ax_force_vs_disp.set_xlim([0,10])
        # ax_force_vs_time.set_xlim([0,None])
        # ax_disp_vs_time.set_xlim([0,None])        
        
        color_rocket = sns.color_palette("rocket")
        color_FF = color_rocket[3]
        color_RDG = color_rocket[1]

        datafile_list = self.import_files(date)
        for datafile in datafile_list:
            datafile_as_pds, sheets_list_with_data = self.get_sheets_from_datafile(datafile)
            correct_sheets_in_data = self.find_only_correct_sheets_in_datafile(datafile)
            sheets_FF = [s for s in correct_sheets_in_data if 'FF' in s]
            sheets_RDG = [s for s in correct_sheets_in_data if 'RDG' in s]
            for sheet in sheets_FF:
                time, force, disp = self.read_sheet_in_datafile(datafile, sheet)
                ax_force_vs_time.plot(time, force, '-', color=color_FF)
                ax_disp_vs_time.plot(time, disp, '-', color=color_FF)
                ax_force_vs_disp.plot(disp, force, '-', color=color_FF)
            for sheet in sheets_RDG:
                time, force, disp = self.read_sheet_in_datafile(datafile, sheet)
                ax_force_vs_time.plot(time, force, '-', color=color_RDG)
                ax_disp_vs_time.plot(time, disp, '-', color=color_RDG)
                ax_force_vs_disp.plot(disp, force, '-', color=color_RDG)
        ax_force_vs_time.plot(time[0], force[0], color=color_FF, label='FF')
        ax_disp_vs_time.plot(time[0], disp[0], color=color_FF, label='FF')
        ax_force_vs_disp.plot(disp[0], force[0], color=color_FF, label='FF')
        ax_force_vs_time.plot(time[0], force[0], color=color_RDG, label='RDG')
        ax_disp_vs_time.plot(time[0], disp[0], color=color_RDG, label='RDG')
        ax_force_vs_disp.plot(disp[0], force[0], color=color_RDG, label='RDG')
        ax_force_vs_time.legend()
        ax_disp_vs_time.legend()
        ax_force_vs_disp.legend()
        savefigure.save_as_png(fig_force_vs_time, date + "all_force_vs_time")
        savefigure.save_as_png(fig_disp_vs_time, date + "all_disp_vs_time")
        savefigure.save_as_png(fig_force_vs_disp, date + "all_force_vs_disp")

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230718']
    types_of_essay = ['C_Indentation_relaxation_500N_force.xlsx']#,'C_Indentation_relaxation_maintienFnulle_500N_trav.xls',  'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    correct_sheets_in_data = files_zwick.find_only_correct_sheets_in_datafile(datafile)
    # sheet1 = correct_sheets_in_data[0]
    # time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet1)
    # for sheet in correct_sheets_in_data:
    #     files_zwick.plot_data_from_sheet_meat( datafile, sheet, createfigure, savefigure, fonts)
    # metadatafile = files_zwick.import_metadatafile(experiment_dates[0])
    # imposed_disp_dict, speed_dict = files_zwick.read_metadatas_zwick(metadatafile)
    for date in experiment_dates:
        files_zwick.plot_all_data(date, createfigure, savefigure, fonts)
    print('hello')