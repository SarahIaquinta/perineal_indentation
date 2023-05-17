import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pandas as pd


class Files_Zwick:
    """
    A class define the .xls files to import.

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
        sheets_list_with_data = [i for i in sheets_list if i.startswith(date)]
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
        time = data_in_sheet.s
        force = data_in_sheet.N
        disp = data_in_sheet.mm
        return time, force, disp

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
        





if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230515']#, '230411']#'230331', '230327', '230403']
    types_of_essay = ['C_Indentation_relaxation_maintienFnulle_500N_trav.xls']#, 'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = sheets_list_with_data[0]
    # time, force, disp = files_zwick.read_sheet_in_datafile(datafile, sheet1)
    for sheet in sheets_list_with_data:
        files_zwick.plot_data_from_sheet( datafile, sheet, createfigure, savefigure, fonts)
    # metadatafile = files_zwick.import_metadatafile(experiment_dates[0])
    # imposed_disp_dict, speed_dict = files_zwick.read_metadatas_zwick(metadatafile)
    print('hello')