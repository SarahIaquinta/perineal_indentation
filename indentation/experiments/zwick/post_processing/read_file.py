import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
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
        
    def read_sheet(self, datafile, sheet):
        date = datafile[0:6]
        path_to_datafile = utils.reach_data_path(date)
        data_in_sheet = pd.read_excel(open(path_to_datafile, 'rb'), sheet_name=sheet, header=2, names=["s", "N", "mm" ], usecols="A:C", decimal=',') 
        print('data') 

if __name__ == "__main__":
    # createfigure = CreateFigure()
    # fonts = Fonts()
    # savefigure = SaveFigure()
    experiment_dates = ['230515']#, '230411']#'230331', '230327', '230403']
    types_of_essay = ['C_Indentation_relaxation_maintienFnulle_500N_trav.xls']#, 'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    sheet1 = sheets_list_with_data[0]
    files_zwick.read_sheet(datafile, sheet1)
    print('hello')