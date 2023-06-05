import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pandas as pd

class Files:
    """
    A class define the .csv files to import.

    Attributes:
        ----------
        beginning: string
            beginning of the csv file to be extracted
            corresponds to the meat piece to be studied. 
            Possible values : 
                - FF (faux filet)
                - RDG (rond de gite)
            To precise the side of the meat piece (right or left),
                add 1 or 2 after the name of the meat peace, otherwise
                both sides will be gathered in the same list.

    Methods:
        -------
        import_files(self):
            Returns a list of the imported the .csv files

    """

    def __init__(self, beginning):
        """
        Constructs all the necessary attributes for the files object.

        Parameters:
            ----------
            beginning: string
                beginning of the csv file to be extracted

        Returns:
            -------
            None
        """

        self.beginning = beginning

    def import_files(self, date):
        """
        Imports the .csv files

        Parameters:
            ----------
            date: string
                date at which the experiments have been performed
                Format : YYMMDD

        Returns:
            -------
            datafile_list: list
                list of .csv files starting by "beginning"

        """
        path_to_data_folder = utils.reach_data_path(date)
        entries = os.listdir(path_to_data_folder)
        beginning_to_test = date + '_' + self.beginning
        len_beginning = len(beginning_to_test)
        datafile_list = []
        for i in range(len(entries)):
            outfile = entries[i]
            if outfile[0:len_beginning] == beginning_to_test:
                datafile_list.append(outfile)
        return datafile_list

    def read_datafile(self, filename):
        """
        Extracts data from the .csv file and extract it as a pkl

        Parameters:
            ----------
            filename: string
                identification name of the testing

        Returns:
            -------
            None

        """        
        date = filename[0:6]
        path_to_data = utils.reach_data_path(date)
        path_to_filename = path_to_data / filename
        cur_file = open(path_to_filename,'r')
        all_line = cur_file.readlines()

        flag_continue = False
        l_vec_Z = []
        l_time = []
        for cur_line in all_line:
            if cur_line[0:12] == 'Frame,Source':
                II = cur_line.find('Axis')
                cur_line0 = cur_line[II+5:]
                II = [pos for pos, char in enumerate(cur_line0) if char == ',']
                vec_pos_axis = np.zeros(len(II)+1)
                vec_pos_axis[0] = float(cur_line0[0:II[0]])
                for it_c in range(len(II)-1):
                    vec_pos_axis[it_c+1] = float(cur_line0[II[it_c]+1:II[it_c+1]])
                vec_pos_axis[-1] = float(cur_line0[II[-1]+1:])
                flag_continue = True
            
            elif flag_continue and (len(cur_line)>10):
                II = [pos for pos, char in enumerate(cur_line) if char == ',']
                l_time.append(float(cur_line[II[1]+1:II[2]]))
                II = cur_line.find('Z')
                cur_line0 = cur_line[II+1:]
                II = [pos for pos, char in enumerate(cur_line0) if char == ',']
                vec_Z = np.zeros(len(II)+1)
                if II[0]==0:
                    vec_Z[0] = nan
                else:
                    vec_Z[0] = float(cur_line0[0:II[0]])
                
                for it_c in range(len(II)-1):
                    if (II[it_c]+1)==(II[it_c+1]):
                        vec_Z[it_c+1] = nan
                    else:    
                        vec_Z[it_c+1] = float(cur_line0[II[it_c]+1:II[it_c+1]])
                
                if (II[-1]+2)==len(cur_line0):
                    vec_Z[0] = nan
                else:
                    vec_pos_axis[-1] = float(cur_line0[II[-1]+1:])
                
                l_vec_Z.append(vec_Z) 
        mat_Z = np.vstack(l_vec_Z)
        vec_time = np.array(l_time)
        vec_time_rescaled = vec_time - min(vec_time)
        
        utils.export_data_output_as_txt(filename, mat_Z, vec_time_rescaled, vec_pos_axis)
        utils.export_data_output_as_pkl(filename, mat_Z, vec_time_rescaled, vec_pos_axis)

def read_all_files(experiment_dates, meat_pieces):
    """
    Reads all the files generated as a given date for a given meatpiece

    Parameters:
        ----------
        experiment_dates: list of strings
            list of the dates for which the files need to be read
        meat_pieces: list of strings
            list of the meatpieces

    Returns:
        -------
        None

    """
    existing_processed_filenames = utils.get_existing_processed_data()
    for experiment_date in tqdm(experiment_dates):
        for meat_piece in tqdm(meat_pieces) :
            files_meat_piece = Files(meat_piece)
            list_of_meat_piece_files = files_meat_piece.import_files(experiment_date)
            for filename in tqdm(list_of_meat_piece_files) :
                # print (filename , ' : currently evaluating')
                if filename[0:-4] not in existing_processed_filenames:
                    # print (filename , ' not processed yet : processing has been launched')
                    files_meat_piece.read_datafile(filename)       


def read_metadatas_laser(metadatafile):
    """
    Extracts the displacement imposed for the laser experiment from a metadatafile

    Parameters:
        ----------
        metadatafile: string
            name of the .xls file containing the metadata
        imposed_disp_dict: float
            displacement that has been imposed. This value corresponds to the distance between
            the upper and lower Zwick tools, measured at the beginning of the experiment.

    Returns:
        -------
        None

    """
    date = metadatafile[0:6]
    path_to_metadatafile = utils.reach_data_path(date) / metadatafile
    metadatas = pd.read_excel(path_to_metadatafile, sheet_name='laser', header=1, names=["Id", "imposed_disp"], usecols="A:B", decimal=',') 
    ids = metadatas.Id
    imposed_disp = metadatas.imposed_disp
    imposed_disp_dict = {ids.tolist()[i]: imposed_disp.tolist()[i] for i in range(len(ids.tolist()))}
    return imposed_disp_dict


if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230515']#, '230411']#'230331', '230327', '230403']
    meat_pieces = ['P002', 'P011']#, 'RDG']
    read_all_files(experiment_dates, meat_pieces)




