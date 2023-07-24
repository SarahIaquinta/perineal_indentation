"""
This file contains tools to extract data from the laser raw datafiles, 
in order to compute spatio temporal evolution of the altitude measured
by the laser during the experiment. 
The script catches the region of interest, based on the location dictionnary
(that needs to be entered manually), then applies a median filter to smooth
noised experimental data and finally returns the spatial and temporal discretisation vector,
with the corresponding altitudes.
"""

import numpy as np
from math import nan
import utils
import os
from indentation.experiments.laser.figures.utils import CreateFigure, Fonts, SaveFigure
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from indentation.experiments.zwick.post_processing.utils import find_nearest


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
        self.locations = {"230331": [0, 20],
                        "230327": [-10, 10],
                        "230403": [8, 40],
                        "230407": [-30, -10],
                        "230411": [-10, 10],
                        "230718":[-30,5]}

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

    def read_datafile_v0(self, filename):
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
        print('starting ', filename)
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

        # x_lim = locations[filename]
        mat_Z = np.vstack(l_vec_Z)
        vec_time = np.array(l_time)
        
        [x_lim_inf, x_lim_sup] = self.locations[date]
        index_where_pos_is_x_lim_inf = np.where(vec_pos_axis==find_nearest(vec_pos_axis, x_lim_inf))[0][0]
        index_where_pos_is_x_lim_sup = np.where(vec_pos_axis==find_nearest(vec_pos_axis, x_lim_sup))[0][0]
        
        mat_Z = mat_Z[:,index_where_pos_is_x_lim_inf:index_where_pos_is_x_lim_sup]
        # vec_time = vec_time[int(A[0][1]):int(A[1][1])]
        vec_pos_axis = vec_pos_axis[index_where_pos_is_x_lim_inf:index_where_pos_is_x_lim_sup]

        # Apply median filter
        mat_Z_n = np.zeros(mat_Z.shape)
        cnt = 0
        for it_i in range(mat_Z.shape[0]):
            if (it_i/mat_Z.shape[0]*100)>cnt:
                print(str(cnt).zfill(2)+'%')
                cnt += 1
            
            for it_j in range(mat_Z.shape[1]):
                nn = 2
                flag_continue = True
                while flag_continue:
                    vec_cur = mat_Z[max(0,it_i-nn):min(mat_Z.shape[0],it_i+nn),max(0,it_j-nn):min(mat_Z.shape[1],it_j+nn)].flatten()
                    II = np.where(np.isnan(vec_cur)<1)[0]
                    if len(II)>3:
                        vec_cur = vec_cur[II]
                        flag_continue = False
                    else:
                        nn += 1
                mat_Z_n[it_i,it_j] = np.median(vec_cur)
                
        mat_Z = mat_Z_n
        
        vec_time_rescaled = vec_time - min(vec_time)
        
        utils.export_data_output_as_txt(filename , mat_Z, vec_time_rescaled, vec_pos_axis)
        utils.export_data_output_as_pkl(filename , mat_Z, vec_time_rescaled, vec_pos_axis)
        print(filename, 'done')


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
            # func = partial(files_meat_piece.read_datafile)
            with mp.Pool(3) as executor:  # the argument of the mp.Pool() function is the amount of CPU to parallelize
                executor.map(files_meat_piece.read_datafile, list_of_meat_piece_files)
            # for filename in tqdm(list_of_meat_piece_files) :
                # print (filename , ' : currently evaluating')
                # if filename[0:-4] not in existing_processed_filenames:
                #     # print (filename , ' not processed yet : processing has been launched')
                #     files_meat_piece.read_datafile(filename, locations)       


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
    experiment_dates = ['230718']#'230331', '230403', '230407', '230411']
    meat_pieces = ['FF']

    
    read_all_files(experiment_dates, meat_pieces)
        
    print('hello')





