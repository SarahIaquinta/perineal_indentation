import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path
import utils
import os

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





def read_datafile(filename):
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
    
    utils.export_data_output_as_txt(filename, mat_Z, vec_time, vec_pos_axis)
    utils.export_data_output_as_pkl(filename, mat_Z, vec_time, vec_pos_axis)
    return vec_time, vec_pos_axis, mat_Z

def plot_Z_profile(filename):
    mat_Z, vec_time, vec_pos_axis = utils.extract_data_from_pkl(filename)
    plt.imshow(mat_Z)
    plt.show()
    plt.close()
    

if __name__ == "__main__":
    current_path = utils.get_current_path()
    experiment_date = '230403'
    path_to_data = utils.reach_data_path(experiment_date)
    print(path_to_data)
    files = Files('FF')
    list_of_FF_files = files.import_files(experiment_date)
    filename_0 =list_of_FF_files[1]
    read_datafile(filename_0)
    plot_Z_profile(filename_0)
    print('hello')