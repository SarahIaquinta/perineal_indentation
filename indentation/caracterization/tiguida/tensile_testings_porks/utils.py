import glob
from indentation import make_data_folder
import os

# DIRECTORY OF THE FILES
def load_data():
    path_to_data = make_data_folder('caracterization\\tiguida\\tensile_testings_porks\\data_traction')
    entries = os.listdir(path_to_data)
    path_to_data_as_string = str(path_to_data)
    datas_cochon4 = sorted(glob.glob(path_to_data_as_string + r'\%04*-1%**.txt',recursive = True))
    datas_cochon5 = sorted(glob.glob(path_to_data_as_string + r'\%05*-1%**.txt',recursive = True))
    datas_cochon7 = sorted(glob.glob(path_to_data_as_string + r'\%07*-1%**.txt',recursive = True))
    datas_cochon8 = sorted(glob.glob(path_to_data_as_string + r'\%08*-1%**.txt',recursive = True))
    datas_cochon9 = sorted(glob.glob(path_to_data_as_string + r'\%09*-1%**.txt',recursive = True))
    all_datas = [datas_cochon4, datas_cochon5, datas_cochon7, datas_cochon8, datas_cochon9]
    cochon_ids = [4, 5, 7, 8, 9]
    nb_of_cochons = len(cochon_ids)
    return all_datas, cochon_ids, nb_of_cochons

