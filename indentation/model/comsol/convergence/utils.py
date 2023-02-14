import glob
from indentation import make_data_folder
import os
from path import Path
import csv
import pandas as pd
import numpy as np
import re

root_path = Path(__file__).abspath().parent
entries = os.listdir(root_path)

def get_path_to_file(file) : 
    path_to_file =  root_path / file
    return path_to_file


def determine_delimiter(file):
    delimiter = '\s+'
    if file[0:5] == 'param':
        delimiter = '\t'
    return delimiter

def get_headers(file) :
    path_to_file =  get_path_to_file(file)
    with open (path_to_file, 'r') as f:
        # rows = csv.reader(f,delimiter=determine_delimiter(file))
        rows = csv.reader(f,delimiter=' ')
        first_characters = [row[0] for row in rows]
        fc = [r[0] for r in first_characters]
        header_rows = int(fc.count('%'))
        headers = first_characters[0:header_rows]
    return header_rows, headers


def get_column_i(file, i, header_rows):
    path_to_file =  get_path_to_file(file)
    delimiter = determine_delimiter(file)
    lines = [open(path_to_file).readlines()][0]
    column_i_as_strings = [re.split(delimiter, x)[i] for x in lines]
    # with open (path_to_file, 'r') as f:
    #     rows = csv.reader(f,delimiter=determine_delimiter(file))
    #     column_i_as_strings = [row[i] for row in rows]
    column_i_as_strings = column_i_as_strings[header_rows:]
    column_i = [float(x) for x in column_i_as_strings]
    return column_i


