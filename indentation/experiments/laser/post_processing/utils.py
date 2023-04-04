import numpy as np
from matplotlib import pyplot as plt
from math import nan
from pathlib import Path

def get_current_path(): 
    """
    Return the current path

    Parameters:
        ----------
        None

    Returns:
        -------
        current_path: str (Path)
            Path format string

    """
    current_path = Path.cwd() / 'indentation\experiments\laser'
    return current_path


def reach_data_path(date):
    """
    Returns the path where the data is stored

    Parameters:
        ----------
        date: string
            date at which the experiments have been performed
            Format :
                YYMMDD

    Returns:
        -------
        path_to_data: str (Path)
            Path format string

    """
    current_path = get_current_path()
    path_to_data = current_path / date
    return path_to_data
    