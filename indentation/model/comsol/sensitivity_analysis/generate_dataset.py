"""
This file contains functions that are used to build an input dataset containing
the samples (tuples lambda and damage) to be used as input parameters for the COMSOL
model. The corresponding temporal evolution of stress and displacement, computed with the COMSOL model
for each set of input parameters, are then post processed to extract indicators.
Note that the elongation parameter cannot be called "lambda" in the script as "lambda"
is used as an internal variable in Python. For this reason, it is called "elongation" in the script.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import matplotlib.font_manager as font_manager
import seaborn as sns

def generate_dataset(expected_sample_size):
    """
    Generates a Sobol quasi Monte Carlo dataset. 

    Parameters:
        ----------
        expected_sample_size: float
            expected size for the dataset
            the real size of the generated dataset is the closest inferior power of 2

    Returns:
        -------
        lambda_list, damage_list: lists
            lists of values of lambda and damage

    """
    sampler = qmc.Sobol(d=2, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    sample = sampler.random_base2(m=n)  
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    lambda_max = 1
    lambda_min = 1.5
    damage_max = 0.8
    damage_min = 0
    lambda_list = [i * (lambda_max - lambda_min) + lambda_min for i in var1]
    damage_list = [i * (damage_max - damage_min) + damage_min for i in var2]
    return lambda_list, damage_list

def export_dataset_as_txt(lambda_list, damage_list):
    """
    Exports a dataset as a txt file

    Parameters:
        ----------
        lambda_list: list
            list of values of lambda
        damage_list: list
            list of values of damage
            
    Returns:
        -------
        None

    """
    path_to_dataset = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\model\comsol\sensitivity_analysis'
    complete_txt_filename = path_to_dataset + "/comsol_input_dataset_size" + str(len(lambda_list)) +".txt"
    f = open(complete_txt_filename, "w")
    f.write("INPUT PARAMETERS FOR COMSOL - SIZE " + str(len(lambda_list)) + " \n lambda in [" + str(min(lambda_list)) + " ; " + str(max(lambda_list)) + " ] damage in [" + str(min(damage_list)) + " ; " + str(max(damage_list)) + " ] \n" )
    f.write("Id \t lambda \t damage \n")
    for i in range(len(lambda_list)):
        f.write(
            str(int(i+1))
            + "\t"
            + str(np.round(lambda_list[i], 5))
            + "\t"
            + str(np.round(damage_list[i], 5))
            + "\n"
        )
    f.close()

def plot_dataset(lambda_list, damage_list):
    """
    Plots the dataset 

    Parameters:
        ----------
        lambda_list: list
            list of values of lambda
        damage_list: list
            list of values of damage

    Returns:
        -------
        None

    """
    path_to_figure = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\model\comsol\sensitivity_analysis'
    complete_figure_filename = path_to_figure + "/comsol_input_dataset_size" + str(len(lambda_list)) +".png"
    fig = plt.figure(figsize=(7, 7), dpi=180, constrained_layout=True)
    palette = sns.color_palette("Blues", as_cmap=False, n_colors=len(lambda_list))
    ax = fig.gca()
    for i in range(len(lambda_list)):
        ax.plot(lambda_list[i], damage_list[i], 'o', mfc=palette[i], mec=palette[i])
        # ax.annotate(str(int(i+1)), (lambda_list[i]+0.005, damage_list[i]), color = palette[i])
    ax.set_xlabel(r'$\lambda$ [-]', font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18))
    ax.set_ylabel('d [-]', font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18))
    ax.set_xticks([1, 1.1, 1.2, 1.3, 1.4, 1.5], font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18))
    ax.set_xticklabels([1, 1.1, 1.2, 1.3, 1.4, 1.5], font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18))
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18))
    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], font = font_manager.FontProperties(family="serif", weight="normal", style="normal", size=18))
    fig.savefig(complete_figure_filename, format="png")


if __name__ == "__main__":
    expected_sample_size = 150
    lambda_list, damage_list = generate_dataset(expected_sample_size)
    export_dataset_as_txt(lambda_list, damage_list)
    plot_dataset(lambda_list, damage_list)