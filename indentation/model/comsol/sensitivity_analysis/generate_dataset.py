import itertools
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import matplotlib.font_manager as font_manager
import seaborn as sns

def generate_dataset(expected_sample_size):
    sampler = qmc.Sobol(d=2, scramble=False)
    n = int(np.log(expected_sample_size) / np.log(2))
    sample_size = int(2 ** n)
    print("Sample size = ", sample_size)
    sample = sampler.random_base2(m=n)  # defining sample size = 4096
    var1 = np.array([sample[i][0] for i in range(len(sample))])
    var2 = np.array([sample[i][1] for i in range(len(sample))])
    lambda_max = 1
    lambda_min = 1.5
    damage_max = 0
    damage_min = 0.9

    lambda_list = [i * (lambda_max - lambda_min) + lambda_min for i in var1]
    damage_list = [i * (damage_max - damage_min) + damage_min for i in var2]

    return lambda_list, damage_list

def export_dataset_as_txt(lambda_list, damage_list):
    path_to_dataset = r'C:\Users\siaquinta\Documents\Projet Périnée\perineal_indentation\indentation\model\comsol\sensitivity_analysis'
    complete_txt_filename = path_to_dataset + "/comsol_input_dataset_size" + str(len(lambda_list)) +".txt"
    f = open(complete_txt_filename, "w")
    f.write("INPUT PARAMETERS FOR COMSOL - SIZE " + str(len(lambda_list)) + " \n")
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