"""
This file contains the functions necessary for the evaluation of the Sobol sensitivity
indices.
"""

import time
import openturns as ot
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure
ot.Log.Show(ot.Log.NONE)
import numpy as np
import pickle
import utils

class Distribution:
    def __init__(self):
        """
        Constructs all the necessary attributes for the Distribution object. This class is used to 
        define the distribution for the input parameter samples 

        Parameters:
            ----------
            None

        Returns:
            -------
            None
        """
    def extract_distributions_from_input(self):
        """Extracts the distribution of the input parameters based on the value contained in
        the original dataset

        Returns:
            alpha_p_distribution (distribution ot object): distribution of the alpha_p indicator (called i_time in the other files)
            beta_distribution (distribution ot object): distribution of the beta indicator (called relaxation_slope in the other files)
            delta_f_distribution (distribution ot object): distribution of the delta_f indicator
            delta_f_star_distribution (distribution ot object): distribution of the delta_f_star indicator
            a_distribution (distribution ot object): distribution of the a indicator
        """
        complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
        with open(complete_pkl_filename, "rb") as f:
            [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
        factory = ot.UserDefinedFactory()
        alpha_p_as_list = list(alpha_p_dict.values())
        beta_as_list = list(beta_dict.values())
        delta_f_as_list = list(delta_f_dict.values())
        delta_f_star_as_list = list(delta_f_star_dict.values())
        a_as_list = list(a_dict.values())
        alpha_p_as_sample = ot.Sample(len(alpha_p_as_list), 1)
        beta_as_sample = ot.Sample(len(beta_as_list), 1)
        delta_f_as_sample = ot.Sample(len(delta_f_as_list), 1)
        delta_f_star_as_sample = ot.Sample(len(delta_f_star_as_list), 1)
        a_as_sample = ot.Sample(len(a_as_list), 1)
        for i in range(len(alpha_p_as_sample)):
            alpha_p_as_sample[i, 0] = alpha_p_as_list[i]
            beta_as_sample[i, 0] = beta_as_list[i]
            delta_f_as_sample[i, 0] = delta_f_as_list[i]
            delta_f_star_as_sample[i, 0] = delta_f_star_as_list[i]
            a_as_sample[i, 0] = a_as_list[i]             
        alpha_p_distribution = factory.build(alpha_p_as_sample)
        beta_distribution = factory.build(beta_as_sample)
        delta_f_distribution = factory.build(delta_f_as_sample)
        delta_f_star_distribution = factory.build(delta_f_star_as_sample)
        a_distribution = factory.build(a_as_sample)
        return alpha_p_distribution, beta_distribution, delta_f_distribution, delta_f_star_distribution, a_distribution

    def composed(self):
        """
        creates a uniform distribution of the 5 input parameters

        Parameters:
            ----------
            None

        Returns:
            -------
            distribution: ot class
                uniform distribution of the 5 input parameters, computed wth Openturns.

        """
        alpha_p_distribution, beta_distribution, delta_f_distribution, delta_f_star_distribution, a_distribution = self.extract_distributions_from_input()
        composed_distribution = ot.ComposedDistribution(
            [
                alpha_p_distribution,
                beta_distribution,
                delta_f_distribution,
                delta_f_star_distribution,
                a_distribution,
            ]
        )
        return composed_distribution


def compute_sensitivity_algo_MauntzKucherenko(distribution, sensitivity_experiment_size):
    """
    Computes the sensitivity algorithms computed after the Mauntz Kucherenko method. 
    Another methods could have been chosen (Saltelli, Martinez etc) 

    Parameters:
        ----------
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        sensitivity_experiment_size: float (int)
            amount of estimations of the ANN to evaluate the sensitivity

    Returns:
        -------
        sensitivityAnalysis: ot class
            sensitivity algorithm computed with the openturns class

    """
    composed_distribution = distribution.composed()
    myExperiment = ot.LowDiscrepancyExperiment(
        ot.SobolSequence(), composed_distribution, sensitivity_experiment_size, True
    )
    f = ot.PythonFunction(5, 1, utils.compute_output_ANN_inverse_model) #5 is the number of input parameters
    sensitivityAnalysis = ot.MauntzKucherenkoSensitivityAlgorithm(myExperiment, f, True)
    return sensitivityAnalysis


def compute_and_export_sensitivity_algo_MauntzKucherenko(
    distribution,
    sensitivity_experiment_size,
):
    """
    computes the sensitivity algorithms computed after the MauntzKucherenko method and exports it to a .pkl
    file

    Parameters:
        ----------
        distribution: class
            class that defines the distribution of the input parameters whose influence is tested
            in the sensitivity analysis
        sensitivity_experiment_size: float (int)
            amount of estimations of the metamodel to evaluate the sensitivity

    Returns:
        -------
        None

    """

    sensitivity_algo_MauntzKucherenko = compute_sensitivity_algo_MauntzKucherenko(
        distribution, sensitivity_experiment_size
    )
    complete_pkl_filename_sensitivy_algo = "sensitivity_Mauntz_algo_inverse_ANN.pkl"
    with open(utils.get_path_to_processed_data() / complete_pkl_filename_sensitivy_algo, "wb") as f:
        pickle.dump(
            [sensitivity_algo_MauntzKucherenko],
            f,
        )


def get_indices_and_confidence_intervals():
    """Computes the Sobol sensitivity indices along with their corresponding 95% confidence intervals

    Returns:
        first_order_indices (list): list of the first order Sobol sensitivity indices
        total_order_indices (list): list of the total order Sobol sensitivity indices
        first_order_indices_confidence_errorbar (list): list of the 95% confidence interval of the first order Sobol sensitivity indices
        total_order_indices_confidence_errorbar (list): list of the 95% confidence interval of the total order Sobol sensitivity indices
    """
    complete_pkl_filename_sensitivy_algo = "sensitivity_Mauntz_algo_inverse_ANN.pkl"
    with open(utils.get_path_to_processed_data() / complete_pkl_filename_sensitivy_algo, "rb") as f:
        [sensitivity_algo] = pickle.load(f)
    first_order_indices = sensitivity_algo.getFirstOrderIndices()
    first_order_indices_confidence_interval = sensitivity_algo.getFirstOrderIndicesInterval()
    first_order_indices_confidence_lowerbounds = [
        first_order_indices_confidence_interval.getLowerBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    first_order_indices_confidence_upperbounds = [
        first_order_indices_confidence_interval.getUpperBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    total_order_indices = sensitivity_algo.getTotalOrderIndices()
    total_order_indices_confidence_interval = sensitivity_algo.getTotalOrderIndicesInterval()
    total_order_indices_confidence_lowerbounds = [
        total_order_indices_confidence_interval.getLowerBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    total_order_indices_confidence_upperbounds = [
        total_order_indices_confidence_interval.getUpperBound()[k] for k in [0, 1, 2, 3, 4]
    ]
    first_order_indices_confidence_errorbar = np.zeros((2, 5))
    total_order_indices_confidence_errorbar = np.zeros((2, 5))
    for k in range(5):
        first_order_indices_confidence_errorbar[0, k] = (
            first_order_indices[k] - first_order_indices_confidence_lowerbounds[k]
        )
        first_order_indices_confidence_errorbar[1, k] = (
            first_order_indices_confidence_upperbounds[k] - first_order_indices[k]
        )
        total_order_indices_confidence_errorbar[0, k] = (
            total_order_indices[k] - total_order_indices_confidence_lowerbounds[k]
        )
        total_order_indices_confidence_errorbar[1, k] = (
            total_order_indices_confidence_upperbounds[k] - total_order_indices[k]
        )
    return first_order_indices, total_order_indices, first_order_indices_confidence_errorbar,total_order_indices_confidence_errorbar
    
def plot_results_sensitivity_analysis(
    createfigure,
    fonts,
    savefigure
):
    """
    Plots the first and total Sobol indices

    Parameters:
        ----------
        createfigure (class): class used to provide consistent settings for the creation of figures
        savefigure (class):class used to provide consistent settings for saving figures
        fonts (class): class used to provide consistent fonts for the figures

    """

    (
        first_order_indices,
        total_order_indices,
        first_order_indices_confidence_errorbar,
        total_order_indices_confidence_errorbar,
    ) = get_indices_and_confidence_intervals()
    fig = createfigure.square_figure_7(pixels=180)
    ax = fig.gca()
    ax.errorbar(
        [0, 1, 2, 3, 4],
        first_order_indices,
        yerr=first_order_indices_confidence_errorbar,
        label="First order indice",
        color="k",
        marker="v",
        markersize=12,
        linestyle="None",
    )
    ax.errorbar(
        [0, 1, 2, 3, 4],
        total_order_indices,
        yerr=total_order_indices_confidence_errorbar,
        label="Total indice",
        color="m",
        marker="D",
        markersize=12,
        linestyle="None",
    )
    ax.set_xlim((-0.2, 4.2))
    ax.set_ylim((-0.05, 1.05))
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(
        [
            r"$\alpha'$",
            r"$\beta$",
            r"$\Delta F$",
            r"$\Delta F^*$",
            r"$a$"
        ],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(
        ["0", "0.25", "0.5", "0.75", "1"],
        font=fonts.serif(),
        fontsize=fonts.axis_legend_size(),
    )
    ax.set_xlabel("Variable", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel("Sobol indice [ - ]", font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.legend(prop=fonts.serif(), loc="upper right", framealpha=0.7)
    savefigure.save_as_png(fig, "sobolindices_inverse_model")

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    distribution = Distribution()
    plot_results_sensitivity_analysis(
    createfigure,
    fonts,
    savefigure
)
    print('hello')
    
    