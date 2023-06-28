"""
This file contains the functions to build an ANN
"""

import os
import numpy as np
import openturns as ot
import seaborn as sns
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure
ot.Log.Show(ot.Log.NONE)
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import utils
import pickle

def define_test_train_datset(training_amount):
    """Separates the dataset into a training and testing dataset. 
    The training dataset will be used to train the ANN whose predictivity
    will be evaluated with the testing dataset

    Args:
        training_amount (float): amount of the dataset that is used for training.
            Between 0 and 1

    Returns:
        input_dataset_train (array):    part of the input dataset used for training
        input_dataset_test (array):     part of the input dataset used for testing
        output_dataset_train (array):   part of the output dataset used for training
        output_dataset_test (array):    part of the output dataset used for testing
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict]
    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
    len_train = int(training_amount*len(indicators_dicts[0].values()))
    len_test = int((1-training_amount)*len(indicators_dicts[0].values()))
    input_dataset_train = np.zeros((len_train, 2))
    input_dataset_test = np.zeros((len_test, 2))
    output_dataset_train = np.zeros((len_train, 5))
    output_dataset_test = np.zeros((len_test, 5))
    input_dataset_train[:, 0] = np.array(list(elongation_dict.values())[:int(training_amount*len(elongation_dict.values()))])
    input_dataset_train[:, 1] = np.array(list(damage_dict.values())[:int(training_amount*len(elongation_dict.values()))])
    input_dataset_test[:, 0] = np.array(list(elongation_dict.values())[int(training_amount*len(elongation_dict.values()))+1:])
    input_dataset_test[:, 1] = np.array(list(damage_dict.values())[int(training_amount*len(elongation_dict.values()))+1:])
    for i in range(len(indicators_dicts)):
        indicator_dict = indicators_dicts[i]
        indicator_vector = list(indicator_dict.values())
        indicator_vector_train = indicator_vector[:int(training_amount*len(indicator_dict.values()))]
        indicator_vector_test = indicator_vector[int(training_amount*len(indicator_dict.values()))+1:]
        output_dataset_train[:, i] = np.array(indicator_vector_train)
        output_dataset_test[:, i] = np.array(indicator_vector_test)
    return input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test

def define_test_train_datset_wo_damage(training_amount):
    """Equivalent to the define_test_train_datset() function, but only the 
    elongation input parameter is taken, while damage is removed.
    This function may be removed in the future.

    Args:
        training_amount (float): amount of the dataset that is used for training.
            Between 0 and 1

    Returns:
        input_dataset_train (array): part of the input dataset used for training
        input_dataset_test (array): part of the input dataset used for testing
        output_dataset_train (array): part of the output dataset used for training
        output_dataset_test (array): part of the output dataset used for testing
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict]
    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
    len_train = int(training_amount*len(indicators_dicts[0].values()))
    len_test = int((1-training_amount)*len(indicators_dicts[0].values()))
    input_dataset_train = np.zeros((len_train))
    input_dataset_test = np.zeros((len_test))
    output_dataset_train = np.zeros((len_train, 5))
    output_dataset_test = np.zeros((len_test, 5))
    input_dataset_train = np.array(list(elongation_dict.values())[:int(training_amount*len(elongation_dict.values()))]).reshape(-1, 1)
    input_dataset_test = np.array(list(elongation_dict.values())[int(training_amount*len(elongation_dict.values()))+1:]).reshape(-1, 1)
    for i in range(len(indicators_dicts)):
        indicator_dict = indicators_dicts[i]
        indicator_vector = list(indicator_dict.values())
        indicator_vector_train = indicator_vector[:int(training_amount*len(indicator_dict.values()))]
        indicator_vector_test = indicator_vector[int(training_amount*len(indicator_dict.values()))+1:]
        output_dataset_train[:, i] = np.array(indicator_vector_train)
        output_dataset_test[:, i] = np.array(indicator_vector_test)
    return input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test

def ANN_gridsearch(X_train, X_test, y_train, y_test):
    """Creates the ANN. In order to find the best settings for the ANN,
    ie those which maiximize the predictivity, a gridsearch is implemented.

    Args:
        X_train (array): part of the input dataset used for training
        X_test (array): part of the input dataset used for testing
        y_train (array): part of the output dataset used for training
        y_test (array): part of the output dataset used for testing
    """
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)
    mlpr = MLPRegressor(max_iter=7000, early_stopping=True)
    net1 = (
        64,
        64,
        64,
        1,
    )
    net2 = (
        64,
        64,
        64,
        2,
    )
    net3 = (
        64,
        64,
        64,
        64,
        1,
    )
    net4 = (
        64,
        64,
        64,
        8,
        1,
    )
    net5 = (64, 32, 16, 8, 4, 2, 1)
    net6 = (128, 128, 128, 1)
    param_list = {
        "hidden_layer_sizes": [net1, net2, net3, net4],
        "activation": ["tanh", "relu"],
        "solver": ["lbfgs", "adam"],
        "learning_rate": ["adaptive"],
    }
    gridCV = GridSearchCV(estimator=mlpr, param_grid=param_list, n_jobs=os.cpu_count() - 2)
    complete_filename_grid = "grid_search.pkl"
    y_train = y_train
    gridCV.fit(X_trainscaled, y_train)
    y_test =y_test
    utils.export_gridsearch(sc_X, gridCV, complete_filename_grid)
    results = sorted(gridCV.cv_results_.keys())
    y_pred = gridCV.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    utils.export_predictions(y_test, y_pred, Q2, "predictions_grid.pkl")   

def ANN_gridsearch_wo_damage(X_train, X_test, y_train, y_test):
    """Equivalent to the ANN_gridsearch() function, but only the 
    elongation input parameter is taken, while damage is removed.
    This function may be removed in the future.

    Args:
        X_train (array): part of the input dataset used for training
        X_test (array): part of the input dataset used for testing
        y_train (array): part of the output dataset used for training
        y_test (array): part of the output dataset used for testing
    """    
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)
    mlpr = MLPRegressor(max_iter=7000, early_stopping=True)
    net1 = (
        64,
        64,
        64,
        1,
    )
    net2 = (
        64,
        64,
        64,
        2,
    )
    net3 = (
        64,
        64,
        64,
        64,
        1,
    )
    net4 = (
        64,
        64,
        64,
        8,
        1,
    )
    net5 = (64, 32, 16, 8, 4, 2, 1)
    net6 = (128, 128, 128, 1)
    param_list = {
        "hidden_layer_sizes": [net1, net2, net3, net4],
        "activation": ["tanh", "relu"],
        "solver": ["sgd"],
        "learning_rate": ["adaptive"],
    }
    gridCV = GridSearchCV(estimator=mlpr, param_grid=param_list, n_jobs=os.cpu_count() - 2)
    complete_filename_grid = "grid_search_wo_damage.pkl"
    y_train = y_train
    gridCV.fit(X_trainscaled, y_train)
    y_test =y_test
    utils.export_gridsearch(sc_X, gridCV, complete_filename_grid)
    results = sorted(gridCV.cv_results_.keys())
    y_pred = gridCV.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    utils.export_predictions(y_test, y_pred, Q2, "predictions_grid_wo_damage.pkl")

def plot_true_vs_predicted(createfigure, savefigure, fonts):
    """Compares the values predicted with the ANN (from the gridsearch)
    using the testing input parameters to the testing output values.

    Args:
        createfigure (class): class used to provide consistent settings for the creation of figures
        savefigure (class):class used to provide consistent settings for saving figures
        fonts (class): class used to provide consistent fonts for the figures
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "predictions_grid.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [y_test, y_pred, Q2] = pickle.load(f)
    labels = ['alpha', 'beta', 'deltaf', 'deltafstar', 'a']
    label_dict = {'alpha': r"$\alpha' [kPa.s^{-1}]$", 'beta': r"$\beta [kPa.s^{-1}]$", 'deltaf':r"$\Delta F$ [kPa]", 'deltafstar':r"$\Delta F^*$ [-]", 'a': 'a [-]'}
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    for i in range(5):
        label = labels[i]
        fig = createfigure.square_figure_7(pixels=180)
        ax = fig.gca()
        color_plot = orange
        ax.plot(y_test[:, i], y_pred[:, i], "o", color=color_plot)
        ax.plot([np.min(y_test[:, i]), np.max(y_test[:, i])], [np.min(y_test[:, i]), np.max(y_test[:, i])], "-k", linewidth=2)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_xticklabels(
    #     [0, 0.2, 0.4, 0.6, 0.8, 1],
    #     font=fonts.serif(),
    #     fontsize=fonts.axis_legend_size(),
    # )
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_yticklabels(
    #     [0, 0.2, 0.4, 0.6, 0.8, 1],
    #     font=fonts.serif(),
    #     fontsize=fonts.axis_legend_size(),
    # )
    # ax.set_xlim((-0.02, 1.02))
    # ax.set_ylim((-0.08, 1.02))
        ax.grid(linestyle="--")
        ax.set_xlabel(r"true values of " + label_dict[label], font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of " + label_dict[label], font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, "results_ANN_" + labels[i])

def plot_true_vs_predicted_wo_damage(createfigure, savefigure, fonts):
    """Equivalent to the plot_true_vs_predicted() function, but only the 
    elongation input parameter is taken, while damage is removed.
    This function may be removed in the future.

    Args:
        createfigure (class): class used to provide consistent settings for the creation of figures
        savefigure (class):class used to provide consistent settings for saving figures
        fonts (class): class used to provide consistent fonts for the figures
    """
    complete_pkl_filename = utils.get_path_to_processed_data() / "predictions_grid_wo_damage.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [y_test, y_pred, Q2] = pickle.load(f)
    labels = ['alpha', 'beta', 'deltaf', 'deltafstar', 'a']
    label_dict = {'alpha': r"$\alpha' [kPa.s^{-1}]$", 'beta': r"$\beta [kPa.s^{-1}]$", 'deltaf':r"$\Delta F$ [kPa]", 'deltafstar':r"$\Delta F^*$ [-]", 'a': 'a [-]'}
    palette = sns.color_palette("Paired")
    orange = palette[-5]
    purple = palette[-3]
    for i in range(5):
        label = labels[i]
        fig = createfigure.square_figure_7(pixels=180)
        ax = fig.gca()
        color_plot = orange
        ax.plot(y_test[:, i], y_pred[:, i], "o", color=color_plot)
        ax.plot([np.min(y_test[:, i]), np.max(y_test[:, i])], [np.min(y_test[:, i]), np.max(y_test[:, i])], "-k", linewidth=2)
    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_xticklabels(
    #     [0, 0.2, 0.4, 0.6, 0.8, 1],
    #     font=fonts.serif(),
    #     fontsize=fonts.axis_legend_size(),
    # )
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_yticklabels(
    #     [0, 0.2, 0.4, 0.6, 0.8, 1],
    #     font=fonts.serif(),
    #     fontsize=fonts.axis_legend_size(),
    # )
    # ax.set_xlim((-0.02, 1.02))
    # ax.set_ylim((-0.08, 1.02))
        ax.grid(linestyle="--")
        ax.set_xlabel(r"true values of " + label_dict[label], font=fonts.serif(), fontsize=fonts.axis_label_size())
        ax.set_ylabel(r"predicted values of " + label_dict[label], font=fonts.serif(), fontsize=fonts.axis_label_size())
        savefigure.save_as_png(fig, "results_ANN_wo_damage" + labels[i])

if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    training_amount = 0.8
    # input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test = define_test_train_datset(training_amount)
    # ANN_gridsearch(input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test)
    # plot_true_vs_predicted(createfigure, savefigure, fonts)
    input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test = define_test_train_datset_wo_damage(training_amount)
    ANN_gridsearch_wo_damage(input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test)
    plot_true_vs_predicted_wo_damage(createfigure, savefigure, fonts)
    print('hello')