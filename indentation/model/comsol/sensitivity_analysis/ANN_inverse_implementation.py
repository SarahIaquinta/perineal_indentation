import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import seaborn as sns
from indentation.model.comsol.sensitivity_analysis.figures.utils import CreateFigure, Fonts, SaveFigure

ot.Log.Show(ot.Log.NONE)
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_evaluation import plot
import utils
import pickle

def define_test_train_datset_inverse_model(training_amount):
    complete_pkl_filename = utils.get_path_to_processed_data() / "indicators.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict] = pickle.load(f)
    indicators_dicts = [alpha_p_dict, beta_dict, delta_f_dict, delta_f_star_dict, a_dict]
    complete_pkl_filename_inputs = utils.get_path_to_processed_data() / "inputs.pkl"
    with open(complete_pkl_filename_inputs, "rb") as f:
        [ids_list, elongation_dict, damage_dict] = pickle.load(f)
    len_train = int(training_amount*len(indicators_dicts[0].values()))
    len_test = int((1-training_amount)*len(indicators_dicts[0].values()))
    output_dataset_train = np.zeros((len_train, 1))
    output_dataset_test = np.zeros((len_test, 1))
    input_dataset_train = np.zeros((len_train, 5))
    input_dataset_test = np.zeros((len_test, 5))
    output_dataset_train = np.array(list(elongation_dict.values())[:int(training_amount*len(elongation_dict.values()))]).ravel()
    output_dataset_test = np.array(list(elongation_dict.values())[int(training_amount*len(elongation_dict.values()))+1:]).ravel()
    for i in range(len(indicators_dicts)):
        indicator_dict = indicators_dicts[i]
        indicator_vector = list(indicator_dict.values())
        indicator_vector_train = indicator_vector[:int(training_amount*len(indicator_dict.values()))]
        indicator_vector_test = indicator_vector[int(training_amount*len(indicator_dict.values()))+1:]
        input_dataset_train[:, i] = np.array(indicator_vector_train)
        input_dataset_test[:, i] = np.array(indicator_vector_test)
    return input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test



def ANN_gridsearch_inverse_model(X_train, X_test, y_train, y_test):
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
        "solver": ["lbfgs"],
        "learning_rate": ["adaptive"],
    }
    gridCV = GridSearchCV(estimator=mlpr, param_grid=param_list, n_jobs=os.cpu_count() - 2)
    complete_filename_grid = "grid_search_inverse.pkl"
    y_train = y_train
    gridCV.fit(X_trainscaled, y_train)
    y_test =y_test
    utils.export_gridsearch(sc_X, gridCV, complete_filename_grid)
    results = sorted(gridCV.cv_results_.keys())
    y_pred = gridCV.predict(X_testscaled)
    Q2 = r2_score(y_pred, y_test)
    print("The Score with ", Q2)
    utils.export_predictions(y_test, y_pred, Q2, "predictions_grid_inverse.pkl")
    


def plot_true_vs_predicted_inverse_model(createfigure, savefigure, fonts):
    complete_pkl_filename = utils.get_path_to_processed_data() / "predictions_grid_inverse.pkl"
    with open(complete_pkl_filename, "rb") as f:
        [y_test, y_pred, Q2] = pickle.load(f)
    palette = sns.color_palette("Paired")
    purple = palette[-3]
    fig = createfigure.square_figure_7(pixels=180)
    ax = fig.gca()
    color_plot = purple
    ax.plot(y_test, y_pred, "o", color=color_plot)
    ax.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], "-k", linewidth=2)
    ax.grid(linestyle="--")
    ax.set_xlabel(r"true values of $\lambda$" , font=fonts.serif(), fontsize=fonts.axis_label_size())
    ax.set_ylabel(r"predicted values of $\lambda$" , font=fonts.serif(), fontsize=fonts.axis_label_size())
    savefigure.save_as_png(fig, "results_inverse_ANN")



if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    training_amount = 0.8
    input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test = define_test_train_datset_inverse_model(training_amount)
    # ANN_gridsearch_inverse_model(input_dataset_train, input_dataset_test, output_dataset_train, output_dataset_test)
    plot_true_vs_predicted_inverse_model(createfigure, savefigure, fonts)
    print('hello')