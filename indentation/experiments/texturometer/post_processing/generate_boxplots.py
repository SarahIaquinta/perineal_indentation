from matplotlib import pyplot as plt
from math import nan
from indentation.experiments.texturometer.figures.utils import CreateFigure, Fonts, SaveFigure
import pickle
import statistics
import seaborn as sns
import utils
from scipy import stats
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# def compute_p_values():
#     cat1 = df[df['Meatpiece']=='FF']
#     cat2 = df[df['Meatpiece']=='RDG']
#     _, p_value = ttest_ind(cat1['values'], cat2['values'])

label_dict = {"F80":r"$F_{80\%}$ [N]", 
              "F20":r"$F_{20\%}$ [N]",
              "A":r"$A$ [-]", 
              "delta_d":r"$\Delta d$ [mm]",
              "delta_d_star":r"$\Delta d^*$ [-]",
              'beta' : r"$\beta$ [$Ns^{-1}$]",
              'delta_f' : r"$\Delta F$ [$N$]",
              'delta_f_star' : r"$\Delta F^*$ [-]",
              'alpha_time': r"$\alpha $ [$Ns^{-1}$]"                   
              }

y_ticks_dict = {"F80":[40, 60, 80, 100, 120], 
                "F20":[0, 5, 10, 15],
                "A":  [0, 0.25, 0.5, 0.75], 
                "delta_d":[0.8, 1.2, 1.6, 2],
                "delta_d_star":[ 0.25, 0.5, 0.75, 1, 1.25, 1.5],
                'beta' : [-0.7, -0.6, -0.5, -0.4],
                'delta_f' : [0.5, 0.6, 0.7, 0.8],
                'delta_f_star' : [0.5, 0.6, 0.7, 0.8],
                'alpha_time': [0, 0.2, 0.4, 0.6, 0.8, 1]                   
              }

h_dict = {"F80":1, 
                "F20":0.2,
                "A":  0.025, 
                "delta_d":0.1,
                "delta_d_star":0.1,
                'beta' : -0.1,
                'delta_f' : 0.02,
                'delta_f_star' : 0.005,
                'alpha_time': 0.01                   
              }

def color(p_value):
    color = 'r'
    if p_value<0.05:
        color = 'g'
    return color

def is_significative(p_value):
    is_significative = False
    if p_value<0.05:
        is_significative = True
    return is_significative

def plot_boxplot_with_pvalue_texturometer(indicator):
    fig = createfigure.square_figure(pixels=180)
    ax =  fig.gca()
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig, ax = plt.subplots()
    path_to_dataset = utils.get_path_to_processed_data() / "indicators_230407.xlsx"
    tips = pd.read_excel(path_to_dataset, sheet_name='texturo', header=0, names=["Meatpiece", "F20", "F80"], usecols="A:C", decimal=',') 

    # tips = sns.load_dataset(path_to_dataset)
    df = tips.pivot(columns='Meatpiece', values=indicator)
    cat1 = df[tips['Meatpiece']=='FF']
    cat2 = df[tips['Meatpiece']=='RDG']
    _, p_value = ttest_ind(cat1["FF"], cat2["RDG"])

    # df = tips.pivot(columns='Meatpiece', values='F80')
    # data as a list of lists for plotting directly with matplotlib (no nan values allowed)
    # data = [df[c].dropna().tolist() for c in df.columns]
    df.plot(kind='box', positions=range(len(df.columns)), ax=ax)

    # sns.boxplot(x="Meatpiece", y=indicator, data=tips, palette="PRGn")
    # statistical annotation
    if is_significative(p_value):
        x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = max(df['RDG'].max(), df['FF'].max()) + h_dict[indicator], h_dict[indicator], color(p_value)
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y+h, '*', ha='center', va='bottom', color='k', font=fonts.serif(), fontsize=20)
    ax.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sirloin",   "cottage ring"], font=fonts.serif(), fontsize=24)
    y_ticks = y_ticks_dict[indicator]
    # y_ticks_reduced = [int(y_ticks[i]) for i in range(0, len(y_ticks), 2) ]
    ax.set_yticks(y_ticks)

    ax.set_yticklabels(y_ticks, font=fonts.serif(), fontsize=24)

    # plt.close(fig)
    savefigure.save_as_png(fig, "article_boxplot_"+indicator)
    

def plot_boxplot_with_pvalue_laser(indicator):
    fig = createfigure.square_figure(pixels=180)
    ax =  fig.gca()
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig, ax = plt.subplots()
    path_to_dataset = utils.get_path_to_processed_data() / "indicators_230407.xlsx"
    tips = pd.read_excel(path_to_dataset, sheet_name='laser', header=0, names=["Meatpiece", "A", "delta_d", "delta_d_star"], usecols="A:D", decimal=',') 

    # tips = sns.load_dataset(path_to_dataset)
    df = tips.pivot(columns='Meatpiece', values=indicator)
    cat1 = df[tips['Meatpiece']=='FF']
    cat2 = df[tips['Meatpiece']=='RDG']
    _, p_value = ttest_ind(cat1["FF"], cat2["RDG"])

    # df = tips.pivot(columns='Meatpiece', values='F80')
    # data as a list of lists for plotting directly with matplotlib (no nan values allowed)
    # data = [df[c].dropna().tolist() for c in df.columns]
    df.plot(kind='box', positions=range(len(df.columns)), ax=ax)

    # sns.boxplot(x="Meatpiece", y=indicator, data=tips, palette="PRGn")
    # statistical annotation
    if is_significative(p_value):
        x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = max(df['RDG'].max(), df['FF'].max()) + h_dict[indicator], h_dict[indicator], color(p_value)
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y+h, '*', ha='center', va='bottom', color='k', font=fonts.serif(), fontsize=20)
    ax.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sirloin",   "cottage ring"], font=fonts.serif(), fontsize=24)
    y_ticks = y_ticks_dict[indicator]
    # y_ticks_reduced = [np.round(y, 1) for y in y_ticks]
    ax.set_yticks(y_ticks)

    ax.set_yticklabels(y_ticks, font=fonts.serif(), fontsize=24)

    # plt.close(fig)
    savefigure.save_as_png(fig, "article_boxplot_"+indicator)
    
def plot_boxplot_with_pvalue_zwick(indicator):
    fig = createfigure.square_figure(pixels=180)
    ax =  fig.gca()
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig, ax = plt.subplots()
    path_to_dataset = utils.get_path_to_processed_data() / "indicators_230407.xlsx"
    tips = pd.read_excel(path_to_dataset, sheet_name='zwick', header=0, names=["Meatpiece", "delta_f", "delta_f_star", "alpha_time", "beta"], usecols="A:E", decimal=',') 

    # tips = sns.load_dataset(path_to_dataset)
    df = tips.pivot(columns='Meatpiece', values=indicator)
    cat1 = df[tips['Meatpiece']=='FF']
    cat2 = df[tips['Meatpiece']=='RDG']
    _, p_value = ttest_ind(cat1["FF"], cat2["RDG"])

    # df = tips.pivot(columns='Meatpiece', values='F80')
    # data as a list of lists for plotting directly with matplotlib (no nan values allowed)
    # data = [df[c].dropna().tolist() for c in df.columns]
    df.plot(kind='box', positions=range(len(df.columns)), ax=ax)

    # sns.boxplot(x="Meatpiece", y=indicator, data=tips, palette="PRGn")
    # statistical annotation
    if is_significative(p_value):
        x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
        y, h, col = max(df['RDG'].max(), df['FF'].max()) + h_dict[indicator], h_dict[indicator], color(p_value)
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        ax.text((x1+x2)*.5, y+h, '*', ha='center', va='bottom', color='k', font=fonts.serif(), fontsize=20)
    ax.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sirloin",   "cottage ring"], font=fonts.serif(), fontsize=24)
    y_ticks = y_ticks_dict[indicator]
    # y_ticks_reduced = [np.round(y, 1) for y in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, font=fonts.serif(), fontsize=24)

    # plt.close(fig)
    savefigure.save_as_png(fig, "article_boxplot_"+indicator)
    

    




if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    plot_boxplot_with_pvalue_texturometer("F20")    
    plot_boxplot_with_pvalue_texturometer("F80")
    plot_boxplot_with_pvalue_laser("A")
    plot_boxplot_with_pvalue_laser("delta_d")
    plot_boxplot_with_pvalue_laser("delta_d_star")
    plot_boxplot_with_pvalue_zwick("delta_f")
    plot_boxplot_with_pvalue_zwick("delta_f_star")
    plot_boxplot_with_pvalue_zwick("alpha_time")
    plot_boxplot_with_pvalue_zwick("beta")

    print('hello')