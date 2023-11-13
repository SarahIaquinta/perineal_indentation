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

label_dict = {"F80":r"$F_{80\%}$ [N]", 
              "F20":r"$F_{20\%}$ [N]",
              "A":r"$A$ [-]", 
              "delta_d":r"$\Delta d$ [mm]",
              "delta_d_star":r"$\Delta d^*$ [-]",
              'beta' : r"$\beta$ [$Ns^{-1}$]",
              'delta_f' : r"$\Delta F$ [$N$]",
              'delta_f_star' : r"$\Delta F^*$ [-]",
              'alpha_time': r"$\alpha $ [$Ns^{-1}$]",
              'def':r"$\Delta U / e$"                   
              }

y_ticks_dict = {"F80":[20, 40, 60, 80, 100, 120], 
                "F20":[0, 5, 10],
                "A":  [0.2, 0.4, 0.6], 
                "delta_d":[0.4, 0.8, 1.2, 1.6],
                "delta_d_star":[ 0, 0.2, 0.4, 0.6, 0.8],
                'beta' : [-0.7, -0.6, -0.5],
                'delta_f' : [0.5, 0.6, 0.7, 0.8],
                'delta_f_star' : [0.55, 0.6, 0.65, 0.7],
                'alpha_time': [0.1, 0.3, 0.5, 0.7],
                'def':[0.2,  0.3,0.4 ]                   
              }




def plot_indicator_vs_texturometer_forces_texturo(indicator):
    fig_F20 = createfigure.square_figure(pixels=180)
    ax_F20 =  fig_F20.gca()
    fig_F80 = createfigure.square_figure(pixels=180)
    ax_F80 =  fig_F80.gca()
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig, ax = plt.subplots()
    path_to_dataset = utils.get_path_to_processed_data() / "indicators_230407.xlsx"
    tips = pd.read_excel(path_to_dataset, sheet_name='texturo', header=0, names=["Meatpiece", "F20", "F80"], usecols="A:C", decimal=',') 

    tips_texturo = pd.read_excel(path_to_dataset, sheet_name='texturo', header=0, names=["Meatpiece", "F20", "F80"], usecols="A:C", decimal=',') 


    # tips = sns.load_dataset(path_to_dataset)
    df_F20 = tips.pivot(columns='Meatpiece', values="F20")
    FF_F20 = df_F20[tips['Meatpiece']=='FF']
    RDG_F20 = df_F20[tips['Meatpiece']=='RDG']
    mean_FF_F20 = np.mean(FF_F20)
    mean_RDG_F20 = np.mean(RDG_F20)
    std_FF_F20 = FF_F20.std()['FF']
    std_RDG_F20 = RDG_F20.std()['RDG']
    
    df_F80 = tips.pivot(columns='Meatpiece', values="F80")
    FF_F80 = df_F80[tips['Meatpiece']=='FF']
    RDG_F80 = df_F80[tips['Meatpiece']=='RDG']
    mean_FF_F80 = np.mean(FF_F80)
    mean_RDG_F80 = np.mean(RDG_F80)  
    std_FF_F80 = FF_F80.std()['FF']
    std_RDG_F80 = RDG_F80.std()['RDG']
    
    df_indicator = tips_texturo.pivot(columns='Meatpiece', values=indicator)
    FF_indicator = df_indicator[tips['Meatpiece']=='FF']
    RDG_indicator = df_indicator[tips['Meatpiece']=='RDG']
    mean_FF_indicator = np.mean(FF_indicator)
    mean_RDG_indicator = np.mean(RDG_indicator)  
    std_FF_indicator = FF_indicator.std()['FF']
    std_RDG_indicator = RDG_indicator.std()['RDG']   
    
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    # kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[3]}
    # kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[1]}
    kwargs_FF = {'marker':'o', 'mfc':'r', 'elinewidth':3, 'ecolor':'r', 'alpha':0.8, 'ms':'20', 'mec':'r'}
    kwargs_RDG = {'marker':'^', 'mfc':'g', 'elinewidth':3, 'ecolor':'g', 'alpha':0.8, 'ms':'20', 'mec':'g'}


    ax_F20.errorbar([mean_FF_F20], [mean_FF_indicator], yerr=std_FF_indicator, xerr=std_FF_F20 ,lw=0, label='sirloin', **kwargs_FF)
    ax_F20.errorbar([mean_RDG_F20], [mean_RDG_indicator], yerr=std_RDG_indicator, xerr=std_RDG_F20 ,lw=0, label='round', **kwargs_RDG)

    ax_F80.errorbar([mean_FF_F80], [mean_FF_indicator], yerr=std_FF_indicator, xerr=std_FF_F80 ,lw=0, label='sirloin', **kwargs_FF)
    ax_F80.errorbar([mean_RDG_F80], [mean_RDG_indicator], yerr=std_RDG_indicator, xerr=std_RDG_F80 ,lw=0, label='round', **kwargs_RDG)
  
    ax_F20.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_F20.set_xticks(y_ticks_dict['F20'])
    ax_F20.set_xticklabels(y_ticks_dict['F20'], font=fonts.serif(), fontsize=24)
    ax_F20.set_yticks(y_ticks_dict[indicator])
    ax_F20.set_yticklabels(y_ticks_dict[indicator], font=fonts.serif(), fontsize=24)
    ax_F20.set_xlabel(r'$F_{20 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_F20.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_F20, indicator + "vs_F20_230407")
    plt.close(fig_F20)
    
    ax_F80.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_F80.set_xticks(y_ticks_dict['F80'])
    ax_F80.set_xticklabels(y_ticks_dict['F80'], font=fonts.serif(), fontsize=24)
    ax_F80.set_yticks(y_ticks_dict[indicator])
    ax_F80.set_yticklabels(y_ticks_dict[indicator], font=fonts.serif(), fontsize=24)
    ax_F80.set_xlabel(r'$F_{80 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_F80.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_F80, indicator + "vs_F80_230407")
    plt.close(fig_F80)


def plot_indicator_vs_texturometer_forces_laser(indicator):
    fig_F20 = createfigure.square_figure(pixels=180)
    ax_F20 =  fig_F20.gca()
    fig_F80 = createfigure.square_figure(pixels=180)
    ax_F80 =  fig_F80.gca()
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig, ax = plt.subplots()
    path_to_dataset = utils.get_path_to_processed_data() / "indicators_230407.xlsx"
    tips = pd.read_excel(path_to_dataset, sheet_name='texturo', header=0, names=["Meatpiece", "F20", "F80"], usecols="A:C", decimal=',') 

    tips_laser = pd.read_excel(path_to_dataset, sheet_name='laser', header=0, names=["Meatpiece", "A", "delta_d", "delta_d_star"], usecols="A:D", decimal=',') 

    # tips = sns.load_dataset(path_to_dataset)
    df_F20 = tips.pivot(columns='Meatpiece', values="F20")
    FF_F20 = df_F20[tips['Meatpiece']=='FF']
    RDG_F20 = df_F20[tips['Meatpiece']=='RDG']
    mean_FF_F20 = np.mean(FF_F20)
    mean_RDG_F20 = np.mean(RDG_F20)
    std_FF_F20 = FF_F20.std()['FF']
    std_RDG_F20 = RDG_F20.std()['RDG']
    
    df_F80 = tips.pivot(columns='Meatpiece', values="F80")
    FF_F80 = df_F80[tips['Meatpiece']=='FF']
    RDG_F80 = df_F80[tips['Meatpiece']=='RDG']
    mean_FF_F80 = np.mean(FF_F80)
    mean_RDG_F80 = np.mean(RDG_F80)  
    std_FF_F80 = FF_F80.std()['FF']
    std_RDG_F80 = RDG_F80.std()['RDG']
    
    df_indicator = tips_laser.pivot(columns='Meatpiece', values=indicator)
    FF_indicator = df_indicator[tips['Meatpiece']=='FF']
    RDG_indicator = df_indicator[tips['Meatpiece']=='RDG']
    mean_FF_indicator = np.mean(FF_indicator)
    mean_RDG_indicator = np.mean(RDG_indicator)  
    std_FF_indicator = FF_indicator.std()['FF']
    std_RDG_indicator = RDG_indicator.std()['RDG']   
        
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    # kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[3]}
    # kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[1]}
    kwargs_FF = {'marker':'o', 'mfc':'r', 'elinewidth':3, 'ecolor':'r', 'alpha':0.8, 'ms':'20', 'mec':'r'}
    kwargs_RDG = {'marker':'^', 'mfc':'g', 'elinewidth':3, 'ecolor':'g', 'alpha':0.8, 'ms':'20', 'mec':'g'}


    ax_F20.errorbar([mean_FF_F20], [mean_FF_indicator], yerr=std_FF_indicator, xerr=std_FF_F20 ,lw=0, label='sirloin', **kwargs_FF)
    ax_F20.errorbar([mean_RDG_F20], [mean_RDG_indicator], yerr=std_RDG_indicator, xerr=std_RDG_F20 ,lw=0, label='round', **kwargs_RDG)

    ax_F80.errorbar([mean_FF_F80], [mean_FF_indicator], yerr=std_FF_indicator, xerr=std_FF_F80 ,lw=0, label='sirloin', **kwargs_FF)
    ax_F80.errorbar([mean_RDG_F80], [mean_RDG_indicator], yerr=std_RDG_indicator, xerr=std_RDG_F80 ,lw=0, label='round', **kwargs_RDG)
  
    ax_F20.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_F20.set_xticks(y_ticks_dict['F20'])
    ax_F20.set_xticklabels(y_ticks_dict['F20'], font=fonts.serif(), fontsize=24)
    ax_F20.set_yticks(y_ticks_dict[indicator])
    ax_F20.set_yticklabels(y_ticks_dict[indicator], font=fonts.serif(), fontsize=24)
    ax_F20.set_xlabel(r'$F_{20 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_F20.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_F20, indicator + "vs_F20_230407")
    plt.close(fig_F20)
    
    ax_F80.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_F80.set_xticks(y_ticks_dict['F80'])
    ax_F80.set_xticklabels(y_ticks_dict['F80'], font=fonts.serif(), fontsize=24)
    ax_F80.set_yticks(y_ticks_dict[indicator])
    ax_F80.set_yticklabels(y_ticks_dict[indicator], font=fonts.serif(), fontsize=24)
    ax_F80.set_xlabel(r'$F_{80 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_F80.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_F80, indicator + "vs_F80_230407")
    plt.close(fig_F80)

def plot_indicator_vs_texturometer_forces_zwick(indicator):
    fig_F20 = createfigure.square_figure(pixels=180)
    ax_F20 =  fig_F20.gca()
    fig_F80 = createfigure.square_figure(pixels=180)
    ax_F80 =  fig_F80.gca()
    # fig = plt.gcf()
    # ax = plt.gca()
    # fig, ax = plt.subplots()
    path_to_dataset = utils.get_path_to_processed_data() / "indicators_230407.xlsx"
    tips = pd.read_excel(path_to_dataset, sheet_name='texturo', header=0, names=["Meatpiece", "F20", "F80"], usecols="A:C", decimal=',') 

    tips_zwick = pd.read_excel(path_to_dataset, sheet_name='zwick', header=0, names=["Meatpiece", "delta_f", "delta_f_star", "alpha_time", "beta", "Umax", "def"], usecols="A:G", decimal=',') 


    # tips = sns.load_dataset(path_to_dataset)
    df_F20 = tips.pivot(columns='Meatpiece', values="F20")
    FF_F20 = df_F20[tips['Meatpiece']=='FF']
    RDG_F20 = df_F20[tips['Meatpiece']=='RDG']
    mean_FF_F20 = np.mean(FF_F20)
    mean_RDG_F20 = np.mean(RDG_F20)
    std_FF_F20 = FF_F20.std()['FF']
    std_RDG_F20 = RDG_F20.std()['RDG']
    
    df_F80 = tips.pivot(columns='Meatpiece', values="F80")
    FF_F80 = df_F80[tips['Meatpiece']=='FF']
    RDG_F80 = df_F80[tips['Meatpiece']=='RDG']
    mean_FF_F80 = np.mean(FF_F80)
    mean_RDG_F80 = np.mean(RDG_F80)  
    std_FF_F80 = FF_F80.std()['FF']
    std_RDG_F80 = RDG_F80.std()['RDG']
    
    df_indicator = tips_zwick.pivot(columns='Meatpiece', values=indicator)
    FF_indicator = df_indicator[tips['Meatpiece']=='FF']
    RDG_indicator = df_indicator[tips['Meatpiece']=='RDG']
    mean_FF_indicator = np.mean(FF_indicator)
    mean_RDG_indicator = np.mean(RDG_indicator)  
    std_FF_indicator = FF_indicator.std()['FF']
    std_RDG_indicator = RDG_indicator.std()['RDG']   
        
    color = sns.color_palette("Paired")
    color_rocket = sns.color_palette("rocket")
    # kwargs_FF = {'marker':'o', 'mfc':color_rocket[3], 'elinewidth':3, 'ecolor':color_rocket[3], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[3]}
    # kwargs_RDG = {'marker':'^', 'mfc':color_rocket[1], 'elinewidth':3, 'ecolor':color_rocket[1], 'alpha':0.8, 'ms':'20', 'mec':color_rocket[1]}
    kwargs_FF = {'marker':'o', 'mfc':'r', 'elinewidth':3, 'ecolor':'r', 'alpha':0.8, 'ms':'20', 'mec':'r'}
    kwargs_RDG = {'marker':'^', 'mfc':'g', 'elinewidth':3, 'ecolor':'g', 'alpha':0.8, 'ms':'20', 'mec':'g'}


    ax_F20.errorbar([mean_FF_F20], [mean_FF_indicator], yerr=std_FF_indicator, xerr=std_FF_F20 ,lw=0, label='sirloin', **kwargs_FF)
    ax_F20.errorbar([mean_RDG_F20], [mean_RDG_indicator], yerr=std_RDG_indicator, xerr=std_RDG_F20 ,lw=0, label='round', **kwargs_RDG)

    ax_F80.errorbar([mean_FF_F80], [mean_FF_indicator], yerr=std_FF_indicator, xerr=std_FF_F80 ,lw=0, label='sirloin', **kwargs_FF)
    ax_F80.errorbar([mean_RDG_F80], [mean_RDG_indicator], yerr=std_RDG_indicator, xerr=std_RDG_F80 ,lw=0, label='round', **kwargs_RDG)
  
    ax_F20.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_F20.set_xticks(y_ticks_dict['F20'])
    ax_F20.set_xticklabels(y_ticks_dict['F20'], font=fonts.serif(), fontsize=24)
    ax_F20.set_yticks(y_ticks_dict[indicator])
    ax_F20.set_yticklabels(y_ticks_dict[indicator], font=fonts.serif(), fontsize=24)
    ax_F20.set_xlabel(r'$F_{20 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_F20.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_F20, indicator + "vs_F20_230407")
    plt.close(fig_F20)
    
    ax_F80.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_F80.set_xticks(y_ticks_dict['F80'])
    ax_F80.set_xticklabels(y_ticks_dict['F80'], font=fonts.serif(), fontsize=24)
    ax_F80.set_yticks(y_ticks_dict[indicator])
    ax_F80.set_yticklabels(y_ticks_dict[indicator], font=fonts.serif(), fontsize=24)
    ax_F80.set_xlabel(r'$F_{80 \%}$ [N]', font=fonts.serif(), fontsize=26)
    ax_F80.set_ylabel(label_dict[indicator], font=fonts.serif(), fontsize=26)

    savefigure.save_as_png(fig_F80, indicator + "vs_F80_230407")
    plt.close(fig_F80)
    
if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    plot_indicator_vs_texturometer_forces_texturo("F20")    
    plot_indicator_vs_texturometer_forces_texturo("F80")
    plot_indicator_vs_texturometer_forces_laser("A")
    plot_indicator_vs_texturometer_forces_laser("delta_d")
    plot_indicator_vs_texturometer_forces_laser("delta_d_star")
    plot_indicator_vs_texturometer_forces_zwick("delta_f")
    plot_indicator_vs_texturometer_forces_zwick("delta_f_star")
    plot_indicator_vs_texturometer_forces_zwick("alpha_time")
    plot_indicator_vs_texturometer_forces_zwick("beta")
    plot_indicator_vs_texturometer_forces_zwick("def")

    print('hello')
    
    