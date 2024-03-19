import glob
from indentation import make_data_folder
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import indentation.caracterization.tiguida.tensile_testings_porks.post_processing.utils as tiguida_postpro_utils
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit, minimize, rosen, rosen_der
from matplotlib.widgets import Slider, Button
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure

def compute_hyperelastic_stress_yeoh(elongation_x, c1, c2, c3):
    stress = 2*(elongation_x - 1/elongation_x**2)*(c1 + (elongation_x**2 + 2/elongation_x - 3)*(2*c2 + 3*c3*(elongation_x**2 + 2/elongation_x - 3)))
    return stress

def compute_stress_list(datafile, params):
    _, _, _, _, _, _, undamaged_stress, undamaged_elongation_x = tiguida_postpro_utils.get_data_from_datafile(datafile)
    [c1, c2, c3] = params
    undamage_stress_model_list = np.zeros_like(undamaged_stress)
    for i in range(len(undamage_stress_model_list)):
        elongation_x = undamaged_elongation_x[i]
        stress = (1/elongation_x)*compute_hyperelastic_stress_yeoh(elongation_x, c1, c2, c3)
        undamage_stress_model_list[i] = stress
    return undamage_stress_model_list
    
    
    
def find_optimal_parameters(datafile, minimization_method):
  _, _, _, _, _, _, undamaged_stress_exp, _ = tiguida_postpro_utils.get_data_from_datafile(datafile)
  # c1_init, c2_init, c3_init, beta_init, tau_init, a_init, eta_init, alpha_init = 10, 5, 0.5, 0.07, 1,  1, 1, 0.5
  c1_init, c2_init, c3_init = 50, 100, 30
  initial_guess_values = [c1_init, c2_init, c3_init]
  # bounds_values = [(0.1, 30), (0.1, 30), (0.1, 10), (0.01, 50), (1, 100), (0.01, 10), (1, 5), (0.1, 5)]
  bounds_values = [(1, 100), (1, 200), (1, 100)]
  # beta eta and alpha have to be positive
  def minimization_function(params):
      stress_list_model = compute_stress_list(datafile, params)
      n = len(stress_list_model)
      stress_list_exp = undamaged_stress_exp
      least_square = mean_squared_error(stress_list_exp[0:int(n/2)], stress_list_model[0:int(n/2)])
      return least_square

  res = minimize(minimization_function, initial_guess_values, method=minimization_method, bounds=bounds_values,
            options={'disp': False})
  params_opti = res.x
  [c1, c2, c3] = params_opti
  return [c1, c2, c3]

from typing_extensions import ParamSpec
def make_adaptive_plot_stress_vs_time(datafile):
  pig_number, tissue, total_time, total_stress, total_stretch, undamaged_time, undamaged_stress, undamaged_stretch = tiguida_postpro_utils.get_data_from_datafile(datafile)
  # c1_init, c2_init, c3_init, c4_init, eta_0_init, eta_init, alpha_init = 4, 6.2, 0.3, 0.07, 1,  1, 0.5
  
  params_init = find_optimal_parameters(datafile, 'TNC')

  [c1_init, c2_init, c3_init] = params_init
  fig, ax = plt.subplots()
  line, = ax.plot(undamaged_stretch, compute_stress_list(datafile, [c1_init, c2_init, c3_init]), '-b', lw=2)
  fig.subplots_adjust(left=0.4, bottom=0.25)
  ax.plot(undamaged_stretch, undamaged_stress, ':k', alpha=0.8)
  ax.plot(total_stretch, total_stress, ':k', alpha=0.7)
  ax.set_xlabel('elongation [-]')
  ax.set_ylabel('Pi stress [kPa]')
#   ax.set_ylim((0, 100))
#   ax.set_xlim((0, 200))


  # Make a horizontal slider to control c1.
  axc1 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
  c1_slider = Slider(
      ax=axc1,
      label='c1',
      valmin=0,
      valmax=1000,
      valinit=c1_init,
      color='y'
  )
  
  axc2 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
  c2_slider = Slider(
      ax=axc2,
      label='c2',
      valmin=0,
      valmax=1000,
      valinit=c2_init,
      color='y'
  )
  
  axc3 = fig.add_axes([0.25, 0.15, 0.65, 0.03])
  c3_slider = Slider(
      ax=axc3,
      label='c3',
      valmin=0,
      valmax=1000,
      valinit=c3_init,
      color='y'
  )

  def compute_adaptive_stress(c1, c2, c3):
  # def compute_adaptive_stress(c1, c2, c3, c4, eta_0, eta, alpha):
    # params = [[c1, c2, c3, c4, eta_0, eta, alpha]]
    params = [c1, c2, c3]
    stress_list = compute_stress_list(datafile, params)
    return stress_list

  # The function to be called anytime a slider's value changes
  def update(val):
      line.set_ydata(compute_adaptive_stress(c1_slider.val, c2_slider.val, c3_slider.val))
      fig.canvas.draw_idle()
      
  c1_slider.on_changed(update)
  c2_slider.on_changed(update)
  c3_slider.on_changed(update)
  
  resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
  button = Button(resetax, 'Reset', hovercolor='0.975')
  
  def reset(event):
    c1_slider.reset()
    c2_slider.reset()
    c3_slider.reset()
  button.on_clicked(reset)

  plt.show()



def plot_comparison_stress_model_experiment(datafile, minimization_method, suffix, export_id):
  createfigure = CreateFigure()
  fonts = Fonts()
  savefigure = SaveFigure()
  pig_number, tissue, total_time, total_stress, total_elongation, undamaged_time, undamaged_stress, undamaged_elongation_x = tiguida_postpro_utils.get_data_from_datafile(datafile)
  params_opti = find_optimal_parameters(datafile, minimization_method)
  large_tension_utils.export_optimization_params_as_pkl(export_id, "uniaxial", params_opti, minimization_method, suffix)
  stress_list_model = compute_stress_list(datafile, params_opti)
  fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
  ax_stress_vs_elongation.plot(total_elongation, total_stress, '-k', lw=1, label='exp')
  ax_stress_vs_elongation.plot(undamaged_elongation_x, stress_list_model, '-r', lw=1, label='model')
  ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
  savefigure.save_as_png(fig_stress_vs_elongation, export_id + "stress_elong_exp_uniaxial_" + minimization_method + "_" + suffix)
  plt.close(fig_stress_vs_elongation)
  
def plot_comparison_stress_model_experiment_given_params(datafile, minimization_method, suffix, export_id, params_opti):
  createfigure = CreateFigure()
  fonts = Fonts()
  savefigure = SaveFigure()
  pig_number, tissue, total_time, total_stress, total_elongation, undamaged_time, undamaged_stress, undamaged_elongation_x = tiguida_postpro_utils.get_data_from_datafile(datafile)
  large_tension_utils.export_optimization_params_as_pkl(export_id, "uniaxial", params_opti, minimization_method, suffix)
  stress_list_model = compute_stress_list(datafile, params_opti)
  fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
  ax_stress_vs_elongation.plot(total_elongation, total_stress, '-k', lw=1, label='exp')
  ax_stress_vs_elongation.plot(undamaged_elongation_x, stress_list_model, '-r', lw=1, label='model')
  ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
  savefigure.save_as_png(fig_stress_vs_elongation, export_id + "stress_elong_exp_uniaxial_" + minimization_method + "_" + suffix)
  plt.close(fig_stress_vs_elongation)

def plot_damaged_vs_undamaged_article(datafile_list):
    suffix_list_article = ["05p", "07p", "08p"]
    color_dict = {"05p": 'limegreen', "07p":'cornflowerblue', '08p': 'deeppink'}
    sample_id_dict = {"05p":"1", "07p":"2", "08p":"3"}
    # manual_params_dict = {"04p":[83, 431, 179], "05p":[40, 204, 75], "07p":[46, 202, 36], "08p":[48, 171, 76], "09p":[35, 342, 150]}
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    for datafile in datafile_list:
        tissue = dict_tissue[datafile]
        pig = dict_pigs[datafile]
        export_id = pig + tissue
        if export_id in suffix_list_article:
          pig_number, tissue, total_time, total_stress, total_elongation, undamaged_time, undamaged_stress, undamaged_elongation_x = tiguida_postpro_utils.get_data_from_datafile(datafile)
          ax_stress_vs_elongation.plot(total_elongation, total_stress, ":", lw=2, color=color_dict[export_id], alpha=0.8)
          ax_stress_vs_elongation.plot(undamaged_elongation_x, undamaged_stress, "-", lw=3, color=color_dict[export_id], label=sample_id_dict[export_id])
          ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    savefigure.save_as_svg(fig_stress_vs_elongation, export_id + "raw_data_stress_elong_exp_uniaxial_undamaged_vs_complete_article")       
    savefigure.save_as_png(fig_stress_vs_elongation, export_id + "raw_data_stress_elong_exp_uniaxial_undamaged_vs_complete_article")       
          

def plot_fitted_hyperelastic_undamaged_article(datafile_list):
    suffix_list_article = ["05p", "07p", "08p"]
    color_dict = {"05p": 'limegreen', "07p":'cornflowerblue', '08p': 'deeppink'}
    markerdict_dict = {"05p": '^', "07p":'o', '08p': 'D'}
    sample_id_dict = {"05p":"1", "07p":"2", "08p":"3"}
    manual_params_dict = {"04p":[83, 431, 179], "05p":[40, 204, 75], "07p":[46, 202, 36], "08p":[48, 171, 76], "09p":[35, 342, 150]}
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
    ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
    for datafile in datafile_list:
        tissue = dict_tissue[datafile]
        pig = dict_pigs[datafile]
        export_id = pig + tissue
        if export_id in suffix_list_article:
          pig_number, tissue, total_time, total_stress, total_elongation, undamaged_time, undamaged_stress, undamaged_elongation_x = tiguida_postpro_utils.get_data_from_datafile(datafile)
          stress_list_model = compute_stress_list(datafile, manual_params_dict[export_id])
          ax_stress_vs_elongation.plot(undamaged_elongation_x, stress_list_model, "-", lw=2, color=color_dict[export_id])
          ax_stress_vs_elongation.plot(undamaged_elongation_x[0:len(undamaged_elongation_x):300], undamaged_stress[0:len(undamaged_elongation_x):300], markerdict_dict[export_id], fillstyle='none', lw=0, color=color_dict[export_id], label=sample_id_dict[export_id])
          ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
    ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
    ax_stress_vs_elongation.set_ylabel(r"$\Pi_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
    savefigure.save_as_svg(fig_stress_vs_elongation, export_id + "fituniaxial_stress_elong_exp_uniaxial_undamaged_article")       
    savefigure.save_as_png(fig_stress_vs_elongation, export_id + "fituniaxial_stress_elong_exp_uniaxial_undamaged_article")       
  

        



if __name__ == "__main__":
    dict_pigs, dict_tissue, dict_datas_total_time, dict_datas_total_stress, dict_datas_total_stretch, dict_datas_undamaged_time, dict_datas_undamaged_stress, dict_datas_undamaged_stretch =  tiguida_postpro_utils.extract_exp_data_as_pkl()
    minimization_method = 'Powell'
    datafile_list = tiguida_postpro_utils.get_exp_datafile_list()
    suffix = "PK1"
    manual_params_dict = {"04p":[83, 431, 179], "05p":[40, 204, 75], "07p":[46, 202, 36], "08p":[48, 171, 76], "09p":[35, 342, 150]}
    # plot_damaged_vs_undamaged_article(datafile_list)
    plot_fitted_hyperelastic_undamaged_article(datafile_list)
    # for datafile in datafile_list:
    #     tissue = dict_tissue[datafile]
    #     pig = dict_pigs[datafile]
    #     export_id = pig + tissue
    #     print(tissue ,  dict_pigs[datafile])
    #     if tissue == 'p':
    #       # if dict_pigs[datafile] == '04':
    #           print('pig ', dict_pigs[datafile], ' tissue ', tissue)
    #           # make_adaptive_plot_stress_vs_time(datafile)
    #           # plot_comparison_stress_model_experiment(datafile, minimization_method, suffix, export_id)
    #           # params_opti = large_tension_utils.extract_optimization_params_from_pkl(export_id, "uniaxial", minimization_method, suffix)
    #           plot_comparison_stress_model_experiment_given_params(datafile, minimization_method, suffix+"_manual", export_id, manual_params_dict[export_id])
    #           # print('pig ', dict_pigs[datafile], ' tissue ', tissue, ' params c1 c2 c3 = ', params_opti)
    # suffix_list_article = ["05p", "07p", "08p"]
        
        
        
    
