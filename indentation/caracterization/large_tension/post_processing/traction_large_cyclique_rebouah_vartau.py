# -*- coding: utf-8 -*-
"""traction_large_cyclique_rebouah.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a0rFThpTnuSuM8pItWBElTJYO9v1qKtT

Import des bibliothèques
"""
import time
import numpy as np
import pandas as pd
from scipy.signal import lfilter, savgol_filter
import matplotlib.pyplot as plt
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick
import indentation.caracterization.large_tension.post_processing.utils as large_tension_utils
from indentation.caracterization.large_tension.figures.utils import CreateFigure, Fonts, SaveFigure
from scipy import integrate
import indentation.caracterization.large_tension.post_processing.fit_experimental_data_continuous_parameters as fit_cont
import pickle
from matplotlib.widgets import Slider, Button
import numba as nb
"""Récupération des données expérimentales"""





def read_sheet_in_datafile(datafile, sheet):
    """
    Extracts the measured time, force and displacement values in a sheet

    Parameters:
        ----------
        datafile: string
            name of the datafile to be read
        sheet: string
            name of the sheet to be read

    Returns:
        -------
        time: pandasArray
            list of the time values (in seconds) in the sheet of the datafile
        force: pandasArray
            list of the force values (in Newtons) in the sheet of the datafile
        disp: pandasArray
            list of the displacement values (in mm) in the sheet of the datafile

    """
    date = datafile[0:6]
    path_to_datafile = large_tension_utils.reach_data_path(date) / datafile
    data_in_sheet = pd.read_excel(path_to_datafile, sheet_name=sheet, header=3, names=["s", "elongation", "MPa-section60mm2", "N", "MPa" ], usecols="A:E", decimal=',')
    time = data_in_sheet.s
    elongation = data_in_sheet.elongation
    stress = data_in_sheet.MPa
    time, elongation, stress = time.to_numpy(), elongation.to_numpy(), stress.to_numpy()
    non_negative_or_null_elongations = np.where(elongation > 0.001)[0]
    rescaled_elongation = np.array([e/100 + 1 for e in elongation[non_negative_or_null_elongations]])
    rescaled_elongation = np.array([e - rescaled_elongation[0] +1 for e in rescaled_elongation])
    stress = savgol_filter(stress, 101, 2)
    rescaled_stress = np.array([s*1000 - stress[non_negative_or_null_elongations][0]*1000 for s in stress[non_negative_or_null_elongations]])
    rescaled_time = time[non_negative_or_null_elongations] - time[non_negative_or_null_elongations][0]
    return rescaled_time, rescaled_elongation, rescaled_stress

# rescaled_time, rescaled_elongation, rescaled_stress = read_sheet_in_datafile(datafile, sheet)

"""Observation graphique des données expérimentales"""

def plot_data(datafile, sheet):
  rescaled_time, rescaled_elongation, rescaled_stress = read_sheet_in_datafile(datafile, sheet)
  plt.figure()
  plt.plot(rescaled_time, rescaled_stress, '-k', lw=3)
  plt.legend([sheet])
  plt.xlabel('time [s]')
  plt.ylabel('stress [kPa]')
  plt.show()

  plt.figure()
  plt.plot(rescaled_elongation, rescaled_stress, '-k', lw=3)
  plt.legend([sheet])
  plt.xlabel('elongation [-]')
  plt.ylabel('stress [kPa]')
  plt.show()

# plot_data(datafile, sheet)

"""✴ Modélisation ✴

Calcul de la fonction d'évolution du chargement
"""

def compute_f_evol(eta, alpha, I1, I1_max):
  if I1_max - 3 == 0:
    f_evol=1
  else:
    f_evol = 1 - eta* ( ( (I1_max - I1 ) / (I1_max - 3) )**alpha)
  return f_evol

"""Calcul de I1"""

def compute_I1(elongation_x):
    elongation_z = 1/elongation_x
    I1 = 1 + elongation_x**2 + elongation_z**2
    return I1

"""Calcul de la partie visqueuse Q"""
def compute_Q_list(elongation_vec, beta, tau_0, tau_1, tau_2, tau_3, c1, c2, c3, datafile, sheet):
  load_phase_time_dict, relaxation_phase_time_dict, discharge_phase_time_dict, load_phase_stress_dict, relaxation_phase_stress_dict, discharge_phase_stress_dict, load_phase_elongation_dict, relaxation_phase_elongation_dict, discharge_phase_elongation_dict = large_tension_utils.extract_data_per_steps(datafile, sheet)
  time_list, elongation_list, _ = read_sheet_in_datafile(datafile, sheet)
  dt_list = np.diff(time_list)
  # print(len(dt_list))
  Qx_list = np.zeros_like(elongation_vec)
  tau_list = np.zeros_like(elongation_vec)
  index = 0
  elongation_0 = 1
  I1_list = [compute_I1(elongation_x) for elongation_x in elongation_vec]
  S_H_list = [(c1 + 2*c2*(I1_list[i]-3) + 3*c3*(I1_list[i]-3)**2) * elongation_vec[i]**2 for i in range(len(I1_list))]
  number_of_steps = len(load_phase_elongation_dict)
  tau_parameters_list = [tau_0, tau_1, tau_2, tau_3]
  number_of_tau_parameters = len(tau_parameters_list) - 1

  for s in range(1, number_of_tau_parameters+1):
    load_phase_time, relaxation_phase_time, discharge_phase_time, _, _, _, load_phase_elongation, _, discharge_phase_elongation = load_phase_time_dict[s], relaxation_phase_time_dict[s], discharge_phase_time_dict[s], load_phase_stress_dict[s], relaxation_phase_stress_dict[s], discharge_phase_stress_dict[s], load_phase_elongation_dict[s], relaxation_phase_elongation_dict[s], discharge_phase_elongation_dict[s]
    step_len = len(load_phase_time) + len(relaxation_phase_time) + len(discharge_phase_time)
    tau_list_step = np.zeros(step_len)
    a = (tau_parameters_list[s] - tau_parameters_list[s-1]) / np.linalg.norm(load_phase_elongation[s])
    b = tau_0 - a*elongation_0
    for i in range(len(load_phase_time)):
        elongation_i = load_phase_elongation[i]
        tau = a * elongation_i  + b
        tau_list_step[i] = tau
    tau_0 = tau
    elongation_0 = elongation_i
    for j in range(len(relaxation_phase_time)):
      tau = tau_0
      tau_list_step[j + len(load_phase_time)] = tau 
    tau_0 = tau
    b = a * elongation_0 + tau_0
    for k in range(len(discharge_phase_time)):
      elongation_k = discharge_phase_elongation[k]
      tau = -a * elongation_k + b
      tau_list_step[k + len(load_phase_time) + len(relaxation_phase_time)] = tau    
    tau_0 = tau
    elongation_0 = elongation_k
    tau_list[index:index+step_len] = tau_list_step
    index = index+step_len

  for s in range(number_of_tau_parameters+1, number_of_steps+1):
    load_phase_time, relaxation_phase_time, discharge_phase_time, _, _, _, load_phase_elongation, _, discharge_phase_elongation = load_phase_time_dict[s], relaxation_phase_time_dict[s], discharge_phase_time_dict[s], load_phase_stress_dict[s], relaxation_phase_stress_dict[s], discharge_phase_stress_dict[s], load_phase_elongation_dict[s], relaxation_phase_elongation_dict[s], discharge_phase_elongation_dict[s]
    step_len = len(load_phase_time) + len(relaxation_phase_time) + len(discharge_phase_time)
    tau_list_step = np.zeros(step_len)
    b = tau_0 - a*elongation_0
    for i in range(len(load_phase_time)):
      elongation_i = load_phase_elongation[i]
      tau = a * elongation_i  + b
      tau_list_step[i] = tau
    tau_0 = tau
    elongation_0 = elongation_i
    for j in range(len(relaxation_phase_time)):
      tau = tau_0
      tau_list_step[j + len(load_phase_time)] = tau 
    tau_0 = tau
    b = a * elongation_0 + tau_0
    for k in range(len(discharge_phase_time)):
      elongation_k = discharge_phase_elongation[k]
      tau = -a * elongation_k + b
      tau_list_step[k + len(load_phase_time) + len(relaxation_phase_time)] = tau    
    tau_0 = tau
    elongation_0 = elongation_k
    tau_list[index:index+step_len] = tau_list_step
    index = index+step_len
  tau_list[index:] = [tau] * (len(elongation_vec) - index)
  for i in range(1, len(Qx_list)):
    # Qz_list[i] = np.exp(-dt_list[i-1]/tau)*Qz_list[i-1] + beta*(S_list[i] - S_list[i-1])
    tau = tau_list[i]
    Qx_list[i] = tau / (dt_list[i-1] + tau) * ( beta*(S_H_list[i] - S_H_list[i-1]) + Qx_list[i-1])
    # Qx_list[i] = tau / (dt_list[0] + tau) * ( dt_list[0]*beta*S_H_list[i] + Qx_list[i-1])
  return Qx_list

def compute_Q(Q_list, current_elongation, datafile, sheet):
  _, elongation, _ = read_sheet_in_datafile(datafile, sheet)
  def find_Q_at_elongation_s(current_elongation):
    index_elongation_is_current_elongation = np.where(elongation == current_elongation)[0][0]
    Q_at_elongation_s = Q_list[index_elongation_is_current_elongation]
    return Q_at_elongation_s
  return find_Q_at_elongation_s(current_elongation)


    

"""Calcul de la forme analytique de la contrainte"""

def compute_stress_visc(elongation_x_model, Q_x, c1, c2, c3, I1, f_evol):
  # Q = compute_Q(time[index_elongation_is_elongation_x_model], beta, tau, datafile, sheet)
  elongation_z_model = 1 / elongation_x_model
  # stress = 2*f_evol * (c1 + 2*c2*(I1-3) + 3*c3*(I1-3)**2) * (elongation_x_model**2 - elongation_z_model**2) + 8*c4 * elongation_x_model * elongation_x_e_model
  stress = 2*f_evol * (c1 + 2*c2*(I1-3) + 3*c3*(I1-3)**2) * (elongation_x_model**2 - elongation_z_model**2) + Q_x*(elongation_x_model**2) 
  return stress


"""Calcul de la contrainte analytique pendant l'essai"""
def compute_analytical_stress(datafile, sheet, params):
  # Load experimental data
  time_exp, elongation_exp, stress_exp = read_sheet_in_datafile(datafile, sheet)

  # Initialisation
  # elongation_x_model_list = elongation_exp
  # elongation_x_e_model_list = np.zeros_like(elongation_x_model_list)
  stress_model_list = np.zeros_like(stress_exp)
  time_model_list = time_exp

  I1_model_list = np.zeros_like(elongation_exp)
  I1_model_list[0] = 3
  I1_max = 3

  # elongation_x_model_list[0] = 1
  # elongation_x_e_model_list[0] = 1
  stress_model_list[0] = 0

  # [c1, c2, c3, c4, eta_0, eta, alpha] = params[0]
  [c1, c2, c3, beta, tau_0, tau_1, tau_2, tau_3, eta, alpha] = params[0]
  Qx_list = compute_Q_list(elongation_exp, beta, tau_0, tau_1, tau_2, tau_3, c1, c2, c3, datafile, sheet)

  # for i in range(1, len(elongation_x_model_list)):
  for i in range(1, len(elongation_exp)):
    elongation_x_model = elongation_exp[i]
    # elongation_at_time_s = elongation[index_time_is_s]
    Q = Qx_list[i]
    # elongation_x_model = elongation_x_model_list[i]
    I1 = compute_I1(elongation_x_model)
    I1_model_list[i] = I1
    I1_max = np.max(I1_model_list)
    f_evol = compute_f_evol(eta, alpha, I1, I1_max)
    # elongation_x_e_previous = elongation_x_e_model_list[i-1]
    # elongation_x_current = elongation_x_model
    # elongation_x_previous = elongation_x_model_list[i-1]
    # dt = time_intervals_list[i-1]
    # elongation_x_e_current = compute_elongation_x_e_current(elongation_x_e_previous, elongation_x_current, elongation_x_previous, c4, eta_0, dt)

    # elongation_x_e_model_list[i] = elongation_x_e_current
    # stress = compute_stress(elongation_x_model, elongation_x_e_current, c1, c2, c3, c4, I1, f_evol)
    stress = compute_stress_visc(elongation_x_model, Q, c1, c2, c3, I1, f_evol)
    stress_model_list[i] = stress
  return stress_model_list



"""Comparaison modèle-exp avec une figure adaptative"""

from typing_extensions import ParamSpec
def make_adaptive_plot_stress_vs_time(datafile, sheet):
  time_exp, elongation_exp, stress_exp = read_sheet_in_datafile(datafile, sheet)
  # c1_init, c2_init, c3_init, c4_init, eta_0_init, eta_init, alpha_init = 4, 6.2, 0.3, 0.07, 1,  1, 0.5
  
  params_init = large_tension_utils.extract_optimization_params_from_pkl(datafile, "C2PA", 'Powell', "0-100_vartau_3cycles")

  [c1_init, c2_init, c3_init, beta_init, tau_0_init, tau_1_init, tau_2_init, tau_3_init, eta_init, alpha_init] = params_init
  fig, ax = plt.subplots()
  line, = ax.plot(time_exp, compute_analytical_stress(datafile, sheet, [[c1_init, c2_init, c3_init, beta_init, tau_0_init, tau_1_init, tau_2_init, tau_3_init, eta_init, alpha_init]]), '-b', lw=2)
  fig.subplots_adjust(left=0.4, bottom=0.25)
  ax.plot(time_exp, stress_exp, ':k', alpha=0.8)
  ax.set_xlabel('time [s]')
  ax.set_ylabel('Pi stress [kPa]')
  ax.set_ylim((0, 100))
  ax.set_xlim((0, 200))


  # Make a horizontal slider to control c1.
  axc1 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
  c1_slider = Slider(
      ax=axc1,
      label='c1',
      valmin=0,
      valmax=20,
      valinit=c1_init,
      color='y'
  )
  
  axc2 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
  c2_slider = Slider(
      ax=axc2,
      label='c2',
      valmin=0,
      valmax=20,
      valinit=c2_init,
      color='y'
  )
  
  axc3 = fig.add_axes([0.25, 0.15, 0.65, 0.03])
  c3_slider = Slider(
      ax=axc3,
      label='c3',
      valmin=0,
      valmax=20,
      valinit=c3_init,
      color='y'
  )


  # Make a vertically oriented slider to control c2
  axbeta = fig.add_axes([0.025, 0.25, 0.0225, 0.63])
  beta_slider = Slider(
      ax=axbeta,
      label="beta",
      valmin=0,
      valmax=20,
      valinit=beta_init,
      orientation="vertical",
      color='m'
  )

  # Make a vertically oriented slider to control c3
  axtau_0 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
  tau_0_slider = Slider(
      ax=axtau_0,
      label="tau_0",
      valmin=0,
      valmax=20,
      valinit=tau_0_init,
      orientation="vertical",
      color='r'
  )
  
  # Make a vertically oriented slider to control c3
  axtau_1 = fig.add_axes([0.12, 0.25, 0.0225, 0.63])
  tau_1_slider = Slider(
      ax=axtau_1,
      label="tau_1",
      valmin=0,
      valmax=20,
      valinit=tau_1_init,
      orientation="vertical",
      color='r'
  )
  
  # Make a vertically oriented slider to control c3
  axtau_2 = fig.add_axes([0.14, 0.25, 0.0225, 0.63])
  tau_2_slider = Slider(
      ax=axtau_2,
      label="tau_2",
      valmin=0,
      valmax=20,
      valinit=tau_2_init,
      orientation="vertical",
      color='r'
  )
  
  # Make a vertically oriented slider to control c3
  axtau_3 = fig.add_axes([0.16, 0.25, 0.0225, 0.63])
  tau_3_slider = Slider(
      ax=axtau_3,
      label="tau_3",
      valmin=0,
      valmax=20,
      valinit=tau_3_init,
      orientation="vertical",
      color='r'
  )
  
  # Make a vertically oriented slider to control eta
  axeta = fig.add_axes([0.2, 0.25, 0.0225, 0.63])
  eta_slider = Slider(
      ax=axeta,
      label=r"$\eta$",
      valmin=1,
      valmax=20,
      valinit=eta_init,
      orientation="vertical",
      color='b'
  )

  # Make a vertically oriented slider to control alpha
  axalpha = fig.add_axes([0.24, 0.25, 0.0225, 0.63])
  alpha_slider = Slider(
      ax=axalpha,
      label=r"$\alpha$",
      valmin=0.1,
      valmax=5,
      valinit=alpha_init,
      orientation="vertical",
      color='r'
  )


  def compute_adaptive_stress(c1, c2, c3, beta, tau_0, tau_1, tau_2, tau_3, eta, alpha):
  # def compute_adaptive_stress(c1, c2, c3, c4, eta_0, eta, alpha):
    # params = [[c1, c2, c3, c4, eta_0, eta, alpha]]
    params = [[c1, c2, c3, beta, tau_0, tau_1, tau_2, tau_3, eta, alpha]]
    stress_list = compute_analytical_stress(datafile, sheet, params)
    return stress_list

  # The function to be called anytime a slider's value changes
  def update(val):
      line.set_ydata(compute_adaptive_stress(c1_slider.val, c2_slider.val, c3_slider.val, beta_slider.val, tau_0_slider.val, tau_1_slider.val, tau_2_slider.val, tau_3_slider.val, eta_slider.val, alpha_slider.val))
      fig.canvas.draw_idle()

  # register the update function with each slider
  c1_slider.on_changed(update)
  c2_slider.on_changed(update)
  c3_slider.on_changed(update)
  beta_slider.on_changed(update)
  tau_0_slider.on_changed(update)
  tau_1_slider.on_changed(update)
  tau_2_slider.on_changed(update)
  tau_3_slider.on_changed(update)
  eta_slider.on_changed(update)
  alpha_slider.on_changed(update)


  # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
  resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
  button = Button(resetax, 'Reset', hovercolor='0.975')

  def reset(event):
    c1_slider.reset()
    c2_slider.reset()
    c3_slider.reset()
    beta_slider.reset()
    eta_slider.reset()
    tau_0_slider.reset()
    tau_1_slider.reset()
    tau_2_slider.reset()
    tau_3_slider.reset()
    alpha_slider.reset()
  button.on_clicked(reset)

  plt.show()

def make_adaptive_plot_stress_vs_elongation(datafile, sheet):
  time_exp, elongation_exp, stress_exp = read_sheet_in_datafile(datafile, sheet)
  params_init = large_tension_utils.extract_optimization_params_from_pkl(datafile, "C2PA", 'Powell', "0-100_vartau_3cycles")
  [c1_init, c2_init, c3_init, beta_init, tau_0_init, tau_1_init, tau_2_init, tau_3_init, eta_init, alpha_init] = params_init
  fig, ax = plt.subplots()
  
  line, = ax.plot(elongation_exp, compute_analytical_stress(datafile, sheet, [params_init]), '-b', lw=2)
  # line, = ax.plot(elongation_exp, compute_analytical_stress(datafile, sheet, [[c1_init, c2_init, c3_init, c4_init, eta_0_init, eta_init, alpha_init]]), '-b', lw=2)
  # adjust the main plot to make room for the sliders
  fig.subplots_adjust(left=0.4, bottom=0.25)
  ax.plot(elongation_exp, stress_exp, ':k', alpha=0.8)
  ax.set_xlabel('elongation [-]')
  ax.set_ylabel('Pi stress [kPa]')
  ax.set_ylim((0, 100))


  # Make a horizontal slider to control c1.
  axc1 = fig.add_axes([0.25, 0.1, 0.65, 0.03])
  c1_slider = Slider(
      ax=axc1,
      label='c1',
      valmin=5,
      valmax=30,
      valinit=c1_init,
      color='y'
  )
  
  # Make a vertically oriented slider to control c4
  axtau_0 = fig.add_axes([0.25, 0.15, 0.65, 0.03])
  tau_0_slider = Slider(
      ax=axtau_0,
      label="tau_0",
      valmin=0,
      valmax=10,
      valinit=tau_0_init,
      color='g'
  )

  # Make a vertically oriented slider to control c4
  axtau_1 = fig.add_axes([0.25, 0.2, 0.65, 0.03])
  tau_1_slider = Slider(
      ax=axtau_1,
      label="tau_1",
      valmin=0,
      valmax=10,
      valinit=tau_1_init,
      color='g'
  )
  
  # Make a vertically oriented slider to control c4
  axtau_2 = fig.add_axes([0.25, 0.25, 0.65, 0.03])
  tau_2_slider = Slider(
      ax=axtau_2,
      label="tau_2",
      valmin=0,
      valmax=10,
      valinit=tau_2_init,
      color='g'
  )
  
  # Make a vertically oriented slider to control c4
  axtau_3 = fig.add_axes([0.25, 0.3, 0.65, 0.03])
  tau_3_slider = Slider(
      ax=axtau_3,
      label="tau_3",
      valmin=0,
      valmax=10,
      valinit=tau_3_init,
      color='g'
  )
  
  # Make a vertically oriented slider to control c2
  axc2 = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
  c2_slider = Slider(
      ax=axc2,
      label="c2",
      valmin=0,
      valmax=10,
      valinit=c2_init,
      orientation="vertical",
      color='m'
  )

  # Make a vertically oriented slider to control c3
  axc3 = fig.add_axes([0.175, 0.25, 0.0225, 0.63])
  c3_slider = Slider(
      ax=axc3,
      label="c3",
      valmin=0,
      valmax=10,
      valinit=c3_init,
      orientation="vertical",
      color='r'
  )

  # Make a vertically oriented slider to control c4
  axbeta = fig.add_axes([0.25, 0.25, 0.0225, 0.63])
  beta_slider = Slider(
      ax=axbeta,
      label="beta",
      valmin=0,
      valmax=10,
      valinit=beta_init,
      orientation="vertical",
      color='g'
  )


  # Make a vertically oriented slider to control eta
  axeta = fig.add_axes([0.325, 0.25, 0.0225, 0.63])
  eta_slider = Slider(
      ax=axeta,
      label=r"$\eta$",
      valmin=0.1,
      valmax=5,
      valinit=eta_init,
      orientation="vertical",
      color='b'
  )

  # Make a vertically oriented slider to control alpha
  axalpha = fig.add_axes([0.4, 0.25, 0.0225, 0.63])
  alpha_slider = Slider(
      ax=axalpha,
      label=r"$\alpha$",
      valmin=0.1,
      valmax=3,
      valinit=alpha_init,
      orientation="vertical",
      color='r'
  )


  def compute_adaptive_stress(c1, c2, c3, c4, eta_0, eta, alpha):
    params = [[c1, c2, c3, c4, eta_0, eta, alpha]]
    stress_list = compute_analytical_stress(datafile, sheet, params)
    return stress_list

  # The function to be called anytime a slider's value changes
  def update(val):
      line.set_ydata(compute_adaptive_stress(c1_slider.val, c2_slider.val, c3_slider.val, c4_slider.val, eta_0_slider.val, eta_slider.val, alpha_slider.val))
      fig.canvas.draw_idle()

  # register the update function with each slider
  c1_slider.on_changed(update)
  c2_slider.on_changed(update)
  c3_slider.on_changed(update)
  beta_slider.on_changed(update)
  tau_0_slider.on_changed(update)
  tau_1_slider.on_changed(update)
  tau_2_slider.on_changed(update)
  tau_3_slider.on_changed(update)
  eta_slider.on_changed(update)
  alpha_slider.on_changed(update)


  # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
  resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
  button = Button(resetax, 'Reset', hovercolor='0.975')

  def reset(event):
    c1_slider.reset()
    c2_slider.reset()
    c3_slider.reset()
    beta_slider.reset()
    tau_0_slider.reset()
    tau_1_slider.reset()
    tau_2_slider.reset()
    tau_3_slider.reset()
    eta_slider.reset()
    alpha_slider.reset()
  button.on_clicked(reset)

  plt.show()

""" Comparaison modèle-exp avec un jeu de paramètres fixe"""
def plot_exp_vs_model_params(datafile, sheet, params):
  time, elongation, stress_exp = read_sheet_in_datafile(datafile, sheet)
  stress_model = compute_analytical_stress(datafile, sheet, params)
  
  createfigure = CreateFigure()
  fonts = Fonts()
  savefigure = SaveFigure()
  fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_time = fig_stress_vs_time.gca()
  
  fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
  
  ax_stress_vs_time.plot(time, stress_exp, ':k', lw=3, label='exp')
  ax_stress_vs_time.plot(time, stress_model, '-r', lw=3, label='num')
  ax_stress_vs_time.set_title(sheet, font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.set_xlabel('time [s]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.set_ylabel('stress [kPa]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)

  ax_stress_vs_elongation.plot(elongation, stress_exp, ':k', lw=3, label='exp')
  ax_stress_vs_elongation.plot(elongation, stress_model, '-r', lw=3, label='num')
  ax_stress_vs_elongation.set_title(sheet, font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_xlabel('elongation [-]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_ylabel('stress [kPa]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)

  savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "stress_time_exp_numRebouah_" + sheet + "_test")
  savefigure.save_as_png(fig_stress_vs_elongation, datafile[0:6] + "stress_elongation_exp_numRebouah_" + sheet + "_test")

  plt.close(fig_stress_vs_elongation)
  plt.close(fig_stress_vs_time)

""" Comparaison modèle-exp avec un jeu de paramètres fixe"""
def plot_exp_vs_model_params_opti(datafile, sheet, params_opti, minimization_method, suffix):
  time, elongation, stress_exp = read_sheet_in_datafile(datafile, sheet)
  stress_model = compute_analytical_stress(datafile, sheet, params_opti)
  
  createfigure = CreateFigure()
  fonts = Fonts()
  savefigure = SaveFigure()
  fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_time = fig_stress_vs_time.gca()
  
  fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
  
  ax_stress_vs_time.plot(time, stress_exp, ':k', lw=3, label='exp')
  ax_stress_vs_time.plot(time, stress_model, '-r', lw=3, label='num')
  ax_stress_vs_time.set_title(sheet, font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.set_xlabel('time [s]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.set_ylabel('stress [kPa]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)

  ax_stress_vs_elongation.plot(elongation, stress_exp, ':k', lw=3, label='exp')
  ax_stress_vs_elongation.plot(elongation, stress_model, '-r', lw=3, label='num')
  ax_stress_vs_elongation.set_title(sheet, font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_xlabel('elongation [-]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_ylabel('stress [kPa]', font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)

  savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "stress_time_exp_numRebouah_" + sheet + "_manual_" + minimization_method + "_" + suffix)
  savefigure.save_as_png(fig_stress_vs_elongation, datafile[0:6] + "stress_elongation_exp_numRebouah_" + sheet + "_manual_" + minimization_method + "_" + suffix)

  plt.close(fig_stress_vs_elongation)
  plt.close(fig_stress_vs_time)


"""Find optimal parameters"""

from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit, minimize, rosen, rosen_der


def find_optimal_parameters(datafile, sheet, minimization_method):
  time_exp, elongation_exp, stress_exp = read_sheet_in_datafile(datafile, sheet)
  c1_init, c2_init, c3_init, beta_init, tau_0_init, tau_1_init, tau_2_init, tau_3_init, eta_init, alpha_init = 10, 5, 0.5, 0.07, 1,  1.1, 1.2, 1.3, 1, 0.5
  initial_guess_values = [c1_init, c2_init, c3_init, beta_init, tau_0_init, tau_1_init, tau_2_init, tau_3_init, eta_init, alpha_init]
  bounds_values = [(0.1, 30), (0.1, 30), (0.1, 10), (0.01, 50), (0.01, 10), (0.01, 10), (0.01, 10), (0.01, 10), (1, 5), (0.1, 5)]
  # beta eta and alpha have to be positive
  def minimization_function(params):
      stress_list_model = compute_analytical_stress(datafile, sheet, [params])
      n = len(stress_list_model)
      stress_list_exp = stress_exp
      least_square = mean_squared_error(stress_list_exp[0:int(n/2)], stress_list_model[0:int(n/2)])
      return least_square

  res = minimize(minimization_function, initial_guess_values, method=minimization_method, bounds=bounds_values,
            options={'disp': False})
  params_opti = res.x
  return params_opti

"""Compare optimized to experimental stress"""

def plot_comparison_stress_model_experiment(datafile, sheet, minimization_method, suffix):
  createfigure = CreateFigure()
  fonts = Fonts()
  savefigure = SaveFigure()
  time_exp, elongation_exp, stress_exp = read_sheet_in_datafile(datafile, sheet)
  params_opti = find_optimal_parameters(datafile, sheet, minimization_method)
  large_tension_utils.export_optimization_params_as_pkl(datafile, sheet, params_opti, minimization_method, suffix)
  stress_list_model = compute_analytical_stress(datafile, sheet, [params_opti])
  n = len(stress_list_model)
  fig_stress_vs_elongation = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_elongation = fig_stress_vs_elongation.gca()
  ax_stress_vs_elongation.plot(elongation_exp[0:int(n/2)], stress_exp[0:int(n/2)], '-k', lw=1, label='exp')
  ax_stress_vs_elongation.plot(elongation_exp[0:int(n/2)], stress_list_model[0:int(n/2)], '-r', lw=1, label='model')
  ax_stress_vs_elongation.set_xlabel(r"$\lambda_x$ [-]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.set_ylabel(r"$\sigma_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_elongation.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
  savefigure.save_as_png(fig_stress_vs_elongation, datafile[0:6] + "stress_elong_exp_numRebouah_" + sheet + "_" + minimization_method + "_" + suffix)
  
  fig_stress_vs_time = createfigure.rectangle_figure(pixels=180)
  ax_stress_vs_time = fig_stress_vs_time.gca()
  ax_stress_vs_time.plot(time_exp[0:int(n/2)], stress_exp[0:int(n/2)], '-k', lw=1, label='exp')
  ax_stress_vs_time.plot(time_exp[0:int(n/2)], stress_list_model[0:int(n/2)], '-r', lw=1, label='model')
  ax_stress_vs_time.set_xlabel("time [s]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.set_ylabel(r"$\sigma_x^{exp}$ [kPa]", font=fonts.serif(), fontsize=26)
  ax_stress_vs_time.legend(prop=fonts.serif(), loc='upper left', framealpha=0.7)
  savefigure.save_as_png(fig_stress_vs_time, datafile[0:6] + "stress_time_exp_numRebouah_" + sheet + "_" + minimization_method+ "_" + suffix)
  plt.close(fig_stress_vs_elongation)
  plt.close(fig_stress_vs_time)
  




  
if __name__ == "__main__":
  sheet = "C1PB"
  datafile = "231012_large_tension_data.xlsx"
  files_zwick = Files_Zwick('large_tension_data.xlsx')

#   c1_test, c2_test, c3_test, beta_test, tau_test, a_test, eta_test, alpha_test = 4, 6.2, 0.3, 0.07, 1, 1, 1, 0.5
#   params_test = [[c1_test, c2_test, c3_test, beta_test, tau_test, a_test, eta_test, alpha_test]]
  start_time = time.time()
  datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
  # plot_exp_vs_model_params(datafile, sheet, params_test)
  # make_adaptive_plot_stress_vs_elongation(datafile, "C2PA")
  make_adaptive_plot_stress_vs_time(datafile, "C2PA")
  minimization_method_list = ['Powell']# ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
  # sheets_list_with_data = ["C2PA"]
  suffix = "0-100_vartau_3cycles"
  
  # for minimization_method in minimization_method_list:
  #   for sheet in sheets_list_with_data:
  #     # try:
  #     print('started', minimization_method)
  #     plot_comparison_stress_model_experiment(datafile, sheet, minimization_method, suffix)
  #     print('done succeed', minimization_method)
  #     print("--- %s seconds ---" % (time.time() - start_time))
  #   # except:
  #     print('failed', minimization_method)
      # None
#   sheet = "C3PA"
  # params = large_tension_utils.extract_optimization_params_from_pkl(datafile, "C2PA", 'Powell', suffix)
  # print(sheet)
  # print ('c1, c2, c3, beta, tau0, tau1, tau2, tau3, eta, alpha')
  # print(params)
  # for minimization_method in minimization_method_list:
  #   for sheet in sheets_list_with_data:
  #     print('started', minimization_method)
  #     print('done succeed', minimization_method)
  #     print("--- %s seconds ---" % (time.time() - start_time))

      



