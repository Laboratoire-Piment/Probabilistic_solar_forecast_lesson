import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_eps_cdf(eps, obs, fig_size, xaxis_lim):
  # INPUTS
  #   eps: Vector with the ensemble members of size
  #   obs: Observation value
  #   plot_title: Title of the figure (string)

  # Assign cumulative probabilities to members
  m = len(eps) # Number of members
  eps_prob = np.arange(0,1,1/(m)) # Vector of cumulative probabilities of the ensemble members

  # Sort members in ascending order
  eps = np.sort(eps)

  # Oservation step function and corresponding cumulative probability
  step_func_obs = np.array([obs,obs])
  step_func_obs_prob = np.array([0.0,1.0])

  #Add lower and upper limits to ensemble and observation CDFs
  low = np.min([eps.min(),obs])-10 # Lower limit
  up = np.max([eps.max(),obs])+10 # Upper limit
  eps = np.concat((low, eps, up), axis=None)
  eps_prob = np.concat((0.0, eps_prob, 1.0), axis=None)
  step_func_obs = np.concat((low, step_func_obs, up), axis=None)
  step_func_obs_prob = np.concat((0.0, step_func_obs_prob, 1.0), axis=None)

  # Plotting
  plt.rcParams.update({'font.size': 10})
  plt.figure(figsize=fig_size)
  plt.step(eps, eps_prob, where='pre',
          color='grey', label='Ensemble') # Plot ensemble CDF
  plt.step(step_func_obs, step_func_obs_prob, where='pre',
          color='magenta', linestyle='dashed', label='Observation') # Plot observation CDF step function
  plt.xlabel('GHI quantiles ($W.m^{-2}$)')  # X-axis label
  plt.xlim(xaxis_lim)
  plt.ylabel('Cumulative probability')  # Y-axis label
  plt.grid()
  plt.legend()

def plot_quantile_forecast_cdf(fcst, tau, obs, fig_size, xaxis_lim):
  # INPUTS
  #   fcst: Quantiles (Vector size M)
  #   tau: Cumulative probabilities (Vector size M)
  #   obs: Observation value (Scalar)
  #   plot_title: Title of the figure (SDtring)

  # Assign cumulative probabilities to members
  m = len(fcst) # Number of members

  # Sort members in ascending order
  fcst = np.sort(fcst)

  # Oservation step function and corresponding cumulative probability
  step_func_obs = np.array([obs,obs])
  step_func_obs_prob = np.array([0.0,1.0])

  #Add lower and upper limits to observation CDFs
  low = np.min([fcst.min(),obs])-10 # Lower limit
  up = np.max([fcst.max(),obs])+10 # Upper limit
  step_func_obs = np.concat((low, step_func_obs, up), axis=None)
  step_func_obs_prob = np.concat((0.0, step_func_obs_prob, 1.0), axis=None)

  # Plotting
  plt.rcParams.update({'font.size': 10})
  plt.figure(figsize=fig_size)
  plt.plot(fcst, tau, 'o',
          color='grey', label='Quantile forecast') # Plot ensemble CDF
  plt.step(step_func_obs, step_func_obs_prob, where='pre',
          color='magenta', linestyle='dashed', label='Observation') # Plot observation CDF step function
  plt.xlabel('GHI quantiles ($W.m^{-2}$)')  # X-axis label
  plt.xlim(xaxis_lim)
  plt.ylabel('Cumulative probability')  # Y-axis label
  plt.grid()
  plt.legend()