import numpy as np
import pandas as pd

def crps_ensemble(fcst,obs):
    """
    Calculates Continuous Ranked Probability Score (CRPS)
    for ensemble forecasts following Hersbach (2000).

    Parameters
    ----------
    fcst : DataFrame (N, M) - Ensemble forecasts (N cases, M members)
    obs : DataFrame (N,1) - Observations

    Returns
    -------
    mean_crps : float - mean CRPS along the N forecast/observation pairs
    rel : float - Reliability
    res : float - Resolution
    unc : float - Uncertainty
    crps : DataFrame (N,1), Individual CRPS values
    """

    n,m = np.shape(fcst) # n: number of steps, m: number of members
    pi = np.arange(m+1)/m # Cumulative probabilitis
    pi_left2 = pi**2
    pi_right2 = (1-pi)**2
    alpha = np.empty([n,m+1])
    beta = np.empty([n,m+1])
    left_outliers = np.zeros(n, dtype=bool)
    right_outliers = np.ones(n, dtype=bool)
    
    crps = pd.DataFrame({'crps': np.empty([n])}) # Pre-allocate crps output
    
    for i in np.arange(n):
      s = np.sort(np.append(fcst.iloc[i,:], obs.iloc[i])) # Sort ensemble members and observation
      idx_obs = np.argmin(np.abs(s-obs.iloc[i].values)) # Find position of the obs
      if idx_obs == 0: #observation is lower than first member
        alpha[i,:] = 0
        beta[i,:] = np.append(np.diff(s), 0)
        left_outliers[i] = True
        crps.iloc[i] = np.sum(beta[i,:]*pi_right2)
      elif idx_obs == m: #observation is higher than last member
        alpha[i,:] = np.append(0, np.diff(s))
        beta[i,:] = 0
        right_outliers[i] = False
        crps.iloc[i] = np.sum(alpha[i,:]*pi_left2)
      else: #observation falls inside members
        alpha[i,:] = np.append(0, np.diff(s)) * np.concat(([0], np.ones([idx_obs]), np.zeros([m-idx_obs])))
        beta[i,:] = np.append(np.diff(s), 0) * np.concat((np.zeros([idx_obs]), np.ones([m-idx_obs]) ,[0]))
        crps.iloc[i] = np.sum(alpha[i,:]*pi_left2 + beta[i,:]*pi_right2)

    mean_crps = np.mean(crps)

    # CRPS decomposition
    g = np.nanmean(alpha, axis=0) + np.nanmean(beta, axis=0)
    o = np.nanmean(beta, axis=0)/(np.nanmean(alpha, axis=0) + np.nanmean(beta, axis=0))
    o[0] = np.sum(left_outliers)/n
    o[-1] = np.sum(right_outliers)/n
    if o[0] != 0: g[0] = np.nanmean(beta[:,0])/o[0]
    if o[-1] != 1: g[-1] = np.nanmean(alpha[:,-1])/(1-o[-1])
    #if (O(end) ~= 1) G(end) = nanmean(alpha(:,end))/(1-O(end));end

    rel = np.sum(g*(o-pi)**2) # Reliability
    crps_pot = np.sum(g*o*(1-o)) # CRPSpot = Uncertainty - Resolution

    # Compute Uncertainty with vectorised code
    X = obs.to_numpy()          # shape (n, d)
    diff = np.abs(X[:, None, :] - X[None, :, :])  # shape (n, n, d)
    unc = diff[np.tril_indices(n, k=-1)].sum() / (n**2) # # keep only lower triangle (j < i)

    res = unc - crps_pot # Resolution

    return mean_crps, rel, res, unc, crps