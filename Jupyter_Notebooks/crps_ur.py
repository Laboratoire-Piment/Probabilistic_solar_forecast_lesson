import numpy as np
import pandas as pd

def crps_ensemble(fcst,obs):
    """
    Calculates Continuous Ranked Probability Score (CRPS) and its decomposition CRPS=REL-RES+UNC for ensemble forecasts

    Parameters
    ----------
    fcst: DataFrame (N, M) - Ensemble forecasts (N cases, M members)
    obs: DataFrame (N,1) - Observations

    Returns
    -------
    df_crps: pd.DataFrame['mean CRPS' (float), 'Reliability' (float),
                          'Resolution' (float), 'Uncertainty' (float)]
    crps: DataFrame (N,1), Individual CRPS values

    See
    ---
    Hersbach, H., 2000. Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems. Weather and Forecasting 15, 559?570. https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
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

    rel = np.sum(g*(o-pi)**2) # Reliability
    crps_pot = np.sum(g*o*(1-o)) # CRPSpot = Uncertainty - Resolution

    # Compute Uncertainty with vectorised code
    X = obs.to_numpy()          # shape (n, d)
    diff = np.abs(X[:, None, :] - X[None, :, :])  # shape (n, n, d)
    unc = diff[np.tril_indices(n, k=-1)].sum() / (n**2) # # keep only lower triangle (j < i)

    res = unc - crps_pot # Resolution

    df_crps = pd.DataFrame({'Mean CRPS': [mean_crps],
                            'Reliability': [rel],
                            'Resolution': [res],
                            'Uncertainty': [unc]})

    return df_crps, crps

def Brier_Score(fcst,obs,tau,xq):
    """
    Compute the Brier Score (BS) and its decomposition BS(xq)=REL(xq)-RES(xq)+UNC(xq) for a specific threshold xq
    
    Parameters
    ----------
        fcst: numpy array of N forecats x M quantiles
        obs: numpy array of N observations
        tau: numpy array with probability level of the quantiles (between 0 and 1)
        xq: threshold
    
    Return
    ------
        bs: float - Brier Score
        bs_rel: float - Reliability component
        bs_res: float - Resolution component
        bs_unc: float - Uncertainty component
    
    See
    ---
    Hersbach, H., 2000. Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems. Weather and Forecasting 15, 559?570. https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
    
    Lauret, P., David, M., Pinson, P., 2019. Verification of solar irradiance probabilistic forecasts. Solar Energy 194, 254?271. https://doi.org/10.1016/j.solener.2019.10.041

    Note: The CRPS can be obtained through the integration of the BS for GHI
    threshold between 0 and GHI_max (i.e. max climatology)
    
    example: GHI_values=[0:10:1200];
    CRPS=trapz(GHI_values,bs);CRPS_rel=trapz(GHI_values,bs_rel );CRPS_res=trapz(GHI_values,bs_res)
    """

    n, m = fcst.shape # n: number of steps, m: number of quantiles
    ct = np.zeros([m+1,2]) # Contingency table

    for i in np.arange(n):
        s = np.sort(np.append(fcst[i,:], xq)) # Sort forecast and threshold
        idx = np.searchsorted(s, xq)
        
        # if idx == 0: yq = 0.05
        # elif idx == m+1: yq = 0.95
        # else: yq = (tau[idx-1] + tau[idx])/2

        if obs[i] <= xq: ct[idx,0] = ct[idx,0] + 1 # Event occured
        else: ct[idx,1] = ct[idx,1] +1  # Event not occured
    
    yq = 0.5
    pyi= np.sum(ct,axis=1)/n # marginal probabilities of event
    zibar=ct[:,0]/np.sum(ct,axis=1)
    zibar = np.where(np.isnan(zibar) , 0.0, zibar)
    yi=np.sort(np.append(tau,yq))

    bs = np.sum(pyi*(zibar*(1-yi).transpose()**2 + (1-zibar)*yi.transpose()**2)) # Equation (40) Herbasch.
    bs_rel = np.sum(pyi.transpose()*(yi-zibar.transpose())**2) # Equation (26) Lauret et al.
    zbar = np.sum(pyi*zibar)
    bs_res = np.sum(pyi*(zibar-zbar)**2) # Equation (27) Lauret et al.
    bs_unc = zbar*(1-zbar) # Equation (28) Lauret et al.

    return bs, bs_rel, bs_res, bs_unc

def crps_quantile_forecast(fcst,tau,obs,nb_thresh):
    """
    Compute the CRPS and its decomposition CRPS = REL - RES + UNC using the integartion of the Brier Score

    Parameters
    ----------
    fcst: DataFrame (N, M) - Quantile forecasts (N cases, M quantiles)
    tau: DataFrame(M) - Probability level of the quantiles (between 0 and 1)
    obs: DataFrame (N,1) - Observations
    nb_thresh: int - Number of thresholds xq used to divide the observation space (between min(obs) and max(obs))

    Returns
    -------
    df_crps: pd.DataFrame['mean CRPS' (float), 'Reliability' (float),
                          'Resolution' (float), 'Uncertainty' (float)]
   

    See
    ---
    Lauret, P., David, M., Pinson, P., 2019. Verification of solar irradiance probabilistic forecasts. Solar Energy 194, 254?271. https://doi.org/10.1016/j.solener.2019.10.04
    """

    fcst = np.asarray(fcst)
    obs = np.asarray(obs)
    tau = np.asarray(tau)

    min_obs = obs.min()
    max_obs = obs.max()
    thresh = np.linspace(min_obs, max_obs, nb_thresh)

    bs = np.empty([nb_thresh])
    bs_rel = np.empty([nb_thresh])
    bs_res = np.empty([nb_thresh])
    bs_unc = np.empty([nb_thresh])
    for i, xq in enumerate(thresh):
        bs[i], bs_rel[i], bs_res[i], bs_unc[i] = Brier_Score(fcst,obs,tau,xq)

    crps = np.trapezoid(y = bs, x = thresh)
    rel = np.trapezoid(y = bs_rel, x = thresh)
    res = np.trapezoid(y = bs_res, x = thresh)
    unc = np.trapezoid(y = bs_unc, x = thresh)

    return pd.DataFrame({'CRPS': [crps],
                         'Reliability': [rel],
                         'Resolution': [res],
                         'Uncertainty': [unc]})
        

