import numpy as np
import pandas as pd

def logit(x, xmin,xmax):
    p = (x - xmin) / (xmax - xmin)
    return np.log(p) - np.log(1 - p)

def inv_logit(x, xmin, xmax):
    p = np.exp(x) / (1 + np.exp(x))
    return xmin + p*(xmax-xmin)


def train(fcst,obs):
    """
    Calculates the coefficient of the Variance Deficit method on a training set
    Sperati et al. (2016).

    Parameters
    ----------
    fcst : DataFrame (N, M) - Ensemble forecasts (N cases, M members)
    obs : DataFrame (N,1) - Observations

    Returns
    -------
    vd_coeff : float - Variance deficit coefficient
    """
    
    vd_coeff = np.sqrt(np.mean((np.mean(fcst,axis=1)-obs)**2))/(np.mean(np.std(fcst,axis=1)))

    return vd_coeff

def predict(fcst,vd_coeff):
    """
    Post-process an ensemble forecast using the Variance Deficit method
    Sperati et al. (2016).

    Parameters
    ----------
    fcst : DataFrame (N, M) - Ensemble forecasts (N cases, M members)
    vd_coeff : float - Variance Deficit coefficient

    Returns
    -------
    cal_fcst : DataFrame (N, M) - Calibrated ensemble forecast
    """
    n,m = np.shape(fcst) # n: number of steps, m: number of members

    emin = fcst.min()-0.01
    emax = fcst.max()+0.01

    log_fcst = logit(fcst,emin,emax) # Log transformation

    # Calibration
    mean_log_fcst = np.repeat(np.array(log_fcst.mean(axis=1))[:, None], m, axis=1)
    cal_log_fcst = mean_log_fcst + vd_coeff*(log_fcst-mean_log_fcst)

    return inv_logit(cal_log_fcst,emin,emax) # Inverse logit transformation
