# utils for plotting behavior
from scipy.stats import norm
from scipy.optimize import minimize
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(fit_params, x_data, y_data):
    s,b,RL,LL = fit_params
    ypred = RL + (1-RL-LL)/(1+np.exp(-(s*x_data+b)))
    return -np.sum(norm.logpdf(y_data, loc=ypred))

def psych_curve(target, coherence, ret_ax=None, annot=False ,xspace=np.linspace(-1,1,50), kwargs_plot={}, kwargs_error={}):
    """
    function to plot psych curves
    target: binomial target (y), aka R_response, or Repeat
    coherence: coherence (x)
    ret_ax: 
        None: returns tupple of vectors: points (2-D, x&y) and confidence intervals (2D, lower and upper), fitted line[2D, x, and y]
        True: returns matplotlib ax object
    xspace: x points to retrieve y_pred from curve fitting
    kwargs: aesthethics for each of the plotting
    """
    # convert vect to arr if they are pd.Series
    for item in [target, coherence]:
        if isinstance(item, pd.Series):
            item = item.values
            if np.isnan(item).sum():
                raise ValueError('Nans detected in provided vector')
    if np.unique(target).size>2:
        raise ValueError('target has >2 unique values (invalids?!)')

    """ not required
    if not (coherence<0).sum(): # 0 in the sum means there are no values under 0
        coherence = coherence * 2 - 1 # (from 01 to -11)
    
    r_resp = np.full(hit.size, np.nan)
    r_resp[np.where(np.logical_or(hit==0, hit==1)==True)[0]] = 0
    r_resp[np.where(np.logical_and(rewside==1, hit==1)==True)[0]] = 1 # right and correct
    r_resp[np.where(np.logical_and(rewside==0, hit==0)==True)[0]] = 1 # left and incorrect
    """

    tmp = pd.DataFrame({'target': target, 'coh':coherence})
    tab = tmp.groupby('coh')['target'].agg(['mean', 'sum','count'])

    tab['low_ci'], tab['high_ci'] = proportion_confint(tab['sum'],tab['count'], method='beta')
    # readapt value to plot errorbars directly
    tab['low_ci'] = tab['mean']-tab['low_ci']
    tab['high_ci'] = tab['high_ci']-tab['mean']
    

    # sigmoid fit
    loglike = minimize(sigmoid, [1,1,0,0], (coherence, target))
    s, b, RL, LL = loglike['x']
    y_fit = RL + (1-RL-LL)/(1+np.exp(-(s*xspace + b)))
    if ret_ax is None:
        return (tab.index.values,tab['mean'].values), np.array([tab.low_ci.values, tab.high_ci.values]), (xspace, y_fit)
        # plot it with plot(*3rd_returnvalue) + errorbar(*1st_retval, yerr=2nd_retval, ls='none')
    else:
        #f, ax = plt.subplots()
        if kwargs_plot:
            pass
        else:
            kwargs_plot = {'c':'tab:blue'}

        if kwargs_error:
            pass
        else:
            kwargs_error = dict(ls='none', marker='o', markersize=3, capsize=4, color='maroon')      

        ret_ax.axhline(y=0.5, linestyle=':', c='k')
        ret_ax.axvline(x=0, linestyle=':', c='k')
        ret_ax.plot(xspace, y_fit ,**kwargs_plot)
        ret_ax.errorbar(tab.index,
                    tab['mean'].values,
                    yerr=np.array([tab.low_ci.values, tab.high_ci.values]),
                    **kwargs_error)

        if annot:
            ret_ax.annotate(f'bias: {round(b,2)}\nsens: {round(s,2)}\nRπ: {round(RL,2)}\nLπ: {round(LL,2)}', (tab.index.values.min(), 0.55), ha='left')
        return ret_ax
