# utils for plotting behavior
# this should be renamed to plotting/figures
from scipy.stats import norm, sem
from scipy.optimize import minimize
from statsmodels.stats.proportion import proportion_confint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from utilsJ.regularimports import groupby_binom_ci
import types


def help():
    """should print at least available functions"""
    print('available methods:')
    print('distbysubj: grid of distributions')
    print('psych_curve')
    print('correcting_kiani')
    print('com_heatmap')
    print('binned curve: curve of means and err of y-var binning by x-var')

#class cplot(): # cplot stands for custom plot
    #def __init__():

def distbysubj(df, data, by, grid_kwargs=dict(col_wrap=2, hue='CoM_sugg', aspect=2), 
                dist_kwargs=dict(kde=False, norm_hist=True,bins=np.linspace(0,400,50))):
    """
    returns facet (g) so user can use extra methods
        ie: .set_axis_labels('x','y')
            . add_legend
        data: what to plot (bin on) ~ str(df.col header)
        by: sorter (ie defines #subplots) ~ str(df.col header)

        returns sns.FacetGrid obj
    """
    g = sns.FacetGrid(df, col=by , **grid_kwargs)
    g = g.map(sns.distplot, data, **dist_kwargs)
    return g

# def raw_rt(df)
# f, ax = plt.subplots(ncols=2, nrows=3,sharex=True, figsize=(16,9))
# ax = ax.flatten()
# for i, subj in enumerate([f'LE{x}' for x in range(82,88)]):
#     vec_toplot = np.concatenate([df.loc[df.subjid==subj, 'sound_len'].values+300, 1000*np.concatenate(df.loc[df.subjid==subj,'fb'].values)])
#     sns.distplot(vec_toplot[(vec_toplot<=700)], kde=False, ax=ax[i])
#     ax[i].set_title(subj)
#     ax[i].axvline(300, c='k', ls=':')

# plt.xlabel('Fixation + RT')    
# plt.show()



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
        return (
            tab.index.values,tab['mean'].values, 
            np.array([tab.low_ci.values, tab.high_ci.values]), 
            (xspace, y_fit),
            {'sens':s, 'bias':b, 'RL':RL, 'LL':LL}
        )
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


# pending: correcting kiani
def correcting_kiani(hit, rresp, com, ev, **kwargs):
    '''
    this docu sux
    hit: hithistory: ....
    now we plot choices because stim, without taking into account reward/correct because of unfair task
    '''
    # pal = sns.color_palette()
    tmp = pd.DataFrame(np.array([hit,rresp,com,ev]).T, columns=['hit','R_response','com','ev'])
    tmp['init_choice'] = tmp['R_response']
    tmp.loc[tmp.com==True, 'init_choice'] = (tmp.loc[tmp.com==True, 'init_choice'].values-1)**2
    # transform to 0_1 space
    tmp.loc[tmp.ev<0, 'init_choice']= (tmp.loc[tmp.ev<0, 'init_choice'].values-1)**2
    tmp.loc[tmp.ev<0, 'R_response']= (tmp.loc[tmp.ev<0, 'R_response'].values-1)**2
    
    tmp.loc[:, 'ev']=tmp.loc[:, 'ev'].abs()
    
    counts_ac, nobs_ac, mean_ac,counts_ae, nobs_ae, mean_ae =[], [], [], [], [], []
    ## adapt to noenv sessions, because there are only 4
    evrange = np.linspace(0,1,6)
    for i in range(5):
        counts_ac += [tmp.loc[(tmp.ev>=evrange[i])&(tmp.ev<=evrange[i+1]), 'R_response'].sum()]
        nobs_ac += [tmp.loc[(tmp.ev>=evrange[i])&(tmp.ev<=evrange[i+1]), 'R_response'].count()]
        mean_ac += [tmp.loc[(tmp.ev>=evrange[i])&(tmp.ev<=evrange[i+1]), 'R_response'].mean()]
        counts_ae += [tmp.loc[(tmp.ev>=evrange[i])&(tmp.ev<=evrange[i+1]), 'init_choice'].sum()]
        nobs_ae += [tmp.loc[(tmp.ev>=evrange[i])&(tmp.ev<=evrange[i+1]), 'init_choice'].count()]
        mean_ae += [tmp.loc[(tmp.ev>=evrange[i])&(tmp.ev<=evrange[i+1]), 'init_choice'].mean()]
    
    ci_l_ac, ci_u_ac = proportion_confint(counts_ac, nobs_ac, method='beta')
    ci_l_ae, ci_u_ae = proportion_confint(counts_ae, nobs_ae, method='beta')
    
    for item in [mean_ac, ci_l_ac, ci_u_ac, mean_ae,ci_l_ae, ci_u_ae]:
        item=np.array(item)
    
    xtickpos = (evrange[:-1]+ evrange[1:])/2
    plt.errorbar(xtickpos, mean_ac, yerr=[mean_ac-ci_l_ac, ci_u_ac-mean_ac], marker='o',markersize=3,capsize=4, color='r', label='with com')
    plt.errorbar(xtickpos, mean_ae, yerr=[mean_ae-ci_l_ae, ci_u_ae-mean_ae], marker='o',markersize=3,capsize=4, color='k', label='init choice')
    plt.legend
    plt.ylim([.45, 1.05])

# pending: pcom kiani

# transition com vs transition regular
def com_heatmap(x, y, com, flip=False, annotate=True,predefbins=None,**kwargs):
    '''x: priors; y: av_stim, com_col, Flip (for single matrx.),all calculated from tmp dataframe
    TODO: improve binning option, or let add custom bins
    TODO: add example call'''
    warnings.warn("when used alone (ie single axis obj) by default sns y-flips it")
    tmp = pd.DataFrame(np.array([x, y, com]).T, columns=['prior', 'stim', 'com'])
    
    # make bins    
    tmp['binned_prior'] = np.nan
    maxedge_prior = tmp.prior.abs().max()
    if predefbins is None:
        predefbinsflag=False
        bins = np.linspace(-maxedge_prior-0.01,maxedge_prior+0.01, 8)
    else:
        predefbinsflag=True
        bins = predefbins[0]
    tmp.loc[:,'binned_prior'], priorbins = pd.cut(tmp.prior, bins=bins, retbins=True, labels=np.arange(7))
    tmp.loc[:, 'binned_prior']= tmp.loc[:, 'binned_prior'].astype(int)
    priorlabels = [round((priorbins[i]+priorbins[i+1])/2, 2) for i in range(7)]


    
    tmp['binned_stim'] = np.nan
    maxedge_stim = tmp.stim.abs().max()
    if not predefbinsflag:
        bins=np.linspace(-maxedge_stim-0.01,maxedge_stim+0.01, 8)
    else:
        bins=predefbins[1]
    tmp.loc[:,'binned_stim'], stimbins = pd.cut(tmp.stim, bins=bins, retbins=True, labels=np.arange(7))
    tmp.loc[:, 'binned_stim']= tmp.loc[:, 'binned_stim'].astype(int)
    stimlabels = [round((stimbins[i]+stimbins[i+1])/2, 2) for i in range(7)]
    
    #populate matrices
    matrix = np.zeros((7,7))
    nmat = np.zeros((7,7))
    plain_com_mat=np.zeros((7,7))
    for i in range(7):
        switch = tmp.loc[(tmp.com==True)&(tmp.binned_stim==i)].groupby('binned_prior')['binned_prior'].count()
        nobs = switch + tmp.loc[(tmp.com==False)&(tmp.binned_stim==i)].groupby('binned_prior')['binned_prior'].count()
        # fill where there are no CoM (instead it will be nan)
        nobs.loc[nobs.isna()]= tmp.loc[(tmp.com==False)&(tmp.binned_stim==i)].groupby('binned_prior')['binned_prior'].count().loc[nobs.isna()] # index should be the same!
        crow = (switch/nobs) #.values
        nmat[i,nobs.index.astype(int)]=nobs
        plain_com_mat[i,switch.index.astype(int)]=switch.values
        matrix[i,crow.index.astype(int)]=crow
    
    if not kwargs:
        kwargs = dict(cmap='viridis', fmt='.0f')
    if flip: # this shit is not workin # this works in custom subplot grid
        # just retrieve ax and ax.invert_yaxis
        # matrix = np.flipud(matrix)
        # nmat = np.flipud(nmat)
        # stimlabels=np.flip(stimlabels)
        if annotate:
            g = (sns.heatmap(np.flipud(matrix), annot=np.flipud(nmat), **kwargs)
            .set(xlabel='prior', ylabel='average stim', xticklabels=priorlabels, yticklabels= np.flip(stimlabels)))
        else:
            g = (sns.heatmap(np.flipud(matrix), annot=None, **kwargs)
            .set(xlabel='prior', ylabel='average stim', xticklabels=priorlabels, yticklabels= np.flip(stimlabels)))
    else:
        if annotate:
            g = (sns.heatmap(matrix, annot=nmat, **kwargs)
            .set(xlabel='prior', ylabel='average stim', xticklabels=priorlabels, yticklabels= stimlabels))
        else:
            g = (sns.heatmap(matrix, annot=None, **kwargs)
            .set(xlabel='prior', ylabel='average stim', xticklabels=priorlabels, yticklabels= stimlabels))

    return g
        


# com_kde2D
# import scipy.stats as stats
# g = sns.jointplot(test3.loc[test3.CoM_sugg==True,'allprior_single'], test3.loc[test3.CoM_sugg==True,'current_stimuli'],kind='reg', color=current_palette[1])
# g.set_axis_labels('prior(lateral+transition*(resp-1)+aftereffect)', 'weighted current stimulus')
# g.annotate(stats.pearsonr)
# plt.show()

###
# com_kernels # avoid! :D
###

# general com across animals
# f, ax = plt.subplots(ncols=2,figsize=(16,12))

# sns.boxplot(x=('subjid','first'), y=('CoM_sugg', 'mean'), data=tmp, ax=ax[0])
# sns.stripplot(x=('subjid','first'), y=('CoM_sugg', 'mean'),data=tmp, color=".25", linewidth=1, edgecolor='w', jitter=True, ax=ax[0])
# ax[0].set_ylim([-0.001, 0.04])
# ax[0].set_xlabel('subject')
# ax[0].set_ylabel('#CoM across training sessions')

# #2nd
# sns.boxplot(x=('subjid','first'), y=('CoM_sugg', 'sum'), data=tmp, ax=ax[1])
# sns.stripplot(x=('subjid','first'), y=('CoM_sugg', 'sum'),data=tmp, color=".25", linewidth=1, edgecolor='w', jitter=True, ax=ax[1])
# ax[1].set_ylim([-0.5, 20])
# ax[1].set_xlabel('subject')
# ax[1].set_ylabel('#CoM across training sessions')
# plt.show()

# # add grid right (gridspec), convolved comrate through time
# # same with num trials/session

def binned_curve(df, var_toplot, var_tobin, bins, errorbar_kw={}, 
                ax=None, sem_err=True, xpos=None, subplot_kw={},
                legend=True, traces=None, traces_kw={'color':'grey', 'alpha':0.15},
                traces_rolling=0, xoffset=True
                ):
    """ bins a var and plots a var according to those bins
    df: dataframe
    var_toplot: str, col in df
    var_tobin: str. col in df,
    bins: edges to bin,
    ax: if provided it plots there
    sem: if false it uses statsmodels proportion confint beta,
    xpos: position to plot in x-ax, else it will use bin index comming fromn groupby
        none= use index
        int/float = number to scale index
        list/array = x-vec array (to plot)
    traces: str, display traces grouping by whatever column
    returns ax

    example call:
    f, g = plt.subplots(figsize=(12,6))
        for filt in ['feedback', 'noenv']:
            g = plotting.binned_curve(df[(df.sessid.str.contains(filt))&(df.hithistory>=0)], 'hithistory', 'sound_len',
                                np.arange(0,400,21), sem_err=True, xpos=20, errorbar_kw={'label':filt}, ax=g) # matplotlib is so smart
                                                                                                            # that keeps rotating color on its own
        g.set_xlabel('RT')
        g.set_ylabel('accu')
        plt.show()
    """
    mdf = df.copy()
    mdf['tmp_bin'] = pd.cut(mdf[var_tobin], bins, include_lowest=True, labels=False)


    if ax is None:
        f, ax = plt.subplots(**subplot_kw)


    # big picture
    if sem_err:
        errfun = sem
    else:
        errfun = groupby_binom_ci
    tmp = mdf.groupby('tmp_bin')[var_toplot].agg(m='mean', e=errfun)

    print(f'attempting to plot {tmp.shape} shaped grouped df')

    if isinstance(xoffset, (int, float)):
        xoffsetval = xoffset
        xoffset = False # flag removed
    elif isinstance(xoffset, bool):
        if not xoffset:
            xoffsetval = 0
        
    
    if xpos is None:
        xpos_plot=tmp.index  
        if xoffset:
            try:
                xpos_plot += (tmp.index[1]-tmp.index[0])/2
            except:
                print(f'could not add offsest, is this index numeric?\n {tmp.index.head()}')
        else:
            xpos_plot += xoffsetval
    elif isinstance(xpos, (list, np.ndarray)): # beware if trying to plot empty data-bin 
        xpos_plot = np.array(xpos)+xoffsetval
        if xoffset:
            xpos_plot+= (xpos_plot[1]-xpos_plot[0])/2
    elif isinstance(xpos, (int, float)):
        if xoffset:
            xpos_plot+= (xpos_plot[1]-xpos_plot[0])/2
        else:
            xpos_plot=tmp.index * xpos + xoffsetval
    elif isinstance(xpos, (types.FunctionType, types.LambdaType)):
        xpos_plot = xpos(tmp.index)

    if sem_err:
        yerrtoplot = tmp['e']
    else:
        yerrtoplot = [
            tmp['e'].apply(lambda x: x[0]),
            tmp['e'].apply(lambda x: x[1])
        ]

    if 'label' not in errorbar_kw.keys():
        errorbar_kw['label'] = var_toplot
    
    ax.errorbar(
        xpos_plot,
        tmp['m'],
        yerr=yerrtoplot,
        **errorbar_kw
    )
    if legend:
        ax.legend()



    # traces section # may malfunction if using weird xpos
    if traces is not None:
        traces_tmp = mdf.groupby([traces, 'tmp_bin'])[var_toplot].mean()
        for tr in mdf[traces].unique():
            if xpos is not None:
                if isinstance(xpos, (int,float)):
                    if not xoffset:
                        xpos_tr = traces_tmp[tr].index * xpos + xoffsetval
                    else:
                        xpos_tr = traces_tmp[tr].index * xpos
                        xpos_tr += (xpos_tr[1]-xpos_tr[0])/2
                else:
                    raise NotImplementedError('traces just work with xpos=None/float/int, offsetval')

            if traces_rolling: #needs debug
                y_traces = traces_tmp[tr].rolling(traces_rolling, min_periods=1).mean().values
            else:
                y_traces = traces_tmp[tr].values
            ax.plot(
                xpos_tr,
                y_traces,
                **traces_kw
            )


    
    return ax