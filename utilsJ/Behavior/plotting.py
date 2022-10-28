# utils for plotting behavior
# this should be renamed to plotting/figures
from scipy.stats import norm, sem
from scipy.optimize import minimize
from scipy import interpolate
from statsmodels.stats.proportion import proportion_confint
from utilsJ.regularimports import groupby_binom_ci
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import types
# import swifter
import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from scipy.interpolate import interp1d

# from utilsJ.Models import alex_bayes as ab

# useless, use dir(plotting) instead
# def help():
#     """should print at least available functions"""
#     print("available methods:")
#     print("distbysubj: grid of distributions")
#     print("psych_curve")
#     print("correcting_kiani")
#     print("com_heatmap")
#     print("binned curve: curve of means and err of y-var binning by x-var")


# class cplot(): # cplot stands for custom plot
# def __init__():


def distbysubj(
    df,
    data, # x?
    by, #
    grid_kwargs={},
    dist_kwargs={},
    override_defaults=False
):
    """
    returns facet (g) so user can use extra methods
        ie: .set_axis_labels('x','y')
            . add_legend
        data: what to plot (bin on) ~ str(df.col header)
        by: sorter (ie defines #subplots) ~ str(df.col header)

        returns sns.FacetGrid obj
    """
    if override_defaults:
        def_grid_kw = grid_kwargs
        def_dist_kw = dist_kwargs
    else:
        def_grid_kw = dict(col_wrap=2, hue="CoM_sugg", aspect=2)
        def_grid_kw.update(grid_kwargs)

        def_dist_kw = dict(kde=False, norm_hist=True, bins=np.linspace(0, 400, 50))
        def_dist_kw.update(dist_kwargs)

    g = sns.FacetGrid(df, col=by, **def_grid_kw)
    g = g.map(sns.distplot, data, **def_dist_kw)
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
    """sigmoid functions for psychometric curves
    fir params is a tupple: (sensitivity, bias, Right Lapse, Left lapse),
    x_data = evidence/coherence whatever
    y_data: hit or miss, right responses whatever"""
    s, b, RL, LL = fit_params
    ypred = RL + (1 - RL - LL) / (
        1 + np.exp(-(s * x_data + b))
    )  # this is harder to interpret
    # replacing function so it is meaningful
    # ypred = RL + (1-RL-LL)/(1+np.exp(-s(x_data-b)))
    return -np.sum(norm.logpdf(y_data, loc=ypred))


def sigmoid_no_lapses(fit_params, x_data, y_data):
    """same without lapses"""
    s, b = fit_params
    ypred = 1 / (1 + np.exp(-(s * x_data + b)))  # this is harder to interpret
    # replacing function so it is meaningful
    # ypred = RL + (1-RL-LL)/(1+np.exp(-s(x_data-b)))
    return -np.sum(norm.logpdf(y_data, loc=ypred))


def raw_psych(y, x, lapses=True):
    """returns params"""
    if lapses:
        loglike = minimize(sigmoid, [1, 1, 0, 0], (x, y))
    else:
        loglike = minimize(sigmoid_no_lapses, [0, 0], (x, y))
    params = loglike["x"]
    return params


def psych_curve(
    target,
    coherence,
    ret_ax=None, # if none returns data
    annot=False,
    xspace=np.linspace(-1, 1, 50),
    kwargs_plot={},
    kwargs_error={},
):
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
                raise ValueError("Nans detected in provided vector")
    if np.unique(target).size > 2:
        raise ValueError("target has >2 unique values (invalids?!)")

    """ not required
    if not (coherence<0).sum(): # 0 in the sum means there are no values under 0
        coherence = coherence * 2 - 1 # (from 01 to -11)
    
    r_resp = np.full(hit.size, np.nan)
    r_resp[np.where(np.logical_or(hit==0, hit==1)==True)[0]] = 0
    r_resp[np.where(np.logical_and(rewside==1, hit==1)==True)[0]] = 1 # right and correct
    r_resp[np.where(np.logical_and(rewside==0, hit==0)==True)[0]] = 1 # left and incorrect
    """

    kwargs_plot_default = {"color": "tab:blue"}
    kwargs_error_default = dict(
                ls="none", marker="o", markersize=3, capsize=4, color="maroon"
            )

    kwargs_plot_default.update(kwargs_plot)
    kwargs_error_default.update(kwargs_error)

    tmp = pd.DataFrame({"target": target, "coh": coherence})
    tab = tmp.groupby("coh")["target"].agg(["mean", "sum", "count"])

    tab["low_ci"], tab["high_ci"] = proportion_confint(
        tab["sum"], tab["count"], method="beta"
    )
    # readapt value to plot errorbars directly
    tab["low_ci"] = tab["mean"] - tab["low_ci"]
    tab["high_ci"] = tab["high_ci"] - tab["mean"]

    # sigmoid fit
    loglike = minimize(sigmoid, [1, 1, 0, 0], (coherence, target))
    s, b, RL, LL = loglike["x"]
    y_fit = RL + (1 - RL - LL) / (1 + np.exp(-(s * xspace + b)))


    if ret_ax is None:
        return (
            tab.index.values,
            tab["mean"].values,
            np.array([tab.low_ci.values, tab.high_ci.values]),
            (xspace, y_fit),
            {"sens": s, "bias": b, "RL": RL, "LL": LL},
        )
        # plot it with plot(*3rd_returnvalue) + errorbar(*1st_retval, yerr=2nd_retval, ls='none')
    else:
        ret_ax.axhline(y=0.5, linestyle=":", c="k")
        ret_ax.axvline(x=0, linestyle=":", c="k")
        ret_ax.plot(xspace, y_fit, **kwargs_plot_default)
        ret_ax.errorbar(
            tab.index,
            tab["mean"].values,
            yerr=np.array([tab.low_ci.values, tab.high_ci.values]),
            **kwargs_error_default,
        )

        if annot:
            ret_ax.annotate(
                f"bias: {round(b,2)}\nsens: {round(s,2)}\nRπ: {round(RL,2)}\nLπ: {round(LL,2)}",
                (tab.index.values.min(), 0.55),
                ha="left",
            )

        ret_ax.set_ylim(-0.05, 1.05)
        return ret_ax


# pending: correcting kiani
def correcting_kiani(hit, rresp, com, ev, **kwargs):
    """
    figure like in resulaj 2009
    hit: hithistory vector; 
    rresp: right response vector,
    com: change of mind vector,
    ev: evidence vector 
    kwargs: plt.errobar kwargs except for label and color! (if passed will make this function crash)
    now we plot choices because stim, without taking into account reward/correct because of unfair task
    """
    def_kws = dict(
        marker="o",
        markersize=3,
        capsize=4)

    def_kws.update(kwargs) # update default kws with those provided in function calll

    # pal = sns.color_palette()
    tmp = pd.DataFrame(
        np.array([hit, rresp, com, ev]).T, columns=["hit", "R_response", "com", "ev"]
    )
    tmp["init_choice"] = tmp["R_response"]
    tmp.loc[tmp.com == True, "init_choice"] = (
        tmp.loc[tmp.com == True, "init_choice"].values - 1
    ) ** 2
    # transform to 0_1 space
    tmp.loc[tmp.ev < 0, "init_choice"] = (
        tmp.loc[tmp.ev < 0, "init_choice"].values - 1
    ) ** 2
    tmp.loc[tmp.ev < 0, "R_response"] = (
        tmp.loc[tmp.ev < 0, "R_response"].values - 1
    ) ** 2

    tmp.loc[:, "ev"] = tmp.loc[:, "ev"].abs()

    counts_ac, nobs_ac, mean_ac, counts_ae, nobs_ae, mean_ae = [], [], [], [], [], []
    ## adapt to noenv sessions, because there are only 4
    evrange = np.linspace(0, 1, 6)
    for i in range(5):
        counts_ac += [
            tmp.loc[
                (tmp.ev >= evrange[i]) & (tmp.ev <= evrange[i + 1]), "R_response"
            ].sum()
        ]
        nobs_ac += [
            tmp.loc[
                (tmp.ev >= evrange[i]) & (tmp.ev <= evrange[i + 1]), "R_response"
            ].count()
        ]
        mean_ac += [
            tmp.loc[
                (tmp.ev >= evrange[i]) & (tmp.ev <= evrange[i + 1]), "R_response"
            ].mean()
        ]
        counts_ae += [
            tmp.loc[
                (tmp.ev >= evrange[i]) & (tmp.ev <= evrange[i + 1]), "init_choice"
            ].sum()
        ]
        nobs_ae += [
            tmp.loc[
                (tmp.ev >= evrange[i]) & (tmp.ev <= evrange[i + 1]), "init_choice"
            ].count()
        ]
        mean_ae += [
            tmp.loc[
                (tmp.ev >= evrange[i]) & (tmp.ev <= evrange[i + 1]), "init_choice"
            ].mean()
        ]

    ci_l_ac, ci_u_ac = proportion_confint(counts_ac, nobs_ac, method="beta")
    ci_l_ae, ci_u_ae = proportion_confint(counts_ae, nobs_ae, method="beta")

    for item in [mean_ac, ci_l_ac, ci_u_ac, mean_ae, ci_l_ae, ci_u_ae]:
        item = np.array(item)

    xtickpos = (evrange[:-1] + evrange[1:]) / 2
    plt.errorbar(
        xtickpos,
        mean_ac,
        yerr=[mean_ac - ci_l_ac, ci_u_ac - mean_ac],
        color="r",
        label="with com",
        **def_kws
    )
    plt.errorbar(
        xtickpos,
        mean_ae,
        yerr=[mean_ae - ci_l_ae, ci_u_ae - mean_ae],
        color="k",
        label="init choice",
        **def_kws
    )
    plt.legend
    plt.ylim([0.45, 1.05])


# pending: pcom kiani

# transition com vs transition regular
def com_heatmap(
    x, y, com, flip=False, annotate=True, predefbins=None, return_mat=False, folding=False, annotate_div=1,**kwargs
):
    """x: priors; y: av_stim, com_col, Flip (for single matrx.),all calculated from tmp dataframe
    TODO: improve binning option, or let add custom bins
    TODO: add example call
    g = sns.FacetGrid(df[df.special_trial==0].dropna(subset=['avtrapz', 'rtbins']), col='rtbins', col_wrap=3, height=5, sharex=False)
    g = g.map(plotting.com_heatmap, 'norm_prior','avtrapz','CoM_sugg', vmax=.15).set_axis_labels('prior', 'average stim')

    annotate_div= number to divide
    """
    warnings.warn("when used alone (ie single axis obj) by default sns y-flips it")
    tmp = pd.DataFrame(np.array([x, y, com]).T, columns=["prior", "stim", "com"])

    # make bins
    tmp["binned_prior"] = np.nan
    maxedge_prior = tmp.prior.abs().max()
    if predefbins is None:
        predefbinsflag = False
        bins = np.linspace(-maxedge_prior - 0.01, maxedge_prior + 0.01, 8)
    else:
        predefbinsflag = True
        bins = np.asarray(predefbins[0])


    tmp.loc[:, "binned_prior"], priorbins = pd.cut(
        tmp.prior, bins=bins, retbins=True, labels=np.arange(bins.size-1), include_lowest=True
    )
    

    tmp.loc[:, "binned_prior"] = tmp.loc[:, "binned_prior"].astype(int)
    priorlabels = [round((priorbins[i] + priorbins[i + 1]) / 2, 2) for i in range(bins.size-1)]

    tmp["binned_stim"] = np.nan
    maxedge_stim = tmp.stim.abs().max()
    if not predefbinsflag:
        bins = np.linspace(-maxedge_stim - 0.01, maxedge_stim + 0.01, 8)
    else:
        bins = np.asarray(predefbins[1])
    tmp.loc[:, "binned_stim"], stimbins = pd.cut(
        tmp.stim, bins=bins, retbins=True, labels=np.arange(bins.size-1), include_lowest=True
    )
    tmp.loc[:, "binned_stim"] = tmp.loc[:, "binned_stim"].astype(int)
    stimlabels = [round((stimbins[i] + stimbins[i + 1]) / 2, 2) for i in range(bins.size-1)]

    if len(stimlabels)!=len(priorlabels):
        warnings.warn('WARNING, this was never tested with "non-squared matrices"')

    # populate matrices
    matrix = np.zeros((len(stimlabels), len(priorlabels)))
    nmat = matrix.copy()
    plain_com_mat = matrix.copy()
    for i in range(len(stimlabels)):
        switch = (
            tmp.loc[(tmp.com == True) & (tmp.binned_stim == i)]
            .groupby("binned_prior")["binned_prior"]
            .count()
        )
        nobs = (
            switch
            + tmp.loc[(tmp.com == False) & (tmp.binned_stim == i)]
            .groupby("binned_prior")["binned_prior"]
            .count()
        )
        # fill where there are no CoM (instead it will be nan)
        nobs.loc[nobs.isna()] = (
            tmp.loc[(tmp.com == False) & (tmp.binned_stim == i)]
            .groupby("binned_prior")["binned_prior"]
            .count()
            .loc[nobs.isna()]
        )  # index should be the same!
        crow = switch / nobs  # .values
        nmat[i, nobs.index.astype(int)] = nobs
        plain_com_mat[i, switch.index.astype(int)] = switch.values
        matrix[i, crow.index.astype(int)] = crow


    if folding: # get indexes
        iu = np.triu_indices(len(stimlabels),1)
        il = np.tril_indices(len(stimlabels),-1)
        tmp_nmat = np.fliplr(nmat.copy())
        tmp_nmat[iu] += tmp_nmat[il]
        tmp_nmat[il] = 0
        tmp_ncom = np.fliplr(plain_com_mat.copy())
        tmp_ncom[iu] += tmp_ncom[il]
        tmp_ncom[il] = 0
        plain_com_mat = np.fliplr(tmp_ncom.copy())
        matrix = tmp_ncom/tmp_nmat
        matrix = np.fliplr(matrix)

    if return_mat:
        # matrix is com/obs, nmat is number of observations
        return matrix, nmat

    if isinstance(annotate, str):
        if annotate=='com':
            annotate = True
            annotmat = plain_com_mat/annotate_div
        if annotate=='counts':
            annotate = True
            annotmat = nmat/annotate_div
    else:
        annotmat = nmat/annotate_div
    

    if not kwargs:
        kwargs = dict(cmap="viridis", fmt=".0f")
    if flip:  # this shit is not workin # this works in custom subplot grid
        # just retrieve ax and ax.invert_yaxis
        # matrix = np.flipud(matrix)
        # nmat = np.flipud(nmat)
        # stimlabels=np.flip(stimlabels)
        if annotate:
            g = sns.heatmap(np.flipud(matrix), annot=np.flipud(annotmat), **kwargs).set(
                xlabel="prior",
                ylabel="average stim",
                xticklabels=priorlabels,
                yticklabels=np.flip(stimlabels),
            )
        else:
            g = sns.heatmap(np.flipud(matrix), annot=None, **kwargs).set(
                xlabel="prior",
                ylabel="average stim",
                xticklabels=priorlabels,
                yticklabels=np.flip(stimlabels),
            )
    else:
        if annotate:
            g = sns.heatmap(matrix, annot=annotmat, **kwargs).set(
                xlabel="prior",
                ylabel="average stim",
                xticklabels=priorlabels,
                yticklabels=stimlabels,
            )
        else:
            g = sns.heatmap(matrix, annot=None, **kwargs).set(
                xlabel="prior",
                ylabel="average stim",
                xticklabels=priorlabels,
                yticklabels=stimlabels,
            )

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


def binned_curve(
    df,
    var_toplot,
    var_tobin,
    bins,
    errorbar_kw={},
    ax=None,
    sem_err=True,
    xpos=None,
    subplot_kw={},
    legend=True,
    traces=None,
    traces_kw={},
    traces_rolling=0,
    xoffset=True,
    median=False,
    return_data=False
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
    mdf = df.copy() # memory leak where?
    mdf["tmp_bin"] = pd.cut(mdf[var_tobin], bins, include_lowest=True, labels=False)

    traces_default_kws = {"color": "grey", "alpha": 0.15}
    traces_default_kws.update(traces_kw)

    if ax is None and not return_data:
        f, ax = plt.subplots(**subplot_kw)

    # big picture
    if sem_err:
        errfun = sem
    else:
        errfun = groupby_binom_ci
    if median:
        tmp = mdf.groupby("tmp_bin")[var_toplot].agg(m="median", e=errfun)
    else:
        tmp = mdf.groupby("tmp_bin")[var_toplot].agg(m="mean", e=errfun)

    # print(f'attempting to plot {tmp.shape} shaped grouped df')

    if isinstance(xoffset, (int, float)):
        xoffsetval = xoffset
        xoffset = False  # flag removed
    elif isinstance(xoffset, bool):
        if not xoffset:
            xoffsetval = 0

    if xpos is None:
        xpos_plot = tmp.index
        if xoffset:
            try:
                xpos_plot += (tmp.index[1] - tmp.index[0]) / 2
            except:
                print(
                    f"could not add offsest, is this index numeric?\n {tmp.index.head()}"
                )
        else:
            xpos_plot += xoffsetval
    elif isinstance(
        xpos, (list, np.ndarray)
    ):  # beware if trying to plot empty data-bin
        xpos_plot = np.array(xpos) + xoffsetval
        if xoffset:
            xpos_plot += (xpos_plot[1] - xpos_plot[0]) / 2
    elif isinstance(xpos, (int, float)):
        if xoffset:
            xpos_plot += (xpos_plot[1] - xpos_plot[0]) / 2
        else:
            xpos_plot = tmp.index * xpos + xoffsetval
    elif isinstance(xpos, (types.FunctionType, types.LambdaType)):
        xpos_plot = xpos(tmp.index)

    if sem_err:
        yerrtoplot = tmp["e"]
    else:
        yerrtoplot = [tmp["e"].apply(lambda x: x[0]), tmp["e"].apply(lambda x: x[1])]

    if "label" not in errorbar_kw.keys():
        errorbar_kw["label"] = var_toplot

    if return_data:
        # returns x, y and errors
        return xpos_plot, tmp["m"], yerrtoplot
    
    ax.errorbar(xpos_plot, tmp["m"], yerr=yerrtoplot, **errorbar_kw)
    if legend:
        ax.legend()

    # traces section # may malfunction if using weird xpos
    if traces is not None:
        traces_tmp = mdf.groupby([traces, "tmp_bin"])[var_toplot].mean()
        for tr in mdf[traces].unique():
            if xpos is not None:
                if isinstance(xpos, (int, float)):
                    if not xoffset:
                        xpos_tr = traces_tmp[tr].index * xpos + xoffsetval
                    else:
                        xpos_tr = traces_tmp[tr].index * xpos
                        xpos_tr += (xpos_tr[1] - xpos_tr[0]) / 2
                else:
                    raise NotImplementedError(
                        "traces just work with xpos=None/float/int, offsetval"
                    )

            if traces_rolling:  # needs debug
                y_traces = (
                    traces_tmp[tr].rolling(traces_rolling, min_periods=1).mean().values
                )
            else:
                y_traces = traces_tmp[tr].values
            ax.plot(xpos_tr, y_traces, **traces_default_kws)

    return ax


def interpolapply(
    row,
    stamps="trajectory_stamps",
    ts_fix_onset="fix_onset_dt",
    trajectory="trajectory_y",
    resp_side="R_response",
    collapse_sides=False,
    interpolatespace=np.linspace(-700000, 1000000, 1701),
    fixation_us=300000,  # from fixation onset (0) to startsound (300) to longest possible considered RT  (+400ms) to longest possible considered motor time (+1s)
    align="action",
    interp_extend=False,
    discarded_tstamp=0,
):  # we can speed up below funcction for trajectories
    # for ii,i in enumerate(idx_dic[b]):

    # think about discarding first few frames from trajectory_y because they are noisy (due to camera delay they likely belong to previous state)
    x_vec = []
    y_vec = []
    try:
        x_vec = row[stamps] - np.datetime64(
            row[ts_fix_onset]
        )  # .astype(float) # aligned to fixation onset (0) using timestamps
        # by def 0 aligned to fixation
        if align == "sound":
            x_vec = (x_vec -
                     np.timedelta64(int(fixation_us +
                                        (row["sound_len"] * 10 ** 3)),
                                    "us")).astype(float)
            # x_vec = x_vec.astype(float)
        elif align == "action":
            # next line crashes when there's a (accidental)silent trial and sound_len is np.nan
            x_vec = (
                x_vec
                - int(fixation_us + (row["sound_len"] * 1e3))
            ).astype(
                float
            )  # shift it in order to align 0 with motor-response/action onset
            # x_vec = (x_vec - np.timedelta64(int((row['sound_len'] * 10**3)), 'us')).astype(float) # shift it in order to align 0 with motor-response/action onset
        elif align == "response":
            x_vec = (
                x_vec
                - np.timedelta64(
                    int(
                        fixation_us
                        + (row["sound_len"] * 10 ** 3)
                        + (row["resp_len"] * 10 ** 6)
                    ),
                    "us",
                )
            ).astype(float)
        else:
            x_vec = x_vec.astype(float)

        x_vec = x_vec[discarded_tstamp:]

        # else it is aliggned with
        if isinstance(trajectory, tuple):  # when is this shit being executed
            y_vec = row[trajectory[0]][
                :, trajectory[1]
            ]  # this wont work if there are several columns called like trajectory[0] in the original df
        else:  # is a column name
            y_vec = row[trajectory]
        if collapse_sides:
            if (
                row[resp_side] == 0
            ):  # do we want to collapse here? # should be filtered so yes
                y_vec = y_vec * -1
        if interp_extend:
            f = interpolate.interp1d(
                x_vec, y_vec, bounds_error=False, fill_value=(y_vec[0], y_vec[-1])
            )  # without fill_value it fills with nan
        else:
            f = interpolate.interp1d(
                x_vec, y_vec, bounds_error=False
            )  # should fill everything else with NaNs
        out = f(interpolatespace)
        return out
    except Exception as e:
        # print(x_vec.shape)
        # raise e
        # print(e) # muting this because it generates a lot of spam
        return np.array([np.nan] * interpolatespace.size)


def trajectory_thr(
    df,
    bincol,
    bins,
    thr=40,
    trajectory="trajectory_y",
    stamps="trajectory_stamps",
    ax=None,
    fpsmin=29,
    fixation_us=300000,
    collapse_sides=False,
    return_trash=False,
    interpolatespace=np.linspace(-700000, 1000000, 1700),
    zeropos_interp=700,
    fixation_delay_offset=0,
    error_kwargs={"ls": "none"},
    ax_traj=None,
    traj_kws={},
    ts_fix_onset="fix_onset_dt",
    align="action",
    interp_extend=False,
    discarded_tstamp=0,
    cmap=None,
    rollingmeanwindow=0,
    bintype="edges",
    xpoints=None,
    raiseerrors=False,
):
    """
    This changed a lot!, review default plots
    Exclude invalids
    atm this will only retrieve data, not plotting if ax=None 
    # if a single bin, both edges must be provided
    fpsmin: minimum fps to consider trajectories
    align ='action' or 'sound'
    if duplicated indexes in df this wont work
    bins= if single element it does not use edges but == that value
    fixation_delay_offset = 300-fixation state lenght (ie new state matrix should use 80)
    trajectory: str (colname) where trajectory is. tuple (colname, which col to use in the array) for cols containing arrays.
    discarded_tstamp: number of tstamps to ommit (for 1st and second derivatives!) # old
    thr (threshold should be able to accept as many thr like bins)
    # rollingmean for noisy traj (d1 and d2)
    bintype='edges', 'categorical', 'dfmask'
    # """

    if bintype not in ["edges", "categorical", "dfmask"]:
        raise ValueError('bintype can take values: "edges", "categorical" and "dfmask"')

    categorical_bins = False
    if bintype != "edges":
        categorical_bins = True
    if align not in ["action", "sound", "response"]:
        raise ValueError('align must be "action","sound" or "response"')
    if (fixation_us != 300000) or (fixation_delay_offset != 0):
        print(
            "fixation and delay offset should be adressed and you should avoid tweaking defaults"
        )

    if (df.index.value_counts() > 1).sum():
        raise IndexError(
            "input dataframe contains duplicate index entries hence this function would not work propperly"
        )

    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
        # cmap overrides color in kwrgs
        if traj_kws is not None:
            traj_kws.pop("c", None)
            traj_kws.pop("color", None)
        if error_kwargs is not None:
            error_kwargs.pop("c", None)
            error_kwargs.pop("color", None)

    matrix_dic = {}
    idx_dic = {}

    # errorplot to threshold!
    if xpoints is None:
        if bintype == "edges":
            xpoints = (bins[:-1] + bins[1:]) / 2
        elif bintype == "categorical":
            xpoints = bins
        else:
            try:  # attempt converting them to floats?!
                xpoints = [float(x) for x in bins.keys()]
            except Exception as e:
                if raiseerrors:
                    raise e
                xpoints = np.arange(len(bins.keys()))
    y_points = []
    y_err = []

    test = df.loc[df.framerate >= fpsmin]

    if bintype == "dfmask":
        bkeys = list(bins.keys())
        niters = len(bkeys)
    elif bintype == "categorical":
        niters = len(bins)
    elif bintype == "edges":
        niters = len(bins) - 1  # we iterate 1 less because of edges!

    for b in range(niters):  # if a single bin, both edges must be provided
        if isinstance(thr, (list, tuple, np.ndarray)):
            cthr = thr[b]  # beware if list passes crashes
        else:
            cthr = thr

        if bintype == "dfmask":
            idx_dic[b] = test.loc[bins[bkeys[b]]].index.values
        elif (len(bins) > 1) & (not categorical_bins):
            idx_dic[b] = test.loc[
                (test[bincol] > bins[b]) & (test[bincol] < bins[b + 1])
            ].index.values
        else:
            idx_dic[b] = test.loc[(test[bincol] == bins[b])].index.values
        matrix_dic[b] = np.zeros((idx_dic[b].size, interpolatespace.size))

        # if collapse_sides:
        #    print('collapsing sides!')
        arrays = (
            test.loc[idx_dic[b]]
            #.swifter.progress_bar(False) # swifter removed
            .apply(
                lambda x: interpolapply(
                    x,
                    collapse_sides=collapse_sides,
                    interpolatespace=interpolatespace,
                    align=align,
                    interp_extend=interp_extend,
                    trajectory=trajectory,
                    discarded_tstamp=discarded_tstamp
                ),
                axis=1,
            )
            .values
        )

        if arrays.size > 0:
            matrix_dic[b] = np.concatenate(arrays).reshape(-1, interpolatespace.size)

        tmp_mat = matrix_dic[b][:, zeropos_interp:]
        # wont iterate anymore
        if (cthr > 0) or collapse_sides:
            r, c = np.where(tmp_mat > cthr)
        else:
            r, c = np.where(tmp_mat < cthr)

        _, idxes = np.unique(r, return_index=True)
        y_point = np.median(c[idxes])
        y_points += [y_point]
        y_err += [sem(c[idxes], nan_policy="omit")]

        extra_kw = {}

        # plot section
        if ax_traj is not None:  # original stuff
            if cmap is not None:
                traj_kws.pop("color", None)
                traj_kws["color"] = cmap(b / (niters-1))
            if "label" not in traj_kws.keys():
                if bintype == "categorical":
                    extra_kw["label"] = f"{bincol}={round(bins[b],2)}"
                elif bintype == "dfmask":
                    extra_kw["label"] = f"{bincol}={bkeys[b]}"
                else:  # edges
                    extra_kw[
                        "label"
                    ] = f"{round(bins[b],2)}<{bincol}<{round(bins[b+1],2)}"
            if rollingmeanwindow:
                ytoplot = (
                    pd.Series(np.nanmedian(matrix_dic[b], axis=0))
                    .rolling(rollingmeanwindow, min_periods=1)
                    .mean()
                    .values
                )
            else:
                ytoplot = np.nanmedian(matrix_dic[b], axis=0)
            ax_traj.plot((interpolatespace) / 1000, ytoplot, **traj_kws, **extra_kw)
            # trigger legend outside of the function

    y_points = np.array(y_points)
    y_err = np.array(y_err)

    if (ax is not None) & return_trash:
        if cmap is not None:
            extra_kw = {}
            for i in range(len(xpoints)):
                if "label" not in error_kwargs.keys():
                    if bintype == "categorical":
                        extra_kw["label"] = f"{bincol}={round(bins[i],2)}"
                    elif bintype == "dfmask":
                        extra_kw["label"] = f"{bincol}={bkeys[b]}"
                    else:
                        extra_kw[
                            "label"
                        ] = f"{round(bins[i],2)}<{bincol}<{round(bins[i+1],2)}"

                ax.errorbar(
                    xpoints[i],
                    y_points[i] + fixation_delay_offset,
                    yerr=y_err[i],
                    **error_kwargs,
                    color=cmap(i / (niters-1)),
                    **extra_kw,
                )
        else:
            ax.errorbar(
                xpoints, y_points + fixation_delay_offset, yerr=y_err, **error_kwargs
            )  # add 80ms offset for those new state machine (in other words, this assumes that fixation = 300000us so it takes extra 80ms to reach threshold)

        return xpoints, y_points + fixation_delay_offset, y_err, matrix_dic, idx_dic
    elif ax is not None:
        if cmap is not None:
            for i in range(len(xpoints)):  ##TODO remove duplicated code
                if "label" not in error_kwargs.keys():
                    if bintype == "categorical":
                        extra_kw["label"] = f"{bincol}={round(bins[i],2)}"
                    elif bintype == "dfmask":
                        extra_kw["label"] = f"{bincol}={bkeys[b]}"
                    else:
                        extra_kw[
                            "label"
                        ] = f"{round(bins[i],2)}<{bincol}<{round(bins[i+1],2)}"

                ax.errorbar(
                    xpoints[i],
                    y_points[i] + fixation_delay_offset,
                    yerr=y_err[i],
                    **error_kwargs,
                    color=cmap(i / (niters-1)),
                    **extra_kw,
                )
        else:
            ax.errorbar(
                xpoints, y_points + fixation_delay_offset, yerr=y_err, **error_kwargs
            )  # add 80ms offset for those new state machine (in other words, this assumes that fixation = 300000us so it takes extra 80ms to reach threshold)
        return ax

    elif not return_trash:
        return xpoints, y_points + fixation_delay_offset, y_err
    else:  # with matrix dic we can compute median trajectories etc. for the other plot
        return xpoints, y_points + fixation_delay_offset, y_err, matrix_dic, idx_dic


def colored_line(x, y, z=None, ax=None, linewidth=3, MAP="viridis"):
    # this uses pcolormesh to make interpolated rectangles
    # https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    xl = len(x)
    [xs, ys, zs] = [np.zeros((xl, 2)), np.zeros((xl, 2)), np.zeros((xl, 2))]

    # z is the line length drawn or a list of vals to be plotted
    if z == None:
        z = [0]

    for i in range(xl - 1):
        # make a vector to thicken our line points
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        perp = np.array([-dy, dx])
        unit_perp = (perp / np.linalg.norm(perp)) * linewidth

        # need to make 4 points for quadrilateral
        xs[i] = [x[i], x[i] + unit_perp[0]]
        ys[i] = [y[i], y[i] + unit_perp[1]]
        xs[i + 1] = [x[i + 1], x[i + 1] + unit_perp[0]]
        ys[i + 1] = [y[i + 1], y[i + 1] + unit_perp[1]]

        if len(z) == i + 1:
            z.append(z[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        # set z values
        zs[i] = [z[i], z[i]]
        zs[i + 1] = [z[i + 1], z[i + 1]]

    if ax is None:
        fig, ax = plt.subplots()
    cm = plt.get_cmap(MAP)
    ax.pcolormesh(xs, ys, zs, shading="gouraud", cmap=cm)


def gradient(row):
    """returns speed, accel and jerk # and time incr"""
    try:
        t = row["trajectory_stamps"].astype(float) / 1000
        dt1 = np.repeat(
            np.diff(row["trajectory_stamps"].astype(float) / 1000), 2
        ).reshape(-1, 2)
        coords = np.c_[row["trajectory_x"], row["trajectory_y"]]
        coords1 = np.diff(coords, axis=0) / dt1  # veloc
        coords2 = np.diff(coords1, axis=0) / dt1[1:]  # accel
        coords3 = np.diff(coords2, axis=0) / dt1[2:]  # jerk
        return coords1, coords2, coords3, t - t[0]
    except:
        # print(f'exception in index {row.index}') # gets too spammy
        # print(e)
        return np.nan, np.nan, np.nan, np.nan


def gradient_1d(row):
    """same than above but the norm rather than 2D"""
    try:
        t = row["trajectory_stamps"].astype(float) / 1000
        dt1 = np.diff(t)
        coords = np.c_[row["trajectory_x"], row["trajectory_y"]]
        coords1 = (np.diff(coords, axis=0) ** 2).sum(axis=1) ** 0.5 / dt1
        coords2 = np.diff(coords1) / dt1[1:]
        coords3 = np.diff(coords2) / dt1[2:]
        return coords1, coords2, coords3, t - t[0]
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan, np.nan


def gradient_np(row):
    """returns speed accel and jerk by np.gradient"""
    try:
        t = row["trajectory_stamps"].astype(float) / 1000
        # dt1 = np.repeat(np.diff(row['trajectory_stamps'].astype(float)/1000),2).reshape(-1,2)
        t = t - t[0]
        coords = np.c_[row["trajectory_x"], row["trajectory_y"]]
        d1 = np.gradient(coords, t, axis=0)
        d2 = np.gradient(d1, t, axis=0)
        d3 = np.gradient(d2, t, axis=0)
        return d1, d2, d3, t
    except:
        return np.nan, np.nan, np.nan, np.nan


def gradient_np_simul(row):
    """same but no stamps due being ms based"""
    try:
        t = np.arange(row.traj.size)
        d1 = np.gradient(row.traj, t)
        d2 = np.gradient(d1, t)
        d3 = np.gradient(d2, t)
        return d1, d2, d3
    except:
        return np.nan, np.nan, np.nan


def speedaccel_plot(
    cdata,
    binningcol,
    bins,
    savpath=None,
    align="action",
    bintype="edges",
    rollingmeanwindow=30,
    axobj=None,
    trajectory_thr_kw={},
    suptitle=None,
    thresholds=[
        -0.05,  # single item list will mak it crash! [[-0.05], ...]
        [-0.2] * 4 + [0.2] * 4,
        -0.001,  # [-0.001]*4 + [0.001]*4,
        [-0.001] * 4 + [0.001] * 4,
    ],
    return_trash=False,
):
    """cdata = filtered dataframe (by subject, RT, hit...)
    binningcol = str 'name of the col to bin
    bins = bins
    bintype = '"""
    if align == "action":
        interpolatespace = np.linspace(-700000, 1000000, 1700)
        zeropos_interp = 700
    elif align == "sound":
        interpolatespace = np.linspace(-300000, 1400000, 1700)
        zeropos_interp = 300
    else:
        ValueError('align should be either "action" or "sound"')
    # cdata = mdf.loc[(mdf.subjid==subject)&(mdf.sound_len<=bins[b+1]) & (mdf.sound_len>bins[b]) & (mdf.hithistory==1)]
    if axobj is None:
        f, ax = plt.subplots(
            ncols=4,
            nrows=2,
            figsize=(16, 10),
            sharex="col",
            gridspec_kw=dict(width_ratios=[1, 4, 1, 4]),
        )
    else:
        ax = axobj

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[0][0],
        ax_traj=ax[0][1],
        bintype=bintype,
        error_kwargs={"capsize": 2, "marker": "o"},
        cmap="viridis",
        trajectory=("traj_d1", 0),
        discarded_tstamp=1,
        interp_extend=False,
        thr=thresholds[0],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[0][0].set_ylabel(f"time to thr = {np.unique(thresholds[0])}")
    ax[0][1].set_ylim([-0.2, 0.2])
    ax[0][1].set_title("x speed")
    for i in np.unique(thresholds[0]):
        ax[0, 1].axhline(i, ls=":", color="gray")
    if return_trash:
        dout = {"x_speed": [xpos, ypos, yerr, mat_d, idx_d]}

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[0][2],
        ax_traj=ax[0][3],
        bintype=bintype,
        error_kwargs={"capsize": 2, "marker": "o"},
        cmap="viridis",
        trajectory=("traj_d1", 1),
        discarded_tstamp=1,
        interp_extend=False,
        thr=thresholds[1],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[0][2].set_ylabel(f"time to thr = {np.unique(thresholds[1])}")
    ax[0][3].set_ylim([-0.5, 0.5])
    ax[0][3].set_title("y speed")
    for i in np.unique(thresholds[1]):
        ax[0, 3].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["y_speed"] = [xpos, ypos, yerr, mat_d, idx_d]

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[1][0],
        ax_traj=ax[1][1],
        bintype=bintype,
        error_kwargs={"capsize": 2, "marker": "o"},
        cmap="viridis",
        trajectory=("traj_d2", 0),
        discarded_tstamp=2,
        interp_extend=False,
        thr=thresholds[2],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[1][0].set_ylabel(f"time to thr = {np.unique(thresholds[2])}")
    ax[1][1].set_ylim([-0.005, 0.005])
    ax[1][1].set_title("x accel")
    for i in np.unique(thresholds[2]):
        ax[1, 1].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["x_accel"] = [xpos, ypos, yerr, mat_d, idx_d]

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[1][2],
        ax_traj=ax[1][3],
        bintype=bintype,
        error_kwargs={"capsize": 2, "marker": "o"},
        cmap="viridis",
        trajectory=("traj_d2", 1),
        discarded_tstamp=2,
        interp_extend=False,
        thr=thresholds[3],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[1][2].set_ylabel(f"time to thr = {np.unique(thresholds[3])}")
    ax[1][3].set_ylim([-0.005, 0.005])
    ax[1][3].set_title("y accel")
    for i in np.unique(thresholds[3]):
        ax[1, 3].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["y_accel"] = [xpos, ypos, yerr, mat_d, idx_d]

    # loop later
    for i in [0, 1]:
        for k in [1, 3]:
            ax[i][k].axhline(0, ls=":", c="black")
            # ax[i][k].legend(fancybox=False, frameon=False) # user decides to show it or not

    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle)
    # plt.suptitle(f'{subject}: {bins[b]}< RT <{bins[b+1]}')
    if savpath is not None:
        f.savefig(savpath)
        plt.close()
        if return_trash:
            return dout
    else:
        # print('figure is active, tune it and plt.show()')
        if return_trash:
            return ax, dout
        else:
            return ax


def full_traj_plot(
    cdata,
    binningcol,
    bins,
    savpath=None,
    align="action",
    bintype="edges",
    rollingmeanwindow=30,
    axobj=None,
    trajectory_thr_kw={},
    suptitle=None,
    discarded_tstamps=[0, 0, 1, 1, 2, 2, 3, 3],
    thresholds=[
        -30,
        [-30] * 4 + [30] * 4,
        -0.05,  # single item list will mak it crash!
        [-0.2] * 4 + [0.2] * 4,
        -0.001,  # [-0.001]*4 + [0.001]*4,
        [-0.001] * 4 + [0.001] * 4,
        0.00001,
        [-0.000025] * 4 + [0.000025] * 4,
    ],
    return_trash=False,
):
    """cdata = filtered dataframe (by subject, RT, hit...)
    binningcol = str 'name of the col to bin
    bins = bins
    bintype = 'edges', 'categorical' or 'dfmask
    TODO: add examples of bintypes
    TODO: add repetitions of default thresholds when collape_sides=False, according to number of bins/lines"""
    if align == "action":
        interpolatespace = np.linspace(-700_000, 1_000_000, 1700)
        zeropos_interp = 700
    elif align == "sound":
        interpolatespace = np.linspace(-300_000, 1_400_000, 1700)
        zeropos_interp = 300
    elif align == "response":
        interpolatespace = np.linspace(-1_000_000, 300_000, 1300)
        zeropos_interp = 1000
    else:
        ValueError('align should be either "action", "sound" or "response"')
    # cdata = mdf.loc[(mdf.subjid==subject)&(mdf.sound_len<=bins[b+1]) & (mdf.sound_len>bins[b]) & (mdf.hithistory==1)]
    if axobj is None:
        f, ax = plt.subplots(
            ncols=4,
            nrows=4,
            figsize=(16, 20),
            sharex="col",
            gridspec_kw=dict(width_ratios=[1, 4, 1, 4]),
        )
    else:
        ax = axobj

    # check defaults
    if (
        "error_kwargs" not in trajectory_thr_kw.keys()
    ):  # default that does not duplicate kw entry
        trajectory_thr_kw["error_kwargs"] = {"capsize": 2, "marker": "o"}
    if "cmap" not in trajectory_thr_kw.keys():
        trajectory_thr_kw["cmap"] = "viridis"
    no_collapse_kw = {
        i: trajectory_thr_kw[i] for i in trajectory_thr_kw if i != "collapse_sides"
    }

    if bintype == "edges":
        nlines = bins.size - 1
    elif bintype == "categorical":
        nlines = bins.size
    elif bintype == "dfmask":
        nlines = len(bins.keys())
    for i, thr in enumerate(thresholds):
        if isinstance(thr, (np.ndarray, list)):
            assert (
                len(thr) == nlines
            ), f"threshold in position {i} does not match the number of bins"

    ##### POSITION
    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[0][0],
        ax_traj=ax[0][1],
        bintype=bintype,
        trajectory="trajectory_x",
        discarded_tstamp=discarded_tstamps[0],
        interp_extend=False,
        thr=thresholds[0],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        collapse_sides=False,
        **no_collapse_kw,
    )
    ax[0][0].set_ylabel(f"time to thr = {np.unique(thresholds[0])}")
    ax[0][1].set_ylim([-50, 5])
    ax[0][1].set_ylabel("x coord")
    for i in np.unique(thresholds[0]):
        ax[0, 1].axhline(i, ls=":", color="gray")
    if return_trash:
        dout = {"x_coord": [xpos, ypos, yerr, mat_d, idx_d]}

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[0][2],
        ax_traj=ax[0][3],
        bintype=bintype,
        trajectory="trajectory_y",
        discarded_tstamp=discarded_tstamps[1],
        interp_extend=False,
        thr=thresholds[1],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[0][2].set_ylabel(f"time to thr = {np.unique(thresholds[1])}")
    ax[1][1].set_ylim([-90, 90])
    ax[0][3].set_ylabel("y coord")
    for i in np.unique(thresholds[1]):
        ax[0, 3].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["y_coord"] = [xpos, ypos, yerr, mat_d, idx_d]

    ##### VELOCITY
    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[1][0],
        ax_traj=ax[1][1],
        bintype=bintype,
        trajectory=("traj_d1", 0),
        discarded_tstamp=discarded_tstamps[2],
        interp_extend=False,
        thr=thresholds[2],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        collapse_sides=False,
        **no_collapse_kw,
    )
    ax[1][0].set_ylabel(f"time to thr = {np.unique(thresholds[2])}")
    ax[1][1].set_ylim([-0.25, 0.25])
    ax[1][1].set_ylabel("x speed")
    for i in np.unique(thresholds[2]):
        ax[1, 1].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["x_speed"] = [xpos, ypos, yerr, mat_d, idx_d]

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[1][2],
        ax_traj=ax[1][3],
        bintype=bintype,
        trajectory=("traj_d1", 1),
        discarded_tstamp=discarded_tstamps[3],
        interp_extend=False,
        thr=thresholds[3],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[1][2].set_ylabel(f"time to thr = {np.unique(thresholds[3])}")
    ax[1][3].set_ylim([-0.5, 0.5])
    ax[1][3].set_ylabel("y speed")
    for i in np.unique(thresholds[3]):
        ax[1, 3].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["y_speed"] = [xpos, ypos, yerr, mat_d, idx_d]

    #### ACCELERATION
    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[2][0],
        ax_traj=ax[2][1],
        bintype=bintype,
        trajectory=("traj_d2", 0),
        discarded_tstamp=discarded_tstamps[4],
        interp_extend=False,
        thr=thresholds[4],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        collapse_sides=False,
        **no_collapse_kw,
    )
    ax[2][0].set_ylabel(f"time to thr = {np.unique(thresholds[4])}")
    ax[2][1].set_ylim([-0.005, 0.005])
    ax[2][1].set_ylabel("x accel")
    for i in np.unique(thresholds[4]):
        ax[2, 1].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["x_accel"] = [xpos, ypos, yerr, mat_d, idx_d]

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[2][2],
        ax_traj=ax[2][3],
        bintype=bintype,
        trajectory=("traj_d2", 1),
        discarded_tstamp=discarded_tstamps[5],
        interp_extend=False,
        thr=thresholds[5],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[2][2].set_ylabel(f"time to thr = {np.unique(thresholds[5])}")
    ax[2][3].set_ylim([-0.005, 0.005])
    ax[2][3].set_ylabel("y accel")
    for i in np.unique(thresholds[5]):
        ax[2, 3].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["y_accel"] = [xpos, ypos, yerr, mat_d, idx_d]

    ##### JERK
    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[3][0],
        ax_traj=ax[3][1],
        bintype=bintype,
        trajectory=("traj_d3", 0),
        discarded_tstamp=discarded_tstamps[6],
        interp_extend=False,
        thr=thresholds[6],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        collapse_sides=False,
        **no_collapse_kw,
    )
    ax[3][0].set_ylabel(f"time to thr = {np.unique(thresholds[4])}")
    ax[3][1].set_ylim([-0.00025, 0.00025])
    ax[3][1].set_ylabel("x jerk")
    for i in np.unique(thresholds[6]):
        ax[3, 1].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["x_jerk"] = [xpos, ypos, yerr, mat_d, idx_d]

    xpos, ypos, yerr, mat_d, idx_d = trajectory_thr(
        cdata,
        binningcol,
        bins,
        return_trash=True,
        ax=ax[3][2],
        ax_traj=ax[3][3],
        bintype=bintype,
        trajectory=("traj_d3", 1),
        discarded_tstamp=discarded_tstamps[7],
        interp_extend=False,
        thr=thresholds[7],
        align=align,
        interpolatespace=interpolatespace,
        zeropos_interp=zeropos_interp,
        rollingmeanwindow=rollingmeanwindow,
        **trajectory_thr_kw,
    )
    ax[3][2].set_ylabel(f"time to thr = {np.unique(thresholds[5])}")
    ax[3][3].set_ylim([-0.00025, 0.00025])
    ax[3][3].set_ylabel("y jerk")
    for i in np.unique(thresholds[7]):
        ax[3, 3].axhline(i, ls=":", color="gray")
    if return_trash:
        dout["y_jerk"] = [xpos, ypos, yerr, mat_d, idx_d]

    # loop later
    for i in [0, 1, 2, 3]:
        for k in [1, 3]:
            ax[i][k].axhline(0, ls=":", c="black")
            # ax[i][k].legend(fancybox=False, frameon=False) # user decides to show it or not
    # ax[0,3].axhline(0, ls=':', c='black')

    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle)
    # plt.suptitle(f'{subject}: {bins[b]}< RT <{bins[b+1]}')
    if savpath is not None:
        f.savefig(savpath)
        plt.close()
        if return_trash:
            return dout
    else:
        # print('figure is active, tune it and plt.show()')
        if return_trash:
            return ax, dout
        else:
            return ax


def shuffled_meandiff(inarg):
    # inarg[0]= cmat; inarg[1]=switchindex
    cmat = inarg[0]
    np.random.shuffle(cmat)
    switchindex = inarg[1]
    meandiff = np.nanmean(cmat[:switchindex, :], axis=0) - np.nanmean(
        cmat[switchindex:, :], axis=0
    )
    return meandiff


def surrogate_test(
    mata,
    matb,
    pval=0.005,
    nshuffles=1000,
    return_extra=False,
    nworkers=8,
    pcalc_discard0=0,
    pcalc_discard1=1000,
    lrG=0.01,
    verbose=False,
):
    """shuffling/permutation strat to compare trajectories
    # mata: matrix of dims (ntrajectories, ninterpolatedspace), e.g. coh=1
    # matb: same but corresponding to alternative group (eg. coh=0)
    # pcalc discard: last noisy items to discard when calculating global pband
    # return_extra:also returns meandiff, 95% band and pointwise significance(yes or no)
    # nworkers for shuffling,
    # lrG: step when searching for global band G
    """
    # TODO: cythonized core_loop?
    wholemat = np.concatenate([mata, matb], axis=0)
    switchindex = mata.shape[0]
    collector = []  # beware running out of memory!
    random_states = np.arange(nshuffles)
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        jobs = [
            executor.submit(shuffled_meandiff, [wholemat, switchindex])
            for x in random_states
        ]
        if verbose:
            iterable = tqdm.tqdm_notebook(as_completed(jobs), total=len(random_states))
        else:
            iterable = as_completed(jobs)
        for job in iterable:
            collector += [job.result()]
    # at some point it ould be adapted to this random states

    # slow part should be done already
    allshuffled = np.concatenate(collector).reshape(-1, mata.shape[1])

    shuffleabs = np.abs(allshuffled[:, pcalc_discard0:pcalc_discard1])
    G = 0
    p = 1  #
    for _ in range(int(1 / lrG) + 1):
        G += lrG
        p = (
            np.any((shuffleabs > G), axis=1) * 1
        ).mean()  # difference from 0 which hshould be considered sign
        if p <= pval:
            break

    if not return_extra:  # we are already done
        return G
    else:
        real_groupdiff = np.nanmean(mata, axis=0) - np.nanmean(matb, axis=0)
        pointwise_points = sorted(
            np.where(real_groupdiff > np.percentile(allshuffled, 97.5, axis=0))[
                0
            ].tolist()
            + np.where(real_groupdiff < np.percentile(allshuffled, 2.5, axis=0))[
                0
            ].tolist()
        )
        pointwise_points = np.unique(pointwise_points)
        out = (
            G,
            np.nanmean(allshuffled, axis=0),
            [
                np.percentile(allshuffled, 97.5, axis=0),
                np.percentile(allshuffled, 2.5, axis=0),
            ],
            pointwise_points,
        )
        return out


def get_Mt0te(t0, te):
    Mt0te = np.array(
        [
            [1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
            [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
            [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
            [1, te, te ** 2, te ** 3, te ** 4, te ** 5],
            [0, 1, 2 * te, 3 * te ** 2, 4 * te ** 3, 5 * te ** 4],
            [0, 0, 2, 6 * te, 12 * te ** 2, 20 * te ** 3],
        ]
    )
    return Mt0te


def v_(t):
    return t.reshape(-1, 1) ** np.arange(6)


def get_traj_subplot(row, invert=True, mu=None, allSIGMA=None, sigma=5, dim=1):
    """evaluates whether it was or not (through pandas apply), copypasting 
    preprocess + expand to loglike
    if want to retrieve fit traj, specify mu (list of left and right mu's)"""
    # TODO: retrieve Zk based on hyperparams
    # def get_data(row, dim=0, sigma=None, SIGMA=None, SIGMA_1=None):
    try:
        t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
        T = row.resp_len * 1000 + 50  # total span
        t = (
            t.astype(int) / 1000_000 - 250 - row.sound_len
        )  #  we take 50ms earlier CportOut
        if t.size < 10:
            raise ValueError("custom exception to discard trial")

        fp = np.argmax(t >= 0)
        lastp = np.argmax(t > T)  # first frame after lateral poke in
        tsegment = np.append(t[fp:lastp], T)
        tsegment = np.insert(tsegment, 0, 0)
        if invert and row.R_response == 0:
            pose = np.c_[row.trajectory_x, row.trajectory_y * -1]
        else:
            pose = np.c_[row.trajectory_x, row.trajectory_y]

        boundaries = [[0, 0]] * 6
        for i, traj in enumerate([pose, row.traj_d1, row.traj_d2]):
            f = interp1d(t, traj, axis=0)
            initial = f(0)
            last = f(T)
            boundaries[i] = initial
            boundaries[i + 3] = last
        pose = np.insert(pose[fp:lastp], 0, boundaries[0], axis=0)
        pose = np.append(pose, boundaries[3].reshape(-1, 2), axis=0)
        if mu is None:
            return tsegment, pose
        else:
            M = get_Mt0te(tsegment[0], tsegment[-1])
            M_1 = np.linalg.inv(M)
            vt = v_(tsegment)
            N = vt @ M_1
            SIGMA = allSIGMA[int(row.R_response)]
            SIGMA_1 = np.linalg.pinv(SIGMA)
            # W = sigma**2 * np.identity(N.shape[0]) + N @ SIGMA @ N.T # shape n x n
            L = np.linalg.pinv(SIGMA_1 + N.T @ N / sigma ** 2)  # shape 6,6

            m = L @ (
                N.T @ pose[:, dim].reshape(-1, 1) / sigma ** 2
                + SIGMA_1 @ mu[:, int(row.R_response)].reshape(-1, 1)
            )
            posterior = N @ m
            prior = N @ mu[:, int(row.R_response)]  # is this the prior
            se = np.sqrt(np.diag(N @ L @ N.T))
            # se = np.zeros(N.shape[0] )
            # for i in range(N.shape[0]):
            #     posterior[i] = None # nt @ m
            #     se[i] = N[i].reshape(1,6) @ L @ N[i].reshape(6,1) # same result!
            return tsegment, pose, posterior, se, prior
    except Exception as e:
        raise e
        if mu is None:
            return np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan


def get_traj_subplot_factors(row, invert=True, B=None, allSIGMA=None, sigma=5, dim=1):
    """evaluates whether it was or not (through pandas apply), copypasting 
    preprocess + expand to loglike
    if want to retrieve fit traj, specify mu (list of left and right mu's)"""
    # TODO: retrieve Zk based on hyperparams
    # def get_data(row, dim=0, sigma=None, SIGMA=None, SIGMA_1=None):
    raise NotImplementedError("not yet")
    try:
        t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
        T = row.resp_len * 1000 + 50  # total span
        t = (
            t.astype(int) / 1000_000 - 250 - row.sound_len
        )  #  we take 50ms earlier CportOut
        if t.size < 10:
            raise ValueError("custom exception to discard trial")

        fp = np.argmax(t >= 0)
        lastp = np.argmax(t > T)  # first frame after lateral poke in
        tsegment = np.append(t[fp:lastp], T)
        tsegment = np.insert(tsegment, 0, 0)
        if invert and row.R_response == 0:
            pose = np.c_[row.trajectory_x, row.trajectory_y * -1]
        else:
            pose = np.c_[row.trajectory_x, row.trajectory_y]

        boundaries = [[0, 0]] * 6
        for i, traj in enumerate([pose, row.traj_d1, row.traj_d2]):
            f = interp1d(t, traj, axis=0)
            initial = f(0)
            last = f(T)
            boundaries[i] = initial
            boundaries[i + 3] = last
        pose = np.insert(pose[fp:lastp], 0, boundaries[0], axis=0)
        pose = np.append(pose, boundaries[3].reshape(-1, 2), axis=0)

        mu = B @ F.reshape(4, 1)

        M = get_Mt0te(tsegment[0], tsegment[-1])
        M_1 = np.linalg.inv(M)
        vt = v_(tsegment)
        N = vt @ M_1
        SIGMA = allSIGMA[int(row.R_response)]
        SIGMA_1 = np.linalg.pinv(SIGMA)
        # W = sigma**2 * np.identity(N.shape[0]) + N @ SIGMA @ N.T # shape n x n
        L = np.linalg.pinv(SIGMA_1 + N.T @ N / sigma ** 2)  # shape 6,6

        m = L @ (
            N.T @ pose[:, dim].reshape(-1, 1) / sigma ** 2
            + SIGMA_1 @ mu[:, int(row.R_response)].reshape(-1, 1)
        )
        posterior = N @ m
        prior = N @ mu[:, int(row.R_response)]  # is this the prior
        se = np.sqrt(np.diag(N @ L @ N.T))
        # se = np.zeros(N.shape[0] )
        # for i in range(N.shape[0]):
        #     posterior[i] = None # nt @ m
        #     se[i] = N[i].reshape(1,6) @ L @ N[i].reshape(6,1) # same result!
        return tsegment, pose, posterior, se, prior
    except Exception as e:
        raise e
        if mu is None:
            return np.nan, np.nan
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan


def tachometric(
    df, # filtered
    ax=None,
    hits = 'hithistory', # column name
    evidence='avtrapz', # column name
    evidence_bins=np.array([0, 0.15, 0.30, 0.60, 1.05]), # beware those with coh .2 and .4
    rt='sound_len', # column
    rtbins=np.arange(0,151,3),
    fill_error=False, # if true it uses fill between instead of errorbars
    error_kws={}, 
    cmap='inferno',
    subplots_kws={},  # ignored if ax is provided,
    labels=None
):
    """jeez how it took me so long"""

    cmap = cm.get_cmap(cmap)
    rtbinsize = rtbins[1]-rtbins[0]
    error_kws_ = dict(marker='o', capsize=3)
    error_kws_.update(error_kws)

    tmp_df = df
    tmp_df['rtbin'] = pd.cut(
        tmp_df[rt], rtbins, labels=np.arange(rtbins.size-1), 
        retbins=False, include_lowest=True, right=True
    ).astype(float)
    if ax is None:
        f, ax = plt.subplots(**subplots_kws)

    for i in range(evidence_bins.size-1):
        tmp = (
            tmp_df.loc[(tmp_df[evidence].abs()>=evidence_bins[i])&(tmp_df[evidence].abs()<evidence_bins[i+1])] # select stim str
            .groupby('rtbin')[hits].agg(['mean', groupby_binom_ci]).reset_index()
            )

        if labels is None:
            clabel = f'{round(evidence_bins[i],2)} < sstr < {round(evidence_bins[i+1],2)}'
        else:
            clabel= labels[i]
        if fill_error:
            ax.plot(
                tmp.rtbin.values * rtbinsize + 0.5 * rtbinsize,
                tmp['mean'].values, label=clabel, c=cmap(i/(evidence_bins.size-1)),
                marker=error_kws.get('marker', 'o')
            )
            ax.fill_between(
                tmp.rtbin.values * rtbinsize + 0.5 * rtbinsize,
                tmp['mean'].values + tmp.groupby_binom_ci.apply(lambda x: x[1]),
                y2=tmp['mean'].values - tmp.groupby_binom_ci.apply(lambda x: x[0]),
                color=cmap(i/(evidence_bins.size-1)),
                alpha=error_kws.get('alpha', 0.3)
            )
        else:
            ax.errorbar(
                tmp.rtbin.values * rtbinsize + 0.5 * rtbinsize,
                tmp['mean'].values,
                yerr=[
                    tmp.groupby_binom_ci.apply(lambda x: x[0]),
                    tmp.groupby_binom_ci.apply(lambda x: x[1])
                ],
                label= clabel,
                c = cmap(i/(evidence_bins.size-1)), **error_kws_
            )


def com_heatmap_paper_marginal_pcom_side(
    df, # data source, must contain 'avtrapz' and allpriors
    pcomlabel = None, fcolorwhite=True, side=0,
    hide_marginal_axis=True, n_points_marginal = None, counts_on_matrix=False,
    adjust_marginal_axes=False, # sets same max=y/x value
    nbins=7, # nbins for the square matrix 
    com_heatmap_kws={}, # avoid, binning and return_mat already handled by this function
    com_col = 'CoM_sugg', priors_col = 'norm_allpriors', stim_col = 'avtrapz',
    average_across_subjects = False
):
    assert side in [0,1], "side value must be either 0 or 1"
    assert df[priors_col].abs().max()<=1, "prior must be normalized between -1 and 1"
    assert df[stim_col].abs().max()<=1, "stimulus must be between -1 and 1"
    if pcomlabel is None:
        if not side:
            pcomlabel = r'$p(CoM_{R \rightarrow L})$'
        else:
            pcomlabel = r'$p(CoM_{L \rightarrow R})$'
    
    if n_points_marginal is None:
        n_points_marginal=nbins
    # ensure some filtering
    #tmp = df.loc[(df.dirty==False)&(df.special_trial==0)].dropna(subset=['CoM_sugg', 'norm_allpriors', 'avtrapz'])
    tmp = df.dropna(subset=['CoM_sugg', 'norm_allpriors', 'avtrapz'])
    tmp['tmp_com'] = False
    tmp.loc[(tmp.R_response==side)&(tmp.CoM_sugg), 'tmp_com'] = True
    f, ax = plt.subplots(
        ncols=2, nrows=2, 
        gridspec_kw={'width_ratios':[8, 3], 'height_ratios': [3, 8]},
        figsize=(7, 5.5), sharex='col', sharey='row'
    )

    # some aestethics
    if fcolorwhite:
        f.patch.set_facecolor('white')
        for i in [0,1]:
            for j in [0,1]:
                ax[i,j].set_facecolor('white')
        
    ax[0,1].axis('off')
    ax[0,0].set_ylabel(pcomlabel)
    ax[1,1].set_xlabel(pcomlabel)
    if hide_marginal_axis:
        ax[0,0].spines['top'].set_visible(False)
        ax[0,0].spines['left'].set_visible(False)
        ax[0,0].spines['right'].set_visible(False)
        ax[0,0].set_yticks([])
        #ax[1,1].xaxis.set_visible(False)
        ax[1,1].spines['right'].set_visible(False)
        ax[1,1].spines['top'].set_visible(False)
        ax[1,1].spines['bottom'].set_visible(False)
        ax[1,1].set_xticks([])



    com_heatmap_kws.update({
        'return_mat':True,
        'predefbins':[
            np.linspace(-1,1,nbins+1),np.linspace(-1,1,nbins+1)
        ]
        })
    if not average_across_subjects:
        mat, nmat = com_heatmap(
            tmp.norm_allpriors.values,
            tmp.avtrapz.values,
            tmp.tmp_com.values,
            **com_heatmap_kws
        )
        # fill nans with 0
        mat[np.isnan(mat)] = 0 
        nmat[np.isnan(nmat)] = 0 
        # change data to match vertical axis image standards (0,0) -> in the top left
    else:
        com_mat_list, number_mat_list = [],[]
        for subject in tmp.subjid.unique():
            cmat, cnmat = com_heatmap(
            tmp.loc[tmp.subjid==subject, 'norm_allpriors'].values,
            tmp.loc[tmp.subjid==subject, 'avtrapz'].values,
            tmp.loc[tmp.subjid==subject,'tmp_com'].values,
            **com_heatmap_kws
            )
            cmat[np.isnan(cmat)] = 0 
            cnmat[np.isnan(cnmat)] = 0 
            com_mat_list += [cmat]
            number_mat_list += [cnmat]

        mat = np.stack(com_mat_list).mean(axis=0)
        nmat = np.stack(number_mat_list).mean(axis=0)


    mat= np.flipud(mat)
    nmat = np.flipud(nmat)
        
        
    

    # marginal plots
    if counts_on_matrix:
        if average_across_subjects:
            raise NotImplementedError(' across subjects + conunts is not implemented')
        ax[0,0].bar(np.arange(nbins), nmat.sum(axis=0), width=1, edgecolor='k')
        ax[1,1].barh(np.arange(nbins), nmat.sum(axis=1), height=1, edgecolor='k')
        if hide_marginal_axis: # hack to avoid missing edge in last bar
            ax[0,0].spines['right'].set_visible(True)
            ax[0,0].spines['right'].set_bounds(0, nmat.sum(axis=0)[-1])
            ax[1,1].spines['bottom'].set_visible(True)
            ax[1,1].spines['bottom'].set_bounds(0, nmat.sum(axis=1)[-1])

    else:
        xpos=(np.linspace(-0.5,nbins-0.5,n_points_marginal+1)[:-1]+np.linspace(-0.5,nbins-0.5,n_points_marginal+1)[1:])/2
        if not average_across_subjects:
            _, means1, yerr1 = binned_curve(
                tmp, 'tmp_com', 'norm_allpriors', np.linspace(-1,1,n_points_marginal+1), # so we get double amount of ticks  
                sem_err=False, return_data=True
            )
            

            _, means2, yerr2 = binned_curve(
                tmp, 'tmp_com', 'avtrapz', np.linspace(-1,1,n_points_marginal+1), # so we get double amount of ticks  
                sem_err=False, return_data=True
            )
            ax[0,0].errorbar(xpos, means1, yerr=yerr1)
            ax[1,1].errorbar(means2, xpos[::-1], xerr=yerr2)
        else:
            bins=np.linspace(-1,1,n_points_marginal+1)
            df['tmp_bin'] = pd.cut(
                df.norm_allpriors,
                bins,
                labels=np.arange(bins.size-1), 
                retbins=False, include_lowest=True, right=True
                ).astype(float)
            tmp = (
                df.groupby(['subject','tmp_bin'])
                .CoM_sugg.mean()
                .reset_index()
                .groupby('tmp_bin').agg(['mean', sem])
            )
            ax[0,0].errorbar(xpos, tmp[('CoM_sugg', 'mean')], yerr=tmp[('CoM_sugg', 'sem')])

            bins=np.linspace(-1,1,n_points_marginal+1)
            df['tmp_bin'] = pd.cut(
                df.avtrapz,
                bins,
                labels=np.arange(bins.size-1), 
                retbins=False, include_lowest=True, right=True
                ).astype(float)
            tmp = (
                df.groupby(['subject','tmp_bin'])
                .CoM_sugg.mean()
                .reset_index()
                .groupby('tmp_bin').agg(['mean', sem])
            )
            ax[1,1].errorbar(tmp[('CoM_sugg', 'mean')], xpos[::-1], xerr=tmp[('CoM_sugg', 'sem')])

            # this was the old strategy, trying a cleaner & more robust
            #means1_list, means2_list = [],[]
            #for subject in tmp.subjid.unique():
            #    _, means1, _ = binned_curve(
            #        tmp.loc[tmp.subjid==subject], 'tmp_com', 'norm_allpriors', np.linspace(-1,1,n_points_marginal+1), # so we get double amount of ticks  
            #        sem_err=False, return_data=True
            #    )
            #
            #    _, means2, _ = binned_curve(
            #        tmp[tmp.subjid==subject], 'tmp_com', 'avtrapz', np.linspace(-1,1,n_points_marginal+1), # so we get double amount of ticks  
            #        sem_err=False, return_data=True
            #    )
            #    means1_list += [means1]
            #    means2_list += [means2]         
            # ax[0,0].errorbar(xpos, means1.mean(axis=0), yerr=sem(means1, axis=0, nan_policy='omit'))
            # ax[1,1].errorbar(means2.mean(axis=0), xpos[::-1], xerr=sem(means2, axis=0, nan_policy='omit'))
        

        #find max val to normalize
    # if normalize_marginal:
    #     mval = np.concatenate([means1, means2]).max()
    #     ax[0,0].errorbar(xpos, means1/mval, yerr=yerr1/mval)
    #     ax[1,1].errorbar(means2/mval, xpos[::-1], xerr=yerr2/mval)
    # else:
    #     ax[0,0].errorbar(xpos, means1, yerr=yerr1)
    #     ax[1,1].errorbar(means2, xpos[::-1], xerr=yerr2)
    #
    # since it is the same, simply adjust max y_
    if adjust_marginal_axes:
        _, ymax = ax[0,0].get_ylim()
        _, xmax = ax[1,1].get_xlim()
        max_val_margin = max(ymax, xmax)
        ax[0,0].set_ylim(0,max_val_margin)
        ax[1,1].set_xlim(0,max_val_margin)


    ax[1,0].set_yticks(np.arange(nbins))
    ax[1,0].set_xticks(np.arange(nbins))
    ax[1,0].set_yticklabels(['right']+['']*(nbins-2)+['left'])
    ax[1,0].set_xticklabels(['left']+['']*(nbins-2)+['right'])

    if counts_on_matrix:
        im = ax[1,0].imshow(nmat, aspect='auto')
    else:
        im = ax[1,0].imshow(mat, aspect='auto')
    ax[1,0].set_xlabel('$\longleftarrow$Prior$\longrightarrow$', labelpad=-5)
    ax[1,0].set_ylabel('$\longleftarrow$Average stimulus$\longrightarrow$', labelpad=-17)
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes('left', size='10%', pad=0.6)

    divider2 = make_axes_locatable(ax[0,0])
    empty = divider2.append_axes('left', size='10%', pad=0.6)
    empty.axis('off')
    cax2 = cax.secondary_yaxis('left')
    cbar = f.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position('left')
    if counts_on_matrix:
        cax2.set_ylabel('# trials')
    else:
        cax2.set_ylabel(pcomlabel)


    f.tight_layout()
    return f, ax