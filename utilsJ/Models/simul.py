"""
the idea is to get simulations working with few filepaths and parameters
then simply save a multi-pannel figure report.

so the following function can be called to do kind of a grid-search
"""
# import os
# from sklearn.model_selection import ParameterGrid
# from utilsJ.regularimports import *
# import swifter
from utilsJ.Behavior import plotting, ComPipe
from utilsJ.Models import traj
from concurrent.futures import as_completed, ThreadPoolExecutor
from scipy.stats import ttest_ind, sem
from matplotlib import cm
import seaborn as sns
from scipy.stats import norm
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import scipy.io as spio
general_path = traj.general_path


def _todict(matobj):
    """A recursive function which constructs from matobjects nested dictionaries"""
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def _check_keys(dict):
    """checks if entries in dictionary are mat-objects. If yes todict is called
    to change them to nested dictionaries"""
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
def loadmat(filename):
    """ this function should be called instead of direct spio.loadmat as it cures
    the problem of not properly recovering python dictionaries m mat files.
    It calls the function check keys to cure all entries which are still
    mat-objects"""
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def get_when_t(a, b, startfrom=700, tot_iter=1000, pval=0.001, nan_policy="omit"):
    """a and b are traj matrices.
    returns ms after motor onset when they split
    startfrom: matrix index to start from (should be 0th position in ms
    tot_iter= remaining)
    if ax, it plots medians + splittime"""
    for i in range(tot_iter):
        pop_a = a[:, startfrom + i]
        pop_b = b[:, startfrom + i]
        _, p2 = ttest_ind(pop_a, pop_b, nan_policy=nan_policy)
        if p2 < pval:
            return i  # , np.nanmedian(a[:,startfrom+i])
    return np.nan  # , -1


def when_did_split_dat(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7),
                       startfrom=700,  ax=None, plot_kwargs={}, align='movement',
                       collapse_sides=False, trajectory="trajectory_y"):
    """
    gets when they are statistically different by t_test,
    df= dataframe
    side= {0,1} left or right,
    rtbins
    startfrom= index to start checking diffs. movement==700;
    ax: where to plot suff if provided
    plot_kwargs: plot kwargs for ax.plot
    align: whether to align 0 to movement(action) or sound
    collapse_sides: collapse sides, so "side" arg has no effect
    """
    kw = {"trajectory": trajectory}

    # TODO: addapt to align= sound
    # get matrices
    if side == 0:
        coh1 = -1
    else:
        coh1 = 1
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        # & (df.resp_len)
    ]  # &(df.R_response==side)
    if align == 'movement':
        kw["align"] = "action"
    elif align == 'sound':
        kw["align"] = "sound"

    if not collapse_sides:
        mata = np.vstack(dat.loc[dat.coh2 == coh1]
                         # removed swifter
                         .apply(lambda x: plotting.interpolapply(x, **kw), axis=1)
                         .values.tolist()
                         )
        matb = np.vstack(
            dat.loc[(dat.coh2 == 0) & (dat.rewside == side)]
            # removed swifter
            .apply(lambda x: plotting.interpolapply(x, **kw), axis=1)
            .values.tolist()
        )
    else:
        mata_0 = np.vstack(
            dat.loc[dat.coh2 == -1]
            # removed swifter
            .apply(lambda x: plotting.interpolapply(x, **kw), axis=1)
            .values.tolist()
        )
        mata_1 = np.vstack(
            dat.loc[dat.coh2 == 1]
            # removed swifter
            .apply(lambda x: plotting.interpolapply(x, **kw), axis=1)
            .values.tolist()
        )
        mata = np.vstack(
            [mata_0*-1, mata_1]
        )

        matb_0 = np.vstack(
            dat.loc[(dat.coh2 == 0) & (dat.rewside == 0)]
            # removed swifter
            .apply(lambda x: plotting.interpolapply(x, **kw), axis=1)
            .values.tolist()
        )
        matb_1 = np.vstack(
            dat.loc[(dat.coh2 == 0) & (dat.rewside == 1)]
            # removed swifter
            .apply(lambda x: plotting.interpolapply(x, **kw), axis=1)
            .values.tolist()
        )
        matb = np.vstack(
            [matb_0*-1, matb_1]
        )

    for a in [mata, matb]:  # discard all nan rows
        a = a[~np.isnan(a).all(axis=1)]

    ind = get_when_t(mata, matb, startfrom=startfrom)

    if ax is not None:
        ax.plot(np.arange(mata.shape[1]) - startfrom,
                np.nanmedian(mata, axis=0), **plot_kwargs)
        ax.plot(np.arange(matb.shape[1]) - startfrom,
                np.nanmedian(matb, axis=0), **plot_kwargs, ls=':')
        ax.scatter(ind, np.nanmedian(mata[:, startfrom+ind]), marker='x',
                   color=plot_kwargs.get('color'),
                   s=50, zorder=3)
    return ind  # mata, matb,


def shortpad(traj, upto=1000, pad_value=np.nan):
    """pads nans to trajectories so it can be stacked in a matrix"""
    missing = upto - traj.size
    return np.pad(traj, ((0, missing)), "constant", constant_values=pad_value)


def shortpad2(row, upto=1000, align='movement', pad_value=np.nan, pad_pre=0):
    """pads nans to trajectories so it can be stacked in a matrix
    align can be either 'movement' (0 is movement onset), or 'sound'
    """
    if align == 'movement':
        missing = upto - row.traj.size
        return np.pad(row.traj, ((0, missing)), "constant",
                      constant_values=pad_value)
    elif align == 'sound':
        missing_pre = int(row.sound_len)
        missing_after = upto - missing_pre - row.traj.size
        return np.pad(row.traj, ((missing_pre, missing_after)), "constant",
                      constant_values=(pad_pre, pad_value))


def when_did_split_simul(
    df, side, rtbin=0, rtbins=np.linspace(0, 150, 7),
    ax=None, plot_kwargs={}, align='movement',
    return_mats=False  # debugging purposes
):
    """gets when they are statistically different by t_test
    here df is simulated df"""
    # get matrices
    if side == 0:
        coh1 = -1
    else:
        coh1 = 1
    shortpad_kws = {}
    if align == 'sound':
        shortpad_kws = dict(upto=1400, align='sound')
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        # & (df.resp_len) # ?
    ]  # &(df.R_response==side) this goes out
    mata = np.vstack(
        dat.loc[(dat.traj.apply(len) > 0) & (dat.coh2 == coh1)]
        .apply(lambda row: shortpad2(row, **shortpad_kws), axis=1)
        .values.tolist()
    )
    matb = np.vstack(
        dat.loc[
            (dat.traj.apply(len) > 0) & (dat.coh2 == 0) & (dat.rewside == side)
        ]
        .apply(lambda row: shortpad2(row, **shortpad_kws), axis=1)
        .values.tolist()
    )
    matlist = [mata, matb]
    # discard all nan rows # this is not working because a is a copy!
    for i in [0, 1]:
        matlist[i] = matlist[i][~np.isnan(matlist[i]).all(axis=1)]
    mata, matb = matlist

    ind = get_when_t(mata, matb, startfrom=0)

    if return_mats:
        print(ind)
        return mata, matb

    if ax is not None:
        ax.plot(np.nanmedian(mata, axis=0), **plot_kwargs)
        ax.plot(np.nanmedian(matb, axis=0), **plot_kwargs, ls=':')
        ax.scatter(ind, np.nanmedian(mata[:, ind]), marker='x',
                   color=plot_kwargs['color'], s=50, zorder=3)
    return ind  # mata, matb,


def whole_splitting(df, rtbins=np.arange(0, 151, 25), simul=False, ax=None,
                    align='movement'):
    """calculates time it takes for each Side*rtbin
    coherence 1vs0 to split significantly
    """
    _index = [0, 1]  # side
    _columns = np.arange(rtbins.size - 1)  # rtbins
    tdf = pd.DataFrame(np.ones((2, _columns.size)) * -
                       1, index=_index, columns=_columns)
    if simul:
        splitfun = when_did_split_simul
    else:
        splitfun = when_did_split_dat
    # tdf.columns = tdf.columns.set_names(['RTbin', 'side'])
    for b in range(rtbins.size - 1):
        if (b == 0) or (b == 3):
            cax = ax
        else:
            cax = None
        for s, side in enumerate(["L", "R"]):
            if b:
                tab = 'tab:'
            else:
                tab = ''
            if side == 'L':
                plot_kwargs = dict(color=tab+'green', label=f'rtbin={b}')
            else:
                plot_kwargs = dict(color=tab+'purple', label=f'rtbin={b}')
            split_time = splitfun(
                df, s, b, rtbins=rtbins, ax=cax, align=align,
                plot_kwargs=plot_kwargs
            )
            tdf.loc[s, b] = split_time

    return tdf


# XXX: divergence plot
def splitplot(df, out, ax, ax1, ax2):
    """plots trajectory split time (coh1 vs 0) per RT-bin"""
    tdf = whole_splitting(df, rtbins=np.arange(0, 151, 10), ax=ax1)
    tdf2 = whole_splitting(out, rtbins=np.arange(
        0, 151, 10), simul=True, ax=ax2)
    colors = ["green", "purple"]
    for i, (dat, name, marker) in enumerate([[tdf, "data", "o"],
                                             [tdf2, "simul", "x"]]):
        for j, side in enumerate(["L", "R"]):
            ax.scatter(
                dat.columns,
                dat.loc[j, :].values,
                marker=marker,
                color=colors[j],
                label=f"{name} {side}",
            )

    ax.set_xlabel("RT (ms)", fontsize=14)
    ax.set_ylabel("time to diverge", fontsize=14)
    xarr = np.array([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_xticks(xarr)
    ax.set_xticklabels(xarr*10+5)
    # ax.legend(fancybox=False, frameon=False) # call it later
    ax1.set_xlim(-20, 400)
    ax1.legend(fancybox=False, frameon=False)
    ax1.set_title(' trajectory divergence in rat data')

    ax2.set_title('trajectory divergence in model')
    ax2.set_xlim(-20, 400)
    ax2.legend(fancybox=False, frameon=False)
    al = [ax1, ax2]
    for i in [0, 1]:
        al[i].set_xlabel('MT (ms)', fontsize=14)
        al[i].set_ylabel('y coordinate', fontsize=14)


def plot_com_contour(df, out, ax):
    """contour of CoM peak vs prior"""
    sns.kdeplot(out.loc[out.CoM_sugg, 'CoM_peakf'].apply(lambda x: x[0]).values,
                out.loc[out.CoM_sugg, 'allpriors'].values, ax=ax)


def plot_median_com_traj(df, out, ax):
    """median trajectory for CoM, spliting by huge and moderate bias
    [only right responses]"""
    ax.plot(np.nanmedian(np.vstack(out.loc[(out.CoM_sugg) &
                                           (out.allpriors < -1.25), 'traj'].dropna().apply(lambda x: shortpad(x, upto=700)).values),
            axis=0), color='navy', label='huge prior')
    ax.plot(np.nanmedian(np.vstack(out.loc[(out.CoM_sugg) &
                                           (out.allpriors.abs() < 1.25), 'traj'].dropna().apply(lambda x: shortpad(x, upto=700)).values),
            axis=0), color='tab:blue', label='moderate to 0 prior')
    ax.set_xlim([0, 300])
    ax.set_ylim([-30, 80])
    ax.set_xlabel('ms from movement onset', fontsize=14)
    ax.set_ylabel('distance in px', fontsize=14)
    ax.legend(fancybox=False, frameon=False)


def plot0(df, out, ax):
    """RT distributions"""
    df.loc[(df.sound_len < 250)].sound_len.hist(
        bins=np.linspace(0, 250, 101),
        ax=ax,
        label="Rat data",
        density=True,
        alpha=0.5,
        grid=False,
    )
    out.sound_len.hist(
        bins=np.linspace(0, 250, 101),
        ax=ax,
        label="Model",
        density=True,
        alpha=0.5,
        grid=False,
    )
    ax.set_xlabel("RT (ms)", fontsize=14)
    ax.set_ylabel('density', fontsize=14)
    # ax.set_title("RT distr")
    ax.legend(frameon=False, fancybox=False)


def pcomRT(df, out, ax):
    """p(CoM) vs RT"""
    plotting.binned_curve(
        df,
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="Rat data", color="tab:blue"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls="-"),
    )
    plotting.binned_curve(
        out,
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="Model", color="tab:orange"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls=":"),
    )
    plotting.binned_curve(
        out.loc[out.reactive == 0],
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="Model (proactive trials)", color="tab:purple"),
        legend=False,
        # traces="sstr",
        # traces_kw=dict(color="grey", alpha=0.3, ls=":"),
    )
    ax.set_ylabel("p(detected-CoM)", fontsize=14)
    ax.set_xlabel("RT(ms)", fontsize=14)
    # ax.set_title("pcom vs rt")
    ax.legend(frameon=False, fancybox=False)


def pcomRT_proactive_only(df, out, ax):
    """deprecated"""
    plotting.binned_curve(
        out.loc[out.reactive == 0],
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="simul", color="tab:orange"),
        legend=False,
        # traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls=":"),
    )
    ax.set_ylabel("p(CoM)", fontsize=14)
    ax.set_xlabel("RT(ms)", fontsize=14)
    # ax.set_title("pcom in proactive")
    ax.legend(frameon=False, fancybox=False)


def plot2(df, out, ax):
    """MT distribution"""
    df.resp_len.hist(
        bins=np.linspace(0, 1, 81),
        ax=ax,
        label="Rat data",
        density=True,
        alpha=0.5,
        grid=False,
    )
    out.resp_len.hist(
        bins=np.linspace(0, 1, 81),
        ax=ax,
        label="Model",
        density=True,
        alpha=0.5,
        grid=False,
    )
    ax.set_xlabel("MT (secs)", fontsize=14)
    ax.set_ylabel('density', fontsize=14)
    # ax.set_title("MT distr")
    ax.legend(frameon=False, fancybox=False)


def plot3(df, out, ax):
    """U shape MT vs RT"""
    titles = ["Rat data", "Model"]
    datacol = ["resp_len", "resp_len"]
    traces_ls = ["-", ":"]
    for i, dfobj in enumerate([df, out]):  # .loc[out.reactive==0]
        plotting.binned_curve(dfobj, datacol[i], "sound_len", ax=ax,
                              bins=np.linspace(0, 150, 16), xpos=10, traces="sstr",
                              traces_kw=dict(color="grey", alpha=0.3,
                                             ls=traces_ls[i]),
                              xoffset=5,
                              errorbar_kw={"marker": "o", "label": titles[i]})
        # other inputs
        # xpos=np.arange(0,150,10), traces='sstr',
        # traces_kw = dict(color='grey', alpha=0.3, ls=traces_ls[i]),

        ax.set_xlabel("RT (ms)", fontsize=14)
        ax.set_ylabel("MT (s)", fontsize=14)
        ax.set_title("MT vs RT", fontsize=14)
        ax.legend(frameon=False, fancybox=False)


def plot4(df, out, ax):
    """proportion of proactive trials"""
    counts_t, xpos = np.histogram(out.sound_len, bins=np.linspace(0, 250, 26))
    counts_p, _ = np.histogram(
        out.loc[out.reactive == 0, "sound_len"], bins=np.linspace(0, 250, 26)
    )
    prop_pro = counts_p / counts_t
    ax.plot(xpos[:-1] + 5, prop_pro, marker="o")
    ax.set_ylabel("proportion proactive", fontsize=14)
    ax.set_xlabel("RT (ms)", fontsize=14)
    ax.set_ylim([-0.05, 1.05])


def plot5(df, out, ax):
    """incomplete"""
    ax.set_title("stimuli split trajectories")
    ax.annotate("splitting time per rtbin", (0, 0))


def plot67(df, out, ax, ax2, rtbins=np.linspace(0, 150, 7)):
    """deprecated"""
    markers = ["o", "x"]
    rtbins = np.linspace(0, 150, 7)
    priorbins = np.linspace(-2, 2, 6)
    cmap = cm.get_cmap("viridis_r")
    datres, simulres = pd.DataFrame([]), pd.DataFrame([])
    for i, (dat, store) in enumerate([[df, datres], [out, simulres]]):
        dat["priorbin"] = pd.cut(
            dat.choice_x_allpriors, priorbins, labels=False, include_lowest=True
        )
        for j in range(rtbins.size - 1):
            tmp = (
                dat[(dat.sound_len >= rtbins[j]) & (
                    dat.sound_len < rtbins[j + 1])]
                .groupby("priorbin")["time_to_thr"]
                .agg(m="mean", e=sem)
            )
            store[f"rtbin{j}"] = tmp["m"]

            if j % 2 == 0:  # plot half of it
                kws = {
                    "ls": "none",
                    "marker": markers[i],
                    "color": cmap(j / (rtbins.size - 1)),
                    "capsize": 2,
                }
                if not i:
                    kws = {
                        "ls": "none",
                        "marker": markers[i],
                        "label": f"rtbin={j}",
                        "color": cmap(j / (rtbins.size - 1)),
                        "capsize": 2,
                    }
                ax.errorbar(tmp.index, tmp["m"], yerr=tmp["e"], **kws)
    ax.legend().remove()
    ax.set_ylabel("ms to threshold (30px)", fontsize=14)
    ax.set_xlabel("congruence (choice * prior)", fontsize=14)
    ax.set_title("prior congruence on MT (o=data, x=simul)")
    diffdf = datres - simulres
    ax2.axhline(0, c="gray", ls=":")
    ax2.set_xlabel("congruence (choice * prior)", fontsize=14)
    ax2.set_ylabel("data - simul", fontsize=14)
    for i in range(rtbins.size - 1):
        ax2.scatter(
            [-2, -1, 0, 1, 2],
            diffdf[f"rtbin{i}"].values,
            color=cmap(i / (rtbins.size - 1)),
            label=f"rtbin {i}",
        )
    ax2.set_title("prior congruence on MT (data - simul)")
    ax2.legend(frameon=False, fancybox=False)


def plot910(df, out, ax, ax2, rtbins=np.linspace(0, 150, 7)):
    """deprecated"""
    markers = ["o", "x"]
    cmap = cm.get_cmap("viridis_r")
    datres, simulres = pd.DataFrame([]), pd.DataFrame([])
    for i, (dat, store) in enumerate([[df, datres], [out, simulres]]):
        kwargs = dict(ls="none", marker=markers[i], capsize=2)
        for j in range(rtbins.size - 1):
            tmp = (
                dat[(dat.sound_len >= rtbins[j]) & (
                    dat.sound_len < rtbins[j + 1])]
                .groupby("choice_x_coh")["time_to_thr"]
                .agg(m="mean", e=sem)
            )
            store[f"rtbin{j}"] = tmp["m"]
            if j % 2 == 0:
                c = cmap(j / (rtbins.size - 1))
                ax.errorbar(tmp.index, tmp["m"], yerr=tmp["e"], **kwargs, c=c)
    ax.set_xlabel("coh * choice", fontsize=14)
    ax.set_ylabel("ms to threshold (30px)", fontsize=14)
    ax.set_title("coherence congruence on MT (o=data, x=simul)")
    diffdf = datres - simulres
    ax2.axhline(0, c="gray", ls=":")
    ax2.set_xlabel("coh * choice", fontsize=14)
    ax2.set_ylabel("data - simul", fontsize=14)
    ax2.set_title("coherence congruence on MT (data - simul)")
    for i in range(rtbins.size - 1):
        ax2.scatter(
            diffdf.index,
            diffdf[f"rtbin{i}"].values,
            color=cmap(i / (rtbins.size - 1)),
            label=f"rtbin {i}",
        )

    ax2.legend(frameon=False, fancybox=False)


def plot1112(df, out, ax, ax2):
    """data and simul CoM Matrix """

    # get max p(com) so colorbars are the same
    subset = df.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    mat_data, _ = plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        return_mat=True
    )
    subset = out.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    mat_simul, _ = plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        return_mat=True
    )
    maxval = np.max(np.concatenate([
        mat_data.flatten(), mat_simul.flatten()
    ]))

    subset = df.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        flip=True,
        ax=ax,
        cmap="magma",
        fmt=".0f",
        vmin=0,
        vmax=maxval
    )
    ax.set_title(f"data p(CoM)")
    subset = out.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        flip=True,
        ax=ax2,
        cmap="magma",
        fmt=".0f",
        vmin=0,
        vmax=maxval
    )
    ax2.set_title(f" SIMULATIONS p(CoM)")


def _callsimul(args):
    """unpacks all args so we can use concurrent futures with traj.simul_psiam"""
    return traj.simul_psiam(*args)


def safe_threshold(row, threshold):
    pass  # will this be implemented?


def whole_simul(
    subject,
    savpath=None,
    dfpath=f"{general_path}paper/",  # dani_clean.pkl",  # parameter grid
    rtbins=np.linspace(0, 150, 7),  # deprecated ~ not yet
    params={
        "t_update": 80,  # ms
        "proact_deltaMT": 0.3,
        "reactMT_interc": 110,
        "reactMT_slope": 0.15,
        "com_gamma": 250,
        "glm2Ze_scaling": 0.25,
        "x_e0_noise": 0.001,
        "naive_jerk": True,
        "confirm_thr": 0,
        "proportional_confirm": False,
        "t_update_noise": 0,
        "com_deltaMT": 0,  # 0 = no modulation
        "jerk_lock_ms": 0
    },
    batches=10,
    batch_size=1500,
    return_data=False,
    vanishing_bounds=False,
    both_traj=False,
    silent_trials=False,
    sample_silent_only=False,
    trajMT_jerk_extension=0,
    mtnoise=True,
    com_height=5,
    drift_multiplier=1,
    extra_t_0_e=0,  # secs
    use_fixed_bias=False,
    return_matrices=False

):
    """
    subject: 'LEXX' subject to simulate, since several params
            (psiam and silentMT are loaded from fits)
    savpath: where to save resulting multi-pannel figure
    dfpath: path where the data is stored. If it doe snot end with .pkl attempts
            appending {subject}_clean.pkl
        *new*: if a df is passed it skips loading + some preprocessing
    rtbins: reaction time bins, semi deprecated
    params: parameters to simulate
        t_update: time it takes from bound hit to exert effect in movement
        pract_deltaMT: coef to reduce expected MT based on accumulated evidence
        reactMT_interc: intercept of reactive MT
        reactMT_slope: slope of reactive MT (* trial_index)
        com_gamma: motor time from updating trajectory to new (reverted) choice
        glm2Ze_scaling: factor to scale down Ze (= Ze*glm2Ze_scaling*bound_height)
        x_e0_noise: variance of the beta distr. centered at scaled Ze
        naive_jerk: whether to use
            True => boundary conditions = (0,0,0,75,0,0) or
            False=> those fitted using alex EM procedure (deprecated)
        confirm_thr: use some confirm threshold
                (ie evidence decision criterion != 0).
                units fraction of bound_height
        proportional_confirm: whether to make it proportional to Ze/x_0_e
        t_update_noise: scale of the gaussian noise to be added to t_update.
                        (derpecated)
        com_deltaMT: coef to reduce CoM MT based on accumulated evidence
        jerk_lock_ms: ms of trajectory which keep locked @ y=0
    batches: number of simulation batches
    batch_size: amount of trials simulated per batch (too high -> memory errors)
    return_data: deprecated
    vanishing_bounds: whether to disable horizontal bounds after AI hits the bound
                    till ev bounds collapse
    both_traj: whether to return both trajectories in out dataframe
                (preplanned + final) or not (just final)
    silent_trials: just simulate silent trials
    sample_silent_only: only sample silent trials from data to simulate
    trajMT_jerk_extension: ms of extension of the trajectory to simulate.
                        Since subjects do not break lateral photogate with x'=0
                        and x''=0 it may help getting simulated trajectories
                        that resemble data
    mtnoise: whether to add noise to predicted "expected MT". If float it adds
            gaussian noise scaled to that value * mae
    com_height: com detection threshold in pxfor simulated data. Real data is
                already thresholded/detected at 5px
    drift_multiplier: multiplies fited values of drift. It can be an array of
                    shape (4,)
    extra_t_0_e: extends t_0_e for this amound of SECONDS
    use_fixed_bias: not implemented
    return_matrices: psiam returns outdf, and evidence_matrix, and
                urgency_matrix # beware this can take a lot of memory
    """
    # dev note
    if use_fixed_bias:
        raise NotImplementedError('fixed bias usage is not implemented because' +
                                  ' it might require to fit expectedMT again')

    df = pd.DataFrame([])
    preprocessed_flag = False
    if isinstance(dfpath, pd.DataFrame):
        df = dfpath
        preprocessed_flag = True
    elif subject == 'all':
        dfpath = f"{general_path}paper/dani_clean.pkl"
    # use default naming # append subject to data path
    elif not dfpath.endswith('.pkl'):
        dfpath = f"{dfpath}{subject}_clean.pkl"

    # if savpath is None:
    #     raise ValueError("provide save path")

    if 'sstr' not in df.columns:
        preprocessed_flag = False

    # load real data
    # unpickling whole dataframe (6 subjects) takes 4 minutes
    if not len(df):  # df is empty
        df = pd.read_pickle(dfpath)
    # ensure we just have a single subject
    if subject != 'all':
        df = df.loc[df.subjid == subject]
    if not preprocessed_flag:
        df["sstr"] = df.coh2.abs()  # stimulus str column
        df["priorZt"] = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        # 'dW_fixedbias' not used in the evidence offset/pre-planned ch anymore*
        df["prechoice"] = np.ceil(df.priorZt.values / 1000)  # pre-planned choice
        df["prechoice"] = df.prechoice.astype(int)
        # initialize variable: time to reach arbitrary threshold in px
        df["time_to_thr"] = np.nan
        # df.swifter.apply(lambda x:
        #                  np.argmax(np.abs(plotting.interpolapply(x)[700:])>30),
        #                  axis=1)
        # split lft and right now!
        df.loc[(df.R_response == 1) &
               (df.trajectory_y.apply(len) > 10), "time_to_thr"] = (
            df.loc[(df.R_response == 1) & (df.trajectory_y.apply(len) > 10)]
            .dropna(subset=["sound_len"])
            .apply(
                # from 700 because they are aligned
                lambda x: np.argmax(plotting.interpolapply(x)[700:] > 30), axis=1
                # to movement onset at 700 position
                # (extreme case [fixation]+[stim]=700)
            )
        )  # axis arg not req. in series
        df.loc[(df.R_response == 0) & (df.trajectory_y.apply(len) > 10),
               "time_to_thr"] = (df.loc[(df.R_response == 0) & (df.trajectory_y.apply(len) > 10)].dropna(subset=["sound_len"]).apply(lambda x: np.argmax(plotting.interpolapply(x)[700:] < -30), axis=1))  # axis arg not req. in series
        df["rtbin"] = pd.cut(df.sound_len, rtbins,
                             labels=False, include_lowest=True)
        df["choice_x_coh"] = (df.R_response * 2 - 1) * df.coh2
        df["allpriors"] = np.nansum(
            df[["dW_trans", "dW_lat"]].values, axis=1
        )  # , 'dW_fixedbias'
        df["choice_x_allpriors"] = (df.R_response * 2 - 1) * df.allpriors

    # load and unpack psiam parameters
    if subject == 'all':
        psiam_path = f"{general_path}paper/fits_psiam/all.pkl"
        (
            c,
            v_u,
            a_u,
            t_0_u,
            *v,
            a_e,
            z_e,
            t_0_e,
            t_0_e_silent,
            v_trial,
            b,
            d
        ) = pd.read_pickle(psiam_path).mean(axis=1).tolist()
    else:
        psiam_path = f"{general_path}paper/fits_psiam/{subject} D2Mconstrainedfit_fitonly.mat"
        psiam_params = loadmat(
            psiam_path
        )["freepar_hat"][0]
        (
            c,
            v_u,
            a_u,
            t_0_u,
            *v,
            a_e,
            z_e,
            t_0_e,
            t_0_e_silent,
            v_trial,
            b,
            d,
            _,
            _,
            _,
        ) = psiam_params
    assert extra_t_0_e < 1,\
        f't_0_e is in secs, it should not be > than 1 and it is {extra_t_0_e}'
    if extra_t_0_e:
        t_0_e += extra_t_0_e

    out = pd.DataFrame([])  # init out dataframe
    if return_matrices:
        u_mat_list = []
        e_mat_list = []

    if sample_silent_only:
        df = df[df.special_trial == 2]
        print(
            f"just sampling from silent trials in subject {subject}\nwhich is around {len(df.loc[df.subjid==subject])}"
        )
    # psiam simulations
    print("psiam_simul began")
    # this runs 7 times so we expect to have n_simul_trials = 7*nbatches*batch_size
    # pack psiam parameters again to pass them as arg
    psiam_params = [
        c, v_u, a_u, t_0_u, v[0], v[1], v[2], v[3], a_e, z_e, t_0_e, t_0_e_silent,
        v_trial, b, d, 0, 0, 0
    ]
    with ThreadPoolExecutor(max_workers=7) as executor:
        jobs = [
            executor.submit(
                _callsimul,
                [
                    df,  # df.loc[df.subjid == subject],
                    psiam_params,
                    1.3,
                    0.3,
                    1e-4,
                    x,  # seed
                    batches,
                    batch_size,
                    params["glm2Ze_scaling"],
                    silent_trials,
                    sample_silent_only,
                    params["x_e0_noise"],
                    params["confirm_thr"],
                    params["proportional_confirm"],
                    params["confirm_ae"],
                    drift_multiplier,
                    return_matrices
                ],
            )
            # x is the seed so we do not simulate the same over and over
            for x in np.arange(7) * 50
        ]
        if not return_matrices:
            for job in tqdm.tqdm_notebook(as_completed(jobs), total=7):
                out = out.append(job.result(), ignore_index=True)
        else:
            for job in tqdm.tqdm_notebook(as_completed(jobs), total=7):
                res1, res2, res3 = job.result()
                out = out.append(res1, ignore_index=True)
                e_mat_list += [res2]
                u_mat_list += [res3]
    # so psiam simulations are already in "out" dataframe

    # initializes a class that will retrieve expected MT
    tr = traj.def_traj(subject, None)
    # initialize some variables
    out["expectedMT"] = np.nan
    out["mu_boundary"] = np.nan
    # object because it will contain boundary conditions (vec)
    out["mu_boundary"] = out["mu_boundary"].astype(object)

    out["priorZt"] = np.nansum(
        out[["dW_lat", "dW_trans"]].values, axis=1
    )  # 'dW_fixedbias',
    out["prechoice"] = np.ceil(out.priorZt.values / 1000)
    out["prechoice"] = out.prechoice.astype(int)

    # invert those factors in left choices to align it to a single dimensino
    for col in ["dW_trans", "dW_lat"]:
        out[f"{col}_i"] = out[col] * (out["prechoice"] * 2 - 1)

    try:  # PROACTIVE RESPONSES
        # create a sliced dataframe of proactive responses to work with
        sdf = out.loc[out.reactive == 0]
        if subject != 'all':
            tr.selectRT(0)  # loads subject's MT LinearModel
        else:
            tr.load_all()  # specific method to load average*
        # Generate design matrix to multiply with weights
        fkmat = sdf[["zidx", "dW_trans_i", "dW_lat_i"]].fillna(0).values
        # (cumbersome we could use ".predict" instead of doing this raw)
        fkmat = np.insert(fkmat, 0, 1, axis=1)
        # fkmat = sdf[['zidx', 'dW_trans', 'dW_lat']].fillna(0).values
        # fkmat = np.insert(fkmat, 0, 1,axis=1)
        tr.expected_mt(fkmat, add_intercept=False)
        # store expectedMT in dataframe
        # add expectedMT to those indexes in data frame "out"
        out.loc[sdf.index, "expectedMT"] = tr.mt * 1000

        # add noise to the predicted value
        if isinstance(mtnoise, bool):
            mtnoise *= 1
        if mtnoise:  # load mserror
            # with open(
            #     f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/MTmse.pkl",
            #     "rb",
            # ) as handle:
            #     msedict = pickle.load(handle)
            # err = mtnoise * msedict[subject] ** 0.5
            # now it is store in clf object [retrieve it with tr.clf.mse_]
            err = mtnoise * tr.clf.mse_ ** 0.5

            out.loc[sdf.index, "expectedMT"] += np.random.normal(
                scale=err, size=out.loc[sdf.index, "expectedMT"].values.size
            ) * 1000  # big bug here, we were using ms already!

            # XXX some noise can make impossible trajectories (ie expectedMT<=0)
            # apply a threshold, expected mT cannot be below 125 ms
            out.loc[(out.reactive == 0) & (
                out.expectedMT < 125), 'expectedMT'] = 125

        if params["naive_jerk"]:
            naive_jerk = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
            # broadcast final position to port since it is aligned to final choice,
            # It will be aligned later when generating trajectories
            out.loc[sdf.index, f"mu_boundary"] = len(sdf) * [naive_jerk]
        else:
            out.loc[sdf.index, f"mu_boundary"] = tr.return_mu(Fk=fkmat)
            # out.loc[sdf.index, f'priortraj{side}'] =\
            #     bo.prior_traj(Fk=fkmat,times=bo.mt+0.05,step=10)

        # REACTIVE ONES
        # create a sliced dataframe of reactive responses to work with
        sdf = out.loc[out.reactive == 1]
        fkmat = sdf[["zidx", "dW_trans_i", "dW_lat_i"]].fillna(0).values
        fkmat = np.insert(fkmat, 0, 1, axis=1)
        times = (
            sdf.loc[sdf.index, "origidx"].values * params["reactMT_slope"]
            + params["reactMT_interc"]
        )
        out.loc[sdf.index, f"expectedMT"] = times
        if params["naive_jerk"]:
            naive_jerk = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
            # broadcast final position to port since it is aligned to final choice
            # It will be aligned later when generating trajectories
            out.loc[sdf.index, f"mu_boundary"] = len(sdf) * [naive_jerk]
        else:
            out.loc[sdf.index, f"mu_boundary"] = tr.return_mu(
                Fk=fkmat
            )  # this one is not used later, right?
            # out.loc[sdf.index, f'priortraj{side}'] =\
            #     bo.prior_traj(step=10, Fk=fkmat, times=(times+50)/1000)
    except Exception as e:
        raise e

    remaining_sensory = (t_0_e - 0.3) * 1000  # that would be t_e delay
    out["remaining_sensory"] = remaining_sensory
    # edit remaining sensory pipe where the stimulus was shorter than it
    out.loc[out.sound_len < remaining_sensory, "remaining_sensory"] = out.loc[
        out.sound_len < remaining_sensory, "sound_len"
    ]
    out["e_after_u"] = 0  # flag for those trials where EA bound is hit after AI
    out.loc[
        (out.e_time * 1000 < out.sound_len + out.remaining_sensory)
        & (out.reactive == 0),
        "e_after_u",
    ] = 1  # control for those which have listened less than the delay # done

    out.prechoice = out.prechoice.astype(int)
    out["sstr"] = out.coh2.abs()
    out["t_update"] = np.nan

    # effective t_update for most of the trials is params['t_update']
    out.loc[out.reactive == 0, "t_update"] = params[
        "t_update"
        # ] + t_0_e # now it happen relative to movement onset!
    ] + t_0_e//0.001 - 300  # new bug

    # UNCOMMENT LINE BELOW IF UPDATE CAN HAPPEN EARLIER WHEN EV-BOUND IS REACHED
    if not vanishing_bounds:
        # adapt t_update in those trials where EA bound is hit after AI bound
        out.loc[out.e_after_u == 1, "t_update"] = (
            out.loc[out.e_after_u == 1, "e_time"] -
            out.loc[out.e_after_u == 1, "u_time"]
        ) * 1000 + params['t_update']

    if params['t_update_noise']:
        out['t_update'] += np.random.normal(
            scale=params['t_update_noise'], size=len(out))

    if (out['t_update'] < 0).sum():
        warnings.warn('replacing t_updates < 0 to 0')
        out.loc[out.t_update < 0, 't_update'] = 0

    # add speed up for confirmed choices
    out["resp_len"] = np.nan
    out["resp_len"] = out["expectedMT"]  # this include reactive
    out["base_mt"] = out["expectedMT"]  # this include reactive
    if not silent_trials:
        # if there's confirm_thr scale it relative to delta_ev
        if params["confirm_thr"] > 0:
            out.loc[
                (out.R_response == out.prechoice) & (
                    out.reactive == 0), "resp_len"
            ] = (
                (
                    1
                    - params["proact_deltaMT"]
                    * out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "delta_ev",
                    ].abs()
                    / a_e
                )
                * (
                    out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "resp_len",
                    ]
                    - out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "t_update",
                    ]
                )
                + out.loc[
                    (out.R_response == out.prechoice) & (
                        out.reactive == 0), "t_update"
                ]
            )  # prechoice==final choice aka confirm
        else:  # samething but with x_e instead of delta_ev
            out.loc[
                (out.R_response == out.prechoice) & (
                    out.reactive == 0), "resp_len"
            ] = (
                (
                    1
                    - params["proact_deltaMT"]
                    * out.loc[
                        (out.R_response == out.prechoice) & (
                            out.reactive == 0), "x_e"
                    ].abs()
                    / a_e
                )
                * (
                    out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "resp_len",
                    ]
                    - out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "t_update",
                    ]
                )
                + out.loc[
                    (out.R_response == out.prechoice) & (
                        out.reactive == 0), "t_update"
                ]
            )  # prechoice==final choice aka confirm

        # changes of mind response length
        out.loc[(out.R_response != out.prechoice) & (out.reactive == 0), "resp_len"] = (
            out.loc[(out.R_response != out.prechoice)
                    & (out.reactive == 0), "t_update"]
            + params["com_gamma"] * (
                1-params['com_deltaMT']
                * out.loc[(out.R_response != out.prechoice) & (out.reactive == 0), 'delta_ev'].abs()
            )
            # why was this shit being added? # else we have a peak around gamma com
            + params["reactMT_slope"]
            * out.loc[
                (out.R_response != out.prechoice) & (
                    out.reactive == 0), "origidx"
            ]
        )  # tri_ind is new

    # transform into seconds so it has same units in df and out
    out["resp_len"] /= 1000
    out["base_mt"] /= 1000  # idem

    # ensure they are float because we had some object datatype column
    for col in ["zidx", "origidx", "expectedMT", "resp_len", "base_mt"]:
        out[col] = out[col].astype(float)

    out["allpriors"] = np.nansum(out[["dW_trans", "dW_lat"]].values, axis=1)
    out["choice_x_allpriors"] = (out.R_response * 2 - 1) * out.allpriors
    out["traj"] = np.nan
    out.traj = out.traj.astype(object)
    print("psiam done, generating trajectories + derivatives")
    if both_traj:
        out["pretraj"] = np.nan
        out.pretraj = out.pretraj.astype(object)
        out[["pretraj", "traj"]] = out.apply(
            lambda x: traj.simul_traj_single(
                x,
                return_both=True,
                silent_trials=silent_trials,
                trajMT_jerk_extension=trajMT_jerk_extension,
                jerk_lock_ms=params["jerk_lock_ms"]
            ),
            axis=1,
            result_type="expand",
        )
    else:
        out["traj"] = out.apply(
            lambda x: traj.simul_traj_single(
                x,
                silent_trials=silent_trials,
                trajMT_jerk_extension=trajMT_jerk_extension,
                jerk_lock_ms=params["jerk_lock_ms"]
            ),
            axis=1,
        )
    # getting and concatenating gradients
    tmp = out.apply(
        lambda x: plotting.gradient_np_simul(x), axis=1, result_type="expand"
    )
    tmp.columns = ["traj_d1", "traj_d2", "traj_d3"]
    out = pd.concat([out, tmp], axis=1)

    print("getting CoM etc.")
    out["time_to_thr"] = np.nan
    for i, thr in enumerate([30, 30]):
        out.loc[
            (out.R_response == i) & (out.traj.apply(len) > 0), "time_to_thr"
        ] = out.loc[
            (out.R_response == i) & (out.traj.apply(len) > 0), "traj"
        ].apply(  # ].swifter.apply(
            lambda x: np.argmax(np.abs(x) > thr)
        )
    out["rtbin"] = pd.cut(out.sound_len, rtbins,
                          labels=False, include_lowest=True)
    out["choice_x_coh"] = (out.R_response * 2 - 1) * out.coh2
    out[["Hesitation", "CoM_sugg", "CoM_peakf"]] = out.apply(
        lambda x: ComPipe.chom.did_he_hesitate(
            x, simul=True, positioncol="traj", speedcol="traj_d1",
            height=com_height),
        axis=1,
        result_type="expand",
    )  # return 1 or more peak frames?

    # not saving data, its 2.9 GB each
    # print('saving data')
    # out.to_pickle(f'{savpath}.pkl')

    # plotting section

    # get data (a) and simul (b) sets to plot.
    # Check each plot function to dig further
    if silent_trials:
        pref_title = "silent_"
        if subject == 'all':
            filter_mask = (df.special_trial == 2)
        else:
            filter_mask = (df.special_trial == 2) & (df.subjid == subject)
        a, b = df.loc[filter_mask], out
    else:
        pref_title = ""
        if subject == 'all':
            filter_mask = (df.special_trial == 0)
        else:
            filter_mask = (df.special_trial == 0) & (df.subjid == subject)
        a, b = df.loc[filter_mask], out
    fig, ax = plt.subplots(ncols=4, nrows=5, figsize=(25, 25))
    plot0(a, b, ax[0, 0])
    # plot1(a,b, ax[1,0])
    plot2(a, b, ax[1, 0])
    pcomRT(a, b, ax[2, 1])
    _, ymax = ax[2, 1].get_ylim()
    if ymax > 0.3:
        ax[2, 1].set_ylim(-0.05, 0.305)
    # pcomRT_proactive_only(a, b, ax[1, 2])

    # p rev matrix
    b['rev'] = 0
    b.loc[(b.prechoice != b.R_response) & (b.reactive == 0), 'rev'] = 1
    subset = b.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.rev,
        flip=True,
        ax=ax[1, 2],
        cmap="magma",
        fmt=".0f",
        vmin=0
    )
    ax[1, 2].set_title('p(rev)')

    # from proactive reversals, which are detected as CoM?
    subset = b.loc[(b.rev == 1) & (b.reactive == 0)].dropna(
        subset=["avtrapz", "allpriors", "CoM_sugg"])
    plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        flip=True,
        ax=ax[1, 3],
        cmap="magma",
        fmt=".0f",
        vmin=0
    )
    ax[1, 3].set_title('p(com) in proactive reversals')
    # pcomRT but with rev and detected com [1,1]
    plotting.binned_curve(
        b[b.reactive == 0],
        "rev",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax[1, 1],
        errorbar_kw=dict(label="CoM", color="tab:olive"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls=":"),
    )
    plotting.binned_curve(
        b[b.reactive == 0],
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax[1, 1],
        errorbar_kw=dict(label="detected-CoM", color="tab:purple"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls="-"),
    )
    ax[1, 1].legend(frameon=False, fancybox=False)
    ax[1, 1].set_title('reversing proactive vs RT')

    plot3(a, b, ax[2, 0])  # ushape
    plot4(a, b, ax[0, 1])  # fraction of proactive responses?
    try:
        splitplot(a, b, ax[4, 0], ax[4, 1], ax[4, 2])
        ax[4, 0].axhline(params['jerk_lock_ms'], color='gray',
                         ls=':', label='jerk lock')
        ax[4, 0].axhline(params['t_update'], color='r',
                         ls=':', label='t_update')
        ax[4, 0].axhline((t_0_e-0.3)*1000, color='teal', ls=':', label='t_e')
        ax[4, 0].legend(fancybox=False, frameon=False)

    except Exception as e:
        print("splitplot typically crashes with silent trials" +
              "\nbecause in dani's tasks they all have the same coh")
        print(e)

    plot_com_contour(a, b, ax[3, 1])
    ax[3, 1].set_title('CoM peak moment')
    ax[3, 1].set_xlabel('MT (ms)', fontsize=14)
    ax[3, 1].set_ylabel('prior', fontsize=14)
    try:
        plot_median_com_traj(a, b, ax[3, 2])
    except:
        print('no CoMs found')

    tacho_kws = dict(
        rtbins=np.arange(0, 151, 5),
        labels=[f'sstr {x}' for x in [0, .25, .5, 1]],
        fill_error=True
    )
    plotting.tachometric(a, ax=ax[0, 2], **tacho_kws)
    ax[0, 2].set_title('Rats')

    ax[0, 2].set_ylabel('accuracy', fontsize=14)
    ax[0, 3].set_ylabel('accuracy', fontsize=14)
    plotting.tachometric(b, ax=ax[0, 3], **tacho_kws)
    ax[0, 3].set_title('Model simul')
    ax[0, 2].sharey(ax[0, 3])
    for i in [2, 3]:
        ax[0, i].set_xlabel('RT (ms)', fontsize=14)
        ax[0, i].legend(frameon=False, fancybox=False)

    plot1112(a, b, ax[2, 2], ax[2, 3])
    for dset, label, col in [[a, 'data', 'tab:blue'], [b, 'simul', 'tab:orange']]:
        plotting.binned_curve(
            dset, 'CoM_sugg', 'origidx', np.linspace(0, 600, 61),
            xpos=np.arange(5, 600, 10),
            ax=ax[3, 0], errorbar_kw=dict(color=col, label=label), legend=False
        )
    ax[3, 0].legend(frameon=False, fancybox=False)
    ax[3, 0].set_ylabel('p(detected-CoM)', fontsize=14)
    ax[3, 0].set_xlabel('trial index', fontsize=14)
    # t update distribution
    sns.histplot(
        data=b.loc[(b.reactive == 0) & (b.rtbin <= 12)], x='t_update',
        hue='rtbin', cumulative=True, element='step', fill=False,
        stat="density", common_norm=False, common_bins=True,
        ax=ax[3, 3], legend=False
    )
    ax[3, 3].set_title('effective t_update since movement onset')
    ax[3, 3].legend(frameon=False, fancybox=False)
    # sns.histplot(
    #    data=b[b.reactive==0], x='origidx', y='expectedMT', ax=ax[2,1]
    # )

    fig.suptitle(pref_title + subject + " " + str(params))
    # suptitle
    fname = f"{pref_title}{subject}-"
    for k, i in params.items():
        fname += f"{k}-{i}-"
    fname = fname[:-1] + ".png"
    if not return_matrices:
        fig.savefig(f"{savpath}{subject}.png")
        plt.show()
        return df, out
    else:
        fig.savefig(f"{savpath}{fname}")
        plt.show()
        return df, out, e_mat_list, u_mat_list  # matrices as well


def p_rev_pro(rt, a_e=0.5, allpriors=0, glm2Ze_scaling=0.1, confirm_thr=0.1,
              k_iters=5):
    """calculates probability to revert initial choice given that it was a
    proactive trial. It wil flip it for negative priors, uses 0 drift
    rt: reaction time (ms)
    a_e: semibound (scaled)
    allpriors: sum of priors in left-right space
    glm2Ze_scaling: factor to scale prior estimate
    confirm_thr: threshold to overcome in order to revert
    k: iters for infinite sum
    """
    # start with drift=0, then update if possible
    t = rt/1000
    x_0 = abs(allpriors*glm2Ze_scaling * a_e) + a_e
    rev_thr = a_e - confirm_thr*a_e
    a = a_e * 2

    iterable = np.c_[np.arange(1, k_iters), -1*np.arange(1, k_iters)].flatten()
    iterable = np.insert(iterable, 0, 0)

    prob_list = []
    for k in iterable:
        first_ = norm(loc=2*k*a+x_0, scale=t)
        first = np.subtract(
            *first_.cdf([rev_thr, 0])
        )
        second_ = norm(loc=2*k*a-x_0, scale=t)
        second = np.subtract(
            *second_.cdf([rev_thr, 0])
        )

        prob_list += [first - second]

    p_wo_drift = np.array(prob_list).sum()

    # do drift stuff here if required [oh no cannot with scipy]
    return p_wo_drift


def p_rev_pro2(rt, a_e=0.5, drift=0, allpriors=0, glm2Ze_scaling=0.1,
               confirm_thr=0.1, k_iters=5, normalize=False,
               return_normaliz_factor=False):
    """calculates probability to revert initial choice given that it was a
    proactive trial. It wil flip it for negative priors, uses 0 drift
    rt: reaction time (ms), converted to seconds internally
    a_e: semibound (scaled)
    drift:
    allpriors: sum of priors in left-right space
    glm2Ze_scaling: factor to scale prior estimate
    confirm_thr: threshold to overcome in order to revert
    k: iters for infinite sum
    normalize: (bool= whether to normalize), (float= normalizing factor)
    """
    if return_normaliz_factor:
        assert normalize, "to return normaliz factor requires normalize=True arg"
    norm_factor = 1.0
    # add external normalization so it can be reused
    if isinstance(normalize, float):
        norm_factor = normalize  # store value
        normalize = False  # set the flag to false so

    if not allpriors:
        allpriors = 1e-6
    t = rt/1000
    if allpriors < 0:  # prechoice left we will invert all the scheme
        drift *= -1
    x_0 = abs(allpriors*glm2Ze_scaling * a_e) + \
        a_e  # scale prior/x_0 and shift it,
    # so that lower bound is 0, top is a and middle/threshold is a_e
    rev_thr = a_e - confirm_thr*a_e
    a = a_e * 2

    iterable = np.c_[np.arange(1, k_iters), -1*np.arange(1, k_iters)].flatten()
    iterable = np.insert(iterable, 0, 0)

    prob_list = []
    if normalize:
        prob_normed = []
    for k in iterable:

        first_ = norm(loc=(2*k*a+x_0+drift*t), scale=t)
        first = np.subtract(
            *first_.cdf([rev_thr, 0])
        )
        second_ = norm(loc=(2*k*a-x_0+drift*t), scale=t)
        second = np.subtract(
            *second_.cdf([rev_thr, 0])
        )
        if normalize:
            first_n = np.subtract(*first_.cdf([a, 0]))
            second_n = np.subtract(*second_.cdf([a, 0]))
            prob_normed += [np.exp(2*k*a*drift) * (
                first_n - np.exp(-2*x_0*drift) * second_n)]

        prob_list += [np.exp(2*k*a*drift) * (
            first - np.exp(-2*x_0*drift) * second)]

    p = np.array(prob_list).sum()
    if not normalize:
        to_return = p/norm_factor
    else:
        to_return = p/np.array(prob_normed).sum()
    if return_normaliz_factor:
        return [to_return, np.array(prob_normed)]
    else:
        return to_return


def threaded_particle_(args):
    """see prob_rev function"""
    k, a, x_0, drift, t, rev_thr = args  # unpack vars
    first_ = norm(loc=2*k*a+x_0+drift*t, scale=t)
    first = np.subtract(
        *first_.cdf([rev_thr, 0])
    )
    second_ = norm(loc=2*k*a-x_0+drift*t, scale=t)
    second = np.subtract(
        *second_.cdf([rev_thr, 0])
    )
    return np.exp(2*k*a*drift) * (first - np.exp(2*x_0*drift) * second)


def threaded_particle_norm_(args):
    """see prob_rev function"""
    k, a, x_0, drift, t, rev_thr = args  # unpack vars
    first_ = norm(loc=2*k*a+x_0+drift*t, scale=t)
    first = np.subtract(
        *first_.cdf([rev_thr, 0])
    )
    second_ = norm(loc=2*k*a-x_0+drift*t, scale=t)
    second = np.subtract(
        *second_.cdf([rev_thr, 0])
    )
    first_n = np.subtract(*first_.cdf([a, 0]))
    second_n = np.subtract(*second_.cdf([a, 0]))

    p = np.exp(2*k*a*drift) * (
        first - np.exp(2*x_0*drift) * second)
    marginaliz = np.exp(2*k*a*drift) * (
        first_n - np.exp(2*x_0*drift) * second_n)

    return (p, marginaliz)


# slower than original one
def prob_rev(rt, a_e=0.5, drift=0, allpriors=0, glm2Ze_scaling=0.1,
             confirm_thr=0.1, k_iters=5, normalize=False, nworkers=7):
    """same than above but using threads
    calculates probability to revert initial choice given that it was a
    proactive trial. It wil flip it for negative priors, uses 0 drift
    rt: reaction time (ms)
    a_e: semibound (scaled)
    drift:
    allpriors: sum of priors in left-right space
    glm2Ze_scaling: factor to scale prior estimate
    confirm_thr: threshold to overcome in order to revert
    k: iters for infinite sum
    """
    if not allpriors:
        allpriors = 1e-6
    t = rt/1000
    if allpriors < 0:  # prechoice left we will invert all the scheme
        drift *= -1
    x_0 = abs(allpriors*glm2Ze_scaling * a_e) + a_e
    rev_thr = a_e - confirm_thr*a_e
    a = a_e * 2

    iterable = np.c_[np.arange(1, k_iters), -1*np.arange(1, k_iters)].flatten()
    iterable = np.insert(iterable, 0, 0)

    if normalize:
        threadfun = threaded_particle_norm_
        norm_probs = []
    else:
        threadfun = threaded_particle_

    probs = []
    with ThreadPoolExecutor(max_workers=nworkers) as executor:
        jobs = [
            executor.submit(
                threadfun,
                [
                    k, a, x_0, drift, t, rev_thr
                ],
            )
            for k in iterable
        ]
        if normalize:
            for job in jobs:
                res = job.result()
                probs += [res[0]]
                norm_probs += [res[1]]
        else:
            for job in jobs:
                probs += [job.result()]

    p = np.array(probs).sum()
    if not normalize:
        return p
    else:
        return p/np.array(norm_probs).sum()
