# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran

2 Readout model:
Version 2 of extended_ddm in which we updated:
    - Default MT fit parameters (slope/intercept)
    - Change of mind readout
    - AI parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import glob
import time
import sys
import multiprocessing as mp
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.pylab as pl
# sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils/")  # Cluster Alex
# import utilsJ
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve,\
    com_heatmap_paper_marginal_pcom_side
# from simul import splitplot
# import os
# SV_FOLDER = '/archive/molano/CoMs/'  # Cluster Manuel
# SV_FOLDER = '/home/garciaduran/'  # Cluster Alex
# SV_FOLDER = '/home/molano/Dropbox/project_Barna/ChangesOfMind/'  # Manuel
SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper'  # Alex
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
# DATA_FOLDER = '/archive/molano/CoMs/data/'  # Cluster Manuel
# DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
BINS = np.linspace(1, 301, 11)


def tests_trajectory_update(remaining_time=100, w_updt=10):
    """
    Evaluate options for trajectory update

    Parameters
    ----------
    remaining_time : int, optional
        DESCRIPTION. The default is 100.
    w_updt : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    None.

    """
    f, ax = plt.subplots()
    leg = ['1st response = -1', '1st response = 1']
    for i_s1r, s1r in enumerate([-1, 1]):
        CoM_bound = -0.5*np.sign(s1r)
        bound = np.sign(s1r)
        evidences = np.linspace(CoM_bound, bound, 50)
        ax.plot(evidences, remaining_time - s1r*w_updt*evidences,
                label=leg[i_s1r])
        ax.axvline(x=CoM_bound, linestyle='--', color='k', lw=0.5)
        ax.axvline(x=bound, linestyle='--', color='k')

    ax.axhline(y=remaining_time, linestyle='--', color='k', lw=0.5)
    ax.axvline(x=0, linestyle='--', color='k', lw=0.5)
    ax.set_xlabel('Update evidence')
    ax.set_ylabel('2nd response time')
    ax.legend()


def rm_top_right_lines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def draw_lines(ax, frst, sec, p_t_aff, p_com_bound, clrs_ro=['c', 'c']):
    ax[0].axhline(y=1, color='purple', linewidth=2)
    ax[0].axhline(y=-1, color='green', linewidth=2)
    ax[0].axhline(y=0, linestyle='--', color='k', linewidth=0.7)
    ax[0].axhline(y=p_com_bound, color='grey', linewidth=.5, linestyle='--')
    ax[0].axhline(y=-p_com_bound, color='grey', linewidth=.5, linestyle='--')
    ax[1].axhline(y=1, color='k', linewidth=1, linestyle='--')
    for a in ax:
        a.axvline(x=frst, color=clrs_ro[0], linewidth=1, linestyle='--')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.axvline(x=sec, color=clrs_ro[1], linewidth=1, linestyle='--')


def plotting(com, E, A, second_ind, first_ind, resp_first, resp_fin, pro_vs_re,
             p_t_aff, init_trajs, total_traj, p_t_eff, frst_traj_motor_time,
             tr_index, p_com_bound, stim_res=50, trial=0):
    f, ax = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
    ax = ax.flatten()
    ax[8].set_xlabel('Time (ms)')
    ax[9].set_xlabel('Time (ms)')
    ax[10].set_xlabel('Time (ms)')
    ax[11].set_xlabel('Time (ms)')
    axes = [np.array([0, 4, 8])+i for i in range(4)]
    trials = [0, 0, 0, 0]
    mat_indx = [np.logical_and(com, pro_vs_re == 0),
                np.logical_and(~com, pro_vs_re == 0),
                np.logical_and(com, pro_vs_re == 1),
                np.logical_and(~com, pro_vs_re == 1)]
    y_lbls = ['CoM Proactive', 'No CoM Proactive', 'CoM Reactive',
              'No CoM Reactive']
    for i_ax in range(4):
        ax[i_ax].set_title(y_lbls[i_ax])
    max_xlim = 0
    for i, (a, t, m, l) in enumerate(zip(axes, trials, mat_indx, y_lbls)):
        trials_temp = np.where(m)[0]
        traj_in = False
        for tr in trials_temp:
            if tr in tr_index:
                trial_total = int(np.where(tr_index == tr)[0])
                trial = tr
                traj_in = True
                break
        if len(trials_temp) > 0 and traj_in:
            draw_lines(ax[np.array(a)], frst=first_ind[trial]*stim_res,
                       sec=second_ind[trial]*stim_res, p_t_aff=p_t_aff*stim_res,
                       p_com_bound=p_com_bound)
            color1 = 'green' if resp_first[trial] < 0 else 'purple'
            color2 = 'green' if resp_fin[trial] < 0 else 'purple'

            x_2 = np.arange(second_ind[trial]+1)*stim_res
            x_1 = np.arange(first_ind[trial]+1)*stim_res
            ax[a[0]].plot(x_2, E[:second_ind[trial]+1, trial], color=color2,
                          alpha=0.7)
            ax[a[0]].plot(x_1, E[:first_ind[trial]+1, trial], color=color1, lw=2)
            ax[a[1]].plot(x_2, A[:second_ind[trial]+1, trial], color=color2,
                          alpha=0.7)
            ax[a[1]].plot(x_1, A[:first_ind[trial]+1, trial], color=color1, lw=2)
            # ax[a[0]].set_ylim([-1.5, 1.5])
            # ax[a[1]].set_ylim([-0.1, 1.5])
            ax[a[0]].set_ylabel(l+' EA')
            ax[a[1]].set_ylabel(l+' AI')
            # trajectories
            sec_ev = round(E[second_ind[trial], trial], 2)
            # updt_motor = first_ind[trial]+frst_traj_motor_time[trial]
            init_motor = first_ind[trial]+p_t_eff
            xs = init_motor*stim_res+np.arange(0, len(total_traj[trial_total]))
            max_xlim = max(max_xlim, np.max(xs))
            ax[a[2]].plot(xs, total_traj[trial_total],
                          label='Updated traj., E:{}'.format(sec_ev))
            first_ev = round(E[first_ind[trial], trial], 2)
            xs = init_motor*stim_res+np.arange(0, len(init_trajs[trial_total]))
            max_xlim = max(max_xlim, np.max(xs))
            ax[a[2]].plot(xs, init_trajs[trial_total],
                          label='Initial traj. E:{}'.format(first_ev))
            ax[a[2]].set_ylabel(l+', y(px)')
            ax[a[2]].set_ylabel(l+', y(px)')
            ax[a[2]].legend()
        else:
            print('There are no '+l)
    for a in ax:
        a.set_xlim([0, max_xlim])
    f.savefig(SV_FOLDER+'/figures/example_trials.svg', dpi=400,
              bbox_inches='tight')


def plot_misc(data_to_plot, stim_res, all_trajs=True, data=False):
    """


    Parameters
    ----------
    data_to_plot : TYPE
        DESCRIPTION.
    stim_res : TYPE
        DESCRIPTION.
    all_trajs : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    if data:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        ax = ax.flatten()
        data_to_df = {key: data_to_plot[key] for key in ['CoM', 'sound_len',
                                                         'hithistory',
                                                         'avtrapz',
                                                         'final_resp',
                                                         'MT', 'trial_idxs',
                                                         'subjid']}
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        ax = ax.flatten()
        data_to_plot['re_vs_pro'] = (data_to_plot['pro_vs_re'] - 1)*(-1)
        data_to_df = {key: data_to_plot[key] for key in ['CoM', 'sound_len',
                                                         'detected_com',
                                                         'hithistory',
                                                         'avtrapz',
                                                         'final_resp',
                                                         'MT', 'trial_idxs',
                                                         're_vs_pro',
                                                         'subjid']}
    df_plot = pd.DataFrame(data_to_df)
    xpos = int(np.diff(BINS)[0]+1)
    binned_curve(df_plot, 'CoM', 'sound_len', bins=BINS,
                 xpos=xpos, ax=ax[0], errorbar_kw={'label': 'CoM'})
    if not data:
        binned_curve(df_plot, 'detected_com', 'sound_len',
                     bins=BINS, ax=ax[0], xpos=xpos,
                     errorbar_kw={'label': 'detected com'})
        data_curve = pd.read_csv(SV_FOLDER + '/results/pcom_vs_rt.csv')
        ax[0].plot(data_curve['rt'], data_curve['pcom'], label='data',
                   linestyle='', marker='o')
    ax[0].legend()
    ax[0].set_xlabel('RT (ms)')
    ax[0].set_ylabel('PCoM')
    tachometric(df_plot, ax=ax[1], fill_error=True)
    ax[1].set_xlabel('RT (ms)')
    ax[1].set_ylabel('Accuracy')
    psych_curve(df_plot.hithistory, np.abs(df_plot.avtrapz), ret_ax=ax[2])
    ax[2].set_xlabel('Evidence')
    ax[2].set_ylabel('Accuracy')
    ax[2].set_xlim(-0.05, 1.05)
    psych_curve((df_plot.final_resp+1)/2, df_plot.avtrapz, ret_ax=ax[3])
    ax[3].set_xlabel('Evidence')
    ax[3].set_ylabel('Probability of right')
    f, ax = plt.subplots()
    bins = np.linspace(-300, 400, 70)
    if not data:
        hist_pro, _ = np.histogram(data_to_plot['sound_len']
                                   [data_to_plot['pro_vs_re'] == 0], bins)
        hist_re, _ = np.histogram(data_to_plot['sound_len']
                                  [data_to_plot['pro_vs_re'] == 1], bins)
        ax.plot(bins[:-1]+(bins[1]-bins[0])/2, hist_pro, label='Pro',
                linestyle='--')
        ax.plot(bins[:-1]+(bins[1]-bins[0])/2, hist_re, label='Re',
                linestyle='--')
    hist_total, _ = np.histogram(data_to_plot['sound_len'], bins)
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2, hist_total,
            label='All')
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Counts')
    ax.legend()
    if not data:
        matrix = data_to_plot['matrix']
        detected_mat = data_to_plot['detected_mat']
    fig1, ax1 = plt.subplots(nrows=2, ncols=3, figsize=(12, 12))
    ax1 = ax1.flatten()
    # df_plot['MT'] = df_plot['MT']*stim_res
    binned_curve(df_plot, 'MT', 'sound_len',
                 bins=BINS, ax=ax1[0], xpos=xpos,
                 errorbar_kw={'label': 'MT'})
    ax1[0].set_xlabel('RT')
    ax1[0].set_ylabel('MT')
    bins_trial = np.linspace(0, 600, 11, dtype=int)
    if not data:
        ax1[1].set_ylabel('Detected CoM')
        binned_curve(df_plot, 'detected_com', 'trial_idxs',
                     bins=bins_trial, ax=ax1[1], xpos=60,
                     errorbar_kw={'label': 'detected_com'})
        binned_curve(df_plot, 're_vs_pro', 'sound_len',
                     bins=BINS, ax=ax1[2], xpos=xpos,
                     errorbar_kw={'label': 'proac_prop'})
        ax1[2].set_xlabel('RT')
        ax1[2].set_ylabel('Proac. proportion')
    if data:
        ax1[1].set_ylabel('CoM')
        binned_curve(df_plot, 'CoM', 'trial_idxs',
                     bins=bins_trial, ax=ax1[1], xpos=60,
                     errorbar_kw={'label': 'CoM'})
    ax1[1].set_xlabel('Trial index')
    bins_MT = np.linspace(50, 600, num=25, dtype=int)
    binned_curve(df_plot, 'CoM', 'MT',
                 bins=bins_MT, ax=ax1[3], xpos=15,
                 xoffset=80, errorbar_kw={'label': 'CoM'})
    if not data:
        binned_curve(df_plot, 'detected_com', 'MT',
                     bins=bins_MT, ax=ax1[3], xpos=15,
                     xoffset=80, errorbar_kw={'label': 'detected CoM'})
    ax1[3].legend()
    ax1[3].set_xlabel('MT')
    ax1[3].set_ylabel('pCoM')
    hist_MT, _ = np.histogram(data_to_plot['MT'], bins=bins_MT)
    ax1[4].plot(bins_MT[:-1]+(bins_MT[1]-bins_MT[0])/2, hist_MT,
                label='MT dist')
    ax1[4].set_xlabel('MT (ms)')
    if not data:
        zt = data_to_plot['zt'][data_to_plot['pro_vs_re'] == 0]
        coh = data_to_plot['avtrapz'][data_to_plot['pro_vs_re'] == 0]
        com = data_to_plot['CoM'][data_to_plot['pro_vs_re'] == 0]
        mat_proac, _ = com_heatmap_jordi(zt, coh, com, return_mat=True, flip=True)
    if not data:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        ax = ax.flatten()
        sns.heatmap(matrix, ax=ax[0])
        ax[0].set_title('pCoM simulation')
        detected_mat[np.isnan(detected_mat)] = 0
        sns.heatmap(detected_mat, ax=ax[1])
        ax[1].set_title('Detected proportion')
        sns.heatmap(mat_proac, ax=ax[2])
        ax[2].set_title('pCoM in proac. trials')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    fig.suptitle('Stim/prior')
    sound_len = data_to_plot['sound_len']
    window = [0, 50, 100, 125, 150]
    if data:
        tit = 'Data: '
    else:
        tit = 'Model: '
    for i in range(4):
        zt = data_to_plot['zt'][(sound_len > window[i]) *
                                (sound_len < window[i+1])]
        coh = data_to_plot['avtrapz'][(sound_len > window[i]) *
                                      (sound_len < window[i+1])]
        if data:
            com = data_to_plot['CoM'][(sound_len > window[i]) *
                                      (sound_len < window[i+1])]
        if not data:
            com = data_to_plot['detected_com'][(sound_len > window[i]) *
                                               (sound_len < window[i+1])]
        com_heatmap_jordi(zt, coh, com, return_mat=False, flip=True, ax=ax[i],
                          annotate=False, cmap='rocket')
        ax[i].set_title('{} {} < RT < {}'.format(tit, window[i], window[i+1]))
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    fig.suptitle('Stim/prior congruent with final decision')
    sound_len = data_to_plot['sound_len']
    window = [0, 50, 100, 125, 150]
    if data:
        tit = 'Data: '
    else:
        tit = 'Model: '
    for i in range(4):
        zt = data_to_plot['zt'][(sound_len > window[i]) *
                                (sound_len < window[i+1])]
        zt *= data_to_plot['final_resp'][(sound_len > window[i]) *
                                         (sound_len < window[i+1])]
        coh = data_to_plot['avtrapz'][(sound_len > window[i]) *
                                      (sound_len < window[i+1])]
        coh *= data_to_plot['final_resp'][(sound_len > window[i]) *
                                          (sound_len < window[i+1])]
        if data:
            com = data_to_plot['CoM'][(sound_len > window[i]) *
                                      (sound_len < window[i+1])]
        if not data:
            com = data_to_plot['detected_com'][(sound_len > window[i]) *
                                               (sound_len < window[i+1])]
        com_heatmap_jordi(zt, coh, com, annotate=False,
                          return_mat=False, flip=True, ax=ax[i],
                          xlabel='prior congruency',
                          ylabel='avg stim congruency', cmap='rocket')
        ax[i].set_title('{} {} < RT < {}'.format(tit, window[i], window[i+1]))
    fig, ax = plt.subplots(1)
    zt = data_to_plot['zt'] * data_to_plot['final_resp']
    coh = data_to_plot['avtrapz'] * data_to_plot['final_resp']
    if data:
        com = data_to_plot['CoM']
    if not data:
        com = data_to_df['detected_com']
    decision = data_to_plot['final_resp']
    com_heatmap_jordi(zt, coh, com, ax=ax, annotate=False, return_mat=False,
                      flip=True, xlabel='prior congruency',
                      ylabel='avg stim congruency', cmap='rocket')
    left_right_matrix(zt, coh, com, decision)


def com_heatmap_jordi(x, y, com, flip=False, annotate=True,
                      predefbins=None, return_mat=False,
                      folding=False, annotate_div=1, ax=None,
                      cbar_location='right',
                      xlabel='prior', ylabel='average stim', **kwargs):
    """x: priors; y: av_stim, com_col, Flip (for single matrx.),all calculated
    from tmp dataframe
    g = sns.FacetGrid(df[df.special_trial==0].dropna(
        subset=['avtrapz', 'rtbins']), col='rtbins',
        col_wrap=3, height=5, sharex=False)
    g = g.map(plotting.com_heatmap, 'norm_prior','avtrapz','CoM_sugg',
              vmax=.15).set_axis_labels('prior', 'average stim')

    annotate_div= number to divide
    """
    tmp = pd.DataFrame(np.array([x, y, com]).T, columns=["prior", "stim", "com"])

    # make bins
    tmp["binned_prior"] = 0
    maxedge_prior = tmp.prior.abs().max()
    if predefbins is None:
        predefbinsflag = False
        bins = np.linspace(-maxedge_prior - 0.01, maxedge_prior + 0.01, 8)
    else:
        predefbinsflag = True
        bins = np.asarray(predefbins[0])

    tmp.loc[:, "binned_prior"], priorbins = pd.cut(
        tmp.prior, bins=bins, retbins=True,
        labels=np.arange(bins.size-1), include_lowest=True)

    tmp.loc[:, "binned_prior"] = tmp.loc[:, "binned_prior"].astype(int)
    priorlabels = [round((priorbins[i] + priorbins[i + 1]) / 2, 2)
                   for i in range(bins.size-1)]

    tmp["binned_stim"] = 0
    maxedge_stim = tmp.stim.abs().max()
    if not predefbinsflag:
        bins = np.linspace(-maxedge_stim - 0.01, maxedge_stim + 0.01, 8)
    else:
        bins = np.asarray(predefbins[1])
    tmp.loc[:, "binned_stim"], stimbins = pd.cut(
        tmp.stim, bins=bins, retbins=True, labels=np.arange(bins.size-1),
        include_lowest=True)
    tmp.loc[:, "binned_stim"] = tmp.loc[:, "binned_stim"].astype(int)
    stimlabels = [round((stimbins[i] + stimbins[i + 1]) / 2, 2)
                  for i in range(bins.size-1)]

    # populate matrices
    matrix = np.zeros((len(stimlabels), len(priorlabels)))
    nmat = matrix.copy()
    plain_com_mat = matrix.copy()
    for i in range(len(stimlabels)):
        switch = (tmp.loc[(tmp.com == 1) & (tmp.binned_stim == i)]
                  .groupby("binned_prior")["binned_prior"].count())
        for ind in tmp.binned_prior.unique():
            if ind not in switch.index.values:
                switch[ind] = 0
        nobs = (switch + tmp.loc[(tmp.com == 0) & (tmp.binned_stim == i)]
                .groupby("binned_prior")["binned_prior"].count())
        for i_n, n in enumerate(nobs.isnull()):
            if n:
                nobs[nobs.index[i_n]] = switch[switch.index[i_n]]
        # fill where there are no CoM (instead it will be nan)
        nobs.loc[nobs.isna()] = (tmp.loc[(tmp.com == 0) &
                                         (tmp.binned_stim == i)]
                                 .groupby("binned_prior")["binned_prior"]
                                 .count()
                                 .loc[nobs.isna()])  # index should be the same!
        crow = switch / nobs  # .values
        nmat[i, nobs.index.astype(int)] = nobs
        plain_com_mat[i, switch.index.astype(int)] = switch.values
        matrix[i, crow.index.astype(int)] = crow

    if folding:  # get indexes
        iu = np.triu_indices(len(stimlabels), 1)
        il = np.tril_indices(len(stimlabels), -1)
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
        matrix = np.flipud(matrix)
        # matrix is com/obs, nmat is number of observations
        return matrix, nmat

    if isinstance(annotate, str):
        if annotate == 'com':
            annotate = True
            annotmat = plain_com_mat/annotate_div
        if annotate == 'counts':
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
            g = sns.heatmap(np.flipud(matrix), annot=np.flipud(annotmat), ax=ax,
                            cbar_kws=dict(use_gridspec=False,
                                          location=cbar_location),
                            **kwargs).set(xlabel=xlabel,
                                          ylabel=ylabel,
                                          xticks=np.arange(len(priorlabels))+0.5,
                                          yticks=np.arange(len(stimlabels))+0.5,
                                          xticklabels=priorlabels,
                                          yticklabels=np.flip(stimlabels))

        else:
            g = sns.heatmap(np.flipud(matrix), ax=ax, annot=None,
                            cbar_kws=dict(use_gridspec=False,
                                          location=cbar_location), **kwargs).set(
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=np.arange(len(priorlabels))+0.5,
                yticks=np.arange(len(stimlabels))+0.5,
                xticklabels=priorlabels,
                yticklabels=np.flip(stimlabels),
            )
    else:
        if annotate:
            g = sns.heatmap(matrix, ax=ax, annot=annotmat,
                            cbar_kws=dict(use_gridspec=False,
                                          location=cbar_location), **kwargs).set(
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=np.arange(len(priorlabels))+0.5,
                yticks=np.arange(len(stimlabels))+0.5,
                xticklabels=priorlabels,
                yticklabels=stimlabels,
            )
        else:
            g = sns.heatmap(matrix, ax=ax, annot=None,
                            cbar_kws=dict(use_gridspec=False,
                                          location=cbar_location), **kwargs).set(
                xlabel=xlabel,
                ylabel=ylabel,
                xticks=np.arange(len(priorlabels))+0.5,
                yticks=np.arange(len(stimlabels))+0.5,
                xticklabels=priorlabels,
                yticklabels=stimlabels,
            )

    return g


def v_(t):
    return t.reshape(-1, 1) ** np.arange(6)


def get_Mt0te(t0, te):
    """

    Parameters
    ----------
    t0 : TYPE
        DESCRIPTION.
    te : TYPE
        DESCRIPTION.

    Returns
    -------
    Mt0te : TYPE
        DESCRIPTION.

    """
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


def compute_traj(jerk_lock_ms, mu, resp_len):
    """

    Parameters
    ----------
    jerk_lock_ms : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    resp_len : TYPE
        DESCRIPTION.

    Returns
    -------
    traj : TYPE
        DESCRIPTION.

    """
    t_arr = np.arange(jerk_lock_ms, resp_len)
    M = get_Mt0te(jerk_lock_ms, resp_len)
    M_1 = np.linalg.inv(M)
    vt = v_(t_arr)
    N = np.matmul(vt, M_1)
    traj = np.matmul(N, mu).ravel()
    traj = np.concatenate([[0]*jerk_lock_ms, traj])  # trajectory
    return traj


def get_data_and_matrix(dfpath='C:/Users/Alexandre/Desktop/CRM/Alex/paper/',
                        num_tr_per_rat=int(1e4), after_correct=True, silent=False,
                        splitting=False, all_trials=False, return_df=False,
                        sv_folder=None, srfail=False):
    # import data for 1 rat
    print('Loading data')
    sv_f = sv_folder or SV_FOLDER
    files = glob.glob(dfpath+'*.pkl')
    start = time.time()
    prior = np.empty((0, ))
    stim = np.empty((0, 20))
    com = np.empty((0, ))
    coh = np.empty((0, ))
    gt = np.empty((0, ))
    sound_len = np.empty((0, ))
    resp_len = np.empty((0, ))
    decision = np.empty((0, ))
    hit = np.empty((0, ))
    trial_index = np.empty((0, ))
    subject = np.empty((0, ))
    if splitting:
        traj_y = np.empty((0, ))
        traj_stamps = np.empty((0, ))
        fix_onset = np.empty((0, ), dtype=np.datetime64)
    if silent:
        special_trial = np.empty((0, ))
    for f in files:
        start_1 = time.time()
        df = pd.read_pickle(f)
        if return_df:
            if after_correct:
                if not silent:
                    if srfail:
                        return df.query(
                               "sound_len <= 400 and\
                                   resp_len <=1 and R_response>= 0\
                                       and hithistory >= 0 and special_trial == 0\
                                           and aftererror==0")
                    else:
                        return df.query(
                                "sound_len <= 400 and soundrfail ==\
                                    False and resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0 and special_trial == 0\
                                            and aftererror==0")
                if silent:
                    if srfail:
                        return df.query(
                                "sound_len <= 400 and \
                                    resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0\
                                            and aftererror==0")
                    else:
                        return df.query(
                                "sound_len <= 400 and soundrfail ==\
                                    False and resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0\
                                            and aftererror==0")
            else:
                if not silent:
                    if srfail:
                        return df.query(
                            "sound_len <= 400 and\
                            resp_len <=1 and R_response>= 0\
                            and hithistory >= 0 and special_trial == 0")
                    else:
                        return df.query(
                            "sound_len <= 400 and soundrfail ==\
                                False and resp_len <=1 and R_response>= 0\
                                    and hithistory >= 0 and special_trial == 0")
                if silent:
                    if srfail:
                        return df.query(
                         "sound_len <= 400 and\
                         resp_len <=1 and R_response>= 0\
                         and hithistory >= 0")
                    else:
                        return df.query(
                            "sound_len <= 400 and soundrfail ==\
                            False and resp_len <=1 and R_response>= 0\
                            and hithistory >= 0")
        if not silent:
            df = df.query(
                    "sound_len <= 400 and soundrfail ==\
                        False and resp_len <=1 and R_response>= 0\
                            and hithistory >= 0 and special_trial == 0")
        if silent:
            df = df.query(
                    "sound_len <= 400 and soundrfail ==\
                        False and resp_len <=1 and R_response>= 0\
                            and hithistory >= 0")
        end = time.time()
        if after_correct:
            indx_prev_error = np.where(df['aftererror'].values == 0)[0]
            if not all_trials:
                selected_indx = np.random.choice(np.arange(len(indx_prev_error)),
                                                 size=(num_tr_per_rat),
                                                 replace=False)
                indx = indx_prev_error[selected_indx]
            if all_trials:
                indx = indx_prev_error
        else:
            if all_trials:
                indx = np.arange(len(df))
            if not all_trials:
                indx = np.random.choice(np.arange(len(df)), size=(num_tr_per_rat),
                                        replace=False)
        prior_tmp = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        stim_tmp = np.array([stim for stim in df.res_sound])
        coh_mat = np.array(df.coh2)
        com_tmp = df.CoM_sugg.values
        decision_tmp = np.array(df.R_response) * 2 - 1
        sound_len_tmp = np.array(df.sound_len)
        trial_index_tmp = np.array(df.origidx)
        resp_len_tmp = np.array(df.resp_len)
        subject_tmp = df.subjid.values
        if silent:
            special_trial_tmp = np.array(df.special_trial)
        if splitting:
            traj_y_tmp = np.array(df.trajectory_y)
            traj_stamps_tmp = np.array(df.trajectory_stamps)
            fix_onset_tmp = np.array(df.fix_onset_dt)
        hit_tmp = np.array(df['hithistory'])
        gt_tmp = np.array(df.rewside) * 2 - 1
        prior = np.concatenate((prior, prior_tmp[indx]))
        stim = np.concatenate((stim, stim_tmp[indx, :]))
        coh = np.concatenate((coh, coh_mat[indx]))
        com = np.concatenate((com, com_tmp[indx]))
        gt = np.concatenate((gt, gt_tmp[indx]))
        decision = np.concatenate((decision, decision_tmp[indx]))
        sound_len = np.concatenate((sound_len, sound_len_tmp[indx]))
        resp_len = np.concatenate((resp_len, resp_len_tmp[indx]))
        hit = np.concatenate((hit, hit_tmp[indx]))
        trial_index = np.concatenate((trial_index, trial_index_tmp[indx]))
        subject = np.concatenate((subject, subject_tmp[indx]))
        if silent:
            special_trial = np.concatenate((special_trial,
                                            special_trial_tmp[indx]))
        if splitting:
            traj_y = np.concatenate((traj_y, traj_y_tmp[indx]))
            traj_stamps = np.concatenate((traj_stamps, traj_stamps_tmp[indx]))
            fix_onset = np.concatenate((fix_onset, fix_onset_tmp[indx]))
        end = time.time()
        print(f)
        print(end - start_1)
        print(len(df))
    print(end - start)
    print('Ended loading data, start computing matrix')
    time_trajs = get_trajs_time(resp_len, traj_stamps, fix_onset, com,
                                sound_len=sound_len)
    com_trajs, _, _, comlist = com_detection(traj_y, decision, time_trajs,
                                             com_threshold=8)
    comlist = np.array(comlist)
    # plot_com_methods(time_trajs, traj_y, com, comlist, subjname='LE44')
    com = comlist
    df_curve = {'CoM': com, 'sound_len': sound_len}
    df_curve = pd.DataFrame(df_curve)
    xpos = int(np.diff(BINS)[0])
    xpos_plot, median_pcom, _ =\
        binned_curve(df_curve, 'CoM', 'sound_len', xpos=xpos,
                     bins=BINS,
                     return_data=True)
    df_pcom_rt = pd.DataFrame({'rt': xpos_plot, 'pcom': median_pcom})
    df_pcom_rt.to_csv(sv_f + '/results/pcom_vs_rt.csv')
    matrix, _ = com_heatmap_jordi(prior, coh, com, return_mat=True, flip=True)
    np.save(sv_f + '/results/CoM_vs_prior_and_stim.npy', matrix)
    stim = stim.T
    com = com.astype(int)
    rt_vals, rt_bins = np.histogram(sound_len,
                                    bins=np.linspace(-100, 300, 81))
    np.save(sv_f + '/results/RT_distribution.npy', rt_vals)
    np.save(sv_f + '/results/RT_bins.npy', rt_bins)
    if not silent:
        special_trial = None
    if not splitting:
        traj_y = None
        fix_onset = None
        traj_stamps = None
    return stim, prior, coh, gt, com, decision, sound_len, resp_len,\
        hit, trial_index, special_trial, traj_y, fix_onset, traj_stamps,\
        subject


def trial_ev_vectorized(zt, stim, coh, trial_index, MT_slope, MT_intercep, p_w_zt,
                        p_w_stim, p_e_noise, p_com_bound, p_t_eff, p_t_aff,
                        p_t_a, p_w_a_intercept, p_w_a_slope, p_a_noise,
                        p_1st_readout, p_2nd_readout, num_tr, stim_res,
                        compute_trajectories=False, num_trials_per_session=600,
                        all_trajs=True, num_computed_traj=int(3e4),
                        fixation_ms=300):
    """
    Generate stim and time integration and trajectories

    Parameters
    ----------
    zt : array
        priors for each trial (transition bias + lateral (CWJ) 1xnum-trials).
    stim : array
        stim sequence for each trial 20xnum-trials.
    MT_slope : float
        slope corresponding to motor time and trial index linear relation (0.15).
    MT_intercep : float
        intercept corresponding to motor-time and trial index relation (110).
    p_w_zt : float
        fitting parameter: gain for prior (zt).
    p_w_stim : float
        fitting parameter: gain for stim (stim).
    p_e_noise : float
        fitting parameter: standard deviation of evidence noise (gaussian).
    p_com_bound : float
        fitting parameter: change-of-mind bound (will have opposite sign of
        first choice).
    p_t_eff : float
        fitting parameter: efferent latency to initiate movement.
    p_t_aff : float
        fitting parameter: afferent latency to integrate stimulus.
    p_t_a : float
        fitting parameter: latency for action integration.
    p_w_a_intercept : float
        fitting parameter: drift of action noise.
    p_a_noise : float
        fitting parameter: standard deviation of action noise (gaussian).
    p_1st_readout : float
        fitting parameter: slope of the linear realtion with time and evidence
        for trajectory update.
    num_tr : int
        number of trials.
    trajectories : boolean, optional
        Whether trajectories are computed or not. The default is False.

    Returns
    -------
    E : array
        evidence integration matrix (num_tr x stim.shape[0]).
    A : array
        action integration matrix (num_tr x stim.shape[0]).
    com : boolean array
        whether each trial is or not a change-of-mind (num_tr x 1).
    first_ind : list
        first choice indexes (num_tr x 1).
    second_ind : list
        second choice indexes (num_tr x 1).
    resp_first : list
        first choice (-1 if left and 1 if right, num_tr x 1).
    resp_fin : list
        second (final) choice (-1 if left and 1 if right, num_tr x 1).
    pro_vs_re : boolean array
        whether each trial is reactive or not (proactive) ( num_tr x 1).
    total_traj: tuple
        total trajectory of the rat, containing the update (num_tr x 1).
    init_trajs: tuple
        pre-planned trajectory of the rat.
    final_trajs: tuple
        trajectory after the update.

    """
    # print('Starting simulation, PSIAM')
    # start_eddm = time.time()
    # TODO: COMMENT EVERY FORKING LINE
    bound = 1
    bound_a = 2.2
    p_leak = 0.6
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    # add noise
    dW = np.random.randn(N, num_tr)*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # adding leak
    rolled_dW = np.roll(dW, 1)
    rolled_dW[fixation + p_t_aff, :] = 0
    dW += -rolled_dW*p_leak
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.cumsum(dW, axis=0)
    com = False
    # check docstring for definitions
    first_ind = []
    second_ind = []
    pro_vs_re = []
    resp_first = np.ones(E.shape[1])
    resp_fin = np.ones(E.shape[1])
    # evidences at 1st/2nd readout
    first_ev = []
    second_ev = []
    # start DDM
    for i_t in range(E.shape[1]):
        # search where evidence bound is reached
        indx_hit_bound = np.abs(E[:, i_t]) >= bound
        hit_bound = max_integration_time
        if (indx_hit_bound).any():
            hit_bound = np.where(indx_hit_bound)[0][0]
        # search where action bound is reached
        indx_hit_action = A[:, i_t] >= bound_a
        hit_action = max_integration_time
        if (indx_hit_action).any():
            hit_action = np.where(indx_hit_action)[0][0]
        # set first readout as the minimum
        hit_dec = min(hit_bound, hit_action)
        # XXX: this is not accurate because reactive trials are defined as
        # EA reaching the bound, which includes influence of zt
        pro_vs_re.append(np.argmin([hit_action, hit_bound]))
        # store first readout index
        first_ind.append(hit_dec)
        # store first readout evidence
        first_ev.append(E[hit_dec, i_t])
        # first categorical response
        resp_first[i_t] *= (-1)**(E[hit_dec, i_t] < 0)
        # CoM bound with sign depending on first response
        com_bound_signed = (-resp_first[i_t])*p_com_bound
        # second response
        indx_final_ch = hit_dec+p_t_eff+p_t_aff
        indx_final_ch = min(indx_final_ch, max_integration_time)
        # get post decision accumulated evidence with respect to CoM bound
        post_dec_integration = E[hit_dec:indx_final_ch, i_t]-com_bound_signed
        # get CoMs indexes
        # in this comparison, post_dec_integration is set with respect
        # to com_bound_signed but E[hit_dec, i_t] isn't. However,
        # it does not matter
        # because:
        # sign(E[hit_dec, i_t]) = sign(E[hit_dec, i_t]) - com_bound_signed
        indx_com =\
            np.where(np.sign(E[hit_dec, i_t]) != np.sign(post_dec_integration))[0]
        # get CoM effective index
        indx_update_ch = indx_final_ch if len(indx_com) == 0\
            else indx_com[0] + hit_dec
        # get final decision
        resp_fin[i_t] = resp_first[i_t] if len(indx_com) == 0 else -resp_first[i_t]
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_t])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind)
    pro_vs_re = np.array(pro_vs_re)
    rt_vals, rt_bins = np.histogram((first_ind-fixation+p_t_eff)*stim_res,
                                    bins=np.linspace(-100, 300, 81))
    matrix, _ = com_heatmap_jordi(zt, coh, com,
                                  return_mat=True, flip=True)
    # end_eddm = time.time()
    # print('Time for "PSIAM": ' + str(end_eddm - start_eddm))
    # XXX: put in a different function
    if compute_trajectories:
        # start_traj = time.time()
        # Trajectories
        # print('Starting with trajectories')
        RLresp = resp_fin
        prechoice = resp_first
        jerk_lock_ms = 0
        # initial positions, speed and acc; final position, speed and acc
        initial_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
        indx_trajs = np.arange(len(first_ind)) if all_trajs\
            else np.random.choice(len(first_ind), num_computed_traj)
        # check docstring for definitions
        init_trajs = []
        final_trajs = []
        total_traj = []
        # first trajectory motor time w.r.t. first readout
        frst_traj_motor_time = []
        # x value of trajectory at second readout update time
        x_val_at_updt = []
        for i_t in indx_trajs:
            # pre-planned Motor Time, the modulo prevents trial-index from
            # growing indefinitely
            MT = MT_slope*trial_index[i_t] + MT_intercep + 35*np.random.randn(1)
            first_resp_len = float(MT-p_1st_readout*np.abs(first_ev[i_t]))
            # first_resp_len: evidence influence on MT. The larger the ev,
            # the smaller the motor time
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                  resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            velocities = np.gradient(prior0)
            accelerations = np.gradient(velocities)  # acceleration
            t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
            t_updt = int(np.min((t_updt, len(velocities)-1)))
            frst_traj_motor_time.append(t_updt)
            vel = velocities[t_updt]  # velocity at the timepoint
            acc = accelerations[t_updt]
            pos = prior0[t_updt]  # position
            mu_update = np.array([pos, vel, acc, 75*RLresp[i_t],
                                  0, 0]).reshape(-1, 1)
            # new mu, considering new position/speed/acceleration
            remaining_m_time = first_resp_len-t_updt
            sign_ = resp_first[i_t]
            # this sets the maximum updating evidence equal to the ev bound
            # and avoids having negative second_resp_len (impossibly fast
            # responses) bc of very strong confirmation evidence.
            updt_ev = np.clip(second_ev[i_t], a_min=-1, a_max=1)
            # second_response_len: motor time update influenced by difference
            # between the evidence at second readout and the signed p_com_bound
            com_bound_signed = (-sign_)*p_com_bound
            offset = 140
            second_response_len =\
                float(remaining_m_time + offset*com[i_t] -
                      p_2nd_readout*(np.abs(updt_ev - com_bound_signed)))
            # SECOND readout
            traj_fin = compute_traj(jerk_lock_ms, mu=mu_update,
                                    resp_len=second_response_len)
            final_trajs.append(traj_fin)
            # joined trajectories
            traj_before_uptd = prior0[0:t_updt]
            traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
            # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
            # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
            # np.random.randn(np.random.randint(15, 60))*0.2
            total_traj.append(traj_updt)
            if com[i_t]:
                opp_side_values = traj_updt.copy()
                opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                max_val_towards_opposite = np.max(np.abs(opp_side_values))
                x_val_at_updt.append(max_val_towards_opposite)
            else:
                x_val_at_updt.append(0)
        detect_CoMs_th = 8
        detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
        df_curve = {'detected_CoM': detected_com,
                    'sound_len': (first_ind[indx_trajs]-fixation+p_t_eff)*stim_res}
        df_curve = pd.DataFrame(df_curve)
        xpos = int(np.diff(BINS)[0])
        xpos_plot, median_pcom, _ =\
            binned_curve(df_curve, 'detected_CoM', 'sound_len', xpos=xpos,
                         bins=BINS,
                         return_data=True)
        # end_traj = time.time()
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, None,\
            rt_vals, rt_bins, None


def run_model(stim, zt, coh, gt, com, trial_index, sound_len, traj_y, traj_stamps,
              fix_onset, configurations, jitters, stim_res,
              compute_trajectories=False, plot=False, existing_data=None,
              detect_CoMs_th=5, shuffle=False, all_trajs=False,
              kernels_model=False):
    """

    Parameters
    ----------
    stim : array
        stim sequence for each trial 20xnum-trials.
    zt : array
        priors for each trial (transition bias + lateral (CWJ) 1xnum-trials).
    coh : array
        Putative coherence (0, +-0.25, +-0.5 or +-1), 1 x num_trials.
    gt : array
        Ground truth for each trial (1 if right and -1 if left).
    configurations : list
        DESCRIPTION.
    jitters : list
        jitter (noise) for each parameter of the configurations.
    stim_res : float
        stimulus resolution (50/data_augmenting_factor) ms, normally 5 ms.
    compute_trajectories : boolean, optional
        If the trajectories have to be computed or not (DDM only).
        The default is False.
    plot : boolean, optional
        If plots are displayed or not. The default is False.
    existing_data : string (path), optional
        Path of the configurations already run. The default is None.
    detect_CoMs_th : float, optional
        Threshold for CoM detection. The default is 5.
    shuffle : boolean, optional
         if inputs have to be shuffled (zt, stim...). The default is False.
    all_trajs : bool, optional
        If all trajectories are computed or not (else 20000). The default is False.

    Returns
    -------
    None.

    """
    def save_data():
        data_final = {'p_w_zt': p_w_zt_vals, 'p_w_stim': p_w_stim_vals,
                      'p_e_noise': p_e_noise_vals,
                      'p_com_bound': p_com_bound_vals,
                      'p_t_aff': p_t_aff_vals, 'p_t_eff': p_t_eff_vals,
                      'p_t_a': p_t_a_vals, 'p_w_a_intercept': p_w_a_intercept_vals,
                      'p_w_a_slope': p_w_a_slope_vals,
                      'p_a_noise': p_a_noise_vals,
                      'p_1st_readout': p_1st_readout_vals,
                      'p_2nd_readout': p_2nd_readout_vals,
                      'pcom_matrix': all_mats,
                      'x_val_at_updt_mat': x_val_at_updt_mat,
                      'xpos_rt_pcom': xpos_rt_pcom,
                      'median_pcom_rt': median_pcom_rt,
                      'rt_vals_all': rt_vals_all,
                      'rt_bins_all': rt_bins_all}
        name = '_'.join([str(c) for c in configurations[0]])
        np.savez(SV_FOLDER+'/results/all_results_'+name+'.npz',
                 **data_final)

    if existing_data is not None:
        ex_data = np.load(existing_data, allow_pickle=1)
        done_confs = np.array([v[:, 0] for k, v in ex_data.items()
                               if k.startswith('p_')])
    else:
        done_confs = np.zeros((10))
    if shuffle:
        indx_sh = np.arange(len(zt))
        np.random.shuffle(indx_sh)
        stim = stim[:, indx_sh]
        zt = zt[indx_sh]
        coh = coh[indx_sh]
        gt = gt[indx_sh]
        trial_index = trial_index[indx_sh]
        if com is not None:
            com = com[indx_sh]
        if sound_len is not None:
            sound_len = sound_len[indx_sh]
        if traj_y is not None and traj_stamps is not None and fix_onset is not None:
            traj_y = traj_y[indx_sh]
            fix_onset = fix_onset[indx_sh]
            traj_stamps = traj_stamps[indx_sh]
    num_tr = stim.shape[1]
    MT_slope = 0.123
    MT_intercep = 254
    p_w_zt_vals = []
    p_w_stim_vals = []
    p_e_noise_vals = []
    p_com_bound_vals = []
    p_t_aff_vals = []
    p_t_eff_vals = []
    p_t_a_vals = []
    p_w_a_intercept_vals = []
    p_w_a_slope_vals = []
    p_a_noise_vals = []
    p_1st_readout_vals = []
    p_2nd_readout_vals = []
    all_mats = []
    x_val_at_updt_mat = []
    xpos_rt_pcom = []
    median_pcom_rt = []
    rt_vals_all = []
    rt_bins_all = []
    one_bins = True
    for i_conf, conf in enumerate(configurations):
        print('--------------')
        print('p_w_zt: '+str(conf[0]))
        print('p_w_stim: '+str(conf[1]))
        print('p_e_noise: '+str(conf[2]))
        print('p_com_bound: '+str(conf[3]))
        print('p_t_aff: '+str(conf[4]))
        print('p_t_eff: '+str(conf[5]))
        print('p_t_a: '+str(conf[6]))
        print('p_w_a_intercept: '+str(conf[7]))
        print('p_w_a_slope: '+str(conf[8]))
        print('p_a_noise: '+str(conf[9]))
        print('p_1st_readout: '+str(conf[10]))
        print('p_2nd_readout: '+str(conf[11]))
        start = time.time()
        if (np.sum(done_confs-np.array(conf).reshape(-1, 1), axis=0) != 0).all():
            p_w_zt = conf[0]+jitters[0]*np.random.rand()
            p_w_stim = conf[1]+jitters[1]*np.random.rand()
            p_e_noise = conf[2]+jitters[2]*np.random.rand()
            p_com_bound = conf[3]+jitters[3]*np.random.rand()
            p_t_aff = int(round(conf[4]+jitters[4]*np.random.rand()))
            p_t_eff = int(round(conf[5]++jitters[5]*np.random.rand()))
            p_t_a = int(round(conf[6]++jitters[6]*np.random.rand()))
            p_w_a_intercept = conf[7]+jitters[7]*np.random.rand()
            p_w_a_slope = conf[8]+jitters[8]*np.random.rand()
            p_a_noise = conf[9]+jitters[9]*np.random.rand()
            p_1st_readout = conf[10]+jitters[10]*np.random.rand()
            p_2nd_readout = conf[11]+jitters[11]*np.random.rand()
            stim_temp =\
                np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                stim.shape[1]))))
            # TODO: get in a dict
            E, A, com_model, first_ind, second_ind, resp_first, resp_fin,\
                pro_vs_re, matrix, total_traj, init_trajs, final_trajs,\
                frst_traj_motor_time, x_val_at_updt, xpos_plot, median_pcom,\
                rt_vals, rt_bins, tr_index =\
                trial_ev_vectorized(zt=zt, stim=stim_temp, coh=coh,
                                    trial_index=trial_index,
                                    MT_slope=MT_slope, MT_intercep=MT_intercep,
                                    p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                    p_e_noise=p_e_noise, p_com_bound=p_com_bound,
                                    p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                                    num_tr=num_tr, p_w_a_intercept=p_w_a_intercept,
                                    p_w_a_slope=p_w_a_slope,
                                    p_a_noise=p_a_noise,
                                    p_1st_readout=p_1st_readout,
                                    p_2nd_readout=p_2nd_readout,
                                    compute_trajectories=compute_trajectories,
                                    stim_res=stim_res, all_trajs=all_trajs)

            print(np.mean(com_model))
            if plot:
                start_plot = time.time()
                reaction_time = (first_ind[tr_index] + p_t_eff -
                                 300/stim_res)*stim_res
                detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
                MT = [len(t) for t in total_traj]
                # ind_com_model = detected_com.astype(int)
                # final_trajs = np.array(final_trajs)
                # out = pd.DataFrame({'trajectory_y': final_trajs[ind_com_model],
                #                     'sound_len': reaction_time[ind_com_model],
                #                     'coh2': coh[ind_com_model],
                #                     'rewside': (gt[ind_com_model]+1)/2})
                # ind_com_data = com.astype(bool)
                # df = pd.DataFrame({'trajectory_y': traj_y[ind_com_data],
                #                    'sound_len': sound_len[ind_com_data],
                #                    'coh2': coh[ind_com_data],
                #                    'rewside': (gt[ind_com_data]+1)/2,
                #                   'trajectory_stamps': traj_stamps[ind_com_data],
                #                    'fix_onset_dt': fix_onset[ind_com_data]})
                # fig, ax = plt.subplots(1)
                # fig, ax1 = plt.subplots(1)
                # fig, ax2 = plt.subplots(1)
                # splitplot(df, out, ax, ax1, ax2)
                if kernels_model:
                    for margin in [5, 15, 30, 50]:
                        coms_tII_tI(first_ind, tr_index, p_t_eff, p_t_aff,
                                    stim_res, com_model, detected_com, zt, MT,
                                    resp_fin, stim, margin=margin)
                if compute_trajectories:
                    plotting(com=com_model, E=E, A=A, second_ind=second_ind,
                             first_ind=first_ind,
                             resp_first=resp_first, resp_fin=resp_fin,
                             pro_vs_re=pro_vs_re,
                             p_t_aff=p_t_aff, init_trajs=init_trajs,
                             total_traj=total_traj,
                             p_t_eff=p_t_eff,
                             frst_traj_motor_time=frst_traj_motor_time,
                             tr_index=tr_index, p_com_bound=p_com_bound,
                             stim_res=stim_res)
                hits = resp_fin == gt
                detected_mat, _ =\
                    com_heatmap_jordi(zt[tr_index], coh[tr_index], detected_com,
                                      return_mat=True, flip=True)
                data_to_plot = {'sound_len': reaction_time,
                                'CoM': com_model[tr_index],
                                'first_resp': resp_first[tr_index],
                                'final_resp': resp_fin[tr_index],
                                'hithistory': hits[tr_index],
                                'avtrapz': coh[tr_index],
                                'detected_com': detected_com,
                                'pro_vs_re': pro_vs_re[tr_index],
                                'detected_mat': detected_mat,
                                'matrix': matrix,
                                'MT': MT,
                                'zt': zt[tr_index], 'decision': decision,
                                'trial_idxs': trial_index[tr_index],
                                'subjid': np.repeat('sim', len(tr_index))}
                left_right_matrix(zt[tr_index], coh[tr_index],
                                  detected_com, resp_fin[tr_index])
                pcom_model_vs_data(detected_com, com[tr_index],
                                   sound_len[tr_index], reaction_time)
                plot_misc(data_to_plot=data_to_plot, stim_res=stim_res)
                MT = np.array(MT)*1e-3
                mean_com_traj_peak(trajectories=None, com=detected_com,
                                   time_trajs=None, zt=zt,
                                   sound_len=reaction_time,
                                   decision=resp_fin, motor_time=MT,
                                   val_at_updt=x_val_at_updt, data=False,
                                   peak_cond=True)
                # MT_vs_ev(resp_len=MT, coh=coh[tr_index],
                #          com=detected_com)
                if kernels_model:
                    kernels(coh=coh[tr_index], zt=zt[tr_index],
                            sound_len=reaction_time,
                            decision=resp_fin[tr_index], stim=stim[:, tr_index],
                            com=detected_com, stim_res=stim_res)
                end_plot = time.time()
                print('Plotting time: ' + str(end_plot-start_plot))
            p_w_zt_vals.append([conf[0], p_w_zt])
            p_w_stim_vals.append([conf[1], p_w_stim])
            p_e_noise_vals.append([conf[2], p_e_noise])
            p_com_bound_vals.append([conf[3], p_com_bound])
            p_t_aff_vals.append([conf[4], p_t_aff])
            p_t_eff_vals.append([conf[5], p_t_eff])
            p_t_a_vals.append([conf[6], p_t_a])
            p_w_a_intercept_vals.append([conf[7], p_w_a_intercept])
            p_w_a_slope_vals.append([conf[8], p_w_a_slope])
            p_a_noise_vals.append([conf[9], p_a_noise])
            p_1st_readout_vals.append([conf[10], p_1st_readout])
            p_2nd_readout_vals.append([conf[11], p_2nd_readout])
            all_mats.append(matrix)
            x_val_at_updt_mat.append(x_val_at_updt)
            xpos_rt_pcom.append(xpos_plot)
            median_pcom_rt.append(median_pcom)
            rt_vals_all.append(rt_vals)
            if one_bins:
                rt_bins_all.append(rt_bins)
                one_bins = False
            if i_conf % 100 == 0:
                save_data()
        end = time.time()
        print(end-start)
    save_data()


def set_parameters(num_vals=3, factor=8):
    """
    p_t_aff = 8
    p_t_eff = 6
    p_t_a = 42
    p_w_zt = 0.15
    p_w_stim = 0.2
    p_e_noise = 0.06
    p_com_bound = 0.1
    p_w_a_intercept = 0.03
    p_a_noise = 0.04
    p_1st_readout = 5
    """
    p_w_zt_list = [0.15, 0.17, 0.2]
    p_w_stim_list = np.linspace(0.06, 0.12, num=num_vals)
    p_e_noise_list = [0.01]
    p_com_bound_list = [0.]
    p_t_aff_list = np.linspace(6, 10, num=num_vals+1, dtype=int)
    p_t_eff_list = np.linspace(6, 10, num=num_vals+1, dtype=int)
    p_t_a_list = [14]
    p_w_a_intercept_list = [0.052]
    p_w_a_slope_list = [2.2e-5]
    p_a_noise_list = [0.04]
    p_1st_readout_list = [20, 30, 40, 50]
    p_2nd_readout_list = [5, 10, 15, 20, 25]
    configurations = list(itertools.product(p_w_zt_list, p_w_stim_list,
                                            p_e_noise_list, p_com_bound_list,
                                            p_t_aff_list, p_t_eff_list, p_t_a_list,
                                            p_w_a_intercept_list,
                                            p_w_a_slope_list,
                                            p_a_noise_list,
                                            p_1st_readout_list,
                                            p_2nd_readout_list))
    if num_vals == 1:
        jitters = np.repeat(0.00001, 12)
    else:
        jitters = [1e-4,
                   1e-4,
                   0.0001,
                   0.001,
                   np.diff(p_t_aff_list)[0]/factor,
                   1e-4,
                   0.5,
                   0.0005,
                   1e-8,
                   0.0001,
                   1,
                   1]
    return configurations, jitters


def data_augmentation(stim, daf, sigma=0):
    """

    Parameters
    ----------
    stim : array
        stim sequence for each trial 20xnum-trials.
    daf : float
        data augmentation factor (quantity of precision augmentation)
    sigma : float, optional
        Weight of the random part of the augmented stimulus. The default is 0.

    Returns
    -------
    augm_stim : array
        Augmented stimulus, from precision of 50ms to precision of 50/daf ms.

    """
    augm_stim = np.zeros((daf*stim.shape[0], stim.shape[1]))
    for tmstp in range(stim.shape[0]):
        augm_stim[daf*tmstp:daf*(tmstp+1), :] =\
            np.random.randn()*sigma+stim[tmstp, :]
    return augm_stim


def plot_distributions(zt_filt, coh_filt, stim_filt, dec_filt, com_array):
    """

    Parameters
    ----------
    zt_filt : TYPE
        DESCRIPTION.
    coh_filt : TYPE
        DESCRIPTION.
    stim_filt : TYPE
        DESCRIPTION.
    dec_filt : TYPE
        DESCRIPTION.
    com_array : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dictionary = {}
    frame_list = np.empty((0, ))
    stim_dec_list = np.empty((0, ))
    stim_dec = []
    for i_s in range(5):
        index_com_rt = (com_array == 1) *\
            (np.abs(zt_filt) < 0.1) * (np.abs(coh_filt) == 0.25)
        stim_dec_tmp = (stim_filt[i_s, index_com_rt])\
            * dec_filt[index_com_rt]
        # stim_dec_tmp = stim_dec_tmp[stim_dec_tmp != -1]
        # stim_dec_tmp = stim_dec_tmp[stim_dec_tmp != 1]
        # stim_dec_tmp = stim_dec_tmp[stim_dec_tmp != 0]
        stim_dec.append(stim_dec_tmp.T)
        stim_dec_list = np.concatenate((stim_dec_list, stim_dec_tmp))
        frame_list = np.concatenate((frame_list,
                                    np.repeat(i_s+1,
                                              len(stim_dec_tmp)))).astype(int)
    dictionary['frame01'] = frame_list
    dictionary['stimulus_decision_01'] = stim_dec_list
    data_frame = pd.DataFrame(dictionary)
    frame_list = np.empty((0, ))
    stim_dec_list = np.empty((0, ))
    stim_dec = []
    for i_s in range(5):
        index_com_rt = (com_array == 1) *\
            (np.abs(zt_filt) > 2.5) * (np.abs(coh_filt) == 0.25)
        stim_dec_tmp = (stim_filt[i_s, index_com_rt])\
            * dec_filt[index_com_rt]
        # stim_dec_tmp = stim_dec_tmp[stim_dec_tmp != -1]
        # stim_dec_tmp = stim_dec_tmp[stim_dec_tmp != 1]
        # stim_dec_tmp = stim_dec_tmp[stim_dec_tmp != 0]
        stim_dec.append(stim_dec_tmp.T)
        stim_dec_list = np.concatenate((stim_dec_list, stim_dec_tmp))
        frame_list = np.concatenate((frame_list,
                                    np.repeat(i_s+1,
                                              len(stim_dec_tmp)))).astype(int)
    dictionary['frame2'] = frame_list
    dictionary['stimulus_decision_2'] = stim_dec_list
    fig, ax = plt.subplots(nrows=5, figsize=(6, 11))
    ax = ax.flatten()
    for i_ax in range(ax.shape[0]):
        counts2, bins2 = np.histogram(dictionary['stimulus_decision_2']
                                      [dictionary['frame2'] == i_ax+1],
                                      range=(-0.75, 0.75), bins=25)
        counts01, bins01 = np.histogram(dictionary['stimulus_decision_01']
                                        [dictionary['frame01'] == i_ax+1],
                                        range=(-0.75, 0.75), bins=25)
        ax[i_ax].plot(bins2[:-1], counts2/sum(counts2),
                      label='zt > 2.5')
        ax[i_ax].plot(bins01[:-1], counts01/sum(counts01),
                      label='zt < 0.1')
        ax[i_ax].fill_between(bins2[:-1], counts2/sum(counts2), alpha=0.3)
        ax[i_ax].fill_between(bins01[:-1], counts01/sum(counts01), alpha=0.3)
        ax[i_ax].set_ylabel('Frame {}'.format(i_ax+1))
        if i_ax == 0:
            ax[i_ax].legend()
        p_val = mannwhitneyu(dictionary['stimulus_decision_2']
                             [dictionary['frame2'] == i_ax+1],
                             dictionary['stimulus_decision_01']
                             [dictionary['frame01'] == i_ax+1],
                             alternative='two-sided')[1]
        ax[i_ax].text(-0.6, 0.1, s='pvalue: {}'.format(round(p_val, 5)))
    ax[-1].set_xlabel('Stimulus * decision')
    plt.figure()
    sns.violinplot(data=data_frame, x='frame01', y='stimulus_decision_01')
    plt.title('coh==0., zt < 0.1')
    plt.ylabel('stimulus * final decision')
    plt.axhline(y=0, linestyle='--', color='k', lw=1)


def energy_vs_time(stim, zt, coh, sound_len, com, decision, hit, plot=False,
                   data_exist=True, compute_array=False):
    """

    Parameters
    ----------
    stim : TYPE
        DESCRIPTION.
    zt : TYPE
        DESCRIPTION.
    coh : TYPE
        DESCRIPTION.
    sound_len : TYPE
        DESCRIPTION.
    com : TYPE
        DESCRIPTION.
    decision : TYPE
        DESCRIPTION.
    hit : TYPE
        DESCRIPTION.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    data_exist : TYPE, optional
        DESCRIPTION. The default is True.
    compute_array : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    sound_int = np.array(sound_len).astype(int)
    sound_int_filt = sound_int[(sound_int >= 0)*(sound_int < 500)]
    com_array = com[(sound_int >= 0)*(sound_int < 500)].astype(int)
    sound_filt_com = sound_int_filt[com_array == 1]
    stim_filt = stim[:, (sound_int >= 0)*(sound_int < 500)]
    coh_filt = coh[(sound_int >= 0)*(sound_int < 500)]
    dec_filt = decision[(sound_int >= 0)*(sound_int < 500)]
    zt_filt = zt[(sound_int >= 0)*(sound_int < 500)]
    if compute_array:
        if data_exist:
            array_energy = np.load(DATA_FOLDER+'energy_array.npz',
                                   allow_pickle=1)['arr_0']
            array_energy_com = np.load(DATA_FOLDER+'energy_array_com.npz',
                                       allow_pickle=1)['arr_0']
        else:
            array_energy = np.empty((len(sound_int), int(500)))
            array_energy[:] = np.nan
            array_energy_com = np.empty((sum(com_array == 1), int(500)))
            array_energy_com[:] = np.nan
            for i, sound in enumerate(sound_int_filt):
                array_energy[i, :sound] = (stim_filt[i, sound//50])\
                    * dec_filt[i]
            for j, sound_com in enumerate(sound_filt_com):
                array_energy_com[j, :sound_com] = (stim_filt[com_array == 1]
                                                   [j, sound_com//50]) *\
                    dec_filt[com_array == 1][j]
        array_energy_mean = np.nanmean(array_energy, axis=0)
        values = array_energy.shape[0] - np.sum(np.isnan(array_energy), axis=0)
        std_energy = np.sqrt(np.nanstd(array_energy, axis=0)/values)
        array_energy_com_mean = np.nanmean(array_energy_com, axis=0)
        values_com = array_energy_com.shape[0] - np.sum(np.isnan(array_energy_com),
                                                        axis=0)
        std_energy_com = np.sqrt(np.nanstd(array_energy_com, axis=0)/values_com)
        plt.figure()
        plt.plot(np.linspace(0, len(std_energy)-1, num=500), array_energy_mean,
                 label='all trials')
        plt.fill_between(np.linspace(0, len(std_energy)-1, num=500),
                         array_energy_mean+std_energy,
                         array_energy_mean-std_energy,
                         alpha=0.3)
        plt.plot(np.linspace(0, len(std_energy_com)-1, num=500),
                 array_energy_com_mean, label='CoM')
        plt.fill_between(np.linspace(0, len(std_energy)-1, num=500),
                         array_energy_com_mean+std_energy_com,
                         array_energy_com_mean-std_energy_com,
                         alpha=0.3)
        plt.axhline(y=0, linestyle='--', color='k', lw=1)
        plt.legend()
        plt.xlim(0, 150)
        plt.ylim(-0.05, 0.05)
        plt.xlabel('RT (ms)')
        plt.ylabel('Energy (a.u.)')
    if plot:
        plot_distributions(zt_filt, coh_filt, stim_filt, dec_filt, com_array)
        plot_kernels_vs_RT(stim_filt, zt_filt, coh_filt, dec_filt, com_array,
                           sound_int_filt, num_RT=4)
        index_t1 = (com_array.astype(bool)) * (np.abs(zt_filt) > 1.5) *\
                   (np.abs(coh_filt) == 0.25)
        index_t2 = (com_array.astype(bool)) * (np.abs(zt_filt) < 0.3) *\
                   (np.abs(coh_filt) == 0.25)
        energy_com_t1 = (stim[:, index_t1])\
            * decision[index_t1]
        energy_com_t2 = (stim[:, index_t2])\
            * decision[index_t2]
        energy = stim*decision
        energy_com = (stim[:, com.astype(bool)])\
            * decision[com.astype(bool)]
        normal_sound = np.mean(stim*decision, axis=0)
        normal_sound_com = np.mean(stim[:, com.astype(bool)] *
                                   decision[com.astype(bool)], axis=0)
        mean_energy = np.nanmean(energy, axis=1)
        mean_energy_com = np.nanmean(energy_com, axis=1)
        err_energy = np.sqrt(np.nanstd(energy, axis=1)/stim.shape[0])
        err_energy_com = np.sqrt(np.nanstd(energy_com, axis=1)/stim.shape[0])
        time_sound_onset = np.arange(0, 50*stim.shape[0], 50)
        # bins_movement_onset = - sound_len
        bins_rt = np.arange(0, 501, 10)
        df_curve = {'sound_len': sound_len}
        df_curve = pd.DataFrame(df_curve)
        binned_rt = pd.cut(df_curve['sound_len'], bins=bins_rt,
                           labels=False)
        energy_to_plot = energy[binned_rt[binned_rt >= 0].astype(int)//50,
                                np.arange(len(binned_rt[binned_rt >= 0]))]
        df_curve_t1 = {'sound_len': sound_len[index_t1]}
        df_curve_t1 = pd.DataFrame(df_curve_t1)
        binned_rt_t1 = pd.cut(df_curve_t1['sound_len'], bins=bins_rt,
                              labels=False)
        energy_t1_plot = energy_com_t1[binned_rt_t1[binned_rt_t1 >= 0]
                                       .astype(int)//50,
                                       np.arange(len(binned_rt_t1
                                                     [binned_rt_t1 >= 0]))]
        df_curve_t1 = {'energy': energy_t1_plot,
                       'sound_len': -binned_rt_t1[binned_rt_t1 >= 0]*50}
        df_curve_t1 = pd.DataFrame(df_curve_t1)
        df_curve_t2 = {'sound_len': sound_len[index_t2]}
        df_curve_t2 = pd.DataFrame(df_curve_t2)
        binned_rt_t2 = pd.cut(df_curve_t2['sound_len'], bins=bins_rt,
                              labels=False)
        energy_t2_plot = energy_com_t2[binned_rt_t2[binned_rt_t2 >= 0]
                                       .astype(int)//50,
                                       np.arange(len(binned_rt_t2
                                                     [binned_rt_t2 >= 0]))]
        df_curve_t2 = {'energy': energy_t2_plot,
                       'sound_len': -binned_rt_t2[binned_rt_t2 >= 0]*50}
        df_curve_t2 = pd.DataFrame(df_curve_t2)
        # Movement onset:
        bins_rt_neg = np.arange(-500, 1, 10)
        # binned_curve(df_curve_t1, 'energy', 'sound_len', xpos=50, xoffset=-450,
        #              bins=bins_rt_neg,
        #              return_data=False, errorbar_kw={'label': 'Type 1'},
        #              ax=ax12[0])
        # binned_curve(df_curve_t2, 'energy', 'sound_len', xpos=50, xoffset=-450,
        #              bins=bins_rt_neg,
        #              return_data=False, errorbar_kw={'label': 'Type 2'},
        #              ax=ax12[0])
        # ax12[0].set_xlabel('Time from movement onset')
        fig, ax = plt.subplots(1)
        df_curve = {'energy': energy_to_plot,
                    'sound_len': -binned_rt[binned_rt >= 0]*10}
        df_curve = pd.DataFrame(df_curve)
        df_curve = df_curve.dropna()
        binned_curve(df_curve, 'energy', 'sound_len', xpos=10, xoffset=-450,
                     bins=bins_rt_neg,
                     return_data=False, errorbar_kw={'label': 'All trials'},
                     ax=ax)
        df_curve_com = {'sound_len': sound_len[com.astype(bool)]}
        df_curve_com = pd.DataFrame(df_curve_com)
        df_curve_com = df_curve_com.dropna()
        binned_com_rt = pd.cut(df_curve_com['sound_len'], bins=bins_rt,
                               labels=False)
        energy_to_plot_com = energy_com[binned_com_rt
                                        [binned_com_rt >= 0].astype(int)//50,
                                        np.arange(len(
                                            binned_com_rt[binned_com_rt >= 0]))]
        df_curve_com = {'energy': energy_to_plot_com,
                        'sound_len': -binned_com_rt[binned_com_rt >= 0]*10}
        df_curve_com = pd.DataFrame(df_curve_com)
        df_curve_com = df_curve_com.dropna()
        binned_curve(df_curve_com, 'energy', 'sound_len', xpos=10, xoffset=-450,
                     bins=bins_rt_neg,
                     return_data=False, errorbar_kw={'label': 'CoM'}, ax=ax)
        ax.set_xlabel('Time from movement onset (ms)')
        ax.set_ylabel('Energy (a.u.)')
        ax.axhline(y=0, color='k', linewidth=1, linestyle='--')
        plt.figure()
        plt.errorbar(time_sound_onset, mean_energy, err_energy, label='All trials')
        plt.errorbar(time_sound_onset, mean_energy_com, err_energy_com,
                     label='CoM')
        plt.legend()
        plt.xlabel('Time from stimulus onset (ms)')
        plt.ylabel('Energy (a.u.)')
        # which part of the mean stimuli is more important:
        fig, ax = plt.subplots(1)
        df_curve = {'sound': normal_sound, 'sound_len': sound_len}
        df_curve = pd.DataFrame(df_curve)
        df_curve_com = {'sound_com': normal_sound_com, 'sound_len':
                        sound_len[com.astype(bool)]}
        df_curve_com = pd.DataFrame(df_curve_com)
        binned_curve(df_curve, 'sound', 'sound_len', xpos=20,
                     bins=BINS,
                     return_data=False, ax=ax)
        binned_curve(df_curve_com, 'sound_com', 'sound_len', xpos=20,
                     bins=BINS,
                     return_data=False, ax=ax)
        ax.set_xlabel('RT (ms)')
        ax.set_ylabel('Mean absolute value stimuli (a.u.)')
        # Resulaj


def plot_kernels_vs_RT(stim_filt, zt_filt, coh_filt, dec_filt, com_array,
                       sound_int_filt, num_RT=4, different_frames=False):
    """

    Parameters
    ----------
    stim_filt : TYPE
        DESCRIPTION.
    zt_filt : TYPE
        DESCRIPTION.
    coh_filt : TYPE
        DESCRIPTION.
    dec_filt : TYPE
        DESCRIPTION.
    com_array : TYPE
        DESCRIPTION.
    sound_int_filt : TYPE
        DESCRIPTION.
    num_RT : TYPE, optional
        DESCRIPTION. The default is 4.
    different_frames : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    fig12, ax12 = plt.subplots(nrows=num_RT, ncols=1)
    fig12.suptitle('coh = 0.')
    ax12 = ax12.flatten()
    fig2, ax2 = plt.subplots(1)
    ax2.axhline(y=0, linestyle='--', color='k', lw=1)
    ax2.set_xlabel('RT (ms)')
    ax2.set_ylabel('Stim*final_decsion')
    ax2.set_title('coh=0.')
    ax2.set_xlim(0, 120)
    for i in range(num_RT):
        RT = 30*(i)
        index_t1 = (com_array.astype(bool)) * (np.abs(zt_filt) > 2) *\
                   (np.abs(coh_filt) == 0.) * (sound_int_filt > RT)   # *\
        # (sound_int_filt < 30*(i+1))
        index_t2 = (com_array.astype(bool)) * (np.abs(zt_filt) < 0.1) *\
                   (np.abs(coh_filt) == 0.) * (sound_int_filt > RT)  # *\
        # (sound_int_filt < 30*(i+1))
        array_energy_t1 = np.empty((sum(index_t1), int(300)))
        array_energy_t1[:] = np.nan
        array_energy_t2 = np.empty((sum(index_t2), int(300)))
        array_energy_t2[:] = np.nan
        for j, sound_com in enumerate(sound_int_filt[index_t1]):
            array_energy_t1[j, :sound_com] = (stim_filt[:, index_t1]
                                              [sound_com//50, j]) *\
               dec_filt[index_t1][j]
        for j, sound_com in enumerate(sound_int_filt[index_t2]):
            array_energy_t2[j, :sound_com] = (stim_filt[:, index_t2]
                                              [sound_com//50, j]) *\
                dec_filt[index_t2][j]
        mean_energy_t1 = np.nanmean(array_energy_t1, axis=0)
        mean_energy_t2 = np.nanmean(array_energy_t2, axis=0)
        values_t1 = array_energy_t1.shape[0]\
            - np.sum(np.isnan(array_energy_t1), axis=0) + 1
        values_t2 = array_energy_t2.shape[0]\
            - np.sum(np.isnan(array_energy_t2), axis=0) + 1
        err_energy_t1 = np.sqrt(np.nanstd(array_energy_t1, axis=0)/values_t1)
        err_energy_t2 = np.sqrt(np.nanstd(array_energy_t2, axis=0)/values_t2)
        ax12[i].plot(np.linspace(0, len(err_energy_t1)-1, num=300),
                     mean_energy_t1,
                     label='T1 (zt > 2')
        ax12[i].fill_between(np.linspace(0, len(err_energy_t1)-1, num=300),
                             mean_energy_t1+err_energy_t1,
                             mean_energy_t1-err_energy_t1,
                             alpha=0.3)
        ax12[i].plot(np.linspace(0, len(err_energy_t2)-1, num=300),
                     mean_energy_t2,
                     label='T2 (zt < 0.1)')
        ax12[i].fill_between(np.linspace(0, len(err_energy_t2)-1, num=300),
                             mean_energy_t2+err_energy_t2,
                             mean_energy_t2-err_energy_t2,
                             alpha=0.3)
        ax2.plot(np.linspace(0, len(err_energy_t2)-1, num=300),
                 mean_energy_t2,
                 label='T2: {} < RT'.format(RT))
        ax2.fill_between(np.linspace(0, len(err_energy_t2)-1, num=300),
                         mean_energy_t2+err_energy_t2,
                         mean_energy_t2-err_energy_t2,
                         alpha=0.3)
        ax12[i].axhline(y=0, linestyle='--', color='k', lw=1)
        ax12[i].set_xlim(0, 120)
        ax12[i].set_ylim(-0.15, 0.15)
        ax12[i].set_xlabel('RT (ms)')
        ax12[i].set_ylabel('{} < RT'.format(RT))
        if i == 0:
            ax12[i].legend()
            plt.figure()
            plt.plot(np.linspace(0, len(err_energy_t2)-1, num=300),
                     mean_energy_t2,
                     label='T2: {} < RT'.format(RT))
            plt.fill_between(np.linspace(0, len(err_energy_t2)-1, num=300),
                             mean_energy_t2+err_energy_t2,
                             mean_energy_t2-err_energy_t2,
                             alpha=0.3)
            plt.title('all RT, type 2')
            plt.axhline(y=0, linestyle='--', color='k', lw=1)
    ax2.legend()
    precision = 20
    RT_step = 10
    RT_init = 0
    max_RT = 350
    coh_unq = 0.25
    if different_frames:
        fig_ths, ax_ths = plt.subplots(nrows=2, ncols=1)
        fig_dis, ax_dis = plt.subplots(nrows=3, ncols=2)
        for ind_com in range(2):
            stim_period_th_list = [50, 100, 150]
            colors = ['blue', 'orange', 'red']
            for irt_th, stim_period_th in enumerate(stim_period_th_list):
                RT_init = stim_period_th-stim_period_th_list[0]
                list_for_df, list_of_rts, bins_RT, _ =\
                    get_type_2(stim_filt, zt_filt, coh_filt, dec_filt, com_array,
                               sound_int_filt, RT_init, RT_step, precision,
                               coh_unq, max_RT, frame=irt_th, is_com=ind_com)
                dict_values = {'stim_vals': list_for_df, 'rt_vals': list_of_rts}
                df_to_plot = pd.DataFrame(dict_values)
                sns.lineplot(data=df_to_plot, x="rt_vals", y="stim_vals",
                             linewidth=1.5, label='stim: {}-{}'.format(
                                     stim_period_th-stim_period_th_list[0],
                                     stim_period_th), color=colors[irt_th],
                             ax=ax_ths[ind_com], err_style='bars')
                ax_ths[ind_com].axhline(0, linestyle='--', color='k', lw=0.5)
                ax_ths[ind_com].set_title('coh = {}'.format(coh_unq))
                sns.kdeplot(data=df_to_plot, x='stim_vals', hue='rt_vals',
                            shade=False, linewidth=2, palette="dark:salmon_r",
                            common_norm=False, ax=ax_dis[irt_th, ind_com])
    else:
        RT_init = 120
        stim_period_th = 50
        list_for_df, list_of_rts, bins_RT, _ =\
            get_type_2(stim_filt, zt_filt, coh_filt, dec_filt, com_array,
                       sound_int_filt, RT_init, RT_step, precision,
                       coh_unq, max_RT, frame=0)
        dict_values = {'stim_vals': list_for_df, 'rt_vals': list_of_rts}
        dict_values['stim_vals_f2'] = list_for_df
        dict_values['rt_vals_f2'] = list_of_rts
        df_to_plot = pd.DataFrame(dict_values)
        # g = sns.FacetGrid(df_to_plot, row="rt_vals", aspect=5, height=1)
        # g.map(sns.kdeplot, "stim_vals", bw_adjust=.5, clip_on=True,
        #       fill=False, alpha=1, linewidth=2)
        # g.fig.subplots_adjust(hspace=-.25)
        # g.set_titles("")
        # g.set(yticks=[], ylabel="")
        # g.despine(bottom=True, left=True)
        plt.figure()
        sns.kdeplot(data=df_to_plot, x='stim_vals', hue='rt_vals',
                    shade=False, linewidth=2, palette="dark:salmon_r",
                    common_norm=False)
        plt.axvline(0, linestyle='--', color='k', lw=0.5)
        plt.title('Stim period: {}-{} ms. coh = {}'.format(
            stim_period_th-stim_period_th_list[0], stim_period_th, coh_unq))
        plt.figure()
        sns.boxplot(data=df_to_plot, y='stim_vals', x='rt_vals',
                    palette="dark:salmon_r", showmeans=True)
        plt.axhline(0, linestyle='--', color='k', lw=0.5)
        plt.title('Stim period: {}-{} ms. coh = {}'.format(
            stim_period_th-stim_period_th_list[0], stim_period_th, coh_unq))
        plt.figure()
        sns.pointplot(data=df_to_plot, x="rt_vals", y="stim_vals",
                      linewidth=0.5, label='stim: {}-{}'.format(
                          stim_period_th-stim_period_th_list[0], stim_period_th))
        plt.axhline(0, linestyle='--', color='k', lw=0.5)
        plt.title('Stim period: {}-{} ms. coh = {}'.format(
            stim_period_th-stim_period_th_list[0], stim_period_th, coh_unq))


def get_type_2(stim_filt, zt_filt, coh_filt, dec_filt, com_array,
               sound_int_filt, RT_init, RT_step, precision,
               coh_unq, max_RT, matrix=False, n_bins=None, frame=0, is_com=True):
    """

    Parameters
    ----------
    stim_filt : TYPE
        DESCRIPTION.
    zt_filt : TYPE
        DESCRIPTION.
    coh_filt : TYPE
        DESCRIPTION.
    dec_filt : TYPE
        DESCRIPTION.
    com_array : TYPE
        DESCRIPTION.
    sound_int_filt : TYPE
        DESCRIPTION.
    RT_init : TYPE
        DESCRIPTION.
    RT_step : TYPE
        DESCRIPTION.
    precision : TYPE
        DESCRIPTION.
    coh_unq : TYPE
        DESCRIPTION.
    max_RT : TYPE
        DESCRIPTION.
    matrix : TYPE, optional
        DESCRIPTION. The default is False.
    n_bins : TYPE, optional
        DESCRIPTION. The default is None.
    frame : TYPE, optional
        DESCRIPTION. The default is 0.
    is_com : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    list_for_df : TYPE
        DESCRIPTION.
    list_of_rts : TYPE
        DESCRIPTION.
    bins_RT : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    list_for_df = np.empty((0))
    list_of_rts = np.empty((0))
    bins_RT = np.empty((0))
    if matrix:
        matrix_stim = np.zeros((len(np.arange(0, max_RT, RT_step)-1),
                                n_bins))
    for j in range(RT_init, max_RT-precision, RT_step):
        RT_all = j
        index_t2_all = (np.abs(zt_filt) < 0.1) *\
            (np.abs(coh_filt) == coh_unq) *\
            (sound_int_filt > RT_all) * (sound_int_filt
                                         < RT_all + precision)
        if is_com:
            index_t2_all *= (com_array.astype(bool))
        else:
            index_t2_all *= (~com_array.astype(bool))
        if sum(index_t2_all) > 0:
            array_energy_t2_all = stim_filt[frame, index_t2_all] *\
                dec_filt[index_t2_all]
            list_for_df = np.concatenate((list_for_df, array_energy_t2_all))
            list_of_rts = np.concatenate((
                list_of_rts, np.repeat(
                        "{}-{}".format(RT_all, RT_all+precision),
                        len(array_energy_t2_all))))
            bins_RT =\
                np.concatenate((
                    bins_RT,
                    np.repeat(RT_all//RT_step, len(array_energy_t2_all))))
            if matrix:
                hist_stim, bins_stim = np.histogram(array_energy_t2_all,
                                                    bins=n_bins)
                hist_stim = hist_stim/np.nansum(hist_stim)
                matrix_stim[j//RT_step, :] = hist_stim
    if matrix:
        return list_for_df, list_of_rts, bins_RT, matrix_stim
    else:
        return list_for_df, list_of_rts, bins_RT, None


def plot_kernels_start_negative(stim_filt, zt_filt, coh_filt, dec_filt,
                                com_array, sound_int_filt, RT_init, RT_step,
                                precision, coh_unq, max_RT):
    """

    Parameters
    ----------
    stim_filt : TYPE
        DESCRIPTION.
    zt_filt : TYPE
        DESCRIPTION.
    coh_filt : TYPE
        DESCRIPTION.
    dec_filt : TYPE
        DESCRIPTION.
    com_array : TYPE
        DESCRIPTION.
    sound_int_filt : TYPE
        DESCRIPTION.
    RT_init : TYPE
        DESCRIPTION.
    RT_step : TYPE
        DESCRIPTION.
    precision : TYPE
        DESCRIPTION.
    coh_unq : TYPE
        DESCRIPTION.
    max_RT : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    RT_1 = 150
    RT_2 = 400
    stim_decision_f1_all = (stim_filt[0, :])*dec_filt
    stim_decision_f2_all = (stim_filt[1, :])*dec_filt
    index_t2_all = (com_array.astype(bool)) * (np.abs(zt_filt) < 0.1) *\
        (np.abs(coh_filt) == 0) *\
        (sound_int_filt >= RT_1) * (sound_int_filt <= RT_2)
    stim_decision_f1 = stim_decision_f1_all[index_t2_all]
    stim_decision_f2 = stim_decision_f2_all[index_t2_all]
    plt.figure()
    plt.boxplot([stim_decision_f1, stim_decision_f2])
    plt.axhline(0, linestyle='--', color='k', lw=0.5)
    index_t2 = ((stim_decision_f2*stim_decision_f1) < 0)*(stim_decision_f1 < 0)
    print('Total T2 in this range: ' + str(sum(index_t2)))
    print('Total num of trials in this range: ' + str(len(stim_decision_f1)))
    for line_ind in range(sum(index_t2_all)):
        if index_t2[line_ind]:
            plt.plot([1, 2], [stim_decision_f1[line_ind],
                              stim_decision_f2[line_ind]],
                     alpha=0.7, color='r')
        else:
            plt.plot([1, 2], [stim_decision_f1[line_ind],
                              stim_decision_f2[line_ind]],
                     alpha=0.3, color='k')
    plt.show()
    return


def left_right_matrix(zt, coh, com, decision):
    """
    Plot pCoM matrix according to L->R and R->L
    """
    df = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                       'norm_allpriors': zt/max(abs(zt)),
                       'R_response': (decision+1)/2})
    for i in range(2):
        com_heatmap_paper_marginal_pcom_side(df, side=i)


def pcom_model_vs_data(detected_com, com, sound_len, reaction_time):
    fig, ax = plt.subplots(1)
    df = pd.DataFrame({'com_model': detected_com, 'CoM_sugg': com,
                       'sound_len': sound_len, 'reaction_time': reaction_time})
    binned_curve(df, 'CoM_sugg', 'sound_len', bins=BINS, xpos=np.diff(BINS)[0],
                 ax=ax)
    binned_curve(df, 'com_model', 'reaction_time', bins=BINS,
                 xpos=np.diff(BINS)[0], ax=ax)


def kernels(coh, zt, sound_len, decision, stim, com, stim_res=5, neg_pos=False,
            indexing=True, type_1_or_2=None, margin=5):
    """


    Parameters
    ----------
    coh : array
        Putative coherence (0, +-0.25, +-0.5 or +-1), 1 x num_trials.
    zt : array
        priors for each trial (transition bias + lateral (CWJ) 1xnum-trials).
    sound_len : array
        Time of stimulus exposure (Reaction Time), 1 x num_trials.
    decision : array
        decision taken at each trial (1 x num_trials), 1 if right and -1 if left.
    stim : array
        stim sequence for each trial 20xnum-trials.

    Returns
    -------
    None.

    """
    stim_res = int(stim_res)
    max_RT = (max(sound_len) + 1).astype(int)
    if indexing:
        # percentile_zt = np.percentile(np.abs(zt), 20)
        index_coh = (np.abs(coh) <= 0.25)*(sound_len > 100) *\
            (sound_len < 150)
        array_energy = np.empty((len(zt[index_coh]), max_RT))
        array_energy[:] = np.nan
        array_com_energy = np.empty((sum(com[index_coh]), max_RT))
        array_com_energy[:] = np.nan
    else:
        index_coh = (sound_len > 0)*(np.abs(coh) <= 0.5)
        array_energy = np.empty((len(zt[index_coh]), max_RT))
        array_energy[:] = np.nan
        array_com_energy = np.empty((sum(com[index_coh]), max_RT))
        array_com_energy[:] = np.nan
    sound_int = sound_len.astype(int)[index_coh]
    decision_coh = decision[index_coh]
    sound_com_int = sound_len.astype(int)[(com*index_coh).astype(bool)]
    stim_com = stim[:, (com*index_coh).astype(bool)]
    stim_coh_nocom = stim[:, index_coh]
    dec_com_coh = decision[(com*index_coh).astype(bool)]
    array_mov_onset = np.copy(array_energy)
    array_com_mov_onset = np.copy(array_com_energy)
    coh_indexed = coh[index_coh]
    coh_com_indexed = coh[(com*index_coh).astype(bool)]
    for j, sound_com in enumerate(sound_int):
        for s in range(sound_com):
            array_energy[j, s] = (stim_coh_nocom[s//stim_res, j]
                                  - coh_indexed[j]) * decision_coh[j]
    for j, sound_com in enumerate(sound_com_int):
        for s in range(sound_com):
            array_com_energy[j, s] = (stim_com[s//stim_res, j]
                                      - coh_com_indexed[j]) * dec_com_coh[j]
    for j, sound_com in enumerate(sound_int):
        array_mov_onset[j, max_RT-sound_com-1:-1] = array_energy[j, :sound_com]
    for j, sound_com in enumerate(sound_com_int):
        array_com_mov_onset[j, max_RT-sound_com-1:-1] =\
            array_com_energy[j, :sound_com]
    array_energy_mean = np.nanmean(array_mov_onset, axis=0)
    values = array_mov_onset.shape[0] - np.sum(np.isnan(array_mov_onset), axis=0)
    std_energy = np.sqrt(np.nanstd(array_mov_onset, axis=0)/values)
    array_energy_com_mean = np.nanmean(array_com_mov_onset, axis=0)
    values_com = array_com_mov_onset.shape[0] - \
        np.sum(np.isnan(array_com_mov_onset), axis=0)
    std_energy_com = np.sqrt(np.nanstd(array_com_mov_onset, axis=0)/values_com)
    plt.figure()
    plt.plot(np.linspace(-max_RT, 0, max_RT), array_energy_mean,
             label='all trials: N = {}'.format(array_energy.shape[0]))
    plt.fill_between(np.linspace(-max_RT, 0, max_RT),
                     array_energy_mean+std_energy,
                     array_energy_mean-std_energy,
                     alpha=0.3)
    plt.plot(np.linspace(-max_RT, 0, max_RT),
             array_energy_com_mean,
             label='CoM: N = {}'.format(array_com_energy.shape[0]))
    plt.fill_between(np.linspace(-max_RT, 0, max_RT),
                     array_energy_com_mean+std_energy_com,
                     array_energy_com_mean-std_energy_com,
                     alpha=0.3)
    plt.axhline(y=0, linestyle='--', color='k', lw=1)
    if type_1_or_2 is None:
        plt.title('coh = {},{} < RT < {} ms, zt < {}'.
                  format(np.unique(np.abs(coh_indexed)),
                         int(np.min(sound_len[index_coh])),
                         int(np.max(sound_len[index_coh])),
                         np.max(np.abs(zt[index_coh]))))
    else:
        plt.title('CoM Type {}. coh = {}. margin = {}'.
                  format(type_1_or_2, np.unique(np.abs(coh_indexed)),
                         margin))
    plt.xlabel('Movement onset (ms)')
    plt.ylabel('Decision*stimulus')
    plt.legend()
    # plt.figure()
    # color = ['b', 'orange']
    # col = color[1]
    # for trial in array_com_mov_onset:
    #     time_mov_onset = np.linspace(-(400 - np.sum(np.isnan(trial))), 0,
    #                                  num=(400 - np.sum(np.isnan(trial))))
    #     if trial[-2] < 0:
    #         col = color[1]
    #     else:
    #         col = color[0]
    #     plt.plot(time_mov_onset, trial[~np.isnan(trial)], color=col)
    # plt.xlabel('Movement onset (ms)')
    # plt.ylabel('Decision*stimulus')
    # com_arr = array_energy_com_mean[~np.isnan(array_energy_com_mean)]
    # nocom_arr = array_energy_mean[~np.isnan(array_energy_mean)]
    # for i in range(300):
    #     res = wilcoxon(com_arr[-(5+i): -1], nocom_arr[-(5+i):-1],
    #                    alternative='less')
    #     if res[1] < 0.01:
    #         print(res[1])
    #         print(i + 5)
    #         break
    if neg_pos:
        trial_list_pos_neg = []
        prior_list_pos_neg = []
        trial_list_neg_pos = []
        prior_list_neg_pos = []
        trial_list_neg_neg = []
        prior_list_neg_neg = []
        trial_list_pos_pos = []
        prior_list_pos_pos = []
        for i, trial in enumerate(array_com_mov_onset):
            trial = trial[::-1]
            s0 = np.sign(trial[1])
            if s0 > 0:
                if np.sum(np.sign(trial[~np.isnan(trial)]) == -1) == 0:
                    trial_list_pos_pos.append(trial[::-1])
                    prior_list_pos_pos.append(zt[(com*index_coh).astype(bool)][i])
                else:
                    for val in trial[~np.isnan(trial)]:
                        if np.sign(val) < 0:
                            trial_list_pos_neg.append(trial[::-1])
                            prior_list_pos_neg.append(zt[(com*index_coh).
                                                         astype(bool)][i])
                            break
            if s0 < 0:
                if np.sum(np.sign(trial[~np.isnan(trial)]) == 1) == 0:
                    trial_list_neg_neg.append(trial[::-1])
                    prior_list_neg_neg.append(zt[(com*index_coh).astype(bool)][i])
                for val in trial[~np.isnan(trial)]:
                    if np.sign(val) > 0:
                        trial_list_neg_pos.append(trial[::-1])
                        prior_list_neg_pos.append(zt[(com*index_coh).
                                                     astype(bool)][i])
                        break
        array_energy_pos_neg = np.array(trial_list_pos_neg)
        array_energy_mean_pos_neg = np.nanmean(array_energy_pos_neg, axis=0)
        values_pos_neg = array_energy_pos_neg.shape[0] -\
            np.sum(np.isnan(array_energy_pos_neg), axis=0)
        std_energy_pos_neg = np.sqrt(
            np.nanstd(array_energy_pos_neg, axis=0)/values_pos_neg)
        array_energy_neg_pos = np.array(trial_list_neg_pos)
        array_energy_mean_neg_pos = np.nanmean(array_energy_neg_pos, axis=0)
        values_neg_pos = array_energy_neg_pos.shape[0] -\
            np.sum(np.isnan(array_energy_neg_pos), axis=0)
        std_energy_neg_pos = np.sqrt(
            np.nanstd(array_energy_neg_pos, axis=0)/values_neg_pos)
        array_energy_neg_neg = np.array(trial_list_neg_neg)
        array_energy_mean_neg_neg = np.nanmean(array_energy_neg_neg, axis=0)
        values_neg_neg = array_energy_neg_neg.shape[0] -\
            np.sum(np.isnan(array_energy_neg_neg), axis=0)
        std_energy_neg_neg = np.sqrt(
            np.nanstd(array_energy_neg_neg, axis=0)/values_neg_neg)
        array_energy_pos_pos = np.array(trial_list_pos_pos)
        array_energy_mean_pos_pos = np.nanmean(array_energy_pos_pos, axis=0)
        values_pos_pos = array_energy_pos_pos.shape[0] -\
            np.sum(np.isnan(array_energy_pos_pos), axis=0)
        std_energy_pos_pos = np.sqrt(
            np.nanstd(array_energy_pos_pos, axis=0)/values_pos_pos)
        plt.figure()
        plt.plot(np.linspace(-400, 0, 400), array_energy_mean_pos_neg,
                 label='type 2 (neg-pos): N = {}'.format(array_energy_pos_neg.
                                                         shape[0]))
        plt.fill_between(np.linspace(-400, 0, 400),
                         array_energy_mean_pos_neg+std_energy_pos_neg,
                         array_energy_mean_pos_neg-std_energy_pos_neg,
                         alpha=0.3)
        plt.plot(np.linspace(-400, 0, 400), array_energy_mean_neg_pos,
                 label='type 2 (pos-neg): N = {}'.format(array_energy_neg_pos.
                                                         shape[0]))
        plt.fill_between(np.linspace(-400, 0, 400),
                         array_energy_mean_neg_pos+std_energy_neg_pos,
                         array_energy_mean_neg_pos-std_energy_neg_pos,
                         alpha=0.3)
        plt.plot(np.linspace(-400, 0, 400), array_energy_mean_neg_neg,
                 label='type 1 (neg-neg): N = {}'.format(array_energy_neg_neg.
                                                         shape[0]))
        plt.fill_between(np.linspace(-400, 0, 400),
                         array_energy_mean_neg_neg+std_energy_neg_neg,
                         array_energy_mean_neg_neg-std_energy_neg_neg,
                         alpha=0.3)
        plt.plot(np.linspace(-400, 0, 400), array_energy_mean_pos_pos,
                 label='type 1 (pos-pos): N = {}'.format(array_energy_pos_pos.
                                                         shape[0]))
        plt.fill_between(np.linspace(-400, 0, 400),
                         array_energy_mean_pos_pos+std_energy_pos_pos,
                         array_energy_mean_pos_pos-std_energy_pos_pos,
                         alpha=0.3)
        plt.axhline(0, linestyle='--', color='k', alpha=0.3)
        plt.xlim(-400, 0)
        plt.ylim(-0.3, 0.3)
        plt.xlabel('Movement onset (ms)')
        plt.ylabel('Decision*stimulus')
        plt.legend()
        plt.suptitle('Imposing positive to negative N_total = {}'.
                     format(array_com_mov_onset.shape[0]))
        plt.title('coh = {}, RT > {} ms, zt < {}'.
                  format(np.unique(np.abs(coh_indexed)),
                         int(np.min(sound_len[index_coh])),
                         np.max(np.abs(zt[index_coh]))))


def MT_vs_ev(resp_len, coh, com):
    MT = np.array(resp_len)*1e3
    data_frame_MT_coh = pd.DataFrame({'MT': MT, 'coh': np.abs(coh),
                                      'com': com})
    plt.figure()
    sns.pointplot(data=data_frame_MT_coh, x="coh", y="MT", hue="com",
                  linestyles='')
    slope, intercept = np.polyfit(np.abs(coh), MT, 1)
    slope_com, intercept_com = np.polyfit(np.abs(coh)[com.astype(bool)],
                                          MT[com.astype(bool)], 1)
    plt.plot([0, 0.75, 1.5, 3], slope*np.array((0, 0.25, 0.5, 1)) + intercept,
             label='{}*coh + {}'.format(round(slope, 3), round(intercept, 3)))
    plt.plot([0, 0.75, 1.5, 3],
             slope_com*np.array((0, 0.25, 0.5, 1)) + intercept_com,
             label='{}*coh + {}'.format(round(slope_com, 3),
                                        round(intercept_com, 3)))
    plt.legend()


def MT_vs_trial_index_silent(resp_len, trial_index):
    MT_1 = np.array(resp_len)*1e3
    MT = MT_1[(special_trial == 2)*(zt < 0.1)*(MT_1 < 600)]
    plt.figure()
    t_i_filt = np.array(trial_index[(special_trial == 2)*(zt < 0.1)*(MT_1 < 600)],
                        dtype=int)
    plt.scatter(t_i_filt, MT, s=10)
    slope, intercept = np.polyfit(t_i_filt, MT, 1)
    plt.plot(t_i_filt, slope*t_i_filt + intercept, color='red',
             label='MT = {}·t_index + {}'.format(round(slope, 3),
                                                 round(intercept, 3)))
    plt.xlabel('Trial index')
    plt.ylabel('MT (ms)')
    plt.legend()
    plt.title('MT fit for silent trials with zt < 0.1 (MT < 600)')


def coms_tII_tI(first_ind, tr_index, p_t_eff, p_t_aff, stim_res, com,
                detected_com, zt, MT, resp_fin, stim, margin=5,
                pcom_vs_prior_mt=False):
    reaction_time = (first_ind[tr_index]+p_t_eff -
                     int(300/stim_res))*stim_res
    t_2_detected = detected_com*(reaction_time > p_t_aff*stim_res + margin)
    t_1_detected = detected_com*(reaction_time <= p_t_aff*stim_res + margin)
    t_2_all = com*(reaction_time > p_t_aff*stim_res + margin)
    t_1_all = com*(reaction_time <= p_t_aff*stim_res + margin)
    df = pd.DataFrame({'T1_all': t_1_all, 'T2_all': t_2_all, 'T2_detected':
                       t_2_detected, 'T1_detected': t_1_detected,
                       'sound_len': reaction_time, 'prior': zt, 'MT': MT})
    kernels(coh=coh[tr_index], zt=zt[tr_index],
            sound_len=reaction_time, decision=resp_fin[tr_index],
            stim=stim[:, tr_index], com=t_2_detected, stim_res=stim_res,
            indexing=False, type_1_or_2=2, margin=margin)
    kernels(coh=coh[tr_index], zt=zt[tr_index],
            sound_len=reaction_time, decision=resp_fin[tr_index],
            stim=stim[:, tr_index], com=t_1_detected, stim_res=stim_res,
            indexing=False, type_1_or_2=1, margin=margin)
    if pcom_vs_prior_mt:
        margin_list = np.linspace(5, 80, num=16)
        fig, ax = plt.subplots(1)
        t_1_det_list = []
        t_2_det_list = []
        for margin_1 in margin_list:
            t_2_detected = detected_com*(reaction_time >
                                         p_t_aff*stim_res + margin_1)
            t_1_detected = detected_com*(reaction_time <=
                                         p_t_aff*stim_res + margin_1)
            # t_2_all = com*(reaction_time > p_t_aff*stim_res + margin_1)
            # t_1_all = com*(reaction_time <= p_t_aff*stim_res + margin_1)
            t_1_det_list.append(np.mean(t_1_detected))
            t_2_det_list.append(np.mean(t_2_detected))
        plt.axvline(x=50, linestyle='--', color='k', alpha=0.5)
        plt.plot(margin_list, t_1_det_list, label='Mean pcom t1')
        plt.plot(margin_list, t_2_det_list, label='Mean pcom t2')
        plt.legend()
        plt.xlabel('Margin (ms)')
        plt.ylabel('pCoM')
        fig, ax = plt.subplots(1)
        ax.set_title('CoMs vs prior')
        bins_zt = np.linspace(min(zt), max(zt), num=80)
        binned_curve(df, 'T1_all', 'prior', bins=bins_zt, ax=ax, xoffset=min(zt),
                     xpos=np.diff(bins_zt)[0], errorbar_kw={'label': 'T1_all'})
        binned_curve(df, 'T2_all', 'prior', bins=bins_zt, ax=ax, xoffset=min(zt),
                     xpos=np.diff(bins_zt)[0], errorbar_kw={'label': 'T2_all'})
        binned_curve(df, 'T1_detected', 'prior', bins=bins_zt, ax=ax,
                     xoffset=min(zt), xpos=np.diff(bins_zt)[0],
                     errorbar_kw={'label': 'T1_detected'})
        binned_curve(df, 'T2_detected', 'prior', bins=bins_zt, ax=ax,
                     xoffset=min(zt), xpos=np.diff(bins_zt)[0],
                     errorbar_kw={'label': 'T2_detected'})
        ax.set_xlabel('Prior)')
        ax.set_ylabel('pCoM')
        fig, ax = plt.subplots(1)
        ax.set_title('CoMs vs MT')
        bins_zt = np.linspace(min(MT), max(MT), num=30)
        binned_curve(df, 'T1_all', 'MT', bins=bins_zt, ax=ax, xoffset=min(MT),
                     xpos=np.diff(bins_zt)[0], errorbar_kw={'label': 'T1_all'})
        binned_curve(df, 'T2_all', 'MT', bins=bins_zt, ax=ax, xoffset=min(MT),
                     xpos=np.diff(bins_zt)[0], errorbar_kw={'label': 'T2_all'})
        binned_curve(df, 'T1_detected', 'MT', bins=bins_zt, ax=ax,
                     xoffset=min(MT), xpos=np.diff(bins_zt)[0],
                     errorbar_kw={'label': 'T1_detected'})
        binned_curve(df, 'T2_detected', 'MT', bins=bins_zt, ax=ax,
                     xoffset=min(MT), xpos=np.diff(bins_zt)[0],
                     errorbar_kw={'label': 'T2_detected'})
        ax.set_xlabel('MT (ms)')
        ax.set_ylabel('pCoM')


def plot_com_methods(time_trajs, traj_y, com, comlist, subjname=''):
    fig, ax = plt.subplots(ncols=2)
    cont1 = 1
    cont2 = 1
    j = 0
    while cont2 <= 100:
        if com[j] == 1 and comlist[j] == 0:
            if traj_y[j][-1] < 0:
                ax[1].plot(time_trajs[j], -traj_y[j], color='red')
            if traj_y[j][-1] > 0:
                ax[1].plot(time_trajs[j], traj_y[j], color='red')
            cont2 += 1
        j += 1
    l_com = len([c for i, c in enumerate(com) if c and not comlist[i]])
    j = 0
    while cont1 <= 100:
        if com[j] == 0 and comlist[j] == 1:
            if traj_y[j][-1] < 0:
                ax[0].plot(time_trajs[j], -traj_y[j], color='blue')
            if traj_y[j][-1] > 0:
                ax[0].plot(time_trajs[j], traj_y[j], color='blue')
            cont1 += 1
        j += 1
    l_alt = len([c for i, c in enumerate(com) if not c and comlist[i]])
    ax[1].set_title('Detected by old method - not by alt. {}/{}'.format(
        l_com, sum(com)), fontsize=10)
    ax[0].set_title('Detected by alt. method - not by old {}/{}'.format(
        l_alt, sum(comlist)), fontsize=10)
    for iax in range(2):
        ax[iax].set_xlabel('Time (ms)')
        ax[iax].set_ylim(-15, 95)
        ax[iax].axhline(-5, linestyle='--', color='k', alpha=0.4)
        ax[iax].axvline(0, linestyle='--', color='k', alpha=0.4)
        ax[iax].set_xlim(-50, ax[iax].get_xlim()[1])
    ax[0].set_ylabel('y-axis (px)')
    # len([c for i, c in enumerate(com) if c and comlist[i]])
    fig.suptitle('Rat: {}. Total intersection {}'.format(
        subjname, sum(com[comlist])), fontsize=10)


def plot_prev_traj(time_trajs, traj_y):
    fig, ax = plt.subplots(1)
    cont = 0
    plot = True
    last_val = []
    first_val = []
    for i_traj, traj in enumerate(traj_y):
        prev_traj = traj_y[i_traj - 1]
        try:
            traj[0]
            prev_traj[-1]
        except Exception:
            continue
        if np.abs(traj[0]) > 30 and plot and traj[-1] > 0 and traj[0] > 0:
            cont += 1
            ax.plot(time_trajs[i_traj], traj, color='blue')
            max_time_prev_traj = min(time_trajs[i_traj])
            time_prev_traj = time_trajs[i_traj - 1] -\
                max(time_trajs[i_traj-1]) + max_time_prev_traj
            ax.plot(time_prev_traj[-5:-1], prev_traj[-5:-1], color='red')
        last_val.append(prev_traj[-1])
        first_val.append(traj[0])
        if cont >= 50:
            plot = False
    fig, ax = plt.subplots(1)
    ax.plot(last_val, first_val, '.', color='k')
    ax.set_ylabel('First value of trajectory')
    ax.set_xlabel('Last value of previous trajectory')


def cdfs(coh, sound_len, title=''):
    plt.figure()
    colors = ['k', 'darkred', 'darkorange', 'gold']
    ev_vals = np.unique(np.abs(coh))
    for i, ev in enumerate(ev_vals):
        index = ev == np.abs(coh)
        hist_data, bins = np.histogram(sound_len[index], bins=200)
        plt.plot(bins[:-1]+(bins[1]-bins[0])/2,
                 np.cumsum(hist_data)/np.sum(hist_data), label=str(ev),
                 color=colors[i], linewidth=3)
    plt.xlabel('RT (ms)')
    plt.ylabel('CDF')
    plt.legend()
    plt.title(str(title))


def com_detection(trajectories, decision, time_trajs, com_threshold=5):
    com_trajs = []
    time_com = []
    peak_com = []
    comlist = []
    for i_t, traj in enumerate(trajectories):
        if len(traj) > 1 and max(np.abs(traj)) > 100:
            comlist.append(False)
        else:
            if len(traj) > 1 and len(time_trajs[i_t]) > 1 and\
              sum(np.isnan(traj)) < 1 and sum(time_trajs[i_t] > 1) >= 1:
                traj -= np.nanmean(traj[
                    (time_trajs[i_t] >= -100)*(time_trajs[i_t] <= 0)])
                signed_traj = traj*decision[i_t]
                if abs(traj[time_trajs[i_t] >= 0][0]) < 20:
                    peak = min(signed_traj[time_trajs[i_t] >= 0])
                    if peak < 0:
                        peak_com.append(peak)
                    if peak < -com_threshold:
                        com_trajs.append(traj)
                        time_com.append(
                            time_trajs[i_t]
                            [np.where(signed_traj == peak)[0]][0])
                        comlist.append(True)
                    else:
                        comlist.append(False)
                else:
                    comlist.append(False)
            else:
                comlist.append(False)
    return com_trajs, time_com, peak_com, comlist


def mean_com_traj_peak(trajectories, com, sound_len, decision, motor_time, zt,
                       time_trajs, val_at_updt=None, data=True, peak_cond=False):
    if data:
        peak_com = []
        time_com = []
        sound_com = []
        mt_com = []
        time_trajs = time_trajs[com.astype(bool)]
        traj_com = trajectories[com.astype(bool)]
        decision_com = decision[com.astype(bool)]
        for i_t, traj in enumerate(traj_com):
            signed_traj = traj[6:-1]*decision_com[i_t]
            if abs(traj[time_trajs[i_t] >= 0][0]) > 15:
                continue
            else:
                peak = abs(min(signed_traj[time_trajs[i_t] >= 0]))
                if peak < 50:
                    peak_com.append(peak)
                    time_com.append(time_trajs[i_t][time_trajs[i_t] >= 0]
                                    [int(np.argmin(
                                        signed_traj[time_trajs[i_t] >= 0]))])
                # else:
                #     peak = abs(min(signed_traj[time_trajs[i_t] < 300]))
                #     peak_com.append(peak)
                #     time_com.append(time_trajs[i_t][time_trajs[i_t] >= 0]
                #                     [int(np.where(np.abs(signed_traj)
                #                                   == peak)[0][0])])
                    sound_com.append(sound_len[i_t])
                    mt_com.append(resp_len[i_t])
    if not data:
        peak_com = np.array(val_at_updt)[com.astype(bool)]
        motor_time = np.array(motor_time)
        sound_com = sound_len[com.astype(bool)]
        mt_com = motor_time[com.astype(bool)]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    if peak_cond:
        var = peak_com
        xlab = 'CoM peak (pixels)'
    else:
        var = time_com
        xlab = 'Time to peak (ms)'
    ax = ax.flatten()
    ax[0].hist(var, bins=80)
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel('Counts')
    df_plot = pd.DataFrame({'RT': sound_com,
                            'ComPeak': var})
    # bins_rt = np.linspace(0, 300, 21)
    binned_curve(df_plot, 'ComPeak', 'RT', bins=BINS,
                 ax=ax[1], xpos=np.diff(BINS)[0])
    ax[1].set_xlabel('RT (ms)')
    ax[1].set_ylabel(xlab)
    # com_heatmap_jordi(zt[com.astype(bool)], coh[com.astype(bool)],
    #                   (np.array(peak_com) < 50)*1.0, flip=True,
    #                   annotate=False, ax=ax[2])
    # ax[2].set_title('Peak < 50 px.')
    if data:
        ax[2].plot(time_com, peak_com, '.', color='k')
    ax[2].set_xlabel('Time to peak (ms)')
    ax[2].set_ylabel('CoM peak (pixels)')
    df_plot = pd.DataFrame({'MT': np.array(mt_com)*1e3,
                            'ComPeak': var})
    bins_mt = np.linspace(150, 600, 31)
    binned_curve(df_plot, 'ComPeak', 'MT', bins=bins_mt,
                 ax=ax[3], xpos=np.diff(bins_mt)[0])
    ax[3].set_xlabel('MT (ms)')
    ax[3].set_ylabel(xlab)


def pdf_cohs(sound_len, ax, coh, bins=np.linspace(1, 301, 61), yaxis=True):
    ev_vals = np.unique(np.abs(coh))
    colormap = pl.cm.gist_gray_r(np.linspace(0.2, 1, len(ev_vals)))
    for i_coh, ev in enumerate(ev_vals):
        index = np.abs(coh) == ev
        counts_coh, bins_coh = np.histogram(sound_len[index], bins=bins)
        norm_counts = counts_coh/sum(counts_coh)
        xvals = bins_coh[:-1]+(bins_coh[1]-bins_coh[0])/2
        ax.plot(xvals, norm_counts, color=colormap[i_coh])
    ax.set_xlabel('Reaction time (ms)')
    ax.set_xlim(0, 150)
    if yaxis:
        ax.set_ylabel('Density')


def pdf_cohs_subjects(subjects, sound_len, coh):
    fig, ax = plt.subplots(nrows=3, ncols=5)
    ax = ax.flatten()
    for isub, subj in enumerate(np.unique(subjects)):
        pdf_cohs(sound_len[subjects == subj], ax[isub], coh[subjects == subj],
                 bins=np.linspace(1, 301, 61), yaxis=True)
        ax[isub].set_title(str(subj))


def get_trajs_time(resp_len, traj_stamps, fix_onset, com, sound_len,
                   com_cond=False):
    time = []
    if com_cond:
        traj_st_com = traj_stamps[com.astype(bool)]
        fix_onset_com = fix_onset[com.astype(bool)]
        sound_len_com = sound_len[com.astype(bool)]
    else:
        traj_st_com = traj_stamps
        fix_onset_com = fix_onset
        sound_len_com = sound_len
    for j in range(len(traj_st_com)):
        t = traj_st_com[j] - fix_onset_com[j]
        t = (t.astype(int) / 1000_000 - 250 - sound_len_com[j])
        time.append(t)
    return np.array(time)


# --- MAIN
if __name__ == '__main__':
    # TODO: organize script
    plt.close('all')
    # tests_trajectory_update(remaining_time=100, w_updt=10)
    num_tr = int(1e5)
    load_data = True
    new_sample = True
    single_run = True
    shuffle = False
    simulate = True
    parallel = False
    plot_t12 = False
    data_augment_factor = 10
    splitting = True
    silent = True
    rat = 'LE42'
    if simulate:
        # GET DATA
        if load_data:  # experimental data
            if new_sample:  # get a new sample
                stim, zt, coh, gt, com, decision, sound_len, resp_len, hit,\
                    trial_index, special_trial, traj_y, fix_onset, traj_stamps,\
                    subjects =\
                    get_data_and_matrix(dfpath=DATA_FOLDER + rat,
                                        num_tr_per_rat=int(1e4),
                                        after_correct=True, splitting=splitting,
                                        silent=silent, all_trials=True)
                data = {'stim': stim, 'zt': zt, 'coh': coh, 'gt': gt, 'com': com,
                        'sound_len': sound_len, 'decision': decision,
                        'resp_len': resp_len, 'hit': hit,
                        'trial_index': trial_index, 'special_trial': special_trial,
                        'trajectory_y': traj_y, 'trajectory_stamps': traj_stamps,
                        'fix_onset_dt': fix_onset}
                np.savez(DATA_FOLDER+'/sample_'+str(time.time())[-5:]+'.npz',
                         **data)
                pdf_cohs_subjects(subjects, sound_len, coh)
            else:  # use existing sample
                if silent:
                    subfolder = '/silent'
                if splitting:
                    subfolder = '/splitting'
                else:
                    subfolder = ''
                files = glob.glob(DATA_FOLDER+subfolder+'/sample_*')
                data = np.load(files[np.random.choice(a=len(files))],
                               allow_pickle=True)
                stim = data['stim']
                zt = data['zt']
                coh = data['coh']
                com = data['com']
                gt = data['gt']
                sound_len = data['sound_len']
                resp_len = data['resp_len']
                decision = data['decision']
                hit = data['hit']
                trial_index = data['trial_index']
                if silent:
                    special_trial = data['special_trial']
                if splitting:
                    traj_y = data['trajectory_y']
                    fix_onset = data['fix_onset_dt']
                    traj_stamps = data['trajectory_stamps']
            if plot_t12:
                energy_vs_time(stim, zt, coh, sound_len, com, decision, hit)
            stim = data_augmentation(stim=stim, daf=data_augment_factor)
            stim_res = 50/data_augment_factor
        else:  # simulated data
            num_timesteps = 1000
            zt =\
                np.random.rand(num_tr)*2*(-1.0)**np.random.randint(-1, 1,
                                                                   size=num_tr)
            stim = \
                np.random.rand(num_tr)*(-1.0)**np.random.randint(-1, 1,
                                                                 size=num_tr) +\
                np.random.randn(num_timesteps, num_tr)*1e-1
            stim_res = 1
        # RUN MODEL
        decision = decision[:int(num_tr)]
        stim = stim[:, :int(num_tr)]
        # stim[:] = 1
        zt = zt[:int(num_tr)]
        sound_len = sound_len[:int(num_tr)]
        resp_len = resp_len[:int(num_tr)]
        coh = coh[:int(num_tr)]
        com = com[:int(num_tr)]
        gt = gt[:int(num_tr)]
        trial_index = trial_index[:int(num_tr)]
        hit = hit[:int(num_tr)]
        if splitting:
            traj_y = traj_y[:int(num_tr)]
            fix_onset = fix_onset[:int(num_tr)]
            traj_stamps = traj_stamps[:int(num_tr)]
        if single_run:  # single run with specific parameters
            p_t_aff = 9
            p_t_eff = 12
            p_t_a = 14  # 90 ms (18) PSIAM fit includes p_t_eff
            p_w_zt = 0.2
            p_w_stim = 0.11
            p_e_noise = 0.01
            p_com_bound = 0.001
            p_w_a_intercept = 0.052
            p_w_a_slope = -2.2e-05  # fixed
            p_a_noise = 0.04  # fixed
            p_1st_readout = 35
            p_2nd_readout = 35
            compute_trajectories = True
            plot = True
            all_trajs = True
            configurations = [(p_w_zt, p_w_stim, p_e_noise, p_com_bound, p_t_aff,
                              p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope,
                              p_a_noise,
                              p_1st_readout,
                              p_2nd_readout)]
            jitters = len(configurations[0])*[0]
            print('Number of trials: ' + str(stim.shape[1]))
            time_trajs = get_trajs_time(resp_len, traj_stamps, fix_onset, com,
                                        sound_len=sound_len)
            com_trajs, _, _, comlist = com_detection(
                traj_y, decision, time_trajs, com_threshold=8)
            comlist = np.array(comlist)
            # plot_com_methods(time_trajs, traj_y, com, comlist, subjname='LE44')
            com = comlist
            if plot:
                # left_right_matrix(zt, coh, com, decision)
                data_to_plot = {'sound_len': sound_len,
                                'CoM': com,
                                'first_resp': decision*[~com*(-1)],
                                'final_resp': decision,
                                'hithistory': hit,
                                'avtrapz': coh,
                                'detected_com': com,
                                'MT': resp_len*1e3,
                                'zt': zt, 'decision': decision,
                                'trial_idxs': trial_index,
                                'subjid': np.repeat(rat, len(com))}
                plot_misc(data_to_plot, stim_res=stim_res, data=True)
                # mean_com_traj_peak(trajectories=traj_y, com=com, zt=zt,
                #                    sound_len=sound_len, decision=decision,
                #                    motor_time=resp_len,
                #                    time_trajs=np.array(time_trajs))
            if splitting:
                traj_y = traj_y[:int(num_tr)]
                fix_onset = fix_onset[:int(num_tr)]
                traj_stamps = traj_stamps[:int(num_tr)]
            else:
                traj_y = None
                fix_onset = None
                traj_stamps = None
        else:  # set grid search of parameters
            configurations, jitters = set_parameters(num_vals=4)
            compute_trajectories = True
            plot = False
            all_trajs = True
        existing_data = None  # SV_FOLDER+'/results/all_results_1.npz'
        if parallel:  # run simulatings using joblib toolbox
            configurations = list(configurations)
            num_cores = int(mp.cpu_count())
            step = int(np.ceil(len(configurations)/num_cores))
            Parallel(n_jobs=num_cores)\
                (delayed(run_model)(stim=stim, zt=zt, coh=coh, gt=gt,
                                    trial_index=trial_index, com=None,
                                    sound_len=None, traj_y=None,
                                    configurations=configurations
                                    [int(i_par*step):int((i_par+1)*step)],
                                    jitters=jitters,
                                    compute_trajectories=compute_trajectories,
                                    plot=plot, stim_res=stim_res,
                                    existing_data=existing_data,
                                    shuffle=shuffle, all_trajs=True,
                                    traj_stamps=None, fix_onset=None)
                 for i_par in range(num_cores))
        else:  # sequential runs
            run_model(stim=stim, zt=zt, coh=coh, gt=gt, com=com,
                      trial_index=trial_index,
                      sound_len=sound_len, traj_y=traj_y,
                      traj_stamps=traj_stamps, fix_onset=fix_onset,
                      configurations=configurations, jitters=jitters,
                      compute_trajectories=compute_trajectories,
                      plot=plot, stim_res=stim_res,
                      existing_data=existing_data,
                      shuffle=shuffle, all_trajs=all_trajs, kernels_model=False)
    # data_path = '/home/molano/Dropbox/project_Barna/ChangesOfMind/results/'
    # res_path = data_path
    # data_path = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/results/'
    # res_path = 'C:/Users/agarcia/Desktop/CRM/brute_force/'
    # data_curve, optimal_params = \
    # fitting(res_path=res_path, data_path=data_path, metrics='mse',
    #         objective='curve')


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Aug 10 15:12:00 2022

# @author: molano
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# plt.close('all')
# folder = '/home/molano/Dropbox/project_Barna/ChangesOfMind/results/'
# results = np.load(folder+'/all_results.npz', allow_pickle=1)
# exps = pd.read_csv(folder+'/pcom_vs_rt.csv')
# pcoms = exps.pcom.values
# pos_rts = results['median_pcom_rt']
# pos_rts_bins = results['xpos_rt_pcom']
# # pos_rts_bins += (pos_rts_bins[1]-pos_rts_bins[0])/2
# corrs = []
# for conf in range(len(pos_rts)):
#     corrs.append(np.corrcoef(pcoms, )[0, 1])


# plt.figure()
# plt.hist(corrs, 100)
# plt.figure()
# best_indxs = np.where(np.array(corrs) > 0.8)[0]
# for conf in best_indxs:
#     norm_hist = pos_rts[conf, :]
#     plt.plot(pos_rts_bins, norm_hist, lw=0.5, color='k')
# plt.plot(exps.rt, pcoms, color='r')
