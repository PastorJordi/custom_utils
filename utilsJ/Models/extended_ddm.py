# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import glob
import time
import sys
from skimage.metrics import structural_similarity as ssim
import multiprocessing as mp
from joblib import Parallel, delayed
# sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
import utilsJ
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve
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
BINS = np.linspace(1, 300, 16)


def tests_trajectory_update(remaining_time=100, w_updt=10):
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


def draw_lines(ax, frst, sec, p_t_aff, p_com_bound):
    ax[0].axhline(y=1, color='purple', linewidth=2)
    ax[0].axhline(y=-1, color='green', linewidth=2)
    ax[0].axhline(y=0, linestyle='--', color='k', linewidth=0.7)
    ax[0].axhline(y=p_com_bound, color='purple', linewidth=1, linestyle='--')
    ax[0].axhline(y=-p_com_bound, color='green', linewidth=1, linestyle='--')
    ax[1].axhline(y=1, color='k', linewidth=1, linestyle='--')
    for a in ax:
        a.axvline(x=frst, color='c', linewidth=1, linestyle='--')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.axvline(x=sec, color='c', linewidth=1, linestyle='--')


def plotting(com, E, A, second_ind, first_ind, resp_first, resp_fin, pro_vs_re,
             p_t_aff, init_trajs, total_traj, p_t_eff, motor_updt_time, tr_index,
             p_com_bound, stim_res=50, trial=0):
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
            # updt_motor = first_ind[trial]+motor_updt_time[trial]
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


def plot_misc(data_to_plot, stim_res, all_trajs=True):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax = ax.flatten()
    trial_idxs = np.arange(len(data_to_plot['sound_len'])) % 600
    data_to_plot['trial_idxs'] = trial_idxs
    data_to_plot['re_vs_pro'] = (data_to_plot['pro_vs_re'] - 1)*(-1)
    data_to_df = {key: data_to_plot[key] for key in ['CoM', 'sound_len',
                                                     'detected_com',
                                                     'hithistory',
                                                     'avtrapz',
                                                     'final_resp',
                                                     'MT', 'trial_idxs',
                                                     're_vs_pro']}
    df_plot = pd.DataFrame(data_to_df)
    xpos = int(np.diff(BINS)[0]+1)
    binned_curve(df_plot, 'CoM', 'sound_len', bins=BINS,
                 xpos=xpos, ax=ax[0], errorbar_kw={'label': 'CoM'})
    binned_curve(df_plot, 'detected_com', 'sound_len',
                 bins=BINS, ax=ax[0], xpos=xpos,
                 errorbar_kw={'label': 'detected com'})
    data_curve = pd.read_csv(SV_FOLDER + '/results/pcom_vs_rt.csv')
    ax[0].plot(data_curve['rt'], data_curve['pcom'], label='data', linestyle='',
               marker='o')
    ax[0].legend()
    ax[0].set_xlabel('RT (ms)')
    ax[0].set_ylabel('PCoM')
    tachometric(df_plot, ax=ax[1])
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
    bins = np.linspace(-300, 400, 40)
    hist_pro, _ = np.histogram(data_to_plot['sound_len'][data_to_plot['pro_vs_re']
                                                         == 0],
                               bins)
    hist_re, _ = np.histogram(data_to_plot['sound_len'][data_to_plot['pro_vs_re']
                                                        == 1],
                              bins)
    hist_total, _ = np.histogram(data_to_plot['sound_len'], bins)
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2, hist_pro/hist_pro.sum(), label='Pro',
            linestyle='--')
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2, hist_re/hist_re.sum(), label='Re',
            linestyle='--')
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2, hist_total/hist_total.sum(),
            label='All')
    ax.legend()
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
    ax1[1].set_xlabel('Trial index')
    ax1[1].set_ylabel('Detected CoM')
    bins_trial = np.linspace(0, 600, 11, dtype=int)
    binned_curve(df_plot, 'detected_com', 'trial_idxs',
                 bins=bins_trial, ax=ax1[1], xpos=60,
                 errorbar_kw={'label': 'detected_com'})
    binned_curve(df_plot, 're_vs_pro', 'sound_len',
                 bins=BINS, ax=ax1[2], xpos=xpos,
                 errorbar_kw={'label': 'proac_prop'})
    ax1[2].set_xlabel('RT')
    ax1[2].set_ylabel('Proac. proportion')
    bins_MT = np.linspace(80, 230, num=16, dtype=int)
    binned_curve(df_plot, 'CoM', 'MT',
                 bins=bins_MT, ax=ax1[3], xpos=10,
                 xoffset=120, errorbar_kw={'label': 'CoM'})
    binned_curve(df_plot, 'detected_com', 'MT',
                 bins=bins_MT, ax=ax1[3], xpos=10,
                 xoffset=120, errorbar_kw={'label': 'detected CoM'})
    ax1[3].legend()
    ax1[3].set_xlabel('MT')
    ax1[3].set_ylabel('pCoM')
    hist_MT, _ = np.histogram(data_to_plot['MT'], bins=bins_MT)
    ax1[4].plot(bins_MT[:-1]+(bins_MT[1]-bins_MT[0])/2, hist_MT,
                label='MT dist')
    ax1[4].set_xlabel('MT (ms)')
    zt = data_to_plot['zt'][data_to_plot['pro_vs_re'] == 0]
    coh = data_to_plot['avtrapz'][data_to_plot['pro_vs_re'] == 0]
    com = data_to_plot['CoM'][data_to_plot['pro_vs_re'] == 0]
    mat_proac, _ = com_heatmap_jordi(zt, coh, com, return_mat=True)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    ax = ax.flatten()
    sns.heatmap(matrix, ax=ax[0])
    ax[0].set_title('pCoM simulation')
    detected_mat[np.isnan(detected_mat)] = 0
    sns.heatmap(detected_mat, ax=ax[1])
    ax[1].set_title('Detected proportion')
    sns.heatmap(mat_proac, ax=ax[2])
    ax[2].set_title('pCoM in proac. trials')


def com_heatmap_jordi(x, y, com, flip=False, annotate=True,
                      predefbins=None, return_mat=False,
                      folding=False, annotate_div=1, **kwargs):
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
            g = sns.heatmap(np.flipud(matrix), annot=np.flipud(annotmat),
                            **kwargs).set(xlabel="prior",
                                          ylabel="average stim",
                                          xticklabels=priorlabels,
                                          yticklabels=np.flip(stimlabels))

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


def v_(t):
    return t.reshape(-1, 1) ** np.arange(6)


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


def compute_traj(jerk_lock_ms, mu, resp_len):
    t_arr = np.arange(jerk_lock_ms, resp_len)
    M = get_Mt0te(jerk_lock_ms, resp_len)
    M_1 = np.linalg.inv(M)
    vt = v_(t_arr)
    N = vt @ M_1
    traj = (N @ mu).ravel()
    traj = np.concatenate([[0]*jerk_lock_ms, traj])  # trajectory
    return traj


def get_data_and_matrix(dfpath='C:/Users/Alexandre/Desktop/CRM/Alex/paper/',
                        num_tr_per_rat=int(1e4), after_correct=True):
    # import data for 1 rat
    print('Loading data')
    files = glob.glob(dfpath+'*.pkl')
    start = time.time()
    prior = np.empty((0, ))
    stim = np.empty((0, 20))
    com = np.empty((0, ))
    coh = np.empty((0, ))
    gt = np.empty((0, ))
    sound_len = np.empty((0, ))
    for f in files:
        start_1 = time.time()
        df = pd.read_pickle(f)
        end = time.time()
        # max_num_tr = max(num_tr_per_rat, len(df))
        if after_correct:
            indx_prev_error = np.where(df['aftererror'].values == 0)[0]
            selected_indx = np.random.choice(np.arange(len(indx_prev_error)),
                                             size=(num_tr_per_rat), replace=False)
            indx = indx_prev_error[selected_indx]
        else:
            indx = np.random.choice(np.arange(len(df)), size=(num_tr_per_rat),
                                    replace=False)
        prior_tmp = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        stim_tmp = np.array([stim for stim in df.res_sound])
        coh_mat = np.array(df.coh2)
        com_tmp = df.CoM_sugg.values
        sound_len_tmp = np.array(df.sound_len)
        gt_tmp = np.array(df.rewside) * 2 - 1
        prior = np.concatenate((prior, prior_tmp[indx]))
        stim = np.concatenate((stim, stim_tmp[indx, :]))
        coh = np.concatenate((coh, coh_mat[indx]))
        com = np.concatenate((com, com_tmp[indx]))
        gt = np.concatenate((gt, gt_tmp[indx]))
        sound_len = np.concatenate((sound_len, sound_len_tmp[indx]))
        end = time.time()
        print(f)
        print(end - start_1)
        print(len(df))
    print(end - start)
    print('Ended loading data, start computing matrix')
    df_curve = {'CoM': com, 'sound_len': sound_len}
    df_curve = pd.DataFrame(df_curve)
    xpos = int(np.diff(BINS)[0])
    xpos_plot, median_pcom, _ =\
        binned_curve(df_curve, 'CoM', 'sound_len', xpos=xpos,
                     bins=BINS,
                     return_data=True)
    df_pcom_rt = pd.DataFrame({'rt': xpos_plot, 'pcom': median_pcom})
    df_pcom_rt.to_csv(SV_FOLDER + '/results/pcom_vs_rt.csv')
    matrix, _ = com_heatmap_jordi(prior, coh, com, return_mat=True)
    np.save(SV_FOLDER + '/results/CoM_vs_prior_and_stim.npy', matrix)
    stim = stim.T
    return stim, prior, coh, gt, com  # , matrix


def trial_ev_vectorized(zt, stim, coh, MT_slope, MT_intercep, p_w_zt, p_w_stim,
                        p_e_noise, p_com_bound, p_t_eff, p_t_aff,
                        p_t_a, p_w_a, p_a_noise, p_w_updt, num_tr, stim_res,
                        compute_trajectories=False, num_trials_per_session=600,
                        proactive_integration=True, all_trajs=True,
                        num_computed_traj=int(2e4)):
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
    p_w_a : float
        fitting parameter: drift of action noise.
    p_a_noise : float
        fitting parameter: standard deviation of action noise (gaussian).
    p_w_updt : float
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
    print('Starting simulation, PSIAM')
    bound = 1
    bound_a = 1
    fixation = int(300 / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    Va = p_w_a
    # trial_dur = 1  # trial duration (s)
    N = Ve.shape[0]  # int(trial_dur/dt)  # number of timesteps
    dW = np.random.randn(N, num_tr)*p_e_noise+Ve
    # Va = (np.linspace(0, Ve.shape[1], num=Ve.shape[1], dtype=int)
    #       % num_trials_per_session)*(-2.5)*1e-6 + 5.2*1e-3
    if proactive_integration:
        dA = np.random.randn(N, num_tr)*p_a_noise+Va
        dA[:p_t_a, :] = 0
        A = np.cumsum(dA, axis=0)
    else:
        rt_a = [np.random.wald(mean=bound_a/va, scale=bound_a**2) + p_t_a
                for va in Va]
        A = rt_a
    dW[0, :] = prior  # +np.random.randn(p_t_aff, num_tr)*p_e_noise
    E = np.cumsum(dW, axis=0)
    com = False
    first_ind = []
    second_ind = []
    pro_vs_re = []
    first_ev = []
    second_ev = []
    resp_first = np.ones(E.shape[1])
    resp_fin = np.ones(E.shape[1])
    for i_t in range(E.shape[1]):
        indx_hit_bound = np.abs(E[:, i_t]) >= bound
        hit_bound = max_integration_time  # -p_t_aff-p_t_eff
        if (indx_hit_bound).any():
            hit_bound = np.where(indx_hit_bound)[0][0]
        if proactive_integration:
            indx_hit_action = np.abs(A[:, i_t]) >= bound_a
            hit_action = max_integration_time  # -p_t_eff-p_t_a
            if (indx_hit_action).any():
                hit_action = np.where(indx_hit_action)[0][0]
        else:
            hit_action = rt_a[i_t]
        hit_dec = min(hit_bound, hit_action)  # reactive or proactive
        # XXX: reactive trials are defined as EA reaching the bound,
        # which includes influence of zt
        pro_vs_re.append(np.argmin([hit_action, hit_bound]))
        first_ind.append(hit_dec)
        first_ev.append(E[hit_dec, i_t])
        # first response
        resp_first[i_t] *= (-1)**(E[hit_dec, i_t] < 0)
        # p_com_bound = 1 - np.abs(hit_dec + p_t_eff - fixation + 1)**(-4/5)
        # 1/(0.03127615*np.exp(-(hit_dec + p_t_eff - fixation + 1)*
        #                                 0.65553348) + 0.01380665)
        com_bound_temp = (-resp_first[i_t])*p_com_bound
        # second response
        indx_fin_ch = hit_dec+p_t_aff+p_t_eff
        indx_fin_ch = min(indx_fin_ch, max_integration_time)
        post_dec_integration = E[hit_dec:indx_fin_ch, i_t]-com_bound_temp
        indx_com =\
            np.where(np.sign(E[hit_dec, i_t]) != np.sign(post_dec_integration))[0]
        indx_update_ch = indx_fin_ch if len(indx_com) == 0\
            else indx_com[0] + hit_dec
        resp_fin[i_t] = resp_first[i_t] if len(indx_com) == 0 else -resp_first[i_t]
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_t])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind).astype(int)
    pro_vs_re = np.array(pro_vs_re)
    matrix, _ = com_heatmap_jordi(zt, coh, com,
                                  return_mat=True)
    # The pcom_RT curve will be computed considering detection
    # df_curve = {'CoM': com, 'sound_len': (first_ind-fixation+p_t_eff)*stim_res}
    # df_curve = pd.DataFrame(df_curve)
    # xpos = int(np.diff(BINS)[0])
    # xpos_plot, median_pcom, _ =\
    #     binned_curve(df_curve, 'CoM', 'sound_len', xpos=xpos,
    #                  bins=BINS,
    #                  return_data=True)
    # df_pcom_rt = {'rt': xpos_plot, 'pcom': median_pcom}
    rt_vals, rt_bins = np.histogram((first_ind-fixation+p_t_eff)*stim_res,
                                    bins=20, range=(-100, 300))
    # TODO: put in a different function
    if compute_trajectories:
        # Trajectories
        print('Starting with trajectories')
        RLresp = resp_fin
        prechoice = resp_first
        jerk_lock_ms = 0
        initial_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
        indx_trajs = np.arange(len(first_ind)) if all_trajs\
            else np.random.choice(len(first_ind), num_computed_traj)
        # initial positions, speed and acc; final position, speed and acc
        init_trajs = []
        final_trajs = []
        total_traj = []
        motor_updt_time = []
        x_val_at_updt = []
        tr_indx_for_coms = []
        indx = []
        for i_t in indx_trajs:
            # pre-planned Motor Time, the modulo prevents trial-index from
            # growing indefinitely
            MT = MT_slope*(i_t % num_trials_per_session) + MT_intercep
            first_resp_len = float(MT-p_w_updt*np.abs(first_ev[i_t]))
            # first_resp_len: evidence affectation on MT. The higher the ev,
            # the lower the MT depending on the parameter p_w_updt
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                  resp_len=first_resp_len)
            init_trajs.append(prior0)
            # TRAJ. UPDATE
            velocities = np.gradient(prior0)
            accelerations = np.gradient(velocities)  # acceleration
            t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
            t_updt = int(np.min((t_updt, len(velocities)-1)))
            motor_updt_time.append(t_updt)
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
            # responses) bc of very strong confirmation evidence. Note that
            # theoretically this problema does not exists bc CoM_bound will
            # be less or equal to the bounds.
            updt_ev = np.sign(second_ev[i_t])*min(1, np.abs(second_ev[i_t]))
            # second_response_len: time left affected by the evidence on the
            second_response_len =\
                float(remaining_m_time-sign_*p_w_updt*updt_ev)
            # SECOND readout
            traj_fin = compute_traj(jerk_lock_ms, mu=mu_update,
                                    resp_len=second_response_len)
            final_trajs.append(traj_fin)
            # joined trajectories
            traj_before_uptd = prior0[0:t_updt]
            traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
            total_traj.append(traj_updt)
            if com[i_t]:
                opp_side_values = traj_updt.copy()
                opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                max_val_towards_opposite = np.max(np.abs(opp_side_values))
                x_val_at_updt.append(max_val_towards_opposite)
                indx.append(i_t)
            else:
                x_val_at_updt.append(0)
        detect_CoMs_th = 5
        detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
        df_curve = {'detected_CoM': detected_com,
                    'sound_len': (first_ind[indx_trajs]-fixation+p_t_eff)*stim_res}
        df_curve = pd.DataFrame(df_curve)
        xpos = int(np.diff(BINS)[0])
        xpos_plot, median_pcom, _ =\
            binned_curve(df_curve, 'detected_CoM', 'sound_len', xpos=xpos,
                         bins=BINS,
                         return_data=True)
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, motor_updt_time,\
            x_val_at_updt, tr_indx_for_coms, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, xpos_plot, median_pcom,\
            rt_vals, rt_bins, None


def fitting(res_path='C:/Users/Alexandre/Desktop/CRM/brute_force/',
            data_path='C:/Users/Alexandre/Desktop/CRM/Alex/paper/results/',
            metrics='mse', objective='curve', bin_size=20, det_th=5):
    data_mat = np.load(data_path + 'all_tr_ac_pCoM_vs_prior_and_stim.npy')
    data_mat_norm = data_mat / np.nanmax(data_mat)
    data_curve = pd.read_csv(data_path + 'pcom_vs_rt.csv')
    tmp_data = data_curve['tmp_bin']
    data_curve_norm = data_curve['pcom'] / np.max(data_curve['pcom'])
    files = glob.glob(res_path+'*.npz')
    nan_penalty = 0.3
    w_rms = 0.5
    diff_mn = []
    diff_rms_mat = []
    diff_norm_mat = []
    curve_total = []
    rt_vals = []
    x_val = []
    max_ssim = metrics == 'ssim'
    file_index = []
    for i_f, f in enumerate(files):
        with np.load(f, allow_pickle=True) as data:
            if objective == 'matrix':
                matrix_list = data.get('pcom_matrix')
                for mat in matrix_list:
                    if metrics == 'mse' and np.mean(mat) > np.mean(data_mat):
                        mat_norm = mat / np.nanmax(mat)
                        diff_norm = np.sqrt(np.nansum(np.subtract(
                                mat_norm, data_mat_norm) ** 2))
                        diff_rms = np.sqrt(np.nansum(np.subtract(
                                mat, data_mat) ** 2))
                        diff = diff_norm / np.mean(diff_norm) + diff_rms /\
                            np.mean(diff_rms)
                        diff_mn.append(diff)
                        max_ssim = False
                    if metrics == 'ssim':
                        ssim_val = ssim(mat, data_mat) if not \
                            np.isnan(ssim(mat, data_mat))\
                            else 0
                        diff_mn.append(ssim_val)
                        max_ssim = True
            else:
                rt_vals_pcom = data.get('xpos_rt_pcom')
                rt_vals_pcom = [rt.to_numpy().astype(int) for rt in rt_vals_pcom]
                median_vals_pcom = data.get('median_pcom_rt')
                median_vals_pcom = [pcom_series.to_numpy() for pcom_series
                                    in median_vals_pcom]
                x_val_at_updt = data.get('x_val_at_updt_mat')
                perc_list = []
                for i_pcom, med_pcom in enumerate(median_vals_pcom):
                    curve_total.append(med_pcom)
                    rt_vals.append(rt_vals_pcom[i_pcom])
                    x_val.append(x_val_at_updt[i_pcom])
                    file_index.append(i_f)
    for curve_ind, _ in enumerate(rt_vals):
        x_perc = np.nanmean(np.abs(x_val[curve_ind]) > det_th)
        x_perc = 1
        perc_list.append(x_perc)
        tmp_simul =\
            np.array((rt_vals[curve_ind] - 1)/bin_size,
                     dtype=int)
        if len(rt_vals[curve_ind]) == 0 or np.isnan(x_perc):
            diff_mn.append(1e3)
            diff_norm_mat.append(1e3)
            diff_rms_mat.append(1e3)
        else:
            curve_tmp = curve_total[curve_ind]*x_perc + 1e-6
            curve_norm = curve_tmp / np.nanmax(curve_tmp)
            # diff_norm = np.subtract(curve_norm,
            #                         np.array(data_curve_norm[tmp_simul])) ** 2
            diff_norm = np.corrcoef(curve_norm,
                                    data_curve_norm[tmp_simul].values)
            diff_norm = diff_norm[0, 1] if not np.isnan(
                diff_norm[0, 1]) else -1
            num_nans = len(tmp_data) - len(tmp_simul)
            # diff_norm_mat.append(1-diff_norm+nan_penalty*num_nans)
            diff_norm_mat.append(1 - diff_norm + nan_penalty*num_nans)
            window = np.exp(-np.arange(len(tmp_simul))**1/40)
            window = 1
            diff_rms = np.subtract(curve_tmp,
                                   np.array(data_curve['pcom']
                                            [tmp_simul]) *
                                   window) ** 2
            diff_rms_mat.append(np.sqrt(np.nansum(diff_rms)) +
                                num_nans * nan_penalty)
            diff = (1 - w_rms)*(1 - diff_norm) + w_rms*np.sqrt(np.nansum(
                diff_rms)) + num_nans * nan_penalty
            # diff = (1-w_rms)*(np.sqrt(np.nansum(diff_norm))+nan_penalty*num_nans)\
            #     + w_rms*np.sqrt(np.nansum(diff_rms)) +\
            #     num_nans * nan_penalty
            diff_mn.append(diff) if not np.isnan(diff) else diff_mn.append(1e3)
            max_ssim = False
    plt.figure()
    plt.plot(curve_total[np.argmin(diff_rms_mat)],
             label='min rms')
    plt.plot(curve_total[np.argmin(diff_norm_mat)],
             label='min norm')
    plt.plot(curve_total[np.argmin(diff_mn)],
             label='min joined')
    # plt.plot(data_curve_norm, label='norm data')
    plt.plot(data_curve['pcom'], label='data')
    plt.ylabel('pCoM')
    plt.legend()
    if max_ssim:
        ind_min = np.argmax(diff_mn)
    else:
        ind_sorted = np.argsort(np.abs(diff_rms_mat))
        ind_min = ind_sorted[7]
        # second_in = (diff_mn*(diff_mn!=diff_mn[ind_min])).argmin()
    optimal_params = {}
    file_index = np.array(file_index)
    min_num = np.where(file_index == file_index[ind_min])[0][0]
    with np.load(files[file_index[ind_min]], allow_pickle=True) as data:
        for k in data.files:
            if k == 'rt_bins_all':
                optimal_params[k] = data[k]
            else:
                optimal_params[k] = data[k][ind_min - min_num]
    # For the best 10 configurations:
    plt.figure()
    plt.plot(data_curve['rt'], data_curve['pcom'], label='data', linestyle='',
             marker='o')
    for i in range(10):
        ind_min = ind_sorted[i+20]
        optimal_params = {}
        file_index = np.array(file_index)
        min_num = np.where(file_index == file_index[ind_min])[0][0]
        with np.load(files[file_index[ind_min]], allow_pickle=True) as data:
            for k in data.files:
                if k == 'rt_bins_all':
                    optimal_params[k] = data[k]
                else:
                    optimal_params[k] = data[k][ind_min - min_num]
        plt.plot(optimal_params['xpos_rt_pcom'],
                 optimal_params['median_pcom_rt'].values,
                 label=f'simul_{i}')
        plt.xlabel('RT (ms)')
        plt.ylabel('pCoM - detected')
        plt.legend()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    sns.heatmap(optimal_params['pcom_matrix'], ax=ax[0])
    ax[0].set_title('Simulation')
    sns.heatmap(data_mat, ax=ax[1])
    ax[1].set_title('Data')
    plt.figure()
    plt.plot(data_curve['rt'], data_curve['pcom'], label='data', linestyle='',
             marker='o')
    plt.plot(optimal_params['xpos_rt_pcom'],
             optimal_params['median_pcom_rt'].values,
             label='simul')
    plt.xlabel('RT (ms)')
    plt.ylabel('pCoM - detected')
    plt.legend()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax[0].plot(optimal_params['rt_bins_all'][0][1::],
               optimal_params['rt_vals_all'])
    ax[0].set_title('RT distribution')
    ax[0].set_xlabel('RT (ms)')
    cdf = np.cumsum(optimal_params['rt_vals_all']) /\
        np.sum(optimal_params['rt_vals_all'])
    ax[1].plot(optimal_params['rt_bins_all'][0][1::], cdf)
    ax[1].set_xlabel('RT (ms)')
    ax[1].set_title('CDF')


def loglikelihood(first_ind, E, zt, p_w_zt, p_w_stim, stim_res, N=int(1e3),
                  n=int(1e3)):
    a = 1  # bound evidence/action
    first_ev = [E[first_ind[i_t], i_t] for i_t in range(E.shape[1])]
    x = np.array(first_ev)
    s = np.std(first_ev)
    s0 = np.std(p_w_zt*zt)
    mu0 = np.mean(p_w_zt*zt)
    first_ind_llk = first_ind * stim_res
    M = p_w_stim * first_ind_llk
    D = 1/(2*s**2*first_ind_llk)
    sinus = np.sum([np.sin(K*np.pi/a*(mu0-s0**2*M/(2*D))) *
                    np.sin(K*np.pi*x/a) *
                    np.exp(-(D+s0**2/2)*(K*np.pi/a)**2)
                    for K in range(int(N))], axis=0)
    pL = 2 / a * np.exp((2*x - (2*mu0 - s0**2+M/(2*D)) * M/(4*D)) - M**2/(4*D))\
        * sinus
    for k in range(n):
        mean_1 = mu0 - s0**2*M/(2*D) + 2*k*a
        std_1_2 = 2*D + s0**2
        mean_2 = -mu0 + s0**2*M/(2*D) + 2*k*a
        N1 = std_1_2*np.random.randn() + mean_1
        N2 = std_1_2*np.random.randn() + mean_2
        pS = np.exp((2*(x-mu0)+s0**2*M/(2*D)-M)*M/(4*D))*(N1 - N2)


def run_model(stim, zt, coh, gt, configurations, jitters, stim_res,
              compute_trajectories=False, plot=False, existing_data=None,
              detect_CoMs_th=5, shuffle=False, all_trajs=False):
    def save_data():
        data_final = {'p_w_zt': p_w_zt_vals, 'p_w_stim': p_w_stim_vals,
                      'p_e_noise': p_e_noise_vals,
                      'p_com_bound': p_com_bound_vals,
                      'p_t_aff': p_t_aff_vals, 'p_t_eff': p_t_eff_vals,
                      'p_t_a': p_t_a_vals, 'p_w_a': p_w_a_vals,
                      'p_a_noise': p_a_noise_vals,
                      'p_w_updt': p_w_updt_vals,
                      'pcom_matrix': all_mats,
                      'x_val_at_updt_mat': x_val_at_updt_mat,
                      'tr_indx_for_coms_mat': tr_indx_for_coms_mat,
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
    num_tr = stim.shape[1]
    MT_slope = 0.15
    MT_intercep = 110
    p_w_zt_vals = []
    p_w_stim_vals = []
    p_e_noise_vals = []
    p_com_bound_vals = []
    p_t_aff_vals = []
    p_t_eff_vals = []
    p_t_a_vals = []
    p_w_a_vals = []
    p_a_noise_vals = []
    p_w_updt_vals = []
    all_mats = []
    x_val_at_updt_mat = []
    tr_indx_for_coms_mat = []
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
        print('p_w_a: '+str(conf[7]))
        print('p_a_noise: '+str(conf[8]))
        print('p_w_updt: '+str(conf[9]))
        start = time.time()
        if (np.sum(done_confs-np.array(conf).reshape(-1, 1), axis=0) != 0).all():
            p_w_zt = conf[0]+jitters[0]*np.random.rand()
            p_w_stim = conf[1]+jitters[1]*np.random.rand()
            p_e_noise = conf[2]+jitters[2]*np.random.rand()
            p_com_bound = conf[3]+jitters[3]*np.random.rand()
            p_t_aff = int(round(conf[4]+jitters[4]*np.random.rand()))
            p_t_eff = int(round(conf[5]++jitters[5]*np.random.rand()))
            p_t_a = int(round(conf[6]++jitters[6]*np.random.rand()))
            p_w_a = conf[7]+jitters[7]*np.random.rand()
            p_a_noise = conf[8]+jitters[8]*np.random.rand()
            p_w_updt = conf[9]+jitters[9]*np.random.rand()
            stim_temp =\
                np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                stim.shape[1]))))
            # TODO: get in a dict
            E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
                matrix, total_traj, init_trajs, final_trajs, motor_updt_time,\
                x_val_at_updt, tr_indx_for_coms, xpos_plot, median_pcom,\
                rt_vals, rt_bins, tr_index =\
                trial_ev_vectorized(zt=zt, stim=stim_temp, coh=coh,
                                    MT_slope=MT_slope, MT_intercep=MT_intercep,
                                    p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                    p_e_noise=p_e_noise, p_com_bound=p_com_bound,
                                    p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                                    num_tr=num_tr, p_w_a=p_w_a,
                                    p_a_noise=p_a_noise, p_w_updt=p_w_updt,
                                    compute_trajectories=compute_trajectories,
                                    stim_res=stim_res, all_trajs=all_trajs)

            print(np.mean(com))
            if plot:
                if compute_trajectories:
                    plotting(com=com, E=E, A=A, second_ind=second_ind,
                             first_ind=first_ind,
                             resp_first=resp_first, resp_fin=resp_fin,
                             pro_vs_re=pro_vs_re,
                             p_t_aff=p_t_aff, init_trajs=init_trajs,
                             total_traj=total_traj,
                             p_t_eff=p_t_eff, motor_updt_time=motor_updt_time,
                             tr_index=tr_index, p_com_bound=p_com_bound,
                             stim_res=stim_res)
                hits = resp_fin == gt
                detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
                detected_mat, _ =\
                    com_heatmap_jordi(zt[tr_index], coh[tr_index], detected_com,
                                      return_mat=True)
                data_to_plot = {'sound_len': (first_ind[tr_index]+p_t_eff -
                                              int(300/stim_res))*stim_res,
                                'CoM': com[tr_index],
                                'first_resp': resp_first[tr_index],
                                'final_resp': resp_fin[tr_index],
                                'hithistory': hits[tr_index],
                                'avtrapz': coh[tr_index],
                                'detected_com': detected_com,
                                'pro_vs_re': pro_vs_re[tr_index],
                                'detected_mat': detected_mat,
                                'matrix': matrix,
                                'MT': [len(t) for t in total_traj],
                                'zt': zt[tr_index]}
                plot_misc(data_to_plot=data_to_plot, stim_res=stim_res)
            p_w_zt_vals.append([conf[0], p_w_zt])
            p_w_stim_vals.append([conf[1], p_w_stim])
            p_e_noise_vals.append([conf[2], p_e_noise])
            p_com_bound_vals.append([conf[3], p_com_bound])
            p_t_aff_vals.append([conf[4], p_t_aff])
            p_t_eff_vals.append([conf[5], p_t_eff])
            p_t_a_vals.append([conf[6], p_t_a])
            p_w_a_vals.append([conf[7], p_w_a])
            p_a_noise_vals.append([conf[8], p_a_noise])
            p_w_updt_vals.append([conf[9], p_w_updt])
            all_mats.append(matrix)
            x_val_at_updt_mat.append(x_val_at_updt)
            tr_indx_for_coms_mat.append(indx_sh[tr_indx_for_coms])
            xpos_rt_pcom.append(xpos_plot)
            median_pcom_rt.append(median_pcom)
            rt_vals_all.append(rt_vals)
            if one_bins:
                rt_bins_all.append(rt_bins)
                one_bins = False
            if i_conf % 1000 == 0:
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
    p_w_a = 0.03
    p_a_noise = 0.04
    p_w_updt = 5
    """
    p_w_zt_list = np.linspace(0.15, 0.25, num=num_vals-2)
    p_w_stim_list = np.linspace(0.1, 0.2, num=num_vals-2)
    p_e_noise_list = np.linspace(0.04, 0.06, num=num_vals-2)
    p_com_bound_list = np.linspace(0., 0.2, num=num_vals-2)
    p_t_aff_list = np.linspace(7, 9, num=num_vals-2, dtype=int)
    p_t_eff_list = np.linspace(7, 9, num=num_vals-2, dtype=int)
    p_t_a_list = np.linspace(30, 50, num=num_vals-2, dtype=int)
    p_w_a_list = np.linspace(0.02, 0.04, num=num_vals-2)
    p_a_noise_list = np.linspace(0.01, 0.09, num=num_vals-2)
    p_w_updt_list = np.linspace(3, 7, num=num_vals-2)
    configurations = list(itertools.product(p_w_zt_list, p_w_stim_list,
                                            p_e_noise_list, p_com_bound_list,
                                            p_t_aff_list, p_t_eff_list, p_t_a_list,
                                            p_w_a_list, p_a_noise_list,
                                            p_w_updt_list))
    if num_vals == 1:
        jitters = np.repeat(0.01, 10)
    else:
        jitters = [np.diff(p_w_zt_list)[0]/factor,
                   np.diff(p_w_stim_list)[0]/factor,
                   0.0001,
                   np.diff(p_com_bound_list)[0]/factor,
                   np.diff(p_t_aff_list)[0]/factor,
                   np.diff(p_t_eff_list)[0]/factor,
                   np.diff(p_t_a_list)[0]/factor,
                   np.diff(p_w_a_list)[0]/factor,
                   0.0001,
                   np.diff(p_w_updt_list)[0]/factor]
    return configurations, jitters


def data_augmentation(stim, daf, sigma=0):
    augm_stim = np.zeros((daf*stim.shape[0], stim.shape[1]))
    for tmstp in range(stim.shape[0]):
        augm_stim[daf*tmstp:daf*(tmstp+1), :] =\
            np.random.randn()*sigma+stim[tmstp, :]
    return augm_stim


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    # tests_trajectory_update(remaining_time=100, w_updt=10)
    num_tr = int(2e4)
    load_data = True
    new_sample = False
    single_run = True
    shuffle = True
    simulate = True
    parallel = False
    data_augment_factor = 10
    if simulate:
        if load_data:
            if new_sample:
                stim, zt, coh, gt, com =\
                    get_data_and_matrix(dfpath=DATA_FOLDER,
                                        num_tr_per_rat=int(7e3),
                                        after_correct=True)
                data = {'stim': stim, 'zt': zt, 'coh': coh, 'gt': gt, 'com': com}
                np.savez(DATA_FOLDER+'/sample_'+str(time.time())[-5:]+'.npz',
                         **data)
            else:
                files = glob.glob(DATA_FOLDER+'/sample_*')
                data = np.load(files[np.random.choice(a=len(files))])
                stim = data['stim']
                zt = data['zt']
                coh = data['coh']
                com = data['com']
                gt = data['gt']
            stim = data_augmentation(stim=stim, daf=data_augment_factor)
            stim_res = 50/data_augment_factor
        else:
            num_timesteps = 1000
            zt =\
                np.random.rand(num_tr)*2*(-1.0)**np.random.randint(-1, 1,
                                                                   size=num_tr)
            stim = \
                np.random.rand(num_tr)*(-1.0)**np.random.randint(-1, 1,
                                                                 size=num_tr) +\
                np.random.randn(num_timesteps, num_tr)*1e-1
            stim_res = 1

        if single_run:
            p_t_aff = 10
            p_t_eff = 6
            p_t_a = 35
            p_w_zt = 0.15
            p_w_stim = 0.15
            p_e_noise = 0.05
            p_com_bound = 0.
            p_w_a = 0.03
            p_a_noise = 0.06
            p_w_updt = 0.1  # 65
            compute_trajectories = True
            plot = True
            all_trajs = True
            configurations = [(p_w_zt, p_w_stim, p_e_noise, p_com_bound, p_t_aff,
                              p_t_eff, p_t_a, p_w_a, p_a_noise, p_w_updt)]
            jitters = len(configurations[0])*[0]
            print('Number of trials: ' + str(stim.shape[1]))
            stim = stim[:, :int(num_tr)]
            zt = zt[:int(num_tr)]
            coh = coh[:int(num_tr)]
            com = com[:int(num_tr)]
            gt = gt[:int(num_tr)]
        else:
            configurations, jitters = set_parameters(num_vals=5)
            compute_trajectories = True
            plot = False
            all_trajs = False
        existing_data = None  # SV_FOLDER+'/results/all_results_1.npz'
        if parallel:
            configurations = list(configurations)
            num_cores = int(mp.cpu_count())
            step = int(np.ceil(len(configurations)/num_cores))
            Parallel(n_jobs=num_cores)\
                (delayed(run_model)(stim=stim, zt=zt, coh=coh, gt=gt,
                                    configurations=configurations
                                    [int(i_par*step):int((i_par+1)*step)],
                                    jitters=jitters,
                                    compute_trajectories=compute_trajectories,
                                    plot=plot, stim_res=stim_res,
                                    existing_data=existing_data,
                                    shuffle=shuffle, all_trajs=False)
                 for i_par in range(num_cores))
        else:
            run_model(stim=stim, zt=zt, coh=coh, gt=gt,
                      configurations=configurations, jitters=jitters,
                      compute_trajectories=compute_trajectories,
                      plot=plot, stim_res=stim_res,
                      existing_data=existing_data,
                      shuffle=shuffle, all_trajs=all_trajs)
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
