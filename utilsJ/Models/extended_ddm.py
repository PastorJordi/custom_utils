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
sys.path.append("C:/Users/alexg/Documents/GitHub/custom_utils/")
import utilsJ
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve
# import os
SV_FOLDER = '/home/molano/Dropbox/project_Barna/ChangesOfMind/'  # Manuel
# SV_FOLDER = 'C:/Users/alexg/Desktop/CRM/Alex/paper'  # Alex
DATA_FOLDER = '/home/molano/ChangesOfMind/data/'
# DATA_FOLDER = 'C:/Users/alexg/Desktop/CRM/Alex/paper/data/'



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


def draw_lines(ax, frst, sec, p_t_aff):
    ax[0].axhline(y=1, color='purple', linewidth=2)
    ax[0].axhline(y=-1, color='green', linewidth=2)
    ax[0].axhline(y=0, linestyle='--', color='k', linewidth=0.7)
    ax[0].axhline(y=0.5, color='purple', linewidth=1, linestyle='--')
    ax[0].axhline(y=-0.5, color='green', linewidth=1, linestyle='--')
    ax[1].axhline(y=1, color='k', linewidth=1, linestyle='--')
    for a in ax:
        a.axvline(x=frst, color='c', linewidth=1, linestyle='--')
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.axvline(x=sec, color='c', linewidth=1, linestyle='--')


def plotting(com, E, A, second_ind, first_ind, resp_first, resp_fin, pro_vs_re,
             p_t_aff, init_trajs, total_traj, p_t_eff, motor_updt_time,
             stim_res=50, trial=0):
    f, ax = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
    ax = ax.flatten()
    ax[6].set_xlabel('Time (ms)')
    ax[7].set_xlabel('Time (ms)')
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
        if len(trials_temp) > 0:
            trial = trials_temp[t]
            draw_lines(ax[np.array(a)], frst=first_ind[trial],
                       sec=second_ind[trial], p_t_aff=p_t_aff)
            color1 = 'green' if resp_first[trial] < 0 else 'purple'
            color2 = 'green' if resp_fin[trial] < 0 else 'purple'

            ax[a[0]].plot(E[:second_ind[trial]+1, trial], color=color2,
                          alpha=0.7)
            ax[a[0]].plot(E[:first_ind[trial]+1, trial], color=color1, lw=2)
            ax[a[1]].plot(A[:second_ind[trial]+1, trial], color=color2,
                          alpha=0.7)
            ax[a[1]].plot(A[:first_ind[trial]+1, trial], color=color1, lw=2)
            # ax[a[0]].set_ylim([-1.5, 1.5])
            # ax[a[1]].set_ylim([-0.1, 1.5])
            ax[a[0]].set_ylabel(l+' EA')
            ax[a[1]].set_ylabel(l+' AI')
            # trajectories
            sec_ev = round(E[second_ind[trial], trial], 2)
            # updt_motor = first_ind[trial]+motor_updt_time[trial]
            init_motor = first_ind[trial]+p_t_eff
            xs = init_motor+np.arange(0, len(total_traj[trial]))/stim_res
            max_xlim = max(max_xlim, np.max(xs))
            ax[a[2]].plot(xs, total_traj[trial],
                          label=f'Updated traj., E:{sec_ev}')
            first_ev = round(E[first_ind[trial], trial], 2)
            xs = init_motor+np.arange(0, len(init_trajs[trial]))/stim_res
            max_xlim = max(max_xlim, np.max(xs))
            ax[a[2]].plot(xs, init_trajs[trial],
                          label=f'Initial traj. E:{first_ev}')
            ax[a[2]].set_ylabel(l+', y(px)')
            ax[a[2]].set_ylabel(l+', y(px)')
            ax[a[2]].legend()
        else:
            print('There are no '+l)
    for a in ax:
        a.set_xlim([0, max_xlim])
    f.savefig(SV_FOLDER+'/figures/example_trials.svg', dpi=400,
              bbox_inches='tight')


def plot_misc(df_plot):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ax = ax.flatten()
    binned_curve(df_plot, 'CoM', 'sound_len', bins=np.linspace(0, 250, 26),
                 ax=ax[0])
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


def get_data_and_matrix(dfpath='C:/Users/alexg/Desktop/CRM/Alex/paper/',
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
        gt_tmp = np.array(df.rewside) * 2 - 1
        prior = np.concatenate((prior, prior_tmp[indx]))
        stim = np.concatenate((stim, stim_tmp[indx, :]))
        coh = np.concatenate((coh, coh_mat[indx]))
        com = np.concatenate((com, com_tmp[indx]))
        gt = np.concatenate((gt, gt_tmp[indx]))
        end = time.time()
        print(f)
        print(end - start_1)
        print(len(df))
    print(end - start)
    print('Ended loading data, start computing matrix')
    matrix, _ = com_heatmap_jordi(prior, coh, com, return_mat=True)
    np.save(SV_FOLDER + '/results/CoM_vs_prior_and_stim.npy', matrix)
    stim = stim.T
    return stim, prior, coh, gt, com  # , matrix


def trial_ev_vectorized(zt, stim, coh, MT_slope, MT_intercep, p_w_zt, p_w_stim,
                        p_e_noise, p_com_bound, p_t_eff, p_t_aff,
                        p_t_a, p_w_a, p_a_noise, p_w_updt, num_tr,
                        compute_trajectories=False, num_trials_per_session=600,
                        proactive_integration=True, all_trajs=False,
                        perc_traj_com=0.1):
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
    prior = zt*p_w_zt
    Ve = np.concatenate((np.zeros((p_t_aff, num_tr)), stim*p_w_stim))
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
        pro_vs_re.append(np.argmin([hit_action, hit_bound]))
        first_ind.append(hit_dec)
        first_ev.append(E[hit_dec, i_t])
        # first response
        resp_first[i_t] *= (-1)**(E[hit_dec, i_t] < 0)
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
    if compute_trajectories:
        # Trajectories
        print('Starting with trajectories')
        RLresp = resp_fin
        prechoice = resp_first
        jerk_lock_ms = 0
        initial_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
        indx_trajs = np.arange(len(first_ind)) if all_trajs else np.where(com)[0]
        # initial positions, speed and acc; final position, speed and acc
        init_trajs = []
        final_trajs = []
        total_traj = []
        motor_updt_time = []
        x_val_at_updt = []
        for i_t in indx_trajs:
            # pre-planned Motor Time, the modulo prevents trial-index from
            # growing indefinitely
            if np.random.rand() < perc_traj_com:
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
                else:
                    x_val_at_updt.append(0)
                # if com[i_t] == 0 and pro_vs_re[i_t] == 1 and\
                #    np.abs(second_ev[i_t]) > 2*np.abs(first_ev[i_t]):
                #     print(MT)
                #     print(first_resp_len)
                #     print(remaining_m_time)
                #     print(second_response_len)
                #     print('------------------')
                #     plt.figure()
                #     plt.plot(init_trajs[-1])
                #     plt.plot(total_traj[-1], '--')
                #     asd
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, motor_updt_time,\
            x_val_at_updt
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None


def matrix_comparison(res_path='C:/Users/alexg/Dropbox/results/',
                      mat_path='C:/Users/alexg/Desktop/CRM/Alex/paper/results/'):
    data_mat = np.load(mat_path + 'CoM_vs_prior_and_stim.npy')
    data_mat[np.isnan(data_mat)] = 0
    data_mat_norm = data_mat / np.nanmax(data_mat)
    files = glob.glob(res_path+'*all_results.npz')
    diff_mn = []
    for f in files:
        with np.load(f, allow_pickle=True) as data:
            matrix_list = data.get('matrix')
            for mat in matrix_list:
                # if np.sum(np.isnan(mat)) > 45:
                #     continue
                # else:
                mat[np.isnan(mat)] = 0
                mat_norm = mat / np.nanmax(mat)
                diff = np.sqrt(np.nansum(np.subtract(mat_norm,
                                                     data_mat_norm) ** 2))
                diff_mn.append(diff)
    ind_min = np.nanargmin(np.abs(diff_mn))
    # data_min = files[ind_min]
    optimal_params = {}
    with np.load(files[0], allow_pickle=True) as data:
        for k in data.files:
            optimal_params[k] = data[k][ind_min]
    optimal_params['matrix'][np.isnan(optimal_params['matrix'])] = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    sns.heatmap(optimal_params['matrix'], ax=ax[0])
    ax[0].set_title('Simulation')
    sns.heatmap(data_mat, ax=ax[1])
    ax[1].set_title('Data')
    return data_mat, optimal_params


def run_model(stim, zt, coh, gt, configurations, jitters, stim_res,
              compute_trajectories=False, plot=False):
    def save_data():
        data_final = {'p_w_zt': p_w_zt_vals, 'p_w_stim': p_w_stim_vals,
                      'p_e_noise': p_e_noise_vals,
                      'p_com_bound': p_com_bound_vals,
                      'p_t_aff': p_t_aff_vals, 'p_t_eff': p_t_eff_vals,
                      'p_t_a': p_t_a_vals, 'p_w_a': p_w_a_vals,
                      'p_a_noise': p_a_noise_vals,
                      'p_w_updt': p_w_updt_vals,
                      'matrix': all_mats, 'x_val_at_updt_mat': x_val_at_updt_mat}
        np.savez(SV_FOLDER+'/results/all_results.npz', **data_final)
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
    for i_conf, conf in enumerate(configurations):
        start = time.time()
        p_w_zt = conf[0]+jitters[0]*np.random.rand()
        p_w_stim = conf[1]+jitters[1]*np.random.rand()
        p_e_noise = conf[2]+jitters[2]*np.random.rand()
        p_com_bound = conf[3]+jitters[3]*np.random.rand()
        p_t_aff = round(conf[4]+jitters[4]*np.random.rand())
        p_t_eff = round(conf[5]++jitters[5]*np.random.rand())
        p_t_a = round(conf[6]++jitters[6]*np.random.rand())
        p_w_a = conf[7]+jitters[7]*np.random.rand()
        p_a_noise = conf[8]+jitters[8]*np.random.rand()
        p_w_updt = conf[9]+jitters[9]*np.random.rand()
        stim_temp =\
            np.concatenate((stim, np.zeros((p_t_aff+p_t_eff, stim.shape[1]))))
        print('--------------')
        print('p_w_zt: '+str(p_w_zt))
        print('p_w_stim: '+str(p_w_stim))
        print('p_e_noise: '+str(p_e_noise))
        print('p_com_bound: '+str(p_com_bound))
        print('p_t_aff: '+str(p_t_aff))
        print('p_t_eff: '+str(p_t_eff))
        print('p_t_a: '+str(p_t_a))
        print('p_w_a: '+str(p_w_a))
        print('p_a_noise: '+str(p_a_noise))
        print('p_w_updt: '+str(p_w_updt))
        E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, motor_updt_time,\
            x_val_at_updt =\
            trial_ev_vectorized(zt=zt, stim=stim_temp, coh=coh, MT_slope=MT_slope,
                                MT_intercep=MT_intercep, p_w_zt=p_w_zt,
                                p_w_stim=p_w_stim, p_e_noise=p_e_noise,
                                p_com_bound=p_com_bound, p_t_aff=p_t_aff,
                                p_t_eff=p_t_eff, p_t_a=p_t_a, num_tr=num_tr,
                                p_w_a=p_w_a, p_a_noise=p_a_noise,
                                p_w_updt=p_w_updt,
                                compute_trajectories=compute_trajectories)
        end = time.time()
        print(np.mean(com))
        print(end-start)
        if plot:
            if compute_trajectories:
                plotting(com=com, E=E, A=A, second_ind=second_ind,
                         first_ind=first_ind,
                         resp_first=resp_first, resp_fin=resp_fin,
                         pro_vs_re=pro_vs_re,
                         p_t_aff=p_t_aff, init_trajs=init_trajs,
                         total_traj=total_traj,
                         p_t_eff=p_t_eff, motor_updt_time=motor_updt_time,
                         stim_res=stim_res)
            hits = resp_fin == gt
            data_to_plot = {'sound_len': first_ind*5, 'CoM': com,
                            'first_resp': resp_first, 'final_resp': resp_fin,
                            'hithistory': hits, 'avtrapz': coh}
            df_plot = pd.DataFrame(data_to_plot)
            plot_misc(df_plot)
            plt.figure()
            sns.heatmap(matrix)
        p_w_zt_vals.append([conf[0], p_w_zt])
        p_w_stim_vals.append([conf[1], p_w_stim])
        p_e_noise_vals.append([conf[2], p_e_noise])
        p_com_bound_vals.append([conf[3], p_e_noise])
        p_t_aff_vals.append([conf[4], p_t_aff])
        p_t_eff_vals.append([conf[5], p_t_eff])
        p_t_a_vals.append([conf[6], p_t_a])
        p_w_a_vals.append([conf[7], p_w_a])
        p_a_noise_vals.append([conf[8], p_a_noise])
        p_w_updt_vals.append([conf[9], p_w_updt])
        all_mats.append(matrix)
        x_val_at_updt_mat.append(x_val_at_updt)
        if i_conf % 100 == 0:
            save_data()
    save_data()


def set_parameters(num_vals=4, factor=8):
    p_w_zt_list = np.linspace(0.005, 0.5, num=num_vals)
    p_w_stim_list = np.linspace(0.005, 0.5, num=num_vals)
    p_e_noise_list = [0.05, 0.1]  # np.linspace(0.005, 10, num=num_vals)
    p_com_bound_list = np.linspace(0.1, 1, num=num_vals)
    p_t_aff_list = np.array([8, 16, 24])
    p_t_eff_list = np.array([8, 16, 24])
    p_t_a_list = np.array([0, 8])
    p_w_a_list = np.linspace(0.005, 0.5, num=num_vals)
    p_a_noise_list = [0.05, 0.1]  # np.linspace(0.01, 0.1, num=num_vals)
    p_w_updt_list = [1]  # np.linspace(0.1, 2, num=num_vals)
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
                   0.01,
                   np.diff(p_com_bound_list)[0]/factor,
                   np.diff(p_t_aff_list)[0]/factor,
                   np.diff(p_t_eff_list)[0]/factor,
                   np.diff(p_t_a_list)[0]/factor,
                   np.diff(p_w_a_list)[0]/factor,
                   0.01,
                   0.1]
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
    num_tr = int(1e5)
    load_data = True
    new_sample = False
    single_run = False
    data_augment_factor = 10
    if load_data:
        if new_sample:
            stim, zt, coh, gt, com =\
                get_data_and_matrix(dfpath=DATA_FOLDER,
                                    num_tr_per_rat=int(1e4), after_correct=True)
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
        zt = np.random.rand(num_tr)*2*(-1.0)**np.random.randint(-1, 1, size=num_tr)
        stim = \
            np.random.rand(num_tr)*(-1.0)**np.random.randint(-1, 1, size=num_tr) +\
            np.random.randn(num_timesteps, num_tr)*1e-1
        stim_res = 1

    if single_run:
        MT_slope = 0.15
        MT_intercep = 110
        p_t_aff = 8
        p_t_eff = 16
        p_t_a = 0
        p_w_zt = 0.05
        p_w_stim = 0.17
        p_e_noise = 0.1
        p_com_bound = 0.7
        p_w_a = 0.005
        p_a_noise = 0.1
        p_w_updt = 3
        compute_trajectories = False
        plot = True
        configurations = [(p_w_zt, p_w_stim, p_e_noise, p_com_bound, p_t_aff,
                          p_t_eff, p_t_a, p_w_a, p_a_noise, p_w_updt)]
        jitters = len(configurations[0])*[0]
        stim = stim[:, :int(num_tr)]
        zt = zt[:int(num_tr)]
        coh = coh[:int(num_tr)]
        com = com[:int(num_tr)]
        gt = gt[:int(num_tr)]
    else:
        configurations, jitters = set_parameters(num_vals=4)
        compute_trajectories = True
        plot = False

    run_model(stim=stim, zt=zt, coh=coh, gt=gt, configurations=configurations,
              jitters=jitters, compute_trajectories=compute_trajectories,
              plot=plot, stim_res=stim_res)
