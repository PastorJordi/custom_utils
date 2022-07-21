# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import scipy as sp

# ddm


def draw_lines(ax, frst, sec, p_t_eff):
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


def plotting(com, E, second_ind, first_ind, resp_first, resp_fin, pro_vs_re,
             p_t_eff, init_trajs, total_traj, p_t_m, motor_updt_time, trial=0):
    f, ax = plt.subplots(nrows=3, ncols=4, figsize=(15, 12))
    ax = ax.flatten()
    ax[6].set_xlabel('Time (ms)')
    ax[7].set_xlabel('Time (ms)')
    axes = [np.array([0, 4, 8])+i for i in range(4)]
    trials = [0, 0, 1, 1]
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
        trial = np.where(m)[0][t]
        draw_lines(ax[np.array(a)], frst=first_ind[trial], sec=second_ind[trial],
                   p_t_eff=p_t_eff)
        color1 = 'green' if resp_first[trial] < 0 else 'purple'
        color2 = 'green' if resp_fin[trial] < 0 else 'purple'

        ax[a[0]].plot(E[:first_ind[trial]+p_t_eff+1, trial], color=color2,
                      alpha=0.7)
        ax[a[0]].plot(E[:first_ind[trial]+1, trial], color=color1, lw=2)
        ax[a[1]].plot(A[:first_ind[trial]+p_t_eff+1, trial], color=color2,
                      alpha=0.7)
        ax[a[1]].plot(A[:first_ind[trial]+1, trial], color=color1, lw=2)
        ax[a[0]].set_ylim([-1.5, 1.5])
        ax[a[1]].set_ylim([-0.1, 1.5])
        ax[a[0]].set_ylabel(l+' EA')
        ax[a[1]].set_ylabel(l+' AI')
        # trajectories
        sec_ev = round(E[second_ind[trial], trial], 2)
        # updt_motor = first_ind[trial]+motor_updt_time[trial]
        init_motor = first_ind[trial]+p_t_m
        xs = np.arange(init_motor, init_motor+len(total_traj[trial]))
        max_xlim = max(max_xlim, np.max(xs))
        ax[a[2]].plot(xs, total_traj[trial], label=f'Updated traj., E:{sec_ev}')
        first_ev = round(E[first_ind[trial], trial], 2)
        xs = np.arange(init_motor, init_motor+len(init_trajs[trial]))
        max_xlim = max(max_xlim, np.max(xs))
        ax[a[2]].plot(xs, init_trajs[trial], label=f'Initial traj. E:{first_ev}')
        ax[a[2]].set_ylabel(l+', y(px)')
        ax[a[2]].set_ylabel(l+', y(px)')
        ax[a[2]].legend()
    for a in ax:
        a.set_xlim([0, max_xlim])


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
    tmp["binned_prior"] = np.nan
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

    tmp["binned_stim"] = np.nan
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
        switch = (tmp.loc[(tmp.com == True) & (tmp.binned_stim == i)]
                  .groupby("binned_prior")["binned_prior"].count())
        nobs = (switch + tmp.loc[(tmp.com == False) & (tmp.binned_stim == i)]
                .groupby("binned_prior")["binned_prior"].count())
        # fill where there are no CoM (instead it will be nan)
        nobs.loc[nobs.isna()] = (tmp.loc[(tmp.com == False) &
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


def get_data_and_matrix(
        dfpath='C:/Users/alexg/Desktop/CRM/Alex/paper/LE42_clean.pkl',
        savepath='C:/Users/alexg/Documents/GitHub/custom_utils/utilsJ/Models'):
    # import data for 1 rat
    print('Loading data')
    df = pd.read_pickle(dfpath)
    df_rat = df[['origidx', 'res_sound', 'R_response', 'trajectory_y', 'coh2',
                 'CoM_sugg']]
    df_rat["priorZt"] = np.nansum(
            df[["dW_lat", "dW_trans"]].values, axis=1)
    print('Ended loading data')
    stim = np.array([stim for stim in df_rat.res_sound])
    prior = df_rat.priorZt
    com = df_rat.CoM_sugg
    matrix, _ = com_heatmap_jordi(prior, df_rat.coh2, com, return_mat=True)
    np.save(savepath + '/CoM_vs_prior_and_stim.npy', matrix)
    return df_rat, stim, prior, com, matrix


def trial_ev_vectorized(zt, stim, MT_slope, MT_intercep, p_w_zt, p_w_stim,
                        p_e_noise, p_com_bound, p_t_m, p_t_eff,
                        p_t_a, p_w_a, p_a_noise, p_w_updt, num_tr,
                        trajectories=False):
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
    p_t_m : float
        fitting parameter: standard deviation of evidence noise (gaussian).
    p_t_eff : float
        fitting parameter: afferent latency time to integrate stimulus.
    p_t_a : float
        fitting parameter: afferent latency time for action integration.
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
    Ve = np.concatenate((np.zeros((p_t_eff, num_tr)), stim*p_w_stim))
    Va = p_w_a
    # trial_dur = 1  # trial duration (s)
    N = Ve.shape[0]  # int(trial_dur/dt)  # number of timesteps
    dW = np.random.randn(N, num_tr)*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+Va
    dA[:p_t_a, :] = 0
    dW[0, :] = prior  # +np.random.randn(p_t_eff, num_tr)*p_e_noise
    dA[0, :] = prior  # +np.random.randn(p_t_a, num_tr)*p_a_noise
    E = np.cumsum(dW, axis=0)
    A = np.cumsum(dA, axis=0)
    com = False
    # reaction_time = 300
    first_ind = []
    second_ind = []
    pro_vs_re = []
    first_ev = []
    second_ev = []
    resp_first = np.ones(E.shape[1])
    resp_fin = np.ones(E.shape[1])
    for i_c in range(E.shape[1]):
        indx_hit_bound = np.abs(E[:, i_c]) >= bound
        indx_hit_action = np.abs(A[:, i_c]) >= bound_a
        hit_bound = E.shape[0]-1
        hit_action = E.shape[0]-1
        if (indx_hit_bound).any():
            hit_bound = np.where(indx_hit_bound)[0][0]
        if (indx_hit_action).any():
            hit_action = np.where(indx_hit_action)[0][0]
        hit_dec = min(hit_bound, hit_action)  # reactive or proactive
        pro_vs_re.append(np.argmin([hit_action, hit_bound]))
        first_ind.append(hit_dec)
        first_ev.append(E[hit_dec, i_c])
        # first response
        resp_first[i_c] *= (-1)**(E[hit_dec, i_c] < 0)
        com_bound_temp = (-resp_first[i_c])*p_com_bound
        # second response
        second_thought = hit_dec+p_t_eff+p_t_m
        indx_fin_ch = min(second_thought, E.shape[0]-1)
        post_dec_integration = E[hit_dec:indx_fin_ch, i_c]-com_bound_temp
        indx_com =\
            np.where(np.sign(E[hit_dec, i_c]) != np.sign(post_dec_integration))[0]
        indx_update_ch = second_thought if len(indx_com) == 0 else\
            (indx_com[0] + hit_dec)
        resp_fin[i_c] = resp_first[i_c] if len(indx_com) == 0 else -resp_first[i_c]
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_c])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind).astype(int)
    pro_vs_re = np.array(pro_vs_re)
    matrix, _ = com_heatmap_jordi(zt, np.mean(stim, axis=0), com,
                                  return_mat=True)
    if trajectories:
        # Trajectories
        print('Starting with trajectories')
        RLresp = resp_fin
        prechoice = resp_first
        jerk_lock_ms = 0
        initial_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
        # initial positions, speed and acc; final position, speed and acc
        init_trajs = []
        final_trajs = []
        total_traj = []
        motor_updt_time = []
        for i_t in range(E.shape[1]):
            MT = MT_slope*i_t + MT_intercep  # pre-planned Motor Time
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
            t_ind = int(p_t_m+second_ind[i_t] - first_ind[i_t])  # time index
            motor_updt_time.append(t_ind)
            vel = velocities[t_ind]  # velocity at the timepoint
            acc = accelerations[t_ind]
            pos = prior0[t_ind]  # position
            mu_update = np.array([pos, vel, acc, 75*RLresp[i_t],
                                  0, 0]).reshape(-1, 1)
            # new mu, considering new position/speed/acceleration
            remaining_m_time = first_resp_len-t_ind
            sign_ = resp_first[i_t]
            # second_response_len: time left affected by the evidence on the
            second_response_len =\
                float(remaining_m_time-sign_*p_w_updt*second_ev[i_t])
            # SECOND readout
            traj_fin = compute_traj(jerk_lock_ms, mu=mu_update,
                                    resp_len=second_response_len)
            final_trajs.append(traj_fin)
            # joined trajectories
            traj_before_uptd = prior0[0:t_ind]
            total_traj.append(np.concatenate((traj_before_uptd,  traj_fin)))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, motor_updt_time
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None


def matrix_comparison(matrix, npypath='C:/Users/alexg/Documents/GitHub/' +
                      'custom_utils/utilsJ/Models/',
                      npyname='CoM_vs_prior_and_stim.npy'):
    print('Starting comparison')
    matrix_data = np.load(npypath+npyname)
    MSE = np.square(np.subtract(matrix, matrix_data))
    rmse = np.sqrt(MSE)
    print('RMSE: ')
    print(rmse)
    sns.heatmap(rmse)
    return rmse


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    num_tr = int(1e4)
    zt = np.random.rand(num_tr)*2*(-1.0)**np.random.randint(-1, 1, size=num_tr)
    p_t_eff = 40
    p_t_a = p_t_eff
    num_timesteps = 1000
    stim = np.random.rand(num_tr)*(-1.0)**np.random.randint(-1, 1, size=num_tr) +\
        np.random.randn(num_timesteps+p_t_eff, num_tr)*1e-1
    MT_slope = 0.15
    MT_intercep = 110
    p_t_m = 10
    p_w_zt = 0.4
    p_w_stim = 0.2
    p_e_noise = 0.2
    p_com_bound = 0.5
    p_w_a = 0.05
    p_a_noise = 0.05
    p_w_updt = 15
    E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re, matrix,\
        total_traj, init_trajs, final_trajs, motor_updt_time =\
        trial_ev_vectorized(zt=zt, stim=stim, MT_slope=MT_slope,
                            MT_intercep=MT_intercep, p_w_zt=p_w_zt,
                            p_w_stim=p_w_stim, p_e_noise=p_e_noise,
                            p_com_bound=p_com_bound, p_t_m=p_t_m,
                            p_t_eff=p_t_eff, p_t_a=p_t_a, num_tr=num_tr,
                            p_w_a=p_w_a, p_a_noise=p_a_noise, p_w_updt=p_w_updt,
                            trajectories=False)
    rmse = matrix_comparison(matrix)
    import sys
    sys.exit()
    plotting(E=E, com=com, second_ind=second_ind, first_ind=first_ind,
             resp_first=resp_first, resp_fin=resp_fin, pro_vs_re=pro_vs_re,
             p_t_eff=p_t_eff, init_trajs=init_trajs, total_traj=total_traj,
             p_t_m=p_t_m, motor_updt_time=motor_updt_time)
