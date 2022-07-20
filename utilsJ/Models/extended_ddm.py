# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

# ddm


def draw_lines(ax, frst, sec, p_t_eff):
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[0].axhline(y=1, color='purple', linewidth=2)
    ax[0].axhline(y=-1, color='green', linewidth=2)
    ax[0].axhline(y=0, linestyle='--', color='k', linewidth=0.7)
    ax[0].axhline(y=0.5, color='purple', linewidth=1, linestyle='--')
    ax[0].axhline(y=-0.5, color='green', linewidth=1, linestyle='--')
    ax[1].axhline(y=1, color='k', linewidth=1, linestyle='--')
    ax[0].axvline(x=frst, color='c', linewidth=1, linestyle='--')
    ax[1].axvline(x=frst, color='c', linewidth=1, linestyle='--')
    if sec < frst+p_t_eff:
        ax[0].axvline(x=sec, color='c', linewidth=1, linestyle='--')
        ax[1].axvline(x=sec, color='c', linewidth=1, linestyle='--')


def plotting(com, E, second_ind, first_ind, resp_first, resp_fin, pro_vs_re,
             p_t_eff, trial=0):
    f, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))
    ax = ax.flatten()
    ax[0].set_title('CoM')
    ax[1].set_title('No-CoM')
    ax[6].set_xlabel('Time (ms)')
    ax[7].set_xlabel('Time (ms)')
    axes = [[0, 2], [1, 3], [4, 6], [5, 7]]
    trials = [0, 0, 1, 1]
    mat_indx = [np.logical_and(com, pro_vs_re == 0),
                np.logical_and(~com, pro_vs_re == 0),
                np.logical_and(com, pro_vs_re == 1),
                np.logical_and(~com, pro_vs_re == 1)]
    y_lbls = ['CoM Proactive', 'No CoM Proactive', 'CoM Reactive',
              'No CoM Reactive']
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


def plotting_trajs(init_trajs, total_traj, com, pro_vs_re):
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))
    ax = ax.flatten()
    ax[0].set_title('CoM')
    ax[1].set_title('No-CoM')
    ax[2].set_xlabel('Time (ms)')
    ax[3].set_xlabel('Time (ms)')
    trials = [0, 0, 1, 1]
    mat_indx = [np.logical_and(com, pro_vs_re == 0),
                np.logical_and(~com, pro_vs_re == 0),
                np.logical_and(com, pro_vs_re == 1),
                np.logical_and(~com, pro_vs_re == 1)]
    y_lbls = ['CoM Proactive', 'No CoM Proactive', 'CoM Reactive',
              'No CoM Reactive']
    for i, (t, m, l) in enumerate(zip(trials, mat_indx, y_lbls)):
        trial = np.where(m)[0][t]
        ax[i].plot(total_traj[trial], label='Updated traj.')
        ax[i].plot(init_trajs[trial], label='Initial traj.')
        ax[i].set_ylabel(l+', y(px)')
        ax[i].set_ylabel(l+', y(px)')
        ax[i].legend()


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


def trial_ev_vectorized(zt, stim, MT_slope, MT_intercep, p_w_zt, p_w_stim,
                        p_e_noise, p_com_bound, p_t_m, p_t_eff,
                        p_t_a, num_tr, p_w_a, p_a_noise, p_w_updt,
                        plot=False):
    bound = 1
    bound_a = 1
    prior = zt*p_w_zt
    Ve = stim*p_w_stim
    Va = p_w_a
    # trial_dur = 1  # trial duration (s)
    N = stim.shape[0]  # int(trial_dur/dt)  # number of timesteps
    dW = np.random.randn(N, num_tr)*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+Va
    dW[:p_t_eff, :] = 0
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
        second_thought = hit_dec+p_t_eff
        indx_fin_ch = min(second_thought, E.shape[0]-1)
        post_dec_integration = E[hit_dec:indx_fin_ch, i_c]-com_bound_temp
        indx_com =\
            np.where(np.sign(E[hit_dec, i_c]) != np.sign(post_dec_integration))[0]
        indx_update_ch = hit_dec + p_t_eff if len(indx_com) == 0 else\
            (indx_com[0] + hit_dec)
        resp_fin[i_c] = resp_first[i_c] if len(indx_com) == 0 else -resp_first[i_c]
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_c])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind).astype(int)
    pro_vs_re = np.array(pro_vs_re)
    RLresp = resp_fin
    prechoice = resp_first
    jerk_lock_ms = 0
    initial_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
    # initial positions, speed and acc; final position, speed and acc
    init_trajs = []
    final_trajs = []
    total_traj = []
    for i_t in range(E.shape[1]):
        MT = (MT_slope*i_t + MT_intercep)  # pre-planned Motor Time
        first_resp_len = float((MT) *
                               np.abs(p_w_updt/(first_ev[i_t])))
        # first_resp_len: evidence affectation on MT. The higher the ev,
        # the lower the MT depending on the parameter p_w_updt
        initial_mu_side = initial_mu * prechoice[i_t]
        prior0 = compute_traj(jerk_lock_ms, mu=initial_mu_side,
                              resp_len=first_resp_len)
        init_trajs.append(prior0)
        # TRAJ. UPDATE
        vel_all = np.gradient(prior0)
        t_ind = int(second_ind[i_t] - first_ind[i_t])  # time index
        vel = vel_all[t_ind]  # velocity at the timepoint
        acc = np.gradient(vel_all)[t_ind]  # acceleration
        pos = prior0[t_ind]  # position
        mu_update = np.array([pos, vel, acc, 75*RLresp[i_t], 0, 0]).reshape(-1, 1)
        # new mu, considering new position/speed/acceleration
        second_response_len = float((first_resp_len-t_ind) *
                                    np.abs(p_w_updt/(second_ev[i_t])))
        # second_response_len: time left affected by the evidence on the
        # SECOND readout
        traj_fin = compute_traj(jerk_lock_ms, mu=mu_update,
                                resp_len=second_response_len)
        final_trajs.append(traj_fin)
        total_traj.append(np.concatenate(
            [prior0[0:second_ind[i_t]-first_ind[i_t]],
             traj_fin]))  # joined trajectories
    return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
        total_traj, init_trajs, final_trajs


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    num_tr = 100
    plot = False
    zt = np.random.randn(num_tr)*1e-2
    p_t_eff = 40
    p_t_a = p_t_eff
    num_timesteps = 1000
    stim = np.random.randn(num_tr)*1e-3+np.random.randn(num_timesteps+p_t_eff,
                                                        num_tr)*1e-1
    MT_slope = 0.15
    MT_intercep = 110
    p_t_m = 40
    p_w_zt = 0.2
    p_w_stim = 0.3
    p_e_noise = 0.2
    p_com_bound = 0.5
    Va = np.abs(np.random.randn(num_tr))*1e-1
    fluc_a = 0.5
    bound_a = 1
    p_w_a = 0.05
    p_a_noise = 0.05
    p_w_updt = 1
    E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re, total_traj,\
        init_trajs, final_trajs =\
        trial_ev_vectorized(zt=zt, stim=stim, MT_slope=MT_slope,
                            MT_intercep=MT_intercep,
                            p_w_zt=p_w_zt,
                            p_w_stim=p_w_stim, p_e_noise=p_e_noise,
                            p_com_bound=p_com_bound,
                            p_t_m=p_t_m, p_t_eff=p_t_eff, p_t_a=p_t_a,
                            num_tr=num_tr, p_w_a=p_w_a, p_a_noise=p_a_noise,
                            p_w_updt=p_w_updt, plot=False)
    plotting_trajs(init_trajs, total_traj, com, pro_vs_re)
    import sys
    sys.exit()
    plotting(E=E, com=com, second_ind=second_ind, first_ind=first_ind,
             resp_first=resp_first, resp_fin=resp_fin, pro_vs_re=pro_vs_re,
             p_t_eff=p_t_eff)
