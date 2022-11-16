#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:54:38 2022

@author: manuel
"""

import numpy as np
from utilsJ.Models import extended_ddm_v2 as edd2
import pandas as pd
import matplotlib.pyplot as plt


def trial_ev(zt, stim, trial_index, MT_slope, MT_intercep, p_w_zt,
             p_w_stim, p_e_noise, p_com_bound, p_t_eff, p_t_aff,
             p_t_a, p_w_a_intercept, p_w_a_slope, p_a_noise,
             p_1st_readout, p_2nd_readout, num_tr, stim_res,
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
    bound = 1
    bound_a = 2.2
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
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.cumsum(dW, axis=0)
    com = False
    i_t = 0
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
    pro_vs_re = np.argmin([hit_action, hit_bound])
    # store first readout index
    first_ind = hit_dec
    # store first readout evidence
    first_ev = E[hit_dec, i_t]
    # first categorical response
    resp_first = (-1)**(E[hit_dec, i_t] < 0)
    # CoM bound with sign depending on first response
    com_bound_signed = (-resp_first)*p_com_bound
    # second response
    indx_final_ch = hit_dec+p_t_eff+p_t_aff
    indx_final_ch = min(indx_final_ch, max_integration_time)
    # get post decision accumulated evidence with respect to CoM bound
    post_dec_integration = E[hit_dec:indx_final_ch, i_t]-com_bound_signed
    # get CoMs indexes
    indx_com =\
        np.where(np.sign(E[hit_dec, i_t]) != np.sign(post_dec_integration))[0]
    # get CoM effective index
    indx_update_ch = indx_final_ch if len(indx_com) == 0\
        else indx_com[0] + hit_dec
    # get final decision
    resp_fin = resp_first if len(indx_com) == 0 else -resp_first
    second_ind = indx_update_ch
    second_ev = E[indx_update_ch, i_t]
    com = resp_first != resp_fin
    first_ind = np.array(first_ind)
    pro_vs_re = np.array(pro_vs_re)
    rt_vals, rt_bins = np.histogram((first_ind-fixation+p_t_eff)*stim_res,
                                    bins=np.linspace(-100, 300, 81))
    # Trajectory
    # print('Starting with trajectories')
    RLresp = resp_fin
    prechoice = resp_first
    jerk_lock_ms = 0
    # initial positions, speed and acc; final position, speed and acc
    initial_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
    # pre-planned Motor Time, the modulo prevents trial-index from
    # growing indefinitely
    MT = MT_slope*trial_index[i_t] + MT_intercep
    first_resp_len = float(MT-p_1st_readout*np.abs(first_ev))
    # first_resp_len: evidence influence on MT. The larger the ev,
    # the smaller the motor time
    initial_mu_side = initial_mu * prechoice
    prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                               resp_len=first_resp_len)
    init_traj = prior0
    # TRAJ. UPDATE
    velocities = np.gradient(prior0)
    accelerations = np.gradient(velocities)  # acceleration
    t_updt = int(p_t_eff+second_ind - first_ind)  # time indx
    t_updt = int(np.min((t_updt, len(velocities)-1)))
    frst_traj_motor_time = t_updt
    vel = velocities[t_updt]  # velocity at the timepoint
    acc = accelerations[t_updt]
    pos = prior0[t_updt]  # position
    mu_update = np.array([pos, vel, acc, 75*RLresp,
                          0, 0]).reshape(-1, 1)
    # new mu, considering new position/speed/acceleration
    remaining_m_time = first_resp_len-t_updt
    sign_ = resp_first
    # this sets the maximum updating evidence equal to the ev bound
    # and avoids having negative second_resp_len (impossibly fast
    # responses) bc of very strong confirmation evidence.
    updt_ev = np.clip(second_ev, a_min=-1, a_max=1)
    # second_response_len: motor time update influenced by difference
    # between the evidence at second readout and the signed p_com_bound
    com_bound_signed = (-sign_)*p_com_bound
    offset = 140
    second_response_len =\
        float(remaining_m_time + offset -
              p_2nd_readout*(np.abs(updt_ev - com_bound_signed)))
    #           float(remaining_m_time +
    # p_2nd_readout*np.abs(1 - np.abs(updt_ev) - com_bound_signed))
    # SECOND readout
    traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                 resp_len=second_response_len)
    final_traj = traj_fin
    # joined trajectories
    traj_before_uptd = prior0[0:t_updt]
    traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
    total_traj = traj_updt
    if com:
        opp_side_values = traj_updt.copy()
        opp_side_values[np.sign(traj_updt) == resp_fin] = 0
        max_val_towards_opposite = np.max(np.abs(opp_side_values))
        x_val_at_updt = max_val_towards_opposite
    else:
        x_val_at_updt = 0
    detect_CoMs_th = 5
    detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
    df_curve = {'detected_CoM': [detected_com],
                'sound_len': [(first_ind-fixation+p_t_eff)*stim_res]}
    df_curve = pd.DataFrame(df_curve)
    xpos = int(np.diff(edd2.BINS)[0])
    xpos_plot, median_pcom, _ =\
        edd2.binned_curve(df_curve, 'detected_CoM', 'sound_len', xpos=xpos,
                     bins=edd2.BINS,
                     return_data=True)
    # end_traj = time.time()
    # print('Time for trajectories: ' + str(end_traj - start_traj))
    return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
        total_traj, init_traj, final_traj, frst_traj_motor_time,\
        x_val_at_updt, xpos_plot, median_pcom, rt_vals, rt_bins, first_resp_len


def plotting(com, E, A, second_ind, first_ind, resp_first, resp_fin, pro_vs_re,
             p_t_aff, init_traj, total_traj, p_t_eff, frst_traj_motor_time,
             p_com_bound, stim_res=50, trial=0, l=''):
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 12), sharex=True)
    ax = ax.flatten()
    ax[1].set_xlabel('Time (ms)')
    max_xlim = 0
    traj_in = False
    edd2.draw_lines(ax, frst=first_ind*stim_res, sec=second_ind*stim_res,
                    p_t_aff=p_t_aff*stim_res, p_com_bound=p_com_bound)
    color1 = 'green' if resp_first < 0 else 'purple'
    color2 = 'green' if resp_fin < 0 else 'purple'

    x_2 = np.arange(second_ind+1)*stim_res
    x_1 = np.arange(first_ind+1)*stim_res
    ax[0].plot(x_2, E[:second_ind+1, trial], color=color2, alpha=0.7)
    ax[0].plot(x_1, E[:first_ind+1, trial], color=color1, lw=2)
    ax[0].set_ylabel(l+' EA')
    # trajectories
    sec_ev = round(E[second_ind, trial], 2)
    # updt_motor = first_ind+frst_traj_motor_time
    init_motor = first_ind+p_t_eff
    xs = init_motor*stim_res+np.arange(0, len(total_traj))
    max_xlim = max(max_xlim, np.max(xs))
    ax[1].plot([0, xs[0]], [0, 0], color='k')
    ax[1].plot(xs, total_traj, color='r',
                  label='Updated traj., E:{}'.format(sec_ev))
    first_ev = round(E[first_ind, trial], 2)
    xs = init_motor*stim_res+np.arange(0, len(init_traj))
    max_xlim = max(max_xlim, np.max(xs))
    ax[1].plot(xs, init_traj, color='k',
                  label='Initial traj. E:{}'.format(first_ev))
    ax[1].set_ylabel(l+', y(px)')
    ax[1].set_ylabel(l+', y(px)')
    print(len(init_traj))
    ax[1].legend()
    # for a in ax:
    #     a.set_xlim([first_ind*stim_res-50, second_ind*stim_res+50])
    f.savefig('/home/manuel/Descargas/example_trials.png', dpi=400,
              bbox_inches='tight')


# --- MAIN
if __name__ == "__main__":
    data_augment_factor = 10
    stim = np.random.randn(20, 1)+1
    stim = edd2.data_augmentation(stim=stim, daf=data_augment_factor)
    stim_res = 50/data_augment_factor

    zt = np.array([-2])
    trial_index = np.array([200])
    p_t_aff = 8
    p_t_eff = 5
    p_t_a = 14  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.1
    p_w_stim = 0.05
    p_e_noise = 0.02
    p_com_bound = 0.
    p_w_a_intercept = 0.05
    p_w_a_slope = -2e-05  # fixed
    p_a_noise = 0.04  # fixed
    p_1st_readout = 140
    p_2nd_readout = 100
    compute_trajectories = True
    plot = True
    all_trajs = True
    MT_slope = 0.123
    MT_intercep = 254
    stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                stim.shape[1]))))
    num_tr = 1
    
    E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
        total_traj, init_traj, final_traj, frst_traj_motor_time,\
        x_val_at_updt, xpos_plot, median_pcom, rt_vals, rt_bins, first_resp_len =\
        trial_ev(zt=zt, stim=stim_temp, trial_index=trial_index,
                 MT_slope=MT_slope, MT_intercep=MT_intercep, p_w_zt=p_w_zt,
                 p_w_stim=p_w_stim, p_e_noise=p_e_noise, p_com_bound=p_com_bound,
                 p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a, num_tr=num_tr,
                 p_w_a_intercept=p_w_a_intercept, p_w_a_slope=p_w_a_slope,
                 p_a_noise=p_a_noise, p_1st_readout=p_1st_readout,
                 p_2nd_readout=p_2nd_readout, stim_res=stim_res)
    
    plotting(com=com, E=E, A=A, second_ind=second_ind, first_ind=first_ind,
             resp_first=resp_first, resp_fin=resp_fin, pro_vs_re=pro_vs_re,
             p_t_aff=p_t_aff, init_traj=init_traj, total_traj=total_traj,
             p_t_eff=p_t_eff, frst_traj_motor_time=frst_traj_motor_time,
             p_com_bound=p_com_bound, stim_res=stim_res)