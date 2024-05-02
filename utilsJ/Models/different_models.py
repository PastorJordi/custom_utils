# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 12:13:10 2023

@author: alexg
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
import os
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.pylab as pl
sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils/")  # Cluster Alex
# import utilsJ
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve,\
    com_heatmap_paper_marginal_pcom_side
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.Models import extended_ddm_v2 as edd2


# from simul import splitplot
# import os
# SV_FOLDER = '/archive/molano/CoMs/'  # Cluster Manuel
# SV_FOLDER = '/home/garciaduran/'  # Cluster Alex
# SV_FOLDER = '/home/molano/Dropbox/project_Barna/ChangesOfMind/'  # Manuel
SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
# DATA_FOLDER = '/archive/molano/CoMs/data/'  # Cluster Manuel
# DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
BINS = np.linspace(1, 301, 11)


def trial_ev_vectorized_without_1st_readout(zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
                                            p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
                                            p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
                                            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
                                            num_tr, stim_res, human=False,
                                            compute_trajectories=False, num_trials_per_session=600,
                                            all_trajs=True, num_computed_traj=int(3e4),
                                            fixation_ms=300, compute_mat_and_pcom=False):
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
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
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
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
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
    if human:
        px_final = 600
    if not human:
        px_final = 75
    # TODO: COMMENT EVERY FORKING LINE
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # adding leak
    # rolled_dW = np.roll(dW, 1)
    # rolled_dW[fixation + p_t_aff, :] = 0
    # dW += -rolled_dW*p_leak
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
    # E = np.cumsum(dW, axis=0)
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
    matrix = None
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
        initial_mu = np.array([0, 0, 0, px_final, 0, 0]).reshape(-1, 1)
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
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-0*np.abs(first_ev[i_t]))  # 
            # first_resp_len: evidence influence on MT. Since we don't have
            # 1st read-out, the impact is null (0)
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                       resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                # and avoids having negative second_resp_len (impossibly fast
                # responses) bc of very strong confirmation evidence.
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence at second readout and the signed p_com_bound
                com_bound_signed = (-sign_)*p_com_bound
                difference = (updt_ev-first_ev[i_t])*sign_
                # the different in this case does not consider first_ev
                # offset = 120
                second_response_len =\
                    float(remaining_m_time -  # offset*com[i_t] -
                          p_2nd_readout*(difference))
                # SECOND readout
                traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                             resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
                # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
                # np.random.randn(np.random.randint(15, 60))*0.2
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        if compute_mat_and_pcom:
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
        else:
            xpos_plot = None
            median_pcom = None
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, None,\
            rt_vals, rt_bins, None


def trial_ev_vectorized_without_1st_readout_random_1st_choice(zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
                p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
                p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
                p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
                num_tr, stim_res, human=False,
                compute_trajectories=False, num_trials_per_session=600,
                all_trajs=True, num_computed_traj=int(3e4),
                fixation_ms=300, compute_mat_and_pcom=False):
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
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
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
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
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
    if human:
        px_final = 600
    if not human:
        px_final = 75
    # TODO: COMMENT EVERY FORKING LINE
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # adding leak
    # rolled_dW = np.roll(dW, 1)
    # rolled_dW[fixation + p_t_aff, :] = 0
    # dW += -rolled_dW*p_leak
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
    # E = np.cumsum(dW, axis=0)
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
        resp_first[i_t] = np.random.choice([-1, 1])
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
        indx_com = ()
        indx_update_ch = indx_final_ch if len(indx_com) == 0\
            else indx_com[0] + hit_dec
        # get final decision
        resp_fin[i_t] = resp_first[i_t] if len(indx_com) == 0 else -resp_first[i_t]
        resp_fin[i_t] = np.sign(E[indx_final_ch, i_t])
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_t])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind)
    pro_vs_re = np.array(pro_vs_re)
    rt_vals, rt_bins = np.histogram((first_ind-fixation+p_t_eff)*stim_res,
                                    bins=np.linspace(-100, 300, 81))
    matrix = None
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
        initial_mu = np.array([0, 0, 0, px_final, 0, 0]).reshape(-1, 1)
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
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-0*np.abs(first_ev[i_t]))  # 
            # first_resp_len: evidence influence on MT. Since we don't have
            # 1st read-out, the impact is null (0)
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                       resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                # and avoids having negative second_resp_len (impossibly fast
                # responses) bc of very strong confirmation evidence.
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence at second readout and the signed p_com_bound
                com_bound_signed = (-sign_)*p_com_bound
                difference = np.abs(updt_ev)
                # the different in this case does not consider first_ev
                # offset = 120
                second_response_len =\
                    float(remaining_m_time -  # offset*com[i_t] -
                          p_2nd_readout*(difference))
                # SECOND readout
                traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                             resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
                # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
                # np.random.randn(np.random.randint(15, 60))*0.2
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        if compute_mat_and_pcom:
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
        else:
            xpos_plot = None
            median_pcom = None
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, None,\
            rt_vals, rt_bins, None



def trial_ev_vectorized_without_2nd_readout(zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
                                            p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
                                            p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
                                            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
                                            num_tr, stim_res, human=False,
                                            compute_trajectories=False, num_trials_per_session=600,
                                            all_trajs=True, num_computed_traj=int(3e4),
                                            fixation_ms=300, compute_mat_and_pcom=False):
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
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
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
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
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
    if human:
        px_final = 600
    if not human:
        px_final = 75
    # TODO: COMMENT EVERY FORKING LINE
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # adding leak
    # rolled_dW = np.roll(dW, 1)
    # rolled_dW[fixation + p_t_aff, :] = 0
    # dW += -rolled_dW*p_leak
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
    # E = np.cumsum(dW, axis=0)
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
        indx_com = ()
        # no coms
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
    matrix = None
    # end_eddm = time.time()
    # print('Time for "PSIAM": ' + str(end_eddm - start_eddm))
    # XXX: put in a different function
    if compute_trajectories:
        # start_traj = time.time()
        # Trajectories
        # print('Starting with trajectories')
        RLresp = resp_first  # so that the second read-out does not affect choice
        prechoice = resp_first
        jerk_lock_ms = 0
        # initial positions, speed and acc; final position, speed and acc
        initial_mu = np.array([0, 0, 0, px_final, 0, 0]).reshape(-1, 1)
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
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-p_1st_readout*np.abs(first_ev[i_t]))
            # first_resp_len: evidence influence on MT. The larger the ev,
            # the smaller the motor time
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                       resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                # and avoids having negative second_resp_len (impossibly fast
                # responses) bc of very strong confirmation evidence.
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence at second readout and the signed p_com_bound
                com_bound_signed = (-sign_)*p_com_bound
                difference = (updt_ev - first_ev[i_t])*sign_
                # offset = 120
                second_response_len =\
                    float(remaining_m_time -  # offset*com[i_t] -
                          0*(difference))
                # impact of 2nd evidence in MT is null (0)
                # SECOND readout
                traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                             resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
                # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
                # np.random.randn(np.random.randint(15, 60))*0.2
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        if compute_mat_and_pcom:
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
        else:
            xpos_plot = None
            median_pcom = None
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, None,\
            rt_vals, rt_bins, None


def trial_ev_vectorized_neg_starting_point(zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
                                           p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
                                           p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
                                           p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
                                           num_tr, stim_res, human=False,
                                           compute_trajectories=False, num_trials_per_session=600,
                                           all_trajs=True, num_computed_traj=int(3e4),
                                           fixation_ms=300, compute_mat_and_pcom=False):
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
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
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
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
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
    if human:
        px_final = 600
    if not human:
        px_final = 75
    # TODO: COMMENT EVERY FORKING LINE
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # adding leak
    # rolled_dW = np.roll(dW, 1)
    # rolled_dW[fixation + p_t_aff, :] = 0
    # dW += -rolled_dW*p_leak
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
    # E = np.cumsum(dW, axis=0)
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

    matrix = None
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
        initial_mu = np.array([-5, 0, 0, px_final, 0, 0]).reshape(-1, 1)
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
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-p_1st_readout*np.abs(first_ev[i_t]))
            # first_resp_len: evidence influence on MT. The larger the ev,
            # the smaller the motor time
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                       resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                # and avoids having negative second_resp_len (impossibly fast
                # responses) bc of very strong confirmation evidence.
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence at second readout and the signed p_com_bound
                com_bound_signed = (-sign_)*p_com_bound
                difference = (updt_ev - first_ev[i_t])*sign_
                # offset = 120
                second_response_len =\
                    float(remaining_m_time -  # offset*com[i_t] -
                          p_2nd_readout*(difference))
                # SECOND readout
                traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                             resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
                # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
                # np.random.randn(np.random.randint(15, 60))*0.2
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        if compute_mat_and_pcom:
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
        else:
            xpos_plot = None
            median_pcom = None
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs


def trial_ev_vectorized_CoM_without_update(
        zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
        p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
        p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
        p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
        num_tr, stim_res, human=False,
        compute_trajectories=False, num_trials_per_session=600,
        all_trajs=True, num_computed_traj=int(3e4),
        fixation_ms=300, compute_mat_and_pcom=False):
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
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
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
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
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
    if human:
        px_final = 600
    if not human:
        px_final = 75
    # TODO: COMMENT EVERY FORKING LINE
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # adding leak
    # rolled_dW = np.roll(dW, 1)
    # rolled_dW[fixation + p_t_aff, :] = 0
    # dW += -rolled_dW*p_leak
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
    # E = np.cumsum(dW, axis=0)
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
    matrix = None
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
        initial_mu = np.array([0, 0, 0, px_final, 0, 0]).reshape(-1, 1)
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
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-p_1st_readout*np.abs(first_ev[i_t]))
            # first_resp_len: evidence influence on MT. The larger the ev,
            # the smaller the motor time
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                       resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                # and avoids having negative second_resp_len (impossibly fast
                # responses) bc of very strong confirmation evidence.
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence at second readout and the signed p_com_bound
                com_bound_signed = (-sign_)*p_com_bound
                difference = (updt_ev - first_ev[i_t])*sign_
                # offset = 120
                if com[i_t]:
                    second_response_len =\
                        float(remaining_m_time -  # offset*com[i_t] -
                              p_2nd_readout*(difference))
                else:
                    second_response_len =\
                        float(remaining_m_time)
                # SECOND readout
                traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                             resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
                # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
                # np.random.randn(np.random.randint(15, 60))*0.2
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        if compute_mat_and_pcom:
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
        else:
            xpos_plot = None
            median_pcom = None
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, None,\
            rt_vals, rt_bins, None


def trial_ev_vectorized_only_prior_1st_choice(
        zt, stim, coh, trial_index, p_MT_slope, p_MT_intercept, p_w_zt,
        p_w_stim, p_e_bound, p_com_bound, p_t_eff, p_t_aff,
        p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
        p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
        num_tr, stim_res, human=False,
        compute_trajectories=False, num_trials_per_session=600,
        all_trajs=True, num_computed_traj=int(3e4),
        fixation_ms=300, compute_mat_and_pcom=False):
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
    p_e_bound : float
        fitting parameter: bounds for the evidence integrator.
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
    p_a_bound : float
        fitting parameter: bounds for the action integrator.
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
    if human:
        px_final = 600
    if not human:
        px_final = 75
    # TODO: COMMENT EVERY FORKING LINE
    bound = p_e_bound
    bound_a = p_a_bound
    dt = stim_res*1e-3
    p_e_noise = np.sqrt(dt)
    p_a_noise = np.sqrt(dt)
    fixation = int(fixation_ms / stim_res)  # ms/stim_resolution
    prior = zt*p_w_zt + np.random.randn(len(zt))*1e-5
    # instantaneous evidence
    Ve = np.concatenate((np.zeros((p_t_aff + fixation, num_tr)), stim*p_w_stim))
    max_integration_time = Ve.shape[0]-1
    N = Ve.shape[0]
    zero_or_noise_evidence = np.concatenate((np.zeros((fixation, num_tr)),
                                             np.random.randn(N - fixation, num_tr)))
    # add noise
    dW = zero_or_noise_evidence*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+p_w_a_intercept +\
        p_w_a_slope*trial_index
    # zeros before p_t_a
    dA[:p_t_a, :] = 0
    # accumulate
    A = np.cumsum(dA, axis=0)
    dW[0, :] = prior
    E = np.copy(dW)
    # adding leak
    E[:fixation, :] = np.cumsum(E[:fixation, :], axis=0)
    for i in range(fixation, N):
        E[i, :] += E[i-1, :]*(1-p_leak)
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
        first_ev.append(prior[i_t])  # prior[i_t]
        # first categorical response
        resp_first[i_t] = np.sign(prior[i_t])
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
            np.where(np.sign(prior[i_t]) != np.sign(post_dec_integration))[0]
        # get CoM effective index
        indx_update_ch = indx_final_ch if len(indx_com) == 0\
            else indx_com[0] + hit_dec
        # get final decision
        resp_fin[i_t] = resp_first[i_t] if len(indx_com) == 0 else -resp_first[i_t]
        # resp_fin[i_t] = np.sign(E[indx_final_ch, i_t])
        second_ind.append(indx_update_ch)
        second_ev.append(E[indx_update_ch, i_t])
    com = resp_first != resp_fin
    first_ind = np.array(first_ind)
    pro_vs_re = np.array(pro_vs_re)
    rt_vals, rt_bins = np.histogram((first_ind-fixation+p_t_eff)*stim_res,
                                    bins=np.linspace(-100, 300, 81))
    matrix = None
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
        initial_mu = np.array([0, 0, 0, px_final, 0, 0]).reshape(-1, 1)
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
            MT = p_MT_slope*trial_index[i_t] + p_MT_intercept +\
                p_mt_noise*np.random.gumbel()
            first_resp_len = float(MT-p_1st_readout*np.abs(first_ev[i_t]))
            # first_resp_len: evidence influence on MT. The larger the ev,
            # the smaller the motor time
            initial_mu_side = initial_mu * prechoice[i_t]
            prior0 = edd2.compute_traj(jerk_lock_ms, mu=initial_mu_side,
                                       resp_len=first_resp_len)
            init_trajs.append(prior0)  # + np.random.randn(len(prior0))*0.15)
            # TRAJ. UPDATE
            try:
                velocities = np.gradient(prior0)
                accelerations = np.gradient(velocities)  # acceleration
                t_updt = int(p_t_eff+second_ind[i_t] - first_ind[i_t])  # time indx
                t_updt = int(np.min((t_updt*stim_res, len(velocities)-1)))
                frst_traj_motor_time.append(t_updt)
                vel = velocities[t_updt]  # velocity at the timepoint
                acc = accelerations[t_updt]
                pos = prior0[t_updt]  # position
                mu_update = np.array([pos, vel, acc, px_final*RLresp[i_t],
                                      0, 0]).reshape(-1, 1)
                # new mu, considering new position/speed/acceleration
                remaining_m_time = first_resp_len-t_updt
                sign_ = resp_first[i_t]
                # this sets the maximum updating evidence equal to the ev bound
                # and avoids having negative second_resp_len (impossibly fast
                # responses) bc of very strong confirmation evidence.
                updt_ev = np.clip(second_ev[i_t], a_min=-bound, a_max=bound)
                # second_response_len: motor time update influenced by difference
                # between the evidence at second readout and the signed p_com_bound
                com_bound_signed = (-sign_)*p_com_bound
                difference = (updt_ev - first_ev[i_t])*sign_
                # offset = 120
                second_response_len =\
                    float(remaining_m_time -  # offset*com[i_t] -
                          p_2nd_readout*(difference))
                # SECOND readout
                traj_fin = edd2.compute_traj(jerk_lock_ms, mu=mu_update,
                                             resp_len=second_response_len)
                # joined trajectories
                traj_before_uptd = prior0[0:t_updt]
                traj_updt = np.concatenate((traj_before_uptd,  traj_fin))
                # traj_updt += np.random.randn(len(traj_updt))*0.15  # noise
                # traj_updt = np.concatenate((np.repeat(0, 30), traj_updt))
                # np.random.randn(np.random.randint(15, 60))*0.2
                if com[i_t]:
                    opp_side_values = traj_updt.copy()
                    opp_side_values[np.sign(traj_updt) == resp_fin[i_t]] = 0
                    max_val_towards_opposite = np.max(np.abs(opp_side_values))
                    x_val_at_updt.append(max_val_towards_opposite)
                else:
                    x_val_at_updt.append(0)
            except Exception:
                traj_fin = [np.nan]
                traj_updt = np.concatenate((prior0, traj_fin))
                x_val_at_updt.append(0)
            total_traj.append(traj_updt)
            final_trajs.append(traj_fin)
        if compute_mat_and_pcom:
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
        else:
            xpos_plot = None
            median_pcom = None
        # print('Time for trajectories: ' + str(end_traj - start_traj))
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, total_traj, init_trajs, final_trajs, frst_traj_motor_time,\
            x_val_at_updt, xpos_plot, median_pcom,\
            rt_vals, rt_bins, indx_trajs
    else:
        return E, A, com, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
            matrix, None, None, None, None, None, None, None,\
            rt_vals, rt_bins, None