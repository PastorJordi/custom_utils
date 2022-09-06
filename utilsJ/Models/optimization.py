# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:29:19 2022

@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
import sys
from cmaes import CMA
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")
sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
import utilsJ
from utilsJ.Models.extended_ddm import trial_ev_vectorized, data_augmentation

# DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/opt_results/'  # Alex
SV_FOLDER = '/home/garciaduran/opt_results/'  # Cluster Alex


def get_data(dfpath=DATA_FOLDER, after_correct=True, num_tr_per_rat=int(1e3),
             all_trials=False):
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
        if all_trials:
            if after_correct:
                indx = np.where(df['aftererror'].values == 0)[0]
            else:
                indx = np.linspace(len(df['aftererror']), dtype=int)
        else:
            if after_correct:
                indx_prev_error = np.where(df['aftererror'].values == 0)[0]
                selected_indx = np.random.choice(np.arange(len(indx_prev_error)),
                                                 size=(num_tr_per_rat),
                                                 replace=False)
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
    print('Ended loading data')
    stim = stim.T
    zt = prior
    return stim, zt, coh, gt, com


def run_likelihood(stim, zt, coh, gt, com, p_w_zt, p_w_stim, p_e_noise,
                   p_com_bound, p_t_aff, p_t_eff, p_t_a, p_w_a, p_a_noise,
                   p_w_updt, num_times_tr=int(1e3), detect_CoMs_th=5):
    num_tr = stim.shape[1]
    # num_tr = 5
    stim = stim[:, :int(num_tr)]
    zt = zt[:int(num_tr)]
    coh = coh[:int(num_tr)]
    com = com[:int(num_tr)]
    gt = gt[:int(num_tr)]
    MT_slope = 0.15
    MT_intercep = 110
    compute_trajectories = True
    all_trajs = True
    data_augment_factor = 10
    stim = data_augmentation(stim=stim, daf=data_augment_factor)
    stim_res = 50/data_augment_factor
    stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                stim.shape[1]))))
    detected_com_mat = np.zeros((num_tr, num_times_tr))
    for i in range(num_times_tr):  # TODO: parallelize loop for cluster
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
        detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
        detected_com_mat[:, i] = detected_com
    prob_detected_com = np.mean(detected_com_mat, axis=1)
    llk = []
    for i_com, com_bool in enumerate(com):
        if com_bool:
            llk.append(prob_detected_com[i_com] + 1e-5)
        else:
            llk.append(1-prob_detected_com[i_com] + 1e-5)
    llk = np.array(llk)
    llk_val = -np.sum(np.log(llk))
    # print(llk_val)
    return llk_val


# --- MAIN
if __name__ == '__main__':
    optimization = True
    stim, zt, coh, gt, com = get_data(dfpath=DATA_FOLDER, after_correct=True,
                                      num_tr_per_rat=int(1e4), all_trials=False)
    # p_t_aff = 10
    # p_t_eff = 6
    # p_t_a = 35
    # p_w_zt = 0.15
    # p_w_stim = 0.15
    # p_e_noise = 0.05
    # p_com_bound = 0.
    # p_w_a = 0.03
    # p_a_noise = 0.06
    # p_w_updt = 0.1
    array_params = np.array((10, 6, 35, 0.15, 0.15, 0.05, 0.3, 0.03, 0.06, 0.1))
    scaled_params = np.repeat(0.5, len(array_params))
    scaled_params[:2] *= 10
    scaling_value = array_params/scaled_params
    bounds = np.array(((5, 20), (3, 10), (20, 60), (0.05, 0.3), (0.05, 0.3),
                      (0.005, 0.1), (0., 1), (0.005, 0.1), (0.05, 0.1),
                      (0.05, 60)))
    bounds_scaled = np.array([bound / scaling_value[i_b] for i_b, bound in
                              enumerate(bounds)])
    # llk_val = run_likelihood(stim, zt, coh, gt, com, p_w_zt, p_w_stim, p_e_noise,
    #                          p_com_bound, p_t_aff, p_t_eff, p_t_a, p_w_a,
    #                          p_a_noise, p_w_updt, num_times_tr=int(1e1),
    #                          detect_CoMs_th=5)
    if optimization:
        optimizer = CMA(mean=scaled_params, sigma=0.1, bounds=bounds_scaled)
        all_solutions = []
        for gen in range(50):
            solutions = []
            for _ in range(optimizer.population_size):
                params_init = optimizer.ask()
                params = params_init * scaling_value
                p_t_aff = int(params[0])
                p_t_eff = int(params[1])
                p_t_a = int(params[2])
                p_w_zt = params[3]
                p_w_stim = params[4]
                p_e_noise = params[5]
                p_com_bound = params[6]
                p_w_a = params[7]
                p_a_noise = params[8]
                p_w_updt = params[9]
                llk_val = run_likelihood(stim, zt, coh, gt, com, p_w_zt, p_w_stim,
                                         p_e_noise, p_com_bound, p_t_aff, p_t_eff,
                                         p_t_a, p_w_a, p_a_noise, p_w_updt,
                                         num_times_tr=int(1e2),
                                         detect_CoMs_th=5)
                solutions.append((params_init, llk_val))
            all_solutions.append(solutions)
            if optimizer.should_stop():
                break
            optimizer.tell(solutions)
        np.savez(SV_FOLDER+'last_solution.npz', solutions)
        np.savez(SV_FOLDER+'all_solutions.npz', solutions)
