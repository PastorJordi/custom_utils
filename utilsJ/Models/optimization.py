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
from skimage.metrics import structural_similarity as ssim
import dirichlet
import seaborn as sns
from sbi.inference import MNLE
from sbi.utils import MultipleIndependent
import torch
from torch.distributions import Beta, Binomial, Gamma, Uniform
from sbi.analysis import pairplot
from pybads import BADS

# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
from utilsJ.Models.extended_ddm_v2 import trial_ev_vectorized,\
    data_augmentation, get_data_and_matrix, com_detection, get_trajs_time
from utilsJ.Behavior.plotting import binned_curve
import utilsJ.Models.dirichletMultinomialEstimation as dme

# DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
# DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM

# SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Results_LE43/'  # Alex
# SV_FOLDER = '/home/garciaduran/opt_results/'  # Cluster Alex
SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/opt_results/'  # Jordi
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM

BINS = np.arange(1, 320, 20)


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
    pright = np.empty((0, ))
    sound_len = np.empty((0, ))
    trial_index = np.empty((0, ))
    for f in files:
        start_1 = time.time()
        df = pd.read_pickle(f)
        df = df.query(
                "sound_len <= 400 and soundrfail ==\
                    False and resp_len <=1 and R_response>= 0\
                        and hithistory >= 0 and special_trial == 0")
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
        pright_tmp = np.array(df.R_response)
        trial_index_tmp = np.array(df.origidx)
        prior = np.concatenate((prior, prior_tmp[indx]))
        stim = np.concatenate((stim, stim_tmp[indx, :]))
        coh = np.concatenate((coh, coh_mat[indx]))
        com = np.concatenate((com, com_tmp[indx]))
        gt = np.concatenate((gt, gt_tmp[indx]))
        pright = np.concatenate((pright, pright_tmp[indx]))
        sound_len = np.concatenate((sound_len, sound_len_tmp[indx]))
        trial_index = np.concatenate((trial_index, trial_index_tmp[indx]))
        end = time.time()
        print(f)
        print(end - start_1)
        print(len(df))
    print(end - start)
    print('Ended loading data')
    stim = stim.T
    zt = prior
    return stim, zt, coh, gt, com, pright, trial_index


def fitting(res_path='C:/Users/Alexandre/Desktop/CRM/Results_LE38/',
            results=False,
            detected_com=None, first_ind=None, p_t_eff=None,
            data_path='C:/Users/Alexandre/Desktop/CRM/results_simul/',
            metrics='mse', objective='curve', bin_size=30, det_th=8,
            plot=False, stim_res=5):
    data_mat = np.load(data_path + 'CoM_vs_prior_and_stim.npy')
    data_mat_norm = data_mat / np.nanmax(data_mat)
    data_curve = pd.read_csv(data_path + 'pcom_vs_rt.csv')
    tmp_data = data_curve['tmp_bin']
    data_curve_norm = data_curve['pcom'] / np.max(data_curve['pcom'])
    nan_penalty = 0.2
    w_rms = 0.5
    if results:
        files = glob.glob(res_path+'*.npz')
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
                if objective == 'curve':
                    rt_vals_pcom = data.get('xpos_rt_pcom')
                    rt_vals_pcom = [rt.astype(int) for
                                    rt in rt_vals_pcom]
                    median_vals_pcom = data.get('median_pcom_rt')
                    x_val_at_updt = data.get('x_val_at_updt_mat')
                    perc_list = []
                    for i_pcom, med_pcom in enumerate(median_vals_pcom):
                        curve_total.append(med_pcom)
                        rt_vals.append(rt_vals_pcom[i_pcom])
                        x_val.append(x_val_at_updt[i_pcom])
                        file_index.append(i_f)
                if objective == 'RT':
                    rt_vals = data.get('rt_vals_all')
                    rt_bins = data.get('rt_bins_all')
                    file_index = np.repeat(0, len(rt_vals))
        if objective == 'RT':
            data_rt_dist = np.load(res_path + 'RT_distribution.npy')
            data_rt_dist_norm = data_rt_dist/data_rt_dist.sum()
            # data_rt_bins = np.load(res_path + 'RT_bins.npy')
            ind_rms = rt_bins[0][:-1] >= 0  # non-considering FB
            for val in range(len(rt_vals)):
                vals_norm = rt_vals[val, :] / rt_vals[val, :].sum()
                diff_rms = np.subtract(data_rt_dist_norm[ind_rms],
                                       vals_norm[ind_rms])**2
                diff_rms_mat.append(np.sqrt(np.nansum(diff_rms)))
                curve_total.append(vals_norm)
        if objective == 'curve':
            for curve_ind, _ in enumerate(rt_vals):
                x_perc = np.nanmean(np.abs(x_val[curve_ind]) > det_th)
                x_perc = 1
                perc_list.append(x_perc)
                tmp_simul =\
                    np.array((rt_vals[curve_ind])/(bin_size-1),
                             dtype=int)
                if len(rt_vals[curve_ind]) == 0 or np.isnan(x_perc):
                    diff_mn.append(1e3)
                    diff_norm_mat.append(1e3)
                    diff_rms_mat.append(1e3)
                else:
                    curve_tmp = curve_total[curve_ind]*x_perc + 1e-6
                    curve_norm = curve_tmp / np.nanmax(curve_tmp)
                    # diff_norm = np.subtract(curve_norm,
                    #                         np.array(data_curve_norm[
                    #                             tmp_simul])) ** 2
                    diff_norm = np.corrcoef(curve_norm,
                                            data_curve_norm[tmp_simul].values)
                    diff_norm = diff_norm[0, 1] if not np.isnan(
                        diff_norm[0, 1]) else -1
                    num_nans = len(tmp_data) - len(tmp_simul)
                    # diff_norm_mat.append(1-diff_norm+nan_penalty*num_nans)
                    diff_norm_mat.append(1 - diff_norm + nan_penalty*num_nans)
                    window = np.exp(-np.arange(len(tmp_simul))**1/10)
                    # window = 1
                    diff_rms = np.subtract(curve_tmp,
                                           np.array(data_curve['pcom']
                                                    [tmp_simul]) *
                                           window) ** 2
                    diff_rms_mat.append(np.sqrt(np.nansum(diff_rms)) +
                                        num_nans * nan_penalty)
                    diff = (1 - w_rms)*(1 - diff_norm) + w_rms*np.sqrt(np.nansum(
                        diff_rms)) + num_nans * nan_penalty
                    diff_mn.append(diff) if not np.isnan(diff) else\
                        diff_mn.append(1e3)
                    max_ssim = False
        if plot:
            plt.figure()
            plt.plot(rt_vals[np.argmin(diff_rms_mat)],
                     curve_total[np.argmin(diff_rms_mat)],
                     label='min rms')
            # plt.plot(data_curve_norm, label='norm data')
            if objective == 'curve':
                plt.plot(rt_vals[np.argmin(diff_norm_mat)],
                         curve_total[np.argmin(diff_norm_mat)],
                         label='min norm')
                plt.plot(rt_vals[np.argmin(diff_mn)],
                         curve_total[np.argmin(diff_mn)],
                         label='min joined')
                plt.plot(data_curve['rt'], data_curve['pcom'], label='data')
                plt.ylabel('pCoM')
            if objective == 'RT':
                plt.plot(data_rt_dist_norm, label='data')
                plt.ylabel('Density')
            plt.legend()
        if max_ssim:
            ind_min = np.argmax(diff_mn)
        else:
            ind_sorted = np.argsort(np.abs(diff_rms_mat))
            ind_min = ind_sorted[0]
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
        if plot:
            # For the best 10 configurations:
            plt.figure()
            if objective == 'curve':
                plt.plot(data_curve['rt'], data_curve['pcom'], label='data',
                         linestyle='', marker='o')
            if objective == 'RT':
                action_slope = []
                action_intercept = []
                zt_weight = []
                stim_weight = []
                aff_eff_sum = []
                action_t_a = []
                plt.plot(rt_bins[0][:-1], data_rt_dist_norm, label='data',
                         linewidth=1.8)
            for i in range(10):
                ind_min = ind_sorted[i]
                optimal_params = {}
                file_index = np.array(file_index)
                min_num = np.where(file_index == file_index[ind_min])[0][0]
                with np.load(files[file_index[ind_min]], allow_pickle=True) as data:
                    for k in data.files:
                        if k == 'rt_bins_all':
                            optimal_params[k] = data[k]
                        else:
                            optimal_params[k] = data[k][ind_min - min_num]
                if objective == 'curve':
                    plt.plot(optimal_params['xpos_rt_pcom'],
                             optimal_params['median_pcom_rt'],
                             label=f'simul_{i}')
                if objective == 'RT':
                    action_t_a.append(optimal_params['p_t_a'][0] * stim_res)
                    action_slope.append(optimal_params['p_w_a_slope'][0])
                    action_intercept.append(optimal_params['p_w_a_intercept'][0])
                    aff_eff_sum.append((optimal_params['p_t_eff'][0] +
                                       optimal_params['p_t_aff'][0])*stim_res)
                    zt_weight.append(optimal_params['p_w_zt'][0])
                    stim_weight.append(optimal_params['p_w_stim'][0])
                    norm_vals = optimal_params['rt_vals_all'] /\
                        optimal_params['rt_vals_all'].sum()
                    plt.plot(rt_bins[0][:-1], norm_vals, label=f'simul_{i}')
            plt.xlabel('RT (ms)')
            plt.legend()
            if objective == 'curve':
                plt.ylabel('pCoM - detected')
            if objective == 'RT':
                plt.ylabel('Density')
                variables = ['p_t_a', 'AI_intercept', 'AI_slope',
                             'sum t_aff + t_eff (ms)', 'weight_zt', 'stim_weight']
                fig, ax = plt.subplots(ncols=len(variables))
                fig.suptitle('Parameters for the 100 configs with lowest RMSE')
                for r, variable in enumerate([action_t_a, action_intercept,
                                              action_slope, aff_eff_sum,
                                              zt_weight, stim_weight]):
                    ax[r].boxplot(variable)
                    ax[r].set_title(variables[r])
                plt.figure()
                diff_rms_mat = np.array(diff_rms_mat)
                plt.plot(diff_rms_mat[ind_sorted[0:i+1].astype(int)],
                         action_t_a/max(action_t_a), '.', label='p_t_a')
                plt.plot(diff_rms_mat[ind_sorted[0:i+1]],
                         aff_eff_sum/max(aff_eff_sum), '.', label='t_eff+t_aff')
                plt.plot(diff_rms_mat[ind_sorted[0:i+1]],
                         action_intercept/max(action_intercept), '.',
                         label='AI_int')
                plt.plot(diff_rms_mat[ind_sorted[0:i+1]],
                         np.abs(action_slope / max(np.abs(action_slope))),
                         '.', label='AI_slo')
                plt.xlabel('RMSE')
                plt.legend()
            # let's see the matrices
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
            sns.heatmap(np.flipud(optimal_params['pcom_matrix']), ax=ax[0])
            ax[0].set_title('Simulation')
            sns.heatmap(data_mat, ax=ax[1])
            ax[1].set_title('Data')
            plt.figure()
            plt.plot(data_curve['rt'], data_curve['pcom'], label='data',
                     linestyle='', marker='o')
            plt.plot(optimal_params['xpos_rt_pcom'],
                     optimal_params['median_pcom_rt'],
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
    else:
        sound_len = (first_ind-fixation+p_t_eff)*stim_res
        df_curve = {'detected_pcom': detected_com, 'sound_len': sound_len}
        df_curve = pd.DataFrame(df_curve)
        rt_vals, median_pcom, _ =\
            binned_curve(df_curve, 'detected_pcom', 'sound_len', xpos=20,
                         bins=BINS,
                         return_data=True)
        if len(rt_vals) == 0:
            diff_rms_mat.append(1e3)
        else:
            tmp_simul = np.array((rt_vals)/bin_size, dtype=int)
            diff_rms = np.subtract(median_pcom,
                                   np.array(data_curve['pcom']
                                            [tmp_simul])) ** 2
            num_nans = len(tmp_data) - len(tmp_simul)
            diff_rms_mat = np.sqrt(np.nansum(diff_rms))\
                + num_nans * nan_penalty
    return diff_rms_mat


def simulation(stim, zt, coh, trial_index, gt, com, pright, p_w_zt,
               p_w_stim, p_e_bound, p_com_bound, p_t_aff, p_t_eff, p_t_a,
               p_w_a_intercept, p_w_a_slope, p_a_bound, p_1st_readout,
               p_2nd_readout, p_leak, p_mt_noise, p_MT_intercept, p_MT_slope,
               num_times_tr=int(1e3), detect_CoMs_th=8, rms_comparison=False,
               epsilon=1e-6, mnle=True):
    start_llk = time.time()
    data_augment_factor = 10
    if isinstance(coh, np.ndarray):
        num_tr = stim.shape[1]
        indx_sh = np.arange(len(zt))
        np.random.shuffle(indx_sh)
        stim = stim[:, indx_sh]
        zt = zt[indx_sh]
        coh = coh[indx_sh]
        gt = gt[indx_sh]
        # num_tr = 5
        stim = stim[:, :int(num_tr)]
        zt = zt[:int(num_tr)]
        coh = coh[:int(num_tr)]
        com = com[:int(num_tr)]
        gt = gt[:int(num_tr)]
        trial_index = trial_index[:int(num_tr)]
        stim = data_augmentation(stim=stim, daf=data_augment_factor)
        stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                    stim.shape[1]))))
    else:
        augm_stim = np.zeros((data_augment_factor*len(stim), 1))
        for tmstp in range(len(stim)):
            augm_stim[data_augment_factor*tmstp:data_augment_factor*(tmstp+1)] =\
                stim[tmstp]
        stim = augm_stim
        stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff), 1))))
        num_tr = 1
        stim_temp = np.array(stim_temp)
    MT_slope = 0.15
    MT_intercep = 254
    compute_trajectories = True
    all_trajs = True

    stim_res = 50/data_augment_factor
    global fixation
    fixation = int(300 / stim_res)
    if not mnle:
        detected_com_mat = np.zeros((num_tr, num_times_tr))
        pright_mat = np.zeros((num_tr, num_times_tr))
    diff_rms_list = []
    if mnle:
        mt = torch.tensor(())
        # rt = torch.tensor(())
        com = torch.tensor(())
        choice = torch.tensor(())
    for i in range(num_times_tr):
        # start_simu = time.time()
        _, _, com_model, first_ind, _, _, resp_fin,\
            _, _, total_traj, _, _,\
            _, x_val_at_updt, _, _,\
            _, _, _ =\
            trial_ev_vectorized(zt=zt, stim=stim_temp, coh=coh,
                                trial_index=trial_index,
                                p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                p_e_bound=p_e_bound, p_com_bound=p_com_bound,
                                p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                                num_tr=num_tr, p_w_a_intercept=p_w_a_intercept,
                                p_w_a_slope=p_w_a_slope,
                                p_a_bound=p_a_bound,
                                p_1st_readout=p_1st_readout,
                                p_2nd_readout=p_2nd_readout,
                                p_leak=p_leak, p_mt_noise=p_mt_noise,
                                p_MT_intercept=p_MT_intercept,
                                p_MT_slope=p_MT_slope,
                                compute_trajectories=compute_trajectories,
                                stim_res=stim_res, all_trajs=all_trajs,
                                compute_mat_and_pcom=False)
        # reaction_time = (first_ind-int(300/stim_res) + p_t_eff)*stim_res
        motor_time = np.array([len(t) for t in total_traj])
        detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
        if not mnle:
            detected_com_mat[:, i] = detected_com
            pright_mat[:, i] = (resp_fin + 1)/2
        if mnle:
            first_ind = []
            total_traj = []
            x_val_at_updt = []
            com_model = []
            mt = torch.cat((mt, torch.tensor(motor_time)))
            # rt = torch.cat((rt, torch.tensor(reaction_time)))
            com = torch.cat((com, torch.tensor(detected_com*1)))
            choice = torch.cat((choice, torch.tensor((resp_fin+1)/2)))
        # end_simu = time.time()
        # print('Trial {} simulation: '.format(i) + str(end_simu - start_simu))
        if rms_comparison:
            diff_rms = fitting(detected_com=detected_com, p_t_eff=p_t_eff,
                               first_ind=first_ind, data_path=DATA_FOLDER)
            diff_rms_list.append(diff_rms)
    if mnle:
        choice_and_com = com + choice*2
        x = torch.column_stack((mt, choice_and_com))
        return x
    mat_right_and_com = detected_com_mat*pright_mat
    mat_right_and_nocom = (1-detected_com_mat)*pright_mat
    mat_left_and_com = detected_com_mat*(1-pright_mat)
    mat_left_and_nocom = (1-detected_com_mat)*(1-pright_mat)
    pright_and_com = np.nansum(mat_right_and_com, axis=1).astype(int)
    pright_and_nocom = np.nansum(mat_right_and_nocom, axis=1).astype(int)
    pleft_and_com = np.nansum(mat_left_and_com, axis=1).astype(int)
    pleft_and_nocom = np.nansum(mat_left_and_nocom, axis=1).astype(int)
    matrix_dirichlet = np.zeros((len(pright), 4)).astype(int)
    matrix_dirichlet[:, 0] = pright_and_com
    matrix_dirichlet[:, 1] = pright_and_nocom
    matrix_dirichlet[:, 2] = pleft_and_com
    matrix_dirichlet[:, 3] = pleft_and_nocom
    # start_dirichlet = time.time()
    K = matrix_dirichlet.shape[1]
    data = dme.CompressedRowData(K)
    for row_ind in range(K):
        data.appendRow(matrix_dirichlet[row_ind, :], weight=1)
    alpha_vector = dme.findDirichletPriors(data=data,
                                           initAlphas=[1, 1, 1, 1],
                                           iterations=1000)
    # end_dirichlet = time.time()
    # print('End Dirichlet: ' + str(end_dirichlet - start_dirichlet))
    alpha_sum = np.sum(alpha_vector)
    com = np.array(com, dtype=float)
    lk_list = []
    for i_p, p in enumerate(pright):
        if p == 1:
            if com[i_p] == 1:
                lk = (np.sum(mat_right_and_com, axis=1)[i_p] + alpha_vector[0])\
                    / (num_times_tr + alpha_sum)
            else:
                lk = (np.sum(mat_right_and_nocom, axis=1)[i_p] + alpha_vector[1])\
                    / (num_times_tr + alpha_sum)
        else:
            if com[i_p] == 1:
                lk = (np.sum(mat_left_and_com, axis=1)[i_p] + alpha_vector[2])\
                    / (num_times_tr + alpha_sum)
            else:
                lk = (np.sum(mat_left_and_nocom, axis=1)[i_p] + alpha_vector[3])\
                    / (num_times_tr + alpha_sum)
        lk_list.append(lk)
    llk_val = -np.nansum(np.log(lk_list))
    end_llk = time.time()
    print(end_llk - start_llk)
    if rms_comparison:
        diff_rms_mean = np.mean(diff_rms_list)
        return llk_val, diff_rms_mean
    else:
        return llk_val, None


def plot_rms_vs_llk(mean, sigma, zt, stim, iterations, scaling_value,
                    n_params=10, save_path=SV_FOLDER):
    """
    DEPRECATED
    """
    rms_comparison = True
    rms_list = []
    llk_list = []
    for i in range(1, int(iterations)+1):
        params = (mean + sigma*np.random.randn(n_params)) * scaling_value
        p_t_aff = int(params[0])
        p_t_eff = int(params[1])
        p_t_a = int(params[2])
        p_w_zt = params[3]
        p_w_stim = params[4]
        p_e_bound = params[5]
        p_com_bound = params[6]
        p_w_a = params[7]
        p_a_bound = params[8]
        p_w_updt = params[9]
        llk_val, diff_rms =\
            simulation(stim, zt, coh, gt, com, pright, p_w_zt, p_w_stim,
                       p_e_bound, p_com_bound, p_t_aff, p_t_eff,
                       p_t_a, p_w_a, p_a_bound, p_w_updt,
                       num_times_tr=int(1e1), detect_CoMs_th=5,
                       rms_comparison=rms_comparison)
        llk_list.append(llk_val)
        rms_list.append(diff_rms)
        if i % 3 == 0:
            plt.figure()
            plt.title('{} iterations'.format(i))
            plt.scatter(rms_list, llk_list)
            plt.xlabel('RMSE')
            plt.ylabel('Log-likelihood')
            plt.xlim(0, 0.15)
            plt.savefig(SV_FOLDER+'/figures/llk_vs_rms.png', dpi=400,
                        bbox_inches='tight')
            plt.close()


def build_prior_sample_theta(num_simulations):
    # 1. Parameters' prior distro definition
    prior =\
        MultipleIndependent([
            Uniform(torch.tensor([1e-1]),
                    torch.tensor([1])),  # prior weight
            Uniform(torch.tensor([1e-3]),
                    torch.tensor([0.8])),  # stim weight
            Uniform(torch.tensor([1e-2]),
                    torch.tensor([3.])),  # evidence integrator bound
            Uniform(torch.tensor([1e-8]),
                    torch.tensor([1.])),  # CoM bound
            Uniform(torch.tensor([4.]),
                    torch.tensor([12.])),  # afferent time
            Uniform(torch.tensor([4.]),
                    torch.tensor([12.])),  # efferent time
            Uniform(torch.tensor([9.]),
                    torch.tensor([16.])),  # time offset action
            Uniform(torch.tensor([1e-2]),
                    torch.tensor([0.08])),  # intercept trial index for action drift
            Uniform(torch.tensor([1e-7]),
                    torch.tensor([5e-5])),  # slope trial index for action drift
            Uniform(torch.tensor([0.5]),
                    torch.tensor([4.])),  # bound for action integrator
            Uniform(torch.tensor([10.]),
                    torch.tensor([100.])),  # weight of evidence at first readout (for MT reduction)
            Uniform(torch.tensor([10.]),
                    torch.tensor([100.])),  # weight of evidence at second readout
            Uniform(torch.tensor([0.2]),
                    torch.tensor([0.9])),  # leak
            Uniform(torch.tensor([1.]),
                    torch.tensor([35.])),  # std of the MT noise
            Uniform(torch.tensor([120.]),
                    torch.tensor([400.])),  # MT offset
            Uniform(torch.tensor([0.06]),
                    torch.tensor([0.5]))],  # MT slope with trial index
            validate_args=False)

    # 2. define all theta space with samples from prior
    theta_all = prior.sample((num_simulations,))
    return prior, theta_all


def fun_theta(theta, data, estimator):
    zt = data[:, 0]
    coh = data[:, 1]
    trial_index = data[:, 2]
    x_o = data[:, 2:-1]
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(num_simulations, 1)
    theta[:, 0] *= torch.tensor(zt[:n_trials])
    theta[:, 1] *= torch.tensor(coh[:n_trials])
    t_i = torch.tensor(
        trial_index[:n_trials].astype(float)).to(torch.float32)
    theta = torch.column_stack((theta, t_i))
    log_liks = estimator.log_prob(x_o[:n_trials], context=theta)
    return -torch.nansum(log_liks).detach().numpy()


def simulations_for_mnle(theta_all, stim, zt, coh, trial_index, gt):
    # run simulations
    x = torch.tensor(())
    print('Starting simulation')
    for i_t, theta in enumerate(theta_all):
        if (i_t+1) % 50000 == 0 and i_t != 0:
            print('Simulation number: ' + str(i_t+1))
        p_w_zt = float(theta[0])
        p_w_stim = float(theta[1])
        p_e_bound = float(theta[2])
        p_com_bound = float(theta[3])*p_e_bound
        p_t_aff = int(np.round(theta[4]))
        p_t_eff = int(np.round(theta[5]))
        p_t_a = int(np.round(theta[6]))
        p_w_a_intercept = float(theta[7])
        p_w_a_slope = -float(theta[8])
        p_a_bound = float(theta[9])
        p_1st_readout = float(theta[10])
        p_2nd_readout = float(theta[11])
        p_leak = float(theta[12])
        p_mt_noise = float(theta[13])
        p_mt_intercept = float(theta[14])
        p_mt_slope = float(theta[15])
        try:
            x_temp = simulation(stim[i_t, :], zt[i_t], coh[i_t],
                                np.array([trial_index[i_t]]), gt[i_t],
                                None, None,
                                p_w_zt, p_w_stim, p_e_bound, p_com_bound,
                                p_t_aff, p_t_eff, p_t_a, p_w_a_intercept,
                                p_w_a_slope, p_a_bound, p_1st_readout,
                                p_2nd_readout, p_leak, p_mt_noise,
                                p_mt_intercept, p_mt_slope,
                                rms_comparison=False,
                                num_times_tr=1, mnle=True)
        except ValueError:
            x_temp = torch.tensor([[np.nan, np.nan]])
        x = torch.cat((x, x_temp))
    x = x.to(torch.float32)
    return x


def opt_mnle(df, num_simulations, n_trials, bads=True):
    mt = df.resp_len.values
    choice = df.R_response.values
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    stim = np.array([stim for stim in df.res_sound])
    coh = np.array(df.coh2)
    trial_index = np.array(df.origidx)
    gt = np.array(df.rewside) * 2 - 1
    traj_stamps = df.trajectory_stamps.values  # [after_correct_id]
    traj_y = df.trajectory_y.values  # [after_correct_id]
    fix_onset = df.fix_onset_dt.values  # [after_correct_id]
    sound_len = np.array(df.sound_len)
    time_trajs = get_trajs_time(resp_len=np.array(df.resp_len),
                                traj_stamps=traj_stamps,
                                fix_onset=fix_onset, com=None,
                                sound_len=sound_len)
    _, _, _, com =\
        com_detection(trajectories=traj_y, decision=choice,
                      time_trajs=time_trajs, com_threshold=8)
    # Prepare data:
    coh = np.resize(coh, num_simulations)
    zt = np.resize(zt, num_simulations)
    trial_index = np.resize(trial_index, num_simulations)
    stim = np.resize(stim, (num_simulations, 20))
    gt = np.resize(gt, num_simulations)
    mt = np.resize(mt, num_simulations)
    choice = np.resize(choice, num_simulations)
    com = np.resize(com, num_simulations)
    choice_and_com = com + choice*2

    x_o = torch.column_stack((torch.tensor(mt*1e3),
                              torch.tensor(choice_and_com)))
    x_o = x_o.to(torch.float32)
    data = torch.column_stack((zt, coh, trial_index, x_o))
    prior, theta_all = build_prior_sample_theta(num_simulations=num_simulations)
    x = simulations_for_mnle(theta_all, stim, zt, coh, trial_index, gt)
    nan_mask = torch.sum(torch.isnan(x), axis=1).to(torch.bool)
    trainer = MNLE(prior=prior)
    theta_all_inp = theta_all.clone().detach()
    theta_all_inp[:, 0] *= torch.tensor(zt[:num_simulations]).to(torch.float32)
    theta_all_inp[:, 1] *= torch.tensor(coh[:num_simulations]).to(torch.float32)
    theta_all_inp = torch.column_stack((
        theta_all_inp, torch.tensor(
            trial_index[:num_simulations].astype(float)).to(torch.float32)))
    theta_all_inp = theta_all_inp.to(torch.float32)
    time_start = time.time()
    estimator = trainer.append_simulations(theta_all_inp[~nan_mask, :],
                                           x[~nan_mask, :]).train(
                                               show_train_summary=True)
    print('For a batch of ' + str(num_simulations) +
          ' simulations, it took ' + str(int(time.time() - time_start)/60)
          + ' mins')
    if bads:
        x0 = theta_all_inp[torch.argmin(-estimator.log_prob(
            x_o[:num_simulations], context=theta_all_inp)), :-1]
        x0[0] = 0.5
        x0[1] = 0.5
        print('Initial guess is: ' + str(x0))

        lb = np.array([np.float64(prior.dists[i].low/10)
                       for i in range(len(prior.dists))])
        ub = np.array([np.float64(prior.dists[i].high*100)
                       for i in range(len(prior.dists))])
        pub = np.array([np.float64(prior.dists[i].high)
                        for i in range(len(prior.dists))])
        plb = np.array([np.float64(prior.dists[i].low)
                        for i in range(len(prior.dists))])
        fun_target = lambda x: fun_theta(x, data, estimator)
        bads = BADS(fun_target, x0, lb, ub, plb, pub)
        optimize_result = bads.optimize()
        return optimize_result.x
    else:
        # Markov chain Monte-Carlo (MCMC) to get posterior distros
        num_samples = 10000
        mcmc_parameters = dict(num_chains=10, thin=10,
                               warmup_steps=num_samples//4,
                               init_strategy="proposal",
                               num_workers=1,)
        mnle_posterior = trainer.build_posterior(prior=prior,
                                                 mcmc_method="slice_np",
                                                 mcmc_parameters=mcmc_parameters)
        mnle_samples = mnle_posterior.sample((num_samples,), x=x_o,
                                             show_progress_bars=True)
        return mnle_samples
    # at this point, we should re-simulate the model with all trials
    # and compare distros


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    optimization_cmaes = False
    optimization_mnle = True
    rms_comparison = False
    plotting = False
    plot_rms_llk = False
    single_run = False
    stim, zt, coh, gt, com, pright, trial_index =\
        get_data(dfpath=DATA_FOLDER + 'LE43', after_correct=True,
                 num_tr_per_rat=int(1e4), all_trials=False)
    array_params = np.array((0.2, 0.11, 2, 1e-2, 8, 8, 14, 0.052, -2.2e-05,
                             2.6, 10, 10, 0.6, 35))
    scaled_params = np.repeat(1, len(array_params)).astype(float)
    scaling_value = array_params/scaled_params
    bounds = np.array(((0.1, 0.4), (1, 3), (0.001, 0.04), (1e-3, 1),
                       (1, 15), (1, 15), (2, 25), (0.01, 0.1), (-1e-04, 0),
                       (1, 3), (1, 100), (1, 100), (0, 2), (1, 60)))
    bounds_scaled = np.array([bound / scaling_value[i_b] for i_b, bound in
                              enumerate(bounds)])
    if single_run:
        p_t_aff = 8
        p_t_eff = 8
        p_t_a = 14  # 90 ms (18) PSIAM fit includes p_t_eff
        p_w_zt = 0.2
        p_w_stim = 0.11
        p_e_bound = 2
        p_com_bound = 0.
        p_w_a_intercept = 0.052
        p_w_a_slope = -2.2e-05  # fixed
        p_a_bound = 2.6  # fixed
        p_1st_readout = 10
        p_2nd_readout = 10
        p_leak = 0.6
        p_mt_noise = 35
        p_MT_intercept = 250
        p_MT_slope = 0.13
        llk_val, diff_rms =\
            simulation(stim, zt, coh, trial_index, gt, com, pright, p_w_zt,
                       p_w_stim, p_e_bound, p_com_bound, p_t_aff, p_t_eff,
                       p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
                       p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
                       p_MT_intercept, p_MT_slope, num_times_tr=int(1e1))
        print(llk_val)
    # TODO: paralelize different initial points
    if optimization_cmaes:
        num_times_tr = 100
        print('Start optimization')
        rms_list = []
        llk_list = []
        optimizer = CMA(mean=scaled_params, sigma=0.2, bounds=bounds_scaled)
        all_solutions = []
        for gen in range(50):
            solutions = []
            for it in range(optimizer.population_size):
                print('Generation {}, iteration {}'.format(gen+1, it+1))
                params_init = optimizer.ask()
                params = params_init * scaling_value
                p_w_zt = params[0]
                p_w_stim = params[1]
                p_e_bound = params[2]
                p_com_bound = params[3]
                p_t_aff = int(round(params[4]))
                p_t_eff = int(round(params[5]))
                p_t_a = int(round(params[6]))
                p_w_a_intercept = params[7]
                p_w_a_slope = params[8]
                p_a_bound = params[9]
                p_1st_readout = params[10]
                p_2nd_readout = params[11]
                p_leak = params[12]
                p_mt_noise = params[13]
                llk_val, diff_rms =\
                    simulation(stim, zt, coh, trial_index, gt, com, pright,
                               p_w_zt, p_w_stim, p_e_bound, p_com_bound,
                               p_t_aff, p_t_eff, p_t_a, p_w_a_intercept,
                               p_w_a_slope, p_a_bound, p_1st_readout,
                               p_2nd_readout, rms_comparison=rms_comparison,
                               num_times_tr=num_times_tr)
                solutions.append((params_init, llk_val))
                np.save(SV_FOLDER+'last_solutions.npy', solutions)
                if rms_comparison:
                    rms_list.append(diff_rms)
                    llk_list.append(llk_val)
            llk_mean_val = np.mean([sol[1] for sol in solutions])
            print('At generation {}:'.format(gen+1))
            print('mean llk: ' + str(llk_mean_val))
            all_solutions.append(solutions)
            if optimizer.should_stop():
                break
            optimizer.tell(solutions)
            np.save(SV_FOLDER+'all_solutions.npy', all_solutions)
            np.save(SV_FOLDER+'all_rms.npy', rms_list)
    if optimization_mnle:
        num_simulations = 2000
        n_trials = 1000
        # load real data
        subject = 'LE43'
        df = get_data_and_matrix(dfpath=DATA_FOLDER + subject, return_df=True,
                                 sv_folder=SV_FOLDER, after_correct=False,
                                 silent=True, all_trials=True,
                                 srfail=True)
        parameters = opt_mnle(df=df, num_simulations=num_simulations,
                              n_trials=n_trials, bads=True)
    if rms_comparison and plotting:
        plt.figure()
        plt.scatter(rms_list, llk_list)
        plt.xlabel('RMSE')
        plt.ylabel('Log-likelihood')
    if plot_rms_llk:
        mean = 1
        sigma = 0.1
        iterations = 100
        plot_rms_vs_llk(mean=mean, sigma=sigma, zt=zt, stim=stim,
                        iterations=iterations, scaling_value=scaling_value)


# RT < p_t_aff : proactive trials
