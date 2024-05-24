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
import pickle
import scipy
from pybads import BADS
import itertools
from scipy.spatial import distance as dist
import os
from scipy.signal import convolve2d
# from pyvbmc import VBMC

sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')  # Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
from utilsJ.Models.extended_ddm_v2 import trial_ev_vectorized,\
    data_augmentation, get_data_and_matrix
from utilsJ.Behavior.plotting import binned_curve
import utilsJ.Models.dirichletMultinomialEstimation as dme
from skimage.transform import resize
from scipy.special import rel_entr
from utilsJ.paperfigs import figure_1 as fig1
from utilsJ.paperfigs import figures_paper as fp
from utilsJ.Models import analyses_humans as ah
from utilsJ.Models import different_models as model_variations

DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
# DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM

SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
# SV_FOLDER = '/home/garciaduran/opt_results/'  # Cluster Alex
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/opt_results/' # Jordi
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM

BINS = np.arange(1, 320, 20)
CTE = 1/2 * 1/600 * 1/995  # contaminants
CTE_FB = 1/600

plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'font.size': 14})


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


def rmse_fitting(res_path='C:/Users/Alexandre/Desktop/CRM/Results_LE38/',
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
               epsilon=1e-6, mnle=True, extra_label=''):
    if extra_label == '':
        model = trial_ev_vectorized
    if 'only_prior' in extra_label:
        model = model_variations.trial_ev_vectorized_only_prior_1st_choice
    start_llk = time.time()
    data_augment_factor = 10
    if isinstance(coh, np.ndarray):
        num_tr = stim.shape[1]
        indx_sh = np.arange(len(zt))
        np.random.shuffle(indx_sh)
        stim = stim[:, indx_sh]
        zt = zt[indx_sh]
        coh = coh[indx_sh]
        # num_tr = 5
        stim = stim[:, :int(num_tr)]
        zt = zt[:int(num_tr)]
        coh = coh[:int(num_tr)]
        com = com[:int(num_tr)]
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
        rt = torch.tensor(())
        # com = torch.tensor(())
        choice = torch.tensor(())
    for i in range(num_times_tr):
        # start_simu = time.time()
        _, _, com_model, first_ind, _, _, resp_fin,\
            _, _, total_traj, _, _,\
            _, x_val_at_updt, _, _,\
            _, _, _ =\
            model(zt=zt, stim=stim_temp, coh=coh,
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
        reaction_time = (first_ind-int(300/stim_res) + p_t_eff)*stim_res
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
            rt = torch.cat((rt, torch.tensor(reaction_time)))
            # com = torch.cat((com, torch.tensor(detected_com*1)))
            choice = torch.cat((choice, torch.tensor((resp_fin+1)/2)))
        # end_simu = time.time()
        # print('Trial {} simulation: '.format(i) + str(end_simu - start_simu))
        if rms_comparison:
            diff_rms = rmse_fitting(detected_com=detected_com, p_t_eff=p_t_eff,
                               first_ind=first_ind, data_path=DATA_FOLDER)
            diff_rms_list.append(diff_rms)
    if mnle:
        # choice_and_com = com + choice*2
        x = torch.column_stack((mt, rt+300, choice))  # add fixation
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
            Uniform(torch.tensor([1e-3]),
                    torch.tensor([1.])),  # prior weight
            Uniform(torch.tensor([1e-3]),
                    torch.tensor([0.8])),  # stim weight
            Uniform(torch.tensor([1e-2]),
                    torch.tensor([4.])),  # evidence integrator bound
            Uniform(torch.tensor([1e-8]),
                    torch.tensor([1.])),  # CoM bound
            Uniform(torch.tensor([3.]),
                    torch.tensor([12.])),  # afferent time
            Uniform(torch.tensor([3.]),
                    torch.tensor([12.])),  # efferent time
            Uniform(torch.tensor([4.]),
                    torch.tensor([24.])),  # time offset action
            Uniform(torch.tensor([1e-2]),
                    torch.tensor([0.1])),  # intercept trial index for action drift
            Uniform(torch.tensor([1e-6]),
                    torch.tensor([5e-5])),  # slope trial index for action drift
            Uniform(torch.tensor([1.]),
                    torch.tensor([4.])),  # bound for action integrator
            Uniform(torch.tensor([1.]),
                    torch.tensor([500.])),  # weight of evidence at first readout (for MT reduction)
            Uniform(torch.tensor([1.]),
                    torch.tensor([500.])),  # weight of evidence at second readout
            Uniform(torch.tensor([1e-6]),
                    torch.tensor([0.9])),  # leak
            Uniform(torch.tensor([5.]),
                    torch.tensor([60.])),  # std of the MT noise
            Uniform(torch.tensor([120.]),
                    torch.tensor([400.])),  # MT offset
            Uniform(torch.tensor([0.01]),
                    torch.tensor([0.5]))],  # MT slope with trial index
            validate_args=False)

    # 2. define all theta space with samples from prior
    theta_all = prior.sample((num_simulations,))
    return prior, theta_all


def closest(lst, K):
    # returns index of closest value of K in a list lst
    return min(range(len(lst)), key=lambda i: abs(lst[i]-K))


def get_log_likelihood_fb_nn(rt_fb, theta_fb, estimator, min_prob=1e-30, binsize=40,
                             eps=1e-3):
    """
    Function that returns the log prob. of the NN (estimator) of having rt_fb
    given theta_fb.

    Parameters
    ----------
    x_o_with_fb : tensor
        Tensor of 1 column (FB RT) and N rows (FB trials).
    theta_fb : tensor
        Tensor of 17 columns and N rows (FB trials).
    estimator : dictionary
        MNLE NN.
    min_prob : float, optional
        Min. probability so we don't have log(0). The default is 1e-12.

    Returns
    -------
    log_liks_fb : float
        Sum of -LLH.

    """
    log_liks_fb = []
    # grid_rt = np.arange(-300, 301, binsize)[:-1] + 300 + binsize/2
    grid_mt = np.arange(0, 901, binsize)[:-1] + binsize/2
    all_rt = np.meshgrid([np.nan], grid_mt)[0].flatten()
    all_mt = np.meshgrid([np.nan], grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o_mat = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    comb_0 = []
    comb_1 = []
    for i_trial, rt in enumerate(rt_fb):
        theta_in = theta_fb[i_trial, :].repeat(len(x_o_mat), 1)
        x_o_mat[:, 1] = rt
        # get log prob of this trial log(P(data | theta))
        lprobs = estimator.log_prob(x_o_mat, context=theta_in).detach().numpy()
        # get prob P(data | theta, side_response)
        lprobs = np.exp(lprobs)
        mat_0_nn = lprobs[x_o_mat[:, 2] == 0].reshape(len(all_mt),
                                                      1)
        mat_1_nn = lprobs[x_o_mat[:, 2] == 1].reshape(len(all_mt),
                                                      1)
        # sum prob P(data | theta, 0, MT) + P(data | theta, 1, MT) so we can have
        # P(RT | theta, 0, MT)
        mat_final = (mat_0_nn + mat_1_nn)*2
        # sum probs for all MT
        marginal_rt = np.nansum(mat_final, axis=0)*binsize
        marginal_rt += min_prob
        log_liks_fb.append(np.log(marginal_rt))  # [closest(grid_rt, rt)]
    log_liks_fb = -np.nansum(np.log(np.exp(log_liks_fb)*(1-eps) + eps*CTE))
    return log_liks_fb


def prob_rt_fb_action(t, v_a, t_a, bound_a):
    # returns p(RT | theta) for RT < 0
    return (bound_a / np.sqrt(2*np.pi*(t - t_a)**3)) *\
        np.exp(- ((v_a**2)*((t-t_a) - bound_a/v_a)**2)/(2*(t-t_a)))


def get_log_likelihood_fb_psiam(rt_fb, theta_fb, eps, dt=5e-3):
    # returns -LLH ( RT | theta ) for RT < 0
    v_a = -theta_fb[:, 8]*theta_fb[:, -1] + theta_fb[:, 7]  # v_a = b_o - b_1*t_index
    v_a = v_a.detach().numpy()/dt
    bound_a = theta_fb[:, 9].detach().numpy()
    t_a = dt*(theta_fb[:, 6] + theta_fb[:, 5]).detach().numpy()
    t = rt_fb*1e-3
    prob = prob_rt_fb_action(t=t, v_a=v_a, t_a=t_a, bound_a=bound_a)
    prob[np.isnan(prob)] = 0
    # prob[prob > 1] = 1
    return -np.nansum(np.log(prob*(1-eps) + eps*CTE_FB))


def fun_theta(theta, data, estimator, n_trials, eps=1e-3, weight_LLH_fb=1):
    zt = data[:, 0]
    coh = data[:, 1]
    trial_index = data[:, 2]
    x_o = data[:, 3::]
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(n_trials, 1)
    theta[:, 0] *= torch.tensor(zt[:n_trials])
    theta[:, 1] *= torch.tensor(coh[:n_trials])
    t_i = torch.tensor(
        trial_index[:n_trials]).to(torch.float32)
    theta = torch.column_stack((theta, t_i))
    x_o = x_o[:n_trials].detach().numpy()
    # trials with RT >= 0
    # we have to pass the same parameters as for the training (14 columns)
    x_o_no_fb = torch.tensor(
        x_o[np.isnan(x_o).sum(axis=1) == 0, :]).to(torch.float32)
    theta_no_fb = torch.tensor(
        theta.detach().numpy()[np.isnan(x_o).sum(axis=1) == 0, :]).to(torch.float32)
    theta_no_fb[:, 14] += theta_no_fb[:, 15]*theta_no_fb[:, -1]
    theta_no_fb[:, 7] -= theta_no_fb[:, 8]*theta_no_fb[:, -1]
    theta_no_fb = torch.column_stack((theta_no_fb[:, :8],
                                      theta_no_fb[:, 9:15]))
    log_liks = estimator.log_prob(x_o_no_fb, context=theta_no_fb).detach().numpy()
    log_liks = np.exp(log_liks)*(1-eps) + eps*CTE
    log_liks = np.log(log_liks)
    log_liks_no_fb = -np.nansum(log_liks)  # -LLH (data | theta) for RT > 0
    # trials with RT < 0
    # we use the analytical computation of p(RT | parameters) for FB
    x_o_with_fb = x_o[np.isnan(x_o).sum(axis=1) > 0, :]
    theta_fb = theta[np.isnan(x_o).sum(axis=1) > 0, :]
    log_liks_fb = get_log_likelihood_fb_psiam(rt_fb=x_o_with_fb[:, 1],
                                              theta_fb=theta_fb, eps=eps)
    # print('-LLH ( FB )')
    # print(log_liks_fb)
    # print('-LLH ( RT > 0 )')
    # print(log_liks_no_fb)
    # print('Ratio LLH: FB/NOFB')
    # print(log_liks_fb/log_liks_no_fb)

    # returns -LLH (data (RT > 0) | theta) + -LLH (data (RT < 0) | theta)
    return log_liks_fb*weight_LLH_fb + log_liks_no_fb  # *(1-weight_LLH_fb)


def simulations_for_mnle(theta_all, stim, zt, coh, trial_index,
                         simulate=False, extra_label=''):
    # run simulations
    x = torch.tensor(())
    simul_data = SV_FOLDER+'/network/NN_simulations'+str(len(zt))+'.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(simul_data), exist_ok=True)
    if os.path.exists(simul_data) and not simulate:
        print('Loading Simulated Data')
        x = np.load(simul_data, allow_pickle=True)
        x = torch.tensor(x).to(torch.float32)
    else:
        print('Starting simulation')
        time_start = time.time()
        for i_t, theta in enumerate(theta_all):
            if (i_t+1) % 100000 == 0 and i_t != 0:
                print('Simulation number: ' + str(i_t+1))
                print('Time elapsed: ' + str((time.time()-time_start)/60) +
                      ' mins')
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
                                    np.array([trial_index[i_t]]), None,
                                    None, None,
                                    p_w_zt, p_w_stim, p_e_bound, p_com_bound,
                                    p_t_aff, p_t_eff, p_t_a, p_w_a_intercept,
                                    p_w_a_slope, p_a_bound, p_1st_readout,
                                    p_2nd_readout, p_leak, p_mt_noise,
                                    p_mt_intercept, p_mt_slope,
                                    rms_comparison=False,
                                    num_times_tr=1, mnle=True,
                                    extra_label=extra_label)
            except ValueError:
                x_temp = torch.tensor([[np.nan, np.nan, np.nan]])
            x = torch.cat((x, x_temp))
        x = x.to(torch.float32)
        np.save(simul_data, x.detach().numpy())
    return x


def theta_for_lh_plot():
    p_t_aff = 5
    p_t_eff = 4
    p_t_a = 14  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.5
    p_w_stim = 0.14
    p_e_bound = 2.
    p_com_bound = 0.1
    p_w_a_intercept = 0.056
    p_w_a_slope = 2e-5
    p_a_bound = 2.6
    p_1st_readout = 40
    p_2nd_readout = 80
    p_leak = 0.5
    p_mt_noise = 20
    p_MT_intercept = 320
    p_MT_slope = 0.07
    return [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
            p_MT_intercept, p_MT_slope]


def theta_for_lh_plot_v2():
    p_t_aff = 7
    p_t_eff = 3
    p_t_a = 15  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.3
    p_w_stim = 0.08
    p_e_bound = 1.6
    p_com_bound = 0.11
    p_w_a_intercept = 0.04
    p_w_a_slope = 2e-5
    p_a_bound = 2.6
    p_1st_readout = 70
    p_2nd_readout = 50
    p_leak = 0.5
    p_mt_noise = 30
    p_MT_intercept = 310
    p_MT_slope = 0.06
    return [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
            p_MT_intercept, p_MT_slope]


def get_x0():
    p_t_aff = 6
    p_t_eff = 6
    p_t_a = 16  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.2
    p_w_stim = 0.12
    p_e_bound = 2.
    p_com_bound = 0.1
    p_w_a_intercept = 0.05
    p_w_a_slope = 2e-5
    p_a_bound = 2.6
    p_1st_readout = 40
    p_2nd_readout = 80
    p_leak = 0.06
    p_mt_noise = 12
    p_MT_intercept = 320
    p_MT_slope = 0.07
    return [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
            p_MT_intercept, p_MT_slope]



def get_x0_v2():
    p_t_aff = 6
    p_t_eff = 6
    p_t_a = 16  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.2
    p_w_stim = 0.12
    p_e_bound = 2.
    p_com_bound = 0.02
    p_w_a_intercept = 0.05
    p_w_a_slope = 2e-5
    p_a_bound = 2.6
    p_1st_readout = 200
    p_2nd_readout = 200
    p_leak = 0.06
    p_mt_noise = 1
    p_MT_intercept = 320
    p_MT_slope = 0.07
    return [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound,
            p_1st_readout, p_2nd_readout, p_leak, p_mt_noise,
            p_MT_intercept, p_MT_slope]



def get_lb():
    """
    Returns list with hard lower bounds (LB) for BADS optimization.

    Returns
    -------
    list
        List with hard lower bounds.

    """
    lb_aff = 3
    lb_eff = 3
    lb_t_a = 4
    lb_w_zt = 0
    lb_w_st = 0
    lb_e_bound = 0.3
    lb_com_bound = 0
    lb_w_intercept = 0.01
    lb_w_slope = 1e-6
    lb_a_bound = 0.1
    lb_1st_r = 25
    lb_2nd_r = 25
    lb_leak = 0
    lb_mt_n = 1
    lb_mt_int = 120
    lb_mt_slope = 0.01
    return [lb_w_zt, lb_w_st, lb_e_bound, lb_com_bound, lb_aff,
            lb_eff, lb_t_a, lb_w_intercept, lb_w_slope, lb_a_bound,
            lb_1st_r, lb_2nd_r, lb_leak, lb_mt_n,
            lb_mt_int, lb_mt_slope]


def get_lb_human():
    """
    Returns list with hard lower bounds (LB) for BADS optimization.

    Returns
    -------
    list
        List with hard lower bounds.

    """
    lb_aff = 3
    lb_eff = 3
    lb_t_a = 4
    lb_w_zt = 0
    lb_w_st = 0
    lb_e_bound = 0.3
    lb_com_bound = 0
    lb_w_intercept = 0.01
    lb_w_slope = 1e-6
    lb_a_bound = 0.1
    lb_1st_r = 25
    lb_2nd_r = 25
    lb_leak = 0
    lb_mt_n = 1
    lb_mt_int = 120
    lb_mt_slope = 0.01
    return [lb_w_zt, lb_w_st, lb_e_bound, lb_com_bound, lb_aff,
            lb_eff, lb_t_a, lb_w_intercept, lb_w_slope, lb_a_bound,
            lb_1st_r, lb_2nd_r, lb_leak, lb_mt_n,
            lb_mt_int, lb_mt_slope]


def get_ub():
    """
    Returns list with hard upper bounds (UB) for BADS optimization.

    Returns
    -------
    list
        List with hard upper bounds.

    """
    ub_aff = 12
    ub_eff = 12
    ub_t_a = 22
    ub_w_zt = 1
    ub_w_st = 0.18
    ub_e_bound = 4
    ub_com_bound = 0.4
    ub_w_intercept = 0.12
    ub_w_slope = 1e-3
    ub_a_bound = 4
    ub_1st_r = 500
    ub_2nd_r = 400
    ub_leak = 0.14
    ub_mt_n = 20
    ub_mt_int = 370
    ub_mt_slope = 0.6
    return [ub_w_zt, ub_w_st, ub_e_bound, ub_com_bound, ub_aff,
            ub_eff, ub_t_a, ub_w_intercept, ub_w_slope, ub_a_bound,
            ub_1st_r, ub_2nd_r, ub_leak, ub_mt_n,
            ub_mt_int, ub_mt_slope]


def get_ub_human():
    """
    Returns list with hard upper bounds (UB) for BADS optimization.

    Returns
    -------
    list
        List with hard upper bounds.

    """
    ub_aff = 12
    ub_eff = 12
    ub_t_a = 22
    ub_w_zt = 1
    ub_w_st = 0.2
    ub_e_bound = 4
    ub_com_bound = 1
    ub_w_intercept = 0.12
    ub_w_slope = 1e-3
    ub_a_bound = 4
    ub_1st_r = 400
    ub_2nd_r = 400
    ub_leak = 0.15
    ub_mt_n = 20
    ub_mt_int = 370
    ub_mt_slope = 0.6
    return [ub_w_zt, ub_w_st, ub_e_bound, ub_com_bound, ub_aff,
            ub_eff, ub_t_a, ub_w_intercept, ub_w_slope, ub_a_bound,
            ub_1st_r, ub_2nd_r, ub_leak, ub_mt_n,
            ub_mt_int, ub_mt_slope]


def get_pub():
    """
    Returns list with plausible upper bounds (PUB) for BADS optimization.

    Returns
    -------
    list
        List with plausible upper bounds.

    """
    pub_aff = 9
    pub_eff = 9
    pub_t_a = 16
    pub_w_zt = 0.7
    pub_w_st = 0.14
    pub_e_bound = 2.6
    pub_com_bound = 0.2
    pub_w_intercept = 0.08
    pub_w_slope = 1e-4
    pub_a_bound = 3
    pub_1st_r = 400
    pub_2nd_r = 400
    pub_leak = 0.1
    pub_mt_n = 13
    pub_mt_int = 320
    pub_mt_slope = 0.12
    return [pub_w_zt, pub_w_st, pub_e_bound, pub_com_bound, pub_aff,
            pub_eff, pub_t_a, pub_w_intercept, pub_w_slope, pub_a_bound,
            pub_1st_r, pub_2nd_r, pub_leak, pub_mt_n,
            pub_mt_int, pub_mt_slope]


def get_plb():
    """
    Returns list with plausible lower bounds (PLB) for BADS optimization.

    Returns
    -------
    list
        List with plausible lower bounds.

    """
    plb_aff = 4
    plb_eff = 4
    plb_t_a = 12
    plb_w_zt = 2e-2
    plb_w_st = 8e-3
    plb_e_bound = 0.5
    plb_com_bound = 1e-3
    plb_w_intercept = 0.03
    plb_w_slope = 1.5e-5
    plb_a_bound = 2.2
    plb_1st_r = 40
    plb_2nd_r = 40
    plb_leak = 1e-5
    plb_mt_n = 10
    plb_mt_int = 290
    plb_mt_slope = 0.04
    return [plb_w_zt, plb_w_st, plb_e_bound, plb_com_bound, plb_aff,
            plb_eff, plb_t_a, plb_w_intercept, plb_w_slope, plb_a_bound,
            plb_1st_r, plb_2nd_r, plb_leak, plb_mt_n,
            plb_mt_int, plb_mt_slope]


def nonbox_constraints_bads(x):
    x_1 = np.atleast_2d(x)
    # cond1 = x_1[:, 6] + x_1[:, 9]/x_1[:, 7] + np.int32(x_1[:, 5]) > 121
    # ~ min. action RT peak can't be < -150 ms
    # cond4 = x_1[:, 0]*3.5/x_1[:, 2] > 0.7
    # ub for prior. i.e. prior*p_zt can't be > 70% of the bound
    cond5 = x_1[:, 1] < 1e-2  # lb for stim
    cond4 = x[:,0] < 1e-2 # lb for prior
    # cond6 = np.int32(x_1[:, 4]) + np.int32(x_1[:, 5]) < 8  # aff + eff < 40 ms
    return np.bool_(cond4 + cond5)  # cond1


def gumbel_plotter():
    # fits = scipy.stats.gumbel_r.fit(df.resp_len.values*1000)
    fig, ax = plt.subplots(ncols=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    val = np.arange(1, 81, 10)
    val = [1]
    for v in val:
        norm = []
        gumb = []
        for i in range(500000):
            norm.append(v*np.random.randn())
        for i in range(500000):
            gumb.append(v*np.random.gumbel())
        sns.kdeplot(np.array(norm)[~np.isnan(norm)], color='k',
                    ax=ax, label='Normal')
        sns.kdeplot(np.array(gumb)[~np.isnan(gumb)], color='r',
                    ax=ax, label='Gumbel')
    ax.legend()
    ax.set_yticks([])
    ax.set_xlabel('x')


def prepare_fb_data(df):
    print('Preparing FB data')
    coh_vec = df.coh2.values
    dwl_vec = df.dW_lat.values
    dwt_vec = df.dW_trans.values
    mt_vec = df.resp_len.values
    ch_vec = df.R_response.values
    tr_in_vec = df.origidx.values
    for ifb, fb in enumerate(df.fb):
        for j in range(len(fb)):
            coh_vec = np.append(coh_vec, [df.coh2.values[ifb]])
            dwl_vec = np.append(dwl_vec, [df.dW_lat.values[ifb]])
            dwt_vec = np.append(dwt_vec, [df.dW_trans.values[ifb]])
            mt_vec = np.append(mt_vec, [np.nan])
            ch_vec = np.append(ch_vec, [np.nan])
            tr_in_vec = np.append(tr_in_vec, [df.origidx.values[ifb]])
    rt_vec =\
        np.vstack(np.concatenate([df.sound_len,
                                  1e3*(np.concatenate(
                                      df.fb.values)-0.3)])).reshape(-1)+300
    zt_vec = np.nansum(np.column_stack((dwl_vec, dwt_vec)), axis=1)
    x_o = torch.column_stack((torch.tensor(mt_vec*1e3),
                              torch.tensor(rt_vec),
                              torch.tensor(ch_vec)))
    data = torch.column_stack((torch.tensor(zt_vec), torch.tensor(coh_vec),
                               torch.tensor(tr_in_vec.astype(float)),
                               x_o))
    data = data[np.round(rt_vec) > 50, :]
    return data


def opt_mnle(df, num_simulations, bads=True, training=False, extra_label=""):
    if training:
        # 1st: loading data
        zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        stim = np.array([stim for stim in df.res_sound])
        coh = np.array(df.coh2)
        trial_index = np.array(df.origidx)
        stim[df.soundrfail, :] = 0
        # Prepare data:
        coh = np.resize(coh, num_simulations)
        # np.random.shuffle(coh)
        zt = np.resize(zt, num_simulations)
        # np.random.shuffle(zt)
        trial_index = np.resize(trial_index, num_simulations)
        # np.random.shuffle(trial_index)
        stim = np.resize(stim, (num_simulations, 20))
        # np.random.shuffle(stim)
        if not bads:
            # motor time: in seconds (must be multiplied then by 1e3)
            mt = df.resp_len.values
            choice = df.R_response.values
            sound_len = np.array(df.sound_len)
            mt = np.resize(mt, num_simulations)
            choice = np.resize(choice, num_simulations)
            # com = np.resize(com, num_simulations)
            # choice_and_com = com + choice*2
            rt = np.resize(sound_len + 300, num_simulations)
            # w.r.t fixation onset
            x_o = torch.column_stack((torch.tensor(mt*1e3),  # MT in ms
                                      torch.tensor(rt),
                                      torch.tensor(choice)))
            x_o = x_o.to(torch.float32)
            # to save some memory
            choice = []
            rt = []
            mt = []
        print('Data preprocessed, building prior distros')
        # build prior: ALL PARAMETERS ASSUMED POSITIVE
        df = []  # ONLY FOR TRAINING
        prior, theta_all = build_prior_sample_theta(num_simulations=num_simulations)
        # add zt, coh, trial index
        theta_all_inp = theta_all.clone().detach()
        theta_all_inp[:, 0] *= torch.tensor(zt[:num_simulations]).to(torch.float32)
        theta_all_inp[:, 1] *= torch.tensor(coh[:num_simulations]).to(torch.float32)
        theta_all_inp = torch.column_stack((
            theta_all_inp, torch.tensor(
                trial_index[:num_simulations].astype(float)).to(torch.float32)))
        theta_all_inp = theta_all_inp.to(torch.float32)
        # SIMULATION
        x = simulations_for_mnle(theta_all, stim, zt, coh, trial_index, simulate=True,
                                 extra_label=extra_label)
        # now we have a matrix of (num_simulations x 3):
        # MT, RT, CHOICE for each simulation

        # NETWORK TRAINING
        # transform parameters related to trial index. 14 params instead of 17
        # MT_in = MT_0 + MT_1*trial_index
        theta_all_inp[:, 14] += theta_all_inp[:, 15]*theta_all_inp[:, -1]
        # V_A = vA_0 - vA_1*trial_index
        theta_all_inp[:, 7] -= theta_all_inp[:, 8]*theta_all_inp[:, -1]
        theta_all_inp = torch.column_stack((theta_all_inp[:, :8],
                                            theta_all_inp[:, 9:15]))
        coh = []
        zt = []
        trial_index = []
        stim = []
        nan_mask = torch.sum(torch.isnan(x), axis=1).to(torch.bool)
        # define network MNLE
        trainer = MNLE(prior=prior)
        time_start = time.time()
        print('Starting network training')
        # network training
        trainer = trainer.append_simulations(theta_all_inp[~nan_mask, :],
                                             x[~nan_mask, :])
        estimator = trainer.train(show_train_summary=True)
        # save the network
        with open(SV_FOLDER + f"/mnle_n{num_simulations}_no_noise" + extra_label + ".p",
                  "wb") as fh:
            pickle.dump(dict(estimator=estimator,
                             num_simulations=num_simulations), fh)
        with open(SV_FOLDER + f"/trainer_n{num_simulations}_no_noise" + extra_label + ".p",
                  "wb") as fh:
            pickle.dump(dict(trainer=trainer,
                             num_simulations=num_simulations), fh)
        print('For a batch of ' + str(num_simulations) +
              ' simulations, it took ' + str(int(time.time() - time_start)/60)
              + ' mins')
    else:
        x_o = []
        with open(SV_FOLDER + f"/mnle_n{num_simulations}_no_noise" + extra_label + ".p",
                  'rb') as f:
            estimator = pickle.load(f)
        if not bads:
            with open(SV_FOLDER + f"/trainer_n{num_simulations}_no_noise" + extra_label + ".p",
                      'rb') as f:
                trainer = pickle.load(f)
            trainer = estimator['trainer']
        estimator = estimator['estimator']
    if bads:
        # find starting point
        x0 = get_x0()
        print('Initial guess is: ' + str(x0))
        time_start = time.time()
        lb = get_lb()
        ub = get_ub()
        pub = get_pub()
        plb = get_plb()
        # get fixation break (FB) data
        data = prepare_fb_data(df=df)
        print('Optimizing')
        n_trials = len(data)
        # define fun_target as function to optimize
        # returns -LLH( data | parameters )
        fun_target = lambda x: fun_theta(x, data, estimator, n_trials)  # f(theta | data, MNLE)
        # define optimizer (BADS)
        bads = BADS(fun_target, x0, lb, ub, plb, pub,
                    non_box_cons=nonbox_constraints_bads)
        # optimization
        optimize_result = bads.optimize()
        print(optimize_result.total_time)
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


def parameter_recovery_test_data_frames(df, subjects, extra_label=''):
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt =\
        fp.run_simulation_different_subjs(stim=None, zt=None, coh=None, gt=None,
                                          trial_index=None, num_tr=len(df),
                                          subject_list=subjects, subjid=None,
                                          simulate=False,
                                          extra_label=extra_label)
    
    MT = [len(t) for t in trajs]
    df['sound_len'] = reaction_time
    df['resp_len'] = np.array(MT)*1e-3
    df['R_response'] = (resp_fin+1)/2
    return df


def matrix_probs_v0(x, bins_rt=np.arange(200, 600, 13),
                    bins_mt=np.arange(100, 600, 26)):
    mt = x[:, 0]
    rt = x[:, 1]
    n_total = len(mt)
    mat_final = np.zeros((len(bins_rt)-1, len(bins_mt)-1))
    for irt, rtb in enumerate(bins_rt[:-1]):
        for imt, mtb in enumerate(bins_mt[:-1]):
            index = (mt >= mtb) & (mt < bins_mt[imt+1]) &\
                (rt >= rtb) & (rt < bins_rt[irt+1])
            mat_final[irt, imt] = sum(index)/n_total
    return mat_final


def matrix_probs(x, bins_rt=np.arange(200, 600, 13),
                 bins_mt=np.arange(100, 600, 26)):
    mt = np.array(x[:, 0])
    rt = np.array(x[:, 1])
    counts = np.histogram2d(mt, rt, bins=[bins_mt, bins_rt])[0]
    counts /= np.sum(counts)
    # for irt, rtb in enumerate(bins_rt[:-1]):
    #     for imt, mtb in enumerate(bins_mt[:-1]):
    #         index = (mt >= mtb) & (mt < bins_mt[imt+1]) &\
    #             (rt >= rtb) & (rt < bins_rt[irt+1])
    #         mat_final[irt, imt] = sum(index)/n_total
    return counts


def get_manual_kl_divergence(mat_model, mat_nn):
    mat_model = mat_model.flatten()
    mat_nn = mat_nn.flatten()
    return np.sum(mat_model * np.log(mat_model/mat_nn))


def plot_network_model_comparison(df, ax, sv_folder=SV_FOLDER, num_simulations=int(5e5),
                                  n_list=[4000000], cohval=0.5, ztval=0.5, tival=10,
                                  plot_nn=False, simulate=False, plot_model=True,
                                  plot_nn_alone=False, xt=False, eps=1e-5, n_trials_sim=100):
    grid_rt = np.arange(-100, 300, 12) + 300
    grid_mt = np.arange(100, 600, 25)
    # all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    # all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    # comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    # comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    # x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    # to simulate
    if simulate:
        coh = df.coh2.values
        zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        trial_index = df.origidx.values
        idxs = np.random.choice(np.arange(len(coh)), size=100)
        cohvals = coh[idxs]
        coh = []
        ztvals = np.round(zt[idxs], 2)
        zt = []
        tivals = trial_index[idxs]
        trial_index = []
        stims = np.array(
            [stim for stim in df.res_sound])[idxs]
        np.save(sv_folder + '/10M/cohvals.npy', cohvals)
        np.save(sv_folder + '/10M/ztvals.npy', ztvals)
        np.save(sv_folder + '/10M/tivals.npy', tivals)
        np.save(sv_folder + '/10M/stims.npy', stims)
        np.save(sv_folder + '/10M/idxs.npy', idxs)
        i = 0
        for cohval, ztval, tival in zip(cohvals, ztvals, tivals):
            stim = stims[i]
            theta = get_x0()
            theta = torch.reshape(torch.tensor(theta),
                                  (1, len(theta))).to(torch.float32)
            theta = theta.repeat(num_simulations, 1)
            stim = np.array(
                [np.concatenate((stim, stim)) for i in range(len(theta))])
            trial_index = np.repeat(tival, len(theta))
            x = simulations_for_mnle(theta_all=np.array(theta), stim=stim,
                                     zt=np.repeat(ztval, len(theta)),
                                     coh=np.repeat(cohval, len(theta)),
                                     trial_index=trial_index, simulate=True)
            np.save(sv_folder + '/10M/coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival), x)
            # let's compute prob for each bin
            mat_0 = matrix_probs(x[x[:, 2] == 0])
            mat_1 = matrix_probs(x[x[:, 2] == 1])
            np.save(sv_folder + '/10M/mat0_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival), mat_0)
            np.save(sv_folder + '/10M/mat1_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival), mat_1)
            x = []
            mat_0 = []
            mat_1 = []
            i += 1
    else:
        if not plot_nn_alone:
            mat_0 = np.load(SV_FOLDER + '/10M/mat0_coh{}_zt{}_ti{}.npy'
                            .format(cohval, ztval, tival))
            mat_1 = np.load(SV_FOLDER + '/10M/mat1_coh{}_zt{}_ti{}.npy'
                            .format(cohval, ztval, tival))
            x = np.load(SV_FOLDER + '/10M/coh{}_zt{}_ti{}.npy'
                        .format(cohval, ztval, tival))
        trial_index = np.repeat(tival, num_simulations)
    # we load estimator
    # n_list = [10000, 50000, 250000]  # , 100000, 4000000]
    grid_rt = np.arange(-100, 300, 1) + 300
    grid_mt = np.arange(100, 600, 1)
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    mat_0_nn = np.empty((len(grid_mt), len(grid_rt)))
    mat_1_nn = np.copy(mat_0_nn)
    if plot_nn:
        for n_sim_train in n_list:
            with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_sim_train),
                      'rb') as f:
                estimator = pickle.load(f)
            estimator = estimator['estimator']
            theta = get_x0()
            theta = torch.reshape(torch.tensor(theta),
                                  (1, len(theta))).to(torch.float32)
            theta = theta.repeat(len(x_o), 1)
            theta[:, 0] *= torch.tensor(ztval)
            theta[:, 1] *= torch.tensor(cohval)
            theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                                torch.tensor(trial_index[
                                                    :len(x_o)]).to(torch.float32)))
            theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
            theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
            theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                                theta_tri_ind[:, 9:15]))
            lprobs = estimator.log_prob(x_o, theta_tri_ind)
            lprobs = torch.exp(lprobs)
            mat_0_nn = lprobs[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                      len(grid_rt)).detach().numpy()
            # reshaped_mat_0_nn = resize(mat_0_nn, mat_0.shape)*(1-eps) + eps*1e-6
            # reshaped_mat_1_nn = resize(mat_1_nn, mat_1.shape)*(1-eps) + eps*1e-6
            # kl_0 = get_manual_kl_divergence(mat_model=(mat_0*(1-eps) + eps*1e-6) /
            #                                 np.sum((mat_1+mat_0)*(1-eps) + eps*1e-6),
            #                                 mat_nn=reshaped_mat_0_nn /
            #                                 np.sum(reshaped_mat_0_nn+reshaped_mat_1_nn))
            mat_1_nn = lprobs[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                      len(grid_rt)).detach().numpy()
            # kl_1 = get_manual_kl_divergence(mat_model=(mat_1*(1-eps) + eps*1e-6) /
            #                                 np.sum((mat_1+mat_0)*(1-eps) + eps*1e-6),
            #                                 mat_nn=reshaped_mat_1_nn /
            #                                 np.sum(reshaped_mat_1_nn+reshaped_mat_0_nn))
            if plot_nn_alone:
                fig, ax1 = plt.subplots(ncols=2)
                fig.suptitle('Network + {}'.format(n_sim_train))
                cte_nn1 = np.sum(mat_0_nn + mat_1_nn)
                mat_0_nn /= cte_nn1
                mat_1_nn /= cte_nn1
                ax1[0].imshow(mat_0_nn, vmin=0, vmax=np.max((mat_0_nn, mat_1_nn)))
                ax1[0].set_title('Choice 0')
                ax1[0].set_yticks(np.arange(0, len(grid_mt), 50), grid_mt[::50])
                ax1[0].set_ylabel('MT (ms)')
                ax1[0].set_xticks(np.arange(0, len(grid_rt), 50), grid_rt[::50]-300)
                ax1[0].set_xlabel('RT (ms)')
                im1 = ax1[1].imshow(mat_1_nn, vmin=0, vmax=np.max((mat_0_nn, mat_1_nn)))
                ax1[1].set_title('Choice 1')
                ax1[1].set_yticks([])
                # ax1[1].set_ylabel('MT (ms)')
                ax1[1].set_xticks(np.arange(0, len(grid_rt), 50), grid_rt[::50]-300)
                ax1[1].set_xlabel('RT (ms)')
                plt.colorbar(im1)
                return
            # fig, ax = plt.subplots(ncols=2)
            # fig.suptitle('Model vs Network(contour) + {}'.format(n_sim_train))
            ax[0].imshow(resize(mat_0, mat_0_nn.shape), vmin=0, cmap='Blues')
            ax[0].contour(mat_0_nn, cmap='Reds', linewidths=0.8)
            ax[0].set_yticks(np.arange(0, len(grid_mt), 100), grid_mt[::100])
            ax[0].set_ylabel('MT (ms)')
            if xt:
                ax[0].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                                 rotation=45)
                ax[0].set_xlabel('RT (ms)')
            else:
                ax[0].set_xticks([])
            im1 = ax[1].imshow(resize(mat_1, mat_1_nn.shape), vmin=0, cmap='Blues')
            plt.sca(ax[1])
            im2 = ax[1].contour(mat_1_nn, cmap='Reds', linewidths=0.8)
            ax[1].set_yticks([])
            # ax[1].set_ylabel('MT (ms)')
            if xt:
                ax[1].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                                 rotation=45)
                ax[1].set_xlabel('RT (ms)')
            else:
                ax[1].set_xticks([])
            # plt.colorbar(im1, fraction=0.04)
            p_ch0_model = np.nanmean(x[:,2] == 0)
            p_fb_model = np.nanmean(x[:, 1] < 300)
            p_ch1_model = 1-p_ch0_model
            p_ch0_nn = np.nansum(mat_0_nn)/np.nansum(mat_0_nn+mat_1_nn)
            p_ch1_nn = np.nansum(mat_1_nn)/np.nansum(mat_0_nn+mat_1_nn)
            idx = grid_rt < 300
            p_fb_nn = (np.nansum(mat_0_nn[:, idx]) + np.nansum(mat_1_nn[:, idx])) /\
                np.nansum(mat_0_nn + mat_1_nn)
            ax[2].set_ylim(-0.05, 1.05)
            ax[2].bar(['Model', 'NN'],
                      [p_ch1_model, p_ch1_nn],
                      color=['cornflowerblue', 'firebrick'])
            ax[2].set_ylabel('p(Right)')
            ax[3].set_ylim(-0.02, 0.15)
            ax[3].bar(['Model', 'NN'],
                      [p_fb_model, p_fb_nn],
                      color=['cornflowerblue', 'firebrick'])
            ax[3].set_ylabel('p(FB)')
            # plt.colorbar(im2, fraction=0.04)
    if plot_model:
        fig, ax = plt.subplots(ncols=2)
        fig.suptitle('Model + coh {}, zt {}, t_ind {}'.format(cohval,
                                                              ztval, tival))
        ax[0].imshow(resize(mat_0, mat_0_nn.shape)
                     * len(x[x[:, 2] == 0])/len(x[:, 2]),
                     vmin=0, vmax=np.max((mat_0*len(x[x[:, 2] == 0])/len(x[:, 2]),
                                          mat_1*len(x[x[:, 2] == 1])/len(x[:, 2]))))
        ax[0].set_title('Choice 0')
        ax[0].set_yticks(np.arange(0, len(grid_mt), 50), grid_mt[::50])
        ax[0].set_ylabel('MT (ms)')
        ax[0].set_xticks(np.arange(0, len(grid_rt), 50), grid_rt[::50]-300)
        ax[0].set_xlabel('RT (ms)')
        im1 = ax[1].imshow(resize(mat_1, mat_1_nn.shape)*len(x[x[:, 2] == 1])
                           / len(x[:, 2]), vmin=0,
                           vmax=np.max((mat_0*len(x[x[:, 2] == 0])/len(x[:, 2]),
                                        mat_1*len(x[x[:, 2] == 1])/len(x[:, 2]))))
        ax[1].set_title('Choice 1')
        ax[1].set_yticks(np.arange(0, len(grid_mt), 50), grid_mt[::50])
        ax[1].set_ylabel('MT (ms)')
        ax[1].set_xticks(np.arange(0, len(grid_rt), 50), grid_rt[::50]-300)
        ax[1].set_xlabel('RT (ms)')
        plt.colorbar(im1)


def get_lprobs_nn(estimator, x_o, theta_tri_ind):
    lprobs1 = estimator.log_prob(x_o, theta_tri_ind)
    lprobs1 = torch.exp(lprobs1)
    return lprobs1


def get_theta_tri_ind(x_o, cohval, ztval, tival):
    theta = get_x0()
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(len(x_o), 1)
    trial_index = np.repeat(tival, len(theta))
    theta[:, 0] *= torch.tensor(ztval)
    theta[:, 1] *= torch.tensor(cohval)
    theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                        torch.tensor(trial_index[
                                            :len(x_o)]).to(torch.float32)))
    theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
    theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
    theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                        theta_tri_ind[:, 9:15]))
    return theta_tri_ind


def plot_nn_to_nn_comparison_model_diff(n_trials=[2000000, 10000000]):
    fig, axl = plt.subplots(nrows=6, ncols=5, figsize=(15, 9))
    axl = axl.flatten()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.95,
                        hspace=0.4, wspace=0.4)
    fig2, axr = plt.subplots(nrows=6, ncols=5, figsize=(15, 9))
    axr = axr.flatten()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.95,
                        hspace=0.4, wspace=0.4)
    axl[0].set_title('Left choice - Model')
    axl[1].set_title(str(n_trials[0]))
    axl[2].set_title('Diff. ' + str(n_trials[0]))
    axl[3].set_title('10M')
    axl[4].set_title('Diff. 10M')
    axr[0].set_title('Right choice - Model')
    axr[1].set_title(str(n_trials[0]))
    axr[2].set_title('Diff. ' + str(n_trials[0]))
    axr[3].set_title('10M')
    axr[4].set_title('Diff. 10M')
    # we load estimator
    grid_rt = np.arange(200, 600, 13)
    grid_rt = grid_rt[:-1] + np.diff(grid_rt)[0]/2
    grid_mt = np.arange(100, 600, 26)
    grid_mt = grid_mt[:-1] + np.diff(grid_mt)[0]/2
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_trials[0]),
              'rb') as f:
        estimator_1 = pickle.load(f)
    estimator_1 = estimator_1['estimator']
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_trials[1]),
              'rb') as f:
        estimator_2 = pickle.load(f)
    estimator_2 = estimator_2['estimator']
    ztvals = [1.5, 0.05, 1.5, -1.5, .5, .5]
    cohvals = [0, 1, 0.5, 0.5, 0.25, 0.25]
    tivals = [400, 400, 400, 400, 10, 800]
    p = 0
    mse_1 = []
    mse_2 = []
    for ztval, cohval, tival in zip(ztvals, cohvals, tivals):
        # load model matrices
        mat_0 = np.load(SV_FOLDER + '/10M/mat0_coh{}_zt{}_ti{}.npy'
                        .format(cohval, ztval, tival))
        mat_1 = np.load(SV_FOLDER + '/10M/mat1_coh{}_zt{}_ti{}.npy'
                        .format(cohval, ztval, tival))
        theta_tri_ind = get_theta_tri_ind(x_o, cohval, ztval, tival)
        lprobs1 = get_lprobs_nn(estimator=estimator_1, x_o=x_o,
                                theta_tri_ind=theta_tri_ind)
        mat_0_nn1 = lprobs1[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        mat_1_nn1 = lprobs1[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        lprobs2 = get_lprobs_nn(estimator=estimator_2, x_o=x_o,
                                theta_tri_ind=theta_tri_ind)
        mat_0_nn2 = lprobs2[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        mat_1_nn2 = lprobs2[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        cte_nn1 = np.sum(mat_0_nn1 + mat_1_nn1)
        mat_0_nn1 /= cte_nn1
        mat_1_nn1 /= cte_nn1
        cte_nn2 = np.sum(mat_0_nn2 + mat_1_nn2)
        mat_0_nn2 /= cte_nn2
        mat_1_nn2 /= cte_nn2
        cte_mod = np.sum(mat_0 + mat_1)
        mat_0 /= cte_mod
        mat_1 /= cte_mod
        # left
        max_im2 = np.max(mat_0_nn1 - mat_0)
        max_im1 = np.max(mat_0_nn2 - mat_0)
        vmax_0 = max(max_im2, max_im1)
        min_im2 = np.min(mat_0_nn1 - mat_0)
        min_im1 = np.min(mat_0_nn2 - mat_0)
        vmin_0 = max(min_im2, min_im1)
        im1 = axl[5*p].imshow(mat_0, cmap='Blues')
        im3 = axl[5*p+1].imshow(mat_0_nn1, cmap='Blues')
        im2 = axl[5*p+2].imshow(mat_0_nn1 - mat_0,
                                cmap='Blues', vmin=vmin_0, vmax=vmax_0)
        plt.colorbar(im2, ax=axl[5*p+2])
        im4 = axl[5*p+3].imshow(mat_0_nn2, cmap='Blues')
        im5 = axl[5*p+4].imshow(mat_0_nn2 - mat_0,
                                cmap='Blues', vmin=vmin_0, vmax=vmax_0)
        plt.colorbar(im5, ax=axl[5*p+4])
        # right
        max_im2 = np.max(mat_1_nn1 - mat_1)
        max_im1 = np.max(mat_1_nn2 - mat_1)
        vmax_1 = max(max_im2, max_im1)
        min_im2 = np.min(mat_1_nn1 - mat_1)
        min_im1 = np.min(mat_1_nn2 - mat_1)
        vmin_1 = max(min_im2, min_im1)
        axr[5*p].imshow(mat_1, cmap='Blues')
        axr[5*p+1].imshow(mat_1_nn1, cmap='Blues')
        im1 = axr[5*p+2].imshow(mat_1_nn1 - mat_1,
                                cmap='Blues', vmin=vmin_1, vmax=vmax_1)
        plt.colorbar(im1, ax=axr[5*p+2])
        axr[5*p+3].imshow(mat_1_nn2, cmap='Blues')
        im2 = axr[5*p+4].imshow(mat_1_nn2 - mat_1,
                               cmap='Blues', vmin=vmin_1, vmax=vmax_1)
        plt.colorbar(im2, ax=axr[5*p+4])
        mse_1.append(np.sum((mat_1_nn1 - mat_1_nn2)**2))
        mse_2.append(np.sum((mat_1_nn1 - mat_1)**2))
        # if p % 4 == 0:
        #     ax[p].set_ylabel('MT (ms)')
        #     ax[p].set_yticks(np.arange(0, len(grid_mt), 100), grid_mt[::100])
        #     ax[p+1].set_yticks([])
        # else:
        #     ax[p].set_yticks([])
        #     ax[p+1].set_yticks([])
        # if p >= 8:
        #     ax[p].set_xlabel('RT (ms)')
        #     ax[p+1].set_xlabel('RT (ms)')
        #     ax[p].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
        #                      rotation=45)
        #     ax[p+1].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
        #                        rotation=45)
        # else:
        #     ax[p].set_xticks([])
        #     ax[p+1].set_xticks([])
        p += 1


def plot_nn_to_nn_comparison(n_trials=10000000):
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
    ax = ax.flatten()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.95,
                        hspace=0.4, wspace=0.4)
    ax[1].set_title('Right choice')
    ax[0].set_title('Left choice')
    ax[3].set_title('Right choice')
    ax[2].set_title('Left choice')
    # we load estimator
    grid_rt = np.arange(-100, 300, 1) + 300
    grid_mt = np.arange(100, 600, 1)
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_trials),
              'rb') as f:
        estimator_1 = pickle.load(f)
    estimator_1 = estimator_1['estimator']
    with open(SV_FOLDER + "/mnle_n{}_no_noise_v2.p".format(n_trials),
              'rb') as f:
        estimator_2 = pickle.load(f)
    estimator_2 = estimator_2['estimator']
    ztvals = [1.5, 0.05, 1.5, -1.5, .5, .5]
    cohvals = [0, 1, 0.5, 0.5, 0.25, 0.25]
    tivals = [400, 400, 400, 400, 10, 800]
    p = 0
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g' , ' ']
    for ztval, cohval, tival in zip(ztvals, cohvals, tivals):
        if p <= 12:
            ax[p].text(-0.16, 1.2, letters[p // 2], transform=ax[p].transAxes,
                       fontsize=16, fontweight='bold', va='top', ha='right')
        pos_ax_1 = ax[p+1].get_position()
        ax[p+1].set_position([pos_ax_1.x0 - pos_ax_1.width/9,
                              pos_ax_1.y0, pos_ax_1.width,
                              pos_ax_1.height])
        theta = get_x0()
        theta = torch.reshape(torch.tensor(theta),
                              (1, len(theta))).to(torch.float32)
        theta = theta.repeat(len(x_o), 1)
        trial_index = np.repeat(tival, len(theta))
        theta[:, 0] *= torch.tensor(ztval)
        theta[:, 1] *= torch.tensor(cohval)
        theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                            torch.tensor(trial_index[
                                                :len(x_o)]).to(torch.float32)))
        theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
        theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
        theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                            theta_tri_ind[:, 9:15]))
        lprobs1 = estimator_1.log_prob(x_o, theta_tri_ind)
        lprobs1 = torch.exp(lprobs1)
        mat_0_nn1 = lprobs1[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        mat_1_nn1 = lprobs1[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        lprobs2 = estimator_2.log_prob(x_o, theta_tri_ind)
        lprobs2 = torch.exp(lprobs2)
        mat_0_nn2 = lprobs2[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        mat_1_nn2 = lprobs2[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                    len(grid_rt)).detach().numpy()
        im1 = ax[p].contour(mat_0_nn1, cmap='hot', linewidths=1.2)
        # plt.sca(ax[p])
        im2 = ax[p].contour(mat_0_nn2, cmap='cool', linewidths=1.2)
        im3 = ax[p+1].contour(mat_1_nn1, cmap='hot', linewidths=1.2)
        # plt.sca(ax[p+1])
        im4 = ax[p+1].contour(mat_1_nn2, cmap='cool', linewidths=1.2)
        if p % 4 == 0:
            ax[p].set_ylabel('MT (ms)')
            ax[p].set_yticks(np.arange(0, len(grid_mt), 100), grid_mt[::100])
            ax[p+1].set_yticks([])
        else:
            ax[p].set_yticks([])
            ax[p+1].set_yticks([])
        if p >= 8:
            ax[p].set_xlabel('RT (ms)')
            ax[p+1].set_xlabel('RT (ms)')
            ax[p].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                             rotation=45)
            ax[p+1].set_xticks(np.arange(0, len(grid_rt), 100), grid_rt[::100]-300,
                               rotation=45)
        else:
            ax[p].set_xticks([])
            ax[p+1].set_xticks([])
        pos_ax_12 = ax[p].get_position()
        ax[p].set_position([pos_ax_12.x0 + pos_ax_12.width/12,
                             pos_ax_12.y0, pos_ax_12.width, pos_ax_12.height])
        pos_ax_12 = ax[p+1].get_position()
        ax[p+1].set_position([pos_ax_12.x0 - pos_ax_12.width/7,
                             pos_ax_12.y0, pos_ax_12.width, pos_ax_12.height])
        p += 2
    ax[13].axis('off')
    ax[15].axis('off')
    pos_ax_12 = ax[12].get_position()
    ax[12].set_position([pos_ax_12.x0 + pos_ax_12.width/3,
                         pos_ax_12.y0-pos_ax_12.height/5, pos_ax_12.width*1.8,
                         pos_ax_12.height])
    pos_ax_12 = ax[14].get_position()
    ax[14].set_position([pos_ax_12.x0 + pos_ax_12.width/4,
                         pos_ax_12.y0-pos_ax_12.height/5, pos_ax_12.width*1.8,
                         pos_ax_12.height])
    supp_plot_dist_vs_n(ax=[ax[12], ax[14]])
    ax[12].text(-0.15, 1.2, letters[-2], transform=ax[12].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')


def plot_nn_to_nn_kldistance(n_trials=10000000):
    fig, ax = plt.subplots(nrows=3, ncols=2)
    ax = ax.flatten()
    ax[0].set_ylabel('stim - Low T.I.')
    ax[2].set_ylabel('stim - Medium T.I.')
    ax[4].set_ylabel('stim - High T.I.')
    ax[4].set_xlabel('prior')
    ax[5].set_xlabel('prior')
    # we load estimator
    grid_rt = np.arange(-100, 300, 5) + 300
    grid_mt = np.arange(100, 600, 5)
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    mat_0_nn = np.empty((len(grid_mt), len(grid_rt)))
    mat_1_nn = np.copy(mat_0_nn)
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_trials),
              'rb') as f:
        estimator_1 = pickle.load(f)
    estimator_1 = estimator_1['estimator']
    with open(SV_FOLDER + "/mnle_n{}_no_noise_v2.p".format(n_trials),
              'rb') as f:
        estimator_2 = pickle.load(f)
    estimator_2 = estimator_2['estimator']
    ztvals = np.linspace(-1.5, 1.5, 17)
    # ztvals = np.linspace(-1.5, 1.5, 9)
    # cohvals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    cohvals = np.linspace(-1, 1, 17)
    tivals = [10, 400, 800]
    p = 0
    for iti, tival in enumerate(tivals):
        mat_kl0 = np.empty((len(ztvals), len(cohvals)))
        mat_kl0[:] = np.nan
        mat_kl1 = np.empty((len(ztvals), len(cohvals)))
        mat_kl1[:] = np.nan
        for izt, ztval in enumerate(ztvals):
            for icoh, cohval in enumerate(cohvals):
                theta = get_x0()
                theta = torch.reshape(torch.tensor(theta),
                                      (1, len(theta))).to(torch.float32)
                theta = theta.repeat(len(x_o), 1)
                trial_index = np.repeat(tival, len(theta))
                theta[:, 0] *= torch.tensor(ztval)
                theta[:, 1] *= torch.tensor(cohval)
                theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                                    torch.tensor(trial_index[
                                                        :len(x_o)]).to(torch.float32)))
                theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
                theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
                theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                                    theta_tri_ind[:, 9:15]))
                lprobs1 = estimator_1.log_prob(x_o, theta_tri_ind)
                lprobs1 = torch.exp(lprobs1)
                mat_0_nn1 = lprobs1[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                            len(grid_rt)).detach().numpy()
                mat_0_nn1 /= np.sum(mat_0_nn1)
                mat_1_nn1 = lprobs1[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                            len(grid_rt)).detach().numpy()
                mat_1_nn1 /= np.sum(mat_1_nn1)
                lprobs2 = estimator_2.log_prob(x_o, theta_tri_ind)
                lprobs2 = torch.exp(lprobs2)
                mat_0_nn2 = lprobs2[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                            len(grid_rt)).detach().numpy()
                mat_0_nn2 /= np.sum(mat_0_nn2)
                mat_1_nn2 = lprobs2[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                            len(grid_rt)).detach().numpy()
                mat_1_nn2 /= np.sum(mat_1_nn2)
                kl_0 = get_manual_kl_divergence(mat_0_nn2, mat_0_nn1)
                kl_1 = get_manual_kl_divergence(mat_1_nn2, mat_1_nn1)
                mat_kl0[izt, icoh] = kl_0
                mat_kl1[izt, icoh] = kl_1
        im_1 = ax[2*iti].imshow(mat_kl0)
        plt.colorbar(im_1, fraction=0.04, label='KL-divergence')
        im_2 = ax[2*iti+1].imshow(mat_kl1)
        plt.colorbar(im_2, fraction=0.04, label='KL-divergence')
        ax[2*iti].set_yticks(np.arange(0, len(cohvals), 4), cohvals[::4])
        ax[2*iti].set_xticks(np.arange(0, len(ztvals), 4), ztvals[::4])
        ax[2*iti+1].set_xticks(np.arange(0, len(ztvals), 4), ztvals[::4])
        ax[2*iti+1].set_yticks(np.arange(0, len(cohvals), 4), cohvals[::4])


def rm_top_right_lines(ax, right=True):
    if right:
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)


def supp_plot_lh_model_network(df, n_trials=2000000):
    fig, ax = plt.subplots(6, 4, figsize=(8, 12))
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95,
                        hspace=0.4, wspace=0.4)
    col_labs = ['a', '', 'b', 'c']
    ax = ax.flatten()
    row_labs = ['i', 'ii', 'iii', 'iv', 'v', 'vi']
    for i in range(4):
        ax[i].text(-0.65, 1.45, col_labs[i], transform=ax[i].transAxes,
                   fontsize=12, fontweight='bold', va='top', ha='right')
    for j in range(6):
        pos_ax_1 = ax[4*j].get_position()
        ax[4*j].set_position([pos_ax_1.x0,
                              pos_ax_1.y0, pos_ax_1.width*1.1, pos_ax_1.height])
        pos_ax_1 = ax[4*j+1].get_position()
        ax[4*j+1].set_position([pos_ax_1.x0 - pos_ax_1.width/4,
                            pos_ax_1.y0, pos_ax_1.width*1.1, pos_ax_1.height])
        pos_ax_1 = ax[4*j+2].get_position()
        ax[4*j+2].set_position([pos_ax_1.x0,
                                pos_ax_1.y0, pos_ax_1.width*0.7,
                                pos_ax_1.height])
        rm_top_right_lines(ax[4*j+2])
        pos_ax_1 = ax[4*j+3].get_position()
        ax[4*j+3].set_position([pos_ax_1.x0,
                                pos_ax_1.y0, pos_ax_1.width*0.7,
                                pos_ax_1.height])
        rm_top_right_lines(ax[4*j+3])
        ax[4*j].text(-0.4, 1.3, row_labs[j], transform=ax[4*j].transAxes,
                     fontsize=12, fontweight='bold', va='top', ha='right')
    i = 0
    xt = False
    ax[1].set_title('Right choice', pad=14, fontsize=11)
    ax[0].set_title('Left choice', pad=14, fontsize=11)
    for cohval, ztval, tival in zip([0, 1, 0.5, 0.25, 0.5, 0.25],
                                    [1.5, 0.05, -1.5, .5, 1.5, 0.5],
                                    [400, 400, 400, 10, 400, 800]):
        if i == 5:
            xt = True
        plot_network_model_comparison(df, ax[4*i:4*(i+1)],
                                      sv_folder=SV_FOLDER, num_simulations=int(5e5),
                                      n_list=[n_trials], cohval=cohval,
                                      ztval=ztval, tival=tival,
                                      plot_nn=True, simulate=False, plot_model=False,
                                      plot_nn_alone=False, xt=xt)
        # ax[3*i].set_title(labels[i])
        i += 1


def bhatt_dist(p, q):
    return -np.log(np.sum(np.sqrt(p*q)))


def supp_plot_dist_vs_n(ax, n_list=[1000, 10000, 100000, 500000, 1000000, 2000000,
                                    4000000, 10000000]):
    cohvals = np.load(SV_FOLDER + '/10M/100_sims/cohvals.npy', allow_pickle=True)
    ztvals = np.load(SV_FOLDER + '/10M/100_sims/ztvals.npy', allow_pickle=True)
    tivals = np.load(SV_FOLDER + '/10M/100_sims/tivals.npy', allow_pickle=True)
    bhat_mat = np.zeros((len(cohvals), len(n_list)))
    js_mat = np.zeros((len(cohvals), len(n_list)))
    for i_n, n_trial in enumerate(n_list):
        i = 0
        for cohval, ztval, tival in zip(cohvals, ztvals, tivals):
            bhat, jens_shan = mse_lh_model_nn(n_trial, cohval, ztval,
                                              tival, num_simulations=int(5e5))
            bhat_mat[i, i_n] = bhat
            js_mat[i, i_n] = jens_shan
            i += 1
        # mse_mat[:, i_n] = (mse_mat[:, i_n] - np.mean(mse_mat[:, i_n])) /\
        #     (np.max(mse_mat[:, i_n])-np.min(mse_mat[:, i_n]))
        # mse_mat[:, i_n] = mse_mat[:, i_n] / np.max(mse_mat[:, i_n])
    # fig, ax = plt.subplots(ncols=2, figsize=(8, 5))
    # ax = ax.flatten()
    for a in ax:
        a.set_xscale('log')
        rm_top_right_lines(a)
        a.set_xlabel('N trials for training')
    # mse_mat_norm_max = np.copy(bhat_mat)
    # for j in range(100):
    #     # mse_mat_norm_max[j, :] /= np.max(mse_mat_norm_max[j, :])
    #     ax[0].plot(n_list,  bhat_mat[j, :] , color='r', alpha=0.02)
    #     ax[1].plot(n_list,  js_mat[j, :] , color='r', alpha=0.02)
    mean_bhat = np.nanmean(bhat_mat, axis=0)
    mean_js = np.nanmean(js_mat, axis=0)
    err_bhat = np.nanstd(bhat_mat, axis=0)/10
    err_js = np.nanstd(js_mat, axis=0)/10
    ax[0].plot(n_list, mean_bhat, linewidth=2, color='r')
    ax[1].plot(n_list, mean_js, linewidth=2, color='r')
    ax[0].fill_between(n_list, mean_bhat-err_bhat, mean_bhat+err_bhat, color='r',
                       alpha=0.2)
    ax[1].fill_between(n_list, mean_js-err_js, mean_js+err_js, color='r',
                       alpha=0.2)
    # ax.set_ylabel(r'KL divergence, $D(x,y)+D(y,x)$')
    # ax.set_ylabel('MSE(NN, model)')
    ax[0].set_ylabel('Bhattacharyya \n distance')
    ax[1].set_ylabel('Jensen-Shannon \n distance')


def mse_lh_model_nn(n_sim_train, cohval, ztval, tival, num_simulations=int(5e5)):
    # we load estimator
    grid_rt = np.arange(200, 600, 13)
    grid_rt = grid_rt[:-1] + np.diff(grid_rt)[0]/2
    grid_mt = np.arange(100, 600, 26)
    grid_mt = grid_mt[:-1] + np.diff(grid_mt)[0]/2
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    trial_index = np.repeat(tival, num_simulations)
    mat_0 = np.load(SV_FOLDER + '/10M/100_sims/mat0_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival))
    mat_1 = np.load(SV_FOLDER + '/10M/100_sims/mat1_coh{}_zt{}_ti{}.npy'
                    .format(cohval, ztval, tival))
    with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_sim_train),
              'rb') as f:
        estimator = pickle.load(f)
    estimator = estimator['estimator']
    theta = get_x0()
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(len(x_o), 1)
    theta[:, 0] *= torch.tensor(ztval)
    theta[:, 1] *= torch.tensor(cohval)
    theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                        torch.tensor(trial_index[
                                            :len(x_o)]).to(torch.float32)))
    theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
    theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
    theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                        theta_tri_ind[:, 9:15]))
    lprobs = estimator.log_prob(x_o, theta_tri_ind)
    lprobs = torch.exp(lprobs)
    mat_0_nn = lprobs[x_o[:, 2] == 0].reshape(len(grid_mt),
                                              len(grid_rt)).detach().numpy()
    mat_1_nn = lprobs[x_o[:, 2] == 1].reshape(len(grid_mt),
                                              len(grid_rt)).detach().numpy()
    cte_nn = np.sum(mat_0_nn + mat_1_nn)
    mat_0_nn /= cte_nn
    mat_1_nn /= cte_nn
    cte_mod = np.sum(mat_0 + mat_1)
    mat_0 /= cte_mod
    mat_1 /= cte_mod
    mse_0 = np.sum((mat_0_nn - mat_0)**2)
    mse_1 = np.sum((mat_1_nn - mat_1)**2)
    mat_model = np.array((((mat_0), (mat_1))))
    mat_nn = np.array((((mat_0_nn), (mat_1_nn))))
    # kl0 = get_manual_kl_divergence(mat_0*(1-1e-3)+1e-9, mat_0_nn*(1-1e-3)+1e-9)
    # kl1 = get_manual_kl_divergence(mat_1*(1-1e-3)+1e-9, mat_1_nn*(1-1e-3)+1e-9)
    # mat_0_nn_smooth = convolve2d(mat_0_nn, np.ones((5, 5)))
    # mat_1_nn_smooth = convolve2d(mat_1_nn, np.ones((5, 5)))
    # mat_0_smooth = convolve2d(mat_0, np.ones((5, 5)))
    # mat_1_smooth = convolve2d(mat_1, np.ones((5, 5)))
    # cte_nn = np.sum(mat_0_nn_smooth + mat_1_nn_smooth)
    # mat_0_nn_smooth /= cte_nn
    # mat_1_nn_smooth /= cte_nn
    # cte_mod = np.sum(mat_0_smooth + mat_1_smooth)
    # mat_0_smooth /= cte_mod
    # mat_1_smooth /= cte_mod
    # mat_model = np.array((((mat_0_smooth), (mat_1_smooth))))
    # mat_nn = np.array((((mat_0_nn_smooth), (mat_1_nn_smooth))))
    # kl_all_1 = get_manual_kl_divergence(mat_model, mat_nn)
    # kl_all_2 = get_manual_kl_divergence(mat_nn*(1-1e-3)+1e-9, mat_model*(1-1e-3)+1e-9)
    estimator = []
    return bhatt_dist(mat_model, mat_nn), np.nansum(dist.jensenshannon(mat_model, mat_nn))


def kl_vs_n_trials(df, n_trials=[2000000, 3000000, 4000000], sv_folder=SV_FOLDER):
    fig, ax = plt.subplots(nrows=4)
    i = 0
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    ztvals = np.quantile(zt, [(i+1)*1/6 for i in range(5)])
    cohvals = np.array((-1, -0.5, -0.25, 0., 0.25, 0.5, 1))
    combinations = list(itertools.product(ztvals, cohvals))
    tival = 350
    mat_kl_mt0 = np.zeros((len(n_trials), len(combinations)))
    mat_kl_mt1 = np.zeros((len(n_trials), len(combinations)))
    mat_kl_rt0 = np.zeros((len(n_trials), len(combinations)))
    mat_kl_rt1 = np.zeros((len(n_trials), len(combinations)))
    for ztval, cohval in combinations:
        trial_index = np.repeat(tival, int(5e5))
        x = np.load(SV_FOLDER + 'coh{}_zt{}_ti{}.npy'.format(cohval, ztval, tival))
        ch_model = x[:, 2]
        mt_model = x[:, 0]
        bins_rt = np.arange(-100, 301, 20) + 300
        bins_mt = np.arange(100, 601, 5)
        mt_model_distro0 = np.histogram(mt_model[ch_model == 0], density=True, bins=bins_mt)[0]
        mt_model_distro0 /= sum(mt_model_distro0)
        mt_model_distro1 = np.histogram(mt_model[ch_model == 1], density=True, bins=bins_mt)[0]
        mt_model_distro1 /= sum(mt_model_distro1)
        rt_model = x[:, 1]
        rt_model_distro0 = np.histogram(rt_model[ch_model == 0], density=True, bins=bins_rt)[0]
        rt_model_distro0 /= sum(rt_model_distro0)
        rt_model_distro1 = np.histogram(rt_model[ch_model == 1], density=True, bins=bins_rt)[0]
        rt_model_distro1 /= sum(rt_model_distro1)
        grid_rt = bins_rt[:-1] + np.diff(bins_rt)[0]/2
        grid_mt = bins_mt[:-1] + np.diff(bins_mt)[0]/2
        all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
        all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
        comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
        comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
        # generated data
        x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
        colors = ['k', 'r']
        for i_n, n_sim_train in enumerate(n_trials):
            with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_sim_train),
                      'rb') as f:
                estimator = pickle.load(f)
            estimator = estimator['estimator']
            theta = theta_for_lh_plot()
            theta = torch.reshape(torch.tensor(theta),
                                  (1, len(theta))).to(torch.float32)
            theta = theta.repeat(len(x_o), 1)
            theta[:, 0] *= torch.tensor(ztval)
            theta[:, 1] *= torch.tensor(cohval)
            theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                                torch.tensor(trial_index[
                                                    :len(x_o)]).to(torch.float32)))
            theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
            theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
            theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                                theta_tri_ind[:, 9:15]))
            lprobs = estimator.log_prob(x_o, theta_tri_ind)
            lprobs = torch.exp(lprobs)
            mat_0_nn = lprobs[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                      len(grid_rt)).detach().numpy()
            mat_1_nn = lprobs[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                      len(grid_rt)).detach().numpy()
            mt_nn_distro0 = np.nansum(mat_0_nn, axis=1)
            mt_nn_distro0 /= sum(mt_nn_distro0)
            mt_nn_distro1 = np.nansum(mat_1_nn, axis=1)
            mt_nn_distro1 /= sum(mt_nn_distro1)
            rt_nn_distro0 = np.nansum(mat_0_nn, axis=0)
            rt_nn_distro0 /= sum(rt_nn_distro0)
            rt_nn_distro1 = np.nansum(mat_1_nn, axis=0)
            rt_nn_distro1 /= sum(rt_nn_distro1)
            kl_div_mt0 = sum(rel_entr(mt_model_distro0, mt_nn_distro0))
            kl_div_mt1 = sum(rel_entr(mt_model_distro1, mt_nn_distro1))
            kl_div_rt0 = sum(rel_entr(rt_model_distro0, rt_nn_distro0))
            kl_div_rt1 = sum(rel_entr(rt_model_distro1, rt_nn_distro1))
            mat_kl_mt0[i_n, i] = kl_div_mt0
            mat_kl_mt1[i_n, i] = kl_div_mt1
            mat_kl_rt0[i_n, i] = kl_div_rt0
            mat_kl_rt1[i_n, i] = kl_div_rt1
        i += 1
        for j in range(4):
            ax[0].plot(n_trials, mat_kl_mt0[:, j], color='gray', alpha=0.05)
            ax[1].plot(n_trials, mat_kl_mt1[:, j], color='gray', alpha=0.05)
            ax[2].plot(n_trials, mat_kl_rt0[:, j], color='gray', alpha=0.05)
            ax[3].plot(n_trials, mat_kl_rt1[:, j], color='gray', alpha=0.05)
    for a in range(len(ax)):
        ax[a].set_ylabel('KL divergence')
    ax[0].errorbar(n_trials, np.nanmean(mat_kl_mt0, axis=1),
                   yerr=np.nanstd(mat_kl_mt0, axis=1), marker='o', color='k',
                   lw=2)
    ax[1].errorbar(n_trials, np.nanmean(mat_kl_mt1, axis=1),
                   yerr=np.nanstd(mat_kl_mt1, axis=1), marker='o', color='k',
                   lw=2)
    ax[2].errorbar(n_trials, np.nanmean(mat_kl_rt0, axis=1),
                   yerr=np.nanstd(mat_kl_rt0, axis=1), marker='o', color='k',
                   lw=2)
    ax[3].errorbar(n_trials, np.nanmean(mat_kl_rt1, axis=1),
                   yerr=np.nanstd(mat_kl_rt1, axis=1), marker='o', color='k',
                   lw=2)
    ax[3].set_xlabel('N trials')


def plot_kl_vs_n_trials(df, cohval, ztval, tival, sv_folder=SV_FOLDER,
                        n_trials=[int(2e6), int(3e6), int(4e6)], simulate=False):
    fig, ax = plt.subplots(ncols=2, nrows=4)
    ax = ax.flatten()
    if simulate:
        stim = np.array(
            [stim for stim in df.res_sound])[df.coh2.values == cohval][0]
        theta = get_x0()
        theta = torch.reshape(torch.tensor(theta),
                              (1, len(theta))).to(torch.float32)
        theta = theta.repeat(num_simulations, 1)
        stim = np.array(
            [np.concatenate((stim, stim)) for i in range(len(theta))])
        trial_index = np.repeat(tival, len(theta))
        x = simulations_for_mnle(theta_all=np.array(theta), stim=stim,
                                 zt=np.repeat(ztval, len(theta)),
                                 coh=np.repeat(cohval, len(theta)),
                                 trial_index=trial_index)
        np.save(SV_FOLDER + 'coh{}_zt{}_ti{}.npy'
                .format(cohval, ztval, tival), x)
    else:
        trial_index = np.repeat(tival, int(5e5))
        x = np.load(SV_FOLDER + 'coh{}_zt{}_ti{}.npy'.format(cohval, ztval, tival))
    ch_model = x[:, 2]
    mt_model = x[:, 0]
    bins_rt = np.arange(-100, 301, 20) + 300
    bins_mt = np.arange(100, 601, 5)
    mt_model_distro0 = np.histogram(mt_model[ch_model == 0], density=True, bins=bins_mt)[0]
    mt_model_distro0 /= sum(mt_model_distro0)
    mt_model_distro1 = np.histogram(mt_model[ch_model == 1], density=True, bins=bins_mt)[0]
    mt_model_distro1 /= sum(mt_model_distro1)
    rt_model = x[:, 1]
    rt_model_distro0 = np.histogram(rt_model[ch_model == 0], density=True, bins=bins_rt)[0]
    rt_model_distro0 /= sum(rt_model_distro0)
    rt_model_distro1 = np.histogram(rt_model[ch_model == 1], density=True, bins=bins_rt)[0]
    rt_model_distro1 /= sum(rt_model_distro1)
    grid_rt = bins_rt[:-1] + np.diff(bins_rt)[0]/2
    grid_mt = bins_mt[:-1] + np.diff(bins_mt)[0]/2
    ax[0].plot(grid_mt, mt_model_distro0, color='b', label='Model')
    ax[2].plot(grid_mt, mt_model_distro1, color='b', label='Model')
    ax[4].plot(grid_rt, rt_model_distro0, color='b', label='Model')
    ax[6].plot(grid_rt, rt_model_distro1, color='b', label='Model')
    all_rt = np.meshgrid(grid_rt, grid_mt)[0].flatten()
    all_mt = np.meshgrid(grid_rt, grid_mt)[1].flatten()
    comb_0 = np.column_stack((all_mt, all_rt, np.repeat(0, len(all_mt))))
    comb_1 = np.column_stack((all_mt, all_rt, np.repeat(1, len(all_mt))))
    # generated data
    x_o = torch.tensor(np.concatenate((comb_0, comb_1))).to(torch.float32)
    colors = ['k', 'r']
    for i_n, n_sim_train in enumerate(n_trials):
        with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_sim_train),
                  'rb') as f:
            estimator = pickle.load(f)
        estimator = estimator['estimator']
        theta = theta_for_lh_plot()
        theta = torch.reshape(torch.tensor(theta),
                              (1, len(theta))).to(torch.float32)
        theta = theta.repeat(len(x_o), 1)
        theta[:, 0] *= torch.tensor(ztval)
        theta[:, 1] *= torch.tensor(cohval)
        theta_tri_ind = torch.column_stack((theta[:len(x_o)],
                                            torch.tensor(trial_index[
                                                :len(x_o)]).to(torch.float32)))
        theta_tri_ind[:, 14] += theta_tri_ind[:, 15]*theta_tri_ind[:, -1]
        theta_tri_ind[:, 7] -= theta_tri_ind[:, 8]*theta_tri_ind[:, -1]
        theta_tri_ind = torch.column_stack((theta_tri_ind[:, :8],
                                            theta_tri_ind[:, 9:15]))
        lprobs = estimator.log_prob(x_o, theta_tri_ind)
        lprobs = torch.exp(lprobs)
        mat_0_nn = lprobs[x_o[:, 2] == 0].reshape(len(grid_mt),
                                                  len(grid_rt)).detach().numpy()
        mat_1_nn = lprobs[x_o[:, 2] == 1].reshape(len(grid_mt),
                                                  len(grid_rt)).detach().numpy()
        mt_nn_distro0 = np.nansum(mat_0_nn, axis=1)
        mt_nn_distro0 /= sum(mt_nn_distro0)
        ax[0].plot(grid_mt, mt_nn_distro0, color=colors[i_n], label=n_sim_train)
        mt_nn_distro1 = np.nansum(mat_1_nn, axis=1)
        mt_nn_distro1 /= sum(mt_nn_distro1)
        ax[2].plot(grid_mt, mt_nn_distro1, color=colors[i_n])
        rt_nn_distro0 = np.nansum(mat_0_nn, axis=0)
        rt_nn_distro0 /= sum(rt_nn_distro0)
        ax[4].plot(grid_rt, rt_nn_distro0, color=colors[i_n])
        rt_nn_distro1 = np.nansum(mat_1_nn, axis=0)
        rt_nn_distro1 /= sum(rt_nn_distro1)
        ax[6].plot(grid_rt, rt_nn_distro1, color=colors[i_n])
        kl_div_mt0 = sum(rel_entr(mt_model_distro0, mt_nn_distro0))
        kl_div_mt1 = sum(rel_entr(mt_model_distro1, mt_nn_distro1))
        kl_div_rt0 = sum(rel_entr(rt_model_distro0, rt_nn_distro0))
        kl_div_rt1 = sum(rel_entr(rt_model_distro1, rt_nn_distro1))
        ax[1].plot(n_sim_train, kl_div_mt0, 'o')
        ax[3].plot(n_sim_train, kl_div_mt1, 'o')
        ax[5].plot(n_sim_train, kl_div_rt0, 'o')
        ax[7].plot(n_sim_train, kl_div_rt1, 'o')
    ax[0].legend()
    ax[1].set_title('mt_0')
    ax[3].set_title('mt_1')
    ax[5].set_title('rt_0')
    ax[7].set_title('rt_1')
    ax[0].set_xlabel('MT, ch=0')
    ax[2].set_xlabel('MT, ch=1')
    ax[4].set_xlabel('RT, ch=0')
    ax[6].set_xlabel('RT, ch=1')


def mnle_sample_simulation(df, theta=theta_for_lh_plot(), num_simulations=int(1e6),
                           n_simul_training=int(3e6), vers_theta='2'):
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    stim = np.array([stim for stim in df.res_sound])
    coh = np.array(df.coh2)
    trial_index = np.array(df.origidx)
    stim[df.soundrfail, :] = 0
    # Prepare data:
    coh = np.resize(coh, num_simulations)
    zt = np.resize(zt, num_simulations)
    trial_index = np.resize(trial_index, num_simulations)
    # stim = np.resize(stim, (num_simulations, 20))
    theta_all = torch.tensor(theta).repeat(num_simulations, 1)
    theta_all_inp = theta_all.clone().detach()
    theta_all_inp[:, 0] *= torch.tensor(zt[:num_simulations]).to(torch.float32)
    theta_all_inp[:, 1] *= torch.tensor(coh[:num_simulations]).to(torch.float32)
    theta_all_inp = torch.column_stack((
        theta_all_inp, torch.tensor(
            trial_index[:num_simulations].astype(float)).to(torch.float32)))
    theta_all_inp = theta_all_inp.to(torch.float32)
    # SIMULATION
    x = simulations_for_mnle(theta_all, stim, zt, coh, trial_index)
    # now we have a matrix of (num_simulations x 3):
    # MT, RT, CHOICE for each simulation

    # NETWORK TRAINING
    # transform parameters related to trial index. 14 params instead of 17
    # MT_in = MT_0 + MT_1*trial_index
    theta_all_inp[:, 14] += theta_all_inp[:, 15]*theta_all_inp[:, -1]
    # V_A = vA_0 - vA_1*trial_index
    theta_all_inp[:, 7] -= theta_all_inp[:, 8]*theta_all_inp[:, -1]
    theta_all_inp = torch.column_stack((theta_all_inp[:, :8],
                                        theta_all_inp[:, 9:15]))
    x_nn = get_sampled_data_mnle(theta=theta_all_inp,
                                 n_simul_training=n_simul_training,
                                 cohztti='{}_ALL_RAT_{}_{}'.format(n_simul_training,
                                                                df.subjid.unique(),
                                                                vers_theta),
                                 new_data=False)
    fig, ax = plt.subplots(nrows=3)
    fig.suptitle('Training trials: ' + str(n_simul_training))
    # sns.kdeplot(df.resp_len.values*1e3, label='Data', ax=ax[0], color='k')
    sns.kdeplot(x_nn[:, 0], label='MNLE', ax=ax[0], color='r')
    sns.kdeplot(x[:, 0], label='Model', ax=ax[0], color='b')
    ax[0].set_xlabel('MT (ms)')
    ax[0].legend()
    # sns.kdeplot(df.sound_len.values, label='Data', ax=ax[1], color='k')
    sns.kdeplot(x_nn[:, 1]-300, label='MNLE', ax=ax[1], color='r')
    sns.kdeplot(x[:, 1]-300, label='Model', ax=ax[1], color='b')
    ax[1].set_xlabel('RT (ms)')
    ax[1].legend()
    # pright_data = []
    pright_nn = []
    pright_model = []
    ev_vals = np.unique(coh)
    for ev in ev_vals:
        index = coh == ev
        # pright_data.append(np.nanmean(df.R_response.values[index]))
        pright_nn.append(np.nanmean(x_nn[:, 2][index]))
        pright_model.append(np.nanmean(x[:, 2][index]))
    # ax[2].plot(ev_vals, pright_data, label='Data', color='k')
    ax[2].plot(ev_vals, pright_nn, marker='o', label='MNLE', color='r')
    ax[2].plot(ev_vals, pright_model, marker='o', label='Model', color='b')
    ax[2].set_xlabel('Stimulus')
    ax[2].set_ylabel('Pright')
    ax[2].legend()


def create_simulations_mt_rt_choice(df, cohval, tival, ztval, theta,
                                    num_simulations=int(1e5)):
    stim = np.array(
        [stim for stim in df.res_sound])[df.coh2.values == cohval][0]
    theta = torch.reshape(torch.tensor(theta),
                          (1, len(theta))).to(torch.float32)
    theta = theta.repeat(num_simulations, 1)
    stim = np.array(
        [np.concatenate((stim, stim)) for i in range(len(theta))])
    trial_index = np.repeat(tival, len(theta))
    simul_data = SV_FOLDER + 'coh{}_zt{}_ti{}.npy'.format(cohval, ztval, tival)
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(simul_data), exist_ok=True)
    if os.path.exists(simul_data):
        print('Data already exists')
        # x = np.load(simul_data, allow_pickle=True)
        # x = torch.tensor(x).to(torch.float32)
    else:
        x = simulations_for_mnle(theta_all=np.array(theta), stim=stim,
                                 zt=np.repeat(ztval, len(theta)),
                                 coh=np.repeat(cohval, len(theta)),
                                 trial_index=trial_index,
                                 simulate=True)
        np.save(SV_FOLDER + 'coh{}_zt{}_ti{}.npy'
                .format(cohval, ztval, tival), x)


def compute_simulations_diff_zt_coh(df, theta=theta_for_lh_plot()):
    tival = 350
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    ztvals = np.quantile(zt, [(i+1)*1/6 for i in range(5)])
    cohvals = np.array((-1, -0.5, -0.25, 0., 0.25, 0.5, 1))
    combinations = list(itertools.product(ztvals, cohvals))
    for ztval, cohval in combinations:
        create_simulations_mt_rt_choice(df=df, cohval=cohval,
                                        tival=tival, ztval=ztval, theta=theta)

def get_sampled_data_mnle(theta, n_simul_training, cohztti, new_data=False):
    # load NN
    simul_data = SV_FOLDER + 'simul_nn_{}.npy'.format(cohztti)
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(simul_data), exist_ok=True)
    if os.path.exists(simul_data) and not new_data:
        # print('Data already exists')
        x_nn = np.load(simul_data, allow_pickle=True)
        # x = torch.tensor(x).to(torch.float32)
    else:
        with open(SV_FOLDER + "/mnle_n{}_no_noise.p".format(n_simul_training),
                  'rb') as f:
            estimator = pickle.load(f)
        estimator = estimator['estimator']
        x_nn = torch.tensor(())
        print('Sampling from network')
        for i_th, th in enumerate(theta):
            if i_th % 10000 == 0 and i_th != 0:
                print('Sampling the ' + str(i_th+1) + 'th trial')
            x_sample_tmp = estimator.sample(th.reshape(1, -1))
            while x_sample_tmp[0][0] > 2000 or x_sample_tmp[0][1] > 2000:
                x_sample_tmp = estimator.sample(th.reshape(1, -1))
            x_nn = torch.cat((x_nn, x_sample_tmp))
        x_nn = x_nn.detach().numpy()
        np.save(SV_FOLDER + 'simul_nn_{}.npy'.format(cohztti), x_nn)
    return x_nn


def plot_kl_vs_zt_coh(df, theta=theta_for_lh_plot(), num_simulations=int(1e5),
                      n_simul_training=int(3e6)):
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    ztvals = np.quantile(zt, [(i+1)*1/6 for i in range(5)])
    cohvals = np.array((-1, -0.5, -0.25, 0., 0.25, 0.5, 1))
    combinations = list(itertools.product(ztvals, cohvals))
    tival = 350
    bins_rt = np.arange(-100, 301, 75) + 300
    bins_mt = np.arange(250, 601, 75)
    kl_mt = []
    kl_rt = []
    kl_ch = []
    for ztval, cohval in combinations:
        # load simulated data
        try:
            x = np.load(SV_FOLDER + 'coh{}_zt{}_ti{}.npy'
                        .format(cohval, ztval, tival))
        except:
            x = np.empty((num_simulations, 3))
            x[:] = np.nan
        mt_model = x[:, 0]
        rt_model = x[:, 1]
        ch_model = x[:, 2]
        # Prepare data:
        coh = np.resize(cohval, num_simulations)
        zt = np.resize(ztval, num_simulations)
        trial_index = np.resize(tival, num_simulations)
        theta_all = torch.tensor(theta).repeat(num_simulations, 1)
        theta_all_inp = theta_all.clone().detach()
        theta_all_inp[:, 0] *= torch.tensor(zt[:num_simulations]).to(torch.float32)
        theta_all_inp[:, 1] *= torch.tensor(coh[:num_simulations]).to(torch.float32)
        theta_all_inp = torch.column_stack((
            theta_all_inp, torch.tensor(
                trial_index[:num_simulations].astype(float)).to(torch.float32)))
        theta_all_inp = theta_all_inp.to(torch.float32)
        theta_all_inp[:, 14] += theta_all_inp[:, 15]*theta_all_inp[:, -1]
        theta_all_inp[:, 7] -= theta_all_inp[:, 8]*theta_all_inp[:, -1]
        theta_all_inp = torch.column_stack((theta_all_inp[:, :8],
                                            theta_all_inp[:, 9:15]))
        x_nn = get_sampled_data_mnle(theta=theta_all_inp,
                                     n_simul_training=n_simul_training,
                                     cohztti=\
                                         str(round(ztval, 2))+str(cohval)+str(tival))
        mt_nn = x_nn[:, 0]
        rt_nn = x_nn[:, 1]
        ch_nn = x_nn[:, 2]
        mt_model_distro = np.histogram(mt_model, density=True, bins=bins_mt)[0]
        mt_model_distro /= sum(mt_model_distro)
        mt_nn_distro = np.histogram(mt_nn, density=True, bins=bins_mt)[0]
        mt_nn_distro /= sum(mt_nn_distro)
        kl_mt.append(sum(rel_entr(mt_model_distro, mt_nn_distro)))
        rt_model_distro = np.histogram(rt_model, density=True, bins=bins_rt)[0]
        rt_model_distro /= sum(rt_model_distro)
        rt_nn_distro = np.histogram(rt_nn, density=True, bins=bins_rt)[0]
        rt_nn_distro /= sum(rt_nn_distro)
        kl_rt.append(sum(rel_entr(rt_model_distro, rt_nn_distro)))
        ch_model_distro = [np.mean(ch_model == 0), np.mean(ch_model == 1)]
        ch_nn_distro = [np.mean(ch_nn == 0), np.mean(ch_nn == 1)]
        kl_ch.append(sum(rel_entr(ch_model_distro, ch_nn_distro)))
    fig, ax = plt.subplots(ncols=3)
    mat_kl_mt = np.zeros((len(ztvals), len(cohvals)))
    mat_kl_rt = np.zeros((len(ztvals), len(cohvals)))
    mat_kl_ch = np.zeros((len(ztvals), len(cohvals)))
    for i, [ztval, cohval] in enumerate(combinations):
        indexzt = ztvals == ztval
        indexcoh = cohvals == cohval
        mat_kl_mt[indexzt, indexcoh] = kl_mt[i]
        mat_kl_rt[indexzt, indexcoh] = kl_rt[i]
        mat_kl_ch[indexzt, indexcoh] = kl_ch[i]
    titles = ['MT', 'RT', 'choice']
    for j in range(len(ax)):
        ax[j].set_xticks([0, 3, 6], [-1, 0, 1])
        ax[j].set_yticks([])
        ax[j].set_xlabel('Stimulus')
        ax[j].set_title(titles[j])
    ax[0].set_yticks([0, 2, 4], [1, 0, -1])
    ax[0].set_ylabel('Prior')
    immt = ax[0].imshow(np.flipud(mat_kl_mt))
    plt.colorbar(immt, fraction=0.04, label='KL-divergence')
    imrt = ax[1].imshow(np.flipud(mat_kl_rt))
    plt.colorbar(imrt, fraction=0.04, label='KL-divergence')
    imch = ax[2].imshow(np.flipud(mat_kl_ch))
    plt.colorbar(imch, fraction=0.04, label='KL-divergence')


def get_human_data(user_id, sv_folder=SV_FOLDER, nm='300'):
    if user_id == 'alex':
        folder = 'C:\\Users\\alexg\\Onedrive\\Escritorio\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'alex_CRM':
        folder = 'C:/Users/agarcia/Desktop/CRM/human/'
    if user_id == 'idibaps':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    if user_id == 'idibaps_alex':
        folder = '/home/jordi/DATA/Documents/changes_of_mind/humans/'+nm+'ms/'
    subj = ['general_traj_all']
    steps = [None]
    # retrieve data
    df = ah.traj_analysis(data_folder=folder,
                          subjects=subj, steps=steps, name=nm,
                          sv_folder=sv_folder)
    return df


def human_fitting(df, subject, sv_folder=SV_FOLDER,  num_simulations=int(10e6)):
    df_data = df.loc[df.subjid == subject]
    reac_time = df_data.sound_len.values
    reaction_time = []
    for rt in reac_time:
        if rt > 500:
            rt = 500
        reaction_time.append(rt+300)
    choice = df_data.R_response.values
    coh = df_data.avtrapz.values*5
    zt = df_data.norm_allpriors.values*3
    times = df_data.times.values
    trial_index = np.arange(len(df_data)) + 1
    motor_time = []
    for tr in range(len(choice)):
        ind_time = [True if t != '' else False for t in times[tr]]
        time_tr = np.array(times[tr])[np.array(ind_time)].astype(float)
        mt = time_tr[-1]
        if mt > 1:
            mt = 1
        motor_time.append(mt*1e3)
    x_o = torch.column_stack((torch.tensor(motor_time),
                              torch.tensor(reaction_time),
                              torch.tensor(choice)))
    data = torch.column_stack((torch.tensor(zt), torch.tensor(coh),
                               torch.tensor(trial_index.astype(float)),
                               x_o))
    # load network
    with open(SV_FOLDER + f"/mnle_n{num_simulations}_no_noise.p", 'rb') as f:
        estimator = pickle.load(f)
    estimator = estimator['estimator']
    x0 = get_x0()
    print('Initial guess is: ' + str(x0))
    lb = get_lb_human()
    ub = get_ub_human()
    pub = get_pub()
    plb = get_plb()
    print('Optimizing')
    n_trials = len(data)
    # define fun_target as function to optimize
    # returns -LLH( data | parameters )
    fun_target = lambda x: fun_theta(x, data, estimator, n_trials)
    # define optimizer (BADS)
    bads = BADS(fun_target, x0, lb, ub, plb, pub,
                non_box_cons=nonbox_constraints_bads)
    # optimization
    optimize_result = bads.optimize()
    print(optimize_result.total_time)
    return optimize_result.x


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    optimization_cmaes = False
    optimization_mnle = True
    rms_comparison = False
    plotting = False
    plot_rms_llk = False
    single_run = False
    human = False
    if not optimization_mnle:
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
        num_simulations = int(10e6)  # number of simulations to train the network
        if not human:
            # load real data
            subjects = ['LE43', 'LE42', 'LE38', 'LE39', 'LE85', 'LE84', 'LE45',
                        'LE40', 'LE46', 'LE86', 'LE47', 'LE37', 'LE41', 'LE36',
                        'LE44']
            # subjects = ['LE43']  # to run only once and train
            training = False
            param_recovery_test = False
            if param_recovery_test:
                n_samples = 50
                subjects = ['LE42' for _ in range(n_samples)]
            for i_s, subject in enumerate(subjects):
                if i_s > 0:
                    training = False
                print('Fitting rat ' + str(subject))
                df = get_data_and_matrix(dfpath=DATA_FOLDER + subject, return_df=True,
                                         sv_folder=SV_FOLDER, after_correct=True,
                                         silent=True, all_trials=True,
                                         srfail=True)
                if param_recovery_test:
                    df = parameter_recovery_test_data_frames(df=df,
                                                             subjects=['Virtual_rat_random_params'],
                                                             extra_label='virt_sim_' + str(i_s))
                df = df.loc[df.special_trial == 0]
                # mnle_sample_simulation(df, theta=theta_for_lh_plot(),
                #                         num_simulations=len(df),
                #                         n_simul_training=int(2e6))
                # mnle_sample_simulation(df, theta=theta_for_lh_plot(),
                #                         num_simulations=len(df),
                #                         n_simul_training=int(3e6))
                # mnle_sample_simulation(df, theta=theta_for_lh_plot(),
                #                         num_simulations=len(df),
                #                         n_simul_training=int(4e6))
                # mnle_sample_simulation(df, theta=theta_for_lh_plot(),
                #                         num_simulations=len(df),
                #                         n_simul_training=int(10e6))
                # plot_lh_model_network(df, n_trials=2000000)
                # plot_lh_model_network(df, n_trials=3000000)
                # plot_lh_model_network(df, n_trials=10000000)
                # plot_kl_vs_zt_coh(df, theta=theta_for_lh_plot(), num_simulations=int(1e5),
                #                       n_simul_training=int(10e6))
                # plot_lh_model_network(df, n_trials=10000000)
                try:
                    extra_label = ''
                    parameters = opt_mnle(df=df, num_simulations=num_simulations,
                                          bads=True, training=training, extra_label=extra_label)
                    print('--------------')
                    print('p_w_zt: '+str(parameters[0]))
                    print('p_w_stim: '+str(parameters[1]))
                    print('p_e_bound: '+str(parameters[2]))
                    print('p_com_bound: '+str(parameters[3]))
                    print('p_t_aff: '+str(parameters[4]))
                    print('p_t_eff: '+str(parameters[5]))
                    print('p_t_a: '+str(parameters[6]))
                    print('p_w_a_intercept: '+str(parameters[7]))
                    print('p_w_a_slope: '+str(parameters[8]))
                    print('p_a_bound: '+str(parameters[9]))
                    print('p_1st_readout: '+str(parameters[10]))
                    print('p_2nd_readout: '+str(parameters[11]))
                    print('p_leak: '+str(parameters[12]))
                    print('p_mt_noise: '+str(parameters[13]))
                    print('p_MT_intercept: '+str(parameters[14]))
                    print('p_MT_slope: '+str(parameters[15]))
                    if param_recovery_test:
                        extra_label = 'virt_params/' +\
                            'parameters_MNLE_BADS_prt_n50_prt_' + str(i_s)
                    else:
                        extra_label_subj = 'parameters_MNLE_BADS' + subject
                    np.save(SV_FOLDER + extra_label_subj + extra_label + '.npy',
                            parameters)
                except Exception:
                    continue
        else:
            df = get_human_data(user_id='alex')
            subjects = df.subjid.values
            df['subjid'] = np.repeat('all', len(subjects))
            for subject in np.unique(df.subjid.unique()):
                parameters = human_fitting(df=df, subject=subject,
                                           num_simulations=num_simulations)
                print('--------------')
                print('p_w_zt: '+str(parameters[0]))
                print('p_w_stim: '+str(parameters[1]))
                print('p_e_bound: '+str(parameters[2]))
                print('p_com_bound: '+str(parameters[3]))
                print('p_t_aff: '+str(parameters[4]))
                print('p_t_eff: '+str(parameters[5]))
                print('p_t_a: '+str(parameters[6]))
                print('p_w_a_intercept: '+str(parameters[7]))
                print('p_w_a_slope: '+str(parameters[8]))
                print('p_a_bound: '+str(parameters[9]))
                print('p_1st_readout: '+str(parameters[10]))
                print('p_2nd_readout: '+str(parameters[11]))
                print('p_leak: '+str(parameters[12]))
                print('p_mt_noise: '+str(parameters[13]))
                print('p_MT_intercept: '+str(parameters[14]))
                print('p_MT_slope: '+str(parameters[15]))
                np.save(SV_FOLDER + 'parameters_MNLE_BADS_human_subj_' +
                        str(subject) + '.npy', parameters)
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
