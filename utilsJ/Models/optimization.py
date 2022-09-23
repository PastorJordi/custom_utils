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
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
# sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
from utilsJ.Models.extended_ddm import trial_ev_vectorized, data_augmentation
from utilsJ.Behavior.plotting import binned_curve
from BayesPy.DirichletEstimation import dirichletMultinomialEstimation as dme

# DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
DATA_FOLDER = '/home/garciaduran/data/'  # Cluster Alex
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
# SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/opt_results/'  # Alex
SV_FOLDER = '/home/garciaduran/opt_results/'  # Cluster Alex
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/opt_results/'  # Jordi
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
        prior = np.concatenate((prior, prior_tmp[indx]))
        stim = np.concatenate((stim, stim_tmp[indx, :]))
        coh = np.concatenate((coh, coh_mat[indx]))
        com = np.concatenate((com, com_tmp[indx]))
        gt = np.concatenate((gt, gt_tmp[indx]))
        pright = np.concatenate((pright, pright_tmp[indx]))
        sound_len = np.concatenate((sound_len, sound_len_tmp[indx]))
        end = time.time()
        print(f)
        print(end - start_1)
        print(len(df))
    print(end - start)
    print('Ended loading data')
    stim = stim.T
    zt = prior
    return stim, zt, coh, gt, com, pright


def fitting(res_path='C:/Users/Alexandre/Desktop/CRM/brute_force/', results=False,
            detected_com=None, first_ind=None, p_t_eff=None,
            data_path='C:/Users/Alexandre/Desktop/CRM/Alex/paper/results/',
            metrics='mse', objective='curve', bin_size=20, det_th=5,
            plot=False, stim_res=5):
    data_mat = np.load(data_path + 'all_tr_ac_pCoM_vs_prior_and_stim.npy')
    data_mat_norm = data_mat / np.nanmax(data_mat)
    data_curve = pd.read_csv(data_path + 'pcom_vs_rt.csv')
    tmp_data = data_curve['tmp_bin']
    data_curve_norm = data_curve['pcom'] / np.max(data_curve['pcom'])
    nan_penalty = 0.3
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
                else:
                    rt_vals_pcom = data.get('xpos_rt_pcom')
                    rt_vals_pcom = [rt.to_numpy().astype(int) for
                                    rt in rt_vals_pcom]
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
                diff_mn.append(diff) if not np.isnan(diff) else diff_mn.append(1e3)
                max_ssim = False
        if plot:
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
            ind_min = ind_sorted[40]
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
            plt.plot(data_curve['rt'], data_curve['pcom'], label='data',
                     linestyle='', marker='o')
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
            plt.plot(data_curve['rt'], data_curve['pcom'], label='data',
                     linestyle='', marker='o')
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


def run_likelihood(stim, zt, coh, gt, com, pright, p_w_zt, p_w_stim, p_e_noise,
                   p_com_bound, p_t_aff, p_t_eff, p_t_a, p_w_a, p_a_noise,
                   p_w_updt, num_times_tr=int(1e3), detect_CoMs_th=5,
                   rms_comparison=False, epsilon=1e-6):
    start_llk = time.time()
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
    MT_slope = 0.15
    MT_intercep = 110
    compute_trajectories = True
    all_trajs = True
    data_augment_factor = 10
    stim = data_augmentation(stim=stim, daf=data_augment_factor)
    stim_res = 50/data_augment_factor
    global fixation
    fixation = int(300 / stim_res)
    stim_temp = np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                                stim.shape[1]))))
    detected_com_mat = np.zeros((num_tr, num_times_tr))
    pright_mat = np.zeros((num_tr, num_times_tr))
    diff_rms_list = []
    for i in range(num_times_tr):
        # start_simu = time.time()
        E, A, com_model, first_ind, second_ind, resp_first, resp_fin, pro_vs_re,\
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
        pright_mat[:, i] = (resp_fin + 1)/2
        # end_simu = time.time()
        # print('Trial {} simulation: '.format(i) + str(end_simu - start_simu))
        if rms_comparison:
            diff_rms = fitting(detected_com=detected_com, p_t_eff=p_t_eff,
                               first_ind=first_ind, data_path=DATA_FOLDER)
            diff_rms_list.append(diff_rms)
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
    data = dme.CompressedRowData(matrix_dirichlet.shape[1])
    for row_ind in range(matrix_dirichlet.shape[1]):
        data.appendRow(matrix_dirichlet[row_ind, :], weight=1)
    alpha_vector = dme.findDirichletPriors(data=data,
                                           initAlphas=[1, 1, 1, 1],
                                           iterations=1000)
    # end_dirichlet = time.time()
    # print('End Dirichlet: ' + str(end_dirichlet - start_dirichlet))
    alpha_sum = np.sum(alpha_vector)
    # prob_detected_com = np.nanmean(detected_com_mat, axis=1)
    # prob_right = np.nanmean(pright_mat, axis=1)
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
        p_e_noise = params[5]
        p_com_bound = params[6]
        p_w_a = params[7]
        p_a_noise = params[8]
        p_w_updt = params[9]
        llk_val, diff_rms =\
            run_likelihood(stim, zt, coh, gt, com, pright, p_w_zt, p_w_stim,
                           p_e_noise, p_com_bound, p_t_aff, p_t_eff,
                           p_t_a, p_w_a, p_a_noise, p_w_updt,
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


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    optimization = False
    rms_comparison = False
    plotting = False
    plot_rms_llk = False
    single_run = True
    stim, zt, coh, gt, com, pright = get_data(dfpath=DATA_FOLDER,
                                              after_correct=True,
                                              num_tr_per_rat=int(2e3),
                                              all_trials=False)
    array_params = np.array((10, 6, 35, 0.15, 0.15, 0.05, 0.3, 0.03, 0.06, 0.1))
    scaled_params = np.repeat(1, len(array_params)).astype(float)
    scaling_value = array_params/scaled_params
    bounds = np.array(((5, 20), (3, 10), (20, 60), (0.05, 0.3), (0.05, 0.3),
                      (0.005, 0.1), (0., 1), (0.005, 0.1), (0.05, 0.1),
                      (0.05, 60)))
    bounds_scaled = np.array([bound / scaling_value[i_b] for i_b, bound in
                              enumerate(bounds)])
    if single_run:
        p_t_aff = 7
        p_t_eff = 7
        p_t_a = 35
        p_w_zt = 0.16
        p_w_stim = 0.15
        p_e_noise = 0.05
        p_com_bound = 0.
        p_w_a = 0.03
        p_a_noise = 0.06
        p_w_updt = 20
        llk_val, diff_rms = run_likelihood(stim, zt, coh, gt, com, pright,
                                           p_w_zt, p_w_stim,
                                           p_e_noise, p_com_bound, p_t_aff,
                                           p_t_eff, p_t_a, p_w_a, p_a_noise,
                                           p_w_updt, num_times_tr=int(5e0),
                                           detect_CoMs_th=5, rms_comparison=False)
        print(llk_val)
    # TODO: paralelize different initial points
    if optimization:
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
                llk_val, diff_rms =\
                    run_likelihood(stim=stim, zt=zt, coh=coh, gt=gt, com=com,
                                   pright=pright, p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                   p_e_noise=p_e_noise, p_com_bound=p_com_bound,
                                   p_t_aff=p_t_aff, p_t_eff=p_t_eff,
                                   p_t_a=p_t_a, p_w_a=p_w_a, p_a_noise=p_a_noise,
                                   p_w_updt=p_w_updt,
                                   num_times_tr=int(1e2), detect_CoMs_th=5,
                                   rms_comparison=rms_comparison)
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
