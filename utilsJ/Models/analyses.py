# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:47:49 2022
@author: Alex Garcia-Duran
"""
import numpy as np
from utilsJ.Behavior import plotting
from utilsJ.Models import traj
import matplotlib.pyplot as plt

# import os
# from sklearn.model_selection import ParameterGrid
# from utilsJ.regularimports import *
# import pandas as pd
# from concurrent.futures import as_completed, ThreadPoolExecutor
# from scipy.stats import ttest_ind, sem
# from matplotlib import cm
# import swifter
# import seaborn as sns
# from scipy.stats import norm
# import warnings
general_path = traj.general_path


def com_matrix_comparison_silent_vs_nonsilent(df):
    # comparing CoM matrix for silent vs non-silent samples
    subset = df.dropna(
        subset=["avtrapz", "allpriors", "CoM_sugg", "special_trial"])
    subset_silent = subset.loc[subset.special_trial == 2]
    subset_nonsilent = subset.loc[subset.special_trial == 0]
    mat_data_silent, _ = plotting.com_heatmap(
        subset_silent.allpriors,
        subset_silent.avtrapz,
        subset_silent.CoM_sugg,
        return_mat=True)
    mat_data_nonsilent, _ = plotting.com_heatmap(
        subset_nonsilent.allpriors,
        subset_nonsilent.avtrapz,
        subset_nonsilent.CoM_sugg,
        return_mat=True)
    plt.figure()
    plt.plot(mat_data_silent)
    plt.plot(mat_data_nonsilent)

    pcom_prior = np.zeros((len(df.subjid.unique()), 7))
    pcom_prior_sil = np.zeros((len(df.subjid.unique()), 7))
    num0 = np.zeros((1,7))
    num0_sil = np.copy(num0)
    for i, subj in enumerate(df.subjid.unique()):
         df_sub = df.loc[df.subjid == subj]
         subset=df_sub.dropna(subset=["avtrapz", "allpriors", "CoM_sugg", "special_trial"])
         subset_silent=subset.loc[subset.special_trial==2]
         subset_nonsilent=subset.loc[subset.special_trial==0]
         mat_data_silent, num_sil = plotting.com_heatmap(
             subset_silent.allpriors,
             subset_silent.avtrapz,
             subset_silent.CoM_sugg,
             return_mat=True)
         mat_data_nonsilent, num_nonsil = plotting.com_heatmap(
             subset_nonsilent.allpriors,
             subset_nonsilent.avtrapz,
             subset_nonsilent.CoM_sugg,
             return_mat=True)
         pcom_prior_sil[i, :] = mat_data_silent[3,:]
         pcom_prior[i, :] = mat_data_nonsilent[3,:]
         num0 += num_nonsil[3,:]
         num0_sil += num_sil[3,:]
    prior_mn = np.nanmean(pcom_prior, axis=0)
    prior_mn_sil = np.nanmean(pcom_prior_sil, axis=0)
    prior_sd = np.nanstd(pcom_prior, axis=0)/np.sqrt(num0[0])
    prior_sd_sil = np.nanstd(pcom_prior_sil, axis=0)/np.sqrt(num0_sil[0])
    plt.figure()
    plt.errorbar(np.linspace(-3,3,num=7),prior_mn, prior_sd, label='nonsilent',
                 marker='.')
    plt.errorbar(np.linspace(-3,3,num=7),prior_mn_sil, prior_sd_sil, label='silent',
                 marker='.')
    plt.legend(fontsize=16)
    plt.ylabel('pCoM', fontsize=16)
    plt.xlabel('prior', fontsize=16)
    return mat_data_silent, mat_data_nonsilent


def pCoM_vs_coh(df, subject):
    """
    It computes the pCoM vs the coherence
    """
    ev_vals = np.sort(np.abs(df.coh2.unique()))  # unique absolute value coh values
    if subject == 'all':
        all_pcom_coh = np.zeros((len(df.subjid.unique()), len(ev_vals)+1))
        for i, subj in enumerate(df.subjid.unique()):
            df_sub = df.loc[df.subjid == subj]
            # separating silent from nonsilent
            silent = df_sub.loc[df_sub.special_trial == 2]
            nonsilent = df_sub.loc[df_sub.special_trial == 0]
            pcom_coh = []  # np.mean(silent.CoM_sugg)
            num = np.array([len(silent)])
            for ev in ev_vals: # for each coherence, a pCoM for each subject
                index = np.abs(nonsilent.coh2) == ev
                pcom_coh.append(np.nanmean(nonsilent.CoM_sugg[index]))
                num = np.concatenate([num, np.array([sum(index)])])
            all_pcom_coh[i, 1::] = np.array(pcom_coh)
            all_pcom_coh[i, 0] = np.mean(silent.CoM_sugg)
        all_pcom_coh_sd = np.std(all_pcom_coh, axis=0)/np.sqrt(num)
        all_pcom_coh_mn = np.mean(all_pcom_coh, axis=0)
        plt.figure()
        ev_vals_sil = np.concatenate([np.array([-0.25]), ev_vals])
        plt.errorbar(ev_vals, all_pcom_coh_mn[1::], yerr=all_pcom_coh_sd[1::], color='b')
        plt.errorbar(-0.25, all_pcom_coh_mn[0], yerr=all_pcom_coh_sd[0], color='b')
        plt.xticks(ticks=ev_vals_sil, labels=(['silent'] + list(ev_vals)))
        plt.xlabel('coh')
        plt.ylabel('pCoM')
    else:
        df_sub = df.loc[df.subjid == subject]
        silent = df_sub.loc[df_sub.special_trial == 2]
        nonsilent = df_sub.loc[df_sub.special_trial == 0]
        pcom_coh = [np.nanmean(silent.CoM_sugg)]   # np.mean(silent.CoM_sugg)
        for ev in ev_vals:
            index = np.abs(nonsilent.coh2) == ev
            pcom_coh.append(np.nanmean(nonsilent.CoM_sugg[index]))
        plt.figure()
        ev_vals_sil = np.concatenate([np.array([-0.25]), ev_vals])
        plt.plot(ev_vals_sil, pcom_coh)
        plt.xticks(ticks=ev_vals_sil, labels=(['silent'] + list(ev_vals)))
        plt.xlabel('coh')
        plt.ylabel('pCoM')
