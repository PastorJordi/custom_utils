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
            for ev in ev_vals:  # for each coherence, a pCoM for each subject
                index = np.abs(df.coh2) == ev
                pcom_coh.append(np.nanmean(nonsilent.CoM_sugg[index]))
                num = np.concatenate([num, np.array([sum(index)])])
            all_pcom_coh[i, 1::] = np.array(pcom_coh)
            all_pcom_coh[i, 0] = np.mean(silent.CoM_sugg)
        all_pcom_coh_sd = np.sqrt(np.std(all_pcom_coh, axis=0)/num)
        all_pcom_coh_mn = np.mean(all_pcom_coh, axis=0)
        plt.figure()
        ev_vals_sil = np.concatenate([np.array([-0.25]), ev_vals])
        plt.errorbar(ev_vals_sil, all_pcom_coh_mn, yerr=all_pcom_coh_sd)
        plt.xlabel('coh')
        plt.ylabel('pCoM')
    else:
        df_sub = df.loc[df.subjid == subject]
        silent = df_sub.loc[df_sub.special_trial == 2]
        nonsilent = df_sub.loc[df_sub.special_trial == 0]
        pcom_coh = [np.nanmean(silent.CoM_sugg)]   # np.mean(silent.CoM_sugg)
        for ev in ev_vals:
            index = np.abs(df.coh2) == ev
            pcom_coh.append(np.nanmean(nonsilent.CoM_sugg[index]))
        plt.figure()
        ev_vals_sil = np.concatenate([np.array([-0.25]), ev_vals])
        plt.plot(ev_vals_sil, pcom_coh)
        plt.xlabel('coh')
        plt.ylabel('pCoM')
