# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:18:25 2023

@author: alexg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# from imp import reload
import sys


sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')  # Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel

from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.paperfigs import figure_5 as fig_5
from utilsJ.paperfigs import fig_5_humans as fig_5h
from utilsJ.paperfigs import figure_6 as fig_6
from utilsJ.paperfigs import figures_paper as fp


def get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                 special_trial, extra_label='_1_ro'):
    num_tr = len(gt)
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt =\
        fp.run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                          trial_index=trial_index, num_tr=num_tr,
                                          subject_list=subjects, subjid=subjid,
                                          simulate=False, extra_label=extra_label)
    MT = [len(t) for t in trajs]
    df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                           'sound_len': reaction_time,
                           'rewside': (gt + 1)/2,
                           'R_response': (resp_fin+1)/2,
                           'resp_len': np.array(MT)*1e-3})
    df_sim['CoM_sugg'] = com_model.astype(bool)
    df_sim['traj_d1'] = [np.diff(t) for t in trajs]
    df_sim['subjid'] = subjid
    df_sim['origidx'] = trial_index
    df_sim['special_trial'] = special_trial
    df_sim['traj'] = df_sim['trajectory_y']
    df_sim['com_detected'] = com_model_detected.astype(bool)
    df_sim['peak_com'] = np.array(x_val_at_updt)
    df_sim['hithistory'] = np.array(resp_fin == gt)
    df_sim['allpriors'] = zt
    df_sim['norm_allpriors'] = fp.norm_allpriors_per_subj(df_sim)
    df_sim['normallpriors'] = df_sim['norm_allpriors']
    df_sim['framerate']=200
    df_sim['dW_lat'] = 0
    df_sim['dW_trans'] = zt
    df_sim['aftererror'] = False
    return df_sim


def plot_trajs_cond_stim(df, data_folder, ax, extra_label):
    fig_5.traj_cond_coh_simul(df_sim=df, ax=ax, median=True, prior=False,
                              save_new_data=False,
                              new_data=False, data_folder=data_folder,
                              prior_lim=np.quantile(df.norm_allpriors.abs(), 0.1),
                              extra_label=extra_label, rt_lim=50)


def plot_traj_cond_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
                                    special_trial, data_folder, extra_labels=['_1_ro', '', '_no_taff',
                                                                              '_no_teff']):
    fig, ax = plt.subplots(4, len(extra_labels), figsize=(10, 12))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    fig2, ax2 = plt.subplots(1)
    n_labs = len(extra_labels)
    ax = ax.flatten()
    # df_mt = df.copy()
    # fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[0], prior_limit=0.1,  # 10% quantile
    #                           condition='choice_x_coh', rt_lim=50)
    # del df_mt
    for a in ax:
        fp.rm_top_right_lines(a)
    for i_l, lab in enumerate(extra_labels):
        df_sim = get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                              special_trial, extra_label=lab)
        plot_trajs_cond_stim(df=df_sim.loc[~df_sim.CoM_sugg], data_folder=data_folder,
                             ax=[ax[n_labs+i_l], ax[int(n_labs*2)+i_l], ax[0+i_l], ax2], extra_label=lab+'_no_com')
        fig_2.trajs_splitting_stim(df_sim.loc[(df_sim.special_trial == 0) &
                                              (~df_sim.CoM_sugg)],
                                   data_folder=data_folder, ax=ax[int(n_labs*3)+i_l], collapse_sides=True,
                                   threshold=800, sim=True, rtbins=np.linspace(0, 150, 16),
                                   connect_points=True, trajectory="trajectory_y",
                                   p_val=0.05, extra_label=lab+'_no_com')
    titles = ['1 read-out', '2 read-out', 't_aff = 0', 't_eff=0']
    for i_a, a in enumerate([ax[0], ax[1], ax[2], ax[3]]):
        a.set_ylim(210, 285)
        a.set_title(titles[i_a])


def plot_mt_ER_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
                                special_trial, data_folder, extra_labels=['_2_ro', '_1_ro','']):
    fig, ax = plt.subplots(2, len(extra_labels), figsize=(10, 6))
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    ax = ax.flatten()
    # df_mt = df.copy()
    # fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[0], prior_limit=0.1,  # 10% quantile
    #                           condition='choice_x_coh', rt_lim=50)
    # del df_mt
    for a in ax:
        fp.rm_top_right_lines(a)
    for i_l, lab in enumerate(extra_labels):
        df_data = get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                               special_trial, extra_label=lab)
        # fig_1.plot_mt_vs_stim(df=df_sim, ax=ax[i_l], prior_min=0.8, rt_max=50,
        #                       sim=True)
        # MT VS PRIOR
        df_mt = df_data.copy()
        fig_1.plot_mt_vs_evidence(df=df_mt.loc[df_mt.special_trial == 2], ax=ax[i_l+3],
                                  condition='choice_x_prior', prior_limit=1,
                                  rt_lim=200)
        del df_mt
        # MT VS COH
        df_mt = df_data.copy()
        fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[i_l], prior_limit=0.1,  # 10% quantile
                                  condition='choice_x_coh', rt_lim=50)
        del df_mt
    titles = ['w/o 1 read-out', 'w/o 2 read-out', '2 read-outs']
    for i_a, a in enumerate(ax[:3]):
        a.set_ylim(245, 275)
        a.set_title(titles[i_a])
    for i_a, a in enumerate([ax[3], ax[4], ax[5]]):
        a.set_ylim(230, 295)
