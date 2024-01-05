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
import matplotlib.pylab as pl


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

def plot_trajs_cond_prior(df, data_folder, ax, extra_label):
    fig_5.traj_cond_coh_simul(df_sim=df[df.special_trial == 2], ax=ax,
                              new_data=False, data_folder=data_folder,
                              save_new_data=False,
                              median=False, prior=True, rt_lim=300,
                              extra_label=extra_label)


def plot_traj_cond_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
                                    special_trial, data_folder, extra_labels=['_2_ro', '_1_ro','']):
    fig, ax = plt.subplots(3, len(extra_labels), figsize=(10, 12))
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
                                              special_trial, extra_label=str(lab))
        plot_trajs_cond_prior(df=df_sim, data_folder=data_folder,
                              ax=[ax[n_labs+i_l], ax[int(n_labs*2)+i_l], ax[i_l], ax2], extra_label=str(lab))
        # ax[n_labs+i_l].set_ylim(0, 0.58)
        # ax[int(n_labs*2)+i_l].set_ylim(0, 0.02)
        # ax[n_labs+i_l].set_xlim(-2, 51)
        # ax[int(n_labs*2)+i_l].set_xlim(-2, 51)
    titles = ['w/o 1st read-out', 'w/o 2nd read-out', '2 read-outs']
    for i_a, a in enumerate([ax[0], ax[1], ax[2]]):
        a.set_ylim(235, 310)
        a.set_title(titles[i_a])


def plot_mt_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
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
        fig_1.plot_mt_vs_evidence(df=df_mt.loc[(df_mt.special_trial == 2) &
                                               (df_mt.sound_len < 50)], ax=ax[i_l+3],
                                  condition='choice_x_prior', prior_limit=1,
                                  rt_lim=200)
        del df_mt
        # MT VS COH
        # for i_rt, rtb in enumerate(rtbins[:-1]):
        #     df_mt = df_data.copy()
        #     plot_mt_vs_evidence_diff_rts(df=df_mt, ax=ax[i_l], prior_limit=0.2,  # 20% quantile
        #                                  condition='choice_x_coh',
        #                                  rt_lim=rtbins[i_rt+1], rtmin=rtb,
        #                                  color=colormap[i_rt])
        #     del df_mt
        df_mt = df_data.copy()
        fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[i_l], prior_limit=0.1,  # 10% quantile
                                  condition='choice_x_coh', rt_lim=200, rtmin=100)
        del df_mt        
    titles = ['w/o 1 read-out', 'w/o 2 read-out', '2 read-outs']
    for i_a, a in enumerate(ax[:3]):
        a.set_ylim(185, 290)
        a.set_title(titles[i_a])
    for i_a, a in enumerate([ax[3], ax[4], ax[5]]):
        a.set_ylim(185, 315)


def plot_mt_vs_stim_cong_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
                                          special_trial, data_folder, ax,
                                          extra_labels=['_2_ro', '_1_ro',''], alpha_list=[1, 0.3]):
    for i_l, lab in enumerate(extra_labels):
        alpha = alpha_list[i_l]
        df_data = get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                               special_trial, extra_label=lab)
        df_mt = df_data.copy()
        fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[i_l], prior_limit=0.1,  # 10% quantile
                                  condition='choice_x_coh', rt_lim=50, alpha=alpha)
        del df_mt


def fig_7(df, subjects, subjid, stim, zt, coh, gt, trial_index,
          special_trial, data_folder, sv_folder,
          extra_labels=['_2_ro', '_1_ro','']):
    fig, ax = plt.subplots(ncols=1, figsize=(4, 3.5))
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    # ax = ax.flatten()
    labs = ['a', 'b']
    for i_ax, a in enumerate([ax]):
        fp.rm_top_right_lines(a)
        a.text(-0.11, 1.12, labs[i_ax], transform=a.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    plot_mt_vs_stim_cong_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
                                          special_trial, data_folder, [ax, ax],
                                          extra_labels=extra_labels)
    df_mt = df.copy()
    fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax, prior_limit=0.1,  # 10% quantile
                              condition='choice_x_coh', rt_lim=50)
    del df_mt
    for a in [ax]:
        a.set_ylim(240, 285)
        a.text(-0.3, 282, r'$\longleftarrow $', fontsize=10)
        a.text(-0.75, 284.5, r'$\it{incongruent}$', fontsize=8)
        a.text(0.07, 282, r'$\longrightarrow $', fontsize=10)
        a.text(0.07, 284.5, r'$\it{congruent}$', fontsize=8)
    fig.savefig(sv_folder+'/fig7.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/fig7.png', dpi=400, bbox_inches='tight')
        

def plot_mt_vs_evidence_diff_rts(df, ax, condition='choice_x_coh', prior_limit=0.25,
                                 rt_lim=50, after_correct_only=True, rtmin=0, color='k'):
    subjects = df['subjid'].unique()
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.6)
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    if condition == 'choice_x_prior':
        df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    bins, _, indx_trajs, _, colormap =\
          fp.get_bin_info(df=df, condition=condition, prior_limit=prior_limit,
                          after_correct_only=after_correct_only,
                          rt_lim=rt_lim, rtmin=rtmin)
    df = df.loc[indx_trajs]
    df.resp_len *= 1e3
    if condition == 'choice_x_coh':
        # compute median MT for each subject and each stim strength
        df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
        mt_time = df.groupby(['choice_x_coh', 'subjid']).resp_len.median()
        # unstack to have a matrix with rows for subjects and columns for bins
        mt_time = mt_time.unstack(fill_value=np.nan).values.T
        plot_bins = sorted(df.coh2.unique())
        ax.set_xlabel('Stimulus evidence towards response')
        ax.set_ylim(238, 303)
    elif condition == 'choice_x_prior':
        mt_time = fp.binning_mt_prior(df, bins)
        plot_bins = bins[:-1] + np.diff(bins)/2
        ax.set_xlabel('Prior evidence towards response')
        ax.set_ylim(219, 312)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylabel('Movement time (ms)')
    ax.set_xticks([-1, 0, 1])
    ax.plot(plot_bins, np.mean(mt_time, axis=0), color=color, ls='-', lw=0.5)


def plot_acc_vs_stim_different_models(df, ax, color='k', sim=False,
                                      label='data'):
    if not sim:
        idx = df.CoM_sugg
    if sim:
        idx = df.com_detected
    # idx = df.sound_len < 50
    df_1 = df.copy().loc[idx]
    # ch_1st_traj = df_1.R_response.values*2-1
    # if not sim:
    #     idx_com = df_1.CoM_sugg
    # if sim:
    #     idx_com = df_1.com_detected
    # ch_1st_traj[idx_com] *= -1
    # df_1['R_response'] = (ch_1st_traj+1)/2
    df_1['hithistory'] = -(df_1.R_response.values*2-1) == (df_1.rewside.values*2-1)
    ev_vals = [0, 0.25, 0.5, 1]
    prop_corr = []
    for i_ev, ev in enumerate(ev_vals):
        acc = np.nanmean(df_1.loc[df_1.coh2 == ev].hithistory.values)
        prop_corr.append(acc)
    ax.plot(ev_vals, prop_corr, color=color, marker='o', label=label)
    
    
def plot_psycho_acc_reversals(df, subjects, subjid, stim, zt, coh, gt, trial_index,
                              special_trial, data_folder, extra_labels=['']):
    fig, ax = plt.subplots(ncols=1, figsize=(7, 7))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    fp.rm_top_right_lines(ax)
    colors = ['k', 'b', 'r']
    labels = ['data', 'w/o 1st r.o.', 'full model']
    sim = False
    for i_l, extra_lab in enumerate(extra_labels):
        if i_l == 0:
            df_sim = df.copy()
        else:
            df_sim = get_simulated_data_extra_lab(subjects, subjid, stim, zt, coh, gt, trial_index,
                                                  special_trial, extra_label=extra_lab)
            sim = True
        plot_acc_vs_stim_different_models(df=df_sim, ax=ax, color=colors[i_l],
                                          label=labels[i_l], sim=sim)
    ax.legend()
    ax.set_xticks([0, 0.25, 0.5, 1])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Stimulus strength')
       