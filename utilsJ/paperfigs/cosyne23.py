#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:49:51 2022

@author: manuel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append("C:/Users/Alexandre/Documents/psycho_priors") 
from utilsJ.Models import analyses_humans as ah
import figures_paper as fp
from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.Behavior.plotting import com_heatmap, tachometric


# SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
# DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
    'ChangesOfMind/figures/from_python/'  # Manuel
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
RAT_COM_IMG = '/home/molano/Dropbox/project_Barna/' +\
    'ChangesOfMind/figures/Figure_3/001965.png'
# RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'


def fig_1(df, ax, coh, hit, sound_len, choice, zt, com):
    for a in ax:
        fp.rm_top_right_lines(a)
    choice_01 = (choice+1)/2
    pos = ax[1].get_position()
    ax[1].set_position([pos.x1, pos.y0, pos.width*3/4, pos.height*3/4])
    edd2.com_heatmap_jordi(zt, coh, choice_01, ax=ax[1], flip=True,
                           annotate=False, xlabel='prior', ylabel='avg stim',
                           cmap='rocket')
    fp.tachometric_data(coh=coh, hit=hit, sound_len=sound_len, ax=ax[3])
    rat = plt.imread(RAT_COM_IMG)
    fig, ax = plt.subplots(ncols=3, figsize=(18, 5.5), gridspec_kw={
                           'width_ratios': [1, 1, 1.8]})
    fig.patch.set_facecolor('white')
    ax[4].set_facecolor('white')
    ax[4].imshow(np.flipud(rat))
    df_data = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                            'norm_allpriors': zt/max(abs(zt)),
                            'R_response': (choice+1)/2})
    pos = ax[5].get_position()
    ax[5].set_position([pos.x1, pos.y0, pos.width/2, pos.height/2])
    fp.com_heatmap_paper_marginal_pcom_side(df_data, side=0, ax=ax[5])
    ax_temp = plt.axes([pos.x1+pos.width/2, pos.y0,
                        pos.width/2, pos.height/2])
    fp.com_heatmap_paper_marginal_pcom_side(df_data, side=1, ax=ax_temp)


def com_heatmap_paper_marginal_pcom_side(
    df, f=None, ax=None,  # data source, must contain 'avtrapz' and allpriors
    pcomlabel=None, fcolorwhite=True, side=0,
    hide_marginal_axis=True, n_points_marginal=None, counts_on_matrix=False,
    adjust_marginal_axes=False,  # sets same max=y/x value
    nbins=7,  # nbins for the square matrix
    com_heatmap_kws={},  # avoid binning & return_mat already handled by the functn
    com_col='CoM_sugg', priors_col='norm_allpriors', stim_col='avtrapz',
    average_across_subjects=False
):
    assert side in [0, 1], "side value must be either 0 or 1"
    assert df[priors_col].abs().max() <= 1,\
        "prior must be normalized between -1 and 1"
    assert df[stim_col].abs().max() <= 1, "stimulus must be between -1 and 1"
    if pcomlabel is None:
        if not side:
            pcomlabel = r'$p(CoM_{R \rightarrow L})$'
        else:
            pcomlabel = r'$p(CoM_{L \rightarrow R})$'

    if n_points_marginal is None:
        n_points_marginal = nbins
    # ensure some filtering
    tmp = df.dropna(subset=['CoM_sugg', 'norm_allpriors', 'avtrapz'])
    tmp['tmp_com'] = False
    tmp.loc[(tmp.R_response == side) & (tmp.CoM_sugg), 'tmp_com'] = True

    com_heatmap_kws.update({
        'return_mat': True,
        'predefbins': [
            np.linspace(-1, 1, nbins+1), np.linspace(-1, 1, nbins+1)
        ]
    })
    if not average_across_subjects:
        mat, nmat = com_heatmap(
            tmp.norm_allpriors.values,
            tmp.avtrapz.values,
            tmp.tmp_com.values,
            **com_heatmap_kws
        )
        # fill nans with 0
        mat[np.isnan(mat)] = 0
        nmat[np.isnan(nmat)] = 0
        # change data to match vertical axis image standards (0,0) ->
        # in the top left
    else:
        com_mat_list, number_mat_list = [], []
        for subject in tmp.subjid.unique():
            cmat, cnmat = com_heatmap(
                tmp.loc[tmp.subjid == subject, 'norm_allpriors'].values,
                tmp.loc[tmp.subjid == subject, 'avtrapz'].values,
                tmp.loc[tmp.subjid == subject, 'tmp_com'].values,
                **com_heatmap_kws
            )
            cmat[np.isnan(cmat)] = 0
            cnmat[np.isnan(cnmat)] = 0
            com_mat_list += [cmat]
            number_mat_list += [cnmat]

        mat = np.stack(com_mat_list).mean(axis=0)
        nmat = np.stack(number_mat_list).mean(axis=0)

    mat = np.flipud(mat)
    nmat = np.flipud(nmat)
    return mat


def matrix_figure(df_data, humans, ax_tach, ax_pright, ax_mat):
    nbins = 7
    matrix_side_0 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=0)
    matrix_side_1 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=1)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = r'$p(CoM_{L \rightarrow R})$'
    ax_mat.set_title(pcomlabel_1)
    im_1 = ax_mat.imshow(matrix_side_1, vmin=0, vmax=vmax)
    divider = make_axes_locatable(ax_mat)
    cax = divider.append_axes('left', size='7%', pad=0.9)
    plt.colorbar(im_1, cax=cax)
    # R -> L
    pos = ax_mat.get_position()
    ax_mat.set_position([pos.x0, pos.y0*2/3, pos.width/2, pos.height*6/5])
    ax_mat_1 = plt.axes([pos.x0+pos.width/2, pos.y0*2/3,
                         pos.width/2, pos.height*6/5])
    pcomlabel_0 = r'$p(CoM_{L \rightarrow R})$'
    divider = make_axes_locatable(ax_mat_1)
    cax = divider.append_axes('right', size='7%', pad=0.9)
    cax.axis('off')
    ax_mat_1.set_title(pcomlabel_0)
    ax_mat_1.imshow(matrix_side_0, vmin=0, vmax=vmax)
    ax_mat_1.yaxis.set_ticks_position('none')
    for ax_i in [ax_pright, ax_mat, ax_mat_1]:
        ax_i.set_xlabel(r'$\longleftarrow$Prior$\longrightarrow$')
        ax_i.set_yticks(np.arange(nbins))
        ax_i.set_xticks(np.arange(nbins))
        ax_i.set_xticklabels(['left']+['']*(nbins-2)+['right'])
    for ax_i in [ax_pright, ax_mat]:
        ax_i.set_yticklabels(['right']+['']*(nbins-2)+['left'])
        ax_i.set_ylabel(r'$\longleftarrow$Average stimulus$\longrightarrow$',
                        labelpad=-17)
    ax_mat_1.set_yticklabels(['']*nbins)
    choice = df_data['R_response'].values
    coh = df_data['avtrapz'].values
    prior = df_data['norm_allpriors'].values
    mat_pright, _ = com_heatmap(prior, coh, choice, return_mat=True,
                                annotate=False)
    mat_pright = np.flipud(mat_pright)
    im_2 = ax_pright.imshow(mat_pright, cmap='rocket')
    divider = make_axes_locatable(ax_pright)
    cax1 = divider.append_axes('left', size='10%', pad=0.6)
    plt.colorbar(im_2, cax=cax1)
    cax1.yaxis.set_ticks_position('left')
    ax_pright.set_title('p(right)')
    if humans:
        num = 8
        rtbins = np.linspace(0, 300, num=num)
        tachometric(df_data, ax=ax_tach, fill_error=True, rtbins=rtbins)
    else:
        tachometric(df_data, ax=ax_tach, fill_error=True)
    ax_tach.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax_tach.set_xlabel('RT (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.4, 1.04)
    ax_tach.spines['right'].set_visible(False)
    ax_tach.spines['top'].set_visible(False)


def fig_3(user_id, sv_folder, ax_tach, ax_pright, ax_mat, humans=False, nm='300'):
    if user_id == 'Alex':
        folder = 'C:\\Users\\Alexandre\\Desktop\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'Manuel':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    subj = ['general_traj']
    steps = [None]
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    matrix_figure(df_data=df_data, ax_tach=ax_tach, ax_pright=ax_pright,
                  ax_mat=ax_mat, humans=humans)


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    subject = 'LE43'
    all_rats = True
    num_tr = int(15e4)
    f1 = True
    f2 = True
    f3 = True
    if f1:
        if all_rats:
            # df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + 'meta_subject/',
            #                               return_df=True, sv_folder=SV_FOLDER,
            #                               after_correct=True, silent=True,
            #                               all_trials=True)
            stim, zt, coh, gt, com, decision, sound_len, resp_len, hit,\
                trial_index, special_trial, traj_y, fix_onset, traj_stamps =\
                edd2.get_data_and_matrix(dfpath=DATA_FOLDER,
                                         num_tr_per_rat=int(1e4),
                                         after_correct=True, splitting=False,
                                         silent=False, all_trials=False,
                                         return_df=False, sv_folder=SV_FOLDER)
        else:
            df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + subject,
                                          return_df=True, sv_folder=SV_FOLDER,
                                          after_correct=True, silent=True,
                                          all_trials=True)
            after_correct_id = np.where(df.aftererror == 0)[0]
            zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
            zt = zt[after_correct_id]
            hit = np.array(df['hithistory'])
            hit = hit[after_correct_id]
            stim = np.array([stim for stim in df.res_sound])
            stim = stim[after_correct_id, :]
            coh = np.array(df.coh2)
            coh = coh[after_correct_id]
            com = df.CoM_sugg.values
            com = com[after_correct_id]
            decision = np.array(df.R_response) * 2 - 1
            decision = decision[after_correct_id]
            sound_len = np.array(df.sound_len)
            sound_len = sound_len[after_correct_id]
            gt = np.array(df.rewside) * 2 - 1
            gt = gt[after_correct_id]
            trial_index = np.array(df.origidx)
            trial_index = trial_index[after_correct_id]
        if stim.shape[0] != 20:
            stim = stim.T
        # FIG 1:
        df_data = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                                'norm_allpriors': zt/max(abs(zt)),
                                'R_response': (decision+1)/2,
                                'sound_len': sound_len,
                                'hithistory': hit})
        f, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        ax[0].axis('off')
        matrix_figure(df_data, ax_tach=ax[1], ax_pright=ax[2],
                      ax_mat=ax[3], humans=False)

    if f2:
        # FIG 2
        existing_model_data = False
        if not existing_model_data:
            hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
                pro_vs_re =\
                fp.run_model(stim=stim, zt=zt, coh=coh, gt=gt,
                             trial_index=trial_index,
                             num_tr=None)
            idx = reaction_time >= 0
            df_data = pd.DataFrame({'avtrapz': coh[idx],
                                    'CoM_sugg': com_model_detected[idx],
                                    'norm_allpriors': zt[idx]/max(abs(zt[idx])),
                                    'R_response': (resp_fin[idx] + 1)/2,
                                    'sound_len': reaction_time[idx],
                                    'hithistory': hit_model[idx]})
        else:
            df_data = pd.read_csv(DATA_FOLDER + 'df_fig_1.csv')
        f, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        humans = False
        ax[0].axis('off')
        matrix_figure(df_data=df_data, humans=humans, ax_tach=ax[1],
                      ax_pright=ax[2], ax_mat=ax[3])
    if f3:
        # FIG 3:
        f, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        ax[0].axis('off')
        fig_3(user_id='Manuel', sv_folder=SV_FOLDER,
              ax_tach=ax[1], ax_pright=ax[2], ax_mat=ax[3], humans=True)