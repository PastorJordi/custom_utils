#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:49:51 2022

@author: manuel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem
import sys
import figures_paper as fp
from utilsJ.Models import extended_ddm_v2 as edd2


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


def fig_1(ax, coh, hit, sound_len, choice, zt):
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
    assert df[priors_col].abs().max() <= 1, "prior must be normalized between -1 and 1"
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
    if f is None and ax is None:
        f, ax = plt.subplots(
            ncols=2, nrows=2,
            gridspec_kw={'width_ratios': [8, 3], 'height_ratios': [3, 8]},
            figsize=(7, 5.5), sharex='col', sharey='row')

    # some aestethics
    if fcolorwhite:
        f.patch.set_facecolor('white')
        for i in [0, 1]:
            for j in [0, 1]:
                ax[i, j].set_facecolor('white')

    ax[0, 1].axis('off')
    ax[0, 0].set_ylabel(pcomlabel)
    ax[1, 1].set_xlabel(pcomlabel)
    if hide_marginal_axis:
        ax[0, 0].spines['top'].set_visible(False)
        ax[0, 0].spines['left'].set_visible(False)
        ax[0, 0].spines['right'].set_visible(False)
        ax[0, 0].set_yticks([])
        # ax[1,1].xaxis.set_visible(False)
        ax[1, 1].spines['right'].set_visible(False)
        ax[1, 1].spines['top'].set_visible(False)
        ax[1, 1].spines['bottom'].set_visible(False)
        ax[1, 1].set_xticks([])

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

   
    # since it is the same, simply adjust max y_
    if adjust_marginal_axes:
        _, ymax = ax[0, 0].get_ylim()
        _, xmax = ax[1, 1].get_xlim()
        max_val_margin = max(ymax, xmax)
        ax[0, 0].set_ylim(0, max_val_margin)
        ax[1, 1].set_xlim(0, max_val_margin)

    ax[1, 0].set_yticks(np.arange(nbins))
    ax[1, 0].set_xticks(np.arange(nbins))
    ax[1, 0].set_yticklabels(['right']+['']*(nbins-2)+['left'])
    ax[1, 0].set_xticklabels(['left']+['']*(nbins-2)+['right'])

    if counts_on_matrix:
        im = ax[1, 0].imshow(nmat, aspect='auto')
    else:
        im = ax[1, 0].imshow(mat, aspect='auto')
    ax[1, 0].set_xlabel('$\longleftarrow$Prior$\longrightarrow$', labelpad=-5)
    ax[1, 0].set_ylabel(
        '$\longleftarrow$Average stimulus$\longrightarrow$', labelpad=-17)
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes('left', size='10%', pad=0.6)

    divider2 = make_axes_locatable(ax[0, 0])
    empty = divider2.append_axes('left', size='10%', pad=0.6)
    empty.axis('off')
    cax2 = cax.secondary_yaxis('left')
    f.colorbar(im, cax=cax)
    cax.yaxis.set_ticks_position('left')
    if counts_on_matrix:
        cax2.set_ylabel('# trials')
    else:
        cax2.set_ylabel(pcomlabel)

    f.tight_layout()
    return f, ax
    

if __name__ == '__main__':
    plt.close('all')
    subject = 'LE43'
    all_rats = False
    if all_rats:
        df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + 'meta_subject/',
                                      return_df=True, sv_folder=SV_FOLDER,
                                      after_correct=True, silent=True,
                                      all_trials=True)
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
    choice = np.array(df.R_response) * 2 - 1
    choice = choice[after_correct_id]
    sound_len = np.array(df.sound_len)
    sound_len = sound_len[after_correct_id]
    gt = np.array(df.rewside) * 2 - 1
    gt = gt[after_correct_id]
    trial_index = np.array(df.origidx)
    trial_index = trial_index[after_correct_id]
        
    # if we want to use data from all rats, we must use dani_clean.pkl
    f1 = True
    f2 = False
    f3 = True


    # fig 1
    if f1:
        f, ax = plt.subplots(nrows=2, ncols=3)
        # fig1.d(df, savpath=SV_FOLDER, average=True)  # psychometrics
        # tachometrics, rt distribution, express performance
        fig_1(coh=coh, hit=hit, sound_len=sound_len,
              choice=choice, zt=zt, com=com, supt='')

    # fig 2
    if f2:
        fgsz = (8, 8)
        inset_sz = 0.1
        f, ax = plt.subplots(nrows=2, ncols=2, figsize=fgsz)
        ax = ax.flatten()
        ax_cohs = np.array([ax[0], ax[2]])
        ax_inset = fp.add_inset(ax=ax_cohs[0], inset_sz=inset_sz, fgsz=fgsz)
        ax_cohs = np.insert(ax_cohs, 0, ax_inset)
        ax_inset = fp.add_inset(ax=ax_cohs[2], inset_sz=inset_sz, fgsz=fgsz,
                                marginy=0.15)
        ax_cohs = np.insert(ax_cohs, 2, ax_inset)
        for a in ax:
            fp.rm_top_right_lines(a)
        fp.trajs_cond_on_coh(df=df, ax=ax_cohs)
        # splits
        ax_split = np.array([ax[1], ax[3]])
        fp.trajs_splitting(df, ax=ax_split[0])
        # XXX: do this panel for all rats?
        fp.trajs_splitting_point(df=df, ax=ax_split[1])
        # fig3.trajs_cond_on_prior(df, savpath=SV_FOLDER)

    # fig 3
