#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:49:51 2022

@author: manuel
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pl
sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
sys.path.append("C:/Users/Alexandre/Documents/psycho_priors") 
import fig4
from utilsJ.Models import analyses_humans as ah
import figures_paper as fp
from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.Behavior.plotting import com_heatmap, tachometric
import matplotlib
matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3


SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
# DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
# SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
#     'ChangesOfMind/figures/from_python/'  # Manuel
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
# SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
# DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
# RAT_COM_IMG = '/home/molano/Dropbox/project_Barna/' +\
#     'ChangesOfMind/figures/Figure_3/001965.png'
RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'
FRAME_RATE = 14


def plot_coms(df, ax, human=False):
    coms = df.CoM_sugg.values
    decision = df.R_response.values
    if human:
        ran_max = 600
        max_val = 600
    if not human:
        ran_max = 200
        max_val = 77
    for tr in reversed(range(ran_max)):  # len(df_rat)):
        if tr > (ran_max/1.2) and not coms[tr] and decision[tr] == 1:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = np.arange(len(traj))*FRAME_RATE
                ax.plot(time, traj, color='tab:cyan', lw=.5)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.1:
                    ax.plot(time*1e3, traj, color='tab:cyan', lw=.5)
        elif tr < (ran_max/1.2-1) and coms[tr] and decision[tr] == 0:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = np.arange(len(traj))*FRAME_RATE
                ax.plot(time, traj, color='tab:olive', lw=2)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.2:
                    ax.plot(time*1e3, traj, color='tab:olive', lw=2)
    fp.rm_top_right_lines(ax)
    if human:
        var = 'x'
        sp = 'Subject'
    if not human:
        var = 'y'
        sp = 'Rats'
    ax.set_ylabel('{} position {}-axis (pixels)'.format(sp, var))
    ax.set_xlabel('Time from movement onset (ms)')
    ax.axhline(y=max_val, linestyle='--', color='Green', lw=1)
    ax.axhline(y=-max_val, linestyle='--', color='Purple', lw=1)
    ax.axhline(y=0, linestyle='--', color='k', lw=0.5)


def tracking_image(ax):
    rat = plt.imread(RAT_COM_IMG)
    ax.set_facecolor('white')
    ax.imshow(np.flipud(rat[100:-100, 350:-50, :]))
    ax.axis('off')


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
    # plot tachometrics
    if humans:
        num = 8
        rtbins = np.linspace(0, 300, num=num)
        tachometric(df_data, ax=ax_tach, fill_error=True, rtbins=rtbins,
                    cmap='gist_yarg')
    else:
        tachometric(df_data, ax=ax_tach, fill_error=True, cmap='gist_yarg')
    ax_tach.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax_tach.set_xlabel('Reaction Time (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.3, 1.04)
    ax_tach.spines['right'].set_visible(False)
    ax_tach.spines['top'].set_visible(False)
    ax_tach.legend()
    # plot Pcoms matrices
    nbins = 7
    matrix_side_0 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=0)
    matrix_side_1 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=1)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[0].set_title(pcomlabel_1)
    im = ax_mat[0].imshow(matrix_side_1, vmin=0, vmax=vmax)
    plt.sca(ax_mat[0])
    plt.colorbar(im, fraction=0.04)
    # pos = ax_mat.get_position()
    # ax_mat.set_position([pos.x0, pos.y0*2/3, pos.width, pos.height])
    # ax_mat_1 = plt.axes([pos.x0+pos.width+0.05, pos.y0*2/3,
    #                      pos.width, pos.height])
    pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[1].set_title(pcomlabel_0)
    im = ax_mat[1].imshow(matrix_side_0, vmin=0, vmax=vmax)
    ax_mat[1].yaxis.set_ticks_position('none')
    plt.sca(ax_mat[1])
    plt.colorbar(im, fraction=0.04)
    # pright matrix
    choice = df_data['R_response'].values
    coh = df_data['avtrapz'].values
    prior = df_data['norm_allpriors'].values
    mat_pright, _ = com_heatmap(prior, coh, choice, return_mat=True,
                                annotate=False)
    mat_pright = np.flipud(mat_pright)
    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    plt.sca(ax_pright)
    plt.colorbar(im_2, fraction=0.04)
    ax_pright.set_title('Proportion of rightward responses')

    # R -> L
    for ax_i in [ax_pright, ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior Evidence')
        # ax_i.set_yticks(np.arange(nbins))
        # ax_i.set_xticks(np.arange(nbins))
        # ax_i.set_xticklabels(['left']+['']*(nbins-2)+['right'])
        ax_i.set_yticklabels(['']*nbins)
        ax_i.set_xticklabels(['']*nbins)
    for ax_i in [ax_pright, ax_mat[0]]:
        # ax_i.set_yticklabels(['right']+['']*(nbins-2)+['left'])
        ax_i.set_ylabel('Stimulus Evidence')  # , labelpad=-17)

    # ax_mat[1].set_aspect('equal', adjustable='box')
    # ax_mat[0].set_aspect('equal', adjustable='box')
    # ax_pright.set_aspect('equal', adjustable='box')


def fig_3(user_id, sv_folder, ax_tach, ax_pright, ax_mat, ax_traj, humans=False,
          nm='300'):
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
    plot_coms(df=df_data, ax=ax_traj, human=humans)


def human_trajs(df_data, max_mt=600, jitter=0.003, wanted_precision=8):
    coh = df_data.avtrapz.values
    decision = df_data.R_response.values
    trajs = df_data.trajectory_y.values
    times = df_data.times.values
    ev_vals = np.unique(np.abs(np.round(coh, 2)))
    bins = [0, 0.25, 0.5, 1]
    congruent_coh = coh * (decision*2 - 1)
    fig, ax = plt.subplots(2)
    ax = ax.flatten()
    colormap = pl.cm.viridis(np.linspace(0, 1, len(ev_vals)))
    for i_ev, ev in enumerate(ev_vals):
        index = np.abs(np.round(congruent_coh, 2)) == ev
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        all_vels = np.empty((sum(index), max_mt))
        all_vels[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)\
                + jitter
            max_time = max(time)*1e3
            if max_time > max_mt:
                continue
            vals_fin = np.interp(np.arange(0, int(max_time), wanted_precision),
                                 xp=time*1e3, fp=vals)
            vels_fin = np.diff(vals_fin)/wanted_precision
            all_trajs[tr, :len(vals_fin)] = vals_fin - vals_fin[0]
            all_vels[tr, :len(vels_fin)] = vels_fin - vels_fin[0]
        mean_traj = np.nanmedian(all_trajs, axis=0)
        std_traj = np.sqrt(np.nanstd(all_trajs, axis=0) / sum(index))
        mean_vel = np.nanmedian(all_vels, axis=0)
        std_vel = np.sqrt(np.nanstd(all_vels, axis=0) / sum(index))
        ax[0].plot(np.arange(len(mean_traj))*wanted_precision, mean_traj,
                   color=colormap[i_ev], label='{}'.format(bins[i_ev]))
        ax[0].fill_between(x=np.arange(len(mean_traj))*wanted_precision,
                           y1=mean_traj-std_traj, y2=mean_traj+std_traj,
                           color=colormap[i_ev])
        ax[1].plot(np.arange(len(mean_vel))*wanted_precision, mean_vel,
                   color=colormap[i_ev], label='{}'.format(bins[i_ev]))
        ax[1].fill_between(x=np.arange(len(mean_vel))*wanted_precision,
                           y1=mean_vel-std_vel, y2=mean_vel+std_vel,
                           color=colormap[i_ev])
    ax[0].set_xlim(-0.1, 550)
    ax[1].set_xlim(-0.1, 550)
    ax[0].axhline(y=200, linestyle='--', color='k', alpha=0.4)
    ax[1].axhline(y=1, linestyle='--', color='k', alpha=0.4)
    ax[0].legend(title='stimulus')
    ax[0].set_title('Median trajectory')
    ax[1].legend(title='stimulus')
    ax[1].set_title('Median velocity')
    # much stronger for prior (?)


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    subject = 'LE44'
    all_rats = True
    num_tr = int(15e4)
    f1 = False
    f2 = False
    f3 = True
    if f1:
        stim, zt, coh, gt, com, decision, sound_len, resp_len, hit,\
            trial_index, special_trial, traj_y, fix_onset, traj_stamps =\
            edd2.get_data_and_matrix(dfpath=DATA_FOLDER,
                                     num_tr_per_rat=int(1e4),
                                     after_correct=True, splitting=False,
                                     silent=False, all_trials=True,
                                     return_df=False, sv_folder=SV_FOLDER)
        data = {'stim': stim, 'zt': zt, 'coh': coh, 'gt': gt, 'com': com,
                'sound_len': sound_len, 'decision': decision,
                'resp_len': resp_len, 'hit': hit, 'trial_index': trial_index,
                'special_trial': special_trial, 'trajectory_y': traj_y,
                'trajectory_stamps': traj_stamps, 'fix_onset_dt': fix_onset}
        np.savez(DATA_FOLDER+'/sample_'+str(time.time())[-5:]+'.npz',
                 **data)
        # data = np.load(DATA_FOLDER+'/sample_73785.npz',
        #                allow_pickle=True)
        # stim = data['stim']
        # zt = data['zt']
        # coh = data['coh']
        # com = data['com']
        # gt = data['gt']
        # sound_len = data['sound_len']
        # resp_len = data['resp_len']
        # decision = data['decision']
        # hit = data['hit']
        # trial_index = data['trial_index']

        df_rat = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + subject,
                                          return_df=True, sv_folder=SV_FOLDER,
                                          after_correct=True, silent=True,
                                          all_trials=True)
        if stim.shape[0] != 20:
            stim = stim.T
        # FIG 1:
        df_data = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                                'norm_allpriors': zt/max(abs(zt)),
                                'R_response': (decision+1)/2,
                                'sound_len': sound_len,
                                'hithistory': hit})
        f, ax = plt.subplots(nrows=2, ncols=4, figsize=(8, 5))  # figsize=(4, 3))
        ax = ax.flatten()
        ax[0].axis('off')
        ax[1].axis('off')
        matrix_figure(df_data, ax_tach=ax[3], ax_pright=ax[2],
                      ax_mat=[ax[6], ax[7]], humans=False)
        plot_coms(df=df_rat, ax=ax[5])
        # ax_trck = plt.axes([.8, .55, .17, .17])
        ax_trck = ax[4]
        tracking_image(ax_trck)
        f.savefig(SV_FOLDER+'fig1.svg', dpi=400, bbox_inches='tight')

    if f2:
        # FIG 2
        existing_model_data = True
        if not existing_model_data:
            hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
                pro_vs_re =\
                fp.run_model(stim=stim, zt=zt, coh=coh, gt=gt,
                             trial_index=trial_index,
                             num_tr=int(2e5))
            idx = reaction_time >= 0
            df_data_model = pd.DataFrame({'avtrapz': coh[idx],
                                          'CoM_sugg': com_model_detected[idx],
                                          'norm_allpriors': zt[idx] /
                                          max(abs(zt[idx])),
                                          'R_response': (resp_fin[idx] + 1)/2,
                                          'sound_len': reaction_time[idx],
                                          'hithistory': hit_model[idx]})
        else:
            df_data_model = pd.read_csv(DATA_FOLDER + 'df_fig_1.csv')
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 5))  # (4, 3))
        ax = ax.flatten()
        fig4.fig4(ax=[ax[0], ax[3]])
        humans = False
        ax[0].axis('off')
        matrix_figure(df_data=df_data_model, humans=humans, ax_tach=ax[1],
                      ax_pright=ax[2], ax_mat=[ax[4], ax[5]])
        f.savefig(SV_FOLDER+'fig2.svg', dpi=400, bbox_inches='tight')
    if f3:
        # FIG 3:
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 5))  # figsize=(3, 3))
        ax = ax.flatten()
        ax[0].axis('off')
        fig_3(user_id='Alex', sv_folder=SV_FOLDER,
              ax_tach=ax[1], ax_pright=ax[3], ax_mat=[ax[4], ax[5]],
              ax_traj=ax[2], humans=True)
        f.savefig(SV_FOLDER+'fig3.svg', dpi=400, bbox_inches='tight')
