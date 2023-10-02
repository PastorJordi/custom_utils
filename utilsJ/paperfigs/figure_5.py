# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 01:01:54 2023

@author: Alex Garcia-Duran
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pl
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from scipy.stats import sem
import sys
import matplotlib

sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel
from utilsJ.paperfigs import figures_paper as fp
from utilsJ.Behavior.plotting import trajectory_thr, interpolapply,\
    binned_curve, com_heatmap
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_1 as fig_1
FRAME_RATE = 14
BINS_RT = np.linspace(1, 301, 11)
xpos_RT = int(np.diff(BINS_RT)[0])


def create_figure_5_model(fgsz):
    matplotlib.rcParams['font.size'] = 13
    plt.rcParams['legend.title_fontsize'] = 10.5
    plt.rcParams['xtick.labelsize'] = 10.5
    plt.rcParams['ytick.labelsize'] = 10.5
    fig, ax = plt.subplots(ncols=4, nrows=4,
                           gridspec_kw={'top': 0.95, 'bottom': 0.055, 'left': 0.07,
                                        'right': 0.95, 'hspace': 0.6, 'wspace': 0.6},
                           figsize=fgsz)
    ax = ax.flatten()
    labs = ['a', 'b', 'c', 'd',
            'e', 'f', 'g', 'k',
            'h', 'i', 'j', 'l',
            'm', 'n', '', 'o']
    # set correct size a, b panels
    pos_ax_0 = ax[0].get_position()
    # ax[0].set_position([pos_ax_0.x0+pos_ax_0.width/1.2, pos_ax_0.y0, pos_ax_0.width,
    #                     pos_ax_0.height])
    # pos_ax_0 = ax[2].get_position()
    # ax[2].set_position([pos_ax_0.x0-pos_ax_0.width/1.5, pos_ax_0.y0, pos_ax_0.width*1.4,
    #                     pos_ax_0.height])
    pos_ax_0 = ax[4].get_position()
    ax[4].set_position([pos_ax_0.x0, pos_ax_0.y0, pos_ax_0.width*0.9,
                        pos_ax_0.height*0.9])
    pos_ax_0 = ax[8].get_position()
    ax[8].set_position([pos_ax_0.x0, pos_ax_0.y0, pos_ax_0.width*0.9,
                        pos_ax_0.height*0.9])
    for i in [5, 9]:
        pos_ax_0 = ax[i].get_position()
        ax[i].set_position([pos_ax_0.x0-pos_ax_0.width/4, pos_ax_0.y0, pos_ax_0.width*1.15,
                            pos_ax_0.height])
    for i in [6, 10]:
        pos_ax_0 = ax[i].get_position()
        ax[i].set_position([pos_ax_0.x0-pos_ax_0.width/6, pos_ax_0.y0, pos_ax_0.width*1.15,
                            pos_ax_0.height])    
    # pos_ax_0 = ax[7].get_position()
    # ax[7].set_position([pos_ax_0.x0+pos_ax_0.width/5.2, pos_ax_0.y0, pos_ax_0.width,
    #                     pos_ax_0.height])
    # pos_ax_0 = ax[10].get_position()
    # ax[10].set_position([pos_ax_0.x0+pos_ax_0.width/5.2, pos_ax_0.y0, pos_ax_0.width,
    #                     pos_ax_0.height])
    # pos_ax_0 = ax[4].get_position()
    # ax[4].set_position([pos_ax_0.x0+pos_ax_0.width/5.2, pos_ax_0.y0, pos_ax_0.width,
    #                     pos_ax_0.height])
    # pos_ax_0 = ax[8].get_position()
    # ax[8].set_position([pos_ax_0.x0+pos_ax_0.width/4,
    #                     pos_ax_0.y0-pos_ax_0.height/2, pos_ax_0.width*4/5,
    #                     pos_ax_0.height])
    # pos_ax_0 = ax[10].get_position()
    # ax[10].set_position([pos_ax_0.x0 + pos_ax_0.width/10, pos_ax_0.y0,
    #                      pos_ax_0.width/2,
    #                      pos_ax_0.height])
    # ax_inset = plt.axes([pos_ax_0.x0 + pos_ax_0.width*0.8 + pos_ax_0.width/10,
    #                      pos_ax_0.y0, pos_ax_0.width/2, pos_ax_0.height])
    # ax[11].set_position([pos_ax_0.x0 + pos_ax_0.width*1.2 + pos_ax_0.width/10,
    #                      pos_ax_0.y0, pos_ax_0.width/2, pos_ax_0.height])
    # letters for panels
    for n, ax_1 in enumerate(ax):
        fp.rm_top_right_lines(ax_1)
        ax_1.text(-0.1, 1.2, labs[n], transform=ax_1.transAxes, fontsize=16,
                  fontweight='bold', va='top', ha='right')
    ax[0].set_ylabel('Stimulus Evidence')
    return fig, ax, ax[14], pos_ax_0


def plot_com_vs_rt_f5(df_plot_pcom, ax, ax2):
    subjid = df_plot_pcom.subjid
    subjects = np.unique(subjid)
    com_data = np.empty((len(subjects), len(BINS_RT)-1))
    com_data[:] = np.nan
    com_model_all = np.empty((len(subjects), len(BINS_RT)-1))
    com_model_all[:] = np.nan
    com_model_det = np.empty((len(subjects), len(BINS_RT)-1))
    com_model_det[:] = np.nan
    for i_s, subject in enumerate(subjects):
        df_plot = df_plot_pcom.loc[subjid == subject]
        xpos_plot, median_pcom_dat, _ =\
            binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                         errorbar_kw={'label': 'Data', 'color': 'k'}, ax=ax,
                         legend=False, return_data=True)
        xpos_plot, median_pcom_mod_det, _ =\
            binned_curve(df_plot, 'com_model_detected', 'rt_model', bins=BINS_RT,
                         xpos=xpos_RT, errorbar_kw={'label': 'Model detected',
                                                    'color': 'red'}, ax=ax,
                         legend=False, return_data=True)
        xpos_plot, median_pcom_mod_all, _ =\
            binned_curve(df_plot, 'com_model', 'rt_model', bins=BINS_RT,
                         xpos=xpos_RT,
                         errorbar_kw={'label': 'Model all', 'color': 'red',
                                      'linestyle': '--'},
                         ax=ax2, legend=False, return_data=True)
        com_data[i_s, :len(median_pcom_dat)] = median_pcom_dat
        com_model_all[i_s, :len(median_pcom_mod_all)] = median_pcom_mod_all
        com_model_det[i_s, :len(median_pcom_mod_det)] = median_pcom_mod_det
    xpos_plot = (BINS_RT[:-1] + BINS_RT[1:]) / 2
    ax.errorbar(xpos_plot, np.nanmedian(com_data, axis=0),
                yerr=np.nanstd(com_data, axis=0)/len(subjects), color='k')
    ax.errorbar(xpos_plot, np.nanmedian(com_model_det, axis=0),
                yerr=np.nanstd(com_model_det, axis=0)/len(subjects), color='r')
    ax2.errorbar(xpos_plot, np.nanmedian(com_model_all, axis=0),
                 yerr=np.nanstd(com_model_all, axis=0)/len(subjects), color='r',
                 linestyle='--')
    ax.xaxis.tick_top()
    ax.xaxis.tick_bottom()
    legendelements = [Line2D([0], [0], color='k', lw=2,
                             label='Rats'),
                      Line2D([0], [0], color='r', lw=2,
                             label='Model'),
                      Line2D([0], [0], color='r', lw=2, linestyle='--',
                             label='Model (All CoMs)')]
    ax.legend(handles=legendelements)
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('P(CoM)')
    ax2.set_ylabel('P(CoM)')
    ax2.set_xlabel('RT (ms)')


def plot_pright_model(df_sim, sound_len_model, decision_model, subjid, coh,
                      zt_model, ax):
    subjects = np.unique(subjid)
    coh_model = coh[sound_len_model >= 0]
    decision_01_model = (decision_model+1)/2
    mat_pright = np.zeros((7, 7, len(subjects)))
    for i_s, subject in enumerate(subjects):
        mat_per_subj, _ = com_heatmap(zt_model[subjid == subject],
                                      coh_model[subjid == subject],
                                      decision_01_model[subjid == subject],
                                      return_mat=True, annotate=False)
        mat_pright[:, :, i_s] = mat_per_subj
    mat_pright_avg = np.nanmean(mat_pright, axis=2)
    # P_right
    ax_pright = ax
    im = ax_pright.imshow(np.flipud(mat_pright_avg), vmin=0., vmax=1, cmap='PRGn_r')
    plt.sca(ax_pright)
    cbar = plt.colorbar(im, fraction=0.04)
    cbar.ax.set_title('p(Right)', pad=17)
    ax_pright.set_yticks([0, 3, 6])
    # ax_pright.set_ylim([-0.5, 6.5])
    ax_pright.set_yticklabels(['L', '', 'R'])
    ax_pright.set_xticks([0, 3, 6])
    # ax_pright.set_xlim([-0.5, 6.5])
    ax_pright.set_xticklabels(['L', '', 'R'])
    ax_pright.set_xlabel('Prior Evidence')
    ax_pright.set_ylabel('Stimulus Evidence')
    # ax[7].set_title('Pright Model')


def plot_pcom_matrices_model(df_model, n_subjs, ax_mat, pos_ax_0, f, nbins=7,
                             margin=.03):
    pos_ax_0 = ax_mat[1].get_position()
    ax_mat[1].set_position([pos_ax_0.x0-pos_ax_0.width/3, pos_ax_0.y0, pos_ax_0.width,
                           pos_ax_0.height])
    mat_side_0_all = np.zeros((7, 7, n_subjs))
    mat_side_1_all = np.zeros((7, 7, n_subjs))
    for i_s, subj in enumerate(df_model.subjid.unique()):
        matrix_side_0 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(
                df=df_model.loc[df_model.subjid == subj], side=0)
        matrix_side_1 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(
                df=df_model.loc[df_model.subjid == subj], side=1)
        mat_side_0_all[:, :, i_s] = matrix_side_0
        mat_side_1_all[:, :, i_s] = matrix_side_1
    matrix_side_0 = np.nanmean(mat_side_0_all, axis=2)
    matrix_side_1 = np.nanmean(mat_side_1_all, axis=2)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
    pcomlabel_0 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[0].set_title(pcomlabel_0)
    ax_mat[0].imshow(matrix_side_1, vmin=0, vmax=vmax, cmap='magma')
    # plt.sca(ax_mat[0])
    # plt.colorbar(im, fraction=0.04)
    # plt.sca(ax_mat[1])
    ax_mat[1].set_title(pcomlabel_1)
    im = ax_mat[1].imshow(matrix_side_0, vmin=0, vmax=vmax, cmap='magma')
    ax_mat[1].yaxis.set_ticks_position('none')
    for ax_i in [ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior Evidence')
        ax_i.set_xticks([0, 3, 6], ['-1', '0', '1'])
    ax_mat[0].set_yticks([0, 3, 6], ['1', '0', '-1'])
    ax_mat[1].set_yticks([0, 3, 6], ['']*3)
    ax_mat[0].set_ylabel('Stimulus Evidence')
    # ax_mat[0].set_position([pos_ax_0.x0 + pos_ax_0.width/10, pos_ax_0.y0,
    #                         pos_ax_0.width,
    #                         pos_ax_0.height])
    # ax_mat[1].set_position([pos_ax_0.x0 + pos_ax_0.width*0.6 + pos_ax_0.width/15,
    #                         pos_ax_0.y0, pos_ax_0.width, pos_ax_0.height])
    pos = ax_mat[1].get_position()
    cbar_ax = f.add_axes([pos.x0+pos.width+margin/2, pos.y0+margin/6,
                      pos.width/15, pos.height/1.5])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('p(detected CoM)', labelpad=12)


def plot_trajs_cond_on_prior_and_stim(df_sim, ax, inset_sz, fgsz, marginx, marginy,
                                      new_data, save_new_data, data_folder):
    ax_cohs = np.array([ax[9], ax[10], ax[8]])
    ax_zt = np.array([ax[5], ax[6], ax[4]])

    ax_inset = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_cohs = np.insert(ax_cohs, 3, ax_inset)

    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_zt = np.insert(ax_zt, 3, ax_inset)
    if sum(df_sim.special_trial == 2) > 0:
        traj_cond_coh_simul(df_sim=df_sim[df_sim.special_trial == 2], ax=ax_zt,
                            new_data=new_data, data_folder=data_folder,
                            save_new_data=save_new_data,
                            median=True, prior=True, rt_lim=300)
    else:
        print('No silent trials')
        traj_cond_coh_simul(df_sim=df_sim, ax=ax_zt, new_data=new_data,
                            save_new_data=save_new_data,
                            data_folder=data_folder, median=True, prior=True)
    traj_cond_coh_simul(df_sim=df_sim, ax=ax_cohs, median=True, prior=False,
                        save_new_data=save_new_data,
                        new_data=new_data, data_folder=data_folder,
                        prior_lim=np.quantile(df_sim.norm_allpriors.abs(), 0.2))


def mean_com_traj_simul(df_sim, data_folder, new_data, save_new_data, ax):
    raw_com = df_sim.CoM_sugg.values
    index_com = df_sim.com_detected.values
    trajs_all = df_sim.trajectory_y.values
    dec = df_sim.R_response.values*2-1
    max_ind = 800
    subjects = df_sim.subjid.unique()
    matrix_com_tr = np.empty((len(subjects), max_ind))
    matrix_com_tr[:] = np.nan
    matrix_com_und_tr = np.empty((len(subjects), max_ind))
    matrix_com_und_tr[:] = np.nan
    matrix_nocom_tr = np.empty((len(subjects), max_ind))
    matrix_nocom_tr[:] = np.nan
    for i_s, subject in enumerate(subjects):
        traj_data = data_folder+subject+'/sim_data/'+subject +\
            '_mean_com_trajs.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data) and not new_data:
            traj_data = np.load(traj_data, allow_pickle=True)
            mean_com_und_traj = traj_data['mean_com_und_traj']
            mean_nocom_tr = traj_data['mean_nocom_tr']
            mean_com_traj = traj_data['mean_com_traj']
        else:
            it_subs = np.where(df_sim.subjid.values == subject)[0][0]
            i_com = 0
            i_nocom = 0
            i_und_com = 0
            mat_nocom_erase = np.empty((sum(~(raw_com)), max_ind))
            mat_nocom_erase[:] = np.nan
            mat_com_erase = np.empty((sum(index_com), max_ind))
            mat_com_erase[:] = np.nan
            mat_com_und_erase = np.empty((sum((~index_com) & (raw_com)), max_ind))
            mat_com_und_erase[:] = np.nan
            for i_t, traj in enumerate(trajs_all[df_sim.subjid == subject]):
                if index_com[i_t+it_subs]:
                    mat_com_erase[i_com, :len(traj)] = traj*dec[i_t+it_subs]
                    i_com += 1
                if not index_com[i_t+it_subs] and not raw_com[i_t]:
                    mat_nocom_erase[i_nocom, :len(traj)] = traj*dec[i_t+it_subs]
                    i_nocom += 1
                if raw_com[i_t+it_subs] and not index_com[i_t+it_subs]:
                    mat_com_und_erase[i_und_com, :len(traj)] = traj*dec[i_t+it_subs]
                    i_und_com += 1
            mean_com_traj = np.nanmean(mat_com_erase, axis=0)
            mean_nocom_tr = np.nanmean(mat_nocom_erase, axis=0)
            mean_com_und_traj = np.nanmean(mat_com_und_erase, axis=0)
        if save_new_data:
            data = {'mean_com_traj': mean_com_traj, 'mean_nocom_tr': mean_nocom_tr,
                    'mean_com_und_traj': mean_com_und_traj}
            np.savez(traj_data, **data)
        matrix_com_tr[i_s, :len(mean_com_traj)] = mean_com_traj
        matrix_nocom_tr[i_s, :len(mean_nocom_tr)] = mean_nocom_tr
        matrix_com_und_tr[i_s, :len(mean_com_und_traj)] = mean_com_und_traj
        ax.plot(np.arange(len(mean_com_traj)), mean_com_traj, color=fig_3.COLOR_COM,
                linewidth=1.4, alpha=0.25)
    mean_com_traj = np.nanmean(matrix_com_tr, axis=0)
    mean_nocom_traj = np.nanmean(matrix_nocom_tr, axis=0)
    mean_com_all_traj = np.nanmean(matrix_com_und_tr, axis=0)
    ax.plot(np.arange(len(mean_com_traj)), mean_com_traj, color=fig_3.COLOR_COM,
            linewidth=2)
    ax.plot(np.arange(len(mean_com_all_traj)), mean_com_all_traj, color=fig_3.COLOR_COM,
            linewidth=1.4, linestyle='--')
    ax.plot(np.arange(len(mean_nocom_traj)), mean_nocom_traj, color=fig_3.COLOR_NO_COM,
            linewidth=2)
    legendelements = [Line2D([0], [0], color=fig_3.COLOR_COM, lw=2,
                             label='Detected Rev.'),
                      Line2D([0], [0], color=fig_3.COLOR_COM, lw=1.5,  linestyle='--',
                             label='All Rev.'),
                      Line2D([0], [0], color=fig_3.COLOR_NO_COM, lw=2,
                             label='No-Rev.')]
    ax.legend(handles=legendelements, loc='upper left')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Position')
    ax.set_xlim(-25, 400)
    ax.set_ylim(-25, 80)
    ax.axhline(-8, color='r', linestyle=':')
    ax.text(200, -19, "Detection threshold", color='r')


def traj_cond_coh_simul(df_sim, data_folder, new_data, save_new_data,
                        ax=None, median=True, prior=True, prior_lim=1, rt_lim=200):
    # TODO: save each matrix? or save the mean and std
    df_sim = df_sim[df_sim.sound_len >= 0]
    if median:
        func_final = np.nanmedian
    if not median:
        func_final = np.nanmean
    # nanidx = df_sim.loc[df_sim.allpriors.isna()].index
    # df_sim.loc[nanidx, 'allpriors'] = np.nan
    df_sim['choice_x_coh'] = (df_sim.R_response*2-1) * df_sim.coh2
    df_sim['choice_x_zt'] = (df_sim.R_response*2-1) * df_sim.norm_allpriors
    bins_coh = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    bins_zt = [1.01]
    # TODO: fix issue with equipopulated bins
    for i_p, perc in enumerate([0.75, 0.5, 0.25, 0.25, 0.5, 0.75]):
        if i_p < 3:
            bins_zt.append(df_sim.norm_allpriors.abs().quantile(perc))
        else:
            bins_zt.append(-df_sim.norm_allpriors.abs().quantile(perc))
    bins_zt.append(-1.01)
    bins_zt = bins_zt[::-1]
    if prior:
        condition = 'choice_x_prior'
        bins_zt, _, _, _, _ =\
              fp.get_bin_info(df=df_sim, condition=condition, prior_limit=1,
                              after_correct_only=True)
    # xvals_zt = [-1, -0.5, 0, 0.5, 1]
    xvals_zt = np.linspace(-1, 1, 5)
    signed_response = df_sim.R_response.values
    # df_sim['normallpriors'] = df_sim['allpriors'] /\
    #     np.nanmax(df_sim['allpriors'].abs())*(signed_response*2 - 1)
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
    labels_zt = ['inc.', ' ', ' ', '0', ' ', ' ', 'cong.']
    labels_coh = ['-1', ' ', ' ', '0', ' ', ' ', '1']
    if prior:
        bins_ref = bins_zt
        colormap = pl.cm.copper(np.linspace(0, 1, len(bins_zt)-1))
    else:
        bins_ref = bins_coh
        colormap = pl.cm.coolwarm(np.linspace(0, 1, len(bins_coh)))
    subjects = df_sim.subjid
    max_mt = 1200
    mat_trajs_subs = np.empty((len(bins_ref), max_mt,
                               len(subjects.unique())))
    mat_vel_subs = np.empty((len(bins_ref), max_mt,
                             len(subjects.unique())))
    mat_trajs_indsub = np.empty((len(bins_ref), max_mt))
    mat_vel_indsub = np.empty((len(bins_ref), max_mt))
    if prior:
        val_traj_subs = np.empty((len(bins_ref)-1, len(subjects.unique())))
        val_vel_subs = np.empty((len(bins_ref)-1, len(subjects.unique())))
        label_save = 'prior'
    else:
        val_traj_subs = np.empty((len(bins_ref), len(subjects.unique())))
        val_vel_subs = np.empty((len(bins_ref), len(subjects.unique())))
        label_save = 'stim'
    for i_s, subject in enumerate(subjects.unique()):
        traj_data = data_folder+subject+'/sim_data/'+subject +\
            '_traj_sim_pos_'+label_save+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data) and not new_data:
            traj_data = np.load(traj_data, allow_pickle=True)
            vals_thr_vel = traj_data['vals_thr_vel']
            vals_thr_traj = traj_data['vals_thr_traj']
            mat_trajs_indsub = traj_data['mat_trajs_indsub']
            mat_vel_indsub = traj_data['mat_vel_indsub']
        else:
            vals_thr_traj = []
            vals_thr_vel = []
            lens = []
            for i_ev, ev in enumerate(bins_ref):
                if not prior:
                    index = (df_sim.choice_x_coh.values == ev) &\
                        (df_sim.normallpriors.abs() <= prior_lim) &\
                        (df_sim.special_trial == 0) & (~np.isnan(df_sim.allpriors)) *\
                        (df_sim.sound_len >= 0) & (df_sim.sound_len <= rt_lim) &\
                        (subjects == subject)
                if prior:
                    if i_ev == len(bins_ref)-1:
                        break
                    index = (df_sim.choice_x_zt.values >= bins_ref[i_ev]) &\
                        (df_sim.choice_x_zt.values < bins_ref[i_ev + 1]) &\
                        (df_sim.sound_len >= 0) & (df_sim.sound_len <= rt_lim) &\
                        (subjects == subject)
                    if sum(index) == 0:
                        continue
                lens.append(max([len(t) for t in df_sim.trajectory_y[index].values]))
                traj_all = np.empty((sum(index), max_mt))
                traj_all[:] = np.nan
                vel_all = np.empty((sum(index), max_mt))
                vel_all[:] = np.nan
                for tr in range(sum(index)):
                    vals_traj = df_sim.traj[index].values[tr] *\
                        (signed_response[index][tr]*2 - 1)
                    if sum(vals_traj) == 0:
                        continue
                    vals_traj = np.concatenate((vals_traj,
                                                np.repeat(75, max_mt-len(vals_traj))))
                    vals_vel = df_sim.traj_d1[index].values[tr] *\
                        (signed_response[index][tr]*2 - 1)
                    vals_vel = np.diff(vals_traj)
                    traj_all[tr, :len(vals_traj)] = vals_traj
                    vel_all[tr, :len(vals_vel)] = vals_vel
                try:
                    index_vel = np.where(np.sum(np.isnan(traj_all), axis=0)
                                          > traj_all.shape[0] - 50)[0][0]
                    mean_traj = func_final(traj_all[:, :index_vel], axis=0)
                    std_traj = np.nanstd(traj_all[:, :index_vel],
                                          axis=0) / np.sqrt(len(subjects.unique()))
                except Exception:
                    mean_traj = func_final(traj_all, axis=0)
                    std_traj = np.nanstd(traj_all, axis=0) /\
                        np.sqrt(len(subjects.unique()))
                val_traj = np.mean(df_sim['resp_len'].values[index])*1e3
                vals_thr_traj.append(val_traj)
                mean_vel = func_final(vel_all, axis=0)
                std_vel = np.nanstd(vel_all, axis=0) / np.sqrt(len(subjects.unique()))
                val_vel = np.nanmax(mean_vel)  # func_final(np.nanmax(vel_all, axis=1))
                vals_thr_vel.append(val_vel)
                mat_trajs_indsub[i_ev, :len(mean_traj)] = mean_traj
                mat_vel_indsub[i_ev, :len(mean_vel)] = mean_vel
            if save_new_data:
                data = {'mat_trajs_indsub': mat_trajs_indsub,
                        'mat_vel_indsub': mat_vel_indsub,
                        'vals_thr_traj': vals_thr_traj, 'vals_thr_vel': vals_thr_vel}
                np.savez(traj_data, **data)
        mat_trajs_subs[:, :, i_s] = mat_trajs_indsub
        mat_vel_subs[:, :, i_s] = mat_vel_indsub
        val_traj_subs[:len(vals_thr_traj), i_s] = vals_thr_traj
        val_vel_subs[:len(vals_thr_vel), i_s] = vals_thr_vel
    for i_ev, ev in enumerate(bins_ref):
        if prior and ev == 1.01:
            break
        val_traj = np.nanmean(val_traj_subs[i_ev, :])
        std_mt = np.nanstd(val_traj_subs[i_ev, :]) /\
            np.sqrt(len(subjects.unique()))
        val_vel = np.nanmean(val_vel_subs[i_ev, :])
        std_vel_points = np.nanstd(val_vel_subs[i_ev, :]) /\
            np.sqrt(len(subjects.unique()))
        mean_traj = np.nanmean(mat_trajs_subs[i_ev, :, :], axis=1)
        std_traj = np.std(mat_trajs_subs[i_ev, :, :], axis=1) /\
            np.sqrt(len(subjects.unique()))
        mean_vel = np.nanmean(mat_vel_subs[i_ev, :, :], axis=1)
        std_vel = np.std(mat_vel_subs[i_ev, :, :], axis=1) /\
            np.sqrt(len(subjects.unique()))
        if prior:
            xval = xvals_zt[i_ev]
        else:
            xval = ev
        ax[2].errorbar(xval, val_traj, std_mt, color=colormap[i_ev], marker='o')
        ax[3].errorbar(xval, val_vel, std_vel_points, color=colormap[i_ev],
                       marker='o')
        if not prior:
            label = labels_coh[i_ev]
        if prior:
            label = labels_zt[i_ev]
        ax[0].plot(np.arange(len(mean_traj)), mean_traj, label=label,
                   color=colormap[i_ev])
        ax[0].fill_between(x=np.arange(len(mean_traj)),
                           y1=mean_traj - std_traj, y2=mean_traj + std_traj,
                           color=colormap[i_ev], alpha=0.3)
        ax[1].plot(np.arange(len(mean_vel)), mean_vel, label=label,
                   color=colormap[i_ev])
        ax[1].fill_between(x=np.arange(len(mean_vel)),
                           y1=mean_vel - std_vel, y2=mean_vel + std_vel,
                           color=colormap[i_ev], alpha=0.3)
    ax[0].axhline(y=75, linestyle='--', color='k', alpha=0.4)
    ax[0].set_xlim(-5, 335)
    ax[0].set_yticks([0, 25, 50, 75])
    ax[0].set_ylim(-10, 85)
    ax[1].set_ylim(-0.08, 0.68)
    ax[1].set_xlim(-5, 335)
    if prior:
        leg_title = 'Prior'
        ax[2].plot(xvals_zt, np.nanmean(val_traj_subs, axis=1),
                   color='k', linestyle='--', alpha=0.6)
        ax[3].plot(xvals_zt, np.nanmean(val_vel_subs, axis=1),
                   color='k', linestyle='--', alpha=0.6)
        ax[2].set_xlabel('Prior')
        ax[3].set_xlabel('Prior')
    if not prior:
        leg_title = 'Stimulus'
        ax[2].plot(bins_coh, np.nanmean(val_traj_subs, axis=1),
                   color='k', linestyle='--', alpha=0.6)
        ax[3].plot(bins_coh,  np.nanmean(val_vel_subs, axis=1),
                   color='k', linestyle='--', alpha=0.6)
        ax[2].set_xlabel('Stimulus')
        ax[3].set_xlabel('Stimulus')
    ax[0].legend(title=leg_title, labelspacing=0.15,
                 loc='center left', bbox_to_anchor=(0.8, 0.45))
    ax[0].set_ylabel('Position')
    ax[0].set_xlabel('Time from movement onset (ms)')
    # ax[0].set_title('Mean trajectory', fontsize=10)
    # ax[1].legend(title=leg_title)
    ax[1].set_ylabel('Velocity')
    ax[1].set_xlabel('Time from movement onset (ms)')
    # ax[1].set_title('Mean velocity', fontsize=8)
    ax[2].set_ylabel('MT (ms)')
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_ylabel('Peak')



def fig_5_model(sv_folder, data_folder, new_data, save_new_data,
                coh, sound_len, hit_model, sound_len_model, zt,
                decision_model, com, com_model, com_model_detected,
                df_sim, means, errors, means_model, errors_model, inset_sz=.06,
                marginx=-0.02, marginy=0.08, fgsz=(13, 12)):
    fig, ax, ax_inset, pos_ax_0 = create_figure_5_model(fgsz=fgsz)
    # select RT > 0 (no FB, as in data)
    hit_model = hit_model[sound_len_model >= 0]
    com_model_detected = com_model_detected[sound_len_model >= 0]
    decision_model = decision_model[sound_len_model >= 0]
    com_model = com_model[sound_len_model >= 0]
    subjid = df_sim.subjid.values
    # Tachometrics
    _ = fp.tachometric_data(coh=coh[sound_len_model >= 0], hit=hit_model,
                            sound_len=sound_len_model[sound_len_model >= 0],
                            subjid=subjid, ax=ax[1], label='', legend=False)
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, 4))
    legendelements = [Line2D([0], [0], color=colormap[3], lw=2,
                             label='1'),
                      Line2D([0], [0], color=colormap[2], lw=2,
                             label='0.5'),
                      Line2D([0], [0], color=colormap[1], lw=2,
                             label='0.25'),
                      Line2D([0], [0], color=colormap[0], lw=2,
                             label='0')]
    ax[1].legend(handles=legendelements, fontsize=8, loc='center left',
                 bbox_to_anchor=(0.96, 0.5), title='Stimulus')
    # ax2 = fp.add_inset(ax=ax[13], inset_sz=inset_sz, fgsz=fgsz, marginx=marginx,
    #                    marginy=0.07, right=True)
    df_plot_pcom = pd.DataFrame({'com': com[sound_len_model >= 0],
                                 'sound_len': sound_len[sound_len_model >= 0],
                                 'rt_model': sound_len_model[sound_len_model >= 0],
                                 'com_model': com_model, 'subjid': subjid,
                                 'com_model_detected': com_model_detected})
    zt_model = df_sim.norm_allpriors.values
    # PCoM vs RT
    plot_com_vs_rt_f5(df_plot_pcom=df_plot_pcom, ax=ax[-1], ax2=ax[-1])
    # slowing in MT
    fig_1.plot_mt_vs_stim(df=df_sim, ax=ax[3], prior_min=0.1, rt_max=50)
    # P(right) matrix
    plot_pright_model(df_sim=df_sim, sound_len_model=sound_len_model,
                      decision_model=decision_model, subjid=subjid, coh=coh,
                      zt_model=zt_model, ax=ax[0])
    df_model = pd.DataFrame({'avtrapz': coh[sound_len_model >= 0],
                             'CoM_sugg': com_model_detected,
                             'norm_allpriors': zt_model/max(abs(zt_model)),
                             'R_response': (decision_model+1)/2, 'subjid': subjid})
    df_model = df_model.loc[~df_model.norm_allpriors.isna()]
    nbins = 7
    # plot Pcoms matrices
    ax_mat = [ax[13], ax_inset]
    n_subjs = len(df_sim.subjid.unique())
    # PCoM matrices
    plot_pcom_matrices_model(df_model=df_model, n_subjs=n_subjs,
                             ax_mat=ax_mat, pos_ax_0=pos_ax_0, nbins=nbins,
                             f=fig)
    # MT matrix vs stim/prior
    fig_1.mt_matrix_ev_vs_zt(df_sim, ax[2], f=fig, silent_comparison=False,
                             collapse_sides=True, margin=0.03)
    # MT distributions
    fig_3.mt_distros(df=df_sim, ax=ax[11], xlmax=625)
    # plot trajs and MT conditioned on stim/prior
    plot_trajs_cond_on_prior_and_stim(df_sim=df_sim, ax=ax, new_data=new_data,
                                      save_new_data=save_new_data,
                                      inset_sz=inset_sz, data_folder=data_folder,
                                      fgsz=fgsz, marginx=marginx, marginy=marginy)
    # plot splitting time vs RT
    fig_2.trajs_splitting_stim(df_sim.loc[df_sim.special_trial == 0],
                               data_folder=data_folder, ax=ax[7], collapse_sides=True,
                               threshold=500, sim=True, rtbins=np.linspace(0, 150, 16),
                               connect_points=True, trajectory="trajectory_y")
    # plot mean com traj
    
    if len(df_sim.subjid.unique()) > 1:
        subject = ''
    else:
        subject = df_sim.subjid.unique()[0]
    fig.savefig(sv_folder+subject+'/fig5.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+subject+'/fig5.png', dpi=400, bbox_inches='tight')
    mean_com_traj_simul(df_sim, ax=ax[12], data_folder=data_folder, new_data=new_data,
                        save_new_data=save_new_data)
    fig.savefig(sv_folder+subject+'/fig5.png', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+subject+'/fig5.svg', dpi=400, bbox_inches='tight')


def fig_5_part_1(sv_folder, data_folder,
                 coh, sound_len, hit_model, sound_len_model, zt,
                 decision_model, com, com_model, com_model_detected,
                 df_sim, inset_sz=.06,
                 marginx=0.006, marginy=0.07, fgsz=(8, 18)):
    fig, ax = plt.subplots(2, 3)
    ax = ax.flatten()
    # select RT > 0 (no FB, as in data)
    hit_model = hit_model[sound_len_model >= 0]
    com_model_detected = com_model_detected[sound_len_model >= 0]
    decision_model = decision_model[sound_len_model >= 0]
    com_model = com_model[sound_len_model >= 0]
    subjid = df_sim.subjid.values
    zt_model = df_sim.norm_allpriors.values
    # P(right) matrix
    plot_pright_model(df_sim=df_sim, sound_len_model=sound_len_model,
                      decision_model=decision_model, subjid=subjid, coh=coh,
                      zt_model=zt_model, ax=ax[0])
    # Tachometrics
    _ = fp.tachometric_data(coh=coh[sound_len_model >= 0], hit=hit_model,
                            sound_len=sound_len_model[sound_len_model >= 0],
                            subjid=subjid, ax=ax[1], label='')
    # MT matrix vs stim/prior
    fig_1.mt_matrix_ev_vs_zt(df_sim, ax[5], f=fig, silent_comparison=False,
                             collapse_sides=True)
    # MT vs stim/prior
    fig2, ax2 = plt.subplots(1)
    ax2 = ax2.flatten()
    ax_final = [ax2, ax2, ax[3], ax[4], ax2, ax2, ax2, ax2]
    plot_trajs_cond_on_prior_and_stim(df_sim=df_sim, ax=ax_final, new_data=False,
                                      save_new_data=False,
                                      inset_sz=inset_sz, data_folder=data_folder,
                                      fgsz=fgsz, marginx=marginx, marginy=marginy)
