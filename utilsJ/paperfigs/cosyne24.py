# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:02:05 2023

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
from utilsJ.Models import analyses_humans as ah
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.paperfigs import figure_5 as fig_5
from utilsJ.paperfigs import fig_5_humans as fig_5h
from utilsJ.paperfigs import figure_6 as fig_6
from utilsJ.paperfigs import figures_paper as fp
from utilsJ.Behavior.plotting import tachometric, com_heatmap


matplotlib.rcParams['font.size'] = 11
plt.rcParams['legend.title_fontsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize']= 13
plt.rcParams['ytick.labelsize']= 13
matplotlib.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/001965.png'
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
    RAT_noCOM_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/screenShot230120.png'
    TASK_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/panel_a.png'
    HUMAN_TASK_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/Human/panel_a.png'
    REPALT_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/repalt.png'
    ST_CARTOON_IMG =\
        'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/st_cartoon_violins.png'
elif pc_name == 'idibaps':
    DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
    SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
        'ChangesOfMind/figures/from_python/'  # Manuel
    RAT_noCOM_IMG = '/home/molano/Dropbox/project_Barna/' +\
        'ChangesOfMind/figures/Figure_1/screenShot230120.png'
    RAT_COM_IMG = '/home/molano/Dropbox/project_Barna/' +\
        'ChangesOfMind/figures/Figure_3/001965.png'
    TASK_IMG = '/home/molano/Dropbox/project_Barna/ChangesOfMind/' +\
        'figures/Figure_1/panel_a.png'
    ST_CARTOON_IMG ='/home/molano/Dropbox/project_Barna/ChangesOfMind/' +\
        'figures/Figure_2/st_cartoon_violins.png'
elif pc_name == 'alex_CRM':
    SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
    RAT_COM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/001965.png'
    RAT_noCOM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/screenShot230120.png'
    HUMAN_TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/rat_image/g41085.png'
    TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/figures/panel_a.png'
    REPALT_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/figures/repalt.png'


def fig_1_cosyne(df, sv_folder, data_folder, task_img, repalt_img,
                 figsize=(12, 5), inset_sz=.1, marginx=-.04, marginy=0.1):
    figure, ax = plt.subplots(nrows=2, ncols=5, figsize=figsize)  # figsize=(4, 3))
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.08, right=0.96,
                        hspace=0.6, wspace=0.6)
    ax = ax.flatten()
    # TUNE PANELS
    # all panels
    letters = ['', '',  '', 'c', 'd', 'e', 'f', 'g', 'h', '']
    for n, ax_1 in enumerate(ax):
        fp.add_text(ax=ax_1, letter=letters[n], x=-0.12, y=1.25)
        if n not in [10, 11]:
            fp.rm_top_right_lines(ax_1)

    for i in [0, 1]:
        ax[i].axis('off')
    for i in [3, 4, 5]:
        pos_ax = ax[i].get_position()
        ax[i].set_position([pos_ax.x0, pos_ax.y0+0.02,
                            pos_ax.width, pos_ax.height])
    # task panel
    ax_task = ax[0]
    pos_task = ax_task.get_position()
    factor = 2.2
    ax_task.set_position([pos_task.x0-0.05, pos_task.y0-0.11,
                          pos_task.width*factor, pos_task.height*factor])
    fp.add_text(ax=ax_task, letter='a', x=0.1, y=1.14)
    # rep-alt img
    ax_repalt = ax[1]
    pos_repalt = ax_repalt.get_position()
    factor = 1
    ax_repalt.set_position([pos_repalt.x0+pos_repalt.width/5,
                            pos_repalt.y0+0.04,
                            pos_repalt.width*factor, pos_repalt.height*factor])
    # TASK PANEL
    task = plt.imread(task_img)
    ax_task.imshow(task)
    # REPALT PANEL
    task = plt.imread(repalt_img)
    ax_repalt.imshow(task)
    # PRIGHT PANEL
    ax_pright = ax[2]
    pos_pright = ax_pright.get_position()
    factor = 0.9
    ax_pright.set_position([pos_pright.x0-pos_pright.width/8,
                            pos_pright.y0 - 0.08,
                            pos_pright.width*factor,
                            pos_pright.height*factor])
    pos_pright = ax_pright.get_position()
    ax_pright.set_yticks([0, 3, 6])
    ax_pright.set_ylim([-0.5, 6.5])
    ax_pright.set_yticklabels(['L', '', 'R'])
    ax_pright.set_xticks([0, 3, 6])
    ax_pright.set_xlim([-0.5, 6.5])
    ax_pright.set_xticklabels(['L', '', 'R'])
    ax_pright.set_xlabel('Prior evidence')
    ax_pright.set_ylabel('Stimulus evidence')
    fp.add_text(ax=ax_pright, letter='b', x=-0.17, y=1.3)
    # P(RIGHT) MATRIX
    mat_pright_all = np.zeros((7, 7))
    for subject in df.subjid.unique():
        df_sbj = df.loc[(df.special_trial == 0) &
                        (df.subjid == subject)]
        choice = df_sbj['R_response'].values
        coh = df_sbj['coh2'].values
        prior = df_sbj['norm_allpriors'].values
        indx = ~np.isnan(prior)
        mat_pright, _ = com_heatmap(prior[indx], coh[indx], choice[indx],
                                    return_mat=True, annotate=False)
        mat_pright_all += mat_pright
    mat_pright = mat_pright_all / len(df.subjid.unique())

    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    # cbar = plt.colorbar(im_2, cax=pright_cbar_ax, orientation='horizontal')
    cbar = figure.colorbar(im_2, ax=ax_pright, location='top', label='p (right response)',
                           shrink=0.7, aspect=15)
    im = ax_pright.images
    cb = im[-1].colorbar
    pos_cb = cb.ax.get_position()
    pos_pright = ax_pright.get_position()
    factor = 1.2
    ax_pright.set_position([pos_pright.x0,
                            pos_pright.y0 + 0.01,
                            pos_pright.width*factor,
                            pos_pright.height*factor])
    cb.ax.set_position([pos_cb.x0-pos_pright.width/3+0.05,
                        pos_cb.y0+0.055, pos_cb.width, pos_cb.height])
    # pright_cbar_ax.set_title('p (right)')
    cbar.ax.tick_params(rotation=45)
    # change pos stim panels
    pos_ax4 = ax[4].get_position()
    ax[4].set_position([pos_ax4.x0+pos_ax4.width/4.5, pos_ax4.y0,
                        pos_ax4.width, pos_ax4.height])
    pos_ax4 = ax[3].get_position()
    ax[3].set_position([pos_ax4.x0+pos_ax4.width/4.5, pos_ax4.y0,
                        pos_ax4.width, pos_ax4.height])
    pos_ax7 = ax[6].get_position()
    ax[6].set_position([pos_ax7.x0-pos_ax7.width/4.5, pos_ax7.y0,
                        pos_ax7.width, pos_ax7.height])
    pos_ax5 = ax[5].get_position()
    ax[5].set_position([pos_ax5.x0-pos_ax5.width/6, pos_ax7.y0,
                        pos_ax5.width, pos_ax5.height])
    # VIGOR PANEL
    ax_mt_zt = ax[3]
    ax_mt_coh = ax[4]
    # MT VS PRIOR
    df_mt = df.copy()
    fig_1.plot_mt_vs_evidence(df=df_mt.loc[df_mt.special_trial == 2], ax=ax_mt_zt,
                              condition='choice_x_prior', prior_limit=1,
                              rt_lim=200)
    del df_mt
    # MT VS COH
    df_mt = df.copy()
    fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax_mt_coh, prior_limit=0.1,  # 10% quantile
                              condition='choice_x_coh', rt_lim=50)
    del df_mt
    ax_mt_coh.set_xlabel('Stimulus evidence towards\nresponse')
    ax_mt_zt.set_xlabel('Prior evidence towards\nresponse')
    # trajs
    # TRAJECTORIES CONDITIONED ON PRIOR
    # add insets
    ax = figure.axes
    fig, ax2 = plt.subplots(1)
    ax_zt = np.array([ax[5], ax2])
    ax_cohs = np.array([ax[6], ax2])
    ax_inset = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=figsize,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    # ax_cohs contains in this order the axes for:
    # index 0: mean position of rats conditioned on stim. evidence,
    # index 1: the inset for the velocity panel 
    # index 2: mean velocity  of rats conditioned on stim. evidence
    ax_cohs = np.insert(ax_cohs, 1, ax_inset)
    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=figsize,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_zt = np.insert(ax_zt, 1, ax_inset)
    fig_2.plots_trajs_conditioned(df=df.loc[df.special_trial == 2],
                                  ax=ax_zt, data_folder=data_folder,
                                  condition='choice_x_prior',
                                  prior_limit=1, cmap='copper')
    # TRAJECTORIES CONDITIONED ON COH
    fig_2. plots_trajs_conditioned(df=df, ax=ax_cohs,
                                   data_folder=data_folder,
                                   condition='choice_x_coh',
                                   prior_limit=0.1,  # 10% quantile
                                   cmap='coolwarm')
    ax[6].set_yticks([0, 25, 50, 75], ['', '', '', ''])
    ax[6].set_ylabel('')
    ax[4].set_ylabel('')
    ax[5].set_xlabel('Time from movement onset\n (ms)')
    ax[6].set_xlabel('Time from movement onset\n (ms)')
    plt.close(fig)
    ax_mat_r = ax[9]
    ax_mat_l = ax[8]
    # pos_ax_mat = ax_mat_r.get_position()
    # factor = 1.5
    # ax_mat_r.set_position([pos_ax_mat.x0+pos_ax_mat.width/5,
    #                        pos_ax_mat.y0+pos_ax_mat.height/12,
    #                        pos_ax_mat.width/factor,
    #                        pos_ax_mat.height/factor])
    # ax_mat_l = figure.add_axes([pos_ax_mat.x0-pos_ax_mat.width/2,
    #                             pos_ax_mat.y0+pos_ax_mat.height/12,
    #                             pos_ax_mat.width/factor,
    #                             pos_ax_mat.height/factor])
    ax_mat = [ax_mat_l, ax_mat_r]
    pos_ax4 = ax_mat_l.get_position()
    ax_mat_l.set_position([pos_ax4.x0+pos_ax4.width/4.5, pos_ax4.y0,
                        pos_ax4.width, pos_ax4.height])
    pos_ax4 = ax_mat_r.get_position()
    ax_mat_r.set_position([pos_ax4.x0+pos_ax4.width/4.5, pos_ax4.y0,
                        pos_ax4.width, pos_ax4.height])
    # PCOM MATRICES
    n_subjs = len(df.subjid.unique())
    mat_side_0_all = np.zeros((7, 7, n_subjs))
    mat_side_1_all = np.zeros((7, 7, n_subjs))
    for i_s, subj in enumerate(df.subjid.unique()):
        matrix_side_0 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(df=df.loc[df.subjid == subj],
                                                     side=0)
        matrix_side_1 =\
            fig_3.com_heatmap_marginal_pcom_side_mat(df=df.loc[df.subjid == subj],
                                                     side=1)
        mat_side_0_all[:, :, i_s] = matrix_side_0
        mat_side_1_all[:, :, i_s] = matrix_side_1
    matrix_side_0 = np.nanmean(mat_side_0_all, axis=2)
    matrix_side_1 = np.nanmean(mat_side_1_all, axis=2)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Right to left'  # r'$p(CoM_{L \rightarrow R})$'
    pcomlabel_0 = 'Left to right'   # r'$p(CoM_{L \rightarrow R})$x'
    ax_mat[0].set_title(pcomlabel_0, fontsize=10)
    im = ax_mat[0].imshow(np.flipud(matrix_side_1), vmin=0, vmax=vmax, cmap='magma')
    ax_mat[1].set_title(pcomlabel_1, fontsize=10)
    im = ax_mat[1].imshow(np.flipud(matrix_side_0), vmin=0, vmax=vmax, cmap='magma')
    ax_mat[1].yaxis.set_ticks_position('none')
    margin = 0.01
    for ax_i in [ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior evidence')
        ax_i.set_xticks([0, 3, 6])
        ax_i.set_xticklabels(['L', '0', 'R'])
        ax_i.set_ylim([-.5, 6.5])
    ax_mat[0].set_yticks([0, 3, 6])
    ax_mat[0].set_yticklabels(['L', '0', 'R'])
    ax_mat[1].set_yticks([])
    pos = ax_mat_r.get_position()
    pright_cbar_ax = figure.add_axes([pos.x0+pos.width*1.07, pos.y0,
                                      pos.width/12, pos.height/2])
    cbar = fig.colorbar(im, cax=pright_cbar_ax)
    cbar.ax.set_title('       p(rev.)', fontsize=8)
    ax_mat[0].set_ylabel('Stimuluse evidence')
    pos_ax5 = ax[5].get_position()
    pos_ax7 = ax[6].get_position()
    ax[6].set_position([pos_ax7.x0, pos_ax5.y0,
                        pos_ax7.width, pos_ax7.height])
    # ax_mean_com = ax[5]
    fig_3.mean_com_traj(df=df, ax=ax[7], data_folder=data_folder,
                        condition='choice_x_prior',
                        prior_limit=1, after_correct_only=True, rt_lim=400,
                        trajectory='trajectory_y',
                        interpolatespace=np.linspace(-700000, 1000000, 1700))
    figure.savefig(sv_folder+'/fig1_cosyne24.png', dpi=400, bbox_inches='tight')
    figure.savefig(sv_folder+'/fig1_cosyne24.svg', dpi=400, bbox_inches='tight')


def fig_2_cosyne(user_id, sv_folder, humans=True, nm='300',
                 inset_sz=.09, marginx=-0.017, marginy=0.07):
    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 2.5))  # figsize=(3, 3))
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9,
                        hspace=0.55, wspace=0.55)
    ax = ax.flatten()
    fig, ax2 = plt.subplots(1)
    ax_tach = ax2
    ax_pright = ax2
    ax_mat = [ax[2], ax[3]]
    # ax_traj = ax2
    plt.close(fig)
    letters = ['a', 'b', 'c']
    fp.add_text(ax=ax[0], letter=letters[0], x=-0.12, y=1.2)
    fp.add_text(ax=ax[1], letter=letters[1], x=-0.12, y=1.2)
    fp.add_text(ax=ax[2], letter=letters[2], x=-0.09, y=1.25)
    # pos_ax_2 = ax[2].get_position()
    shift = 0.1
    # ax[2].set_position([pos_ax_2.x0+shift/2, pos_ax_2.y0,
    #                     pos_ax_2.width,
    #                     pos_ax_2.height/1.5])
    # pos_ax_3 = ax[3].get_position()
    # ax[3].set_position([pos_ax_3.x0+shift*0.9, pos_ax_3.y0,
    #                     pos_ax_3.width,
    #                     pos_ax_3.height/1.5])
    pos_ax_0 = ax[0].get_position()
    factor = 1.4
    ax[0].set_position([pos_ax_0.x0-shift*0.2, pos_ax_0.y0, pos_ax_0.width*factor,
                        pos_ax_0.height])
    pos_ax_1 = ax[1].get_position()
    ax[1].set_position([pos_ax_1.x0+shift*0.35, pos_ax_1.y0, pos_ax_1.width*factor,
                        pos_ax_1.height])
    for a in ax:
        fp.rm_top_right_lines(a)
    if user_id == 'alex':
        folder = 'C:\\Users\\alexg\\Onedrive\\Escritorio\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'alex_CRM':
        folder = 'C:/Users/agarcia/Desktop/CRM/human/'
    if user_id == 'idibaps':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    if user_id == 'idibaps_alex':
        folder = '/home/jordi/DATA/Documents/changes_of_mind/humans/'+nm+'ms/'
    subj = ['general_traj']
    steps = [None]
    # prepare axis for trajs conditioned on stim and prior
    ax_cohs = ax[1]
    ax_zt = ax[0]
    # trajs. conditioned on coh
    ax_inset_coh = fp.add_inset(ax=ax_cohs, inset_sz=inset_sz, fgsz=(4, 1),
                                marginx=marginx, marginy=marginy, right=True)
    # trajs. conditioned on zt
    ax_inset_zt = fp.add_inset(ax=ax_zt, inset_sz=inset_sz, fgsz=(4, 1),
                               marginx=marginx, marginy=marginy, right=True)
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    # TRAJECTORIES
    index1 = (df_data.subjid != 5) & (df_data.subjid != 6) &\
             (df_data.sound_len <= 300) &\
             (df_data.sound_len >= 0)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    coh = df_data.avtrapz.values[index1]
    decision = df_data.R_response.values[index1]
    trajs = df_data.trajectory_y.values[index1]
    times = df_data.times.values[index1]
    prior_cong = df_data['norm_allpriors'][index1] * (decision*2 - 1)
    prior_cong = prior_cong.values
    ev_vals = np.unique(np.round(coh, 2))
    ground_truth = (df_data.R_response.values*2-1) *\
        (df_data.hithistory.values*2-1)
    ground_truth = ground_truth[index1]
    congruent_coh = np.round(coh, 2) * (decision*2 - 1)
    # Trajs conditioned on stimulus congruency
    fig_6.human_trajs_cond(congruent_coh=congruent_coh, decision=decision,
                           trajs=trajs, prior=prior_cong, bins=ev_vals,
                           times=times, ax=[ax_cohs, ax_inset_coh],
                           n_subjects=len(df_data.subjid.unique()),
                           condition='stimulus', max_mt=400)
    ax_cohs.get_legend().remove()
    bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
    # Trajs conditioned on prior congruency
    fig_6.human_trajs_cond(congruent_coh=congruent_coh, decision=decision,
                           trajs=trajs, prior=prior_cong, bins=bins,
                           times=times, ax=[ax_zt, ax_inset_zt],
                           n_subjects=len(df_data.subjid.unique()),
                           condition='prior', max_mt=400)
    ax_zt.get_legend().remove()
    fig_3.matrix_figure(df_data=df_data, ax_tach=ax_tach, ax_pright=ax_pright,
                        ax_mat=ax_mat, humans=humans)
    pcomlabel_1 = 'Left to right'   # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[0].set_title(pcomlabel_1)
    pcomlabel_0 = 'Right to left'  # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[1].set_title(pcomlabel_0)
    # fig_6.mean_com_traj_human(df=df_data, ax=ax_traj)
    pos_ax_2 = ax[2].get_position()
    shift = 0.1
    pos_ax_3 = ax[3].get_position()
    ax[2].set_position([pos_ax_2.x0+shift*0.8, pos_ax_3.y0,
                        pos_ax_3.width,
                        pos_ax_3.height])
    ax[3].set_position([pos_ax_3.x0+shift*0.6, pos_ax_3.y0,
                        pos_ax_3.width,
                        pos_ax_3.height])
    f.savefig(sv_folder+'/fig2_cosyne24.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/fig2_cosyne24.svg', dpi=400, bbox_inches='tight')


def fig_3_cosyne(df_sim, data_folder, sv_folder):
    f, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 5))  # figsize=(4, 3))
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.08, right=0.96,
                        hspace=0.6, wspace=0.6)
    ax = ax.flatten()
    fig, ax2 = plt.subplots(1)
    ax_cohs = np.array([ax[6], ax2, ax[3], ax2])
    ax_zt = np.array([ax[5], ax2, ax[2], ax2])
    for a in ax:
        fp.rm_top_right_lines(a)
    for i in [0, 4]:
        ax[i].axis('off')
    # pright matrix
    fig_5.plot_pright_model(df_sim=df_sim,
                            sound_len_model=df_sim.sound_len.values,
                            decision_model=df_sim.R_response.values*2-1,
                            subjid=df_sim.subjid.values, coh=df_sim.coh2.values,
                            zt_model=df_sim.norm_allpriors.values,
                            ax=ax[1])
    if sum(df_sim.special_trial == 2) > 0:
        fig_5.traj_cond_coh_simul(df_sim=df_sim[df_sim.special_trial == 2], ax=ax_zt,
                                  new_data=False, data_folder=data_folder,
                                  save_new_data=False,
                                  median=True, prior=True, rt_lim=300)
    else:
        print('No silent trials')
        fig_5.traj_cond_coh_simul(df_sim=df_sim, ax=ax_zt, new_data=False,
                                  data_folder=data_folder,
                                  save_new_data=False, median=True, prior=True)
    fig_5.traj_cond_coh_simul(df_sim=df_sim, ax=ax_cohs, median=True, prior=False,
                              save_new_data=False,
                              new_data=False, data_folder=data_folder,
                              prior_lim=np.quantile(df_sim.norm_allpriors.abs(), 0.2))
    ax[5].get_legend().remove()
    ax[6].get_legend().remove()
    plt.close(fig)
    # PCoM matrices
    pos_ax_0 = ax[2].get_position()
    df_model = pd.DataFrame({'avtrapz': df_sim.coh2.values,
                             'CoM_sugg': df_sim.com_detected,
                             'norm_allpriors': df_sim.norm_allpriors,
                             'R_response': df_sim.R_response.values,
                             'subjid': df_sim.subjid,
                             'sound_len': df_sim.sound_len.values})
    df_model = df_model.loc[df_model.sound_len >= 0]
    ax_mat_r = ax[9]
    ax_mat_l = ax[8]
    # pos_ax_mat = ax_mat_r.get_position()
    # factor = 2
    # ax_mat_r.set_position([pos_ax_mat.x0+pos_ax_mat.width/1.5,
    #                        pos_ax_mat.y0+pos_ax_mat.height/12,
    #                        pos_ax_mat.width/factor,
    #                        pos_ax_mat.height/factor])
    # ax_mat_l = f.add_axes([pos_ax_mat.x0-pos_ax_mat.width/6,
    #                        pos_ax_mat.y0+pos_ax_mat.height/12,
    #                        pos_ax_mat.width/factor,
    #                        pos_ax_mat.height/factor])
    fig_5.plot_pcom_matrices_model(df_model=df_model, n_subjs=len(df_sim.subjid.unique()),
                                   ax_mat=[ax_mat_l, ax_mat_r],
                                   pos_ax_0=pos_ax_0, nbins=7,
                                   f=f)
    ax_mean_com = ax[7]
    fig_5.mean_com_traj_simul(df_sim, ax=ax_mean_com,
                              data_folder=data_folder, new_data=False,
                              save_new_data=False)
    f.savefig(sv_folder+'/fig3_cosyne24.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/fig3_cosyne24.svg', dpi=400, bbox_inches='tight')


def plot_com_traj_rat_model_human(df, df_sim, user_id, sv_folder, data_folder,
                                  nm='300'):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 9))
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.85,
                        hspace=0.5, wspace=0.5)
    ax = ax.flatten()
    for a in ax:
        fp.rm_top_right_lines(a)
    ax_rat = ax[0]
    ax_human = ax[1]
    ax_model = ax[2]
    # GET HUMAN DF
    if user_id == 'alex':
        folder = 'C:\\Users\\alexg\\Onedrive\\Escritorio\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'alex_CRM':
        folder = 'C:/Users/agarcia/Desktop/CRM/human/'
    if user_id == 'idibaps':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    if user_id == 'idibaps_alex':
        folder = '/home/jordi/DATA/Documents/changes_of_mind/humans/'+nm+'ms/'
    subj = ['general_traj']
    steps = [None]
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    # plot human traj
    fig_6.mean_com_traj_human(df_data=df_data, ax=ax_human)
    # plot rat traj
    fig_3.mean_com_traj(df=df, ax=ax_rat, data_folder=data_folder,
                        condition='choice_x_prior',
                        prior_limit=1, after_correct_only=True, rt_lim=400,
                        trajectory='trajectory_y',
                        interpolatespace=np.linspace(-700000, 1000000, 1700))
    # plot model traj
    fig_5.mean_com_traj_simul(df_sim, ax=ax_model,
                              data_folder=data_folder, new_data=False,
                              save_new_data=False)
    for a in ax:
        a.set_xticks([0, 200, 400])
        a.set_ylabel('Position')
        a.set_xlim(-2, 410)
    ax_rat.set_yticks([-25, 0, 25, 50, 75])
    ax_model.set_yticks([-25, 0, 25, 50, 75])
    ax_human.set_xlabel('')
    ax_rat.set_xlabel('')
    ax_human.get_legend().remove()
    ax_rat.get_legend().remove()
    ax_model.set_xlabel('Time from movement onset (ms)')
    
    

f1 = True  # rats
f2 = False  # humans
f3 = False  # model
com_threshold = 8
if f1 or f3:
    # with silent: 42, 43, 44, 45, 46, 47
    # good ones for fitting: 42, 43, 38
    subjects = ['LE42', 'LE43', 'LE38', 'LE39', 'LE85', 'LE84', 'LE45',
                'LE40', 'LE46', 'LE86', 'LE47', 'LE37', 'LE41', 'LE36',
                'LE44']
    # subjects = ['LE42', 'LE43', 'LE38', 'LE39', 'LE45',
    #             'LE40', 'LE46', 'LE47', 'LE37', 'LE41', 'LE36',
    #             'LE44']
    # subjects = ['LE42', 'LE37', 'LE46'
    # subjects = ['LE43']
    df_all = pd.DataFrame()
    for sbj in subjects:
        df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + sbj, return_df=True,
                                        sv_folder=SV_FOLDER, after_correct=True,
                                        silent=True, all_trials=True,
                                        srfail=True)
        df_all = pd.concat((df_all, df), ignore_index=True)
    df = df_all
    del df_all
    # XXX: can we remove the code below or move it to the fig5 part?
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    df['allpriors'] = zt
    hit = np.array(df['hithistory'])
    stim = np.array([stim for stim in df.res_sound])
    coh = np.array(df.coh2)
    com = df.CoM_sugg.values
    decision = np.array(df.R_response) * 2 - 1
    traj_stamps = df.trajectory_stamps.values
    traj_y = df.trajectory_y.values
    fix_onset = df.fix_onset_dt.values
    fix_breaks = np.vstack(np.concatenate([df.sound_len/1000,
                                            np.concatenate(df.fb.values)-0.3]))
    sound_len = np.array(df.sound_len)
    gt = np.array(df.rewside) * 2 - 1
    trial_index = np.array(df.origidx)
    resp_len = np.array(df.resp_len)
    time_trajs = edd2.get_trajs_time(resp_len=resp_len,
                                     traj_stamps=traj_stamps,
                                     fix_onset=fix_onset, com=com,
                                     sound_len=sound_len)
    df['time_trajs'] = time_trajs
    if f1 or f3:
        subjid = df.subjid.values
        print('Computing CoMs')
        time_com, peak_com, com =\
            fig_3.com_detection(df=df, data_folder=DATA_FOLDER,
                                com_threshold=com_threshold)
        print('Ended Computing CoMs')
        com = np.array(com)  # new CoM list
        df['CoM_sugg'] = com
    df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)
    df['time_trajs'] = time_trajs
if f1:
    fig_1_cosyne(df, sv_folder=SV_FOLDER, data_folder=DATA_FOLDER,
                 task_img=TASK_IMG, repalt_img=REPALT_IMG)
if f2:
    fig_2_cosyne(user_id=pc_name, sv_folder=SV_FOLDER, humans=True, nm='300')
if f3:
    simulate = False
    with_fb = False
    save_new_data = False
    print('Plotting Figure 5')
    # we can add extra silent to get cleaner fig5 prior traj
    n_sil = 0
    # trials where there was no sound... i.e. silent for simul
    stim[df.soundrfail, :] = 0
    num_tr = int(len(decision))
    decision = np.resize(decision[:int(num_tr)], num_tr + n_sil)
    zt = np.resize(zt[:int(num_tr)], num_tr + n_sil)
    sound_len = np.resize(sound_len[:int(num_tr)], num_tr + n_sil)
    coh = np.resize(coh[:int(num_tr)], num_tr + n_sil)
    com = np.resize(com[:int(num_tr)], num_tr + n_sil)
    gt = np.resize(gt[:int(num_tr)], num_tr + n_sil)
    trial_index = np.resize(trial_index[:int(num_tr)], num_tr + n_sil)
    hit = np.resize(hit[:int(num_tr)], num_tr + n_sil)
    special_trial = np.resize(df.special_trial[:int(num_tr)], num_tr + n_sil)
    subjid = np.resize(np.array(df.subjid)[:int(num_tr)], num_tr + n_sil)
    special_trial[int(num_tr):] = 2
    # special_trial[df.soundrfail.values] = 2
    if stim.shape[0] != 20:
        stim = stim.T
    stim = np.resize(stim[:, :int(num_tr)], (20, num_tr + n_sil))
    stim[:, int(num_tr):] = 0  # for silent simulation
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt =\
        fp.run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                          trial_index=trial_index, num_tr=num_tr,
                                          subject_list=subjects, subjid=subjid, simulate=simulate)
    # fp.basic_statistics(decision=decision, resp_fin=resp_fin)  # dec
    # fp.basic_statistics(com, com_model_detected)  # com
    # fp.basic_statistics(hit, hit_model)  # hit
    MT = [len(t) for t in trajs]
    df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                            'sound_len': reaction_time,
                            'rewside': (gt + 1)/2,
                            'R_response': (resp_fin+1)/2,
                            'resp_len': np.array(MT)*1e-3})
    df_sim['CoM_sugg'] = com_model.astype(bool)
    df_sim['traj_d1'] = [np.diff(t) for t in trajs]
    df_sim['aftererror'] =\
        np.resize(np.array(df.aftererror)[:int(num_tr)], num_tr + n_sil)
    df_sim['subjid'] = subjid
    df_sim['origidx'] = trial_index
    df_sim['special_trial'] = special_trial
    df_sim['traj'] = df_sim['trajectory_y']
    df_sim['com_detected'] = com_model_detected.astype(bool)
    df_sim['peak_com'] = np.array(x_val_at_updt)
    df_sim['hithistory'] = np.array(resp_fin == gt)
    df_sim['soundrfail'] = np.resize(df.soundrfail.values[:int(num_tr)],
                                     num_tr + n_sil)
    df_sim['allpriors'] = zt
    df_sim['norm_allpriors'] = fp.norm_allpriors_per_subj(df_sim)
    df_sim['normallpriors'] = df_sim['norm_allpriors']
    df_sim['framerate']=200
    # fp.plot_model_trajs(df_sim, df, model_alone=True, align_y_onset=False,
    #                     offset=0)
    # fp.plot_model_density(df_sim, offset=0, df=df, plot_data_trajs=True,
    #                       n_trajs_plot=50, pixel_precision=1, cmap='Reds')
    # fp.plot_data_trajs_density(df=df)
    # simulation plots
    # fp.plot_rt_sim(df_sim)
    # fp.plot_fb_per_subj_from_df(df)
    # fig_3.supp_com_marginal(df=df_sim, sv_folder=SV_FOLDER)
    means, errors = fig_1.mt_weights(df, means_errs=True, ax=None)
    means_model, errors_model = fig_1.mt_weights(df_sim, means_errs=True, ax=None)
    if not with_fb:
        df_sim = df_sim[df_sim.sound_len.values >= 0]
    # memory save:
    stim = []
    traj_y = []
    trial_index = []
    special_trial = []
    # df = []
    gt = []
    subjid = []
    traj_stamps = []
    fix_onset = []
    fix_breaks = []
    resp_len = []
    time_trajs = []
    # fig_3_cosyne(df_sim, data_folder=DATA_FOLDER, sv_folder=SV_FOLDER)
    plot_com_traj_rat_model_human(df, df_sim, user_id='alex',
                                  sv_folder=SV_FOLDER, data_folder=DATA_FOLDER)
