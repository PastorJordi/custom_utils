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


matplotlib.rcParams['font.size'] = 10.5
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['xtick.labelsize']= 10.5
plt.rcParams['ytick.labelsize']= 10.5
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


def fig_1_cosyne(df, data_folder, task_img, repalt_img, figsize=(10, 8),
                 inset_sz=.1, marginx=-.04, marginy=0.1):
    f, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize)  # figsize=(4, 3))
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.08, right=0.98,
                        hspace=0.45, wspace=0.6)
    ax = ax.flatten()
    # TUNE PANELS
    # all panels
    letters = ['', '',  '', 'c', 'd', 'g', 'e', 'f', '']
    for n, ax_1 in enumerate(ax):
        fp.add_text(ax=ax_1, letter=letters[n], x=-0.12, y=1.25)
        if n not in [10, 11]:
            fp.rm_top_right_lines(ax_1)

    for i in [0, 1]:
        ax[i].axis('off')
    # task panel
    ax_task = ax[0]
    pos_task = ax_task.get_position()
    factor = 2.5
    ax_task.set_position([pos_task.x0, pos_task.y0-0.08,
                          pos_task.width*factor, pos_task.height*factor])
    fp.add_text(ax=ax_task, letter='a', x=0.1, y=1.12)
    # rep-alt img
    ax_repalt = ax[1]
    pos_repalt = ax_repalt.get_position()
    factor = 1.3
    ax_repalt.set_position([pos_repalt.x0-pos_repalt.width/2,
                            pos_repalt.y0+0.02,
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
    factor = 1.1
    ax_pright.set_position([pos_pright.x0,
                            pos_pright.y0 + 0.05,
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
    cbar = f.colorbar(im_2, ax=ax_pright, location='top', label='p (right)',
                      shrink=0.55, aspect=5)
    im = ax_pright.images
    cb = im[-1].colorbar
    pos_cb = cb.ax.get_position()
    pos_pright = ax_pright.get_position()
    factor = 1.3
    ax_pright.set_position([pos_pright.x0-pos_pright.width/3,
                            pos_pright.y0 + 0.07,
                            pos_pright.width*factor,
                            pos_pright.height*factor])
    cb.ax.set_position([pos_cb.x0-pos_pright.width/3+0.02,
                        pos_cb.y0+0.115, pos_cb.width, pos_cb.height])
    # pright_cbar_ax.set_title('p (right)')
    cbar.ax.tick_params(rotation=45)
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
    # trajs
    # TRAJECTORIES CONDITIONED ON PRIOR
    # add insets
    ax = f.axes
    fig, ax2 = plt.subplots(1)
    ax_zt = np.array([ax[6], ax2])
    ax_cohs = np.array([ax[7], ax2])
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
    plt.close(fig)
    # ax_mean_com = ax[5]
    fig_3.mean_com_traj(df=df, ax=ax[5], data_folder=data_folder, condition='choice_x_prior',
                        prior_limit=1, after_correct_only=True, rt_lim=400,
                        trajectory='trajectory_y',
                        interpolatespace=np.linspace(-700000, 1000000, 1700))


def fig_2_cosyne(user_id, sv_folder, humans=True, nm='300'):
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(6, 5))  # figsize=(3, 3))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.5, wspace=0.5)
    ax = ax.flatten()
    ax[0].axis('off')
    ax_tach=ax[1]
    ax_pright=ax[3]
    ax_mat=[ax[4], ax[5]]
    ax_traj=ax[2]
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
    fig_3.matrix_figure(df_data=df_data, ax_tach=ax_tach, ax_pright=ax_pright,
                        ax_mat=ax_mat, humans=humans)
    fig_6.plot_coms(df=df_data, ax=ax_traj, human=humans)


def fig_3_cosyne(df_sim, data_folder):
    f, ax = plt.subplots(ncols=4, nrows=2)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.98,
                        hspace=0.45, wspace=0.3)
    ax = ax.flatten()
    fig, ax2 = plt.subplots(1)
    ax_cohs = np.array([ax[5], ax2, ax[1], ax2])
    ax_zt = np.array([ax[4], ax2, ax[0], ax2])
    for a in ax:
        fp.rm_top_right_lines(a)
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
    plt.close(fig)
    # PCoM matrices
    pos_ax_0 = ax[2].get_position()
    ax[2].set_position([pos_ax_0.x0-pos_ax_0.width/6, pos_ax_0.y0, pos_ax_0.width*1.15,
                        pos_ax_0.height])
    df_model = pd.DataFrame({'avtrapz': df_sim.coh2.values,
                             'CoM_sugg': df_sim.com_detected,
                             'norm_allpriors': df_sim.norm_allpriors,
                             'R_response': df_sim.R_response.values,
                             'subjid': df_sim.subjid,
                             'sound_len': df_sim.sound_len.values})
    df_model = df_model.loc[df_model.sound_len >= 0]
    fig_5.plot_pcom_matrices_model(df_model=df_model, n_subjs=len(df_sim.subjid.unique()),
                                   ax_mat=[ax[6], ax[7]],
                                   pos_ax_0=pos_ax_0, nbins=7,
                                   f=f)
    ax_mean_com = ax[3]
    fig_5.mean_com_traj_simul(df_sim, ax=ax_mean_com,
                              data_folder=data_folder, new_data=False,
                              save_new_data=False)


f1 = False  # rats
f2 = True  # humans
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
    # subjects = ['LE42', 'LE37', 'LE46']
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
    fig_1_cosyne(df, data_folder=DATA_FOLDER, task_img=TASK_IMG, repalt_img=REPALT_IMG)
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
    fig_3_cosyne(df_sim, data_folder=DATA_FOLDER)
