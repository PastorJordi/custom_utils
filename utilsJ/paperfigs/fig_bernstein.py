# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:08:25 2023

@author: alexg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys

# sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')  # Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel

from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.paperfigs import figure_5 as fig_5
from utilsJ.paperfigs import figure_6 as fig_6
from utilsJ.paperfigs import figures_paper as fp
matplotlib.rcParams['font.size'] = 10   
plt.rcParams['legend.title_fontsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize']= 8
plt.rcParams['ytick.labelsize']= 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
    RAT_noCOM_IMG = 'C:/Users/alexg/OneDrive/Escritorio/CRM/figures/screenShot230120.png'
    TASK_IMG = 'C:/Users/alexg/OneDrive/Escritorio/CRM/figures/panel_a.png'
    HUMAN_TASK_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/Human/g41085.png'
    MODEL_IMG = 'C:/Users/alexg/OneDrive/Escritorio/CRM/figures/model_fig.png'
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
elif pc_name == 'idibaps_alex':
    SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
    DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
    RAT_COM_IMG = '/home/jordi/Documents/changes_of_mind/demo/materials/' +\
        'craft_vid/CoM/a/001965.png'
    RAT_noCOM_IMG = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/' +\
        'screenShot230120.png'
    HUMAN_TASK_IMG = '/home/jordi/DATA/Documents/changes_of_mind/humans/g41085.png'
elif pc_name == 'alex_CRM':
    SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
    RAT_COM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/001965.png'
    RAT_noCOM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/screenShot230120.png'
    HUMAN_TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/rat_image/g41085.png'
    TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/panel_a.png'

plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'font.size': 14})

def fig_bernstein(df, data_folder=DATA_FOLDER, task_img=TASK_IMG, model_img=MODEL_IMG,
                  rat_nocom_img=RAT_noCOM_IMG, fgsz=(15, 12), inset_sz=.1,
                  marginx=-.04, marginy=0.1):
    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=fgsz)
    ax = ax.flatten()
    ax_task = ax[0]
    ax_task.axis('off')
    ax[3].axis('off')
    ax[2].axis('off')
    ax[6].axis('off')
    ax[7].axis('off')
    ax[10].axis('off')
    ax[11].axis('off')
    for a in ax:
        fp.rm_top_right_lines(a)
    # pos_task = ax_task.get_position()
    # factor = 1.75
    # ax_task.set_position([pos_task.x0+0.05, pos_task.y0-0.05,
    #                       pos_task.width*factor, pos_task.height*factor])
    # fp.add_text(ax=ax_task, letter='a', x=0.1, y=1.15)
    # TASK PANEL
    task = plt.imread(task_img)
    ax_task.imshow(task)
    # mt versus evidence panels
    ax_mt_coh = ax[5]
    ax_mt_zt = ax[4]
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
    # rat trajectory deeplabcut
    # tune screenshot panel
    ax_scrnsht = ax[1]
    ax_scrnsht.set_xticks([])
    right_port_y = 50
    center_port_y = 250
    left_port_y = 460
    margin = 0.05
    ax_scrnsht.set_yticks([right_port_y, center_port_y, left_port_y])
    ax_scrnsht.set_yticklabels([-85, 0, 85])
    ax_scrnsht.set_xlabel('x dimension (pixels)')
    ax_scrnsht.set_ylabel('y dimension (pixels)')
    # add colorbar for screenshot
    n_stps = 100
    pos = ax_scrnsht.get_position()
    ax_clbr = plt.axes([pos.x0+margin/4, pos.y0+pos.height+margin/8,
                        pos.width*0.7, pos.height/15])
    ax_clbr.imshow(np.linspace(0, 1, n_stps)[None, :], aspect='auto')
    x_tcks = np.linspace(0, n_stps, 6)
    ax_clbr.set_xticks(x_tcks)
    x_tcks_str = ['0', '', '', '', '', str(4*n_stps)]
    x_tcks_str[-1] += ' ms'
    ax_clbr.set_xticklabels(x_tcks_str)
    ax_clbr.tick_params(labelsize=6)
    ax_clbr.set_yticks([])
    ax_clbr.xaxis.set_ticks_position("top")
    # TRACKING SCREENSHOT
    rat = plt.imread(rat_nocom_img)
    img = rat[150:646, 120:-10, :]
    ax_scrnsht.imshow(np.flipud(img)) # rat.shape = (796, 596, 4)
    ax_scrnsht.axhline(y=left_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(y=right_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(center_port_y, color='k', lw=.5)
    ax_scrnsht.set_ylim([0, img.shape[0]])
    # trajectories
    # add insets
    ax_zt = np.array([ax[8], ax[10]])
    ax_cohs = np.array([ax[9], ax[11]])
    ax_inset_1 = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=fgsz,
                              marginx=marginx, marginy=marginy, right=True)
    ax_inset_1.yaxis.set_ticks_position('none')
    # ax_cohs contains in this order the axes for:
    # index 0: mean position of rats conditioned on stim. evidence,
    # index 1: the inset for the velocity panel 
    # index 2: mean velocity  of rats conditioned on stim. evidence
    ax_cohs = np.insert(ax_cohs, 1, ax_inset_1)
    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=fgsz,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_zt = np.insert(ax_zt, 1, ax_inset)
     # ax_zt contains in this order the axes for:
    # index 0: mean position of rats conditioned on prior evidence,
    # index 1: the inset for the velocity panel 
    # index 2: mean velocity  of rats conditioned on priors evidence
    df_trajs = df.copy()
    # TRAJECTORIES CONDITIONED ON PRIOR
    fig_2.plots_trajs_conditioned(df=df_trajs.loc[df_trajs.special_trial == 2],
                                  ax=ax_zt, data_folder=data_folder,
                                  condition='choice_x_prior',
                                  prior_limit=1, cmap='copper')
    # TRAJECTORIES CONDITIONED ON COH
    fig_2.plots_trajs_conditioned(df=df_trajs, ax=ax_cohs,
                                  data_folder=data_folder,
                                  condition='choice_x_coh',
                                  prior_limit=0.1,  # 10% quantile
                                  cmap='coolwarm')
    ax[10].cla()
    ax[10].axis('off')
    ax[11].cla()
    ax[11].axis('off')
    ax_inset_1.cla()
    ax_inset_1.axis('off')
    ax_inset.cla()
    ax_inset.axis('off')
    # model
    ax_model = ax[6]
    pos_ax9 = ax[10].get_position()
    ax_model.set_position([pos_ax9.x0, pos_ax9.y0, pos_ax9.width*1.5,
                           pos_ax9.height*3])
    model_image = plt.imread(model_img)
    ax_model.imshow(model_image)
    for i in [3, 7, 11]:
        pos = ax[i].get_position()
        ax[i].set_position([pos.x0, pos.y0, pos.width*0.4, pos.height])


subjects = ['LE42', 'LE43', 'LE38', 'LE39', 'LE85', 'LE84', 'LE45',
            'LE40', 'LE46', 'LE86', 'LE47', 'LE37', 'LE41', 'LE36',
            'LE44']
# subjects = ['LE42', 'LE85', 'LE86']
# subjects = ['LE42', 'LE41']
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
subjid = df.subjid.values
# print('Computing CoMs')
# time_com, peak_com, com =\
#     fig_3.com_detection(df=df, data_folder=DATA_FOLDER,
#                         com_threshold=8)
# print('Ended Computing CoMs')
# com = np.array(com)  # new CoM list
# df['CoM_sugg'] = com
df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)
df['time_trajs'] = time_trajs
fig_bernstein(df, task_img=TASK_IMG)