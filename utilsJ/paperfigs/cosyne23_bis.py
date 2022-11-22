# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:05:37 2022

@author: Alexandre
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# from scipy.stats import sem
# import sys
import matplotlib
import cosyne23 as c23
# from scipy import interpolate
# sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
import figures_paper as fp
from utilsJ.Models import extended_ddm_v2 as edd2

# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
# sys.path.append("C:/Users/Alexandre/Documents/psycho_priors")
# from utilsJ.Models import simul
# import analyses
# from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve,\
#     trajectory_thr, com_heatmap
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import fig1, fig3, fig2
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
# SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
# DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
# RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'
DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
    'ChangesOfMind/figures/from_python/'  # Manuel
RAT_COM_IMG = '/home/molano/Dropbox/project_Barna/' +\
    'ChangesOfMind/figures/Figure_3/001965.png'


matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3


# ---MAIN
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
        c23.matrix_figure(df_data, ax_tach=ax[1], ax_pright=ax[2],
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
        c23.matrix_figure(df_data=df_data, humans=humans, ax_tach=ax[1],
                          ax_pright=ax[2], ax_mat=ax[3])
    if f3:
        # FIG 3:
        f, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
        ax[0].axis('off')
        c23.fig_3(user_id='Manuel', sv_folder=SV_FOLDER,
                  ax_tach=ax[1], ax_pright=ax[2], ax_mat=ax[3], humans=True)
