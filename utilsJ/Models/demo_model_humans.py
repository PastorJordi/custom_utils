# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:17:05 2023

@author: alexg
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')
sys.path.append('C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/Documentos/GitHub/custom_utils')
from utilsJ.paperfigs import figures_paper as fp

# ---GLOBAL VARIABLES
pc_name = 'sara'
if pc_name == 'alex':
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
if pc_name =='sara':
    SV_FOLDER = 'C:\\Users\\Sara Fuentes\\OneDrive - Universitat de Barcelona\\Documentos\\EBM\\4t\\IDIBAPS'
    DATA_FOLDER = 'C:\\Users\\Sara Fuentes\\OneDrive - Universitat de Barcelona\\Documentos\\EBM\\4t\\IDIBAPS'

def plot_rt_all_subjs(reaction_time, subjid):
    subjects = np.unique(subjid)
    for subj in subjects:
        rt = reaction_time[subjid == subj]
        sns.kdeplot(rt, color='red', alpha=0.4)
        plt.axvline(300, color='k', linestyle='--')

load_params = False  # wether to load or not parameters
df_data = fp.get_human_data(user_id=pc_name, sv_folder=SV_FOLDER)
choice = df_data.R_response.values*2-1
hit = df_data.hithistory.values*2-1
subjects = df_data.subjid.unique()
subjid = df_data.subjid.values
len_task = [len(df_data.loc[subjid == subject]) for subject in subjects]
# subjid = np.repeat('all', len(choice))  # meta subject
df_data['subjid'] = subjid
gt = (choice*hit+1)/2
coh = df_data.avtrapz.values*5
trial_index = np.empty((0))
for j in range(len(len_task)):
    trial_index = np.concatenate((trial_index, np.arange(len_task[j])+1))
df_data['origidx'] = trial_index
stim = np.repeat(coh.reshape(-1, 1), 20, 1).T
# MT ANALYSIS
index1 = (df_data.subjid != 5) & (df_data.subjid != 6)
df_data.avtrapz /= max(abs(df_data.avtrapz))
coh = df_data.avtrapz.values[index1]
decision = df_data.R_response.values[index1]
trajs = df_data.trajectory_y.values[index1]
times = df_data.times.values[index1]
sound_len = df_data.sound_len.values[index1]
prior_cong = df_data['norm_allpriors'][index1] * (decision*2 - 1)
prior_cong = prior_cong.values
ev_vals = np.unique(np.round(coh, 2))
congruent_coh = np.round(coh, 2) * (decision*2 - 1)




# SIMULATION
hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
    _, trajs, x_val_at_updt =\
    fp.simulate_model_humans(df_data=df_data, stim=stim,
                             load_params=load_params)
MT = np.array([len(t) for t in trajs])
mt_human = np.array(fp.get_human_mt(df_data))
df_data['resp_len'] = mt_human
df_data['coh2'] = coh
df_data['origidx'] = trial_index
df_data['allpriors'] = df_data.norm_allpriors.values
# simulation output: hit_model, reaction_time, mt_human, resp_fin, com_model
# trajs

