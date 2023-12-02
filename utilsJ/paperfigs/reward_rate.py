
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 2023
@author: Alex Garcia-Duran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import itertools
# from imp import reload
import sys


sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')
# Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils") # Cluster Alex
sys.path.append("/home/molano/custom_utils")  # Cluster Manuel
sys.path.append('C:/Users/Sara Fuentes/OneDrive - Universitat de '
                'Barcelona/Documentos/GitHub/custom_utils')
sys.path.append('C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/'
                'Documentos/GitHub/custom_utils/'
                'utilsJ/Models')


from utilsJ.paperfigs import figures_paper as fp
# reload(fig_5)

matplotlib.rcParams['font.size'] = 10.5
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 10.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
matplotlib.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
pc_name = 'sara'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/001965.png'
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
    RAT_noCOM_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/' +\
        'screenShot230120.png'
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
    ST_CARTOON_IMG = '/home/molano/Dropbox/project_Barna/ChangesOfMind/' +\
        'figures/Figure_2/st_cartoon_violins.png'
elif pc_name == 'idibaps_alex':
    SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
    DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'
    # Jordi
    RAT_COM_IMG = '/home/jordi/Documents/changes_of_mind/demo/materials/' +\
        'craft_vid/CoM/a/001965.png'
    RAT_noCOM_IMG = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/' +\
        'screenShot230120.png'
    HUMAN_TASK_IMG = '/home/jordi/DATA/Documents/changes_of_mind/humans/' +\
        'g41085.png'
    TASK_IMG = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/' +\
        'panel_a.png'
elif pc_name == 'alex_CRM':
    SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
    RAT_COM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/001965.png'
    RAT_noCOM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/screenShot230120.png'
    HUMAN_TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/rat_image/g41085.png'
    TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/figures/panel_a.png'
    REPALT_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/figures/repalt.png'
elif pc_name == 'sara':
    SV_FOLDER = 'C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/' +\
                'Documentos/EBM/4t/IDIBAPS/'
    DATA_FOLDER = 'C:/Users/Sara Fuentes/OneDrive - Universitat de ' +\
                  'Barcelona/Documentos/EBM/4t/IDIBAPS/'

PUNISHMENT = 2000



def RR_core(df_data, stim, param_to_explore, value_to_explore):
    """
    Parameters
    ----------
    param_to_explore : list
        list with names of parameters to explore
    value_to_explore : list
        list with values of parameters to explore
    Returns
    -------
    rw_rate : float

    """
    param_idx_dict = {"p_w_zt": 0, "p_w_stim": 1, "p_e_bound": 2,
                      "p_com_bound": 3, "p_t_aff": 4, "p_t_eff": 5,
                      "p_t_a": 6, "p_w_a_intercept": 7, "p_w_a_slope": 8,
                      "p_a_bound": 9, "p_1st_readout": 10,
                      "p_2nd_readout": 11, "p_leak": 12, "p_mt_noise": 13,
                      "p_MT_intercept": 14, "p_MT_slope": 15}

    param_idxes = [param_idx_dict[pte] for pte in param_to_explore]
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt, frst_traj_motor_time =\
        fp.simulate_model_humans(df_data, stim=stim, load_params=True,
                                 params_to_explore=[param_idxes,
                                                    value_to_explore])
    sum_hit_model = np.sum(hit_model)
    sum_rt = np.sum(reaction_time)
    # TODO: do this for the model
    sum_mt = np.sum([len(tr) for tr in trajs])
    rr = 1000 * sum_hit_model / (
        sum_rt + sum_mt + PUNISHMENT * (np.sum(hit_model == 0))
    )

    return rr, sum_hit_model, sum_rt, sum_mt


def RR_mat(df_data, stim, params_to_explore, values_to_explore):
    """
    Parameters
    ----------
    params_to_explore : list
        list of parameters to explore
    values_to_explore : list of lists
        values of the paremeters to test, in the same order
    punishment: integer
        penalty time for each error

    Returns
    -------
    reward_rate_mat: matrix  of size: len(values_to_explore[0]) x
                    len(values_to_explore[1])
        matrix containing the reward rates for the values of the tested
        parameters

    """

    rr_dict = {'rr': [], 'sum_hit_model': [], 'sum_rt': [], 'sum_mt': []}
    list_of_values = itertools.product(*values_to_explore)
    for i_v in list_of_values:
        rr, sum_hit_model, sum_rt, sum_mt =\
            RR_core(df_data=df_data, stim=stim,
                    param_to_explore=params_to_explore,
                    value_to_explore=i_v)

        rr_dict['rr'].append(rr)
        rr_dict['sum_hit_model'].append(sum_hit_model)
        rr_dict['sum_rt'].append(sum_rt)
        rr_dict['sum_mt'].append(sum_mt)

    return rr_dict


# TODO: plot_reward_rate() funtion


# --- MAIN
if __name__ == '__main__':

    # param_idx_dict = {"p_w_zt": 0, "p_w_stim": 1, "p_e_bound": 2,
    #                   "p_com_bound": 3, "p_t_aff": 4, "p_t_eff": 5,
    #                   "p_t_a": 6, "p_w_a_intercept": 7, "p_w_a_slope": 8,
    #                   "p_a_bound": 9, "p_1st_readout": 10,
    #                   "p_2nd_readout": 11, "p_leak": 12, "p_mt_noise": 13,
    #                   "p_MT_intercept": 14, "p_MT_slope": 15}
    load_params = True
    params_to_explore = ['p_a_bound', 'p_1st_readout']
    values_to_explore = [np.arange(1, 11), [181.43]]

    df_data = fp.get_human_data(user_id=pc_name, sv_folder=SV_FOLDER)
    choice = df_data.R_response.values*2-1
    df_data['subjid'] = np.repeat('all', len(choice))
    hit = df_data.hithistory.values*2-1
    subjects = df_data.subjid.unique()
    subjid = df_data.subjid.values
    gt = (choice*hit+1)/2
    coh = df_data.avtrapz.values*5
    stim = np.repeat(coh, 20).reshape(len(coh), 20).T
    stim += np.random.randn(stim.shape[0], stim.shape[1])*0.001
    len_task = [len(df_data.loc[subjid == subject]) for subject in subjects]
    trial_index = np.empty((0))
    for j in range(len(len_task)):
        trial_index = np.concatenate((trial_index,
                                      np.arange(len_task[j])+1))
    df_data['origidx'] = trial_index
    rr = RR_mat(df_data, stim, params_to_explore, values_to_explore)
    print('params_to_explore: ', params_to_explore)
    print('values_to_explore: ', values_to_explore)
    print('reward rate: ', rr)
    # TODO: make labels
    f, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    ax[0].plot(np.arange(1, 11), rr['rr'])
    ax[1].plot(np.arange(1, 11), rr['sum_hit_model'])
    ax[2].plot(np.arange(1, 11), rr['sum_rt'])
    ax[3].plot(np.arange(1, 11), rr['sum_mt'])
    plt.show()
