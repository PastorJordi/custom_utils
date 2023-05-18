
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 2023
@author: Alex Garcia-Duran
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
# sys.path.append("/home/molano/custom_utils") # Cluster Manuel

from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figures_paper as fp
# ---GLOBAL VARIABLES
pc_name = 'idibaps_alex'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'
    SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
    DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
    RAT_noCOM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/screenShot230120.png'
    TASK_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/panel_a.png'
    HUMAN_TASK_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/g41085.png'
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


plt.close('all')
f1 = False
f2 = False
f3 = False
f4 = False
f5 = True
f6 = False
f7 = False
com_threshold = 8
if f1 or f2 or f3 or f5:
    all_rats = True
    if all_rats:
        subjects = ['LE42', 'LE43', 'LE38', 'LE39', 'LE85', 'LE84', 'LE45',
                    'LE40', 'LE46', 'LE86', 'LE47', 'LE37', 'LE41', 'LE36',
                    'LE44']
        subjects = ['LE43']
        # with silent: 42, 43, 44, 45, 46, 47
    else:
        subjects = ['LE43']
        # good ones for fitting: 42, 43, 38
    df_all = pd.DataFrame()
    for sbj in subjects:
        df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + sbj, return_df=True,
                                        sv_folder=SV_FOLDER, after_correct=True,
                                        silent=True, all_trials=True,
                                        srfail=True)
        if all_rats:
            df_all = pd.concat((df_all, df), ignore_index=True)
        else:
            subjects = [None]
    if all_rats:
        df = df_all
        df_all = []
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
    if f3 or f5:
        subjid = df.subjid.values
        print('Computing CoMs')
        _, time_com, peak_com, com =\
            edd2.com_detection(trajectories=traj_y, decision=decision,
                                time_trajs=time_trajs,
                                com_threshold=com_threshold)
        print('Ended Computing CoMs')
        com = np.array(com)  # new CoM list
        df['CoM_sugg'] = com
    df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)
    df['time_trajs'] = time_trajs

# fig 1
if f1:
    print('Plotting Figure 1')
    fig_1.fig_1_rats_behav(df_data=df, task_img=TASK_IMG, sv_folder=SV_FOLDER)

# fig 2
if f2:
    print('Plotting Figure 2')
    fp.fig_2_trajs(df=df.loc[df.soundrfail == 0])

# fig 3
if f3:
    print('Plotting Figure 3')
    fp.fig_3_CoMs(df=df, peak_com=peak_com, time_com=time_com)
    fp.supp_com_marginal(df)

# fig 5 (model)
if f5:
    with_fb = False
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
                                          subject_list=subjects, subjid=subjid, simulate=True)
    # basic_statistics(decision=decision, resp_fin=resp_fin)  # dec
    # basic_statistics(com, com_model_detected)  # com
    # basic_statistics(hit, hit_model)  # hit
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
    # simulation plots
    fp.plot_rt_sim(df_sim)
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
    # actual plot
    fp.fig_5_model(coh=coh, sound_len=sound_len, zt=zt,
                   hit_model=hit_model, sound_len_model=reaction_time.astype(int),
                   decision_model=resp_fin, com=com, com_model=com_model,
                   com_model_detected=com_model_detected,
                   means=means, errors=errors, means_model=means_model,
                   errors_model=errors_model, df_sim=df_sim)
    fig, ax = plt.subplots(ncols=2, nrows=1)
    ax = ax.flatten()
    ax[0].set_title('Data')
    fig_1.mt_matrix_ev_vs_zt(df, ax[0], silent_comparison=False, collapse_sides=True)
    ax[1].set_title('Model')
    fig_1.mt_matrix_ev_vs_zt(df_sim, ax[1], silent_comparison=False,
                        collapse_sides=True)
    # fig.suptitle('DATA (top) vs MODEL (bottom)')
    fp.mt_vs_stim_cong(df_sim, rtbins=np.linspace(0, 80, 9), matrix=False)
    # supp_trajs_prior_cong(df_sim, ax=None)
    # model_vs_data_traj(trajs_model=trajs, df_data=df)
    if f4:
        fp.fig_trajs_model_4(trajs_model=trajs, df_data=df,
                            reaction_time=reaction_time)
if f6:
    print('Plotting Figure 6')
    # human traj plots
    fp.fig_humans_6(user_id='idibaps_alex', sv_folder=SV_FOLDER, max_mt=600,
                    wanted_precision=12, nm='300')
if f7:
    fp.fig_7(df, df_sim)
