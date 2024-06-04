
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 2023
@author: Alex Garcia-Duran
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
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
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.paperfigs import figure_5 as fig_5
from utilsJ.paperfigs import fig_5_humans as fig_5h
from utilsJ.paperfigs import figure_6 as fig_6
from utilsJ.paperfigs import figure_7 as fig_7
from utilsJ.paperfigs import figures_paper as fp
# reload(fig_5)

matplotlib.rcParams['font.size'] = 11
plt.rcParams['legend.title_fontsize'] = 10.5
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize']= 10
plt.rcParams['ytick.labelsize']= 10
matplotlib.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
pc_name = 'alex'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/image_com.png'
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
    RAT_noCOM_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/screenShot230120.png'
    TASK_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/panel_a.png'
    PCOM_RT_IMG = 'C:/Users/alexg/Onedrive/Escritorio/CRM/figures/pcom_rt.png'
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
elif pc_name == 'idibaps_alex':
    SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
    DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
    RAT_COM_IMG = '/home/jordi/Documents/changes_of_mind/demo/materials/' +\
        'craft_vid/CoM/a/001965.png'
    RAT_noCOM_IMG = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/' +\
        'screenShot230120.png'
    HUMAN_TASK_IMG = '/home/jordi/DATA/Documents/changes_of_mind/humans/g41085.png'
    TASK_IMG = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/panel_a.png'
elif pc_name == 'alex_CRM':
    SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
    RAT_COM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/001965.png'
    RAT_noCOM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/screenShot230120.png'
    HUMAN_TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/rat_image/g41085.png'
    TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/figures/panel_a.png'
    REPALT_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/figures/repalt.png'


def plot_psyc(decision_rat, decision_model, coh):
    ev_vals = np.unique(coh)
    dec_rat_01 = (decision_rat + 1)/2
    dec_mod_01 = (decision_model + 1)/2
    acc_rat = []
    acc_mod = []
    plt.figure()
    for ev in ev_vals:
        index = coh == ev
        acc_rat.append(np.nanmean(dec_rat_01[index]))
        acc_mod.append(np.nanmean(dec_mod_01[index]))
    plt.plot(ev_vals, acc_rat, marker='o', color='blue', label='Data')
    plt.plot(ev_vals, acc_mod, marker='o', color='red', label='Model')
    plt.legend()


def plot_MT_density_comparison(MT_data, MT_model):
    plt.figure()
    sns.kdeplot(MT_data, color='blue', label='Data')
    sns.kdeplot(MT_model, color='red', label='Model')
    plt.legend()
    plt.xlabel('MT (ms)')


def plot_RT_density_comparison(RT_data, RT_model):
    plt.figure()
    sns.kdeplot(RT_data, color='blue', label='Data')
    sns.kdeplot(RT_model, color='red', label='Model')
    plt.xlabel('RT (ms)')
    plt.axvline(x=300, linestyle='--', color='k')
    plt.legend()


def check_distros(df, df_sim):
    # plot_psyc(df.R_response.values*2-1,
    #           df_sim.R_response.values*2-1, df.coh2.values)
    plot_MT_density_comparison(df.resp_len, df_sim.resp_len)
    plot_RT_density_comparison(df.sound_len, df_sim.sound_len)


plt.close('all')
f1 = False
f2 = False
f3 = False
f4 = False
f5 = True
f6 = False
f7 = False
f8 = False
com_threshold = 8
if f1 or f2 or f3 or f5 or f7:
    # with silent: 42, 43, 44, 45, 46, 47
    subjects = ['LE42', 'LE43', 'LE38', 'LE39', 'LE85', 'LE84', 'LE45',
                'LE40', 'LE46', 'LE86', 'LE47', 'LE37', 'LE41', 'LE36',
                'LE44']
    # subjects = ['LE42', 'LE38', 'LE39', 'LE40']
    # subjects = ['LE46']  # for params analysis
    subjects = ['LE42']  # for fig7
    # subjects = ['LE42', 'LE43', 'LE44', 'LE45', 'LE46', 'LE47']  # for silent
    df_all = pd.DataFrame()
    for sbj in subjects:
        df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + sbj, return_df=True,
                                        sv_folder=SV_FOLDER, after_correct=True,
                                        silent=True, all_trials=True,
                                        srfail=True)
        df_all = pd.concat((df_all, df), ignore_index=True)
    df = df_all
    del df_all
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
    special_trial = df.special_trial
    df['time_trajs'] = time_trajs
    if (f5 or f3) and not f7:
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
    if f7:
        df['subjid'] = 'LE42'
        subjects = ['LE42']  # to simulate trials from selected subjects with LE42 parameters
# fig 1
if f1:
    print('Plotting Figure 1')
    fig_1.fig_1_rats_behav(df_data=df, task_img=TASK_IMG, sv_folder=SV_FOLDER,
                            repalt_img=REPALT_IMG)
    # fig_1.supp_trial_index_analysis(df=df, data_folder=DATA_FOLDER)

# fig 2
if f2:
    print('Plotting Figure 2')
    fig_2.fig_2_trajs(df=df.loc[df.soundrfail == 0], data_folder=DATA_FOLDER,
                      sv_folder=SV_FOLDER, rat_nocom_img=RAT_noCOM_IMG,
                      st_cartoon_img=ST_CARTOON_IMG)

# fig 3
if f3:
    print('Plotting Figure 3')
    fig_3.fig_3_CoMs(df=df, sv_folder=SV_FOLDER, data_folder=DATA_FOLDER,
                     rat_com_img=RAT_COM_IMG)
    # fig_3.supp_com_marginal(df=df, sv_folder=SV_FOLDER)

# fig 5 (model)
if f5 or f7:
    simulate = False
    with_fb = False
    save_new_data = False
    silent_sim = False
    if silent_sim:
        stim[:] = 0  # silent simulation
        special_trial[:] = 2
        extra_label = 'silent'
    else:
        extra_label =  'orig_v1'
    print('Plotting Figure 5')
    # we can add extra silent to get cleaner fig5 prior traj
    if f7:
        n_sil = 0  # int(400000 - len(df))
    else:
        n_sil = 0  # int(100000 - len(df))
    # trials where there was no sound... i.e. silent for simul
    stim[df.soundrfail, :] = 0
    num_tr = int(len(decision))
    decision = np.resize(decision[:int(num_tr)], num_tr + n_sil)
    zt = np.resize(zt[:int(num_tr)], num_tr + n_sil)
    sound_len = np.resize(sound_len[:int(num_tr)], num_tr + n_sil)
    coh = np.resize(coh[:int(num_tr)], num_tr + n_sil)
    com = np.resize(com[:int(num_tr)], num_tr + n_sil)
    # com = np.resize(com[:int(num_tr)], num_tr + n_sil)
    gt = np.resize(gt[:int(num_tr)], num_tr + n_sil)
    trial_index = np.resize(trial_index[:int(num_tr)], num_tr + n_sil)
    hit = np.resize(hit[:int(num_tr)], num_tr + n_sil)
    special_trial = np.resize(special_trial[:int(num_tr)], num_tr + n_sil)
    subjid = np.resize(np.array(df.subjid)[:int(num_tr)], num_tr + n_sil)
    # special_trial[df.soundrfail.values] = 2
    if stim.shape[0] != 20:
        stim = stim.T
    stim = np.resize(stim[:, :int(num_tr)], (20, num_tr + n_sil))
    # stim[:, int(num_tr):] = 0  # for silent simulation
    if f5:
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            _, trajs, x_val_at_updt =\
            fp.run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                              trial_index=trial_index, num_tr=num_tr,
                                              subject_list=subjects, subjid=subjid, simulate=simulate,
                                              extra_label=extra_label)
        
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
        df_sim['framerate'] = 200
        df_sim['res_sound'] = [st for st in stim.T]
        # fp.plot_model_density(df_sim, offset=0, df=df, plot_data_trajs=True,
        #                       n_trajs_plot=150, pixel_precision=1, cmap='Reds')
        # fp.supp_plot_rt_distros_data_model(df, df_sim)
        # simulation plots
        # fp.plot_rt_sim(df_sim)
        # fp.plot_fb_per_subj_from_df(df)
        # fig_3.supp_com_marginal(df=df_sim, sv_folder=SV_FOLDER)
        # fig_2.supp_regression_weights_trajectory(DATA_FOLDER, SV_FOLDER, 
        #                                           subjects, subjid, stim, zt, coh,
        #                                           gt, trial_index,
        #                                           special_trial, extra_labels=['_50_ms_orig_',
        #                                                                       '_50_ms_continuous_',
        #                                                                       '_orig_stimulus_cont_',
        #                                                                       '_continuous_stimulus_'])
        # means, errors = fig_1.mt_weights(df, means_errs=True, ax=None)
        # means_model, errors_model = fig_1.mt_weights(df_sim, means_errs=True, ax=None)
        # if not with_fb:
        #     df_sim = df_sim.loc[df_sim.sound_len.values >= 0]
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
        fp.supp_silent(df=df, df_sim=df_sim)
        # fig, ax = plt.subplots(1)
        # fig_2.trajs_splitting_stim(df_sim, ax, DATA_FOLDER, collapse_sides=True, threshold=300,
        #                           sim=True,
        #                           rtbins=np.linspace(0, 150, 25), connect_points=False,
        #                           trajectory="trajectory_y", p_val=0.05)

        # fig_5.supp_com_analysis(df_sim, sv_folder=SV_FOLDER, pcom_rt=PCOM_RT_IMG)
        # fig_5.supp_p_reversal_silent(df, df_sim, data_folder=DATA_FOLDER,
        #                               sv_folder=SV_FOLDER)
        # fig_5.supp_model_trial_index(df_sim)
        # actual plot
        # fig_2.plot_lin_reg_weights_frames(df.loc[df.coh2.abs() <= 0.25], frame_len=50,
        #                                   rtbins=np.linspace(0, 150, 4),
        #                                   trajectory="trajectory_y",
        #                                   max_MT=400, sim=False)
        # rtbins=np.linspace(50, 110, 4)
        # fig_2.plot_lin_reg_weights_continuous_stimulus(df_sim.loc[(df_sim.sound_len >= min(rtbins)) & 
        #                                                           (df_sim.sound_len <= max(rtbins))],
        #                                                frame_len=5,
        #                                                rtbins=rtbins,
        #                                                trajectory="trajectory_y",
        #                                                max_MT=400, sim=True,
        #                                                continuous=True)
        # fig_2.splitting_time_frames_model(df_sim.loc[df_sim.subjid == 'LE42'],
        #                                   data_folder=DATA_FOLDER,
        #                                   frame_len=5,
        #                                   rtbins=np.arange(0, 305, 15),
        #                                   trajectory="trajectory_y",
        #                                   new_data=True,
        #                                   pval=.05, max_MT=400)
        # fig_2.log_reg_vs_rt(df, rtbins=np.linspace(0, 150, 16))
        # fig_2.log_reg_vs_rt(df_sim, rtbins=np.linspace(0, 150, 16))
        # fig_5.fig_5_model(sv_folder=SV_FOLDER, data_folder=DATA_FOLDER,
        #                   new_data=simulate, save_new_data=save_new_data,
        #                   coh=coh, sound_len=sound_len, zt=zt,
        #                   hit_model=hit_model, sound_len_model=reaction_time.astype(int),
        #                   decision_model=resp_fin, com=com, com_model=com_model,
        #                   com_model_detected=com_model_detected,
        #                   means=means, errors=errors, means_model=means_model,
        #                   errors_model=errors_model, df_sim=df_sim, extra_label=extra_label)
    if f7:
        # index to filter by stim/silent for p(com) vs p(proac) supp figure
        # idx = df.special_trial.values >= 0
        # fp.supp_parameter_analysis(stim, zt, coh, gt, trial_index, subjects,
        #                             subjid, sv_folder=SV_FOLDER, idx=idx)
        # fp.supp_pcom_teff_taff(stim, zt, coh, gt, trial_index, subjects,
        #                        subjid, sv_folder=SV_FOLDER, idx=idx)
        # fig_7.plot_traj_cond_different_models(subjects, subjid, stim, zt, coh, gt, trial_index,
        #                                       special_trial,
        #                                       data_folder=DATA_FOLDER,
        #                                       extra_labels=['_2_ro_rand_'+str(len(coh)),
        #                                                     '_1_ro_'+str(len(coh)),
        #                                                     len(coh)])
        fig3, ax_mat = plt.subplots(2)
        fig3.tight_layout()
        fig_5.plot_pcom_matrices_model(df.loc[df.sound_len >= 150],
                                       1, ax_mat, fig3, nbins=7, pos_ax_0=[],
                                       margin=.03, title='p(reversal)',
                                       mat_titles=['Left to right reversal',
                                                   'Right to left reversal'],
                                       return_matrix=False)
        fig_7.fig_7(subjects, subjid, stim, zt, coh, gt, trial_index,
                    special_trial,
                    data_folder=DATA_FOLDER, sv_folder=SV_FOLDER,
                    extra_labels=[len(coh),
                                  '_2_ro_rand_'+str(len(coh)),
                                  # '_1_ro_'+str(len(coh)),
                                  '_1_ro__com_modulation_'+str(len(coh)),
                                  '_prior_sign_1_ro_'+str(len(coh))])
if f6:
    print('Plotting Figure 6')
    # human traj plots
    # fig_6.supp_human_behavior(user_id=pc_name, sv_folder=SV_FOLDER)
    fig_6.fig_6_humans(user_id=pc_name, sv_folder=SV_FOLDER,
                       human_task_img=HUMAN_TASK_IMG, max_mt=1000, nm='300')
if f8:
    df_data = fp.get_human_data(user_id=pc_name, sv_folder=SV_FOLDER)
    choice = df_data.R_response.values*2-1
    # df_data['subjid'] = np.repeat('all', len(choice))
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
        trial_index = np.concatenate((trial_index, np.arange(len_task[j])+1))
    df_data['origidx'] = trial_index
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        pro_vs_re, trajs, x_val_at_updt =\
        fp.simulate_model_humans(df_data, stim=stim, load_params=True)
    MT = np.array([len(t) for t in trajs])
    mt_human = np.array(fp.get_human_mt(df_data))
    df_data['resp_len'] = mt_human
    df_data['coh2'] = np.copy(coh)
    df_data['allpriors'] = df_data.norm_allpriors.values
    # plot_MT_density_comparison(
    #     mt_human[mt_human < 800], MT[MT < 800])
    # plot_psyc(df_data.R_response.values*2-1, resp_fin, coh=df_data.avtrapz.values*5)
    # plot_RT_density_comparison(df_data.sound_len.values,
    #                             reaction_time)
    df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                            'sound_len': reaction_time,
                            'rewside': (gt + 1)/2,
                            'R_response': (resp_fin+1)/2,
                            'resp_len': np.array(MT)*1e-3})
    df_sim['CoM_sugg'] = com_model.astype(bool)
    df_sim['traj_d1'] = [np.diff(t) for t in trajs]
    df_sim['subjid'] = subjid  # .astype(int).astype(str)
    df_sim['origidx'] = trial_index
    df_sim['special_trial'] = np.repeat(0, len(subjid))
    df_sim['traj'] = df_sim['trajectory_y']
    df_sim['com_detected'] = com_model_detected.astype(bool)
    df_sim['peak_com'] = np.array(x_val_at_updt)
    df_sim['hithistory'] = np.array(resp_fin == (gt*2-1))
    df_sim['allpriors'] = df_data.norm_allpriors.values
    df_sim['norm_allpriors'] = df_data.norm_allpriors.values
    df_sim['normallpriors'] = df_sim['norm_allpriors']
    prev_perf = ~ (np.concatenate((np.array([True]),
                                      df_sim['hithistory'][:-1])) == 1)
    prev_perf = prev_perf.astype('int')
    df_sim['aftererror'] = prev_perf
    df_sim['framerate'] = 200
    df_sim = df_sim[df_sim.sound_len.values >= 0]
    means, errors = fig_1.mt_weights(df_data, means_errs=True, ax=None)
    means_model, errors_model = fig_1.mt_weights(df_sim, means_errs=True, ax=None)
    sound_len = df_data.sound_len.values
    zt = df_data.norm_allpriors.values
    com = df_data.CoM_sugg.values
    # fig_1.plot_mt_weights_rt_bins(df=df_sim, ax=None, rtbins=np.linspace(100, 300, 8))
    # fig_1.plot_mt_weights_rt_bins(df=df_data, ax=None, rtbins=np.linspace(100, 300, 8))
    fig_5h.fig_5_model(sv_folder=SV_FOLDER, data_folder=DATA_FOLDER,
                       new_data=False, save_new_data=False,
                       coh=coh, sound_len=sound_len, zt=zt,
                       hit_model=hit_model, sound_len_model=reaction_time.astype(int),
                       decision_model=resp_fin, com=com, com_model=com_model,
                       com_model_detected=com_model_detected,
                       means=means, errors=errors, means_model=means_model,
                       errors_model=errors_model, df_sim=df_sim)
