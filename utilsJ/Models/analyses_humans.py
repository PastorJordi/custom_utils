#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:28:46 2020
@author: molano
"""
import pandas as pd
# from psycho_priors import helper_functions as hf
# import helper_functions as hf
# import plotting_functions as pf
from scipy.optimize import curve_fit
from numpy import logical_and as and_
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from numpy import concatenate as conc
from numpy import logical_and as and_
import sys
sys.path.append("C:/Users/Alexandre/Documents/GitHub/")
sys.path.append("C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/" +\
                "Documentos/GitHub/custom_utils")
from utilsJ.Behavior.plotting import binned_curve,\
    com_heatmap_paper_marginal_pcom_side
from utilsJ.Models import extended_ddm_v2 as edd2
# GLOBAL VARIABLES
FIX_TIME = 0.5
NUM_BINS_RT = 6
NUM_BINS_MT = 7
MT_MIN = 0.05
MT_MAX = 1
START_ANALYSIS = 0  # trials para ignorar
RESP_W = 0.3
START_ANALYSIS = 0  # trials to ignore
GREEN = np.array((77, 175, 74))/255
PURPLE = np.array((152, 78, 163))/255
model_cols = ['evidence',
              'L+', 'L-', 'T+-', 'T-+', 'T--', 'T++', 'intercept']


def get_data(subj, main_folder):
    # subject folder
    folder = os.path.join(main_folder, subj, '*trials.csv')
    # find all data files
    files = glob.glob(folder)
    # take files names
    file_list = [os.path.basename(x) for x in files
                 if x.endswith('trials.csv')]
    # sort files
    sfx = [x[x.find('202'):x.find('202')+15] for x in file_list]

    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    print(sorted_list)
    # create data
    data = {'correct': np.empty((0,)), 'answer_response': np.empty((0,)),
            'soundPlay_object1_leftRightBalance': np.empty((0,)),
            'respondedInTime': np.empty((0,)), 'block': np.empty((0,)),
            'soundPlay_responseTime': np.empty((0,)),
            'soundPlay_duration': np.empty((0,)),
            'answer_responseTime': np.empty((0,))}
    # go over all files
    for f in sorted_list:
        # read file
        df1 = pd.read_csv(os.path.join(main_folder, subj, f), sep=',')
        for k in data.keys():
            values = df1[k].values
            if k == 'soundPlay_object1_leftRightBalance':
                values = values-.5
                values[np.abs(values) < 0.01] = 0
            data[k] = np.concatenate((data[k], values))
    return data


def tune_panel(ax, xlabel, ylabel, font=10):
    ax.set_xlabel(xlabel, fontsize=font)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def get_hist(values, bins):
    hist, x_hist = np.histogram(values, bins=bins)
    hist = hist/np.sum(hist)
    return hist, x_hist


def extract_vars_from_dict(data, steps=None):
    """
    Extracts data
    """
    steps = get_steps(steps, num_tr=len(data['correct']))
    ev = data['soundPlay_object1_leftRightBalance'][steps[0]:steps[1]]
    # TODO: change ev to decibels
    choice = data['answer_response'][steps[0]:steps[1]]
    perf = data['correct'][steps[0]:steps[1]]
    valid = data['respondedInTime'][steps[0]:steps[1]] == 1
    reaction_time = data['soundPlay_responseTime'][steps[0]:steps[1]]-FIX_TIME
    answ_rt = data['answer_responseTime'][steps[0]:steps[1]]
    sound_dur = data['soundPlay_duration'][steps[0]:steps[1]]-FIX_TIME
    blocks = data['block'][steps[0]:steps[1]]
    return ev, choice, perf, valid, reaction_time, blocks, answ_rt, sound_dur


def get_steps(steps, num_tr):
    if steps is None:
        steps = [0, num_tr]
    else:
        if steps < 0:
            steps = [num_tr+steps, num_tr]
        else:
            steps = [0, steps]
    return steps


def process_trajectories_rep_alt(data_tr, data_traj):
    """
    With this function, data from trials and trajectories is extracted.
    We want to study the slope of the lines so we take the diff on each
    trajectory and then we measure the average slope for each trajectory.
    Since the objective points are at ratio 2:1 to the center, each objective
    point is at ratio 1:1 to the starting point so the slope must be
    approximately 1. With this function we are computing the slope, the
    trajectories and separating them by correct/incorrect, to see if there
    is some bias in these.
    ***
    Inputs:
            data_tr : data from trials
            data traj: data from trajectories
            condition: wheter the user wants to see repetition (rep),
            alternation (alt), or total (all) data
   Outputs:
           dict_all: has information about trajectories, slopes, left/right
           choices, correct/incorrect choices
    """
    choice = np.array(data_tr['answer_response'])
    correct = np.array(data_tr['correct'])
    x_traj = [x for x in data_traj['answer_positionsX']
              if x not in [np.nan]]
    y_traj = [x for x in data_traj['answer_positionsY']
              if x not in [np.nan]]
    times = [x for x in data_traj['answer_times']
             if x not in [np.nan]]
    # TODO: define lists
    difference_x = []  # XXX: these 2 variables are not saved later
    difference_y = []
    slope = []
    slope_mean = []
    choice_f = []  # XXX: this variable is not saved later
    difference_y = []  # XXX: repeated
    correct_f = []  # XXX: this variable is not saved later
    slope_correct = []
    slope_incorrect = []
    xtraj_correct = []
    ytraj_correct = []
    xtraj_incorrect = []
    ytraj_incorrect = []
    right_correct = []
    right_incorrect = []
    left_correct = []
    left_incorrect = []
    right = []
    left = []
    ind_cor = []
    ind_incor = []
    # first we process the data from the repetition pattern
    for inde in range(len(x_traj)):
        x_traj[inde] = str(x_traj[inde]).split(';')
        y_traj[inde] = str(y_traj[inde]).split(';')
        times[inde] = str(times[inde]).split(';')
        for i in range(len(x_traj[inde])):
            x_traj[inde][i] = float(x_traj[inde][i])
            times[inde][i] = float(times[inde][i])
        difference_x.append(np.diff(x_traj[inde]))  # compute the diff in x
        for j in range(len(y_traj[inde])):
            y_traj[inde][j] = float(y_traj[inde][j])
        difference_y.append(np.diff(y_traj[inde]))  # compute the diff in y
        slope_dummy = []
        for r in range(len(np.diff(y_traj[inde]))):
            # compute the slope by steps of a trajectory
            slope_dummy.append(np.diff(y_traj[inde])[r] /
                               np.diff(x_traj[inde])[r])
        slope.append(slope_dummy)
        slope_mean.append(np.mean(slope_dummy))  # compute the mean of slope
        choice_f.append(int(choice[inde]))  # save choice
        correct_f.append(correct[inde])  # save correct answer
        if correct[inde] == 1:
            # if subject is correct
            slope_correct.append(np.mean(slope_dummy))  # save slope
            xtraj_correct.append(x_traj[inde])  # save trajectory in x
            ytraj_correct.append(y_traj[inde])  # save trajectory in y
            ind_cor.append(inde)
            if int(choice[inde]) == 1:  # if choice is right
                right_correct.append(np.mean(slope_dummy))  # append mean
                right.append(inde)  # also append the index
            else:
                left_correct.append(np.mean(slope_dummy))  # same with left
                left.append(inde)
        else:  # if choice is incorrect
            slope_incorrect.append(np.mean(slope_dummy))  # same as before
            xtraj_incorrect.append(x_traj[inde])
            ytraj_incorrect.append(y_traj[inde])
            ind_incor.append(inde)
            if int(choice[inde]) == 1:
                right_incorrect.append(np.mean(slope_dummy))
                right.append(inde)
            else:
                left_incorrect.append(np.mean(slope_dummy))
                left.append(inde)
    slope_mean = np.nan_to_num(slope_mean, copy=True, posinf=10)
    slope_incorrect = np.nan_to_num(slope_incorrect, copy=True, posinf=10)
    slope_correct = np.nan_to_num(slope_correct, copy=True, posinf=10)
    right_incorrect = np.nan_to_num(right_incorrect, copy=True, posinf=10)
    right_correct = np.nan_to_num(right_correct, copy=True, posinf=10)
    left_incorrect = np.nan_to_num(left_incorrect, copy=True, posinf=10)
    left_correct = np.nan_to_num(left_correct, copy=True, posinf=10)
    # replace infinite by 10 and save dictionary
    dict_all = {'slope': slope, 'mean': slope_mean, 'choice': choice,
                'correct': correct, 'x_correct': xtraj_correct,
                'y_correct': ytraj_correct,
                'x_incorrect': xtraj_incorrect,
                'y_incorrect': ytraj_incorrect,
                'slope_correct': slope_correct,
                'slope_incorrect': slope_incorrect,
                'right_correct': right_correct, 'left_correct': left_correct,
                'left_incorrect': left_incorrect,
                'right_incorrect': right_incorrect, 'right': right,
                'left': left, 'x_traj': x_traj, 'y_traj': y_traj,
                'ind_cor': ind_cor, 'ind_incor': ind_incor, 'times': times}
    return dict_all


def data_processing(data_tr, data_traj, rgrss_folder, sv_folder,
                   com_threshold=50, plot=False):
    """


    Parameters
    ----------
    data_tr : dataframe
        must contain all trial's data already extracted from 2AFC
        human task.
    data_traj : dataframe
        must contain all trial's trajectory data already extracted from 2AFC
        human task.
    com_threshold : int or float, optional
        Threshold in pixels to detect CoMs in trajectories. The default is 50.
    plot : bool, optional
        Wether to plot or not. The default is False.

    Returns
    -------
    com_list : list
        Boolean list that indicates the CoMs in its corresponding index.

    """
    ev, choice, perf, valid, reaction_time, blocks, answ_rt, sound_dur =\
        extract_vars_from_dict(data_tr, steps=None)
    subjid = data_tr['subjid']
    choice_12 = choice + 1
    choice_12[~valid] = 0
    data = {'signed_evidence': ev, 'choice': choice_12,
            'performance': perf}
    if rgrss_folder is None:
        df_regressors = get_GLM_regressors(data, tau=2)
        df_regressors.to_csv(sv_folder + 'df_regressors_all_sub.csv')
    else:
        df_regressors = pd.read_csv(rgrss_folder+'df_regressors_all_sub.csv')
    ind_af_er = df_regressors['aftererror'] == 0
    subjid = subjid[ind_af_er]
    ev = ev[ind_af_er]
    perf = perf[ind_af_er]
    valid = valid[ind_af_er]
    prior = np.nansum((df_regressors['T++'], df_regressors['T++']), axis=0)/2
    prior = prior[ind_af_er]
    blocks = blocks[ind_af_er]
    pos_x = data_traj['answer_positionsX']
    pos_y = data_traj['answer_positionsY']
    answer_times = [x for x in data_traj['answer_times']
                    if x not in [np.nan]]
    for inde in range(len(choice_12)):
        answer_times[inde] = answer_times[inde].split(';')
        for i in range(len(pos_x[inde])):
            if i == 0:
                answer_times[inde][i] = 0
            else:
                answer_times[inde][i] = float(answer_times[inde][i])
    answer_times = np.array(answer_times, dtype=object)[ind_af_er]
    pos_x = pos_x[ind_af_er]
    choice = choice[ind_af_er]
    pos_y = pos_y[ind_af_er]
    choice_signed = choice*2 - 1
    reaction_time = reaction_time[ind_af_er]
    com_list = []
    com_peak = []
    time_com = []
    for i, traj in enumerate(pos_x):
        traj = np.array(traj)
        max_val = max((traj) * (-choice_signed[i]))
        com_list.append(max_val > com_threshold)
        if max_val > 0:
            com_peak.append(max_val)
        else:
            com_peak.append(0)
        if answer_times[i][-1] != '' and max_val > com_threshold:
            time_com_ind = np.array(answer_times[i])[traj == max_val]
            try:
                if len(time_com_ind) >= 1:
                    time_com.append(time_com_ind[0])
                else:
                    time_com.append(-1)
            except Exception:
                time_com.append(time_com_ind)
        else:
            time_com.append(-1)
    df_plot = pd.DataFrame({'sound_len': reaction_time*1e3, 'CoM': com_list,
                            'ev': ev})
    indx = ~np.isnan(ev)
    com_list = np.array(com_list)
    avtrapz = ev[indx]
    CoM_sugg = com_list[indx]
    norm_allpriors = prior[indx]/max(abs(prior[indx]))
    R_response = choice[indx]
    blocks = blocks[indx]
    for i, e_val in enumerate(avtrapz):
        if abs(e_val) > 1:
            avtrapz[i] = np.sign(e_val)
    df_data = pd.DataFrame({'avtrapz': avtrapz, 'CoM_sugg': CoM_sugg,
                            'norm_allpriors': norm_allpriors,
                            'R_response': R_response,
                            'sound_len': reaction_time[indx]*1e3,
                            'hithistory': perf[indx],
                            'trajectory_y': pos_x[indx],
                            'times': answer_times[indx],
                            'traj_y': pos_y[indx],
                            'subjid': subjid[indx],
                            'com_peak': np.array(com_peak)[indx],
                            'time_com': np.array(time_com)[indx],
                            'blocks': blocks})
    if plot:
        fig, ax = plt.subplots(1)
        bins = np.linspace(0, 350, 8)  # rt bins
        xpos = np.diff(bins)[0]  # rt binss
        binned_curve(df_plot, 'CoM', 'sound_len', bins=bins,
                     xoffset=min(bins), xpos=xpos, ax=ax,
                     errorbar_kw={'marker': 'o'})
        ax.set_title('Detection threshold: ' + str(com_threshold) + ' px')
        ax.set_xlabel('RT (ms)')
        ax.set_ylabel('pCoM')
        ax.set_ylim(0, 0.12)
        plt.figure()
        edd2.com_heatmap_jordi(prior[indx], ev[indx], com_list[indx],
                               annotate=False, flip=True)
        plt.title('pCoM')
        plt.figure()
        edd2.com_heatmap_jordi(prior[indx], ev[indx], choice[indx],
                               annotate=False, flip=True,
                               xlabel='Normalized prior')
        plt.title('pRight')

        # f, ax = plt.subplots(ncols=5, nrows=3, figsize=(6, 3),
        #                      gridspec_kw={'width_ratios': [12, 3, 3, 12, 3],
        #                                   'height_ratios': [12, 3, 12],
        #                                   'top': 0.969, 'bottom': 0.081,
        #                                   'left': 0.081, 'right': 0.88,
        #                                   'hspace': 0.215, 'wspace': 0.25})

        # ax_side1 = ax[1:3, 0:2]
        # com_heatmap_paper_marginal_pcom_side(df_data, f=f, ax=ax_side1, side=1,
        #                                      hide_marginal_axis=True)
        # ax_side1[1, 0].set_xlabel(r'$\longleftarrow$Prior$\longrightarrow$',
        #                           labelpad=15)
        # ax_side1[1, 0].set_ylabel(
        #     r'$\longleftarrow$Average stimulus$\longrightarrow$', labelpad=-5)
        # ax_side0 = ax[1:3, 3:5]
        # com_heatmap_paper_marginal_pcom_side(df_data, f=f, ax=ax_side0, side=0,
        #                                      hide_marginal_axis=True)
        # for j in range(3):
        #     ax[j, 2].axis('off')
        # ax_side0[1, 0].set_xlabel(r'$\longleftarrow$Prior$\longrightarrow$',
        #                           labelpad=15)
        # ax_side0[1, 0].set_ylabel(
        #     r'$\longleftarrow$Average stimulus$\longrightarrow$', labelpad=15)
        # # edd2.com_heatmap_jordi(prior[indx], ev[indx], com_list[indx],
        # #                        annotate=False, flip=True, ax=ax[0, 0],
        # #                        cbar_location='left')
        # # ax[0, 0].set_title('pCoM')
        # ax[0, 1].axis('off')
        # ax[0, 4].axis('off')
        # ax[0, 0].axis('off')
        # edd2.com_heatmap_jordi(prior[indx], ev[indx], choice[indx],
        #                        annotate=False, flip=True, ax=ax[0, 3],
        #                        cbar_location='left', cmap='rocket')
        # ax[0, 3].set_title('pRight')
        # f.savefig(sv_folder + '\\figure_3_cosyne.svg')
        plot_coms(com_list=com_list, pos_x=pos_x, answer_times=answer_times,
                  com_threshold=com_threshold)
        plot_init_point_prior(pos_x, prior, choice, com_list)
    return df_data


def plot_coms(com_list, pos_x, answer_times, com_threshold, init_zero=False,
              time_bounds=[0.2, 0.3]):
    plt.figure()
    j = 0
    for i, p in enumerate(com_list):
        time_max = answer_times[i][-1]
        if time_max != '':
            if p and time_max < time_bounds[1] and time_max > time_bounds[0]:
                color = 'green' if pos_x[i][-1] < 0 else 'purple'
                if init_zero:
                    plt.plot(np.array(answer_times[i])*1e3,
                             np.array(pos_x[i]) - pos_x[i][0], color=color)
                else:
                    plt.plot(np.array(answer_times[i])*1e3,
                             np.array(pos_x[i]), color=color)
                j += 1
        if j == 100:
            break
    plt.xlabel('Time from movement onset (ms)')
    plt.ylabel('x-coordinate (px)')
    plt.axhline(-com_threshold, linestyle='--', color='k', alpha=0.6)
    plt.axhline(com_threshold, linestyle='--', color='k', alpha=0.6)
    plt.axhline(y=0, linestyle='solid', color='b', alpha=0.1)


def plot_init_point_prior(pos_x, prior, choice, com_list):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    right_resp = [x[0] for x in pos_x[choice == 1]]
    left_resp = [x[0] for x in pos_x[choice == 0]]
    right_resp_com = [x[0] for x in pos_x[(choice == 1)*com_list]]
    left_resp_com = [x[0] for x in pos_x[(choice == 0)*com_list]]
    ax[0].boxplot([right_resp, right_resp_com, left_resp, left_resp_com])
    ax[0].set_xticklabels(labels=['R', 'R-CoM', 'L', 'L-CoM'],
                          minor=False)
    ax[0].set_ylabel('x position at t=0 (px)')
    bins_prior = np.linspace(-max(prior)-0.01, max(prior)+0.01, num=10)
    xoffset = bins_prior[0]
    xpos = np.diff(bins_prior)[0]
    data_plot = pd.DataFrame({'prior_no_CoM': prior[~com_list],
                              'pos_x_0_no_CoM': [x[0] for x in pos_x[~com_list]]})
    binned_curve(data_plot, 'pos_x_0_no_CoM', 'prior_no_CoM', bins=bins_prior,
                 xpos=xpos, xoffset=xoffset, errorbar_kw={'label': 'no-CoM'},
                 ax=ax[1])
    data_plot = pd.DataFrame({'prior_CoM': prior[com_list],
                              'pos_x_0_CoM': [x[0] for x in pos_x[com_list]]})
    binned_curve(data_plot, 'pos_x_0_CoM', 'prior_CoM', bins=bins_prior,
                 xpos=xpos, xoffset=xoffset, errorbar_kw={'label': 'CoM'},
                 ax=ax[1])
    ax[1].set_xlabel('Prior')
    ax[1].set_ylabel('x position at t=0 (px)')


def traj_analysis(data_folder, sv_folder, subjects, steps=[None], name=''):
    for i_s, subj in enumerate(subjects):
        print('-----------')
        print(subj)
        folder = data_folder+subj+'/'
        for i_stp, stp in enumerate(steps):
            data_tr, data_traj = get_data_traj(folder=folder)
            df_data = data_processing(data_tr, data_traj, com_threshold=100,
                                      plot=False, rgrss_folder=data_folder,
                                      sv_folder=sv_folder)
    return df_data


def get_data_traj(folder, plot=False):
    """
    Extracts trajectories and psychometric data.
    Inputs: subject name (subj) and main_folder
    Outputs: psychometric data and trajectories data in two different dictionaries
    """
    # subject folder
    # folder = main_folder+'\\'+subj+'\\'  # Alex
    # find all data files
    # copy_files(ori_f=folder[:folder.find('general_traj')], fin_f=folder)
    files_trials = glob.glob(folder+'*trials.csv')
    files_traj = glob.glob(folder+'*trials-trajectories.csv')
    # take files names
    file_list_trials = [os.path.basename(x) for x in files_trials
                        if x.endswith('trials.csv')]
    file_list_traj = [os.path.basename(x) for x in files_traj
                      if x.endswith('trials-trajectories.csv')]
    # sort files
    sfx_tls = [x[x.find('202'):x.find('202')+15] for x in file_list_trials]
    sfx_trj = [x[x.find('202'):x.find('202')+15] for x in file_list_traj]

    sorted_list_tls = [x for _, x in sorted(zip(sfx_tls, file_list_trials))]
    sorted_list_trj = [x for _, x in sorted(zip(sfx_trj, file_list_traj))]
    # create data
    data_tls = {'correct': np.empty((0,)), 'answer_response': np.empty((0,)),
                'soundPlay_object1_leftRightBalance': np.empty((0,)),
                'respondedInTime': np.empty((0,)), 'block': np.empty((0,)),
                'soundPlay_responseTime': np.empty((0,)),
                'soundPlay_duration': np.empty((0,)),
                'answer_responseTime': np.empty((0,))}
    # go over all files
    subjid = np.empty((0,))
    for i_f, f in enumerate(sorted_list_tls):
        # read file
        df1 = pd.read_csv(folder+'/'+f, sep=',')  # Manuel
        # df1 = pd.read_csv(folder+'\\'+f, sep=',')  # Alex
        if np.mean(df1['correct']) < 0.4:
            continue
        else:
            for k in data_tls.keys():
                values = df1[k].values[START_ANALYSIS:]
                if k == 'soundPlay_object1_leftRightBalance':
                    values = values-.5
                    values[np.abs(values) < 0.01] = 0
                data_tls[k] = np.concatenate((data_tls[k], values))
            subjid = np.concatenate((subjid, np.repeat(i_f+1, len(values))))
    data_tls['subjid'] = subjid
    num_tr = len(data_tls['correct'])
    data_trj = {'answer_positionsX': np.empty((0,)),
                'answer_positionsY': np.empty((0,)),
                'answer_times': np.empty((0,))}
    if plot:
        _, ax = plt.subplots()
    for i_f, f in enumerate(sorted_list_trj):
        # read file
        df1 = pd.read_csv(folder+'/'+f, sep=',')  # Manuel
        df2 = pd.read_csv(folder+'/'+sorted_list_tls[i_f], sep=',')
        # df1 = pd.read_csv(folder+'\\'+f, sep=',')  # Alex
        if np.mean(df2['correct']) < 0.4:
            print('subject discarded: ' + str(i_f+1))
            print('acc: ' + str(np.mean(df2['correct'])))
            continue
        else:
            pos_x = df1['answer_positionsX'].dropna().values[:num_tr]
            pos_y = df1['answer_positionsY'].dropna().values[:num_tr]
            cont = 0
            for ind_trl in range(len(pos_x)):
                if cont == 1 and df1['trial'][ind_trl] == 1:
                    break
                if df1['trial'][ind_trl] == 1:
                    cont = 1
                pos_x[ind_trl] = [float(x) for x in pos_x[ind_trl].split(';')]
                pos_y[ind_trl] = [float(x) for x in pos_y[ind_trl].split(';')]
                if plot:
                    color = PURPLE if data_tls['answer_response'][ind_trl] == 1\
                        else GREEN
                    ax.plot(pos_x[ind_trl], pos_y[ind_trl], color=color)
            k = 'answer_positionsX'
            data_trj[k] = np.concatenate((data_trj[k], pos_x))
            k = 'answer_positionsY'
            data_trj[k] = np.concatenate((data_trj[k], pos_y))
            k = df1.columns[-1]
            values = df1[k].dropna().values
            k = 'answer_times'
            data_trj[k] = np.concatenate((data_trj[k], values))
    return data_tls, data_trj


def get_repetitions(mat):
    """
    Return mask indicating the repetitions in mat.
    Makes diff of the input vector, mat, to obtain the repetition vector X,
    i.e. X will be 1 at t if the value of mat at t is equal to that at t-1
    Parameters
    ----------
    mat : array
        array of elements.
    Returns
    -------
    repeats : array
        mask indicating the repetitions in mat.
    """
    mat = mat.flatten()
    values = np.unique(mat)
    # We need to account for size reduction of np.diff()
    rand_ch = np.array(np.random.choice(values, size=(1,)))
    repeat_choice = conc((rand_ch, mat))
    diff = np.diff(repeat_choice)
    repeats = (diff == 0)*1.
    repeats[np.isnan(diff)] = np.nan
    return repeats


def nanconv(vec_1, vec_2):
    """
    This function returns a convolution result of two vectors without
    considering nans
    """
    mask = ~np.isnan(vec_1)
    return np.nansum(np.multiply(vec_2[mask], vec_1[mask]))


def get_GLM_regressors(data, tau, chck_corr=False):
    """
    Compute regressors.
    Parameters
    ----------
    data : dict
        dictionary containing behavioral data.
    chck_corr : bool, optional
        whether to check correlations (False)
    Returns
    -------
    df: dataframe
        dataframe containg evidence, lateral and transition regressors.
    """
    ev = data['signed_evidence'][START_ANALYSIS::]  # coherence/evidence with sign
    perf = data['performance'].astype(float)  # performance (0/1)
    ch = data['choice'][START_ANALYSIS::].astype(float)  # choice (1, 2)
    # discard (make nan) non-standard-2afc task periods
    if 'std_2afc' in data.keys():
        std_2afc = data['std_2afc'][START_ANALYSIS::]
    else:
        std_2afc = np.ones_like(ch)
    inv_choice = and_(ch != 1., ch != 2.)
    nan_indx = np.logical_or.reduce((std_2afc == 0, inv_choice))

    ev[nan_indx] = np.nan
    perf[nan_indx] = np.nan
    ch[nan_indx] = np.nan
    ch = ch-1  # choices should belong to {0, 1}
    prev_perf = ~ (conc((np.array([True]), data['performance'][:-1])) == 1)
    prev_perf = prev_perf.astype('int')
    prevprev_perf = (conc((np.array([False]), prev_perf[:-1])) == 1)
    ev /= np.nanmax(ev)
    rep_ch_ = get_repetitions(ch)
    # variables:
    # 'origidx': trial index within session
    # 'rewside': ground truth
    # 'hithistory': performance
    # 'R_response': choice (right == 1, left == 0, invalid == nan)
    # 'subjid': subject
    # 'sessid': session
    # 'res_sound': stimulus (left - right) [frame_i, .., frame_i+n]
    # 'sound_len': stim duration
    # 'frames_listened'
    # 'aftererror': not(performance) shifted
    # 'rep_response'
    df = {'origidx': np.arange(ch.shape[0]),
          'R_response': ch,
          'hit': perf,
          'evidence': ev,
          'aftererror': prev_perf,
          'rep_response': rep_ch_,
          'prevprev_perf': prevprev_perf}
    df = pd.DataFrame(df)

    # Lateral module
    df['L+1'] = np.nan  # np.nan considering invalids as errors
    df.loc[(df.R_response == 1) & (df.hit == 1), 'L+1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 1), 'L+1'] = -1
    df.loc[df.hit == 0, 'L+1'] = 0
    df['L+1'] = df['L+1'].shift(1)
    df.loc[df.origidx == 1, 'L+1'] = np.nan
    # L-
    df['L-1'] = np.nan
    df.loc[(df.R_response == 1) & (df.hit == 0), 'L-1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 0), 'L-1'] = -1
    df.loc[df.hit == 1, 'L-1'] = 0
    df['L-1'] = df['L-1'].shift(1)
    df.loc[df.origidx == 1, 'L-1'] = np.nan

    # pre transition module
    df.loc[df.origidx == 1, 'rep_response'] = np.nan
    df['rep_response_11'] = df.rep_response
    df.loc[df.rep_response == 0, 'rep_response_11'] = -1
    df.rep_response_11.fillna(value=0, inplace=True)
    df.loc[df.origidx == 1, 'aftererror'] = np.nan

    # transition module
    df['T++1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 1), 'T++1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 0), 'T++1'] = 0
    df['T++1'] = df['T++1'].shift(1)

    df['T+-1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 0), 'T+-1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 1), 'T+-1'] = 0
    df['T+-1'] = df['T+-1'].shift(1)

    df['T-+1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 1), 'T-+1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 0), 'T-+1'] = 0
    df['T-+1'] = df['T-+1'].shift(1)

    df['T--1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 0), 'T--1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 1), 'T--1'] = 0
    df['T--1'] = df['T--1'].shift(1)

    # exponential fit for T++
    decay_tr = np.exp(-np.arange(10)/tau)  # exp(-x/tau)
    regs = [x for x in model_cols if x != 'intercept' and x != 'evidence']
    N = len(decay_tr)
    for reg in regs:  # all regressors (T and L)
        df[reg] = df[reg+str(1)]
        for j in range(N, len(df[reg+str(1)])):
            df[reg][j-1] = nanconv(df[reg+str(1)][j-N:j], decay_tr[::-1])
            # its j-1 for shifting purposes

    # transforming transitions to left/right space
    for col in [x for x in df.columns if x.startswith('T')]:
        df[col] = df[col] * (df.R_response.shift(1)*2-1)
        # {-1 = Left; 1 = Right, nan=invalid}

    df['intercept'] = 1

    df.loc[:, model_cols].fillna(value=0, inplace=True)
    # check correlation between regressors

    return df  # resulting df with lateralized T


def copy_files(ori_f, fin_f):
    def cp(name):
        files = glob.glob(f+'*'+name)
        for fl in files:
            shutil.copyfile(fl, fin_f+os.path.basename(fl))
    import shutil
    subjects = ['ruben', 'sophia', 'cris', 'beatriz', 'ilaria', 'carlo',
                'valeria', 'eugenia', 'richard', 'alessia_03', 'lorenzo2',
                'marina', 'alexv', 'clara', 'sergio', 'stan',
                'luca', 'alice', 'maria', 'arnau']
    for sbj in subjects:
        f = ori_f+sbj+'/'
        cp(name='trajectories.csv')
        cp(name='trials.csv')


def psycho_curves_rep_alt(df_data, ax):
    
    # MEAN PSYCHO-CURVES FOR REP/ALT, AFTER CORRECT/ERROR
    rojo = np.array((228, 26, 28))/255
    azul = np.array((55, 126, 184))/255
    colors = [rojo, azul]
    ttls = ['']
    bias_final = []
    slope_final = []
    subjects = df_data.subjid.unique()
    fig2, ax2 = plt.subplots(ncols=3)
    lbs = ['after error', 'after correct']
    median_rep_alt = np.empty((len(subjects), 2, 7))
    cohs_rep_alt = np.empty((len(subjects), 2, 7))
    for i_s, subj in enumerate(subjects):
        df_sub = df_data.loc[df_data.subjid == subj]
        ev = df_sub.avtrapz
        choice_12 = df_sub.R_response.values + 1
        blocks = df_sub.blocks
        perf = df_sub.hithistory
        prev_perf = np.concatenate((np.array([0]), perf[:-1]))
        all_means = []
        all_xs = []
        biases = []
        for i_b, blk in enumerate([1, 2]):  # blk = 1 --> alt / blk = 2 --> rep
            p = 1
            alpha = 1 if p == 0 else 1
            lnstyl = '-' if p == 0 else '-'
            plt_opts = {'color': colors[i_b],
                        'alpha': alpha, 'linestyle': lnstyl}
            # rep/alt
            popt, pcov, ev_mask, repeat_mask =\
                bias_psychometric(choice=choice_12.copy(), ev=-ev.copy(),
                                     mask=and_(prev_perf == p,
                                               blocks == blk),
                                     maxfev=100000)
            # this is to avoid rounding differences
            ev_mask = np.round(ev_mask, 2)
            d =\
                plot_psycho_curve(ev=ev_mask, choice=repeat_mask,
                                     popt=popt, ax=ax2[p],
                                     color_scatter=colors[i_b],
                                     label=lbs[p], plot_errbars=True,
                                     **plt_opts)
            means = d['means']
            xs = d['xs']
            all_means.append(means)
            all_xs.append(xs)
            biases.append(popt[1])
        median_rep_alt[i_s] = np.array([np.array(a) for a in all_means])
        cohs_rep_alt[i_s] = np.array([np.array(a) for a in all_xs])
    plt.close(fig2)
    labels = ['Alternating', 'Repeating']
    for i_b, blk in enumerate([1, 2]):
        ip = 0
        p = 1
        if i_b == 0:
            ax.axvline(x=0., linestyle='--', lw=0.2,
                             color=(.5, .5, .5))
            ax.axhline(y=0.5, linestyle='--', lw=0.2,
                             color=(.5, .5, .5))
            ax.set_title(ttls[ip])
            ax.set_yticks([0, 0.5, 1])
            tune_panel(ax=ax, xlabel='Repeating stimulus evidence',
                       ylabel='p(repeat response)')
        ax.plot(cohs_rep_alt[:, i_b, :].flatten(),
                median_rep_alt[:, i_b, :].flatten(),
                color=colors[i_b], alpha=0.2, linestyle='',
                marker='+')
        medians = np.median(median_rep_alt, axis=0)[i_b]
        sems = np.std(median_rep_alt, axis=0)[i_b] /\
            np.sqrt(median_rep_alt.shape[0])
        ax.errorbar(cohs_rep_alt[0][0], medians, sems,
                    color=colors[i_b], marker='.', linestyle='')
        ev_gen = cohs_rep_alt[0, i_b, :].flatten()
        popt, pcov = curve_fit(probit_lapse_rates,
                               ev_gen,
                               np.median(median_rep_alt, axis=0)[i_b],
                               maxfev=10000)
        bias_final.append(popt[1])
        slope_final.append(popt[0])
        x_fit = np.linspace(-np.max(ev), np.max(ev), 20)
        y_fit = probit_lapse_rates(x_fit, popt[0], popt[1], popt[2],
                                   popt[3])
        ax.plot(x_fit, y_fit, color=colors[i_b], label=labels[i_b])
    ax.legend(title='Context', bbox_to_anchor=(0.5, 1.12), frameon=False,
              handlelength=1.2)


def probit(x, beta, alpha):
    from scipy.special import erf
    """
    Return probit function with parameters alpha and beta.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha.

    """
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def probit_lapse_rates(x, beta, alpha, piL, piR):
    """
    Return probit with lapse rates.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.
    piL : float
        lapse rate for left side.
    piR : TYPE
        lapse rate for right side.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha and lapse rates.

    """
    piL = 0
    piR = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def plot_psycho_curve(ev, choice, popt, ax, color_scatter, plot_errbars=False,
                      **plt_opts):
    """
    Plot psycho-curves (fits and props) using directly the fit parameters.

    THIS FUNCTION ASSUMES PUTATIVE EVIDENCE (it will compute response proportions
                                             for all values of ev)

    Parameters
    ----------
    ev : array
        array with **putative** evidence for each trial.
    choice : array
        array with choices made by agent.
    popt : list
        list containing fitted parameters (beta, alpha, piL, piR).
    ax : axis
        where to plot.
    **plt_opts : dict
        plotting options.

    Returns
    -------
    means : list
        response means for each evidence value.
    sems : list
        sem for the responses.
    x : array
        evidences values for which the means/sems are computed.
    y_fit : array
        y values for the fit.
    x_fit : array
        x values for the fit.

    """
    x_fit = np.linspace(np.min(ev), np.max(ev), 20)
    y_fit = probit_lapse_rates(x_fit, popt[0], popt[1], popt[2], popt[3])
    ax.plot(x_fit, y_fit, markersize=6, **plt_opts)
    means = []
    sems = []
    n_samples = []
    for e in np.unique(ev):
        means.append(np.mean(choice[ev == e]))
        sems.append(np.std(choice[ev == e])/np.sqrt(np.sum(ev == e)))
        n_samples.append(np.sum(ev == e))
    x = np.unique(ev)
    plt_opts['linestyle'] = ''
    if 'label' in plt_opts.keys():
        del plt_opts['label']
    if plot_errbars:
        ax.errorbar(x, means, sems, **plt_opts)
    ax.scatter(x, means, marker='.', alpha=1, s=60, c=color_scatter)
    ax.plot([0, 0], [0, 1], '--', lw=0.2, color=(.5, .5, .5))
    d_list = [means, sems, x, y_fit, x_fit, n_samples]
    d_str = ['means, sems, xs, y_fit, x_fit, n_samples']
    d = list_to_dict(d_list, d_str)
    return d



def plot_rep_alt_psycho_curve(choice_12, ev, prev_perf, blocks,
                              rep_alt_panel, lbs):
    from numpy import logical_and as and_
    rojo = np.array((228, 26, 28))/255
    azul = np.array((55, 126, 184))/255
    colors = [rojo, azul]
    all_means = []
    all_xs = []
    biases = []
    for i_b, blk in enumerate([1, 2]):  # blk = 1 --> alt / blk = 2 --> rep
        for p in [0, 1]:
            plt.sca(rep_alt_panel[p])
            alpha = 1 if p == 0 else 1
            lnstyl = '-' if p == 0 else '-'
            plt_opts = {'color': colors[i_b],
                        'alpha': alpha, 'linestyle': lnstyl}
            # rep/alt
            popt, pcov, ev_mask, repeat_mask =\
                bias_psychometric(choice=choice_12.copy(), ev=-ev.copy(),
                                  mask=and_(prev_perf == p,
                                            blocks == blk),
                                  maxfev=100000)
            # this is to avoid rounding differences
            ev_mask = np.round(ev_mask, 2)
            d =\
                plot_psycho_curve(ev=ev_mask, choice=repeat_mask,
                                  popt=popt, ax=rep_alt_panel[p],
                                  color_scatter=colors[i_b],
                                  label=lbs[p], plot_errbars=True,
                                  **plt_opts)
            means = d['means']
            xs = d['xs']
            if blk == 1:
                if p == 1:
                    x_alt_ac = d['x_fit']
                    y_alt_ac = d['y_fit']
                else:
                    x_alt_ae = d['x_fit']
                    y_alt_ae = d['y_fit']
            elif blk == 2:
                if p == 1:
                    x_rep_ac = d['x_fit']
                    y_rep_ac = d['y_fit']
                else:
                    x_rep_ae = d['x_fit']
                    y_rep_ae = d['y_fit']
            all_means.append(means)
            all_xs.append(xs)
            biases.append(popt[1])
    d_curves = {'x_alt_ac': x_alt_ac, 'y_alt_ac': y_alt_ac,
                'x_rep_ac': x_rep_ac, 'y_rep_ac': y_rep_ac,
                'x_alt_ae': x_alt_ae, 'y_alt_ae': y_alt_ae,
                'x_rep_ae': x_rep_ae, 'y_rep_ae': y_rep_ae}
    return all_means, all_xs, biases, d_curves  # TODO: return bias (popt[1])


def list_to_dict(lst, string):
    """
    Transform a list of variables into a dictionary.

    Parameters
    ----------
    lst : list
        list with all variables.
    string : str
        string containing the names, separated by commas.

    Returns
    -------
    d : dict
        dictionary with items in which the keys and the values are specified
        in string and lst values respectively.

    """
    string = string[0]
    string = string.replace(']', '')
    string = string.replace('[', '')
    string = string.replace('\\', '')
    string = string.replace(' ', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    string = string.split(',')
    d = {s: v for s, v in zip(string, lst)}
    return d

def bias_psychometric(choice, ev, mask=None, maxfev=10000):
    """
    Compute repeating bias by fitting probit function.

    Parameters
    ----------
    choice : array
        array of choices made bythe network.
    ev : array
        array with (signed) stimulus evidence.
    mask : array, optional
        array of booleans indicating the trials on which the bias
    # should be computed (None)

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of probit(xdata) - ydata is minimized
    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.

    """
    choice = choice.astype(float)
    choice[and_(choice != 1, choice != 2)] = np.nan
    repeat = get_repetitions(choice).astype(float)
    repeat[np.isnan(choice)] = np.nan
    # choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    choice_repeating = conc(
        (np.array(np.random.choice([1, 2])).reshape(1, ),
         choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev = ev*(-1)**(choice_repeating == 2)
    if mask is None:
        mask = ~np.isnan(repeat)
    else:
        mask = and_(~np.isnan(repeat), mask)
    rep_ev_mask = rep_ev[mask]  # xdata
    repeat_mask = repeat[mask]  # ydata
    try:
        # Use non-linear least squares to fit probit to xdata, ydata
        popt, pcov = curve_fit(probit_lapse_rates, rep_ev_mask,
                               repeat_mask, maxfev=maxfev)
    except RuntimeError as err:
        print(err)
        popt = [np.nan, np.nan, np.nan, np.nan]
        pcov = 0
    return popt, pcov, rep_ev_mask, repeat_mask