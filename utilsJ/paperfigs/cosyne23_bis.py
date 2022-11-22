# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:05:37 2022

@author: Alexandre
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem
import sys
# from scipy import interpolate
# sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("C:/Users/Alexandre/Documents/psycho_priors") 
from utilsJ.Models import simul
from utilsJ.Models import extended_ddm_v2 as edd2
import figures_paper as fp
import analyses
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve,\
    trajectory_thr, com_heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fig1, fig3, fig2
import matplotlib
SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
# DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
# SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
#     'ChangesOfMind/figures/from_python/'  # Manuel
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM


matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3


def com_heatmap_paper_marginal_pcom_side(
    df, f=None, ax=None,  # data source, must contain 'avtrapz' and allpriors
    pcomlabel=None, fcolorwhite=True, side=0,
    hide_marginal_axis=True, n_points_marginal=None, counts_on_matrix=False,
    adjust_marginal_axes=False,  # sets same max=y/x value
    nbins=7,  # nbins for the square matrix
    com_heatmap_kws={},  # avoid binning & return_mat already handled by the functn
    com_col='CoM_sugg', priors_col='norm_allpriors', stim_col='avtrapz',
    average_across_subjects=False
):
    assert side in [0, 1], "side value must be either 0 or 1"
    assert df[priors_col].abs().max() <= 1,\
        "prior must be normalized between -1 and 1"
    assert df[stim_col].abs().max() <= 1, "stimulus must be between -1 and 1"
    if pcomlabel is None:
        if not side:
            pcomlabel = r'$p(CoM_{R \rightarrow L})$'
        else:
            pcomlabel = r'$p(CoM_{L \rightarrow R})$'

    if n_points_marginal is None:
        n_points_marginal = nbins
    # ensure some filtering
    tmp = df.dropna(subset=['CoM_sugg', 'norm_allpriors', 'avtrapz'])
    tmp['tmp_com'] = False
    tmp.loc[(tmp.R_response == side) & (tmp.CoM_sugg), 'tmp_com'] = True

    com_heatmap_kws.update({
        'return_mat': True,
        'predefbins': [
            np.linspace(-1, 1, nbins+1), np.linspace(-1, 1, nbins+1)
        ]
    })
    if not average_across_subjects:
        mat, nmat = com_heatmap(
            tmp.norm_allpriors.values,
            tmp.avtrapz.values,
            tmp.tmp_com.values,
            **com_heatmap_kws
        )
        # fill nans with 0
        mat[np.isnan(mat)] = 0
        nmat[np.isnan(nmat)] = 0
        # change data to match vertical axis image standards (0,0) ->
        # in the top left
    else:
        com_mat_list, number_mat_list = [], []
        for subject in tmp.subjid.unique():
            cmat, cnmat = com_heatmap(
                tmp.loc[tmp.subjid == subject, 'norm_allpriors'].values,
                tmp.loc[tmp.subjid == subject, 'avtrapz'].values,
                tmp.loc[tmp.subjid == subject, 'tmp_com'].values,
                **com_heatmap_kws
            )
            cmat[np.isnan(cmat)] = 0
            cnmat[np.isnan(cnmat)] = 0
            com_mat_list += [cmat]
            number_mat_list += [cnmat]

        mat = np.stack(com_mat_list).mean(axis=0)
        nmat = np.stack(number_mat_list).mean(axis=0)

    mat = np.flipud(mat)
    nmat = np.flipud(nmat)
    return mat


# fig 2
def matrix_figure(df_data, humans, ax_tach, ax_pright, ax_mat):
    nbins = 7
    matrix_side_0 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=0)
    matrix_side_1 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=1)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = r'$p(CoM_{L \rightarrow R})$'
    ax_mat.set_title(pcomlabel_1)
    im_1 = ax_mat.imshow(matrix_side_1, vmin=0, vmax=vmax)
    divider = make_axes_locatable(ax_mat)
    cax = divider.append_axes('left', size='7%', pad=0.9)
    plt.colorbar(im_1, cax=cax)
    # R -> L
    pos = ax_mat.get_position()
    ax_mat.set_position([pos.x0, pos.y0*2/3, pos.width/2, pos.height*6/5])
    ax_mat_1 = plt.axes([pos.x0+pos.width/2, pos.y0*2/3,
                         pos.width/2, pos.height*6/5])
    pcomlabel_0 = r'$p(CoM_{L \rightarrow R})$'
    divider = make_axes_locatable(ax_mat_1)
    cax = divider.append_axes('right', size='7%', pad=0.9)
    cax.axis('off')
    ax_mat_1.set_title(pcomlabel_0)
    ax_mat_1.imshow(matrix_side_0, vmin=0, vmax=vmax)
    ax_mat_1.yaxis.set_ticks_position('none')
    for ax_i in [ax_pright, ax_mat, ax_mat_1]:
        ax_i.set_xlabel(r'$\longleftarrow$Prior$\longrightarrow$')
        ax_i.set_yticks(np.arange(nbins))
        ax_i.set_xticks(np.arange(nbins))
        ax_i.set_xticklabels(['left']+['']*(nbins-2)+['right'])
    for ax_i in [ax_pright, ax_mat]:
        ax_i.set_yticklabels(['right']+['']*(nbins-2)+['left'])
        ax_i.set_ylabel(r'$\longleftarrow$Average stimulus$\longrightarrow$',
                        labelpad=-17)
    ax_mat_1.set_yticklabels(['']*nbins)
    choice = df_data['R_response'].values
    coh = df_data['avtrapz'].values
    prior = df_data['norm_allpriors'].values
    mat_pright, _ = com_heatmap(prior, coh, choice, return_mat=True,
                                annotate=False)
    mat_pright = np.flipud(mat_pright)
    im_2 = ax_pright.imshow(mat_pright, cmap='rocket')
    divider = make_axes_locatable(ax_pright)
    cax1 = divider.append_axes('left', size='10%', pad=0.6)
    plt.colorbar(im_2, cax=cax1)
    cax1.yaxis.set_ticks_position('left')
    ax_pright.set_title('p(right)')
    if humans:
        num = 8
        rtbins = np.linspace(0, 300, num=num)
        tachometric(df_data, ax=ax_tach, fill_error=True, rtbins=rtbins)
    else:
        tachometric(df_data, ax=ax_tach, fill_error=True)
    ax_tach.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax_tach.set_xlabel('RT (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.4, 1.04)
    ax_tach.spines['right'].set_visible(False)
    ax_tach.spines['top'].set_visible(False)


def run_model(stim, zt, coh, gt, trial_index, num_tr=None):
    if num_tr is not None:
        num_tr = num_tr
    else:
        num_tr = int(len(zt))
    stim = stim[:, :int(num_tr)]
    zt = zt[:int(num_tr)]
    coh = coh[:int(num_tr)]
    gt = gt[:int(num_tr)]
    trial_index = trial_index[:int(num_tr)]
    data_augment_factor = 10
    MT_slope = 0.123
    MT_intercep = 254
    detect_CoMs_th = 5
    p_t_aff = 8
    p_t_eff = 8
    p_t_a = 12  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.2
    p_w_stim = 0.11
    p_e_noise = 0.02
    p_com_bound = 0.
    p_w_a_intercept = 0.05
    p_w_a_slope = -2.5e-05  # fixed
    p_a_noise = 0.042  # fixed
    p_1st_readout = 60
    p_2nd_readout = 150

    stim = edd2.data_augmentation(stim=stim, daf=data_augment_factor)
    stim_res = 50/data_augment_factor
    compute_trajectories = True
    all_trajs = True
    conf = [p_w_zt, p_w_stim, p_e_noise, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_noise, p_1st_readout,
            p_2nd_readout]
    jitters = len(conf)*[0]
    print('Number of trials: ' + str(stim.shape[1]))
    p_w_zt = conf[0]+jitters[0]*np.random.rand()
    p_w_stim = conf[1]+jitters[1]*np.random.rand()
    p_e_noise = conf[2]+jitters[2]*np.random.rand()
    p_com_bound = conf[3]+jitters[3]*np.random.rand()
    p_t_aff = int(round(conf[4]+jitters[4]*np.random.rand()))
    p_t_eff = int(round(conf[5]++jitters[5]*np.random.rand()))
    p_t_a = int(round(conf[6]++jitters[6]*np.random.rand()))
    p_w_a_intercept = conf[7]+jitters[7]*np.random.rand()
    p_w_a_slope = conf[8]+jitters[8]*np.random.rand()
    p_a_noise = conf[9]+jitters[9]*np.random.rand()
    p_1st_readout = conf[10]+jitters[10]*np.random.rand()
    p_2nd_readout = conf[11]+jitters[11]*np.random.rand()
    stim_temp =\
        np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                        stim.shape[1]))))
    # TODO: get in a dict
    E, A, com_model, first_ind, second_ind, resp_first, resp_fin,\
        pro_vs_re, matrix, total_traj, init_trajs, final_trajs,\
        frst_traj_motor_time, x_val_at_updt, xpos_plot, median_pcom,\
        rt_vals, rt_bins, tr_index =\
        edd2.trial_ev_vectorized(zt=zt, stim=stim_temp, coh=coh,
                                 trial_index=trial_index,
                                 MT_slope=MT_slope, MT_intercep=MT_intercep,
                                 p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                 p_e_noise=p_e_noise, p_com_bound=p_com_bound,
                                 p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                                 num_tr=num_tr, p_w_a_intercept=p_w_a_intercept,
                                 p_w_a_slope=p_w_a_slope,
                                 p_a_noise=p_a_noise,
                                 p_1st_readout=p_1st_readout,
                                 p_2nd_readout=p_2nd_readout,
                                 compute_trajectories=compute_trajectories,
                                 stim_res=stim_res, all_trajs=all_trajs)
    hit_model = resp_fin == gt
    reaction_time = (first_ind[tr_index]-int(300/stim_res) + p_t_eff)*stim_res
    detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
    return hit_model, reaction_time, detected_com, resp_fin, com_model, pro_vs_re


def fig_3(user_id, existing_data_path, ax_tach, ax_pright, ax_mat, humans=False):
    subj = ['general_traj']
    steps = [None]
    nm = '300'
    if user_id == 'Alex':
        folder = 'C:\\Users\\Alexandre\\Desktop\\CRM\\Human\\80_20'
        df_data = analyses.traj_analysis(main_folder=folder+'\\'+nm+'ms\\',
                                         subjects=subj, steps=steps, name=nm,
                                         existing_data_path=existing_data_path)
    if user_id == 'Manuel':
        folder = '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'
        df_data = analyses.traj_analysis(main_folder=folder+'/'+nm+'ms/',
                                         subjects=subj, steps=steps, name=nm,
                                         existing_data_path=existing_data_path)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    matrix_figure(df_data=df_data, ax_tach=ax_tach, ax_pright=ax_pright,
                  ax_mat=ax_mat, humans=humans)


# ---MAIN
if __name__ == '__main__':
    plt.close('all')
    subject = 'LE43'
    all_rats = True
    num_tr = int(15e4)
    return_df = False
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
                                             return_df=return_df)
    else:
        df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + subject,
                                      return_df=True, sv_folder=SV_FOLDER,
                                      after_correct=True, silent=True,
                                      all_trials=True)
    if return_df:
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
    matrix_figure(df_data, ax_tach=ax[1], ax_pright=ax[2],
                  ax_mat=ax[3], humans=False)

    # FIG 2
    existing_model_data = True
    if not existing_model_data:
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            pro_vs_re =\
            run_model(stim=stim, zt=zt, coh=coh, gt=gt, trial_index=trial_index,
                      num_tr=None)
        index = reaction_time >= 0
        df_data = pd.DataFrame({'avtrapz': coh[index],
                                'CoM_sugg': com_model_detected[index],
                                'norm_allpriors': zt[index]/max(abs(zt[index])),
                                'R_response': (resp_fin[index] + 1)/2,
                                'sound_len': reaction_time[index],
                                'hithistory': hit_model[index]})
    else:
        df_data = pd.read_csv(DATA_FOLDER + 'df_fig_1.csv')
    f, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    humans = False
    ax[0].axis('off')
    matrix_figure(df_data=df_data, humans=humans, ax_tach=ax[1],
                  ax_pright=ax[2], ax_mat=ax[3])

    fgsz = (8, 8)
    inset_sz = 0.1
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=fgsz)
    ax = ax.flatten()
    ax_cohs = np.array([ax[0], ax[2]])
    ax_inset = fp.add_inset(ax=ax_cohs[0], inset_sz=inset_sz, fgsz=fgsz)
    ax_cohs = np.insert(ax_cohs, 0, ax_inset)
    ax_inset = fp.add_inset(ax=ax_cohs[2], inset_sz=inset_sz, fgsz=fgsz,
                            marginy=0.15)
    ax_cohs = np.insert(ax_cohs, 2, ax_inset)
    for a in ax:
        fp.rm_top_right_lines(a)
    df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + subject,
                                  return_df=True, sv_folder=SV_FOLDER,
                                  after_correct=True, silent=True,
                                  all_trials=True)
    fp.trajs_cond_on_coh(df=df, ax=ax_cohs)
    # splits
    ax_split = np.array([ax[1], ax[3]])
    fp.trajs_splitting(df, ax=ax_split[0])
    # XXX: do this panel for all rats?
    fp.trajs_splitting_point(df=df, ax=ax_split[1])
    # fig3.trajs_cond_on_prior(df, savpath=SV_FOLDER)
    # FIG 3:
    f, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    ax[0].axis('off')
    fig_3(user_id='Alex',
          existing_data_path=DATA_FOLDER+'df_regressors.csv',
          ax_tach=ax[1], ax_pright=ax[2],
          ax_mat=ax[3], humans=True)
