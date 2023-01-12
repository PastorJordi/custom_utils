# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:24:12 2022
@author: Alex Garcia-Duran
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem
import sys
from scipy.optimize import curve_fit
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr, ttest_ind
from matplotlib.lines import Line2D
# from scipy import interpolate
sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
from utilsJ.Models import simul
from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve,\
    trajectory_thr, com_heatmap
from utilsJ.Models import analyses_humans as ah
import fig1, fig3, fig2
import matplotlib
import matplotlib.pylab as pl

matplotlib.rcParams['font.size'] = 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3
pc_name = 'idibaps_Jordi'  # 'alex'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'
    SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
    DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
elif pc_name == 'idibaps':
    DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
    SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
        'ChangesOfMind/figures/from_python/'  # Manuel
    RAT_COM_IMG = '/home/molano/Dropbox/project_Barna/' +\
        'ChangesOfMind/figures/Figure_3/001965.png'
elif pc_name == 'idibaps_Jordi':
    SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
    DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
    RAT_COM_IMG = '/home/jordi/Documents/changes_of_mind/demo/materials/' +\
        'craft_vid/CoM/a/001965.png'
# SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
# DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM


# RAT_COM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/001965.png'

FRAME_RATE = 14
BINS_RT = np.linspace(1, 301, 11)
xpos_RT = int(np.diff(BINS_RT)[0])


def plot_coms(df, ax, human=False):
    coms = df.CoM_sugg.values
    decision = df.R_response.values
    if human:
        ran_max = 600
        max_val = 600
    if not human:
        ran_max = 400
        max_val = 77
    for tr in reversed(range(ran_max)):  # len(df_rat)):
        if tr > (ran_max/2) and not coms[tr] and decision[tr] == 1:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = np.arange(len(traj))*FRAME_RATE
                ax.plot(time, traj, color='tab:cyan', lw=.5)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.1:
                    ax.plot(time*1e3, traj, color='tab:cyan', lw=.5)
        elif tr < (ran_max/2-1) and coms[tr] and decision[tr] == 0:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = np.arange(len(traj))*FRAME_RATE
                ax.plot(time, traj, color='tab:olive', lw=2)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.2:
                    ax.plot(time*1e3, traj, color='tab:olive', lw=2)
    rm_top_right_lines(ax)
    if human:
        var = 'x'
        sp = 'Subject'
    if not human:
        var = 'y'
        sp = 'Rats'
    ax.set_ylabel('{} position {}-axis (pixels)'.format(sp, var))
    ax.set_xlabel('Time from movement onset (ms)')
    ax.axhline(y=max_val, linestyle='--', color='Green', lw=1)
    ax.axhline(y=-max_val, linestyle='--', color='Purple', lw=1)
    ax.axhline(y=0, linestyle='--', color='k', lw=0.5)


def tracking_image(ax):
    rat = plt.imread(RAT_COM_IMG)
    ax.set_facecolor('white')
    ax.imshow(np.flipud(rat[100:-100, 350:-50, :]))
    ax.axis('off')


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


def matrix_figure(df_data, humans, ax_tach, ax_pright, ax_mat):
    # plot tachometrics
    if humans:
        num = 8
        rtbins = np.linspace(0, 300, num=num)
        tachometric(df_data, ax=ax_tach, fill_error=True, rtbins=rtbins,
                    cmap='gist_yarg')
    else:
        tachometric(df_data, ax=ax_tach, fill_error=True, cmap='gist_yarg')
    ax_tach.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax_tach.set_xlabel('Reaction Time (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.3, 1.04)
    ax_tach.spines['right'].set_visible(False)
    ax_tach.spines['top'].set_visible(False)
    ax_tach.legend()
    # plot Pcoms matrices
    nbins = 7
    matrix_side_0 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=0)
    matrix_side_1 = com_heatmap_paper_marginal_pcom_side(df=df_data, side=1)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[0].set_title(pcomlabel_1)
    im = ax_mat[0].imshow(matrix_side_1, vmin=0, vmax=vmax)
    plt.sca(ax_mat[0])
    plt.colorbar(im, fraction=0.04)
    # pos = ax_mat.get_position()
    # ax_mat.set_position([pos.x0, pos.y0*2/3, pos.width, pos.height])
    # ax_mat_1 = plt.axes([pos.x0+pos.width+0.05, pos.y0*2/3,
    #                      pos.width, pos.height])
    pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[1].set_title(pcomlabel_0)
    im = ax_mat[1].imshow(matrix_side_0, vmin=0, vmax=vmax)
    ax_mat[1].yaxis.set_ticks_position('none')
    plt.sca(ax_mat[1])
    plt.colorbar(im, fraction=0.04)
    # pright matrix
    choice = df_data['R_response'].values
    coh = df_data['coh2'].values
    prior = df_data['norm_allpriors'].values
    mat_pright, _ = com_heatmap(prior, coh, choice, return_mat=True,
                                annotate=False)
    mat_pright = np.flipud(mat_pright)
    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    plt.sca(ax_pright)
    plt.colorbar(im_2, fraction=0.04)
    ax_pright.set_title('Proportion of rightward responses')

    # R -> L
    for ax_i in [ax_pright, ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior Evidence')
        # ax_i.set_yticks(np.arange(nbins))
        # ax_i.set_xticks(np.arange(nbins))
        # ax_i.set_xticklabels(['left']+['']*(nbins-2)+['right'])
        ax_i.set_yticklabels(['']*nbins)
        ax_i.set_xticklabels(['']*nbins)
    for ax_i in [ax_pright, ax_mat[0]]:
        # ax_i.set_yticklabels(['right']+['']*(nbins-2)+['left'])
        ax_i.set_ylabel('Stimulus Evidence')  # , labelpad=-17)


def rm_top_right_lines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def pcom_model_vs_data(detected_com, com, sound_len, reaction_time):
    fig, ax = plt.subplots(1)
    rm_top_right_lines(ax)
    df = pd.DataFrame({'com_model': detected_com, 'CoM_sugg': com,
                       'sound_len': sound_len, 'reaction_time': reaction_time})
    binned_curve(df, 'CoM_sugg', 'sound_len', bins=BINS_RT, xpos=xpos_RT, ax=ax,
                 errorbar_kw={'label': 'CoM data'})
    binned_curve(df, 'com_model', 'reaction_time', bins=BINS_RT, xpos=xpos_RT,
                 ax=ax, errorbar_kw={'label': 'Detected CoM model'})


def MT_model_vs_data(MT_model, MT_data, bins_MT=np.linspace(50, 600, num=26,
                                                            dtype=int)):
    fig, ax = plt.subplots(1)
    rm_top_right_lines(ax)
    ax.set_title('MT distributions')
    hist_MT_model, _ = np.histogram(MT_model, bins=bins_MT)
    ax.plot(bins_MT[:-1]+(bins_MT[1]-bins_MT[0])/2, hist_MT_model,
            label='model MT dist')
    hist_MT_data, _ = np.histogram(MT_data, bins=bins_MT)
    ax.scatter(bins_MT[:-1]+(bins_MT[1]-bins_MT[0])/2, hist_MT_data,
               label='data MT dist')
    ax.set_xlabel('MT (ms)')


def plot_RT_distributions(sound_len, RT_model, pro_vs_re):
    fig, ax = plt.subplots(1)
    rm_top_right_lines(ax)
    bins = np.linspace(-300, 400, 40)
    ax.hist(sound_len, bins=bins, density=True, ec='k', label='Data')
    hist_pro, _ = np.histogram(RT_model[0][pro_vs_re == 1], bins)
    hist_re, _ = np.histogram(RT_model[0][pro_vs_re == 0], bins)
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2,
            hist_pro/(np.sum(hist_pro)*np.diff(bins)), label='Proactive only')
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2,
            hist_re/(np.sum(hist_re)*np.diff(bins)), label='Reactive only')
    hist_total, _ = np.histogram(RT_model[0], bins)
    ax.plot(bins[:-1]+(bins[1]-bins[0])/2,
            hist_total/(np.sum(hist_total)*np.diff(bins)), label='Model')
    ax.legend()


def tachometrics_data_and_model(coh, hit_history_model, hit_history_data,
                                RT_data, RT_model):
    fig, ax = plt.subplots(ncols=2)
    rm_top_right_lines(ax[0])
    rm_top_right_lines(ax[1])
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit_history_data,
                                 'sound_len': RT_data})
    tachometric(df_plot_data, ax=ax[0])
    ax[0].set_xlabel('RT (ms)')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Data')
    df_plot_model = pd.DataFrame({'avtrapz': coh, 'hithistory': hit_history_model,
                                 'sound_len': RT_model})
    tachometric(df_plot_model, ax=ax[1])
    ax[1].set_xlabel('RT (ms)')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Model')


def add_inset(ax, inset_sz=0.2, fgsz=(4, 8), marginx=0.01, marginy=0.05):
    ratio = fgsz[0]/fgsz[1]
    pos = ax.get_position()
    ax_inset = plt.axes([pos.x1-inset_sz-marginx, pos.y0+marginy, inset_sz,
                         inset_sz*ratio])
    rm_top_right_lines(ax_inset)
    return ax_inset


def trajs_cond_on_coh(df, ax, average=False, acceleration=('traj_d2', 1),
                      accel=False):
    """median position and velocity in silent trials splitting by prior"""
    # TODO: adapt for mean + sem
    df_43 = df.loc[df.subjid == 'LE43']
    ax_zt = ax[0]
    ax_cohs = ax[1]
    ax_ti = ax[2]
    trajs_cond_on_coh_computation(df=df_43, ax=ax_zt, condition='prior_x_coh',
                                  prior_limit=1, cmap='copper')
    trajs_cond_on_coh_computation(df=df_43, ax=ax_cohs, condition='choice_x_coh',
                                  cmap='coolwarm')
    trajs_cond_on_coh_computation(df=df_43, ax=ax_ti, condition='origidx',
                                  cmap='jet', prior_limit=1)


def trajs_cond_on_coh_computation(df, ax, condition='choice_x_coh', cmap='viridis',
                                  prior_limit=0.25, rt_lim=25, 
                                  after_correct_only=True,
                                  trajectory="trajectory_y",
                                  velocity=("traj_d1", 1),
                                  acceleration=('traj_d2', 1), accel=False):
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    df['norm_allpriors'] = df.allpriors / np.max(np.abs(df.allpriors))
    df['prior_x_coh'] = (df.R_response*2-1) * df.norm_allpriors
    df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
    if condition == 'choice_x_coh':
        bins = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        xlab = 'ev. towards response'
        bintype = 'categorical'
    if condition == 'prior_x_coh':
        bins = np.array([-1, -0.4, -0.05, 0.05, 0.4, 1])
        xlab = 'prior towards response'
        bintype = 'edges'
    if condition == 'origidx':
        bins = np.linspace(0, 1e3, num=6)
        xlab = 'trial index'
        bintype = 'edges'
    if after_correct_only:
        ac_cond = df.aftererror == False
    else:
        ac_cond = (df.aftererror*1) >= 0
    # position
    indx_trajs = (df.norm_allpriors.abs() <= prior_limit) &\
        ac_cond & (df.special_trial == 0) &\
        (df.sound_len < rt_lim)
    xpoints, ypoints, _, mat, dic, mt_time, mt_time_err =\
        trajectory_thr(df.loc[indx_trajs], condition, bins,
                       collapse_sides=True, thr=30, ax=ax[0], ax_traj=ax[1],
                       return_trash=True, error_kwargs=dict(marker='o'),
                       cmap=cmap, bintype=bintype,
                       trajectory=trajectory, plotmt=True)
    if condition == 'choice_x_coh':
        ax[1].legend(labels=['-1', '-0.5', '-0.25', '0', '0.25', '0.5', '1'],
                     title='Coherence')
    if condition == 'prior_x_coh':
        ax[1].legend(labels=['inc. high', 'inc. low', 'zero', 'con. low',
                             'con. high'], title='Prior')
    if condition == 'origidx':
        ax[1].legend(labels=['100', '300', '500', '700', '900'],
                     title='Trial index')
    ax[1].set_xlim([-50, 500])
    ax[1].set_xlabel('time from movement onset (MT, ms)')
    for i in [0, 30]:
        ax[1].axhline(i, ls=':', c='gray')
    ax[1].set_ylabel('y coord. (px)')
    ax[0].set_xlabel(xlab)
    ax[0].set_ylabel('Motor time')
    ax[1].set_ylim([-10, 80])
    # ax2 = ax[0].twinx()
    ax[0].plot(xpoints, mt_time, color='k', ls=':')
    # ax2.set_label('Motor time')
    # velocities
    threshold = .2
    xpoints, ypoints, _, mat, dic, _, _ = trajectory_thr(
        df.loc[indx_trajs], condition, bins, collapse_sides=True,
        thr=threshold, ax=ax[2], ax_traj=ax[3], return_trash=True,
        error_kwargs=dict(marker='o'), cmap=cmap,
        bintype=bintype, trajectory=velocity, plotmt=False)
    # ax[3].legend(labels=['-1', '-0.5', '-0.25', '0', '0.25', '0.5', '1'],
    #              title='Coherence', loc='upper left')
    ax[3].set_xlim([-200, 500])
    ax[3].set_xlabel('time from movement onset (MT, ms)')
    ax[3].set_ylim([-0.05, 0.5])
    for i in [0, threshold]:
        ax[3].axhline(i, ls=':', c='gray')
    ax[3].set_ylabel('y coord velocity (px/ms)')
    ax[2].set_xlabel(xlab)
    ax[2].set_ylabel(f'time to threshold ({threshold} px/ms)')
    ax[2].plot(xpoints, ypoints, color='k', ls=':')
    plt.show()
    if accel:
        # acceleration
        threshold = .0015
        xpoints, ypoints, _, mat, dic, _, _ = trajectory_thr(
            df.loc[indx_trajs], condition, bins, collapse_sides=True,
            thr=threshold, ax=ax[4], ax_traj=ax[5], return_trash=True,
            error_kwargs=dict(marker='o'), cmap='viridis',
            bintype=bintype, trajectory=acceleration)
        # ax[3].legend(labels=['-1', '-0.5', '-0.25', '0', '0.25', '0.5', '1'],
        #              title='Coherence', loc='upper left')
        ax[5].set_xlim([-50, 500])
        ax[5].set_xlabel('time from movement onset (MT, ms)')
        ax[5].set_ylim([-0.003, 0.0035])
        for i in [0, threshold]:
            ax[5].axhline(i, ls=':', c='gray')
        ax[5].set_ylabel('y coord accelration (px/ms)')
        ax[4].set_xlabel('ev. towards response')
        ax[4].set_ylabel(f'time to threshold ({threshold} px/ms)')
        ax[4].plot(xpoints, ypoints, color='k', ls=':')
        plt.show()


def get_split_ind_corr(mat, evl, pval=0.001, max_MT=400, startfrom=700):
    for i in range(max_MT):
        pop_a = mat[:, startfrom + i]
        nan_idx = ~np.isnan(pop_a)
        pop_evidence = evl[nan_idx]
        pop_a = pop_a[nan_idx]
        _, p2 = pearsonr(pop_a, pop_evidence)
        # plist.append(p2)
        if p2 < pval:
            return i
    return np.nan


def trajs_splitting(df, ax, rtbin=0, rtbins=np.linspace(0, 150, 2),
                    subject='LE43', startfrom=700):
    """
    Plot moment at which median trajectories for coh=0 and coh=1 split, for RTs
    between 0 and 90.


    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    rtbin : TYPE, optional
        DESCRIPTION. The default is 0.
    rtbins : TYPE, optional
        DESCRIPTION. The default is np.linspace(0, 90, 2).

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    subject = subject
    lbl = 'RTs: ['+str(rtbins[rtbin])+'-'+str(rtbins[rtbin+1])+']'
    colors = pl.cm.gist_yarg(np.linspace(0.4, 1, 3))
    evs = [0.25, 0.5, 1]
    mat = np.empty((1701,))
    evl = np.empty(())
    appb = True
    for iev, ev in enumerate(evs):
        indx = df.subjid == subject
        if np.sum(indx) > 0:
            _, matatmp, matb =\
                simul.when_did_split_dat(df=df[indx], side=0, collapse_sides=True,
                                         ax=ax, rtbin=rtbin, rtbins=rtbins,
                                         color=colors[iev], label=lbl, coh1=ev)
        if appb:
            mat = matb
            evl = np.repeat(0, matb.shape[0])
            appb = False
        mat = np.concatenate((mat, matatmp))
        evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
    ind = get_split_ind_corr(mat, evl, pval=0.001, max_MT=400, startfrom=700)
    ax.axvline(ind, linestyle='--', alpha=0.4, color='red')
    ax.set_xlim(-10, 100)
    ax.set_ylim(-2, 5)
    ax.set_xlabel('time from movement onset (ms)')
    ax.set_ylabel('y dimension (px)')
    ax.set_title(subject)
    plt.show()


def trajs_splitting_point(df, ax, collapse_sides=True, threshold=300,
                          sim=False,
                          rtbins=np.linspace(0, 150, 16), connect_points=True,
                          draw_line=((0, 90), (90, 0)),
                          trajectory="trajectory_y"):

    # split time/subject by coherence
    # threshold= bigger than that are turned to nan so it doesnt break figure range
    # this wont work if when_did_split_dat returns Nones instead of NaNs
    # plot will not work fine with uneven bins
    if sim:
        splitfun = simul.when_did_split_simul
    if not sim:
        splitfun = simul.when_did_split_dat
    out_data = []
    for subject in df.subjid.unique():
        for i in range(rtbins.size-1):
            if collapse_sides:
                evs = [0.25, 0.5, 1]
                mat = np.empty((1701,))
                evl = np.empty(())
                appb = True
                for iev, ev in enumerate(evs):
                    _, matatmp, matb =\
                        splitfun(df=df.loc[(df.special_trial == 0)
                                           & (df.subjid == subject)],
                                 side=0, collapse_sides=True,
                                 rtbin=i, rtbins=rtbins, coh1=ev,
                                 trajectory=trajectory)
                    if appb:
                        mat = matb
                        evl = np.repeat(0, matb.shape[0])
                        appb = False
                    mat = np.concatenate((mat, matatmp))
                    evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
                current_split_index =\
                    get_split_ind_corr(mat, evl, pval=0.001, max_MT=400,
                                       startfrom=700)
                out_data += [current_split_index]
            else:
                for j in [0, 1]:  # side values
                    current_split_index, _, _ = splitfun(
                        df.loc[df.subjid == subject],
                        j,  # side has no effect because this is collapsing_sides
                        rtbin=i, rtbins=rtbins)
                    out_data += [current_split_index]

    # reshape out data so it makes sense. '0th dim=rtbin, 1st dim= n datapoints
    # ideally, make it NaN resilient
    out_data = np.array(out_data).reshape(
        df.subjid.unique().size, rtbins.size-1, -1)
    # set axes: rtbins, subject, sides
    out_data = np.swapaxes(out_data, 0, 1)

    # change the type so we can have NaNs
    out_data = out_data.astype(float)

    out_data[out_data > threshold] = np.nan

    binsize = rtbins[1]-rtbins[0]

    scatter_kws = {'color': (.6, .6, .6, .3), 'edgecolor': (.6, .6, .6, 1)}
    if collapse_sides:
        nrepeats = df.subjid.unique().size  # n subjects
    else:
        nrepeats = df.subjid.unique().size * 2  # two responses per subject
    # because we might want to plot each subject connecting lines, lets iterate
    # draw  datapoints
    if not connect_points:
        ax.scatter(  # add some offset/shift on x axis based on binsize
            binsize/2 + binsize * (np.repeat(
                np.arange(rtbins.size-1), nrepeats
            ) + np.random.normal(loc=0, scale=0.2, size=out_data.size)),  # jitter
            out_data.flatten(),
            **scatter_kws,
        )
    else:
        for i in range(df.subjid.unique().size):
            for j in range(out_data.shape[2]):
                ax.plot(
                    binsize/2 + binsize * np.arange(rtbins.size-1),
                    out_data[:, i, j],
                    marker='o', mfc=(.6, .6, .6, .3), mec=(.6, .6, .6, 1),
                    mew=1, color=(.6, .6, .6, .3)
                )

    error_kws = dict(ecolor='k', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='k', marker='o', label='mean & SEM')
    ax.errorbar(
        binsize/2 + binsize * np.arange(rtbins.size-1),
        # we do the mean across rtbin axis
        np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1),
        # other axes we dont care
        yerr=sem(out_data.reshape(rtbins.size-1, -1),
                 axis=1, nan_policy='omit'),
        **error_kws
    )
    if draw_line is not None:
        ax.plot(*draw_line, c='r', ls='--', zorder=0, label='slope -1')

    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('time to split (ms)')
    ax.legend()
    plt.show()
# 3d histogram-like*?


def fig3_b(trajectories, motor_time, decision, com, coh, sound_len, traj_stamps,
           fix_onset, fixation_us=300000):
    'mean velocity and position for all trials'
    # interpolatespace = np.linspace(-700000, 1000000, 1701)
    ind_nocom = (~com.astype(bool))
    # *(motor_time < 400)*(np.abs(coh) == 1) *\
    #     (motor_time > 300)
    mean_position_array = np.empty((len(motor_time[ind_nocom]),
                                    max(motor_time)))
    mean_position_array[:] = np.nan
    mean_velocity_array = np.empty((len(motor_time[ind_nocom]), max(motor_time)))
    mean_velocity_array[:] = np.nan
    for i, traj in enumerate(trajectories[ind_nocom]):
        xvec = traj_stamps[i] - np.datetime64(fix_onset[i])
        xvec = (xvec -
                np.timedelta64(int(fixation_us + (sound_len[i]*1e3)),
                               "us")).astype(float)
        # yvec = traj
        # f = interpolate.interp1d(xvec, yvec, bounds_error=False)
        # out = f(interpolatespace)
        vel = np.diff(traj)
        mean_position_array[i, :len(traj)] = -traj*decision[i]
        mean_velocity_array[i, :len(vel)] = -vel*decision[i]
    mean_pos = np.nanmean(mean_position_array, axis=0)
    mean_vel = np.nanmean(mean_velocity_array, axis=0)
    std_pos = np.nanstd(mean_position_array, axis=0)
    fig, ax = plt.subplots(nrows=2)
    ax = ax.flatten()
    ax[0].plot(mean_pos)
    ax[0].fill_between(np.arange(len(mean_pos)), mean_pos + std_pos,
                       mean_pos - std_pos, alpha=0.4)
    ax[1].plot(mean_vel)


def tachometric_data(coh, hit, sound_len, ax, label='Data'):
    rm_top_right_lines(ax)
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len})
    tachometric(df_plot_data, ax=ax, fill_error=True, cmap='gist_yarg')
    ax.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title(label)
    ax.set_ylim(0.4, 1.04)
    # ax.legend([1, 0.5, 0.25, 0])
    return ax.get_position()


def reaction_time_histogram(sound_len, label, ax, bins=np.linspace(1, 301, 61),
                            pro_vs_re=None):
    rm_top_right_lines(ax)
    if label == 'Data':
        color = 'k'
    if label == 'Model':
        color = 'red'
        color_pro = 'coral'
        color_re = 'maroon'
        sound_len_pro = sound_len[pro_vs_re == 0]
        sound_len_re = sound_len[pro_vs_re == 1]
        ax.hist(sound_len_pro, bins=bins, alpha=0.3, density=False, linewidth=0.,
                histtype='stepfilled', label=label + '-pro', color=color_pro)
        ax.hist(sound_len_re, bins=bins, alpha=0.3, density=False, linewidth=0.,
                histtype='stepfilled', label=label + '-reac', color=color_re)
    ax.hist(sound_len, bins=bins, alpha=0.3, density=False, linewidth=0.,
            histtype='stepfilled', label=label, color=color)
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel('Frequency')
    # ax.set_xlim(0, max(bins))


def pdf_cohs(sound_len, ax, coh, bins=np.linspace(1, 301, 61), yaxis=True):
    ev_vals = np.unique(np.abs(coh))
    colormap = pl.cm.gist_gray_r(np.linspace(0.2, 1, len(ev_vals)))
    for i_coh, ev in enumerate(ev_vals):
        index = np.abs(coh) == ev
        counts_coh, bins_coh = np.histogram(sound_len[index], bins=bins)
        norm_counts = counts_coh/sum(counts_coh)
        xvals = bins_coh[:-1]+(bins_coh[1]-bins_coh[0])/2
        ax.plot(xvals, norm_counts, color=colormap[i_coh])
    ax.set_xlabel('Reaction time (ms)')
    if yaxis:
        ax.set_ylabel('Density')


def express_performance(hit, coh, sound_len, pos_tach_ax, ax, label,
                        inset=False):
    " all rats..? "
    pos = pos_tach_ax
    rm_top_right_lines(ax)
    ev_vals = np.unique(np.abs(coh))
    accuracy = []
    error = []
    for ev in ev_vals:
        index = (coh == ev)*(sound_len < 90)
        accuracy.append(np.mean(hit[index]))
        error.append(np.sqrt(np.std(hit[index])/np.sum(index)))
    if inset:
        ax.set_position([pos.x0+2*pos.width/3, pos.y0+pos.height/9,
                         pos.width/3, pos.height/6])
    if label == 'Data':
        color = 'k'
    if label == 'Model':
        color = 'red'
    ax.errorbar(x=ev_vals, y=accuracy, yerr=error, color=color, fmt='-o',
                capsize=3, capthick=2, elinewidth=2, label=label)
    ax.set_xlabel('Coherence')
    ax.set_ylabel('Performance')
    ax.set_title('Express performance')
    ax.set_ylim(0.5, 1)
    ax.legend()


def cdfs(coh, sound_len, ax, f5, title='', linestyle='solid', label_title='',
         model=False):
    colors = ['k', 'darkred', 'darkorange', 'gold']
    index_1 = (sound_len <= 300)*(sound_len > 0)
    sound_len = sound_len[index_1]
    coh = coh[index_1]
    ev_vals = np.unique(np.abs(coh))
    for i, ev in enumerate(ev_vals):
        if f5:
            if ev == 0 or ev == 1:
                index = ev == np.abs(coh)
                hist_data, bins = np.histogram(sound_len[index], bins=200)
                cdf_vals = np.cumsum(hist_data)/np.sum(hist_data)
                xvals = bins[:-1]+(bins[1]-bins[0])/2
                if model:
                    x_interp = np.arange(0, 300, 10)
                    cdf_vals_interp = np.interp(x_interp, xvals, cdf_vals)
                    ax.plot(x_interp, cdf_vals_interp,
                            label=str(ev) + ' ' + label_title,
                            color=colors[i], linewidth=2, linestyle=linestyle)
                else:
                    ax.plot(xvals, cdf_vals,
                            label=str(ev) + ' ' + label_title,
                            color=colors[i], linewidth=2, linestyle=linestyle)
        else:
            index = ev == np.abs(coh)
            hist_data, bins = np.histogram(sound_len[index], bins=200)
            ax.plot(bins[:-1]+(bins[1]-bins[0])/2,
                    np.cumsum(hist_data)/np.sum(hist_data),
                    label=str(ev) + ' ' + label_title,
                    color=colors[i], linewidth=2, linestyle=linestyle)
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('CDF')
    ax.set_xlim(-1, 152)
    ax.legend(title='Coherence')
    ax.set_title(str(title))


def fig_rats_behav_1(df_data):
    nbins = 7
    f, ax = plt.subplots(nrows=3, ncols=4, figsize=(6, 5))  # figsize=(4, 3))
    ax = ax.flatten()
    for i in [0, 1, 2, 4, 5, 6]:
        ax[i].axis('off')
    # P_right
    # TODO: check ticks for matrix
    ax_pright = ax[3]
    choice = df_data['R_response'].values
    coh = df_data['coh2'].values
    prior = df_data['norm_allpriors'].values
    mat_pright, _ = com_heatmap(prior, coh, choice, return_mat=True,
                                annotate=False)
    mat_pright = np.flipud(mat_pright)
    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    plt.sca(ax_pright)
    plt.colorbar(im_2, fraction=0.04)
    ax_pright.set_title('Proportion of rightward responses')

    ax_pright.set_xlabel('Prior Evidence')
    ax_pright.set_yticklabels(['']*nbins)
    ax_pright.set_xticklabels(['']*nbins)
    ax_pright.set_ylabel('Stimulus Evidence')  # , labelpad=-17)

    # tachometrics
    # TODO: check legend
    ax_tach = ax[7]
    tachometric(df_data, ax=ax_tach, fill_error=True, cmap='gist_yarg')
    ax_tach.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax_tach.set_xlabel('Reaction Time (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.3, 1.04)
    rm_top_right_lines(ax_tach)
    ax_tach.legend()

    # TODO: RTs distros conditioned on stim evidence. CHECK
    ax_rts = ax[8]
    sound_len = np.array(df.sound_len)
    pdf_cohs(sound_len=sound_len, ax=ax_rts, coh=coh, yaxis=True)

    # track screenshot
    rat = plt.imread(RAT_COM_IMG)
    ax_scrnsht = ax[9]
    ax_scrnsht.imshow(np.flipud(rat))
    ax_scrnsht.set_xticklabels([])
    ax_scrnsht.set_yticklabels([])
    ax_scrnsht.set_xticks([])
    ax_scrnsht.set_yticks([])
    ax_scrnsht.set_xlabel('x dimension (pixels)')  # , fontsize=14)
    ax_scrnsht.set_ylabel('y dimension (pixels)')  # , fontsize=14)
    ax_scrnsht.set_xlim(435, 585)
    ax_scrnsht.set_ylim(100, 360)

    # raw trajectories
    ax_rawtr = ax[10]
    ax_ydim = ax[11]
    ran_max = 100
    for tr in range(ran_max):  # len(df_rat)):
        if tr > (ran_max/2):
            trial = df.iloc[tr]
            traj_x = trial['trajectory_x']
            traj_y = trial['trajectory_y']
            ax_rawtr.plot(traj_x, traj_y, color='k', lw=.5)
            time = np.arange(len(traj_x))*FRAME_RATE
            ax_ydim.plot(time, traj_y, color='k', lw=.5)

    ax_rawtr.set_xticklabels([])
    ax_rawtr.set_yticklabels([])
    ax_rawtr.set_xticks([])
    ax_rawtr.set_yticks([])
    ax_rawtr.set_xlabel('x dimension (pixels)')  # , fontsize=14)
    # ax_rawtr.set_ylabel('y dimension (pixels)')  # , fontsize=14)
    ax_ydim.set_xlabel('time (ms)')  # , fontsize=14)
    # ax_ydim.set_ylabel('y dimension (pixels)')  # , fontsize=14)

    f.savefig(SV_FOLDER+'fig1.svg', dpi=400, bbox_inches='tight')
    f.savefig(SV_FOLDER+'fig1.png', dpi=400, bbox_inches='tight')


def mt_weights(df, ax, plot=False, means_errs=True):
    w_coh = []
    w_t_i = []
    w_zt = []
    if ax is None and plot:
        fig, ax = plt.subplots(1)
    for subject in df.subjid.unique():
        df_1 = df.loc[df.subjid == subject]
        resp_len = np.array(df_1.resp_len)
        decision = np.array(df_1.R_response)*2 - 1
        coh = np.array(df_1.coh2)
        trial_index = np.array(df_1.origidx)
        com = df_1.CoM_sugg.values
        zt = np.nansum(df_1[["dW_lat", "dW_trans"]].values, axis=1)
        params = mt_linear_reg(mt=resp_len, coh=coh*decision/max(np.abs(coh)),
                               trial_index=trial_index/max(trial_index),
                               prior=zt*decision/max(np.abs(zt)), plot=False,
                               com=com)
        w_coh.append(params[1])
        w_t_i.append(params[2])
        w_zt.append(params[3])
    mean_1 = np.nanmean(w_coh)
    mean_2 = np.nanmean(w_t_i)
    mean_3 = np.nanmean(w_zt)
    std_1 = np.nanstd(w_coh)/np.sqrt(len(w_coh))
    std_2 = np.nanstd(w_t_i)/np.sqrt(len(w_t_i))
    std_3 = np.nanstd(w_zt)/np.sqrt(len(w_zt))
    errors = [std_1, std_2, std_3]
    means = [mean_1, mean_2, mean_3]
    if plot:
        if means_errs:
            # fig, ax = plt.subplots(figsize=(3, 2))
            # TODO: not the most informative name for a function
            plot_bars(means=means, errors=errors, ax=ax)
            rm_top_right_lines(ax=ax)
        else:
            # fig, ax = plt.subplots(figsize=(3, 2))
            plot_violins(w_coh=w_coh, w_t_i=w_t_i, w_zt=w_zt, ax=ax)
    if means_errs:
        return means, errors
    else:
        return w_coh, w_t_i, w_zt


def plot_bars(means, errors, ax, f5=False, means_model=None, errors_model=None,
              width=0.35):
    labels = ['Stimulus Congruency', 'Trial index', 'Prior Congruency']
    if not f5:
        ax.bar(x=labels, height=means, yerr=errors, capsize=3, color='k',
               ecolor='blue')
        ax.set_ylabel('Weight (a.u.)')
    if f5:
        x = np.arange(len(labels))
        ax.bar(x=x-width/2, height=means, yerr=errors, width=width,
               capsize=3, color='k', label='Data', ecolor='blue')
        ax.bar(x=x+width/2, height=means_model, yerr=errors_model, width=width,
               capsize=3, color='red', label='Model')
        ax.set_ylabel('Weight (a.u.)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()


def plot_violins(w_coh, w_t_i, w_zt, ax):
    labels = ['Stimulus Congruency', 'Trial index', 'Prior Congruency']
    arr_weights = np.concatenate((w_coh, w_t_i, w_zt))
    label_1 = []
    for j in range(len(labels)):
        for i in range(len(w_coh)):
            label_1.append(labels[j])
    df_weights = pd.DataFrame({' ': label_1, 'weight': arr_weights})
    sns.violinplot(data=df_weights, x=" ", y="weight", ax=ax, color='grey')
    ax.set_ylabel('Weight (a.u.)')
    ax.axhline(y=0, linestyle='--', color='k', alpha=.4)


def fig_trajs_2(df, fgsz=(15, 5), accel=False, inset_sz=.06, marginx=0.06,
                marginy=0.2):
    f, ax = plt.subplots(nrows=2, ncols=4, figsize=fgsz)
    ax = ax.flatten()
    ax_cohs = np.array([ax[1], ax[5]])
    ax_zt = np.array([ax[2], ax[6]])
    # splitting
    trajs_splitting(df, ax=ax[0])
    rm_top_right_lines(ax[0])
    trajs_splitting_point(df=df, ax=ax[4])

    # trajs. conditioned on coh
    ax_inset = add_inset(ax=ax_cohs[0], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_cohs = np.insert(ax_cohs, 0, ax_inset)
    ax_inset = add_inset(ax=ax_cohs[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_cohs = np.insert(ax_cohs, 2, ax_inset)
    # trajs. conditioned on prior
    ax_inset = add_inset(ax=ax_zt[0], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_zt = np.insert(ax_zt, 0, ax_inset)
    ax_inset = add_inset(ax=ax_zt[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_zt = np.insert(ax_zt, 2, ax_inset)
    for a in ax:
        rm_top_right_lines(a)
    # TODO: the function below does not work with all subjects
    df_trajs = df.loc[df.subjid == 'LE43']
    trajs_cond_on_coh_computation(df=df_trajs, ax=ax_zt, condition='prior_x_coh',
                                  prior_limit=1, cmap='copper')
    trajs_cond_on_coh_computation(df=df_trajs, ax=ax_cohs, condition='choice_x_coh',
                                  cmap='coolwarm')
    # regression weights
    mt_weights(df, ax=ax[7], plot=True, means_errs=False)
    f.savefig(SV_FOLDER+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(SV_FOLDER+'/Fig2.svg', dpi=400, bbox_inches='tight')


def supp_fig_traj_tr_idx(df, fgsz=(15, 5), accel=False, marginx=0.01, marginy=0.05):
    fgsz = fgsz
    inset_sz = 0.08
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=fgsz)
    ax = ax.flatten()
    ax_ti = np.array([ax[0], ax[1]])

    # trajs. conditioned on trial index
    ax_inset = add_inset(ax=ax[0], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_ti = np.insert(ax_ti, 0, ax_inset)
    ax_inset = add_inset(ax=ax[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_ti = np.insert(ax_ti, 2, ax_inset)
    for a in ax:
        rm_top_right_lines(a)
    trajs_cond_on_coh_computation(df=df, ax=ax_ti, condition='prior_x_coh',
                                  prior_limit=1, cmap='copper')
    # splits
    mt_weights(df, ax=ax[3], plot=True, means_errs=False)
    trajs_splitting_point(df=df, ax=ax[7])
    f.savefig(SV_FOLDER+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(SV_FOLDER+'/Fig2.svg', dpi=400, bbox_inches='tight')


def tach_1st_2nd_choice(df, ax):
    # TODO: average across rats
    choice = df.R_response.values * 2 - 1
    coh = df.coh2.values
    gt = df.rewside.values * 2 - 1
    hit = df.hithistory.values
    sound_len = df.sound_len.values
    com = df.CoM_sugg.values
    choice_com = choice
    choice_com[com] = -choice[com]
    hit_com = choice_com == gt
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len})
    tachometric(df_plot_data, ax=ax, fill_error=True, cmap='YlGn')
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit_com,
                                 'sound_len': sound_len})
    tachometric(df_plot_data, ax=ax, fill_error=True, cmap='Blues',
                linestyle='--')
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Accuracy')
    legendelements = [Line2D([0], [0], linestyle='--', color='k', lw=2,
                             label='1st resp'),
                      Line2D([0], [0], color='k', lw=2, label='2nd resp')]
    ax.legend(handles=legendelements)


def fig_CoMs_3(df, inset_sz=.08, marginx=0.005, marginy=0.08, figsize=(10, 5),
               com_th=5):
    if com_th != 5:
        _, _, _, com = edd2.com_detection(trajectories=traj_y, decision=decision,
                                          time_trajs=time_trajs)
        com = np.array(com)  # new CoM list
        df['CoM_sugg'] = com
    fig, ax = plt.subplots(2, 4, figsize=figsize)
    ax = ax.flatten()
    ax_mat = [ax[2], ax[3]]
    rm_top_right_lines(ax=ax[5])
    tach_1st_2nd_choice(df=df, ax=ax[5])
    ax[6].axis('off')
    ax[7].axis('off')
    fig2.e(df, sv_folder=SV_FOLDER, ax=ax[4])
    plot_coms(df=df, ax=ax[1])
    ax_trck = ax[0]
    tracking_image(ax_trck)
    # plot Pcoms matrices
    nbins = 7
    matrix_side_0 = com_heatmap_marginal_pcom_side_mat(df=df, side=0)
    matrix_side_1 = com_heatmap_marginal_pcom_side_mat(df=df, side=1)
    # L-> R
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[0].set_title(pcomlabel_1)
    im = ax_mat[0].imshow(matrix_side_1, vmin=0, vmax=vmax)
    plt.sca(ax_mat[0])
    plt.colorbar(im, fraction=0.04)
    # pos = ax_mat.get_position()
    # ax_mat.set_position([pos.x0, pos.y0*2/3, pos.width, pos.height])
    # ax_mat_1 = plt.axes([pos.x0+pos.width+0.05, pos.y0*2/3,
    #                      pos.width, pos.height])
    pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
    ax_mat[1].set_title(pcomlabel_0)
    im = ax_mat[1].imshow(matrix_side_0, vmin=0, vmax=vmax)
    ax_mat[1].yaxis.set_ticks_position('none')
    plt.sca(ax_mat[1])
    plt.colorbar(im, fraction=0.04)
    for ax_i in [ax_mat[0], ax_mat[1]]:
        ax_i.set_xlabel('Prior Evidence')
        ax_i.set_yticklabels(['']*nbins)
        ax_i.set_xticklabels(['']*nbins)
    for ax_i in [ax_mat[0]]:
        ax_i.set_ylabel('Stimulus Evidence')  # , labelpad=-17)
    ax_inset = add_inset(ax=ax[1], inset_sz=inset_sz, fgsz=(4, 6),
                         marginx=marginx, marginy=marginy)
    fig_COMs_per_rat_inset_3(df=df, ax_inset=ax_inset)
    fig.savefig(SV_FOLDER+'fig3.svg', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER+'fig3.png', dpi=400, bbox_inches='tight')


def fig_COMs_per_rat_inset_3(df, ax_inset):
    subjects = df.subjid.unique()
    comlist_rats = []
    for subj in subjects:
        df_1 = df.loc[df.subjid == subj]
        mean_coms = np.nanmean(df_1.CoM_sugg.values)
        comlist_rats.append(mean_coms)
    ax_inset.plot(subjects, comlist_rats, 'o', color='k', markersize=4)
    ax_inset.set_ylabel('P(CoM)')
    ax_inset.set_xlabel('Rat')
    ax_inset.axhline(np.nanmean(comlist_rats), linestyle='--', color='k',
                     alpha=0.8)
    ax_inset.set_xticklabels(subjects, rotation=90)
    ax_inset.set_ylim(0, 0.125)


def fig_5_in(coh, hit, sound_len, decision, hit_model, sound_len_model, zt,
             decision_model, com, com_model, com_model_detected, pro_vs_re):
    """
    Deprecated
    """
    fig, ax = plt.subplots(ncols=4, nrows=3, gridspec_kw={'top': 0.95,
                                                          'bottom': 0.055,
                                                          'left': 0.055,
                                                          'right': 0.975,
                                                          'hspace': 0.38,
                                                          'wspace': 0.225})
    ax = ax.flatten()
    for ax_1 in ax:
        rm_top_right_lines(ax_1)
    psych_curve((decision+1)/2, coh, ret_ax=ax[1], kwargs_plot={'color': 'k'},
                kwargs_error={'label': 'Data', 'color': 'k'})
    ax[1].set_xlabel('Coherence')
    ax[1].set_ylabel('Probability of right')
    hit_model = hit_model[sound_len_model >= 0]
    com_model_detected = com_model_detected[sound_len_model >= 0]
    decision_model = decision_model[sound_len_model >= 0]
    com_model = com_model[sound_len_model >= 0]
    psych_curve((decision_model+1)/2, coh[sound_len_model >= 0], ret_ax=ax[1],
                kwargs_error={'label': 'Model', 'color': 'red'},
                kwargs_plot={'color': 'red'})
    ax[1].legend()
    pos_tach_ax = tachometric_data(coh=coh, hit=hit, sound_len=sound_len, ax=ax[2])
    ax[2].set_title('Data')
    pos_tach_ax_model = tachometric_data(coh=coh[sound_len_model >= 0],
                                         hit=hit_model,
                                         sound_len=sound_len_model[
                                             sound_len_model >= 0],
                                         ax=ax[3])
    ax[3].set_title('Model')
    reaction_time_histogram(sound_len=sound_len, label='Data', ax=ax[0],
                            bins=np.linspace(-150, 300, 91))
    reaction_time_histogram(sound_len=sound_len_model[sound_len_model >= 0],
                            label='Model', ax=ax[0],
                            bins=np.linspace(-150, 300, 91), pro_vs_re=pro_vs_re)
    ax[0].legend()
    express_performance(hit=hit, coh=coh, sound_len=sound_len,
                        pos_tach_ax=pos_tach_ax, ax=ax[4], label='Data')
    express_performance(hit=hit_model, coh=coh[sound_len_model >= 0],
                        sound_len=sound_len_model[sound_len_model >= 0],
                        pos_tach_ax=pos_tach_ax_model, ax=ax[4], label='Model')
    df_plot = pd.DataFrame({'com': com[sound_len_model >= 0],
                            'sound_len': sound_len[sound_len_model >= 0],
                            'rt_model': sound_len_model[sound_len_model >= 0],
                            'com_model': com_model,
                            'com_model_detected': com_model_detected})
    binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                 errorbar_kw={'label': 'Data', 'color': 'k'}, ax=ax[5])
    binned_curve(df_plot, 'com_model_detected', 'rt_model', bins=BINS_RT,
                 xpos=xpos_RT, errorbar_kw={'label': 'Model detected',
                                            'color': 'red'}, ax=ax[5])
    binned_curve(df_plot, 'com_model', 'rt_model', bins=BINS_RT, xpos=xpos_RT,
                 errorbar_kw={'label': 'Model all', 'color': 'green'}, ax=ax[5])
    ax[5].legend()
    ax[5].set_xlabel('RT (ms)')
    ax[5].set_ylabel('PCoM')
    binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                 errorbar_kw={'label': 'Data', 'color': 'k'}, ax=ax[6])
    binned_curve(df_plot, 'com_model_detected', 'rt_model', bins=BINS_RT,
                 xpos=xpos_RT, errorbar_kw={'label': 'Model detected',
                                            'color': 'red'}, ax=ax[6])
    ax[6].legend()
    ax[6].set_xlabel('RT (ms)')
    ax[6].set_ylabel('PCoM')
    decision_01 = (decision+1)/2
    edd2.com_heatmap_jordi(zt, coh, decision_01, ax=ax[8], flip=True,
                           annotate=False, xlabel='prior', ylabel='avg stim',
                           cmap='PRGn_r', vmin=0., vmax=1)
    cdfs(coh, sound_len, f5=True, ax=ax[7], label_title='Data', linestyle='solid')
    cdfs(coh, sound_len_model, f5=True, ax=ax[7], label_title='Model',
         linestyle='--', model=True)
    ax[8].set_title('Pright Data')
    zt_model = zt[sound_len_model >= 0]
    coh_model = coh[sound_len_model >= 0]
    decision_01_model = (decision_model+1)/2
    edd2.com_heatmap_jordi(zt_model, coh_model, decision_01_model, ax=ax[9],
                           flip=True, annotate=False, xlabel='prior',
                           ylabel='avg stim', cmap='PRGn_r', vmin=0., vmax=1)
    ax[9].set_title('Pright Model')
    edd2.com_heatmap_jordi(zt, coh, hit, ax=ax[10],
                           flip=True, xlabel='prior', annotate=False,
                           ylabel='avg stim', cmap='coolwarm', vmin=0.2, vmax=1)
    ax[10].set_title('Pcorrect Data')
    edd2.com_heatmap_jordi(zt_model, coh_model, hit_model, ax=ax[11],
                           flip=True, xlabel='prior', annotate=False,
                           ylabel='avg stim', cmap='coolwarm', vmin=0.2, vmax=1)
    ax[11].set_title('Pcorrect Model')
    df_data = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                            'norm_allpriors': zt/max(abs(zt)),
                            'R_response': (decision+1)/2})
    com_heatmap_paper_marginal_pcom_side(df_data, side=0)
    com_heatmap_paper_marginal_pcom_side(df_data, side=1)
    # matrix_data, _ = edd2.com_heatmap_jordi(zt, coh, com,
    #                                         return_mat=True, flip=True)
    # matrix_model, _ = edd2.com_heatmap_jordi(zt, coh, com_model,
    #                                          return_mat=True, flip=True)
    # sns.heatmap(matrix_data, ax=ax[8])
    # ax[8].set_title('Data')
    # sns.heatmap(matrix_model, ax=ax[9])
    # ax[9].set_title('Model')
    df_model = pd.DataFrame({'avtrapz': coh[sound_len_model >= 0],
                             'CoM_sugg':
                                 com_model_detected,
                             'norm_allpriors':
                                 zt_model/max(abs(zt_model)),
                             'R_response': (decision_model+1)/2})
    com_heatmap_paper_marginal_pcom_side(df_model, side=0)
    com_heatmap_paper_marginal_pcom_side(df_model, side=1)


def com_heatmap_marginal_pcom_side_mat(
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


def fig_5(coh, hit, sound_len, decision, hit_model, sound_len_model, zt,
          decision_model, com, com_model, com_model_detected, pro_vs_re,
          df_sim, means, errors, means_model, errors_model):
    fig, ax = plt.subplots(ncols=4, nrows=4, gridspec_kw={'top': 0.95,
                                                          'bottom': 0.055,
                                                          'left': 0.055,
                                                          'right': 0.975,
                                                          'hspace': 0.38,
                                                          'wspace': 0.225})
    ax = ax.flatten()
    for ax_1 in ax:
        rm_top_right_lines(ax_1)
    hit_model = hit_model[sound_len_model >= 0]
    com_model_detected = com_model_detected[sound_len_model >= 0]
    decision_model = decision_model[sound_len_model >= 0]
    com_model = com_model[sound_len_model >= 0]
    _ = tachometric_data(coh=coh[sound_len_model >= 0], hit=hit_model,
                         sound_len=sound_len_model[sound_len_model >= 0],
                         ax=ax[10], label='Model')
    # pdf_cohs(sound_len=sound_len, ax=ax[8], coh=coh, yaxis=True)
    # pdf_cohs(sound_len=sound_len_model[sound_len_model >= 0], ax=ax[9],
    #          coh=coh[sound_len_model >= 0], yaxis=False)
    # ax[8].set_title('Data')
    # ax[9].set_title('Model')
    df_plot = pd.DataFrame({'com': com[sound_len_model >= 0],
                            'sound_len': sound_len[sound_len_model >= 0],
                            'rt_model': sound_len_model[sound_len_model >= 0],
                            'com_model': com_model,
                            'com_model_detected': com_model_detected})
    binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                 errorbar_kw={'label': 'Data', 'color': 'k'}, ax=ax[12])
    binned_curve(df_plot, 'com_model_detected', 'rt_model', bins=BINS_RT,
                 xpos=xpos_RT, errorbar_kw={'label': 'Model detected',
                                            'color': 'red'}, ax=ax[12])
    binned_curve(df_plot, 'com_model', 'rt_model', bins=BINS_RT, xpos=xpos_RT,
                 errorbar_kw={'label': 'Model all', 'color': 'green'}, ax=ax[12])
    ax[12].xaxis.tick_top()
    ax[12].xaxis.tick_bottom()
    ax[12].legend()
    ax[12].set_xlabel('RT (ms)')
    ax[12].set_ylabel('PCoM')
    zt_model = zt[sound_len_model >= 0]
    coh_model = coh[sound_len_model >= 0]
    decision_01_model = (decision_model+1)/2
    edd2.com_heatmap_jordi(zt_model, coh_model, decision_01_model, ax=ax[11],
                           flip=True, annotate=False, xlabel='prior',
                           ylabel='avg stim', cmap='PRGn_r', vmin=0., vmax=1)
    ax[11].set_title('Pright Model')
    df_model = pd.DataFrame({'avtrapz': coh[sound_len_model >= 0],
                             'CoM_sugg':
                                 com_model_detected,
                             'norm_allpriors':
                                 zt_model/max(abs(zt_model)),
                             'R_response': (decision_model+1)/2})
    nbins = 7
    matrix_side_0 = com_heatmap_marginal_pcom_side_mat(df=df_model, side=0)
    matrix_side_1 = com_heatmap_marginal_pcom_side_mat(df=df_model, side=1)
    vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
    ax[13].set_title(pcomlabel_1)
    im = ax[13].imshow(matrix_side_1, vmin=0, vmax=vmax)
    plt.sca(ax[13])
    plt.colorbar(im, fraction=0.04)
    pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
    ax[14].set_title(pcomlabel_0)
    im = ax[14].imshow(matrix_side_0, vmin=0, vmax=vmax)
    ax[14].yaxis.set_ticks_position('none')
    plt.sca(ax[14])
    plt.colorbar(im, fraction=0.04)
    for ax_i in [ax[13], ax[14]]:
        ax_i.set_xlabel('Prior Evidence')
        ax_i.set_yticklabels(['']*nbins)
        ax_i.set_xticklabels(['']*nbins)
    ax[13].set_ylabel('Stimulus Evidence')
    plot_bars(means=means, errors=errors, ax=ax[15], f5=True,
              means_model=means_model, errors_model=errors_model)
    ax_pr = [ax[i] for i in [0, 4, 1, 5]]
    traj_cond_coh_simul(df_sim=df_sim, ax=ax_pr, median=False, prior=True)
    ax_coh = [ax[i] for i in [2, 6, 3, 7]]
    traj_cond_coh_simul(df_sim=df_sim, ax=ax_coh, median=False, prior=False,
                        prior_lim=0.25)
    # bins_MT = np.linspace(50, 600, num=25, dtype=int)
    trajs_splitting_point(df_sim, ax=ax[8], collapse_sides=False, threshold=300,
                          sim=True,
                          rtbins=np.linspace(0, 150, 16), connect_points=True,
                          draw_line=((0, 90), (90, 0)),
                          trajectory="trajectory_y")
    fig.savefig(SV_FOLDER+'fig5.svg', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER+'fig5.png', dpi=400, bbox_inches='tight')


def traj_model_plot(df_sim):
    fgsz = (8, 8)
    inset_sz = 0.1
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=fgsz)
    ax = ax.flatten()
    ax_cohs = np.array([ax[0], ax[2]])
    ax_inset = add_inset(ax=ax_cohs[0], inset_sz=inset_sz, fgsz=fgsz)
    ax_cohs = np.insert(ax_cohs, 0, ax_inset)
    ax_inset = add_inset(ax=ax_cohs[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginy=0.15)
    ax_cohs = np.insert(ax_cohs, 2, ax_inset)
    # trajs_cond_on_coh(df_sim, ax=ax)
    simul.whole_splitting(df=df_sim, ax=ax[1], simul=True)
    ax[1].set_xlim(-10, 200)
    ax[1].set_ylim(-20, 20)
    trajs_splitting_point(df=df_sim, ax=ax[3], sim=True)
    trajs_cond_on_coh(df=df_sim, ax=ax_cohs)


def traj_cond_coh_simul(df_sim, ax=None, median=True, prior=True, traj_thr=30,
                        vel_thr=0.2, prior_lim=1):
    # TODO: save each matrix? or save the mean and std
    if median:
        func_final = np.nanmedian
    if not median:
        func_final = np.nanmean
    nanidx = df_sim.loc[df_sim[['dW_trans',
                                'dW_lat']].isna().sum(axis=1) == 2].index
    df_sim['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df_sim.loc[nanidx, 'allpriors'] = np.nan
    df_sim['choice_x_coh'] = (df_sim.R_response*2-1) * df_sim.coh2
    bins_coh = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    bins_zt = [-1, -0.6, -0.15, 0.15, 0.6, 1]
    xvals_zt = [-1, -0.5, 0, 0.5, 1]
    signed_response = df_sim.R_response.values
    df_sim['normallpriors'] = df_sim['allpriors'] /\
        np.nanmax(df_sim['allpriors'].abs())*(signed_response*2 - 1)
    lens = []
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax = ax.flatten()
    vals_thr_traj = []
    vals_thr_vel = []
    labels_zt = ['inc. high', 'inc. low', 'zero', 'con. low', 'con. high']
    if prior:
        bins_ref = bins_zt
    else:
        bins_ref = bins_coh
    for i_ev, ev in enumerate(bins_ref):
        if not prior:
            index = (df_sim.choice_x_coh.values == ev) *\
                (df_sim.R_response.values == 1) *\
                (df_sim.allpriors.abs() <= prior_lim)
            colormap = pl.cm.coolwarm(np.linspace(0, 1, len(bins_coh)))
        if prior:
            if ev == 1:
                break
            index = (df_sim.normallpriors.values >= bins_zt[i_ev]) *\
                (df_sim.normallpriors.values < bins_zt[i_ev + 1]) *\
                (df_sim.R_response.values == 1)
            colormap = pl.cm.copper(np.linspace(0, 1, len(bins_zt)-1))
            # (df_sim.R_response.values == 1) *\
        lens.append(max([len(t) for t in df_sim.trajectory_y[index].values]))
        traj_all = np.empty((sum(index), max(lens)))
        traj_all[:] = np.nan
        vel_all = np.empty((sum(index), max(lens)))
        vel_all[:] = np.nan
        for tr in range(sum(index)):
            vals_traj = df_sim.traj[index].values[tr] *\
                (signed_response[index][tr]*2 - 1)
            vals_vel = df_sim.traj_d1[index].values[tr] *\
                (signed_response[index][tr]*2 - 1)
            traj_all[tr, :len(vals_traj)] = vals_traj
            vel_all[tr, :len(vals_vel)] = vals_vel
        mean_traj = func_final(traj_all, axis=0)
        std_traj = np.nanstd(traj_all, axis=0) / np.sqrt(sum(index))
        # val_traj = np.argmax(mean_traj >= traj_thr)
        val_traj = np.mean(df_sim['resp_len'].values[index])*1e3
        if prior:
            xval = xvals_zt[i_ev]
        else:
            xval = ev
        ax[2].scatter(xval, val_traj, color=colormap[i_ev], marker='D', s=60)
        vals_thr_traj.append(val_traj)
        mean_vel = func_final(vel_all, axis=0)
        std_vel = np.nanstd(vel_all, axis=0) / np.sqrt(sum(index))
        val_vel = np.argmax(mean_vel >= vel_thr)
        ax[3].scatter(xval, val_vel, color=colormap[i_ev], marker='D', s=60)
        vals_thr_vel.append(val_vel)
        if not prior:
            label = '{}'.format(ev)
        if prior:
            label = labels_zt[i_ev]
        ax[0].plot(np.arange(len(mean_traj)), mean_traj, label=label,
                   color=colormap[i_ev])
        ax[0].fill_between(x=np.arange(len(mean_traj)),
                           y1=mean_traj - std_traj, y2=mean_traj + std_traj,
                           color=colormap[i_ev])
        ax[1].plot(np.arange(len(mean_vel)), mean_vel, label=label,
                   color=colormap[i_ev])
        ax[1].fill_between(x=np.arange(len(mean_vel)),
                           y1=mean_vel - std_vel, y2=mean_vel + std_vel,
                           color=colormap[i_ev])
    ax[0].axhline(y=30, linestyle='--', color='k', alpha=0.4)
    ax[1].axhline(y=0.2, linestyle='--', color='k', alpha=0.4)
    if prior:
        leg_title = 'prior congruency'
        ax[2].plot(xvals_zt, vals_thr_traj, color='k', linestyle='--',
                   alpha=0.6)
        ax[3].plot(xvals_zt, vals_thr_vel, color='k', linestyle='--',
                   alpha=0.6)
        ax[2].set_xlabel('Prior congruency', fontsize=10)
        ax[3].set_xlabel('Prior congruency', fontsize=10)
    if not prior:
        leg_title = 'stim congruency'
        ax[2].plot(bins_coh, vals_thr_traj, color='k', linestyle='--', alpha=0.6)
        ax[3].plot(bins_coh, vals_thr_vel, color='k', linestyle='--', alpha=0.6)
        ax[2].set_xlabel('Evidence congruency', fontsize=10)
        ax[3].set_xlabel('Evidence congruency', fontsize=10)
    ax[0].legend(title=leg_title)
    ax[0].set_ylabel('y-coord (px)', fontsize=10)
    ax[0].set_xlabel('Time from movement onset (ms)', fontsize=10)
    ax[0].set_title('Mean trajectory', fontsize=10)
    ax[1].legend(title=leg_title)
    ax[1].set_ylabel('Velocity (px/s)', fontsize=10)
    ax[1].set_xlabel('Time from movement onset (ms)', fontsize=10)
    ax[1].set_title('Mean velocity', fontsize=10)
    ax[2].set_ylabel('MT (ms)', fontsize=10)
    ax[3].set_ylabel('Time to reach threshold (ms)', fontsize=10)


def supp_trajs_prior_cong(df_sim, ax=None):
    signed_response = df_sim.R_response.values
    nanidx = df_sim.loc[df_sim[['dW_trans',
                                'dW_lat']].isna().sum(axis=1) == 2].index
    df_sim['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df_sim.loc[nanidx, 'allpriors'] = np.nan
    df_sim['normallpriors'] = df_sim['allpriors'] /\
        np.nanmax(df_sim['allpriors'].abs())*(signed_response*2 - 1)
    if ax is None:
        fig, ax = plt.subplots(1)
    bins_zt = [0.6, 1]
    lens = []
    for i_ev, ev in enumerate(bins_zt):
        if ev == 1:
            break
        index = (df_sim.normallpriors.values >= bins_zt[i_ev]) *\
            (df_sim.normallpriors.values < bins_zt[i_ev + 1])
        lens.append(max([len(t) for t in df_sim.trajectory_y[index].values]))
        traj_all = np.empty((sum(index), max(lens)))
        traj_all[:] = np.nan
        for tr in range(sum(index)):
            vals_traj = df_sim.traj[index].values[tr] *\
                (signed_response[index][tr]*2 - 1)
            traj_all[tr, :len(vals_traj)] = vals_traj
            ax.plot(vals_traj, color='k', alpha=0.4)
        mean_traj = np.nanmean(traj_all, axis=0)
    ax.plot(np.arange(len(mean_traj)), mean_traj, label='Mean', color='yellow',
            linewidth=4)
    ax.set_ylabel('y-coord (px)', fontsize=10)
    ax.set_xlabel('Time from movement onset (ms)', fontsize=10)


def fig_humans_6(user_id, sv_folder, nm='300', max_mt=600, jitter=0.003,
                wanted_precision=8, traj_thr=240, vel_thr=2):
    if user_id == 'Alex':
        folder = 'C:\\Users\\Alexandre\\Desktop\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'AlexCRM':
        folder = 'C:/Users/agarcia/Desktop/CRM/human/'
    if user_id == 'Manuel':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    subj = ['general_traj']
    steps = [None]
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)


def human_trajs(df_data, user_id, sv_folder, nm='300', max_mt=600, jitter=0.003,
                wanted_precision=8, traj_thr=240, vel_thr=2):
    
    # TRAJECTORIES
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    coh = df_data.avtrapz.values
    decision = df_data.R_response.values
    trajs = df_data.trajectory_y.values
    times = df_data.times.values
    ev_vals = np.unique(np.abs(np.round(coh, 2)))
    bins = [0, 0.25, 0.5, 1]
    # congruent_coh = coh * (decision*2 - 1)
    fig, ax = plt.subplots(nrows=2, ncols=4)
    ax = ax.flatten()
    colormap = pl.cm.coolwarm(np.linspace(0, 1, len(ev_vals)))
    vals_thr_traj = []
    vals_thr_vel = []
    precision = 16
    for i_ev, ev in enumerate(ev_vals):
        index = np.abs(np.round(coh, 2)) == ev
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        all_vels = np.empty((sum(index), max_mt))
        all_vels[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)
            max_time = max(time)*1e3
            if max_time > max_mt:
                continue
            # vals_fin = np.interp(np.arange(0, int(max_time), wanted_precision),
            #                      xp=time*1e3, fp=vals)
            vels_fin = np.diff(vals-vals[0])/precision
            all_trajs[tr, :len(vals)] = vals  # - vals[0]
            all_vels[tr, :len(vels_fin)] = vels_fin
        mean_traj = np.nanmean(all_trajs, axis=0)
        std_traj = np.sqrt(np.nanstd(all_trajs, axis=0) / sum(index))
        # val_traj = np.where(mean_traj >= traj_thr)[0][2]*wanted_precision
        val_traj = np.nanmean(np.array([float(t[-1]) for t in
                                        df_data.times.values[index]
                                        if t[-1] != '']))*1e3
        vals_thr_traj.append(val_traj)
        ax[1].scatter(ev, val_traj, color=colormap[i_ev], marker='D', s=60)
        mean_vel = np.nanmean(all_vels, axis=0)
        std_vel = np.sqrt(np.nanstd(all_vels, axis=0) / sum(index))
        for ind_v, velocity in enumerate(mean_vel):
            if velocity >= vel_thr and ind_v*precision >= 160:
                val_vel = ind_v*precision
                break
        vals_thr_vel.append(val_vel)
        ax[5].scatter(ev, val_vel, color=colormap[i_ev], marker='D', s=60)
        ax[0].plot(np.arange(len(mean_traj))*precision, mean_traj,
                   color=colormap[i_ev], label='{}'.format(bins[i_ev]))
        ax[0].fill_between(x=np.arange(len(mean_traj))*precision,
                           y1=mean_traj-std_traj, y2=mean_traj+std_traj,
                           color=colormap[i_ev])
        ax[4].plot(np.arange(len(mean_vel))*precision, mean_vel,
                   color=colormap[i_ev], label='{}'.format(bins[i_ev]))
        ax[4].fill_between(x=np.arange(len(mean_vel))*precision,
                           y1=mean_vel-std_vel, y2=mean_vel+std_vel,
                           color=colormap[i_ev])
    ax[1].plot(bins, vals_thr_traj, color='k', linestyle='--', alpha=0.6)
    ax[5].plot(bins, vals_thr_vel, color='k', linestyle='--', alpha=0.6)
    ax[0].set_xlim(-0.1, 550)
    ax[4].set_xlim(-0.1, 550)
    ax[4].set_ylim(1, 4)
    # ax[0].axhline(y=traj_thr, linestyle='--', color='k', alpha=0.4)
    ax[4].axhline(y=vel_thr, linestyle='--', color='k', alpha=0.4)
    ax[0].legend(title='stimulus')
    ax[0].set_ylabel('y-coord (px)')
    ax[0].set_xlabel('Time from movement onset (ms)')
    ax[0].set_title('Mean trajectory')
    ax[4].legend(title='stimulus')
    ax[4].set_ylabel('Velocity (px/s)')
    ax[4].set_xlabel('Time from movement onset (ms)')
    ax[4].set_title('Mean velocity')
    ax[1].set_xlabel('Evidence')
    ax[1].set_ylabel('Motor time (ms)')
    ax[5].set_xlabel('Evidence')
    ax[5].set_ylabel('Time to reach threshold (ms)')
    # now for the prior
    prior = df_data['norm_allpriors'] * (decision*2 - 1)
    # cong_prior = prior * (decision*2 - 1)
    bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
    colormap = pl.cm.copper(np.linspace(0, 1, len(bins)-1))
    vals_thr_traj = []
    vals_thr_vel = []
    labels = ['inc. high', 'inc. low', 'zero', 'con. low', 'con. high']
    for i_pr, pr_min in enumerate(bins):
        if pr_min == 1:
            break
        index = (prior >= bins[i_pr])*(prior < bins[i_pr+1])
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        all_vels = np.empty((sum(index), max_mt))
        all_vels[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)
            max_time = max(time)*1e3
            if max_time > max_mt:
                continue
            # vals_fin = np.interp(np.arange(0, int(max_time), wanted_precision),
            #                      xp=time*1e3, fp=vals)
            vels_fin = np.diff(vals-vals[0])/precision
            all_trajs[tr, :len(vals)] = vals  # - vals[0]
            all_vels[tr, :len(vels_fin)] = vels_fin
        mean_traj = np.nanmean(all_trajs, axis=0)
        std_traj = np.sqrt(np.nanstd(all_trajs, axis=0) / sum(index))
        # val_traj = np.where(mean_traj >= traj_thr)[0][2]*wanted_precision
        val_traj = np.nanmean(np.array([float(t[-1]) for t in
                                        df_data.times.values[index]
                                        if t[-1] != '']))*1e3
        vals_thr_traj.append(val_traj)
        ax[3].scatter(i_pr, val_traj, color=colormap[i_pr], marker='D', s=60)
        mean_vel = np.nanmean(all_vels, axis=0)
        std_vel = np.sqrt(np.nanstd(all_vels, axis=0) / sum(index))
        for ind_v, velocity in enumerate(mean_vel):
            if velocity >= vel_thr and ind_v*precision >= 50:
                val_vel = ind_v*precision
                break
        vals_thr_vel.append(val_vel)
        ax[7].scatter(i_pr, val_vel, color=colormap[i_pr], marker='D', s=60)
        ax[2].plot(np.arange(len(mean_traj))*precision, mean_traj,
                   color=colormap[i_pr], label='{}'.format(labels[i_pr]))
        ax[2].fill_between(x=np.arange(len(mean_traj))*precision,
                           y1=mean_traj-std_traj, y2=mean_traj+std_traj,
                           color=colormap[i_pr])
        ax[6].plot(np.arange(len(mean_vel))*precision, mean_vel,
                   color=colormap[i_pr], label='{}'.format(labels[i_pr]))
        ax[6].fill_between(x=np.arange(len(mean_vel))*precision,
                           y1=mean_vel-std_vel, y2=mean_vel+std_vel,
                           color=colormap[i_pr])
    ax[3].plot(np.arange(5), vals_thr_traj, color='k', linestyle='--', alpha=0.6)
    ax[7].plot(np.arange(5), vals_thr_vel, color='k', linestyle='--', alpha=0.6)
    ax[2].set_xlim(-0.1, 550)
    ax[6].set_xlim(-0.1, 550)
    ax[6].set_ylim(1, 4)
    # ax[0].axhline(y=traj_thr, linestyle='--', color='k', alpha=0.4)
    ax[6].axhline(y=vel_thr, linestyle='--', color='k', alpha=0.4)
    ax[2].legend(title='Prior')
    ax[2].set_ylabel('y-coord (px)')
    ax[2].set_xlabel('Time from movement onset (ms)')
    ax[2].set_title('Mean trajectory')
    ax[6].legend(title='Prior')
    ax[6].set_ylabel('Velocity (px/s)')
    ax[6].set_xlabel('Time from movement onset (ms)')
    ax[6].set_title('Mean velocity')
    ax[3].set_xlabel('Prior congruency')
    ax[3].set_ylabel('Motor time (ms)')
    ax[7].set_xlabel('Prior congruency')
    ax[7].set_xticks(np.arange(5))
    ax[7].set_xticklabels(labels)
    ax[3].set_xticks(np.arange(5))
    ax[3].set_xticklabels(labels)
    ax[7].set_ylabel('Time to reach threshold (ms)')


def accuracy_1st_2nd_ch(gt, decision, coh, com):  # ??
    coh_com = coh[com]
    gt_com = gt[com]
    decision_com = decision[com]
    ev_vals = np.unique(np.abs(coh_com))
    acc_ch1 = []
    acc_ch2 = []
    for ev in ev_vals:
        index = np.abs(coh_com) == ev
        acc_ch1.append(np.mean((-decision_com[index]) == gt_com[index]))
        acc_ch2.append(np.mean(decision_com[index] == gt_com[index]))


def linear_fun(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[2]


def mt_linear_reg(mt, coh, trial_index, com, prior, plot=False):
    """

    Parameters
    ----------
    mt : array
        DESCRIPTION.
    coh : array (abs)
        DESCRIPTION.
    trial_index : array
        DESCRIPTION.
    prior : array (abs)
        congruent prior with final decision.

    Returns
    -------
    popt : TYPE
        DESCRIPTION.

    """
    trial_index = trial_index.astype(float)[~com.astype(bool)]
    xdata = np.array([[coh[~com.astype(bool)]],
                      [trial_index],
                      [prior[~com.astype(bool)]]]).reshape(3, sum(~com))
    ydata = np.array(mt[~com.astype(bool)]*1e3)
    popt, pcov = curve_fit(f=linear_fun, xdata=xdata, ydata=ydata)
    if plot:
        df = pd.DataFrame({'coh': coh/max(coh), 'prior': prior/max(prior),
                           'MT': resp_len*1e3,
                           'trial_index': trial_index/max(trial_index)})
        plt.figure()
        sns.pointplot(data=df, x='coh', y='MT', label='coh')
        sns.pointplot(data=df, x='prior', y='MT', label='prior')
        sns.pointplot(data=df, x='trial_index', y='MT', label='trial_index')
        plt.ylabel('MT (ms)')
        plt.xlabel('normalized variables')
        plt.legend()
    return popt


def basic_statistics(decision, resp_fin):
    mat = confusion_matrix(decision, resp_fin)
    print(mat)
    fpr, tpr, _ = roc_curve(resp_fin, decision)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


def run_model(stim, zt, coh, gt, trial_index, num_tr=None):
    if num_tr is not None:
        num_tr = num_tr
    else:
        num_tr = int(len(zt))
    data_augment_factor = 10
    MT_slope = 0.123
    MT_intercep = 254
    detect_CoMs_th = 5
    p_t_aff = 8
    p_t_eff = 8
    p_t_a = 14  # 90 ms (18) PSIAM fit includes p_t_eff
    p_w_zt = 0.2
    p_w_stim = 0.11
    p_e_noise = 0.01
    p_com_bound = 0.001
    p_w_a_intercept = 0.052
    p_w_a_slope = -2.2e-05  # fixed
    p_a_noise = 0.04  # fixed
    p_1st_readout = 40
    p_2nd_readout = 40

    stim = edd2.data_augmentation(stim=stim.reshape(20, num_tr),
                                  daf=data_augment_factor)
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
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj


def pdf_cohs_subj(df, bins=np.linspace(1, 301, 61), pval_max=0.001):
    ev_vals = [0, 1]
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, len(ev_vals)))
    num_subjects = len(df.subjid.unique())
    density_matrix_0 = np.zeros((num_subjects, len(bins)-1))
    density_matrix_1 = np.zeros((num_subjects, len(bins)-1))
    xvals = bins[:-1]+(bins[1]-bins[0])/2
    subjects = df.subjid.unique()
    for i_sub, subj in enumerate(subjects):
        df1 = df.loc[df.subjid == subj]
        sound_len = df1.sound_len.values
        coh = df1.coh2.values
        for i_coh, ev in enumerate(ev_vals):
            index = np.abs(coh) == ev
            counts_coh, _ = np.histogram(sound_len[index], bins=bins)
            norm_counts = counts_coh/sum(counts_coh)
            if ev == 0:
                density_matrix_0[i_sub, :] = norm_counts
            else:
                density_matrix_1[i_sub, :] = norm_counts
    # plist = []
    for i_rt, rt_bin in enumerate(xvals):
        density_vals_0 = density_matrix_0[:, i_rt+1]
        density_vals_1 = density_matrix_1[:, i_rt+1]
        _, p_value = ttest_ind(density_vals_0, density_vals_1)
        # plist.append(p_value)
        if p_value < pval_max:
            ind = rt_bin
            break
    fig, ax = plt.subplots(1)
    # for i_sub in range(num_subjects):
    #     ax.plot(xvals, density_matrix_0[i_sub, :], color=colormap[0],
    #             linewidth=.7, alpha=0.2)
    #     ax.plot(xvals, density_matrix_1[i_sub, :], color=colormap[1],
    #             linewidth=.7, alpha=0.2)
    mean_density_0 = np.nanmean(density_matrix_0, axis=0)
    mean_density_1 = np.nanmean(density_matrix_1, axis=0)
    std_density_0 = np.nanstd(density_matrix_0, axis=0)/np.sqrt(num_subjects)
    std_density_1 = np.nanstd(density_matrix_1, axis=0)/np.sqrt(num_subjects)
    ax.plot(xvals, mean_density_0, color=colormap[0], linewidth=2, label='coh=0')
    ax.plot(xvals, mean_density_1, color=colormap[1], linewidth=2, label='coh=1')
    ax.fill_between(xvals, mean_density_0-std_density_0,
                    mean_density_0+std_density_0, color=colormap[0], alpha=0.5)
    ax.fill_between(xvals, mean_density_1-std_density_1,
                    mean_density_1+std_density_1, color=colormap[1], alpha=0.3)
    ax.axvline(ind, color='r', linestyle='--', alpha=0.8,
               label='{} ms'.format(ind))
    ax.legend()
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Density')


def fig_7(df, df_sim):
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
    coh = df.coh2.values
    com = df.CoM_sugg.values
    com_model = df_sim['com_detcted'].values
    sound_len_model = df_sim.sound_len.values
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax = ax.flatten()
    fig.suptitle('Stim/prior')
    sound_len = df.sound_len.values
    window = [0, 25, 50, 100, 125, 150, 175, 200, 250]
    for i in range(8):
        zt_tmp = zt[(sound_len > window[i]) * (sound_len < window[i+1])]
        coh_tmp = coh[(sound_len > window[i]) * (sound_len < window[i+1])]
        com_tmp = com[(sound_len > window[i]) * (sound_len < window[i+1])]
        edd2.com_heatmap_jordi(zt_tmp, coh_tmp, com_tmp, ax=ax[i],
                               flip=True, annotate=False, xlabel='prior',
                               ylabel='avg stim', cmap='rocket')
        ax[i].set_title('{} < RT < {}'.format(window[i], window[i+1]))
    pos = ax[8].get_position()
    x, y, w, h = pos.x0, pos.y0, pos.width, pos.height
    inset_height = h/2
    inset_width = w/2
    ax[8].set_title('Model')
    ax[8].axis('off')
    ax_inset_1 = plt.axes([x, y, inset_height, inset_width])
    ax_inset_2 = plt.axes([x+inset_width, y, inset_height, inset_width])
    ax_inset_3 = plt.axes([x, y+inset_height, inset_height, inset_width])
    ax_inset_4 = plt.axes([x+inset_width, y+inset_height, inset_height,
                           inset_width])
    ax_inset = [ax_inset_3, ax_inset_4, ax_inset_1, ax_inset_2]
    window = [0, 50, 100, 125, 150]
    for i in range(4):
        zt_tmp = zt[(sound_len_model > window[i]) *
                    (sound_len_model < window[i+1])]
        coh_tmp = coh[(sound_len_model > window[i]) *
                      (sound_len_model < window[i+1])]
        com_model_tmp = com_model[(sound_len_model > window[i]) *
                                  (sound_len_model < window[i+1])]
        edd2.com_heatmap_jordi(zt_tmp, coh_tmp, com_model_tmp, ax=ax_inset[i],
                               flip=True, annotate=False, xlabel='',
                               ylabel='', cmap='rocket')
    ax_inset_1.set_xlabel('Prior evidence')
    ax_inset_1.set_ylabel('Stimulus evidence')
    fig.savefig(SV_FOLDER+'fig7.svg', dpi=400, bbox_inches='tight')
    fig.savefig(SV_FOLDER+'fig7.png', dpi=400, bbox_inches='tight')


def supp_com_marginal(df):
    fig, ax = plt.subplots(nrows=len(df.subjid.unique()), ncols=2,
                           figsize=(4, 12))
    ax = ax.flatten()
    for i_ax, subj in enumerate(df.subjid.unique()):
        df_1 = df.loc[df.subjid == subj]
        nbins = 7
        matrix_side_0 = com_heatmap_marginal_pcom_side_mat(df=df_1, side=0)
        matrix_side_1 = com_heatmap_marginal_pcom_side_mat(df=df_1, side=1)
        # L-> R
        vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
        pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
        im = ax[i_ax*2].imshow(matrix_side_1, vmin=0, vmax=vmax)
        plt.sca(ax[i_ax*2])
        plt.colorbar(im, fraction=0.04)
        # R -> L
        pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
        im = ax[i_ax*2+1].imshow(matrix_side_0, vmin=0, vmax=vmax)
        ax[i_ax*2+1].yaxis.set_ticks_position('none')
        plt.sca(ax[i_ax*2+1])
        plt.colorbar(im, fraction=0.04)
        if i_ax == 0:
            ax[i_ax].set_title(pcomlabel_1)
            ax[i_ax+1].set_title(pcomlabel_0)
        for ax_i in [ax[i_ax*2], ax[i_ax*2+1]]:
            ax_i.set_yticklabels(['']*nbins)
            ax_i.set_xticklabels(['']*nbins)
        ax[i_ax*2].set_ylabel('stim. {}'.format(subj))
        if i_ax == len(df.subjid.unique()) - 1:
            ax[i_ax*2].set_xlabel('Prior evidence')
            ax[i_ax*2+1].set_xlabel('Prior evidence')
    fig.savefig(SV_FOLDER+'fig_supp_com_marginal.svg', dpi=400,
                bbox_inches='tight')
    fig.savefig(SV_FOLDER+'fig_supp_com_marginal.png', dpi=400,
                bbox_inches='tight')


def norm_allpriors_per_subj(df):
    norm_allpriors = np.empty((0,))
    for subj in df.subjid.unique():
        df_1 = df.loc[df.subjid == subj]
        zt_tmp = np.nansum(df_1[["dW_lat", "dW_trans"]].values, axis=1)
        norm_allpriors = np.concatenate((norm_allpriors,
                                         zt_tmp/max(abs(zt_tmp))))
    return norm_allpriors


def supp_different_com_thresholds(traj_y, time_trajs, decision, sound_len,
                                  com_th_list=np.linspace(0.5, 10, 20)):
    fig, ax = plt.subplots(1)
    rm_top_right_lines(ax=ax)
    colormap = pl.cm.Reds(np.linspace(0.2, 1, len(com_th_list)))
    com_d = {}
    for i_th, com_th in enumerate(com_th_list):
        print('Com threshold = ' + str(com_th))
        _, _, _, com = edd2.com_detection(trajectories=traj_y, decision=decision,
                                          time_trajs=time_trajs,
                                          com_threshold=com_th)
        df_plot = pd.DataFrame({'sound_len': sound_len, 'com': com})
        binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                     errorbar_kw={'color': colormap[i_th], 'label': str(com_th)},
                     ax=ax)
        com_d['com_'+str(com_th)] = com
    # ax.legend()
    ax.set_xlabel('RT(ms)')
    ax.set_ylabel('P(CoM)')
    com_dframe = pd.DataFrame(com_d)
    com_dframe.to_csv(SV_FOLDER + 'com_diff_thresholds.csv')


def pcom_vs_prior_coh(df, bins_zt=np.linspace(-1, 1, 14),
                      bins_coh=[-1, -0.5, -0.25, 0, 0.25, 0.5, 1]):
    fig, ax = plt.subplots(2, 2)
    ax = ax.flatten()
    for a in ax:
        rm_top_right_lines(ax=a)
    subjects = df.subjid.unique()
    for j in [0, 2]:
        com_vs_zt = np.zeros((len(subjects), len(bins_zt)-1))
        error_com_vs_zt = np.zeros((len(subjects), len(bins_zt)-1))
        for i_sub, subj in enumerate(subjects):
            df_1 = df.loc[df.subjid == subj]
            zt_tmp = np.nansum(df_1[["dW_lat", "dW_trans"]].values, axis=1)
            if j != 0:
                zt_tmp *= (df_1.R_response.values*2-1)
                ax[j].set_xlabel('Prior Congruency')
            if j == 0:
                ax[j].set_xlabel('Prior')
            ax[j].set_ylabel('P(CoM)')
            norm_zt = zt_tmp/max(abs(zt_tmp))
            for i_b, bin_zt in enumerate(bins_zt[:-1]):
                index_zt = (norm_zt >= bin_zt)*(norm_zt < bins_zt[i_b+1])
                com_binned = np.nanmean(df_1.CoM_sugg.values[index_zt])
                error_com = np.nanstd(df_1.CoM_sugg.values[index_zt]) /\
                    np.sqrt(sum(index_zt))
                com_vs_zt[i_sub, i_b] = com_binned
                error_com_vs_zt[i_sub, i_b] = error_com
            com_vs_zt[i_sub, :] /= np.max(com_vs_zt[i_sub, :])
            ax[j].errorbar(bins_zt[:-1], com_vs_zt[i_sub, :],
                           error_com_vs_zt[i_sub, :],
                           color='k', alpha=0.5)
        total_mean_com_zt = np.nanmean(com_vs_zt, axis=0)
        total_error_com_zt = np.nanstd(com_vs_zt, axis=0)/np.sqrt(len(subjects))
        ax[j].errorbar(bins_zt[:-1], total_mean_com_zt, total_error_com_zt,
                       color='k', alpha=1, linewidth=3)
    for j in [1, 3]:
        com_vs_coh = np.zeros((len(subjects), len(bins_coh)))
        error_com_vs_coh = np.zeros((len(subjects), len(bins_coh)))
        for i_sub, subj in enumerate(subjects):
            df_1 = df.loc[df.subjid == subj]
            coh = df_1.coh2.values
            if j != 1:
                coh *= (df_1.R_response.values*2-1)
                ax[j].set_xlabel('Stimulus Congruency')
            if j == 1:
                ax[j].set_xlabel('Stimulus')
            for i_b, bin_coh in enumerate(bins_coh):
                index_coh = coh == bin_coh
                com_binned = np.nanmean(df_1.CoM_sugg.values[index_coh])
                error_com = np.nanstd(df_1.CoM_sugg.values[index_coh]) /\
                    np.sqrt(sum(index_coh))
                com_vs_coh[i_sub, i_b] = com_binned
                error_com_vs_coh[i_sub, i_b] = error_com
            com_vs_coh[i_sub, :] /= np.max(com_vs_coh[i_sub, :])
            ax[j].errorbar(bins_coh, com_vs_coh[i_sub, :],
                           error_com_vs_coh[i_sub, :], color='k', alpha=0.5)
        total_mean_com_coh = np.nanmean(com_vs_coh, axis=0)
        total_error_com_coh = np.nanstd(com_vs_coh, axis=0)/np.sqrt(len(subjects))
        ax[j].errorbar(bins_coh, total_mean_com_coh, total_error_com_coh,
                       color='k', alpha=1, linewidth=3)


# ---MAIN
if __name__ == '__main__':
    plt.close('all')
    f1 = True
    f2 = True
    f3 = False
    f5 = False
    f6 = False
    f7 = False
    if f1 or f2 or f3 or f5:
        all_rats = True
        if all_rats:
            subjects = ['LE42', 'LE43', 'LE38', 'LE39', 'LE85', 'LE84', 'LE45',
                        'LE40', 'LE46', 'LE86', 'LE47', 'LE37', 'LE41', 'LE36',
                        'LE44']
        else:
            subjects = ['LE43']
        df_all = pd.DataFrame()
        for sbj in subjects:
            df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + sbj, return_df=True,
                                          sv_folder=SV_FOLDER, after_correct=True,
                                          silent=True, all_trials=True)
            if all_rats:
                df_all = pd.concat((df_all, df))
        if all_rats:
            df = df_all
        # XXX: can we remove the code below or move it to the fig5 part?
        after_correct_id = np.where((df.aftererror == 0))
        # *(df.special_trial == 0))[0]
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
        traj_stamps = df.trajectory_stamps.values[after_correct_id]
        traj_y = df.trajectory_y.values[after_correct_id]
        fix_onset = df.fix_onset_dt.values[after_correct_id]

        sound_len = np.array(df.sound_len)
        sound_len = sound_len[after_correct_id]
        gt = np.array(df.rewside) * 2 - 1
        gt = gt[after_correct_id]
        trial_index = np.array(df.origidx)
        trial_index = trial_index[after_correct_id]
        resp_len = np.array(df.resp_len)
        resp_len = resp_len[after_correct_id]
        time_trajs = edd2.get_trajs_time(resp_len=resp_len,
                                         traj_stamps=traj_stamps,
                                         fix_onset=fix_onset, com=com,
                                         sound_len=sound_len)
        print('Computing CoMs')
        _, _, _, com = edd2.com_detection(trajectories=traj_y, decision=decision,
                                          time_trajs=time_trajs)
        print('Ended Computing CoMs')
        com = np.array(com)  # new CoM list

        df['norm_allpriors'] = norm_allpriors_per_subj(df)
        df['CoM_sugg'] = com
        # if we want to use data from all rats, we must use dani_clean.pkl

    # fig 1
    if f1:
        fig_rats_behav_1(df_data=df)

    # fig 2
    if f2:
        fig_trajs_2(df=df, fgsz=(10, 5))

    # fig 3
    if f3:
        fig_CoMs_3(df)
        supp_com_marginal(df)

    # fig 5 (model)
    if f5:
        num_tr = int(2e5)
        decision = decision[:int(num_tr)]
        zt = zt[:int(num_tr)]
        sound_len = sound_len[:int(num_tr)]
        coh = coh[:int(num_tr)]
        com = com[:int(num_tr)]
        gt = gt[:int(num_tr)]
        trial_index = trial_index[:int(num_tr)]
        hit = hit[:int(num_tr)]
        if stim.shape[0] != 20:
            stim = stim.T
        stim = stim[:, :int(num_tr)]
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            pro_vs_re, trajs =\
            run_model(stim=stim, zt=zt, coh=coh, gt=gt, trial_index=trial_index,
                      num_tr=None)
        # basic_statistics(decision=decision, resp_fin=resp_fin)  # dec
        # basic_statistics(com, com_model_detected)  # com
        # basic_statistics(hit, hit_model)  # hit
        MT = [len(t) for t in trajs]
        df_sim = pd.DataFrame({'coh2': coh, 'trajectory_y': trajs,
                               'sound_len': reaction_time,
                               'rewside': (gt + 1)/2,
                               'R_response': (resp_fin+1)/2,
                               'resp_len': np.array(MT)*1e-3})
        df_sim['CoM_sugg'] = com_model
        df_sim['traj_d1'] = [np.diff(t) for t in trajs]
        df_sim['aftererror'] =\
            np.array(df.aftererror)[after_correct_id][:int(num_tr)]
        df_sim['subjid'] = 'simul'
        df_sim['dW_trans'] =\
            np.array(df.dW_trans)[after_correct_id][:int(num_tr)]
        df_sim['origidx'] =\
            np.array(df.origidx)[after_correct_id][:int(num_tr)]
        df_sim['dW_lat'] = np.array(df.dW_lat)[after_correct_id][:int(num_tr)]
        df_sim['special_trial'] =\
            np.array(df.special_trial)[after_correct_id][:int(num_tr)]
        df_sim['traj'] = df_sim['trajectory_y']
        df_sim['com_detcted'] = com_model_detected
        # simulation plots
        means, errors = mt_weights(df, means_errs=True, ax=None)
        means_model, errors_model = mt_weights(df_sim, means_errs=True, ax=None)
        fig_5(coh=coh, hit=hit, sound_len=sound_len, decision=decision, zt=zt,
              hit_model=hit_model, sound_len_model=reaction_time.astype(int),
              decision_model=resp_fin, com=com, com_model=com_model,
              com_model_detected=com_model_detected, pro_vs_re=pro_vs_re,
              means=means, errors=errors, means_model=means_model,
              errors_model=errors_model, df_sim=df_sim)
        supp_trajs_prior_cong(df_sim, ax=None)
    if f6:
        # human traj plots
        human_trajs(user_id='Manuel', sv_folder=SV_FOLDER, max_mt=600,
                    wanted_precision=12, traj_thr=250, vel_thr=2.8, nm='300')
    if f7:
        fig_7(df, df_sim)
    # from utilsJ.Models import extended_ddm_v2 as edd2
    # import numpy as np
    # import matplotlib.pyplot as plt
    # DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
    # SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
    #     'ChangesOfMind/figures/from_python/'  # Manuel

    # df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + 'LE43_',
    #                               return_df=True, sv_folder=SV_FOLDER)

    # coms = df.loc[df.CoM_sugg]
    # rts = coms.sound_len

    # max_ = 0
    # for tr in range(len(coms)):
    #     trial = df.iloc[tr]
    #     traj = trial['trajectory_y']
    #     plt.plot(traj, 'k')
    #     max_temp = np.nanmax(traj)
    #     if max_temp > max_:
    #         max_ = max_temp
    #         print(max_)
    #     if np.nanmax(traj) > 200:
    #         print(trial)
