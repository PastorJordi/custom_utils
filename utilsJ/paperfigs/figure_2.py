import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from scipy.stats import sem
import sys
sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel
from utilsJ.Models import simul
from utilsJ.paperfigs import figures_paper as fp
from utilsJ.Behavior.plotting import trajectory_thr, interpolapply

def plots_trajs_conditioned_old(df, ax, data_folder, condition='choice_x_coh', cmap='viridis',
                            prior_limit=0.25, rt_lim=50,
                            after_correct_only=True,
                            trajectory="trajectory_y",
                            velocity=("traj_d1", 1),
                            acceleration=('traj_d2', 1)):
    """
    Plots mean trajectories, MT, velocity and peak velocity
    conditioning on Coh/Zt/T.index,
    """
    interpolatespace = np.linspace(-700000, 1000000, 1700)
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)
    df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
    # prior_lim = np.quantile(df.norm_allpriors, prior_limit)
    if after_correct_only:
        ac_cond = df.aftererror == False
    else:
        ac_cond = (df.aftererror*1) >= 0
    if condition == 'choice_x_coh':
        bins = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        xlab = 'ev. resp.'
        bintype = 'categorical'
        indx_trajs = (df.norm_allpriors.abs() <= prior_limit) &\
            ac_cond & (df.special_trial == 0) &\
            (df.sound_len < rt_lim)
        mt = df.resp_len.values*1e3
        n_iters = len(bins)
        colormap = pl.cm.coolwarm(np.linspace(0., 1, n_iters))
    if condition == 'choice_x_prior':
        bins_zt = [-1.01]
        for i_p, perc in enumerate([0.5, 0.25, 0.25, 0.5]):
            if i_p > 2:
                bins_zt.append(df.norm_allpriors.abs().quantile(perc))
            else:
                bins_zt.append(-df.norm_allpriors.abs().quantile(perc))
        bins_zt.append(1.01)
        bins = np.array(bins_zt)
        bins = np.array([-1, -0.4, -0.05, 0.05, 0.4, 1])
        xlab = 'prior resp.'
        bintype = 'edges'
        rt_lim = 200
        indx_trajs = (df.norm_allpriors.abs() <= prior_limit) &\
            ac_cond & (df.special_trial == 2) &\
            (df.sound_len < rt_lim)
        n_iters = len(bins)-1
        colormap = pl.cm.copper(np.linspace(0., 1, n_iters))
    if condition == 'origidx':
        bins = np.linspace(0, 1e3, num=6)
        bintype = 'edges'
        n_iters = len(bins) - 1
        indx_trajs = (df.norm_allpriors.abs() <= prior_limit) &\
            ac_cond & (df.special_trial == 0) &\
            (df.sound_len < rt_lim)
        colormap = pl.cm.jet(np.linspace(0., 1, n_iters))

    # position
    subjects = df['subjid'].unique()
    mat_all = np.empty((n_iters, 1700, len(subjects)))
    mt_all = np.empty((n_iters, len(subjects)))
    for i_subj, subj in enumerate(subjects):
        traj_data = data_folder+subj+'/traj_data/'+subj+'_traj_pos_'+condition+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data):
            traj_data = np.load(traj_data, allow_pickle=True)
            mean_traj = traj_data['mean_traj']
            xpoints = traj_data['xpoints']
            mt_time = traj_data['mt_time']
        else:
            xpoints, _, _, mat, _, mt_time =\
                trajectory_thr(df.loc[(indx_trajs) & (df.subjid == subj)],
                               condition, bins, collapse_sides=True, thr=30,
                               ax=None, ax_traj=None, return_trash=True,
                               error_kwargs=dict(marker='o'), cmap=cmap, bintype=bintype,
                               trajectory=trajectory, plotmt=True, alpha_low=False)
            mean_traj = np.array([np.nanmean(mat[m], axis=0) for m in mat])
            data = {'xpoints': xpoints, 'mean_traj': mean_traj, 'mt_time': mt_time}
            np.savez(traj_data, **data)
        mat_all[:, :, i_subj] = mean_traj
        mt_all[:, i_subj] = mt_time
    all_trajs = np.nanmean(mat_all, axis=2)
    all_trajs_err = np.nanstd(mat_all, axis=2) / np.sqrt(len(subjects))
    mt_time = np.nanmedian(mt_all, axis=1)
    mt_time_err = np.nanstd(mt_all, axis=1) / np.sqrt(len(subjects))
    for i_tr, traj in enumerate(all_trajs):
        ax[1].plot(interpolatespace/1000, traj, color=colormap[i_tr])
        ax[1].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr], color=colormap[i_tr],
                           alpha=0.5)
        if len(subjects) > 1:
            c = colormap[i_tr]
            xp = [xpoints[i_tr]]
            ax[0].boxplot(mt_all[i_tr, :], positions=xp, 
                          boxprops=dict(markerfacecolor=c, markeredgecolor=c))
            ax[0].plot(xp + 0.1*np.random.randn(len(subjects)),
                       mt_all[i_tr, :], color=colormap[i_tr], marker='o',
                       linestyle='None')
        else:
            ax[0].errorbar(xpoints[i_tr], mt_time[i_tr], yerr=mt_time_err[i_tr],
                           color=colormap[i_tr], marker='o')
    if condition == 'choice_x_coh':
        legendelements = [Line2D([0], [0], color=colormap[0], lw=2, label='-1'),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[2], lw=2, label=''),
                          Line2D([0], [0], color=colormap[3], lw=2, label='0'),
                          Line2D([0], [0], color=colormap[4], lw=2, label=''),
                          Line2D([0], [0], color=colormap[5], lw=2, label=''),
                          Line2D([0], [0], color=colormap[6], lw=2, label='1')]
        ax[1].legend(handles=legendelements, title='Stimulus \n evidence',
                     loc='upper left', fontsize=7)
        ax[0].set_yticklabels('')
        ax[0].set_yticks([])
        ax[0].set_xticks([-1, 0, 1])
        ax[0].set_xticklabels(['-1', '0', '1'], fontsize=9)
        ax[0].set_xlabel('Stimulus')
        ax[2].set_xticks([0])
        ax[2].set_xticklabels(['Stimulus'], fontsize=9)
        ax[2].xaxis.set_ticks_position('none')
        # ax[0].set_ylim(220, 285)
        ax[0].set_yticks([240, 275])
        ax[0].set_yticklabels(['240', '275'])
        # ax[2].set_ylim([0.5, 0.8])
    if condition == 'choice_x_prior':
        ax[0].set_yticklabels('')
        ax[0].set_yticks([])
        legendelements = [Line2D([0], [0], color=colormap[4], lw=2,
                                 label='congruent'),
                          Line2D([0], [0], color=colormap[3], lw=2,
                                 label=''),
                          Line2D([0], [0], color=colormap[2], lw=2,
                                 label='0'),
                          Line2D([0], [0], color=colormap[1], lw=2,
                                 label=''),
                          Line2D([0], [0], color=colormap[0], lw=2,
                                 label='incongruent')]
        ax[1].legend(handles=legendelements, title='Prior', loc='upper left',
                     fontsize=7)
        xpoints = (bins[:-1] + bins[1:]) / 2
        # ax[0].set_ylim(230, 310)
        ax[0].set_yticks([250, 300])
        ax[0].set_yticklabels(['250', '300'])
        ax[0].set_xticks([xpoints[0], 0, xpoints[-1]])
        ax[0].set_xticklabels(['Incongruent', '0', 'Congruent'])
        ax[0].set_xlabel('Prior')
        # ax[2].set_ylim([0.5, 0.8])
        ax[2].set_xticks([0])
        ax[2].set_xticklabels(['Prior'], fontsize=9)
        ax[2].xaxis.set_ticks_position('none')
    if condition == 'origidx':
        legendelements = []
        labs = ['100', '300', '500', '700', '900']
        for i in range(len(colormap)):
            legendelements.append(Line2D([0], [0], color=colormap[i], lw=2,
                                  label=labs[i]))
        ax[1].legend(handles=legendelements, title='Trial index')
        ax[2].set_xlabel('Trial index')
    ax[1].set_xlim([-20, 450])
    ax[1].set_xticklabels('')
    ax[1].axhline(0, c='gray')
    ax[1].set_ylabel('Position (pixels)')
    ax[0].set_ylabel('MT (ms)', fontsize=9)
    # ax[1].set_ylim([-10, 85])
    ax[1].set_yticks([0, 25, 50, 75])
    ax[1].axhline(78, color='gray', linestyle=':')
    ax[0].plot(xpoints, mt_time, color='k', ls=':')
    # velocities
    mat_all = np.empty((n_iters, 1700, len(subjects)))
    mt_all = np.empty((n_iters, len(subjects)))
    for i_subj, subj in enumerate(subjects):
        traj_data = data_folder + subj + '/traj_data/' + subj + '_traj_vel_'+condition+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data):
            traj_data = np.load(traj_data, allow_pickle=True)
            mean_traj = traj_data['mean_traj']
            xpoints = traj_data['xpoints']
            mt_time = traj_data['mt_time']
            ypoints = traj_data['ypoints']
        else:
            xpoints, ypoints, _, mat, _, mt_time =\
                trajectory_thr(df.loc[(indx_trajs) & (df.subjid == subj)],
                               condition, bins,
                               collapse_sides=True, thr=30, ax=None, ax_traj=None,
                               return_trash=True, error_kwargs=dict(marker='o'),
                               cmap=cmap, bintype=bintype,
                               trajectory=velocity, plotmt=True, alpha_low=False)
            mean_traj = np.array([np.nanmean(mat[m], axis=0) for m in mat])
            data = {'xpoints': xpoints, 'ypoints': ypoints, 'mean_traj': mean_traj, 'mt_time': mt_time}
            np.savez(traj_data, **data)
        mat_all[:, :, i_subj] = mean_traj
        mt_all[:, i_subj] = ypoints
    all_trajs = np.nanmean(mat_all, axis=2)
    all_trajs_err = np.nanstd(mat_all, axis=2) / np.sqrt(len(subjects))
    mt_time = np.nanmedian(mt_all, axis=1)
    mt_time_err = np.nanstd(mt_all, axis=1) / np.sqrt(len(subjects))
    for i_tr, traj in enumerate(all_trajs):
        ax[3].plot(interpolatespace/1000, traj, color=colormap[i_tr])
        ax[3].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr], color=colormap[i_tr],
                           alpha=0.5)
        if len(subjects) > 1:
            xp = [xpoints[i_tr]]
            c = colormap[i_tr]
            ax[2].boxplot(mt_all[i_tr, :], positions=xp,
                          boxprops=dict(markerfacecolor=c, markeredgecolor=c))
            ax[2].plot(xpoints[i_tr] + 0.1*np.random.randn(len(subjects)),
                       mt_all[i_tr, :], color=colormap[i_tr], marker='o',
                       linestyle='None')
        else:
            ax[2].errorbar(xpoints[i_tr], mt_time[i_tr], yerr=mt_time_err[i_tr],
                           color=colormap[i_tr], marker='o')
    ax[3].set_xlim([-20, 450])
    ax[2].set_ylabel('Peak (pixels/ms)')
    # ax[3].set_ylim([-0.05, 0.5])
    ax[3].axhline(0, c='gray')
    ax[3].set_ylabel('Velocity (pixels/ms)')
    ax[3].set_xlabel('Time from movement onset (ms)')
    ax[2].plot(xpoints, mt_time, color='k', ls=':')


def plots_trajs_conditioned(df, ax, data_folder, condition='choice_x_coh', cmap='viridis',
                            prior_limit=0.25, rt_lim=50,
                            after_correct_only=True,
                            trajectory="trajectory_y",
                            velocity=("traj_d1", 1)):
    """
    Plots mean trajectories, MT, velocity and peak velocity
    conditioning on Coh/Zt/T.index,
    """
    interpolatespace = np.linspace(-700000, 1000000, 1700)
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)
    df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
    bins, bintype, indx_trajs, n_iters, colormap =\
          fp.get_bin_info(df=df, condition=condition, prior_limit=prior_limit,
                          after_correct_only=after_correct_only,
                          rt_lim=rt_lim)
    # POSITION
    subjects = df['subjid'].unique()
    mat_all = np.empty((n_iters, 1700, len(subjects)))
    mt_all = np.empty((n_iters, len(subjects)))
    for i_subj, subj in enumerate(subjects):
        traj_data = data_folder+subj+'/traj_data/'+subj+'_traj_pos_'+condition+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data):
            traj_data = np.load(traj_data, allow_pickle=True)
            mean_traj = traj_data['mean_traj']
            xpoints = traj_data['xpoints']
            mt_time = traj_data['mt_time']
        else:
            xpoints, _, _, mat, _, mt_time =\
                trajectory_thr(df.loc[(indx_trajs) & (df.subjid == subj)],
                               condition, bins, collapse_sides=True, thr=30,
                               ax=None, ax_traj=None, return_trash=True,
                               error_kwargs=dict(marker='o'), cmap=cmap, bintype=bintype,
                               trajectory=trajectory, plotmt=True, alpha_low=False)
            mean_traj = np.array([np.nanmean(mat[m], axis=0) for m in mat])
            data = {'xpoints': xpoints, 'mean_traj': mean_traj, 'mt_time': mt_time}
            np.savez(traj_data, **data)
        mat_all[:, :, i_subj] = mean_traj
        mt_all[:, i_subj] = mt_time
    all_trajs = np.nanmean(mat_all, axis=2)
    all_trajs_err = np.nanstd(mat_all, axis=2) / np.sqrt(len(subjects))
    mt_time = np.nanmedian(mt_all, axis=1)
    for i_tr, traj in enumerate(all_trajs):
        traj -= np.nanmean(traj[(interpolatespace > -100000) * (interpolatespace < 0)])
        ax[0].plot(interpolatespace/1000, traj, color=colormap[i_tr])
        ax[0].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr], color=colormap[i_tr],
                           alpha=0.5)
    ax[1].set_ylim([0.5, 0.8])
    if condition == 'choice_x_coh':
        legendelements = [Line2D([0], [0], color=colormap[0], lw=2, label='-1'),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[2], lw=2, label=''),
                          Line2D([0], [0], color=colormap[3], lw=2, label='0'),
                          Line2D([0], [0], color=colormap[4], lw=2, label=''),
                          Line2D([0], [0], color=colormap[5], lw=2, label=''),
                          Line2D([0], [0], color=colormap[6], lw=2, label='1')]
        ax[0].legend(handles=legendelements, title='Stimulus \n evidence',
                     loc='upper left')
        ax[1].set_xlabel('Stimulus')
    if condition == 'choice_x_prior':
        legendelements = [Line2D([0], [0], color=colormap[4], lw=2,
                                 label='congruent'),
                          Line2D([0], [0], color=colormap[3], lw=2,
                                 label=''),
                          Line2D([0], [0], color=colormap[2], lw=2,
                                 label='0'),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[0], lw=2,
                                 label='incongruent')]
        ax[0].legend(handles=legendelements, title='Prior', loc='upper left')
        ax[1].set_xlabel('Prior')
    if condition == 'origidx':
        legendelements = []
        labs = ['100', '300', '500', '700', '900']
        for i in range(len(colormap)):
            legendelements.append(Line2D([0], [0], color=colormap[i], lw=2,
                                  label=labs[i]))
        ax[0].legend(handles=legendelements, title='Trial index')
        ax[1].set_xlabel('Trial index')
    ax[0].set_xlim([-20, 450])
    ax[0].set_xticklabels('')
    ax[0].axhline(0, c='gray')
    ax[0].set_ylabel('Position (pixels)')
    ax[0].set_xlabel('Time from movement onset (ms)')
    ax[0].set_ylim([-10, 85])
    ax[0].set_yticks([0, 25, 50, 75])
    ax[0].axhline(78, color='gray', linestyle=':')
    # VELOCITIES
    mat_all = np.empty((n_iters, 1700, len(subjects)))
    mt_all = np.empty((n_iters, len(subjects)))
    for i_subj, subj in enumerate(subjects):
        traj_data = data_folder + subj + '/traj_data/' + subj + '_traj_vel_'+condition+'.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(traj_data), exist_ok=True)
        if os.path.exists(traj_data):
            traj_data = np.load(traj_data, allow_pickle=True)
            mean_traj = traj_data['mean_traj']
            xpoints = traj_data['xpoints']
            mt_time = traj_data['mt_time']
            ypoints = traj_data['ypoints']
        else:
            xpoints, ypoints, _, mat, _, mt_time =\
                trajectory_thr(df.loc[(indx_trajs) & (df.subjid == subj)],
                               condition, bins,
                               collapse_sides=True, thr=30, ax=None, ax_traj=None,
                               return_trash=True, error_kwargs=dict(marker='o'),
                               cmap=cmap, bintype=bintype,
                               trajectory=velocity, plotmt=True, alpha_low=False)
            mean_traj = np.array([np.nanmean(mat[m], axis=0) for m in mat])
            data = {'xpoints': xpoints, 'ypoints': ypoints, 'mean_traj': mean_traj, 'mt_time': mt_time}
            np.savez(traj_data, **data)
        mat_all[:, :, i_subj] = mean_traj
        mt_all[:, i_subj] = ypoints
    all_trajs = np.nanmean(mat_all, axis=2)
    all_trajs_err = np.nanstd(mat_all, axis=2) / np.sqrt(len(subjects))
    mt_time = np.nanmedian(mt_all, axis=1)
    mt_time_err = np.nanstd(mt_all, axis=1) / np.sqrt(len(subjects))
    for i_tr, traj in enumerate(all_trajs):
        traj -= np.nanmean(traj[(interpolatespace > -100000) * (interpolatespace < 0)])
        ax[2].plot(interpolatespace/1000, traj, color=colormap[i_tr])
        ax[2].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr], color=colormap[i_tr],
                           alpha=0.5)
        if False:  # len(subjects) > 1:
            xp = [xpoints[i_tr]]
            c = colormap[i_tr]
            ax[1].boxplot(mt_all[i_tr, :], positions=xp,
                          boxprops=dict(markerfacecolor=c, markeredgecolor=c))
            ax[1].plot(xpoints[i_tr] + 0.1*np.random.randn(len(subjects)),
                       mt_all[i_tr, :], color=colormap[i_tr], marker='o',
                       linestyle='None')
        else:
            ax[1].errorbar(xpoints[i_tr], mt_time[i_tr], yerr=mt_time_err[i_tr],
                           color=colormap[i_tr], marker='o')
    ax[2].set_xlim([-20, 450])
    ax[1].set_ylabel('Peak (pixels/ms)')
    ax[2].set_ylim([-0.05, 0.5])
    ax[2].axhline(0, c='gray')
    ax[2].set_ylabel('Velocity (pixels/ms)')
    ax[2].set_xlabel('Time from movement onset (ms)')
    print(ax[1].get_xticks())
    # ax[1].plot(xpoints, mt_time, color='k', ls=':')


def get_split_ind_corr(mat, evl, pval=0.01, max_MT=400, startfrom=700, sim=True):
    # Returns index at which the trajectories and coh vector become uncorrelated
    # backwards in time
    # mat: trajectories (n trials x time)
    plist = []
    for i in reversed(range(max_MT)):  # reversed so it goes backwards in time
        pop_a = mat[:, startfrom + i]
        nan_idx = ~np.isnan(pop_a)
        pop_evidence = evl[nan_idx]
        pop_a = pop_a[nan_idx]
        try:
            _, p2 = pearsonr(pop_a, pop_evidence)  # p2 = pvalue from pearson corr
            plist.append(p2)
        except Exception:
            continue
            # return np.nan
        if p2 > pval:
            return i + 1
        if sim and np.isnan(p2):
            return i + 1
    return np.nan


def plot_trajs_splitting_example(df, ax, rtbin=0, rtbins=np.linspace(0, 150, 2),
                                 subject='LE37', xlab=False):
    """
    Plot trajectories depending on COH and the corresponding Splitting Time as arrow.


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
    def plot_boxcar_rt(rt, ax, low_val=0, high_val=2):
        # plot box representing stimulus duration
        x_vals = np.linspace(-1, rt+5, num=100)
        y_vals = [low_val]
        for x in x_vals[:-1]:
            if x <= rt:
                y_vals.append(high_val)
            else:
                y_vals.append(low_val)
        ax.step(x_vals, y_vals, color='k')
        ax.fill_between(x_vals, y_vals, np.repeat(0, len(y_vals)),
                        color='grey', alpha=0.6)

    subject = subject
    lbl = 'RTs: ['+str(rtbins[rtbin])+'-'+str(rtbins[rtbin+1])+']'
    colors = pl.cm.gist_yarg(np.linspace(0.4, 1, 3))
    evs = [0.25, 0.5, 1]
    mat = np.empty((1701,))
    evl = np.empty(())
    appb = True
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    for iev, ev in enumerate(evs):
        indx = (df.special_trial == 0) & (df.subjid == subject)
        if np.sum(indx) > 0:
            _, matatmp, matb =\
                simul.when_did_split_dat(df=df[indx], side=0, collapse_sides=True,
                                         ax=ax, rtbin=rtbin, rtbins=rtbins,
                                         color=colors[iev], label=lbl, coh1=ev,
                                         align='sound')
        if appb:
            mat = matb
            evl = np.repeat(0, matb.shape[0])
            appb = False
        mat = np.concatenate((mat, matatmp))
        evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
    ind = get_split_ind_corr(mat, evl, pval=0.01, max_MT=400, startfrom=700)
    ax.set_xlim(-10, 255)
    ax.set_ylim(-0.6, 5.2)
    if xlab:
        ax.set_xlabel('Time from stimulus onset (ms)')
    if rtbins[-1] > 25 and xlab:
        ax.set_title('\n RT > 150 ms', fontsize=8)
        ax.arrow(ind, 3, 0, -2, color='k', width=1, head_width=5,
                 head_length=0.4)
        ax.text(ind-17, 3.4, 'Splitting Time', fontsize=8)
        ax.set_ylabel("{} Snout position (px)".format("           "))
        plot_boxcar_rt(rt=rtbins[0], ax=ax)
    else:
        ax.set_title('RT < 15 ms', fontsize=8)
        ax.text(ind-60, 3.3, 'Splitting Time', fontsize=8)
        ax.arrow(ind, 2.85, 0, -1.4, color='k', width=1, head_width=5,
                 head_length=0.4)
        ax.set_xticklabels([''])
        plot_boxcar_rt(rt=rtbins[-1], ax=ax)
        labels = ['0', '0.25', '0.5', '1']
        legendelements = []
        for i_l, lab in enumerate(labels):
            legendelements.append(Line2D([0], [0], color=colormap[i_l], lw=2,
                                  label=lab))
        ax.legend(handles=legendelements, fontsize=7, loc='upper right')


def trajs_splitting_prior(df, ax, data_folder, rtbins=np.linspace(0, 150, 16),
                          trajectory='trajectory_y', threshold=300):
    # split time/subject by prior
    ztbins = [0.1, 0.4, 1.1]
    kw = {"trajectory": trajectory, "align": "sound"}
    out_data = []
    df_1 = df.copy()
    for subject in df_1.subjid.unique():
        out_data_sbj = []
        split_data = data_folder + subject + '/traj_data/' + subject + '_traj_split_prior.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data):
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data']
        else:
            for i in range(rtbins.size-1):
                dat = df_1.loc[(df_1.subjid == subject) &
                            (df_1.sound_len < rtbins[i + 1]) &
                            (df_1.sound_len >= rtbins[i])]
                matb_0 = np.vstack(
                    dat.loc[(dat.norm_allpriors < 0.1) &
                            (dat.norm_allpriors >= 0)]
                    .apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
                matb_1 = np.vstack(
                    dat.loc[(dat.norm_allpriors > -0.1) &
                            (dat.norm_allpriors <= 0)]
                    .apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
                matb = np.vstack([matb_0*-1, matb_1])
                mat = matb
                ztl = np.repeat(0, matb.shape[0])
                for i_z, zt1 in enumerate(ztbins[:-1]):
                    mata_0 = np.vstack(
                        dat.loc[(dat.norm_allpriors > zt1) &
                                (dat.norm_allpriors <= ztbins[i_z+1])]
                        .apply(lambda x: interpolapply(x, **kw),
                            axis=1).values.tolist())
                    mata_1 = np.vstack(
                        dat.loc[(dat.norm_allpriors < -zt1) &
                                (dat.norm_allpriors >= -ztbins[i_z+1])]
                        .apply(lambda x: interpolapply(x, **kw),
                            axis=1).values.tolist())
                    mata = np.vstack([mata_0*-1, mata_1])
                    ztl = np.concatenate((ztl, np.repeat(zt1, mata.shape[0])))
                    mat = np.concatenate((mat, mata))
                current_split_index =\
                    get_split_ind_corr(mat, ztl, pval=0.01, max_MT=400,
                                    startfrom=700)
                if current_split_index >= rtbins[i]:
                    out_data_sbj += [current_split_index]
                else:
                    out_data_sbj += [np.nan]
            np.savez(split_data, out_data=out_data_sbj)
        out_data += [out_data_sbj]
    out_data = np.array(out_data).reshape(
        df_1.subjid.unique().size, rtbins.size-1, -1)
    out_data = np.swapaxes(out_data, 0, 1)
    out_data = out_data.astype(float)
    out_data[out_data > threshold] = np.nan
    binsize = rtbins[1]-rtbins[0]
    for i in range(df_1.subjid.unique().size):
        for j in range(out_data.shape[2]):
            ax.plot(binsize/2 + binsize * np.arange(rtbins.size-1),
                    out_data[:, i, j],
                    marker='o', mfc=(.6, .6, .6, .3), mec=(.6, .6, .6, 1),
                    mew=1, color=(.6, .6, .6, .3))
    error_kws = dict(ecolor='goldenrod', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='goldenrod', marker='o', label='mean & SEM')
    ax.errorbar(binsize/2 + binsize * np.arange(rtbins.size-1),
                np.nanmedian(out_data.reshape(rtbins.size-1, -1), axis=1),
                yerr=sem(out_data.reshape(rtbins.size-1, -1),
                         axis=1, nan_policy='omit'), **error_kws)
    ax.set_xlabel('RT (ms)')
    # ax.set_title('Impact of prior', fontsize=9)
    ax.set_ylabel('Splitting time (ms)')
    ax.plot([0, 155], [0, 155], color='k')
    ax.fill_between([0, 250], [0, 250], [0, 0],
                    color='grey', alpha=0.6)
    ax.set_xlim(-5, 155)
    # plt.show()


def trajs_splitting_stim(df, ax, data_folder, collapse_sides=True, threshold=300,
                         sim=False,
                         rtbins=np.linspace(0, 150, 16), connect_points=False,
                         trajectory="trajectory_y"):

    # split time/subject by coherence
    if sim:
        splitfun = simul.when_did_split_simul
        df['traj'] = df.trajectory_y.values
    if not sim:
        splitfun = simul.when_did_split_dat
    out_data = []
    for subject in df.subjid.unique():
        out_data_sbj = []
        split_data = data_folder + subject + '/traj_data/' + subject + '_traj_split_stim.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data):
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data_sbj']
        else:
            for i in range(rtbins.size-1):
                if collapse_sides:
                    evs = [0.25, 0.5, 1]
                    mat = np.empty((1701,))
                    evl = np.empty(())
                    appb = True
                    for ev in evs:
                        if not sim:  # TODO: do this if within splitfun
                            _, matatmp, matb =\
                                splitfun(df=df.loc[(df.special_trial == 0)
                                                   & (df.subjid == subject)],
                                         side=0, collapse_sides=True,
                                         rtbin=i, rtbins=rtbins, coh1=ev,
                                         trajectory=trajectory, align="sound")
                        if sim:
                            _, matatmp, matb =\
                                splitfun(df=df.loc[(df.special_trial == 0)
                                                   & (df.subjid == subject)],
                                         side=0, rtbin=i, rtbins=rtbins, coh=ev,
                                         align="sound")
                        if appb:
                            mat = matb
                            evl = np.repeat(0, matb.shape[0])
                            appb = False
                        mat = np.concatenate((mat, matatmp))
                        evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
                    if not sim:
                        current_split_index =\
                            get_split_ind_corr(mat, evl, pval=0.01, max_MT=400,
                                            startfrom=700)
                    if sim:
                        max_mt = 800
                        current_split_index =\
                            get_split_ind_corr(mat, evl, pval=0.01, max_MT=max_mt,
                                            startfrom=0)+1
                    if current_split_index >= rtbins[i]:
                        out_data_sbj += [current_split_index]
                    else:
                        out_data_sbj += [np.nan]
                else:
                    for j in [0, 1]:  # side values
                        current_split_index, _, _ = splitfun(
                            df.loc[df.subjid == subject],
                            j,  # side has no effect because this is collapsing_sides
                            rtbin=i, rtbins=rtbins, align='sound')
                        out_data_sbj += [current_split_index]
            np.savez(split_data, out_data_sbj=out_data_sbj)
        out_data += [out_data_sbj]

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

    error_kws = dict(ecolor='firebrick', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='firebrick', marker='o', label='mean & SEM')
    ax.errorbar(
        binsize/2 + binsize * np.arange(rtbins.size-1),
        # we do the mean across rtbin axis
        np.nanmedian(out_data.reshape(rtbins.size-1, -1), axis=1),
        # other axes we dont care
        yerr=sem(out_data.reshape(rtbins.size-1, -1),
                 axis=1, nan_policy='omit'),
        **error_kws
    )
    # if draw_line is not None:
    #     ax.plot(*draw_line, c='r', ls='--', zorder=0, label='slope -1')
    ax.plot([0, 155], [0, 155], color='k')
    ax.fill_between([0, 250], [0, 250], [0, 0],
                    color='grey', alpha=0.6)
    ax.set_xlim(-5, 155)
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Splitting time (ms)')
    # ax.set_title('Impact of stimulus')
    # plt.show()


def fig_2_trajs_old(df, data_folder, sv_folder, rat_nocom_img, fgsz=(8, 8), inset_sz=.06, marginx=0.008,
                marginy=0.05):
    f = plt.figure(figsize=fgsz)
    # FIGURE LAYOUT
    # mt vs zt
    ax_label = f.add_subplot(3, 3, 1)
    fp.add_text(ax_label, 'a', x=-0.1, y=1.2)
    # mt vs stim
    ax_label = f.add_subplot(3, 3, 2)
    fp.add_text(ax_label, 'b', x=-0.1, y=1.2)
    ax_label = f.add_subplot(3, 3, 3)
    ax_label.axis('off')
    # trajs prior
    ax_label = f.add_subplot(3, 3, 4)
    fp.add_text(ax_label, 'c', x=-0.1, y=1.2)
    # trajs stim
    ax_label = f.add_subplot(3, 3, 5)
    fp.add_text(ax_label, 'd', x=-0.1, y=1.2)
    ax_label = f.add_subplot(3, 3, 6)
    ax_label.axis('off')
    # vel prior
    ax_label = f.add_subplot(3, 3, 7)
    fp.add_text(ax_label, 'e', x=-0.1, y=1.2)
    ax_label = f.add_subplot(3, 3, 8)
    fp.add_text(ax_label, 'f', x=-0.1, y=1.2)
    # adjust panels positions
    plt.subplots_adjust(top=0.95, bottom=0.09, left=0.075, right=0.98,
                        hspace=0.5, wspace=0.4)
    ax = f.axes
    pos_ax_0 = ax[0].get_position()
    ax[0].set_position([pos_ax_0.x0, pos_ax_0.y0, pos_ax_0.width*1.6,
                        pos_ax_0.height])
    ax[1].set_position([pos_ax_0.x0 + pos_ax_0.width*2.2, pos_ax_0.y0,
                        pos_ax_0.width*1.6, pos_ax_0.height])
    pos_ax_3 = ax[3].get_position()
    ax[3].set_position([pos_ax_3.x0, pos_ax_3.y0, pos_ax_3.width*1.6,
                        pos_ax_3.height])
    ax[4].set_position([pos_ax_3.x0 + pos_ax_3.width*2.2, pos_ax_3.y0,
                        pos_ax_3.width*1.6, pos_ax_3.height])
    pos_ax_5 = ax[6].get_position()
    ax[6].set_position([pos_ax_5.x0, pos_ax_5.y0, pos_ax_5.width*1.6,
                        pos_ax_5.height])
    ax[7].set_position([pos_ax_5.x0 + pos_ax_5.width*2.2, pos_ax_5.y0,
                        pos_ax_5.width*1.6, pos_ax_5.height])
    ax_cohs = np.array([ax[1], ax[4], ax[7]])
    ax_zt = np.array([ax[0], ax[3], ax[6]])
    ax_inset = fp.add_inset(ax=ax_cohs[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_cohs = np.insert(ax_cohs, 2, ax_inset)
    ax_inset = fp.add_inset(ax=ax_zt[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_zt = np.insert(ax_zt, 2, ax_inset)
    ax_weights = ax[2]
    pos = ax_weights.get_position()
    ax_weights.set_position([pos.x0, pos.y0+pos.height/4, pos.width,
                             pos.height*1/2])
    for i_a, a in enumerate(ax):
        if i_a != 8:
            fp.rm_top_right_lines(a)

    df_trajs = df.copy()
    # TRAJECTORIES CONDITIONED ON PRIOR
    plots_trajs_conditioned(df=df_trajs.loc[df_trajs.special_trial == 2],
                            data_folder=data_folder,
                            ax=ax_zt, condition='choice_x_prior',
                            prior_limit=1, cmap='copper')
    # TRAJECTORIES CONDITIONED ON COH
    plots_trajs_conditioned(df=df_trajs, ax=ax_cohs,
                            data_folder=data_folder,
                            prior_limit=0.1,  # 10% quantile
                            condition='choice_x_coh',
                            cmap='coolwarm')
    plt.show()
    f.savefig(sv_folder+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/Fig2.svg', dpi=400, bbox_inches='tight')


def fig_2_trajs(df, rat_nocom_img, data_folder, sv_folder, fgsz=(8, 12),
                inset_sz=.1, marginx=-.1, marginy=0.1, subj='LE46'):
    f, ax = plt.subplots(4, 3, figsize=fgsz)
    letters = 'abcdeXfgXhij'
    ax = ax.flatten()
    for lett, a in zip(letters, ax):
        if lett != 'X':
            fp.add_text(ax=a, letter=lett, x=-0.1, y=1.2)
    ax[5].axis('off')
    ax[8].axis('off')
    # adjust panels positions
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.5, wspace=0.4)
    factor = 1.2
    for i_ax in [3, 4, 6, 7]:
        pos = ax[i_ax].get_position()
        if i_ax in [3, 6]:
            ax[i_ax].set_position([pos.x0, pos.y0, pos.width*factor,
                                    pos.height])
        else:
            ax[i_ax].set_position([pos.x0+pos.width/2, pos.y0, pos.width*factor,
                                    pos.height])

    ax = f.axes
    ax_zt = np.array([ax[3], ax[6]])
    ax_cohs = np.array([ax[4], ax[7]])
    ax_inset = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_cohs = np.insert(ax_cohs, 1, ax_inset)
    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_zt = np.insert(ax_zt, 1, ax_inset)
    ax_weights = ax[2]
    pos = ax_weights.get_position()
    ax_weights.set_position([pos.x0, pos.y0+pos.height/4, pos.width,
                             pos.height*1/2])
    for i_a, a in enumerate(ax):
        if i_a != 8:
            fp.rm_top_right_lines(a)
    margin = 0.05
    # TRACKING SCREENSHOT
    rat = plt.imread(rat_nocom_img)
    ax_scrnsht = ax[0]
    img = rat[150:646, 120:-10, :]
    ax_scrnsht.imshow(np.flipud(img)) # rat.shape = (796, 596, 4)
    ax_scrnsht.set_xticks([])
    right_port_y = 50
    center_port_y = 250
    left_port_y = 460
    ax_scrnsht.set_yticks([right_port_y, center_port_y, left_port_y])
    ax_scrnsht.set_yticklabels([-85, 0, 85])
    ax_scrnsht.set_ylim([0, img.shape[0]])
    ax_scrnsht.set_xlabel('x dimension (pixels)')
    ax_scrnsht.set_ylabel('y dimension (pixels)')
    ax_scrnsht.axhline(y=left_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(y=right_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(center_port_y, color='k', lw=.5)

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

    # TRAJECTORIES
    df_subj = df[df.subjid == subj]
    ax_rawtr = ax[1]
    ax_ydim = ax[2]
    ran_max = 100
    for tr in range(ran_max):
        if tr > (ran_max/2):
            trial = df_subj.iloc[tr]
            traj_x = trial['trajectory_x']
            traj_y = trial['trajectory_y']
            ax_rawtr.plot(traj_x, traj_y, color='grey', lw=.5, alpha=0.6)
            time = trial['time_trajs']
            ax_ydim.plot(time, traj_y, color='grey', lw=.5, alpha=0.6)
    x_lim = [-80, 20]
    y_lim = [-100, 100]
    ax_rawtr.set_xlim(x_lim)
    ax_rawtr.set_ylim(y_lim)
    ax_rawtr.set_xticklabels([])
    ax_rawtr.set_yticklabels([])
    ax_rawtr.set_xticks([])
    ax_rawtr.set_yticks([])
    ax_rawtr.set_xlabel('x dimension (pixels)')
    pos_rawtr = ax_rawtr.get_position()
    ax_rawtr.set_position([pos_rawtr.x0-margin, pos_rawtr.y0,
                           pos_rawtr.width/2, pos_rawtr.height])
    fp.add_text(ax=ax_rawtr, letter='rat LE46', x=0.7, y=1., fontsize=8)
    x_lim = [-100, 800]
    y_lim = [-100, 100]
    ax_ydim.set_xlim(x_lim)
    ax_ydim.set_ylim(y_lim)
    ax_ydim.set_yticks([])
    ax_ydim.set_xlabel('Time from movement onset (ms)')
    pos_ydim = ax_ydim.get_position()
    ax_ydim.set_position([pos_ydim.x0-3*margin, pos_rawtr.y0,
                          pos_ydim.width, pos_rawtr.height])
    fp.add_text(ax=ax_ydim, letter='rat LE46', x=0.32, y=1., fontsize=8)
    # plot dashed lines
    for i_a in [1, 2]:
        ax[i_a].axhline(y=85, linestyle='--', color='k', lw=.5)
        ax[i_a].axhline(y=-80, linestyle='--', color='k', lw=.5)
        ax[i_a].axhline(0, color='k', lw=.5)

    df_trajs = df.copy()
    # TRAJECTORIES CONDITIONED ON PRIOR
    plots_trajs_conditioned(df=df_trajs.loc[df_trajs.special_trial == 2],
                            ax=ax_zt, data_folder=data_folder,
                            condition='choice_x_prior',
                            prior_limit=1, cmap='copper')
    # TRAJECTORIES CONDITIONED ON COH
    plots_trajs_conditioned(df=df_trajs, ax=ax_cohs,
                            data_folder=data_folder,
                            condition='choice_x_coh',
                            prior_limit=0.1,  # 10% quantile
                            cmap='coolwarm')
    # SPLITTING TIME EXAMPLE
    ax_split = ax[9]
    pos = ax_split.get_position()
    ax_split.set_position([pos.x0, pos.y0, pos.width,
                           pos.height*2/5])
    ax_inset = plt.axes([pos.x0, pos.y0+pos.height*3/5, pos.width,
                         pos.height*2/5])
    axes_split = [ax_split, ax_inset]
    plot_trajs_splitting_example(df, ax=axes_split[1], rtbins=np.linspace(0, 15, 2),
                                 xlab=False)
    fp.rm_top_right_lines(axes_split[1])
    fp.rm_top_right_lines(axes_split[0])
    plot_trajs_splitting_example(df, ax=axes_split[0], rtbins=np.linspace(150, 300, 2),
                                 xlab=True)
    # TRAJECTORY SPLITTING PRIOR
    trajs_splitting_prior(df=df, ax=ax[11], data_folder=data_folder)
    # TRAJECTORY SPLITTING STIMULUS
    trajs_splitting_stim(df=df, data_folder=data_folder, ax=ax[10],
                         connect_points=True)
    f.savefig(sv_folder+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/Fig2.svg', dpi=400, bbox_inches='tight')


def supp_plot_trajs_dep_trial_index(df, data_folder):
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax = ax.flatten()
    for a in ax:
        fp.rm_top_right_lines(a)
    ax_ti = [ax[1], ax[0], ax[3], ax[2]]
    plots_trajs_conditioned(df, ax_ti, data_folder=data_folder,
                            condition='origidx', cmap='jet',
                            prior_limit=1, rt_lim=300,
                            after_correct_only=True,
                            trajectory="trajectory_y",
                            velocity=("traj_d1", 1),
                            acceleration=('traj_d2', 1), accel=False)

