# from utilsJ.regularimports import *
from utilsJ.Behavior import plotting
from utilsJ.Models import simul
from matplotlib import cm
from scipy.stats import sem
import numpy as np
import matplotlib.pyplot as plt


# where to save figs
SAVPATH = '/home/jordi/Documents/changes_of_mind/changes_of_mind_scripts/' +\
    'PAPER/paperfigs/test_fig_scripts/'
SAVEPLOTS = False  # whether to save plots or just show them

# TODO: how to merge them weigthed?
# mean and SEMs


def trajs_cond_on_prior(df, average=False, trajectory="trajectory_y",
                        velocity=("traj_d1", 1), savpath=SAVPATH):
    """median position and velocity in silent trials splitting by prior"""
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    df['choice_x_prior'] = (df.R_response*2-1) * df.allpriors

    if not average:
        for subject in df.subjid.unique():  # dani_rats:
            f, ax = plt.subplots(ncols=2, nrows=2, figsize=(11, 8),
                                 gridspec_kw={'width_ratios': [1, 3]})
            ax = ax.flatten()
            sil_tr_indx = (df.subjid == subject) & (df.special_trial == 2)
            xpoints, ypoints, _, mat, dic =\
                plotting.trajectory_thr(df.loc[sil_tr_indx],
                                        'choice_x_prior',
                                        np.linspace(-3, 3, 6), collapse_sides=True,
                                        thr=30, ax=ax[0], ax_traj=ax[1],
                                        return_trash=True,
                                        error_kwargs=dict(marker='o'),
                                        cmap='viridis',
                                        trajectory=trajectory)
            ax[1].legend(labels=['-2', '-1', '0', '1', '2'],
                         title='Prior')
            ax[1].set_xlim([-50, 500])
            ax[1].set_xlabel('time from movement onset (MT, ms)')
            for i in [0, 30]:
                ax[1].axhline(i, ls=':', c='gray')
            ax[1].set_ylabel('y coord. (px)')
            ax[0].set_xlabel('prior towards response')
            ax[0].set_ylabel('time to threshold (30px)')
            ax[0].plot(xpoints, ypoints, color='k', ls=':')

            ax[1].set_ylim([-10, 80])
            threshold = .2
            xpoints, ypoints, _, mat, dic = plotting.trajectory_thr(
                df.loc[(df.subjid == subject) & (df.special_trial == 2)],
                'choice_x_prior', np.linspace(-3, 3, 6), collapse_sides=True,
                thr=threshold, ax=ax[2], ax_traj=ax[3], return_trash=True,
                error_kwargs=dict(marker='o'), cmap='viridis',
                trajectory=velocity)  # TODO ACCELY col name?
            ax[3].legend(labels=['-2', '-1', '0', '1', '2'],
                         title='Prior')

            ax[3].set_xlim([-50, 500])
            ax[3].set_xlabel('time from movement onset (MT, ms)')
            ax[3].set_ylim([-0.05, 0.5])
            for i in [0, threshold]:
                ax[3].axhline(i, ls=':', c='gray')
            ax[3].set_ylabel('y coord velocity (px/ms)')
            ax[2].set_xlabel('prior towards response')
            ax[2].set_ylabel(f'time to threshold ({threshold} px/ms)')
            ax[2].plot(xpoints, ypoints, color='k', ls=':')
            if SAVEPLOTS:
                f.savefig(f'{savpath}{subject}_median_traj_silent_prior.svg')
            plt.show()
    else:
        f, ax = plt.subplots(ncols=2, nrows=2, figsize=(
            11, 8), gridspec_kw={'width_ratios': [1, 3]})
        ax = ax.flatten()
        cmap = cm.get_cmap('viridis')
        median_list_pose = []
        median_list_vel = []
        mean_thr_pose = []
        mean_thr_vel = []
        for subject in df.subjid.unique():  # dani_rats:
            # position
            xpoints, ypoints1, _, mat, dic = plotting.trajectory_thr(
                df.loc[(df.subjid == subject) & (df.special_trial == 2)],
                'choice_x_prior', np.linspace(-3, 3, 6), collapse_sides=True,
                thr=30, ax=None, ax_traj=None, return_trash=True,
                error_kwargs=dict(marker='o'), cmap='viridis',
                trajectory=trajectory)
            median_mat1 = np.zeros((5, 1700)) * np.nan

            for i, m in mat.items():
                median_mat1[i] = np.nanmedian(m, axis=0)
            median_list_pose += [median_mat1]
            mean_thr_pose += [ypoints1]

            # velocity
            xpoints, ypoints2, _, mat, dic = plotting.trajectory_thr(
                df.loc[(df.subjid == subject) & (df.special_trial == 2)],
                'choice_x_prior', np.linspace(-3, 3, 6), collapse_sides=True,
                thr=.2, ax=None, ax_traj=None, return_trash=True,
                error_kwargs=dict(marker='o'), cmap='viridis',
                trajectory=velocity)  # TODO ACCELY col name?
            median_mat2 = np.zeros((5, 1700)) * np.nan
            for i, m in mat.items():
                median_mat2[i] = np.nanmedian(m, axis=0)

            median_list_vel += [median_mat2]
            mean_thr_vel += [ypoints2]

        median_list_pose = np.nanmedian(np.stack(median_list_pose), axis=0)
        median_list_vel = np.nanmedian(np.stack(median_list_vel), axis=0)
        mean_thr_pose = np.stack(mean_thr_pose)  # .mean(axis=0)
        mean_thr_vel = np.stack(mean_thr_vel)  # .mean(axis=0)

        error_kws = dict(marker='o', capsize=2, ls=':')
        for i in range(5):  # hardcoded number
            ccolor = cmap(i/4)
            ax[0].errorbar(
                xpoints[i], mean_thr_pose.mean(axis=0)[i],
                yerr=sem(mean_thr_pose, axis=0)[i],
                color=ccolor, **error_kws)
            ax[2].errorbar(
                xpoints[i], mean_thr_vel.mean(axis=0)[i],
                yerr=sem(mean_thr_vel, axis=0)[i],
                color=ccolor, **error_kws)

            ax[1].plot(np.arange(1700)-700, median_list_pose[i], color=ccolor)
            ax[3].plot(np.arange(1700)-700, median_list_vel[i], color=ccolor)

        ax[0].plot(xpoints, mean_thr_pose.mean(
            axis=0), ls=':', c='k', zorder=0)
        ax[2].plot(xpoints, mean_thr_vel.mean(axis=0), ls=':', c='k', zorder=0)
        ax[1].set_xlim([-50, 500])
        ax[1].set_xlabel('time from movement onset (MT, ms)')
        for i in [0, 30]:
            ax[1].axhline(i, ls=':', c='gray')
        ax[1].set_ylabel('y coord. (px)')
        ax[0].set_xlabel('prior towards response')
        ax[0].set_ylabel('time to threshold (30px)')

        ax[1].set_ylim([-10, 80])
        threshold = .2
        ax[3].set_xlim([-50, 500])
        ax[3].set_xlabel('time from movement onset (MT, ms)')
        ax[3].set_ylim([-0.05, 0.6])
        for i in [0, threshold]:
            ax[3].axhline(i, ls=':', c='gray')
        ax[3].set_ylabel('y coord velocity (px/ms)')
        ax[2].set_xlabel('prior towards response')
        ax[2].set_ylabel(f'time to threshold ({threshold} px/ms)')

        if SAVEPLOTS:
            f.savefig(f'{savpath}3a_average_median_traj_silent_prior.svg')
        plt.show()


def trajs_cond_on_coh(df, average=False, prior_limit=0.25, rt_lim=25,
                      after_correct_only=True, trajectory="trajectory_y",
                      velocity=("traj_d1", 1), savpath=SAVPATH):
    """median position and velocity in silent trials splitting by prior"""
    # TODO: adapt for mean + sem
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
    bins = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]

    if not average:
        # dani_rats: # this is with sound, it can use all sunbjects data
        for subject in df.subjid.unique():
            f, ax = plt.subplots(ncols=2, nrows=2, figsize=(
                11, 8), gridspec_kw={'width_ratios': [1, 3]})
            ax = ax.flatten()
            if after_correct_only:
                ac_cond = df.aftererror == False
            else:
                ac_cond = (df.aftererror*1) >= 0
            indx_trajs = (df.subjid == subject) &\
                (df.allpriors.abs() < prior_limit) &\
                ac_cond & (df.special_trial == 0) &\
                (df.sound_len < rt_lim)
            xpoints, ypoints, _, mat, dic, mt_time, mt_time_err =\
                plotting.trajectory_thr(df.loc[indx_trajs],
                                        'choice_x_coh', bins, collapse_sides=True,
                                        thr=30, ax=ax[0], ax_traj=ax[1],
                                        return_trash=True,
                                        error_kwargs=dict(marker='o'),
                                        cmap='viridis',
                                        bintype='categorical',
                                        trajectory=trajectory)
            ax[1].legend(labels=['-1', '-0.5', '-0.25', '0', '0.25', '0.5', '1'],
                         title='Coherence')
            ax[1].set_xlim([-50, 500])
            ax[1].set_xlabel('time from movement onset (MT, ms)')
            for i in [0, 30]:
                ax[1].axhline(i, ls=':', c='gray')
            ax[1].set_ylabel('y coord. (px)')
            ax[0].set_xlabel('ev. towards response')
            ax[0].set_ylabel('time to threshold (30px)')
            ax[0].plot(xpoints, ypoints, color='k', ls=':')
            ax[1].set_ylim([-10, 80])
            ax2 = ax[0].twinx()
            ax2.errorbar(xpoints, mt_time, mt_time_err, color='c', ls=':')
            threshold = .2
            xpoints, ypoints, _, mat, dic, _, _ = plotting.trajectory_thr(
                df.loc[(df.subjid == subject) & (df.allpriors.abs() < prior_limit)
                       & ac_cond & (df.special_trial == 0)
                       & (df.sound_len < rt_lim)],
                'choice_x_coh', bins, collapse_sides=True,
                thr=threshold, ax=ax[2], ax_traj=ax[3], return_trash=True,
                error_kwargs=dict(marker='o'), cmap='viridis',
                bintype='categorical', trajectory=velocity)  # XXX ACCELY col name?
            ax[3].legend(labels=['-1', '-0.5', '-0.25', '0', '0.25', '0.5', '1'],
                         title='Coherence')
            ax[3].set_xlim([-50, 500])
            ax[3].set_xlabel('time from movement onset (MT, ms)')
            ax[3].set_ylim([-0.05, 0.5])
            for i in [0, threshold]:
                ax[3].axhline(i, ls=':', c='gray')
            ax[3].set_ylabel('y coord velocity (px/ms)')
            ax[2].set_xlabel('ev. towards response')
            ax[2].set_ylabel(f'time to threshold ({threshold} px/ms)')
            ax[2].plot(xpoints, ypoints, color='k', ls=':')
            if SAVEPLOTS:
                f.savefig(f'{savpath}{subject}_traj_coherence_lowprior.svg')
            plt.show()

    else:
        # raise NotImplementedError('need to rewrite and support diff. coherences')
        # use dfmask instead of categories in trajectory_thr()
        f, ax = plt.subplots(ncols=2, nrows=2, figsize=(
            11, 8), gridspec_kw={'width_ratios': [1, 3]})
        ax = ax.flatten()

        cmap = cm.get_cmap('viridis')
        median_list_pose = []
        median_list_vel = []
        mean_thr_pose = []
        mean_thr_vel = []

        ax[1].set_xlim([-50, 500])
        ax[1].set_xlabel('time from movement onset (MT, ms)')
        for i in [0, 30]:
            ax[1].axhline(i, ls=':', c='gray')
        ax[1].set_ylabel('y coord. (px)')
        ax[0].set_xlabel('ev. towards response')
        ax[0].set_ylabel('time to threshold (30px)')

        ax[1].set_ylim([-10, 80])
        ax[3].set_xlim([-50, 500])
        ax[3].set_xlabel('time from movement onset (MT, ms)')
        ax[3].set_ylim([-0.05, 0.6])
        for i in [0, .2]:
            ax[3].axhline(i, ls=':', c='gray')
        ax[3].set_ylabel('y coord velocity (px/ms)')
        ax[2].set_xlabel('ev. towards response')
        ax[2].set_ylabel(f'time to threshold (0.2 px/ms)')
        # dani_rats: # this is with sound, it can use all sunbjects data
        for subject in df.subjid.unique():
            if after_correct_only:
                ac_cond = df.aftererror == False
            else:
                ac_cond = (df.aftererror*1) >= 0

            xpoints, ypoints1, _, mat, dic = plotting.trajectory_thr(
                df.loc[
                    (df.subjid == subject)
                    & (df.allpriors.abs() < prior_limit)
                    & ac_cond
                    & (df.special_trial == 0)
                    & (df.sound_len < rt_lim)
                ],
                'choice_x_coh', bins, collapse_sides=True,
                thr=30, ax=None, ax_traj=None, return_trash=True,
                error_kwargs=dict(marker='o'), cmap='viridis',
                bintype='categorical',
                trajectory=trajectory)
            median_mat1 = np.zeros((7, 1700)) * np.nan

            for i, m in mat.items():
                median_mat1[i] = np.nanmedian(m, axis=0)
            median_list_pose += [median_mat1]
            mean_thr_pose += [ypoints1]

            xpoints, ypoints2, _, mat, dic = plotting.trajectory_thr(
                df.loc[
                    (df.subjid == subject)
                    & (df.allpriors.abs() < prior_limit)
                    & (df.aftererror == False)
                    & (df.special_trial == 0)
                    & (df.sound_len < rt_lim)
                ],
                'choice_x_coh', bins, collapse_sides=True,
                thr=.2, ax=None, ax_traj=None, return_trash=True,
                error_kwargs=dict(marker='o'), cmap='viridis',
                bintype='categorical',
                trajectory=velocity)  # TODO ACCELY col name?

            median_mat2 = np.zeros((7, 1700)) * np.nan
            for i, m in mat.items():
                median_mat2[i] = np.nanmedian(m, axis=0)

            median_list_vel += [median_mat2]
            mean_thr_vel += [ypoints2]

        # merge results
        median_list_pose = np.nanmedian(np.stack(median_list_pose), axis=0)
        median_list_vel = np.nanmedian(np.stack(median_list_vel), axis=0)
        mean_thr_pose = np.stack(mean_thr_pose)  # .mean(axis=0)
        mean_thr_vel = np.stack(mean_thr_vel)  # .mean(axis=0)

        error_kws = dict(marker='o', capsize=2, ls=':')
        for i in range(7):  # hardcoded number
            ccolor = cmap(i/6)
            ax[0].errorbar(
                xpoints[i], mean_thr_pose.mean(axis=0)[i],
                yerr=sem(mean_thr_pose, axis=0)[i],
                color=ccolor, **error_kws)
            ax[2].errorbar(
                xpoints[i], mean_thr_vel.mean(axis=0)[i],
                yerr=sem(mean_thr_vel, axis=0)[i],
                color=ccolor, **error_kws)

            ax[1].plot(np.arange(1700)-700, median_list_pose[i], color=ccolor)
            ax[3].plot(np.arange(1700)-700, median_list_vel[i], color=ccolor)

        ax[0].plot(xpoints, mean_thr_pose.mean(
            axis=0), ls=':', c='k', zorder=0)
        ax[2].plot(xpoints, mean_thr_vel.mean(axis=0), ls=':', c='k', zorder=0)

        if SAVEPLOTS:
            f.savefig(f'{savpath}3b_average_traj_coherence_lowprior.svg')
        plt.show()

# 3c: split with average traj x2 example RTs
#


def trajs_splitting(df, average=False, collapse_sides=False, savpath=SAVPATH):
    # dual, when are rts{0-25} and rts{100-125} splitting?
    if not average:
        for subject in df.subjid.unique():

            if not collapse_sides:
                f, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 12))
                # split_time_0_L =
                simul.when_did_split_dat(df=df[df.subjid == subject], side=0,
                                         ax=ax[0],
                                         plot_kwargs=dict(color='tab:green'))
                # split_time_0_R =
                simul.when_did_split_dat(df=df[df.subjid == subject], side=1,
                                         ax=ax[0],
                                         plot_kwargs=dict(color='tab:purple'))
                # split_time_4_L =
                simul.when_did_split_dat(df=df[df.subjid == subject], side=0,
                                         ax=ax[1], rtbin=4,
                                         plot_kwargs=dict(color='tab:green'))
                # split_time_4_R =
                simul.when_did_split_dat(df=df[df.subjid == subject], side=1,
                                         ax=ax[1], rtbin=4,
                                         plot_kwargs=dict(color='tab:purple'))
                ax[0].set_xlim(-10, 150)
                ax[0].set_title('RT {0, 25} ms')
                ax[1].set_title('RT {100, 125} ms')
                ax[1].set_xlabel('time from movement onset (ms)')
                ax[0].set_ylabel('y dimension (px)')
                ax[1].set_ylabel('y dimension (px)')
            else:  # collapse_sides == True
                f, ax = plt.subplots(figsize=(6, 6))
                # split_time_0 =
                simul.when_did_split_dat(df=df[df.subjid == subject], side=0,
                                         collapse_sides=True, ax=ax,
                                         plot_kwargs=dict(color='tab:green',
                                                          label='rtbin=0'))
                # split_time_4 =
                simul.when_did_split_dat(df=df[df.subjid == subject], side=1,
                                         ax=ax, collapse_sides=True, rtbin=4,
                                         plot_kwargs=dict(color='tab:blue',
                                                          label='rtbin=4'))
                ax.set_xlim(-10, 150)
                ax.set_xlabel('time from movement onset (ms)')
                ax.set_ylabel('y dimension (px)')
                ax.set_title(subject)
            if not collapse_sides:
                figname = f'{savpath}{subject}3c_split_median_traj.svg'
            else:
                figname = f'{savpath}{subject}3c_split_median_traj_collapsed.svg'
            if SAVEPLOTS:
                f.savefig(figname)
            plt.show()
    else:
        raise NotImplementedError('averaging is not implemented')


def trajs_splitting_point(df, collapse_sides=False, rtbins=np.linspace(0, 150, 16),
                          connect_points=True, draw_line=((0, 90), (90, 0)),
                          threshold=300, trajectory="trajectory_y",
                          savpath=SAVPATH):

    # split time/subject by coherence
    # threshold= bigger than that are turned to nan so it doesnt break figure range
    # this wont work if when_did_split_dat returns Nones instead of NaNs
    # plot will not work fine with uneven bins
    out_data = []
    for subject in df.subjid.unique():
        for i in range(rtbins.size-1):
            if collapse_sides:
                current_split_index = simul.when_did_split_dat(
                    df.loc[(df.special_trial == 0) & (df.subjid == subject)],
                    0,  # side has no effect because this is collapsing_sides
                    rtbin=i, rtbins=rtbins,
                    collapse_sides=True,
                    trajectory=trajectory
                )
                out_data += [current_split_index]
            else:
                for j in [0, 1]:  # side values
                    current_split_index = simul.when_did_split_dat(
                        df.loc[df.subjid == subject],
                        j,  # side has no effect because this is collapsing_sides
                        rtbin=i, rtbins=rtbins,
                        collapse_sides=True,
                        trajectory=trajectory
                    )
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

    f, ax = plt.subplots(figsize=(8, 6))
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
                    binsize/2 + binsize * np.arange(rtbins.size-1)
                    + np.random.normal(loc=0, scale=0.2, size=rtbins.size-1),
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
    if SAVEPLOTS:
        f.savefig(f'{savpath}3d_split_times.svg')
    plt.show()
# 3d histogram-like*?
