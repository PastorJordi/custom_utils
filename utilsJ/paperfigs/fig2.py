import seaborn as sns
from utilsJ.Behavior import plotting, ComPipe
from scipy.interpolate import interp1d
from scipy.stats import sem
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
SAVPATH = '/home/jordi/Documents/changes_of_mind/changes_of_mind_scripts/PAPER/paperfigs/test_fig_scripts/' # where to save figs
SAVEPLOTS = True # whether to save plots or just show them

# util functions


def get_com_ty(row):  # to retrieve time and distance of each CoM
    row = row.copy()
    fps = (row['trajectory_stamps'][-1] - row['trajectory_stamps']
           [0]).astype(int) / (1000 * row.trajectory_stamps.size)

    fixtime = np.datetime64(row['fix_onset_dt'])
    peak_candidates = row['CoM_peakf']
    traj = row['trajectory_y']
    if len(peak_candidates) > 1:  # we know these are not 0s
        correct_peak = peak_candidates[np.where(
            np.abs(traj[peak_candidates]).max())[0]]
        yval = traj[correct_peak]

        if isinstance(row['trajectory_stamps'], np.ndarray):
            tval = (row['trajectory_stamps'][correct_peak] -
                    fixtime).astype(int) / 1000
        else:
            tval = 1000*correct_peak/fps
    else:
        yval = traj[peak_candidates]
        if isinstance(row['trajectory_stamps'], np.ndarray):
            tval = (row['trajectory_stamps'][peak_candidates] -
                    fixtime).astype(int) / 1000
        else:
            tval = 1000*peak_candidates/fps

    return [tval[0], yval[0]]  # tval will be in ms
# tiny one to ensure minor transformations are done? eg. allpriors, normallpriors ----


def a(
        df, average=False,
        rtbins=np.arange(0, 151, 3),
        evidence_bins=np.array([0, 0.15, 0.30, 0.60, 1.05]),
        savpath=SAVPATH):
    """calls tachometric defined in utilsJ.behavior.plotting"""
    # TODO adapt so proper mean/CI are shown when plotting average rat
    f, ax = plt.subplots(figsize=(9, 6))
    f.patch.set_facecolor('white')
    ax.set_facecolor('white')
    if not average:
        tacho_kws = dict(
            rtbins=rtbins,
            labels=[f'sstr {x}' for x in [0, .25, .5, 1]],
            fill_error=True
        )
        plotting.tachometric(df, ax=ax, **tacho_kws)
    else:
        rtbinsize = rtbins[1]-rtbins[0]
        df['rtbin'] = pd.cut(
            df.sound_len,
            rtbins,
            labels=np.arange(rtbins.size-1),
            retbins=False, include_lowest=True, right=True
        ).astype(float)

        evidence_bins = np.array([0, 0.15, 0.30, 0.60, 1.05])
        df['sstr'] = 0
        for i, sst in enumerate([0, .25, .5, 1]):
            df.loc[
                (df.coh2.abs() >= evidence_bins[i])
                & (df.coh2.abs() < evidence_bins[i+1]),
                'sstr'] = sst
        cmap = cm.get_cmap('inferno')
        tmp = (df.groupby(['subjid', 'sstr', 'rtbin'])['hithistory'].mean()
               .reset_index()
               .groupby(['sstr', 'rtbin'])['hithistory'].agg(['mean', sem])
               )
        for i, sst in enumerate(tmp.index.get_level_values(0).unique()):
            ax.plot(
                tmp.loc[sst, 'mean'].index * rtbinsize + 0.5 * rtbinsize,
                tmp.loc[sst, 'mean'],
                marker='o', color=cmap(i/(evidence_bins.size-1))
            )
            ax.fill_between(
                tmp.loc[sst, 'mean'].index * rtbinsize + 0.5 * rtbinsize,
                tmp.loc[sst, 'mean'] + tmp.loc[sst, 'sem'],
                y2=tmp.loc[sst, 'mean'] - tmp.loc[sst, 'sem'],
                alpha=0.3, color=cmap(i/(evidence_bins.size-1))
            )

    sns.despine(ax=ax, trim=True)
    ax.set_ylabel('accuracy', fontsize=14)
    ax.set_xlabel('RT (ms)', fontsize=14)

    if SAVEPLOTS:
        f.savefig(f'{savpath}2a_tachometric.svg')
    plt.show()


def bcd(parentpath, sv_folder=None):
    sv_folder = sv_folder or SAVPATH
    portspng = '/home/molano/Dropbox/project_Barna/' +\
    'ChangesOfMind/figures/Figure_3/ports.png'
    ratcompng = '/home/molano/Dropbox/project_Barna/' +\
        'ChangesOfMind/figures/Figure_3/001965.png'
    img = plt.imread(portspng)
    rat = plt.imread(ratcompng)
    fig, ax = plt.subplots(ncols=3, figsize=(18, 5.5), gridspec_kw={
                           'width_ratios': [1, 1, 1.8]})
    fig.patch.set_facecolor('white')
    #ax[1] = plt.subplot2grid((2,2), (0,0), rowspan=2)
    for i in range(3):
        ax[i].set_facecolor('white')
    ax[0].imshow(np.flipud(rat))
    ax[1].imshow(img)
    for i in [0, 1]:
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xlabel('x dimension (pixels)', fontsize=14)
        ax[i].set_ylabel('y dimension (pixels)', fontsize=14)

    ax[0].set_xlim(435, 585)
    ax[0].set_ylim(130, 330)
    ax[1].set_xlim(450, 600)
    ax[1].set_ylim(100, 300)

    # a = ComPipe.chom('LE37', parentpath=parentpath)
    # # a.load_available()
    # # load and preprocess data
    # a.load('LE37_p4_u_20190330-150513')
    # a.process(normcoords=True)
    # a.get_trajectories()
    # a.suggest_coms()
    # a.trial_sess['t_com'] = [[np.nan, np.nan]]*len(a.trial_sess)
    # a.trial_sess['y_com'] = np.nan
    # a.trial_sess.loc[
    #     (a.trial_sess.resp_len < 2) & (a.trial_sess.CoM_sugg == True),
    #     't_com'
    # ] = a.trial_sess.loc[
    #     (a.trial_sess.resp_len < 2) & (a.trial_sess.CoM_sugg == True)
    # ].apply(lambda x: get_com_ty(x), axis=1)  # peak is just available for CoM trials# 'framerate'
    # a.trial_sess[['t_com', 'y_com']] = pd.DataFrame(
    #     a.trial_sess['t_com'].values.tolist(), index=a.trial_sess.index)

    # a.trial_sess['ftraj'] = np.nan
    # a.trial_sess['ftraj'] = a.trial_sess.trajectory_y.apply(lambda x: len(x))

    # for i in range(205, 220):  # 235):

    #     if a.trial_sess.dirty[i] or ((a.trial_sess.CoM_sugg[i] == True) & (a.trial_sess.y_com.abs()[i] < 7)):
    #         print(f'skipping {i}')
    #         continue

    #     if a.trial_sess.R_response[i]:
    #         toappendx, toappendy = [-20], [75]
    #     else:
    #         toappendx, toappendy = [-20], [-75]
            
    #     # c_resp_len = a.trial_sess.resp_len[i]
    #     c_sound_len = a.trial_sess.sound_len[i]
    #     tvec = (
    #         a.trial_sess.trajectory_stamps[i] -
    #         np.datetime64(a.trial_sess.fix_onset_dt[i])
    #     ).astype(int) / 1000
    #     x, y = a.trial_sess.trajectory_x[i], a.trial_sess.trajectory_y[i]
    #     #print(x.size, y.size, tvec.size)
    #     f = interp1d(tvec, np.c_[x, y], kind='cubic',
    #                  axis=0, fill_value=np.nan, bounds_error=False)

    #     tvec_new = np.linspace(
    #         0, 30 + 300+a.trial_sess.sound_len[i]+a.trial_sess.resp_len[i] * 1000, 100)

    #     itraj = f(tvec_new)
    #     if a.trial_sess.CoM_sugg[i]:
    #         #print(f'CoM in index {i}')
    #         # print(itraj.shape)
    #         kws = dict(color='r', alpha=1, zorder=3)
    #     else:
    #         kws = dict(color='tab:blue', alpha=0.8)

    #     # aligned to RT
    #     movement_offset = -300 - c_sound_len
    #     #ax[0].plot(tvec_new + movement_offset, itraj[:,0], **kws)
    #     ax[2].plot(tvec_new + movement_offset, itraj[:, 1], **kws)

    #     ax[1].plot(itraj[:, 0]+530, itraj[:, 1]+200, **kws)  # add offset

    # ax[2].axvline(0, color='k', ls=':', label='response onset')
    # ax[2].legend()

    # ax[2].set_xlabel('time', fontsize=14)
    # ax[2].set_ylabel('y dimension (pixels)', fontsize=14)
    # ax[2].set_xticks([])
    if SAVEPLOTS:
        fig.savefig(f'{sv_folder}2bcd_ports_and_traj.svg')
    plt.show()


def e(df, ax, average=False, rtbins= np.arange(0,201,10), sv_folder=None, dist=False):
    """p(com) and RT distribution"""
    sv_folder = sv_folder or SAVPATH
    if not average:
        plotting.binned_curve(
            df[df.special_trial == 0],
            'CoM_sugg',
            'sound_len',
            rtbins,
            sem_err=False,
            legend=False,
            xpos=10,  # np.arange(5,201,10),
            xoffset=5,
            errorbar_kw={'color': 'tab:orange',
                         'label': 'p(CoM)', 'zorder': 3},
            traces='subjid',
            traces_kw=dict(alpha=0.3), ax=ax
        )
    else:  # mean of means + sem
        rtbinsize = rtbins[1]-rtbins[0]
        df['rtbin'] = pd.cut(
            df.sound_len,
            rtbins,
            labels=np.arange(rtbins.size-1),
            retbins=False, include_lowest=True, right=True
        ).astype(float)

        f, ax = plt.subplots()
        # traces
        tmp = (
            df.groupby(['subjid', 'rtbin'])['CoM_sugg'].mean()
            .reset_index()
        )
        for subject in tmp.subjid.unique():
            ax.plot(
                tmp.loc[tmp.subjid == subject, 'rtbin'] *
                rtbinsize + 0.5 * rtbinsize,
                tmp.loc[tmp.subjid == subject, 'CoM_sugg'],
                ls=':', color='gray'
            )
        # average
        tmp = tmp.groupby(['rtbin'])['CoM_sugg'].agg(['mean', sem])
        ax.errorbar(
            tmp.index * rtbinsize + 0.5 * rtbinsize,
            tmp['mean'],
            yerr=tmp['sem'],
            label='p(CoM)', color='tab:orange'
        )
    ax.set_ylim(0, 0.075)
    if dist:
        hist_list = []
        for subject in df.subjid.unique():
            counts, bns = np.histogram(df[(df.subjid == subject) & (
                df.special_trial == 0)].sound_len.dropna().values, bins=rtbins)
            hist_list += [counts]
        ax.set_ylim(0, 0.075)
        _, ymax = ax.get_ylim()
        counts = np.stack(hist_list).mean(axis=0)
        ax.hist(bns[:-1], bns, weights=0.5*ymax * counts /
                counts.max(), alpha=.4, label='RT distribution')
    ax.set_xlabel('Reaction Time (ms)')
    ax.legend(fancybox=False, frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_bounds(0, 0.075)
    ax.spines['bottom'].set_bounds(0, 200)
    plt.gcf().patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
    if SAVEPLOTS:
        plt.gcf().savefig(f'{sv_folder}2e_Pcom_and_RT_distr.svg')
    plt.show()

def f(df, average=False, sv_folder=None):
    """com matrix and marginal axes
    be sure to pre-filter df (silent trials etc.)"""
    sv_folder = sv_folder or SAVPATH
    if 'allpriors' not in df.columns:
        nanidx = df.loc[df[['dW_trans', 'dW_lat']
                           ].isna().sum(axis=1) == 2].index
        df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
        df.loc[nanidx, 'allpriors'] = np.nan
    if 'norm_allpriors' not in df.columns:
        df['norm_allpriors'] = (
            df.allpriors
            # perhaps should it be per subject x sesstype (ie x glm)?
            / df.groupby('subjid').allpriors.transform(lambda x: np.max(np.abs(x)))
        )

    f, _ = plotting.com_heatmap_paper_marginal_pcom_side(
        df.loc[df.special_trial == 0],
        hide_marginal_axis=False, nbins=10, average_across_subjects=average
    )
    if SAVEPLOTS:
        f.savefig(f'{sv_folder}2f_Pcom_matrix_L.svg')
    plt.show()

def g(df, average=False, sv_folder=None):
    """com matrix and marginal axes
    be sure to pre-filter df (silent trials etc.)"""
    sv_folder = sv_folder or SAVPATH
    if 'allpriors' not in df.columns:
        nanidx = df.loc[df[['dW_trans', 'dW_lat']
                           ].isna().sum(axis=1) == 2].index
        df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
        df.loc[nanidx, 'allpriors'] = np.nan
    if 'norm_allpriors' not in df.columns:
        df['norm_allpriors'] = (
            df.allpriors
            / df.groupby('subjid').allpriors.transform(lambda x: np.max(np.abs(x)))
        )
    f, _ = plotting.com_heatmap_paper_marginal_pcom_side(
        df.loc[(df.special_trial == 0)],
        hide_marginal_axis=False, nbins=10, side=1, average_across_subjects=average
    )
    if SAVEPLOTS:
        f.savefig(f'{sv_folder}2g_Pcom_matrix_R.svg')
    plt.show()
