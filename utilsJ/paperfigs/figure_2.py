import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib as mtp
from matplotlib.lines import Line2D
from scipy.stats import sem, ttest_1samp
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit as cfit
import sys
sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel
from utilsJ.paperfigs import figures_paper as fp
from utilsJ.Behavior.plotting import trajectory_thr, interpolapply

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
        col_face = colormap[i_tr].copy()
        col_face[-1] = 0.2
        col_edge = [0, 0, 0, 0]
        traj -= np.nanmean(traj[(interpolatespace > -100000) * (interpolatespace < 0)])
        ax[0].plot(interpolatespace/1000, traj, color=colormap[i_tr])
        ax[0].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr], facecolor=col_face,
                           edgecolor=col_edge)
    ax[1].set_ylim([0.5, 0.8])
    # plt.show()
    if condition == 'choice_x_coh':
        legendelements = [Line2D([0], [0], color=colormap[0], lw=2, label='-1'),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[2], lw=2, label=''),
                          Line2D([0], [0], color=colormap[3], lw=2, label='0'),
                          Line2D([0], [0], color=colormap[4], lw=2, label=''),
                          Line2D([0], [0], color=colormap[5], lw=2, label=''),
                          Line2D([0], [0], color=colormap[6], lw=2, label='1')]
        title = 'Stimulus'
    if condition == 'choice_x_prior':
        legendelements = [Line2D([0], [0], color=colormap[4], lw=2,
                                 label='congr.'),
                          Line2D([0], [0], color=colormap[3], lw=2,
                                 label=''),
                          Line2D([0], [0], color=colormap[2], lw=2,
                                 label='0'),
                          Line2D([0], [0], color=colormap[1], lw=2, label=''),
                          Line2D([0], [0], color=colormap[0], lw=2,
                                 label='incongr.')]
        title = 'Prior'
    if condition == 'origidx':
        legendelements = []
        labs = ['1-200', '201-400', '401-600', '601-800', '801-1000']
        for i in range(len(colormap)):
            legendelements.append(Line2D([0], [0], color=colormap[i], lw=2,
                                  label=labs[i]))
        title = 'Trial index'
        ax[1].set_xlabel('Trial index')
    ax[0].legend(handles=legendelements, loc='upper left', title=title,
                labelspacing=.1, bbox_to_anchor=(0., 1.3))
    ax[1].set_xlabel(title)
    ax[0].set_xlim([-20, 450])
    ax[0].set_xticklabels('')
    ax[0].axhline(0, c='gray')
    ax[0].set_ylabel('Position')
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
        col_face = colormap[i_tr].copy()
        col_face[-1] = 0.2
        col_edge = [0, 0, 0, 0]
        traj -= np.nanmean(traj[(interpolatespace > -100000) * (interpolatespace < 0)])
        ax[2].plot(interpolatespace/1000, traj, color=colormap[i_tr])
        ax[2].fill_between(interpolatespace/1000, traj-all_trajs_err[i_tr],
                           traj+all_trajs_err[i_tr],
                           facecolor=col_face, edgecolor=col_edge)
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
    ax[1].set_ylabel('Peak')
    ax[2].set_ylim([-0.05, 0.5])
    ax[2].axhline(0, c='gray')
    ax[2].set_ylabel('Velocity')
    ax[2].set_xlabel('Time from movement onset (ms)')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].plot(xpoints, mt_time, color='k', ls='-', lw=0.5)


def get_split_ind_corr_frames(mat, stim, pval=0.01, max_MT=400, startfrom=700):
    idx_1 = np.nan
    i1 = True
    idx_2 = np.nan
    i2 = True
    idx_3 = np.nan
    i3 = True
    w1 = []
    w2 = []
    w3 = []
    for i in reversed(range(max_MT)):  # reversed so it goes backwards in time
        pop_a = mat[:, startfrom + i]
        nan_idx = ~np.isnan(pop_a)
        pop_evidence = stim[:, nan_idx]
        if i < 100:
            pop_evidence = stim[:2, nan_idx]
        if i < 50:
            pop_evidence = stim[0, nan_idx]
        pop_a = pop_a[nan_idx]
        mod = sm.OLS(pop_a.T, pop_evidence.T).fit()
        # sm.add_constant(pop_evidence.T)
        p2 = mod.pvalues
        # params = mod.params
        w1.append(p2[0])
        if i < 50 and stim.shape[0] > 1:
            w2.append(np.nan)
        if i > 50 and stim.shape[0] > 1:
            w2.append(p2[1])
        if stim.shape[0] == 1:
            w2.append(np.nan)
        if i < 100 and stim.shape[0] > 2:
            w3.append(np.nan)
        if i > 100 and stim.shape[0] > 2:
            w3.append(p2[2])
        if stim.shape[0] <= 2:
            w3.append(np.nan)
        if len(p2) == 1 and p2 > pval and i1:
            idx_1 = i
            i1 = False
        if len(p2) > 1:
            if p2[0] > pval and i1:
                idx_1 = i
                i1 = False
            if p2[1] > pval and i2:
                idx_2 = i
                i2 = False
        if len(p2) > 2:
            if p2[2] > pval and i3:
                idx_3 = i
                i3 = False
        if i1 + i2 + i3 == 0:
            break
    return [idx_1, idx_2, idx_3]


def get_params_lin_reg_frames(mat, stim, max_MT=400, startfrom=700):
    w1 = []
    w2 = []
    w3 = []
    for i in reversed(range(max_MT)):  # reversed so it goes backwards in time
        pop_a = mat[:, startfrom + i]
        nan_idx = ~np.isnan(pop_a)
        pop_evidence = stim[:, nan_idx]
        if i < 100:
            pop_evidence = stim[:2, nan_idx]
        if i < 50:
            pop_evidence = stim[0, nan_idx]
        pop_a = pop_a[nan_idx]
        mod = sm.OLS(pop_a.T, pop_evidence.T).fit()
        # sm.add_constant(pop_evidence.T)
        params = mod.params
        w1.append(params[0])
        if i <= 50 and stim.shape[0] > 1:
            w2.append(0)
        if i > 50 and stim.shape[0] > 1:
            w2.append(params[1])
        if stim.shape[0] == 1:
            w2.append(0)
        if i <= 100 and stim.shape[0] > 2:
            w3.append(0)
        if i > 100 and stim.shape[0] > 2:
            w3.append(params[2])
        if stim.shape[0] <= 2:
            w3.append(0)
    return [w1, w2, w3]


def get_dv_from_params_ttest(params, max_MT=400, pval=0.01):
    """
    params is a matrix with N_subjects rows and 400 columns, corresponding
    to timepoints (and it has 3 "channels" corresponding to each weight).
    It has values of the linear regression of the trajectory
    with the stimulus frames:
        y ~ beta_1 * frame_1 + beta_2 * frame_2 + beta_3 * frame_3
    """
    b1 = params[:, :, 0]  # beta_1
    b2 = params[:, :, 1]  # beta_2
    b3 = params[:, :, 2]  # beta_3
    idx_1 = np.nan
    i1 = True
    idx_2 = np.nan
    i2 = True
    if np.nansum(b2) == 0:
        i2 = False
    idx_3 = np.nan
    i3 = True
    if np.nansum(b3) == 0:
        i3 = False
    for i in range(max_MT):
        if i1:
            _, p1 = ttest_1samp(b1[:, i], 0)
            if p1 > pval:
                i1 = False
                idx_1 = 400-i
        if i2:
            _, p2 = ttest_1samp(b2[:, i], 0)
            if p2 > pval:
                i2 = False
                idx_2 = 400-i
        if i3:
            _, p3 = ttest_1samp(b3[:, i], 0)
            if p3 > pval:
                i3 = False
                idx_3 = 400-i
        if i1 + i2 + i3 == 0:
            break
    return [idx_1, idx_2, idx_3]


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
            r, p2 = pearsonr(pop_a, pop_evidence)  # p2 = pvalue from pearson corr
            plist.append(r)
        except Exception:  # TODO: really??
            continue
            # return np.nan
        if p2 > pval:
            return i + 1
        if sim and np.isnan(p2):
            return i + 1
    return np.nan


def get_corr_coef(mat, evl, pval=0.05, max_MT=400, startfrom=700, sim=True):
    # Returns index at which the trajectories and coh vector become uncorrelated
    # backwards in time
    # mat: trajectories (n trials x time)
    rlist = []
    for i in reversed(range(max_MT)):  # reversed so it goes backwards in time
        pop_a = mat[:, startfrom + i]
        nan_idx = ~np.isnan(pop_a)
        pop_evidence = evl[nan_idx]
        pop_a = pop_a[nan_idx]
        r, p2 = pearsonr(pop_a, pop_evidence)  # p2 = pvalue from pearson corr
        rlist.append(r)
    return rlist


def corr_rt_time_prior(df, fig, ax, data_folder, rtbins=np.linspace(0, 150, 16, dtype=int),
                       trajectory='trajectory_y', threshold=300):
    # TODO: do analysis with equipopulated bins
    # split time/subject by prior
    cmap = mtp.colors.LinearSegmentedColormap.from_list("", ["chocolate", "white", "olivedrab"])
    kw = {"trajectory": trajectory, "align": "sound"}
    zt = df.norm_allpriors.values
    out_data = np.empty((400, len(rtbins)-1, 15))
    out_data[:] = np.nan
    df_1 = df.copy()
    out_data_sbj = []
    split_data = data_folder + 'prior_matrix.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(split_data), exist_ok=True)
    if os.path.exists(split_data):
        out_data = np.load(split_data, allow_pickle=True)
    else:
        for i_s, subject in enumerate(df_1.subjid.unique()):
            for i in range(rtbins.size-1):
                dat = df_1.loc[(df_1.subjid == subject) &
                            (df_1.sound_len < rtbins[i + 1]) &
                            (df_1.sound_len >= rtbins[i]) &
                            (~np.isnan(zt))]
                ztl = zt[(df_1.subjid == subject) &
                        (df_1.sound_len < rtbins[i + 1]) &
                        (df_1.sound_len >= rtbins[i]) &
                        (~np.isnan(zt))]
                mat = np.vstack(
                    dat.apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
                ztl = ztl[~np.isnan(mat).all(axis=1)]
                mat = mat[~np.isnan(mat).all(axis=1)]
                current_split_index =\
                    get_corr_coef(mat, ztl, pval=0.01, max_MT=400,
                                  startfrom=700)
                out_data[:, i, i_s] = current_split_index
        np.save(split_data, out_data) 
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    r_coef_mean = np.nanmean(out_data, axis=2)
    # timevals = np.arange(400)[::-1]
    # xvalsrt = rtbins[:-1] + np.diff(rtbins)[0]
    # rtgrid, timegrid = np.meshgrid(xvalsrt, timevals)
    # ax.plot_surface(rtgrid, timegrid, r_coef_mean)
    # ax.set_xlabel('RT (ms)')
    # ax.set_ylabel('Time from stimulus onset (ms)')
    # ax.set_zlabel('Corr. coef.')
    # fig, ax = plt.subplots(1)
    ax.set_title('Prior-position \ncorrelation', fontsize=12)
    ax.plot([0, 14], [0, 150], color='k', linewidth=2)
    im = ax.imshow(r_coef_mean, aspect='auto', cmap=cmap,
                   vmin=-0.5, vmax=0.5, extent=[0, 14, 0, 304])
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylim(0, 304)
    ax.set_yticks([])
    # ax.set_ylabel('Time from stimulus onset (ms)')ç
    pos = ax.get_position()
    ax.set_xticks([0, 4, 9, 14], [rtbins[0], rtbins[5], rtbins[10], rtbins[15]])
    # ax.set_yticks([0, 100, 200, 300], [300, 200, 100, 0])
    pright_cbar_ax = fig.add_axes([pos.x0+pos.width/1.25,
                                   pos.y0 + pos.height/10,
                                   pos.width/20, pos.height/1.3])
    cbar = plt.colorbar(im, cax=pright_cbar_ax)
    cbar.set_label('Corr. coeff.')


def corr_rt_time_stim(df, ax, split_data_all_s, data_folder, rtbins=np.linspace(0, 150, 16, dtype=int),
                      trajectory='trajectory_y', threshold=300):
    # TODO: do analysis with equipopulated bins
    # split time/subject by prior
    cmap = mtp.colors.LinearSegmentedColormap.from_list("", ["chocolate", "white", "olivedrab"])
    out_data = np.empty((400, len(rtbins)-1, 15))
    out_data[:] = np.nan
    splitfun = get_splitting_mat_data
    df_1 = df.copy()
    evs = [0, 0.25, 0.5, 1]
    split_data = data_folder + 'stim_matrix.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(split_data), exist_ok=True)
    if os.path.exists(split_data):
        out_data = np.load(split_data, allow_pickle=True)
    else:
        for i_s, subject in enumerate(df_1.subjid.unique()):
            for i in range(rtbins.size-1):
                for iev, ev in enumerate(evs):
                    matatmp =\
                        splitfun(df=df.loc[(df.special_trial == 0)
                                           & (df.subjid == subject)],
                                 side=0,
                                 rtbin=i, rtbins=rtbins, coh1=ev,
                                 trajectory=trajectory, align="sound")
                    if iev == 0:
                        mat = matatmp
                        evl = np.repeat(0, matatmp.shape[0])
                    else:
                        mat = np.concatenate((mat, matatmp))
                        evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
                current_split_index =\
                    get_corr_coef(mat, evl, pval=0.05, max_MT=400,
                                  startfrom=700)
                out_data[:, i, i_s] = current_split_index
        np.save(split_data, out_data)
    # fig = plt.figure()
    # fig.suptitle('Stimulus corr. coef.')
    # ax = plt.axes(projection='3d')
    r_coef_mean = np.nanmean(out_data, axis=2)
    # timevals = np.arange(400)[::-1]
    # xvalsrt = rtbins[:-1] + np.diff(rtbins)[0]
    # rtgrid, timegrid = np.meshgrid(xvalsrt, timevals)
    # ax.plot_surface(rtgrid, timegrid, r_coef_mean)
    # ax.set_xlabel('RT (ms)')
    # ax.set_ylabel('Time from stimulus onset (ms)')
    # ax.set_zlabel('Corr. coef.')
    # fig2, ax = plt.subplots(1)
    # fig2.suptitle('Stimulus corr. coef.')
    ax.set_title('Stimulus-position \ncorrelation', fontsize=12)
    ax.plot([0, 14], [0, 150], color='k', linewidth=2)
    ax.plot(np.arange(len(split_data_all_s)),
            split_data_all_s, color='firebrick', linewidth=1.4, alpha=0.5)
    ax.imshow(r_coef_mean, aspect='auto', cmap=cmap,
              vmin=-0.5, vmax=0.5, extent=[0, 14, 0, 304])
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylim(0, 304)
    ax.set_ylabel('Time from stimulus onset (ms)')
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xticks([0, 4, 9, 14], [rtbins[0], rtbins[5], rtbins[10], rtbins[15]])
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label('Corr. coeff.')


def get_splitting_mat_data(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7),
                           align='movement', trajectory="trajectory_y",
                           coh1=1):
    """
    Create matrix that will be used to compute splitting time.
    Version of function:
    utilsJ.Models.simul.when_did_split_dat

    df= dataframe
    side= {0,1} left or right,
    rtbins
    startfrom= index to start checking diffs. movement==700;
    plot_kwargs: plot kwargs for ax.plot
    align: whether to align 0 to movement(action) or sound
    """
    kw = {"trajectory": trajectory}

    # TODO: addapt to align= sound
    # get matrices
    if side == 0:
        coh1 = -coh1
    else:
        coh1 = coh1
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        # & (df.resp_len)
    ]  # &(df.R_response==side)
    if align == 'movement':
        kw["align"] = "action"
    elif align == 'sound':
        kw["align"] = "sound"
    idx = (dat.coh2 == coh1) & (dat.rewside == 0)
    mata_0 = np.vstack(dat.loc[idx].apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
    idx = (dat.coh2 == -coh1) & (dat.rewside == 1)
    mata_1 = np.vstack(dat.loc[idx].apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
    mata = np.vstack([mata_0*-1, mata_1])
    mata = mata[~np.isnan(mata).all(axis=1)]

    return mata


def get_splitting_mat_simul(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7),
                            align='movement', coh=1):  # debugging purposes
    """
    Create matrix that will be used to compute splitting time.
    Version of function:
    utilsJ.Models.simul.when_did_split_simul

    """
    def shortpad2(row, upto=1400, align='movement', pad_value=np.nan,
                  pad_pre=0):
        """pads nans to trajectories so it can be stacked in a matrix
        align can be either 'movement' (0 is movement onset), or 'sound'
        """
        if align == 'movement':
            missing = upto - row.traj.size
            return np.pad(row.traj, ((0, missing)), "constant",
                        constant_values=pad_value)
        elif align == 'sound':
            missing_pre = int(row.sound_len)
            missing_after = upto - missing_pre - row.traj.size
            return np.pad(row.traj, ((missing_pre, missing_after)), "constant",
                        constant_values=(pad_pre, pad_value))
    
    
    # get matrices
    if side == 0:
        coh1 = -coh
    else:
        coh1 = coh
    shortpad_kws = {}
    if align == 'sound':
        shortpad_kws = dict(upto=1400, align='sound')
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        # & (df.resp_len) # ?
    ]  # &(df.R_response==side) this goes out
    idx = (dat.coh2 == coh1) & (dat.rewside == 0)
    mata_0 = np.vstack(dat.loc[idx].apply(lambda row: shortpad2(row, **shortpad_kws), axis=1).values.tolist())
    idx = (dat.coh2 == -coh1) & (dat.rewside == 1)
    mata_1 = np.vstack(dat.loc[idx].apply(lambda row: shortpad2(row, **shortpad_kws), axis=1).values.tolist())
    for mat in [mata_0, mata_1]:
        for i_t, t in enumerate(mat):
            ind_last_val = np.where(t == t[~np.isnan(t)][-1])[0][0]
            mat[i_t, ind_last_val:-1] = np.repeat(t[ind_last_val],
                                                  len(t)-ind_last_val-1)
    mata = np.vstack([mata_0*-1, mata_1])

    # discard all nan rows # this is not working because a is a copy!
    mata = mata[~np.isnan(mata).all(axis=1)]

    # if ax is not None:
    #     ax.plot(np.nanmean(mata, axis=0), **plot_kwargs)
    return mata



def plot_trajs_splitting_example(df, ax, rtbin=0, rtbins=np.linspace(0, 150, 2),
                                 subject='LE37', xlabel='', ylabel='', show_legend=False,
                                 startfrom=700, fix_per_offset_subtr=150):
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
    assert startfrom == 700, 'startfrom must be 700, which is the stimulus onset'
    indx = (df.special_trial == 0) & (df.subjid == subject)
    assert np.sum(indx) > 0, 'No trials for subject ' + subject + ' with special_trial == 0'
    lbl = 'RTs: ['+str(rtbins[rtbin])+'-'+str(rtbins[rtbin+1])+']'
    evs = [0, 0.25, 0.5, 1]
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    medians = []
    for iev, ev in enumerate(evs):
        matatmp =\
            get_splitting_mat_data(df=df[indx], side=0, rtbin=rtbin,
                                    rtbins=rtbins, coh1=ev, align='sound')
        median_plt = np.nanmedian(matatmp, axis=0) -\
                np.nanmedian(matatmp[:,startfrom-fix_per_offset_subtr:startfrom])
        ax.plot(np.arange(matatmp.shape[1]) - startfrom,
                median_plt, color=colormap[iev], label=lbl)
        medians.append(median_plt)
        
        if iev == 0:
            mat = matatmp
            evl = np.repeat(0, matatmp.shape[0])
        else:
            mat = np.concatenate((mat, matatmp))
            evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
    ind = get_split_ind_corr(mat, evl, pval=0.05, max_MT=400, startfrom=startfrom)
    ind_y = np.max([m[ind+startfrom] for m in medians])
    ax.set_xlim(-10, 255)
    ax.set_ylim(-0.6, 5.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-0.5, 4])
    # plot horizontal line
    ax.axhline(0, color='k', lw=0.5, ls='--')
    # plot stimulus duration as line at y=3
    mean_stim_dur = np.mean(df.loc[(df.special_trial == 0)&(df.subjid == subject)&
                            (df.sound_len < rtbins[rtbin + 1])&
                            (df.sound_len >= rtbins[rtbin])].sound_len)
    # ax.plot([0, mean_stim_dur], [3.5, 3.5], color=(.7, .7, .7), lw=2)
    col_face = [0.7, 0.7, 0.7, 0.4]
    col_edge = [0.7, 0.7, 0.7, 0]
    ax.fill_between([0, mean_stim_dur], [3.5, 3.5], [4, 4],
                     facecolor=col_face, edgecolor=col_edge)
    if rtbins[1] == 300:
        ax.set_title('RT > 150 ms', fontsize=9.5)
    if rtbins[1] == 65:
        ax.set_title('RT = 50 ms', fontsize=9.5)
    if rtbins[1] == 15:
        ax.set_title('RT < 15 ms', fontsize=9.5)

    # plot arrow
    al = 0.5
    hl = 0.4
    ax.arrow(ind, ind_y+al+3*hl, 0, -al-hl,  color='k', width=1, head_width=8,
             head_length=hl)
    if show_legend:
        labels = ['0', '0.25', '0.5', '1']
        legendelements = []
        for i_l, lab in enumerate(labels[::-1]):
            legendelements.append(Line2D([0], [0], color=colormap[::-1][i_l], lw=2,
                                  label=lab))
        ax.legend(handles=legendelements, fontsize=8, loc='lower right',
                  labelspacing=0.1)
    
    # if xlab:
        
    # if rtbins[-1] > 25:
    #     # ax.set_title('\n RT > 150 ms', fontsize=8)
    #     ax.arrow(ind, 3, 0, -2, color='k', width=1, head_width=5,
    #              head_length=0.4)
    #     ax.text(ind-17, 3.4, 'Splitting Time', fontsize=8)
    #     plot_boxcar_rt(rt=rtbins[0], ax=ax)
    # else:
    #     ax.set_title('RT < 15 ms', fontsize=8)
    #     ax.text(ind-60, 3.3, 'Splitting Time', fontsize=8)
    #     ax.arrow(ind, 2.85, 0, -1.4, color='k', width=1, head_width=5,
    #              head_length=0.4)
    #     ax.set_xticklabels([''])
    #     plot_boxcar_rt(rt=rtbins[-1], ax=ax)
    #     labels = ['0', '0.25', '0.5', '1']
    #     legendelements = []
    #     for i_l, lab in enumerate(labels):
    #         legendelements.append(Line2D([0], [0], color=colormap[i_l], lw=2,
    #                               label=lab))
    #     

def trajs_splitting_prior(df, ax, data_folder, rtbins=np.linspace(0, 150, 16),
                          trajectory='trajectory_y', threshold=300):
    # TODO: do analysis with equipopulated bins
    # split time/subject by prior
    kw = {"trajectory": trajectory, "align": "sound"}
    zt = df.norm_allpriors.values
    out_data = []
    df_1 = df.copy()
    for subject in df_1.subjid.unique():
        out_data_sbj = []
        split_data = data_folder + subject + '/traj_data/' + subject + '_traj_split_prior_005_fwd_9.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data):
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data']
        else:
            for i in range(rtbins.size-1):
                dat = df_1.loc[(df_1.subjid == subject) &
                            (df_1.sound_len < rtbins[i + 1]) &
                            (df_1.sound_len >= rtbins[i]) &
                            (~np.isnan(zt))]
                ztl = zt[(df_1.subjid == subject) &
                        (df_1.sound_len < rtbins[i + 1]) &
                        (df_1.sound_len >= rtbins[i]) &
                        (~np.isnan(zt))]
                mat = np.vstack(
                    dat.apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
                ztl = ztl[~np.isnan(mat).all(axis=1)]
                mat = mat[~np.isnan(mat).all(axis=1)]
                current_split_index =\
                    get_split_ind_corr(mat, ztl, pval=0.01, max_MT=400,
                                       startfrom=700)
                out_data_sbj += [current_split_index]
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
    ax.set_xlabel('Reaction time(ms)')
    # ax.set_title('Impact of prior', fontsize=9)
    ax.set_ylabel('Splitting time (ms)')
    ax.plot([0, 155], [0, 155], color='k')
    ax.fill_between([0, 155], [0, 155], [0, 0],
                    color='grey', alpha=0.2)
    ax.set_xlim(-5, 155)
    # plt.show()


def retrieve_trajs(df, rtbins=np.linspace(0, 150, 16),
                   rtbin=0, align='sound', trajectory='trajectory_y'):
    kw = {"trajectory": trajectory}
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        # & (df.resp_len)
    ]  # &(df.R_response==side)
    if align == 'movement':
        kw["align"] = "action"
    elif align == 'sound':
        kw["align"] = "sound"
    # idx = dat.rewside == 0
    # mata_0 = np.vstack(dat.loc[idx].apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
    # idx = dat.rewside == 1
    # mata_1 = np.vstack(dat.loc[idx].apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
    # mata = np.vstack([mata_0*-1, mata_1])
    mata = np.vstack(dat.apply(lambda x: interpolapply(x, **kw), axis=1).values.tolist())
    mata = mata * (dat.rewside.values*2-1).reshape(-1, 1)
    index_nan = ~np.isnan(mata).all(axis=1)
    mata = mata[index_nan]
    return mata, index_nan


def trajs_splitting_stim(df, ax, data_folder, collapse_sides=True, threshold=300,
                         sim=False,
                         rtbins=np.linspace(0, 150, 16), connect_points=False,
                         trajectory="trajectory_y"):

    # split time/subject by coherence
    if sim:
        splitfun = get_splitting_mat_simul
        df['traj'] = df.trajectory_y.values
    if not sim:
        splitfun = get_splitting_mat_data
    out_data = []
    for subject in df.subjid.unique():
        out_data_sbj = []
        if not sim:
            split_data = data_folder + subject + '/traj_data/' + subject + '_traj_split_stim_005.npz'
        if sim:
            split_data = data_folder + subject + '/sim_data/' + subject + '_traj_split_stim_005.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data):
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data_sbj']
        else:
            for i in range(rtbins.size-1):
                if collapse_sides:
                    evs = [0, 0.25, 0.5, 1]
                    for iev, ev in enumerate(evs):
                        if not sim:  # TODO: do this if within splitfun
                            matatmp =\
                                splitfun(df=df.loc[(df.special_trial == 0)
                                                   & (df.subjid == subject)],
                                         side=0,
                                         rtbin=i, rtbins=rtbins, coh1=ev,
                                         trajectory=trajectory, align="sound")
                        if sim:
                            matatmp =\
                                splitfun(df=df.loc[(df.special_trial == 0)
                                                   & (df.subjid == subject)],
                                         side=0, rtbin=i, rtbins=rtbins, coh=ev,
                                         align="sound")
                        
                        if iev == 0:
                            mat = matatmp
                            evl = np.repeat(0, matatmp.shape[0])
                        else:
                            mat = np.concatenate((mat, matatmp))
                            evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
                    if not sim:
                        current_split_index =\
                            get_split_ind_corr(mat, evl, pval=0.05, max_MT=400,
                                            startfrom=700)
                    if sim:
                        max_mt = 800
                        current_split_index =\
                            get_split_ind_corr(mat, evl, pval=0.05, max_MT=max_mt,
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
    ax.fill_between([0, 155], [0, 155], [0, 0],
                    color='grey', alpha=0.2)
    ax.set_xlim(-5, 155)
    ax.set_ylim(-1, 305)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('Splitting time (ms)')
    # ax.set_title('Impact of stimulus')
    # plt.show()
    return np.nanmedian(out_data.reshape(rtbins.size-1, -1), axis=1)


def splitting_time_frames_ttest_across(df, frame_len=50, rtbins=np.linspace(0, 150, 7),
                                       trajectory="trajectory_y", pval=0.01,
                                       max_MT=400):
    fig, ax = plt.subplots(ncols=4)
    binsize = np.diff(rtbins)[0]
    subjects = df.subjid.unique()
    out_data = np.empty((len(rtbins)-1, 3))
    out_data[:] = np.nan
    for i in range(rtbins.size-1):
        out_data_sbj = np.empty((len(subjects), max_MT, 3))
        out_data_sbj[:] = np.nan
        for i_s, subject in enumerate(subjects):
            df_sub = df.loc[df.subjid == subject]
            reaction_time = df_sub.sound_len.values
            stimulus = np.array(df_sub.res_sound)
            category = df_sub.rewside.values*2-1
            first_frame = [stimulus[i][0] for i in range(len(stimulus))]
            second_frame = [stimulus[i][1] for i in range(len(stimulus))]
            third_frame = [stimulus[i][2] for i in range(len(stimulus))]
            matatmp, idx =\
                retrieve_trajs(df_sub, rtbins=rtbins,
                               rtbin=i, align='sound', trajectory=trajectory)
            stim = np.zeros((int(i*binsize/frame_len)+1, len(stimulus)))
            stim[0] = first_frame
            if i*binsize >= 50:
                stim[1] = second_frame  
            if i*binsize >= 100:
                stim[2] = third_frame
            stim = stim[:, (reaction_time >= rtbins[i]) &
                       (reaction_time < rtbins[i+1])][:, idx] *\
                category[(reaction_time >= rtbins[i]) &
                         (reaction_time < rtbins[i+1])][idx]
            params = get_params_lin_reg_frames(matatmp, stim)
            out_data_sbj[i_s, :, 0] = np.array(params[0])
            out_data_sbj[i_s, :, 1] = np.array(params[1])
            out_data_sbj[i_s, :, 2] = np.array(params[2])
        splitting_index =\
            get_dv_from_params_ttest(params=out_data_sbj, max_MT=max_MT, pval=pval)
        for j in range(out_data.shape[1]):
            out_data[i, j] = splitting_index[j]
    ax[0].set_ylabel('Splitting Time (ms)')
    titles = ['1st frame', '2nd frame', '3rd frame', ' ']
    colors = ['r', 'k', 'b']
    for i in range(len(ax)):
        ax[i].plot([0, 300], [0, 300], color='k')
        ax[i].fill_between([0, 300], [0, 300], [0, 0],
                           color='grey', alpha=0.6)
        ax[i].set_title(titles[i])
        ax[i].set_xlim(-5, 305)
        ax[i].set_ylim(-5, 405)
        fp.rm_top_right_lines(ax[i])
        ax[i].set_xlabel('Reaction Time (ms)')
        if i <= 2:
            ax[i].plot(rtbins[:-1]+binsize/2, out_data[:, i], marker='o',
                           color='firebrick')
        else:
            for j in range(out_data.shape[1]):
                ax[i].plot(rtbins[:-1]+binsize/2, out_data[:, j],
                           color=colors[j], label=titles[j])
            ax[i].legend()


def splitting_time_frames(df, data_folder, frame_len=50, rtbins=np.linspace(0, 150, 7),
                          trajectory="trajectory_y", new_data=True, pval=0.01,
                          max_MT=400):
    fig, ax = plt.subplots(ncols=4)
    binsize = np.diff(rtbins)[0]
    subjects = df.subjid.unique()
    out_data = np.empty((len(rtbins)-1, 3, len(subjects)))
    for i_s, subject in enumerate(subjects):
        out_data_sbj = np.empty((len(rtbins)-1, 3))
        out_data_sbj[:] = np.nan
        split_data = data_folder + subject + '/traj_data/' + subject + '_traj_split_stim_frames.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data) and not new_data:
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data_sbj']
        else:
            df_sub = df.loc[df.subjid == subject]
            reaction_time = df_sub.sound_len.values
            stimulus = np.array(df_sub.res_sound)
            category = df_sub.rewside.values*2-1
            first_frame = [stimulus[i][0] for i in range(len(stimulus))]
            second_frame = [stimulus[i][1] for i in range(len(stimulus))]
            third_frame = [stimulus[i][2] for i in range(len(stimulus))]
            for i in range(rtbins.size-1):
                matatmp, idx =\
                    retrieve_trajs(df_sub, rtbins=rtbins,
                                   rtbin=i, align='sound', trajectory=trajectory)
                stim = np.zeros((int(i*binsize/frame_len)+1, len(stimulus)))
                stim[0] = first_frame
                if i*binsize >= 50:
                    stim[1] = second_frame  
                if i*binsize >= 100:
                    stim[2] = third_frame
                stim = stim[:, (reaction_time >= rtbins[i]) &
                           (reaction_time < rtbins[i+1])][:, idx] *\
                    category[(reaction_time >= rtbins[i]) &
                             (reaction_time < rtbins[i+1])][idx]
                current_split_index =\
                    get_split_ind_corr_frames(matatmp, stim, pval=pval,
                                              max_MT=max_MT)
                out_data_sbj[i, 0] = current_split_index[0]
                out_data_sbj[i, 1] = current_split_index[1]
                out_data_sbj[i, 2] = current_split_index[2]
            np.savez(split_data, out_data_sbj=out_data_sbj)
        out_data[:, :, i_s] = out_data_sbj
    ax[0].set_ylabel('Splitting Time (ms)')
    titles = ['1st frame', '2nd frame', '3rd frame', ' ']
    splt_data_all = np.empty((len(rtbins)-1, 3))
    splt_data_all[:] = np.nan
    err_data_all = np.empty((len(rtbins)-1, 3))
    err_data_all[:] = np.nan
    colors = ['r', 'k', 'b']
    for i in range(len(ax)):
        ax[i].plot([0, 305], [0, 300], color='k')
        ax[i].fill_between([0, 300], [0, 300], [0, 0],
                           color='grey', alpha=0.6)
        ax[i].set_title(titles[i])
        ax[i].set_xlim(-5, 305)
        ax[i].set_ylim(-5, 405)
        fp.rm_top_right_lines(ax[i])
        ax[i].set_xlabel('Reaction Time (ms)')
        if i <= 2:
            splt_data = np.nanmedian(out_data[:, i, :], axis=1)
            splt_data_all[:, i] = splt_data
            err_data = np.nanstd(out_data[:, i, :], axis=1) / np.sqrt(len(subjects))
            err_data_all[:, i] = err_data
            ax[i].errorbar(rtbins[:-1]+binsize/2, splt_data, err_data, marker='o',
                           color='firebrick', ecolor='firebrick')
            for j in range(len(subjects)):
                ax[i].plot(rtbins[:-1]+binsize/2, out_data[:, i, j],
                           marker='o', mfc=(.6, .6, .6, .3), mec=(.6, .6, .6, 1),
                           mew=1, color=(.6, .6, .6, .3))
        else:
            for j in range(splt_data_all.shape[1]):
                ax[i].plot(rtbins[:-1]+binsize/2, splt_data_all[:, j],
                           color=colors[j], label=titles[j])
                ax[i].fill_between(rtbins[:-1]+binsize/2,
                                   splt_data_all[:, j]-err_data_all[:, j],
                                   splt_data_all[:, j]+err_data_all[:, j],
                                   color=colors[j], alpha=0.3)
            ax[i].legend()


def logifunc(x,A, B,x01,k1, x02, k2, x03, k3):
    return A / (1 + B*np.exp(-k1*(x[0]-x01)-k2*(x[1]-x02)-k3*(x[2]-x03)))


def log_reg_frames(df, frame_len = 50):
    response = df.R_response.values
    stim = np.array(df.res_sound)
    stim_final = np.array([st[:3] for st in stim])
    frames_listened = df.sound_len.astype(int).values // 50+1
    for irt, fr in enumerate(frames_listened):
        if fr < 3:
            stim_final[irt, 2] = 0
        if fr < 2:
            stim_final[irt, 1] = 0
    logreg = LogisticRegression()
    # fit
    logreg.fit(stim_final, response)
    # extract coeffs
    params = logreg.coef_
    params[params == 0] = np.nan
    # print(params)
    # popt, pcov = cfit(logifunc, stim_final.T, response)
    return params


def log_reg_vs_rt(df, rtbins=np.linspace(0, 150, 16)):
    fig, ax = plt.subplots(1)
    fig2, ax2 = plt.subplots(3, 5)
    ax2 = ax2.flatten()
    colors = ['r', 'b', 'k']
    labels = ['1st', '2nd', '3rd']
    subjects = df.subjid.unique()
    binsize = np.diff(rtbins)[0]
    mat_total = np.empty((len(rtbins)-1, 3, len(subjects)))
    mat_total[:] = np.nan
    for i_s, subj in enumerate(subjects):
        mat_per_sub = np.empty((len(rtbins)-1, 3))
        mat_per_sub[:] = np.nan
        df_sub = df.loc[df.subjid == subj]
        for i in range(rtbins.size-1):
            df_sub_2 = df_sub.loc[(df_sub.sound_len >= rtbins[i]) &
                                  (df_sub.sound_len < rtbins[i+1])]
            mat_per_sub[i, :] = log_reg_frames(df=df_sub_2)
        mat_total[:, :, i_s] = mat_per_sub
        ax.plot(rtbins[:-1] + binsize/2, mat_per_sub.T[0],
                color=colors[0], alpha=0.3)
        ax.plot(rtbins[:-1] + binsize/2, mat_per_sub.T[1],
                color=colors[1], alpha=0.3)
        ax.plot(rtbins[:-1] + binsize/2, mat_per_sub.T[2],
                color=colors[2], alpha=0.3)
        for t in range(3):
            ax2[i_s].plot(rtbins[:-1] + binsize/2, mat_per_sub.T[t],
                          color=colors[t], label=labels[t]+' frame')
        ax2[i_s].set_title(subj)
        ax2[i_s].legend()
        fp.rm_top_right_lines(ax2[i_s])
        ax2[i_s].set_xlabel('Reaction time (ms)')
        ax2[i_s].set_ylabel('Logistic Regression weight (a.u.)')
    weights_mean = np.nanmean(mat_total, axis=2).T
    weights_err = np.nanstd(mat_total, axis=2).T / np.sqrt(len(subjects))
    for j in range(3):
        ax.plot(rtbins[:-1] + binsize/2, weights_mean[j],  # weights_err[j],
                label=labels[j]+' frame', marker='o', linewidth=2.5,
                color=colors[j])
    ax.legend()
    fp.rm_top_right_lines(ax)
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('Logistic Regression weight (a.u.)')


def fig_2_trajs(df, rat_nocom_img, data_folder, sv_folder, st_cartoon_img, fgsz=(8, 12),
                inset_sz=.1, marginx=-.04, marginy=0.1, subj='LE46'):
    f, ax = plt.subplots(4, 3, figsize=fgsz)
    # add letters to panels
    letters = 'abcdehfgXij'
    ax = ax.flatten()
    for lett, a in zip(letters, ax):
        if lett != 'X' and lett != 'h':
            fp.add_text(ax=a, letter=lett, x=-0.1, y=1.2)
        if lett == 'h':
            fp.add_text(ax=a, letter=lett, x=-0.1, y=1.)
    ax[8].axis('off')
    # ax[11].axis('off')
    # adjust panels positions
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
                        hspace=0.5, wspace=0.4)
    factor = 1.
    for i_ax in [3, 4, 6, 7]:
        pos = ax[i_ax].get_position()
        if i_ax in [3, 6]:
            ax[i_ax].set_position([pos.x0, pos.y0, pos.width*factor,
                                    pos.height])
        else:
            ax[i_ax].set_position([pos.x0+pos.width/8, pos.y0, pos.width*factor,
                                    pos.height])
    # add insets
    ax = f.axes
    ax_zt = np.array([ax[3], ax[6]])
    ax_cohs = np.array([ax[4], ax[7]])
    ax_inset = fp.add_inset(ax=ax_cohs[1], inset_sz=inset_sz, fgsz=fgsz,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    # ax_cohs contains in this order the axes for:
    # index 0: mean position of rats conditioned on stim. evidence,
    # index 1: the inset for the velocity panel 
    # index 2: mean velocity  of rats conditioned on stim. evidence
    ax_cohs = np.insert(ax_cohs, 1, ax_inset)
    ax_inset = fp.add_inset(ax=ax_zt[1], inset_sz=inset_sz, fgsz=fgsz,
                            marginx=marginx, marginy=marginy, right=True)
    ax_inset.yaxis.set_ticks_position('none')
    ax_zt = np.insert(ax_zt, 1, ax_inset)
     # ax_zt contains in this order the axes for:
    # index 0: mean position of rats conditioned on prior evidence,
    # index 1: the inset for the velocity panel 
    # index 2: mean velocity  of rats conditioned on priors evidence

    ax_weights = ax[2]
    pos = ax_weights.get_position()
    ax_weights.set_position([pos.x0, pos.y0+pos.height/4, pos.width,
                             pos.height*1/2])
    for i_a, a in enumerate(ax):
        if i_a != 8:
            fp.rm_top_right_lines(a)
    margin = 0.05
    # tune screenshot panel
    ax_scrnsht = ax[0]
    ax_scrnsht.set_xticks([])
    right_port_y = 70
    center_port_y = 250
    left_port_y = 440
    ax_scrnsht.set_yticks([right_port_y, center_port_y, left_port_y])
    ax_scrnsht.set_yticklabels([-75, 0, 75])
    ax_scrnsht.set_xlabel('x dimension (pixels)')
    ax_scrnsht.set_ylabel('y dimension (pixels)')
    # add colorbar for screenshot
    n_stps = 100
    pos = ax_scrnsht.get_position()
    ax_clbr = plt.axes([pos.x0+margin/2, pos.y0+pos.height+margin/8,
                        pos.width*0.7, pos.height/15])
    ax_clbr.imshow(np.linspace(0, 1, n_stps)[None, :], aspect='auto')
    x_tcks = np.linspace(0, n_stps, 5)
    ax_clbr.set_xticks(x_tcks)
    x_tcks_str = ['0', '', '', '', str(int(2.5*n_stps))]
    x_tcks_str[-1] += ' ms'
    ax_clbr.set_xticklabels(x_tcks_str)
    ax_clbr.tick_params(labelsize=8)
    ax_clbr.set_yticks([])
    ax_clbr.xaxis.set_ticks_position("top")
    # tune trajectories panels
    ax_rawtr = ax[1]
    ax_ydim = ax[2]
    x_lim = [-80, 20]
    y_lim = [-100, 100]
    ax_rawtr.set_xlim(x_lim)
    ax_rawtr.set_ylim(y_lim)
    ax_rawtr.set_xticklabels([])
    ax_rawtr.set_yticklabels([])
    ax_rawtr.set_xticks([])
    ax_rawtr.set_yticks([])
    ax_rawtr.set_xlabel('x dimension (pixels)')
    pos_coh = ax_cohs[2].get_position()
    pos_rawtr = ax_rawtr.get_position()
    ax_rawtr.set_position([pos_coh.x0, pos_rawtr.y0,
                           pos_rawtr.width/2, pos_rawtr.height])
    fp.add_text(ax=ax_rawtr, letter='rat LE46', x=0.7, y=1., fontsize=10)
    x_lim = [-100, 800]
    y_lim = [-100, 100]
    ax_ydim.set_xlim(x_lim)
    ax_ydim.set_ylim(y_lim)
    ax_ydim.set_yticks([])
    ax_ydim.set_xlabel('Time from movement onset (ms)')
    pos_ydim = ax_ydim.get_position()
    ax_ydim.set_position([pos_ydim.x0-1.5*margin, pos_rawtr.y0,
                          pos_ydim.width, pos_rawtr.height])
    fp.add_text(ax=ax_ydim, letter='rat LE46', x=0.32, y=1., fontsize=10)
    # tune splitting time panels
    factor_y = 0.5
    factor_x = 0.8
    plt.axes
    ax_cartoon = ax[5]
    ax_cartoon.axis('off')
    pos = ax_cartoon.get_position()
    ax_cartoon.set_position([pos.x0+pos.width/8, pos.y0+pos.height*factor_y*0.9,
                             pos.width*0.9, pos.height*0.9])
    ax_top = plt.axes([.1, .1, .1, .1])
    ax_middle = plt.axes([.2, .2, .1, .1])
    ax_bottom = plt.axes([.3, .3, .1, .1])
    for i_a, a in enumerate([ax_top, ax_middle, ax_bottom]):
        a.set_position([pos.x0+pos.width/4, pos.y0-(i_a)*pos.height*factor_y*1.5, pos.width*factor_x,
                        pos.height*factor_y])
        fp.rm_top_right_lines(a)
    # move ax[10] (spllting time coherence) to the right
    pos = ax[10].get_position()
    ax[10].set_position([pos_coh.x0, pos.y0+pos.height/12,
                         pos.width*0.9, pos.height*0.9])
    # plt.show()
    # TRACKING SCREENSHOT
    rat = plt.imread(rat_nocom_img)
    img = rat[80:576, 120:-10, :]
    ax_scrnsht.imshow(np.flipud(img)) # rat.shape = (796, 596, 4)
    ax_scrnsht.axhline(y=left_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(y=right_port_y, linestyle='--', color='k', lw=.5)
    ax_scrnsht.axhline(center_port_y, color='k', lw=.5)
    ax_scrnsht.set_ylim([0, img.shape[0]])


    # TRAJECTORIES
    df_subj = df[df.subjid == subj]
    ran_max = 100
    for tr in range(ran_max):
        if tr > (ran_max/2):
            trial = df_subj.iloc[tr]
            traj_x = trial['trajectory_x']
            traj_y = trial['trajectory_y']
            ax_rawtr.plot(traj_x, traj_y, color='grey', lw=.5, alpha=0.6)
            time = trial['time_trajs']
            ax_ydim.plot(time, traj_y, color='grey', lw=.5, alpha=0.6)
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
    img = plt.imread(st_cartoon_img)
    ax_cartoon.imshow(img)
    plot_trajs_splitting_example(df, ax=ax_top, rtbins=np.linspace(150, 300, 2))
    plot_trajs_splitting_example(df, ax=ax_bottom, rtbins=np.linspace(0, 15, 2),
                                 xlabel='Time from stimulus onset (ms)', show_legend=True)
    plot_trajs_splitting_example(df, ax=ax_middle, rtbins=np.linspace(45, 65, 2),
                                  ylabel='Position')
    # TRAJECTORY SPLITTING STIMULUS
    split_data_all_s = trajs_splitting_stim(df=df, data_folder=data_folder, ax=ax[9],
                                            connect_points=True)
    # trajs_splitting_prior(df=df, ax=ax[9], data_folder=data_folder)
    corr_rt_time_stim(df=df, split_data_all_s=split_data_all_s,
                      ax=ax[10], data_folder=data_folder)
    corr_rt_time_prior(df=df, fig=f, ax=ax[11], data_folder=data_folder)
    pos = ax[11].get_position()
    ax[11].set_position([pos.x0-pos.width/5, pos.y0+pos.height/12,
                         pos.width*0.9, pos.height*0.9])
    f.savefig(sv_folder+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/Fig2.svg', dpi=400, bbox_inches='tight')
