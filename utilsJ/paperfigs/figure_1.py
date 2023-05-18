import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
from scipy.stats import zscore
from scipy.optimize import curve_fit
import pandas as pd
import sys
sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel

from utilsJ.paperfigs import figures_paper as fp
from utilsJ.Behavior.plotting import tachometric


# ---FUNCTIONS
def plot_rt_cohs_with_fb(df, ax, subj='LE46'):
    colormap = pl.cm.gist_yarg(np.linspace(0.3, 1, 4))
    df_1 = df[df.subjid == subj]
    coh_vec = df_1.coh2.values
    for ifb, fb in enumerate(df_1.fb):
        for j in range(len(fb)):
            coh_vec = np.append(coh_vec, [df_1.coh2.values[ifb]])
    for iev, ev in enumerate([0, 0.25, 0.5, 1]):
        index = np.abs(coh_vec) == ev
        fix_breaks =\
            np.vstack(np.concatenate([df_1.sound_len/1000,
                                      np.concatenate(df_1.fb.values)-0.3]))
        fix_breaks = fix_breaks[index]
        counts_coh, bins = np.histogram(fix_breaks*1000,
                                        bins=30, range=(-100, 200))
        norm_counts = counts_coh/sum(counts_coh)
        ax.plot(bins[:-1]+(bins[1]-bins[0])/2, norm_counts,
                color=colormap[iev])
    ax.set_ylabel('RT density')
    ax.set_xlabel('Reaction time (ms)')


def plot_mt_vs_evidence(df, ax, condition='choice_x_coh', prior_limit=0.25,
                        rt_lim=50, after_correct_only=True):
    subjects = df['subjid'].unique()
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1) == 2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values, axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan
    bins, _, indx_trajs, _, colormap =\
          fp.get_bin_info(df=df, condition=condition, prior_limit=prior_limit,
                          after_correct_only=after_correct_only,
                            rt_lim=rt_lim)
    df = df.loc[indx_trajs]
    if condition == 'choice_x_coh':
        # compute median MT for each subject and each stim strength
        df['choice_x_coh'] = (df.R_response*2-1) * df.coh2
        mt_time = df.groupby(['choice_x_coh', 'subjid']).resp_len.median()
        # unstack to have a matrix with rows for subjects and columns for bins
        mt_time = mt_time.unstack(fill_value=np.nan).values.T
        plot_bins = sorted(df.coh2.unique())
    elif condition == 'choice_x_prior':
        df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
        mt_time = fp.binning_mt_prior(df, bins)
        plot_bins = bins[:-1] + np.diff(bins)/2
    mt_time_err = np.nanstd(mt_time, axis=0) / np.sqrt(len(subjects))
    for i_tr, bin in enumerate(plot_bins):
        c = colormap[i_tr]  
        if len(subjects) > 1:            
            ax.boxplot(mt_time[:, i_tr], positions=[bin], 
                       boxprops=dict(markerfacecolor=c, markeredgecolor=c))
            ax.plot(bin + 0.1*np.random.randn(len(subjects)),
                    mt_time[:, i_tr], color=colormap[i_tr], marker='o',
                    linestyle='None')
        else:
            ax.errorbar(bin, mt_time[:, i_tr], yerr=mt_time_err[i_tr],
                            color=c, marker='o')

        ax.set_ylabel('MT (ms)', fontsize=9)
    # ax.plot(xp, mt_time, color='k', ls=':')



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
    ydata = np.array(mt[~com.astype(bool)])
    popt, pcov = curve_fit(f=linear_fun, xdata=xdata, ydata=ydata)
    if plot:
        df = pd.DataFrame({'coh': coh/max(coh), 'prior': prior/max(prior),
                           'MT': mt,
                           'trial_index': trial_index/max(trial_index)})
        plt.figure()
        sns.pointplot(data=df, x='coh', y='MT', label='coh')
        sns.pointplot(data=df, x='prior', y='MT', label='prior')
        sns.pointplot(data=df, x='trial_index', y='MT', label='trial_index')
        plt.ylabel('MT (ms)')
        plt.xlabel('normalized variables')
        plt.legend()
    return popt

def plot_mt_weights_bars(means, errors, ax, f5=False, means_model=None,
                         errors_model=None, width=0.35):
    labels = ['Prior', 'Stimulus']  # , 'Trial index'
    if not f5:
        ax.bar(x=labels, height=means, yerr=errors, capsize=3, color='gray',
               ecolor='blue')
        ax.set_ylabel('Impact on MT (weights, a.u)')
    if f5:
        x = np.arange(len(labels))
        ax.bar(x=x-width/2, height=means, yerr=errors, width=width,
               capsize=3, color='k', label='Data', ecolor='blue')
        ax.bar(x=x+width/2, height=means_model, yerr=errors_model, width=width,
               capsize=3, color='red', label='Model')
        ax.set_ylabel('Impact on MT (weights, a.u)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8)

def plot_mt_weights_violins(w_coh, w_t_i, w_zt, ax, mt=True, t_index_w=False):
    if t_index_w:
        labels = ['Prior', 'Stimulus', 'Trial index']  # ]
        arr_weights = np.concatenate((w_zt, w_coh, w_t_i))
        palette = ['darkorange', 'red', 'steelblue']
    else:
        labels = ['Prior', 'Stimulus']  # ]
        arr_weights = np.concatenate((w_zt, w_coh))  #
        palette = ['darkorange', 'red']
    label_1 = []
    for j in range(len(labels)):
        for i in range(len(w_coh)):
            label_1.append(labels[j])
    df_weights = pd.DataFrame({' ': label_1, 'weight': arr_weights})

    sns.violinplot(data=df_weights, x=" ", y="weight", ax=ax,
                   palette=palette, linewidth=0.8)
    if t_index_w:
        arr_weights = np.array((w_zt, w_coh, w_t_i))
    else:
        arr_weights = np.array((w_zt, w_coh))
    for i in range(len(labels)):
        ax.plot(np.repeat(i, len(arr_weights[i])) +
                0.1*np.random.randn(len(arr_weights[i])),
                arr_weights[i], color='k', marker='o', linestyle='',
                markersize=2)
        ax.collections[0].set_edgecolor('k')
    if t_index_w:
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticklabels([labels[0], labels[1], labels[2]], fontsize=9)
    else:
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticklabels([labels[0], labels[1]], fontsize=9)
    if mt:
        ax.set_ylabel('Impact on MT (weights, a.u)')
    else:
        ax.set_ylabel('Impact on RT (weights, a.u)')
    ax.axhline(y=0, linestyle='--', color='k', alpha=.4)

def mt_weights(df, ax, plot=False, means_errs=True, mt=True, t_index_w=False):
    w_coh = []
    w_t_i = []
    w_zt = []
    if ax is None and plot:
        fig, ax = plt.subplots(1)
    for subject in df.subjid.unique():
        df_1 = df.loc[df.subjid == subject]
        if mt:
            var = zscore(np.array(df_1.resp_len))
        else:
            var = zscore(np.array(df_1.sound_len))
        decision = np.array(df_1.R_response)*2 - 1
        coh = np.array(df_1.coh2)
        trial_index = np.array(df_1.origidx)
        com = df_1.CoM_sugg.values
        zt = df_1.allpriors.values
        params = mt_linear_reg(mt=var[~np.isnan(zt)],
                               coh=zscore(coh[~np.isnan(zt)] *
                                          decision[~np.isnan(zt)]),
                               trial_index=zscore(trial_index[~np.isnan(zt)].
                                                  astype(float)),
                               prior=zscore(zt[~np.isnan(zt)] *
                                            decision[~np.isnan(zt)]),
                               plot=False,
                               com=com[~np.isnan(zt)])
        w_coh.append(params[1])
        w_t_i.append(params[2])
        w_zt.append(params[3])
    mean_2 = np.nanmean(w_coh)
    mean_1 = np.nanmean(w_zt)
    std_2 = np.nanstd(w_coh)/np.sqrt(len(w_coh))
    std_1 = np.nanstd(w_zt)/np.sqrt(len(w_zt))
    errors = [std_1, std_2]  # , std_3
    means = [mean_1, mean_2]  # , mean_3
    if t_index_w:
        errors = [std_1, std_2, np.nanstd(w_t_i)/np.sqrt(len(w_t_i))]
        means = [mean_1, mean_2, np.nanmean(w_t_i)]
    if plot:
        if means_errs:
            plot_mt_weights_bars(means=means, errors=errors, ax=ax)
            fp.rm_top_right_lines(ax=ax)
        else:
            plot_mt_weights_violins(w_coh=w_coh, w_t_i=w_t_i, w_zt=w_zt,
                                    ax=ax, mt=mt, t_index_w=t_index_w)
    if means_errs:
        return means, errors
    else:
        return w_coh, w_t_i, w_zt

def plot_mt_vs_stim(df, ax, prior_min=0.5, rt_max=50):
    subjects = df.loc[df.special_trial == 2, 'subjid'].unique()
    subjid = df.subjid
    zt_cong = df.norm_allpriors.values * (df.R_response*2-1)
    coh_cong = df.coh2.values * (df.R_response*2-1)
    rt = df.sound_len.values
    spec_trial = df.special_trial.values
    mt = df.resp_len.values
    mt_mat = np.empty((len(subjects), len(np.unique(coh_cong))))
    sil = []
    for i_s, subject in enumerate(subjects):
        # silent
        prior_min_quan = np.quantile(zt_cong[(subjid == subject) &
                                             (~np.isnan(zt_cong))], prior_min)
        sil.append(np.nanmean(mt[(zt_cong >= prior_min_quan) & (rt <= rt_max) &
                                 (spec_trial == 2) & (subjid == subject)])*1e3)
        for iev, ev in enumerate(np.unique(coh_cong)):
            index = (subjid == subject) & (zt_cong >= prior_min_quan) &\
                (rt <= rt_max) & (spec_trial == 0) & (coh_cong == ev)
            mt_mat[i_s, iev] = np.nanmean(mt[index])*1e3
    mean_mt_vs_coh = np.nanmean(mt_mat, axis=0)
    sd_mt_vs_coh = np.nanstd(mt_mat, axis=0)/np.sqrt(len(subjects))
    ax.axhline(np.nanmean(sil), color='k', linestyle='--', alpha=0.6)
    coh_unq = np.unique(coh_cong)
    colormap = pl.cm.coolwarm(np.linspace(0, 1, len(coh_unq)))
    for x, y, e, color in zip(coh_unq, mean_mt_vs_coh, sd_mt_vs_coh, colormap):
        ax.plot(x, y, 'o', color=color)
        ax.errorbar(x, y, e, color=color)
    # coh_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    ax.set_xticks([-1, 0, 1])
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('MT (ms)')


def mt_matrix_ev_vs_zt(df, ax, silent_comparison=False, rt_bin=None,
                       collapse_sides=False):
    coh_raw = df.coh2.copy()
    norm_allp_raw = df.norm_allpriors.copy()
    df['norm_allpriors'] *= df.R_response.values*2-1
    df['coh2'] *= df.R_response.values*2-1
    if not collapse_sides:
        ax0, ax1 = ax
    if rt_bin is not None:
        if rt_bin > 100:
            df_fil = df.loc[df.sound_len > rt_bin]
        if rt_bin < 100:
            df_fil = df.loc[df.sound_len < rt_bin]
        if not collapse_sides:
            df_0 = df_fil.loc[(df_fil.R_response == 0)]
            df_1 = df_fil.loc[(df_fil.R_response == 1)]
        else:
            df_s = df_fil
    else:
        if not collapse_sides:
            df_0 = df.loc[(df.R_response == 0)]
            df_1 = df.loc[(df.R_response == 1)]
        else:
            df_s = df.copy()
    bins_zt = [1.01]
    for i_p, perc in enumerate([0.75, 0.5, 0.25, 0.25, 0.5, 0.75]):
        if i_p < 3:
            bins_zt.append(df.norm_allpriors.abs().quantile(perc))
        else:
            bins_zt.append(-df.norm_allpriors.abs().quantile(perc))
    bins_zt.append(-1.01)
    # bins_zt = [1.01, 0.65, 0.35, 0.1, -0.1, -0.35, -0.65, -1.01]
    # np.linspace(1.01, -1.01, 8)
    coh_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    nsubs = len(df.subjid.unique())
    if not collapse_sides:
        mat_0 = np.zeros((len(bins_zt)-1, 7, nsubs))
        mat_1 = np.zeros((len(bins_zt)-1, 7, nsubs))
        silent_mat_per_rows_0 = np.zeros((len(bins_zt)-1, 7))
        silent_mat_per_rows_1 = np.zeros((len(bins_zt)-1, 7))
    else:
        mat_s = np.zeros((len(bins_zt)-1, 7, nsubs))
        silent_mat_per_rows = np.zeros((len(bins_zt)-1, 7))
    # reference MT computation
    for i_zt, zt in enumerate(bins_zt[:-1]):
        if not collapse_sides:
            mt_sil_per_sub_0 = []
            mt_sil_per_sub_1 = []
        else:
            mt_sil_per_sub = []
        for subj in df.subjid.unique():
            if not collapse_sides:
                mt_silent_0 = df_0.loc[(df_0.special_trial == 2) &
                                       (df_0.norm_allpriors < zt) &
                                       (df_0.subjid == subj) &
                                       (df_0.norm_allpriors > bins_zt[i_zt+1]),
                                       'resp_len']
                mt_silent_1 = df_1.loc[(df_1.special_trial == 2) &
                                       (df_1.norm_allpriors < zt) &
                                       (df_1.subjid == subj) &
                                       (df_1.norm_allpriors > bins_zt[i_zt+1]),
                                       'resp_len']
                mt_sil_per_sub_0.append(np.nanmean(mt_silent_0))
                mt_sil_per_sub_1.append(np.nanmean(mt_silent_1))
            else:
                mt_silent = df_s.loc[(df_s.special_trial == 2) &
                                     (df_s.norm_allpriors < zt) &
                                     (df_s.subjid == subj) &
                                     (df_s.norm_allpriors > bins_zt[i_zt+1]),
                                     'resp_len']
                mt_sil_per_sub.append(np.nanmean(mt_silent))
        if not collapse_sides:
            silent_mat_per_rows_0[i_zt, :] = np.nanmean(mt_sil_per_sub_0)*1e3
            silent_mat_per_rows_1[i_zt, :] = np.nanmean(mt_sil_per_sub_1)*1e3
        else:
            silent_mat_per_rows[i_zt, :] = np.nanmean(mt_sil_per_sub)*1e3
    # MT matrix computation, reference MT can be substracted if desired
    for i_s, subj in enumerate(df.subjid.unique()):
        if not collapse_sides:
            df_sub_0 = df_0.loc[(df_0.subjid == subj)]
            df_sub_1 = df_1.loc[(df_1.subjid == subj)]
        else:
            df_sub = df_s.loc[df_s.subjid == subj]
        for i_ev, ev in enumerate(coh_vals):
            for i_zt, zt in enumerate(bins_zt[:-1]):
                if not collapse_sides:
                    mt_vals_0 = df_sub_0.loc[(df_sub_0.coh2 == ev) &
                                             (df_sub_0.norm_allpriors < zt)
                                             & (df_sub_0.norm_allpriors >=
                                                bins_zt[i_zt+1]),
                                             'resp_len']
                    mt_vals_1 = df_sub_1.loc[(df_sub_1.coh2 == ev) &
                                             (df_sub_1.norm_allpriors < zt)
                                             & (df_sub_1.norm_allpriors >=
                                                bins_zt[i_zt+1]),
                                             'resp_len']
                    mat_0[i_zt, i_ev, i_s] = np.nanmean(mt_vals_0)*1e3
                    mat_1[i_zt, i_ev, i_s] = np.nanmean(mt_vals_1)*1e3
                else:
                    mt_vals = df_sub.loc[(df_sub.coh2 == ev) &
                                         (df_sub.norm_allpriors < zt)
                                         & (df_sub.norm_allpriors >=
                                            bins_zt[i_zt+1]),
                                         'resp_len']
                    mat_s[i_zt, i_ev, i_s] = np.nanmean(mt_vals)*1e3
    if not collapse_sides:
        mat_0 = np.nanmean(mat_0, axis=2)
        mat_1 = np.nanmean(mat_1, axis=2)
    else:
        mat_s = np.nanmean(mat_s, axis=2)
    if silent_comparison:
        if not collapse_sides:
            # We substract reference MT (in silent trials) at each row
            mat_0 -= silent_mat_per_rows_0
            mat_1 -= silent_mat_per_rows_1
        else:
            mat_s -= silent_mat_per_rows
    # else:
    #     if collapse_sides:
    #         mat_s = np.column_stack((mat_s, silent_mat_per_rows[:, 0]))
    df['norm_allpriors'] = norm_allp_raw
    df['coh2'] = coh_raw
    if not collapse_sides:
        # SIDE 0
        im_0 = ax0.imshow(mat_0, cmap='RdGy', vmin=np.nanmin((mat_1, mat_0)),
                          vmax=np.nanmax((mat_1, mat_0)))
        plt.sca(ax0)
        cbar_0 = plt.colorbar(im_0, fraction=0.04)
        cbar_0.remove()
        # cbar_0.set_label(r'$MT \; - MT_{silent}(ms)$')
        ax0.set_xlabel('Evidence')
        ax0.set_ylabel('Prior')
        if rt_bin is not None:
            if rt_bin > 100:
                ax0.set_title('Left, RT > ' + str(rt_bin) + ' ms')
            else:
                ax0.set_title('Left, RT < ' + str(rt_bin) + ' ms')
        else:
            ax0.set_title('Left')
        ax0.set_yticks([0, 3, 6], ['R', '0', 'L'])
        ax0.set_xticks([0, 3, 6], ['L', '0', 'R'])
        # SIDE 1
        ax1.set_title('Right')
        im_1 = ax1.imshow(mat_1, cmap='RdGy', vmin=np.nanmin((mat_1, mat_0)),
                          vmax=np.nanmax((mat_1, mat_0)))
        plt.sca(ax1)
        ax1.set_xlabel('Evidence')
        # ax1.set_ylabel('Prior')
        cbar_1 = plt.colorbar(im_1, fraction=0.04, pad=-0.05)
        if silent_comparison:
            cbar_1.set_label(r'$MT \; - MT_{silent}(ms)$')
        else:
            cbar_1.set_label(r'$MT \;(ms)$')
        ax1.set_yticks([0, 3, 6], ['', '', ''])
        ax1.set_xticks([0, 3, 6], ['L', '0', 'R'])
        ax0pos = ax0.get_position()
        ax1pos = ax1.get_position()
        ax0.set_position([ax0pos.x0, ax1pos.y0, ax1pos.width, ax1pos.height])
        ax1.set_position([ax1pos.x0-ax1pos.width*0.2,
                          ax1pos.y0, ax1pos.width, ax1pos.height])
    else:
        im_s = ax.imshow(np.flip(mat_s).T, cmap='RdGy')
        plt.sca(ax)
        cbar_s = plt.colorbar(im_s, fraction=0.04)
        cbar_s.set_label(r'$MT \;(ms)$')
        ax.set_yticks([0, 3, 6], ['1', '0', '-1'])
        ax.set_xticks([0, 3, 6], ['-1', '0', '1'])
        ax.set_xlabel('Prior Evidence')
        ax.set_ylabel('Stimulus Evidence')



def fig_1_rats_behav(df_data, task_img, sv_folder, figsize=(7, 9), margin=.05):
    mat_pright_all = np.zeros((7, 7))
    for subject in df_data.subjid.unique():
        df_sbj = df_data.loc[(df_data.special_trial == 0) &
                             (df_data.subjid == subject)]
        choice = df_sbj['R_response'].values
        coh = df_sbj['coh2'].values
        prior = df_sbj['norm_allpriors'].values
        indx = ~np.isnan(prior)
        mat_pright, _ = fp.com_heatmap(prior[indx], coh[indx], choice[indx],
                                    return_mat=True, annotate=False)
        mat_pright_all += mat_pright
    mat_pright = mat_pright_all / len(df_data.subjid.unique())
    f, ax = plt.subplots(nrows=4, ncols=3, figsize=figsize)  # figsize=(4, 3))
    ax = ax.flatten()
    labs = ['', '',  'c', '', '', 'd', 'e', 'f', 'g', '', '', '']
    for n, ax_1 in enumerate(ax):
        fp.rm_top_right_lines(ax_1)
        fp.add_text(ax=ax_1, letter=labs[n], x=-0.15, y=1.2)
    for i in [0, 1, 3]:
        ax[i].axis('off')
    # TASK PANEL
    ax_task = ax[0]
    pos = ax_task.get_position()
    factor = 1.75
    ax_task.set_position([pos.x0+0.05, pos.y0-0.05, pos.width*factor, pos.height*factor])
    task = plt.imread(task_img)
    ax_task.imshow(task)
    fp.add_text(ax=ax_task, letter='a', x=0.1, y=1.15)

    # P(RIGHT) MATRIX
    ax_pright = ax[4]
    im_2 = ax_pright.imshow(mat_pright, cmap='PRGn_r')
    pos = ax_pright.get_position()
    ax_pright.set_position([pos.x0-pos.width/1.6, pos.y0+margin/1.5, pos.width*.9,
                            pos.height*.9])
    cbar_ax = f.add_axes([pos.x0+pos.width/2.3, pos.y0+margin/2,
                          pos.width/10, pos.height/2])
    cbar_ax.set_title('p(Right)', fontsize=9)
    f.colorbar(im_2, cax=cbar_ax)
    ax_pright.set_yticks([0, 3, 6])
    ax_pright.set_ylim([-0.5, 6.5])
    ax_pright.set_yticklabels(['L', '', 'R'])
    ax_pright.set_xticks([0, 3, 6])
    ax_pright.set_xlim([-0.5, 6.5])
    ax_pright.set_xticklabels(['Left', '', 'Right'])
    ax_pright.set_xlabel('Prior Evidence')
    ax_pright.set_ylabel('Stimulus Evidence')
    fp.add_text(ax=ax_pright, letter='b', x=-0.2, y=1.15)

    # RTs
    ax_rts = ax[2]
    fp.rm_top_right_lines(ax=ax_rts)
    plot_rt_cohs_with_fb(df=df_data, ax=ax_rts, subj='LE46')
    ax_rts.set_xlabel('Reaction Time (ms)')
    ax_rts.set_ylabel('Density')
    ax_rts.set_xlim(-101, 201)
    ax_rts.axvline(x=0, linestyle='--', color='k', lw=0.5)
    pos = ax_rts.get_position()
    ax_rts.set_position([pos.x0, pos.y0+margin, pos.width, pos.height])
    fp.add_text(ax=ax_rts, letter='rat LE46', x=0.32, y=1., fontsize=8)

    # TACHOMETRICS
    bin_size = 10
    ax_tach = ax[5]
    labels = ['0', '0.25', '0.5', '1']
    tachometric(df_data, ax=ax_tach, fill_error=True, cmap='gist_yarg',
                labels=labels, rtbins=np.arange(0, 201, bin_size))
    ax_tach.set_xlabel('Reaction Time (ms)')
    ax_tach.set_ylabel('Accuracy')
    ax_tach.set_ylim(0.5, 1.04)
    ax_tach.set_xlim(-101, 201)
    ax_tach.axvline(x=0, linestyle='--', color='k', lw=0.5)
    ax_tach.set_yticks([0.5, 0.75, 1])
    ax_tach.set_yticklabels(['0.5', '0.75', '1'])
    fp.rm_top_right_lines(ax_tach)
    pos = ax_tach.get_position()
    ax_tach.set_position([pos.x0, pos.y0, pos.width, pos.height])
    fp.add_text(ax=ax_tach, letter='rat LE46', x=0.32, y=1., fontsize=8)

    # MT VS COH
    plot_mt_vs_evidence(df=df_data, ax=ax[7], prior_limit=0.1,  # 10% quantile
                        condition='choice_x_coh', rt_lim=50)
    # MT VS PRIOR
    plot_mt_vs_evidence(df=df_data.loc[df_data.special_trial == 2], ax=ax[6],
                        condition='choice_x_prior', prior_limit=1,
                        rt_lim=200)
    # REGRESSION WEIGHTS
    mt_weights(df_data, ax=ax[8], plot=True, means_errs=False)
    # MT MATRIX
    mt_matrix_ev_vs_zt(df=df_data, ax=ax[9], silent_comparison=False,
                          rt_bin=60, collapse_sides=True)
    # SLOWING
    plot_mt_vs_stim(df_data, ax[10], prior_min=0.8, rt_max=50)

    f.savefig(sv_folder+'fig1.svg', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'fig1.png', dpi=400, bbox_inches='tight')



def supp_fig_traj_tr_idx(df, sv_folder, fgsz=(15, 5), accel=False, marginx=0.01,
                         marginy=0.05):
    fgsz = fgsz
    inset_sz = 0.08
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=fgsz)
    ax = ax.flatten()
    ax_ti = np.array([ax[0], ax[1]])

    # trajs. conditioned on trial index
    ax_inset = fp.add_inset(ax=ax[0], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_ti = np.insert(ax_ti, 0, ax_inset)
    ax_inset = fp.add_inset(ax=ax[2], inset_sz=inset_sz, fgsz=fgsz,
                         marginx=marginx, marginy=marginy)
    ax_ti = np.insert(ax_ti, 2, ax_inset)
    for a in ax:
        fp.rm_top_right_lines(a)
    fp.plots_trajs_conditioned(df=df, ax=ax_ti, condition='choice_x_prior',
                            prior_limit=1, cmap='copper')
    # splits
    mt_weights(df, ax=ax[3], plot=True, means_errs=False)
    fp.trajs_splitting_stim(df=df, ax=ax[7])
    f.savefig(sv_folder+'/Fig2.png', dpi=400, bbox_inches='tight')
    f.savefig(sv_folder+'/Fig2.svg', dpi=400, bbox_inches='tight')


def plot_mt_weights_rt_bins(df, rtbins=np.linspace(0, 150, 16)):
    fig, ax = plt.subplots(nrows=2)
    for a in ax:
        fp.rm_top_right_lines(a)
    coh_weights = []
    zt_weights = []
    ti_weights = []
    coh_err = []
    zt_err = []
    ti_err = []
    for irt, rtbin in enumerate(rtbins[:-1]):
        df_rt = df.loc[(df.sound_len >= rtbin) &
                       (df.sound_len < rtbins[irt+1])]
        mean_sub, err_sub = mt_weights(df_rt, ax=None, plot=False,
                                       means_errs=True, mt=True, t_index_w=True)
        coh_weights.append(mean_sub[1])
        zt_weights.append(mean_sub[0])
        ti_weights.append(mean_sub[2])
        coh_err.append(err_sub[1])
        zt_err.append(err_sub[0])
        ti_err.append(err_sub[2])
    error_kws = dict(ecolor='goldenrod', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='goldenrod', marker='o', label='Prior')
    ax[0].errorbar(rtbins[:-1]+(rtbins[1]-rtbins[0])/2, zt_weights, zt_err,
                   **error_kws)
    error_kws = dict(ecolor='firebrick', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='firebrick', marker='o', label='Stimulus')
    ax[0].errorbar(rtbins[:-1]+(rtbins[1]-rtbins[0])/2, coh_weights, coh_err,
                   **error_kws)
    error_kws = dict(ecolor='steelblue', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='steelblue', marker='o', label='Trial index')
    ax[1].errorbar(rtbins[:-1]+(rtbins[1]-rtbins[0])/2, ti_weights, ti_err,
                   **error_kws)
    ax[0].set_ylabel('MT weight')
    ax[0].set_xlabel('RT (ms)')
    ax[0].legend()
    ax[1].set_ylabel('MT weight')
    ax[1].set_xlabel('RT (ms)')
    ax[1].legend()

