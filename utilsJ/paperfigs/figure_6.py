import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pylab as pl
import sys
from scipy import interpolate
from scipy.stats import sem
sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
sys.path.append("C:/Users/alexg/Onedrive/Documentos/GitHub/")  # Alex
# sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
sys.path.append("/home/molano/custom_utils") # Cluster Manuel
from utilsJ.paperfigs import figures_paper as fp
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.Models import analyses_humans as ah

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
                time = df.time_trajs.values[tr]
                ax.plot(time, traj, color=fig_3.COLOR_NO_COM, lw=.5)
                ax.set_xlim(-100, 800)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.1:
                    ax.plot(time*1e3, traj, color=fig_3.COLOR_NO_COM, lw=.5)
        elif tr < (ran_max/2-1) and coms[tr] and decision[tr] == 0:
            trial = df.iloc[tr]
            traj = trial['trajectory_y']
            if not human:
                time = df.time_trajs.values[tr]
                ax.plot(time, traj, color=fig_3.COLOR_COM, lw=2)
                ax.set_xlim(-100, 800)
            if human:
                time = np.array(trial['times'])
                if time[-1] < 0.3 and time[-1] > 0.2:
                    ax.plot(time*1e3, traj, color=fig_3.COLOR_COM, lw=2)
    fp.rm_top_right_lines(ax)
    if human:
        var = 'x'
    if not human:
        var = 'y'
    ax.set_ylabel('{}-coord (pixels)'.format(var))
    ax.set_xlabel('Time from movement \n onset (ms)')
    ax.axhline(y=max_val, linestyle='--', color='Green', lw=1)
    ax.axhline(y=-max_val, linestyle='--', color='Purple', lw=1)
    ax.axhline(y=0, linestyle='--', color='k', lw=0.5)


def com_statistics_humans(peak_com, time_com, ax, mean_mt):
    ax1, ax2 = ax
    fp.rm_top_right_lines(ax1)
    fp.rm_top_right_lines(ax2)
    ax1.hist(peak_com[peak_com != 0]/600*100, bins=67, range=(-100, -16.667),
             color=fig_3.COLOR_COM)
    ax1.hist(peak_com[peak_com != 0]/600*100, bins=14, range=(-16.667, 0),
             color=fig_3.COLOR_NO_COM)
    ax1.set_yscale('log')
    ax1.axvline(-100/6, linestyle=':', color='r')
    ax1.set_xlim(-100, 1)
    ax1.set_xlabel('Deflection point (%)')
    ax1.set_ylabel('# Trials')
    ax2.set_ylabel('# Trials')
    ax2.hist(time_com[time_com != -1]*1e3, bins=30, range=(0, 510),
             color=fig_3.COLOR_COM)
    ax2.set_xlabel('Deflection time (ms)')
    # ax3 = ax2.twiny()
    # ax3.set_xlim(ax2.get_xlim())
    # ax3.set_xticks([0, mean_mt*0.5, mean_mt, mean_mt*1.5],
    #                [0, 50, 100, 150])
    # ax3.spines['right'].set_visible(False)
    # ax3.set_xlabel('Proportion (%)')


def mean_com_traj_human(df_data, ax, max_mt=400):
    # TRAJECTORIES
    fp.rm_top_right_lines(ax=ax)
    index1 = (df_data.subjid != 5) & (df_data.subjid != 6)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    decision = df_data.R_response.values[index1]
    trajs = df_data.trajectory_y.values[index1]
    times = df_data.times.values[index1]
    com = df_data.CoM_sugg.values[index1]
    # congruent_coh = coh * (decision*2 - 1)
    precision = 16
    mat_mean_trajs_subjs = np.empty((len(df_data.subjid.unique()), max_mt))
    mat_mean_trajs_subjs[:] = np.nan
    for i_s, subj in enumerate(df_data.subjid.unique()):
        index = com & (df_data.subjid.values[index1] == subj)
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)
            max_time = max(time)*1e3
            if max_time > max_mt:
                continue
            all_trajs[tr, :len(vals)] = vals
            all_trajs[tr, len(vals):-1] = np.repeat(vals[-1],
                                                    int(max_mt-len(vals)-1))
        mean_traj = np.nanmean(all_trajs, axis=0)
        xvals = np.arange(len(mean_traj))*precision
        yvals = mean_traj
        ax.plot(xvals, yvals, color=fig_3.COLOR_COM, alpha=0.1)
        mat_mean_trajs_subjs[i_s, :] = yvals
    mean_traj_across_subjs = np.nanmean(mat_mean_trajs_subjs, axis=0)
    ax.plot(xvals, mean_traj_across_subjs, color=fig_3.COLOR_COM, linewidth=2)
    index = ~com
    all_trajs = np.empty((sum(index), max_mt))
    all_trajs[:] = np.nan
    for tr in range(sum(index)):
        vals = np.array(trajs[index][tr]) * (decision[index][tr]*2 - 1)
        ind_time = [True if t != '' else False for t in times[index][tr]]
        time = np.array(times[index][tr])[np.array(ind_time)].astype(float)
        max_time = max(time)*1e3
        if max_time > max_mt:
            continue
        all_trajs[tr, :len(vals)] = vals
        all_trajs[tr, len(vals):-1] = np.repeat(vals[-1],
                                                int(max_mt-len(vals)-1))
    mean_traj = np.nanmean(all_trajs, axis=0)
    xvals = np.arange(len(mean_traj))*precision
    yvals = mean_traj
    ax.plot(xvals, yvals, color=fig_3.COLOR_NO_COM, linewidth=2)
    ax.set_xlabel('Time from movement \n onset (ms)')
    ax.set_ylabel('Position')
    legendelements = [Line2D([0], [0], color=fig_3.COLOR_COM, lw=2, label='Rev.'),
                      Line2D([0], [0], color=fig_3.COLOR_NO_COM, lw=2, label='No-rev.')]
    ax.axhline(-100, color='r', linestyle=':')
    ax.set_xlim(-5, 415)
    ax.text(150, -200, 'Detection threshold', color='r', fontsize=10.5)
    ax.legend(handles=legendelements, loc='upper left', borderpad=0.1,
              labelspacing=0.01, bbox_to_anchor=(0, 1.1))


def human_trajs_cond(congruent_coh, decision, trajs, prior, bins, times, ax,
                     n_subjects, max_mt=400, max_px=800,
                     condition='prior', interpolatespace=np.arange(500)):
    """
    Plots trajectories conditioned on stimulus and prior congruency.
    For condition = 'prior', it will plot conditioning on prior congruency.
    For any other value of condition, it will plot conditioning on stimulus.
    """
    
    if condition == 'prior':
       colormap = pl.cm.copper_r(np.linspace(0., 1, len(bins)-1))[::-1]
       # colormap_2 = pl.cm.copper_r(np.linspace(0., 1, len(bins)-1))
    else:
        colormap = pl.cm.coolwarm(np.linspace(0., 1, len(bins)))[::-1]
        # colormap_2 = pl.cm.copper_r(np.linspace(0., 1, len(bins)-1))
        ev_vals = bins
        labels_stim = ['-1', ' ', ' ', '0', ' ', ' ', '1']
    mov_time_list = []
    for i_ev, ev in enumerate(bins):
        if condition == 'prior':
            if ev == 1:
                break
            index = (prior >= bins[i_ev])*(prior < bins[i_ev+1])

        else:
            index = (congruent_coh == ev) &\
                (np.abs(prior) <= np.quantile(np.abs(prior), 0.25))
        all_trajs = np.empty((sum(index), max_mt))
        all_trajs[:] = np.nan
        for tr in range(sum(index)):
            vals = np.array(trajs[index][tr])*(decision[index][tr]*2-1)
            ind_time = [True if t != '' else False for t in times[index][tr]]
            time = np.array(times[index][tr])[np.array(ind_time)].astype(float)*1e3
            f = interpolate.interp1d(time, vals, bounds_error=False)
            vals_in = f(interpolatespace)
            vals_in = vals_in[~np.isnan(vals_in)]
            max_time = max(time)
            if max_time > max_mt:
                continue
            all_trajs[tr, :len(vals_in)] = vals_in  # - vals[0]
            all_trajs[tr, len(vals_in):-1] = np.repeat(vals[-1],
                                                       int(max_mt - len(vals_in)-1))
        mean_traj = np.nanmean(all_trajs, axis=0)
        std_traj = np.nanstd(all_trajs, axis=0) / np.sqrt(sum(index))
        mov_time = np.nanmean(np.array([float(t[-1]) for t in
                                        times[index]
                                        if t[-1] != '']))*1e3
        err_traj = np.nanstd(np.array([float(t[-1]) for t in
                                        times[index]
                                        if t[-1] != '']))*1e3 / np.sqrt(sum(index))
        mov_time_list.append(mov_time)
        x_val = i_ev if condition == 'prior' else ev
        ax[1].errorbar(x_val, mov_time, err_traj, color=colormap[i_ev],
                       marker='o')
        xvals = np.arange(len(mean_traj))
        yvals = mean_traj
        if condition == 'prior':
            ax[0].plot(xvals[yvals <= max_px], mean_traj[yvals <= max_px],
                       color=colormap[i_ev])
        else:
            ax[0].plot(xvals[yvals <= max_px], mean_traj[yvals <= max_px],
                       color=colormap[i_ev], label='{}'.format(labels_stim[i_ev]))
        ax[0].fill_between(x=xvals[yvals <= max_px],
                           y1=mean_traj[yvals <= max_px]-std_traj[yvals <= max_px],
                           y2=mean_traj[yvals <= max_px]+std_traj[yvals <= max_px],
                           color=colormap[i_ev])
    x_vals = np.arange(5) if condition == 'prior' else ev_vals
    ax[1].plot(x_vals, mov_time_list, color='k', linestyle=':', alpha=0.6)
    ax[0].axhline(600, color='k', linestyle='--', alpha=0.4)
    ax[0].set_xlim(-0.1, 470)
    ax[0].set_ylim(-1, 620)
    ax[0].set_ylabel('Position')
    ax[0].set_xlabel('Time from movement \n onset (ms)')
    ax[1].set_xticks([])
    ax[1].set_title('MT (ms)', fontsize=10)
    if condition == 'prior':
        ax[1].set_xlabel('Prior')
        legendelements = [Line2D([0], [0], color=colormap[4], lw=1.5, label='cong.'),
                          Line2D([0], [0], color=colormap[3], lw=1.5, label=' '),
                          Line2D([0], [0], color=colormap[2], lw=1.5, label='0'),
                          Line2D([0], [0], color=colormap[1], lw=1.5, label=' '),
                          Line2D([0], [0], color=colormap[0], lw=1.5, label='inc.')]
        ax[1].set_ylim(180, 315)
        ax[1].set_xlim(-0.4, 4.4)
        ax[0].legend(handles=legendelements, title='Prior', loc='center left',
                     labelspacing=0.05, bbox_to_anchor=(0., 1.2))
    else:
        legendelements = [Line2D([0], [0], color=colormap[0], lw=1.5, label='1'),
                          Line2D([0], [0], color=colormap[1], lw=1.5, label=' '),
                          Line2D([0], [0], color=colormap[2], lw=1.5, label=' '),
                          Line2D([0], [0], color=colormap[3], lw=1.5, label='0'),
                          Line2D([0], [0], color=colormap[4], lw=1.5, label=' '),
                          Line2D([0], [0], color=colormap[5], lw=1.5, label=' '),
                          Line2D([0], [0], color=colormap[6], lw=1.5, label='-1')]
        ax[1].set_xlabel('Stimulus')
        ax[1].set_ylim(170, 285)
        ax[1].set_xlim(-1.2, 1.2)
        ax[0].legend(handles=legendelements, title='Stimulus', loc='center left',
                     labelspacing=0.05, bbox_to_anchor=(0.8, 1.2))


def human_trajs(df_data, ax, sv_folder, max_mt=400, max_px=800,
                interpolatespace=np.arange(500)):
    """
    Plots:
        - Human trajectories conditioned to stim and prior
        - Splitting time examples
        - Splitting time vs RT
    """
    # TRAJECTORIES
    index1 = (df_data.subjid != 5) & (df_data.subjid != 6) &\
             (df_data.sound_len <= 300) &\
             (df_data.sound_len >= 0)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    coh = df_data.avtrapz.values[index1]
    decision = df_data.R_response.values[index1]
    trajs = df_data.trajectory_y.values[index1]
    times = df_data.times.values[index1]
    sound_len = df_data.sound_len.values[index1]
    prior_cong = df_data['norm_allpriors'][index1] * (decision*2 - 1)
    prior_cong = prior_cong.values
    ev_vals = np.unique(np.round(coh, 2))
    subjects = df_data.subjid.values[index1]
    ground_truth = (df_data.R_response.values*2-1) *\
        (df_data.hithistory.values*2-1)
    ground_truth = ground_truth[index1]
    congruent_coh = np.round(coh, 2) * (decision*2 - 1)
    # Trajs conditioned on stimulus congruency
    human_trajs_cond(congruent_coh=congruent_coh, decision=decision,
                     trajs=trajs, prior=prior_cong, bins=ev_vals,
                     times=times, ax=ax[0:2],
                     n_subjects=len(df_data.subjid.unique()),
                     condition='stimulus', max_mt=400)
    bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
    # Trajs conditioned on prior congruency
    human_trajs_cond(congruent_coh=congruent_coh, decision=decision,
                     trajs=trajs, prior=prior_cong, bins=bins,
                     times=times, ax=ax[2:4],
                     n_subjects=len(df_data.subjid.unique()),
                     condition='prior', max_mt=400)
    # extract splitting time
    out_data, rtbins = splitting_time_humans(sound_len=sound_len, coh=coh,
                                             trajs=trajs, times=times, subjects=subjects,
                                             ground_truth=ground_truth,
                                             interpolatespace=interpolatespace,
                                             max_mt=max_mt)
    # plot splitting time vs RT
    splitting_time_plot(sound_len=sound_len, out_data=out_data,
                        ax=ax[-1], subjects=subjects)
    rtbins = np.array((rtbins[0], rtbins[1], rtbins[2]))
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    # plot splitting time examples
    splitting_time_example_human(rtbins=rtbins, ax=ax, sound_len=sound_len,
                                 ground_truth=ground_truth, coh=coh, trajs=trajs,
                                 times=times, max_mt=max_mt,
                                 interpolatespace=interpolatespace,
                                 colormap=colormap)


def plot_xy(df_data, ax):
    """
    Plots raw trajectories in x-y
    """
    cont = 0
    subj_xy = 1
    index_sub = df_data.subjid == subj_xy
    ax.scatter(-500, 400, s=1100, color='grey', alpha=0.2)
    ax.scatter(500, 400, s=1100, color='grey', alpha=0.2)
    ax.scatter(0, -200, s=600, color='grey', alpha=0.8)
    for traj in range(800):
        # np.random.seed(1)
        tr_ind = np.random.randint(0, len(df_data['trajectory_y'][index_sub])-1)
        x_coord = df_data['trajectory_y'][tr_ind]
        y_coord = df_data['traj_y'][tr_ind]
        time_max = df_data['times'][tr_ind][-1]
        if time_max != '':
            if time_max < 0.3 and time_max > 0.1 and not df_data.CoM_sugg[tr_ind]:
                time = df_data['times'][tr_ind]
                ind_time = [True if t != '' else False for t in time]
                time = np.array(time)[np.array(ind_time)]
                ax.plot(x_coord, y_coord, color='grey', lw=.5, alpha=0.6)
                # ax[5].plot(time*1e3, x_coord, color='k', linewidth=0.5)
                cont += 1
        if cont == 50:
            break
    ax.set_xlabel('Position along x-axis')
    ax.set_ylabel('Position along y-axis')


def splitting_time_plot(sound_len, out_data, ax, subjects):
    rtbins = np.concatenate(([0], np.quantile(sound_len, [.25, .50, .75, 1])))
    xvals = []
    for irtb, rtb in enumerate(rtbins[:-1]):
        sound_len_bin = sound_len[(sound_len >= rtb) &
                                  (sound_len < rtbins[irtb+1])]
        rtbins_window = np.median(sound_len_bin)
        xvals.append(rtbins_window)
    # xplot = rtbins[:-1] + np.diff(rtbins)/2
    out_data = np.array(out_data).reshape(np.unique(subjects).size,
                                          rtbins.size-1, -1)
    out_data = np.swapaxes(out_data, 0, 1)
    out_data = out_data.astype(float)
    # binsize = np.diff(rtbins)
    ax2 = ax
    # fig2, ax2 = plt.subplots(1)
    for i in range(len(np.unique(subjects))):
        for j in range(out_data.shape[2]):
            ax2.plot(xvals,
                     out_data[:, i, j],
                     marker='o', mfc=(.6, .6, .6, .3), mec=(.6, .6, .6, 1),
                     mew=1, color=(.6, .6, .6, .3))
    error_kws = dict(ecolor='firebrick', capsize=2, mfc=(1, 1, 1, 0), mec='k',
                     color='firebrick', marker='o', label='mean & SEM')
    ax2.errorbar(xvals,
                 np.nanmedian(out_data.reshape(rtbins.size-1, -1), axis=1),
                 yerr=sem(out_data.reshape(rtbins.size-1, -1),
                          axis=1, nan_policy='omit'), **error_kws)
    ax2.set_xlabel('Reaction time (ms)')
    ax2.set_title('Impact of stimulus', fontsize=11.5)
    # ax2.set_xticks([107, 128], labels=['Early RT', 'Late RT'], fontsize=9)
    # ax2.set_ylim(190, 410)
    ax2.plot([0, 310], [0, 310], color='k')
    ax2.fill_between([0, 310], [0, 310], [0, 0],
                     color='grey', alpha=0.6)
    ax2.set_ylabel('Splitting time (ms)')
    fp.rm_top_right_lines(ax2)


def splitting_time_example_human(rtbins, ax, sound_len, ground_truth, coh, trajs,
                                 times, max_mt, interpolatespace, colormap):
    # to plot trajectories with splitting time: (all subjects together)
    ev_vals = np.array([0, 0.25, 0.5, 1])
    rtbins = np.concatenate(([0], np.quantile(sound_len, [.3333])))
    labs = ['Short RT', 'Long RT']
    for i in range((rtbins.size-1)*2):
        if i >= rtbins.size-1:
            rtbins = np.concatenate(([0],
                                     np.quantile(sound_len, [.6666, 1.])))
        ax1 = ax[-3+i]
        for i_ev, ev in enumerate(ev_vals):
            index = (sound_len < rtbins[i+1]) & (sound_len >= rtbins[i]) &\
                    (np.abs(np.round(coh, 2)) == ev)
            all_trajs = np.empty((sum(index), int(max_mt+300)))
            all_trajs[:] = np.nan
            for tr in range(sum(index)):
                vals = np.array(trajs[index][tr]) * (ground_truth[index][tr])
                ind_time = [True if t != '' else False
                            for t in times[index][tr]]
                time = np.array(times[index][tr])[
                    np.array(ind_time)].astype(float)*1e3
                f = interpolate.interp1d(time, vals, bounds_error=False)
                vals_in = f(interpolatespace)
                vals_in = vals_in[~np.isnan(vals_in)]
                vals_in -= vals_in[0]
                vals_in = np.concatenate((np.zeros((int(sound_len[index][tr]))),
                                          vals_in))
                max_time = max(time)
                if max_time > max_mt:
                    continue
                all_trajs[tr, :len(vals_in)] = vals_in  # - vals[0]
                all_trajs[tr, len(vals_in):-1] =\
                    np.repeat(vals[-1], int(max_mt + 300 - len(vals_in)-1))
            if ev == 0:
                ev_mat = np.repeat(0, sum(index))
                traj_mat = all_trajs
            else:
                ev_mat = np.concatenate((ev_mat, np.repeat(ev, sum(index))))
                traj_mat = np.concatenate((traj_mat, all_trajs))
            ax1.plot(np.arange(len(np.nanmean(all_trajs, axis=0))),
                     np.nanmean(all_trajs, axis=0),
                     color=colormap[i_ev])
        ax1.set_xlim(-5, 405)
        ax1.set_ylim(-2, 400)
        ax1.set_title(labs[i], fontsize=11.5)
        ind = fig_2.get_split_ind_corr(traj_mat, ev_mat, startfrom=0,
                                       max_MT=max_mt+300, pval=0.01)
        ax1.set_xlabel('Time from stimulus \n onset (ms)')
        ax1.set_ylabel('Position')
        if i == 0:
            ax1.arrow(ind, 35, 0, 25, color='k', width=1.5, head_width=15,
                      head_length=15)
            ax1.text(ind-30, 10, 'Splitting Time', fontsize=10)
            labels = ['0', '0.25', '0.5', '1']
            legendelements = []
            for i_l, lab in enumerate(reversed(labels)):
                legendelements.append(Line2D([0], [0], color=colormap[::-1][i_l], lw=2,
                                      label=lab))
            ax1.legend(handles=legendelements, fontsize=9, loc='upper left',
                       title='Stimulus', labelspacing=0.01)
        else:
            if np.isnan(ind):
                ind = rtbins[i]
            ax1.arrow(ind, 110, 0, -65, color='k', width=1.5, head_width=15,
                      head_length=15)
            ax1.text(ind-150, 140, 'Splitting Time', fontsize=10)


def splitting_time_humans(sound_len, coh, trajs, times, subjects, ground_truth,
                          interpolatespace, max_mt):
    # splitting time computation
    rtbins = np.concatenate(([0], np.quantile(sound_len, [.25, .50, .75, 1])))
    split_ind = []
    ev_vals = [0, 0.25, 0.5, 1]
    for subj in np.unique(subjects):
        for i in range(rtbins.size-1):
            # fig, ax1 = plt.subplots(1)
            for i_ev, ev in enumerate(ev_vals):
                index = (sound_len < rtbins[i+1]) & (sound_len >= rtbins[i]) &\
                        (np.abs(np.round(coh, 2)) == ev) &\
                        (subjects == subj)  # & (prior <= 0.3)
                all_trajs = np.empty((sum(index), int(max_mt+300)))
                all_trajs[:] = np.nan
                for tr in range(sum(index)):
                    vals = np.array(trajs[index][tr]) * (ground_truth[index][tr])
                    ind_time = [True if t != '' else False
                                for t in times[index][tr]]
                    time = np.array(times[index][tr])[
                        np.array(ind_time)].astype(float)*1e3
                    f = interpolate.interp1d(time, vals, bounds_error=False)
                    vals_in = f(interpolatespace)
                    vals_in = vals_in[~np.isnan(vals_in)]
                    vals_in -= vals_in[0]
                    vals_in = np.concatenate((np.zeros((int(sound_len[index][tr]))),
                                              vals_in))
                    max_time = max(time)
                    if max_time > max_mt:
                        continue
                    all_trajs[tr, :len(vals_in)] = vals_in  # - vals[0]
                    all_trajs[tr, len(vals_in):-1] =\
                        np.repeat(vals[-1], int(max_mt + 300 - len(vals_in)-1))
                if ev == 0:
                    ev_mat = np.repeat(0, sum(index))
                    traj_mat = all_trajs
                else:
                    ev_mat = np.concatenate((ev_mat, np.repeat(ev, sum(index))))
                    traj_mat = np.concatenate((traj_mat, all_trajs))
                # ax1.plot(np.arange(len(np.nanmean(traj_mat, axis=0)))*16,
                #          np.nanmean(traj_mat, axis=0),
                #          color=colormap[i_ev])
                # ax1.set_xlim(0, 650)
                # ax1.set_title('{} < RT < {}'.format(rtbins[i], rtbins[i+1]))
            ind = fig_2.get_split_ind_corr(traj_mat, ev_mat, startfrom=0,
                                           max_MT=max_mt+300, pval=0.01)
            if ind < 410:
                split_ind.append(ind)
            else:
                # ind = get_split_ind_corr(traj_mat, ev_mat, startfrom=0,
                #                          max_MT=500, pval=0.001)
                # if ind > 410:
                split_ind.append(np.nan)
            # ax1.axvline(ind*16, color='r')
    out_data = np.array(split_ind)
    return out_data, rtbins


def fig_6_humans(user_id, human_task_img, sv_folder, nm='300',
                 max_mt=600, inset_sz=.06, marginx=0.004, marginy=0.025,
                 fgsz=(11, 13.5)):
    if user_id == 'alex':
        folder = 'C:\\Users\\alexg\\Onedrive\\Escritorio\\CRM\\Human\\80_20\\'+nm+'ms\\'
    if user_id == 'alex_CRM':
        folder = 'C:/Users/agarcia/Desktop/CRM/human/'
    if user_id == 'idibaps':
        folder =\
            '/home/molano/Dropbox/project_Barna/psycho_project/80_20/'+nm+'ms/'
    if user_id == 'idibaps_alex':
        folder = '/home/jordi/DATA/Documents/changes_of_mind/humans/'+nm+'ms/'
    subj = ['general_traj']
    steps = [None]
    humans = True
    # retrieve data
    df_data = ah.traj_analysis(data_folder=folder,
                               subjects=subj, steps=steps, name=nm,
                               sv_folder=sv_folder)
    df_data.avtrapz /= max(abs(df_data.avtrapz))
    # create figure
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=fgsz)
    ax = ax.flatten()
    plt.subplots_adjust(top=0.95, bottom=0.09, left=0.09, right=0.95,
                        hspace=0.6, wspace=0.5)
    labs = ['', '', '',  'b', 'c', 'd', 'e', 'f', 'g', '', 'h', 'i', 'j', 'k',
            'l', '']
    for n, ax_1 in enumerate(ax):
        fp.rm_top_right_lines(ax_1)
        if n == 13:
            ax_1.text(-0.1, 3, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
        elif n == 0:
            ax_1.text(-0.1, 1.15, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
        elif n == 4:
            ax_1.text(-0.1, 1.2, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')     
        elif n == 14:
            ax_1.text(-0.1, 1.4, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')            
        else:
            ax_1.text(-0.1, 1.2, labs[n], transform=ax_1.transAxes, fontsize=16,
                      fontweight='bold', va='top', ha='right')
    for i in [0, 1, 2]:
        ax[i].axis('off')
    # TASK PANEL
    ax_task = ax[0]
    pos_ax_0 = ax_task.get_position()
    # setting ax0 a bit bigger
    ax_task.set_position([pos_ax_0.x0-0.04, pos_ax_0.y0-0.02,
                          pos_ax_0.width*4, pos_ax_0.height+0.024])
    
    pos = ax_task.get_position()
    ax_task.set_position([pos.x0, pos.y0, pos.width, pos.height])
    task = plt.imread(human_task_img)
    ax_task.imshow(task)
    ax_task.text(0.08, 1.3, 'a', transform=ax_task.transAxes, fontsize=16,
                 fontweight='bold', va='top', ha='right')

    # changing ax x-y plot width
    pos_ax_1 = ax[2].get_position()
    ax[3].set_position([pos_ax_1.x0 + pos_ax_1.width, pos_ax_1.y0,
                        pos_ax_1.width+pos_ax_1.width/3, pos_ax_1.height])
    # plotting x-y trajectories
    plot_xy(df_data=df_data, ax=ax[3])
    # tachs and pright
    ax_tach = ax[5]
    ax_pright = ax[4]
    ax_mat = [ax[14], ax[15]]
    pos_com_0 = ax_mat[0].get_position()
    ax_mat[0].set_position([pos_com_0.x0 + pos_com_0.width*0.3, pos_com_0.y0,
                            pos_com_0.width, pos_com_0.height])
    ax_mat[1].set_position([pos_com_0.x0 + pos_com_0.width*1.4, pos_com_0.y0,
                            pos_com_0.width, pos_com_0.height])
    fig_3.matrix_figure(df_data=df_data, ax_tach=ax_tach, ax_pright=ax_pright,
                  ax_mat=ax_mat, humans=humans)
    pos_com_0 = ax_mat[0].get_position()
    pos_com_1 = ax_mat[1].get_position()
    ax_mat[0].set_position([pos_com_0.x0, pos_com_1.y0,
                            pos_com_1.width, pos_com_1.height])
    # plots CoM trajectory examples
    ax_examples_com = ax[11]
    plot_coms(df=df_data, ax=ax_examples_com, human=humans)
    # prepare data for CoM peak/time distros plot
    peak_com = -df_data.com_peak.values
    time_com = df_data.time_com.values
    ax_com_stat = ax[13]
    pos = ax_com_stat.get_position()
    ax_com_stat.set_position([pos.x0, pos.y0, pos.width,
                              pos.height*2/5])
    ax_inset = plt.axes([pos.x0, pos.y0+pos.height*3.35/5, pos.width,
                         pos.height*2/5])
    ax_coms = [ax_com_stat, ax_inset]
    # CoM peak/time distributions
    mean_mt = np.median(fp.get_human_mt(df_data.loc[~df_data.CoM_sugg]))
    com_statistics_humans(peak_com=peak_com, time_com=time_com, ax=ax_coms,
                          mean_mt=mean_mt)
    # mean CoM trajectories
    mean_com_traj_human(df_data=df_data, ax=ax[12])
    # prepare axis for trajs conditioned on stim and prior
    ax_cohs = ax[7]
    ax_zt = ax[6]
    # trajs. conditioned on coh
    ax_inset = fp.add_inset(ax=ax_cohs, inset_sz=inset_sz, fgsz=(1, 1),
                         marginx=marginx, marginy=marginy, right=True)
    ax_cohs = np.insert(ax_cohs, 0, ax_inset)
    # trajs. conditioned on zt
    ax_inset = fp.add_inset(ax=ax_zt, inset_sz=inset_sz, fgsz=(1, 1),
                         marginx=marginx, marginy=marginy, right=True)
    ax_zt = np.insert(ax_zt, 0, ax_inset)
    axes_trajs = [ax_cohs[1], ax_cohs[0], ax_zt[1], ax_zt[0], ax[8],
                  ax[9], ax[10]]
    # trajectories conditioned on stim/prior, splitting time (vs RT and example)
    human_trajs(df_data, sv_folder=sv_folder, ax=axes_trajs, max_mt=max_mt)
    fig.savefig(sv_folder+'fig6.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'fig6.png', dpi=400, bbox_inches='tight')
