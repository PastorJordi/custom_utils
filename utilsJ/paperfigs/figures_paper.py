# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:24:12 2022
@author: Alex Garcia-Duran
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys, os
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from scipy.stats import ttest_rel
from matplotlib.lines import Line2D
from statsmodels.stats.proportion import proportion_confint
from skimage import exposure
import scipy
# from scipy import interpolate
# import shutil

# sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
# sys.path.append("/home/molano/custom_utils") # Cluster Manuel

# from utilsJ.Models import simul
from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.Behavior.plotting import binned_curve, tachometric
from utilsJ.Behavior.plotting import trajectory_thr, interpolapply
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
from utilsJ.Models import analyses_humans as ah
import matplotlib
import matplotlib.pylab as pl



matplotlib.rcParams['font.size'] = 9
plt.rcParams['legend.title_fontsize'] = 8
plt.rcParams['xtick.labelsize']= 8
plt.rcParams['ytick.labelsize']= 8
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
pc_name = 'alex_CRM'
if pc_name == 'alex':
    RAT_COM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/001965.png'
    SV_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/'  # Alex
    DATA_FOLDER = 'C:/Users/alexg/Onedrive/Escritorio/CRM/data/'  # Alex
    RAT_noCOM_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/screenShot230120.png'
    TASK_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/panel_a.png'
    HUMAN_TASK_IMG = 'C:/Users/Alexandre/Desktop/CRM/rat_image/g41085.png'
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
elif pc_name == 'idibaps_alex':
    SV_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/'  # Jordi
    DATA_FOLDER = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/'  # Jordi
    RAT_COM_IMG = '/home/jordi/Documents/changes_of_mind/demo/materials/' +\
        'craft_vid/CoM/a/001965.png'
    RAT_noCOM_IMG = '/home/jordi/DATA/Documents/changes_of_mind/data_clean/' +\
        'screenShot230120.png'
    HUMAN_TASK_IMG = '/home/jordi/DATA/Documents/changes_of_mind/humans/g41085.png'
elif pc_name == 'alex_CRM':
    SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
    DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
    RAT_COM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/001965.png'
    RAT_noCOM_IMG = 'C:/Users/agarcia/Desktop/CRM/proves/screenShot230120.png'
    HUMAN_TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/rat_image/g41085.png'
    TASK_IMG = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/panel_a.png'

FRAME_RATE = 14
BINS_RT = np.linspace(1, 301, 11)
xpos_RT = int(np.diff(BINS_RT)[0])
COLOR_COM = 'coral'
COLOR_NO_COM = 'tab:cyan'

def plot_fixation_breaks_single(subject, ax):
    path = DATA_FOLDER + subject + '_clean.pkl'
    df = pd.read_pickle(path)
    hist_edges = np.linspace(-0.3, 0.4, 71)
    rt_density_per_session =\
        np.vstack(df.groupby('sessid').apply(lambda x: np.concatenate(
            [x.sound_len/1000, np.concatenate(x.fb.values)-0.3]))
            .apply(lambda x: np.histogram(x, density=True, bins=hist_edges)[0])
            .values)
    ax.errorbar(hist_edges[:-1]+0.005, rt_density_per_session.mean(axis=0),
                np.std(rt_density_per_session, axis=0))
    ax.axvline(0, c='r')
    ax.set_xlabel('RT (s)')
    ax.set_ylabel('mean density (+/- std)')
    # plt.show()


def plot_rt_all_rats(subjects):
    fig, ax = plt.subplots(nrows=5, ncols=3)
    ax = ax.flatten()
    for i_s, subject in enumerate(subjects):
        plot_fixation_breaks_single(subject=subject, ax=ax[i_s])


def rm_top_right_lines(ax, right=True):
    if right:
        ax.spines['right'].set_visible(False)
    else:
        ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)


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


def add_inset(ax, inset_sz=0.2, fgsz=(4, 8), marginx=0.01, marginy=0.05,
              right=True):
    # adds inset to an axis
    ratio = fgsz[0]/fgsz[1]
    pos = ax.get_position()
    ax_inset = plt.axes([pos.x1-inset_sz-marginx, pos.y0+marginy, inset_sz,
                         inset_sz*ratio])
    rm_top_right_lines(ax_inset, right=right)
    return ax_inset


def binning_mt_prior(df, bins):
    # matrix with rows for subjects and columns for bins
    mat_mt = np.empty((len(df.subjid.unique()), len(bins)-1))
    for i_s, subject in enumerate(df.subjid.unique()):
        df_sub = df.loc[df.subjid == subject]
        for bin in range(len(bins)-1):
            mt_sub = df_sub.loc[(df_sub.choice_x_prior >= bins[bin]) &
                                (df_sub.choice_x_prior < bins[bin+1]), 'resp_len']
            mat_mt[i_s, bin] = np.nanmean(mt_sub)
            if np.isnan(mat_mt[i_s, bin]):
                print(1)
    return mat_mt  # if you want mean across subjects, np.nanmean(mat_mt, axis=0)


def get_bin_info(df, condition, prior_limit=0.25, after_correct_only=True, rt_lim=50,
                 fpsmin=29, num_bins_prior=5):
    # after correct condition
    ac_cond = df.aftererror == False if after_correct_only else (df.aftererror*1) >= 0
    # common condition 
    # put together all common conditions: prior, reaction time and framerate
    common_cond = ac_cond & (df.norm_allpriors.abs() <= prior_limit) &\
        (df.sound_len < rt_lim) & (df.framerate >= fpsmin)
    # define bins, bin type, trajectory index and colormap depending on condition
    if condition == 'choice_x_coh':
        bins = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        bintype = 'categorical'
        indx_trajs = common_cond & (df.special_trial == 0) 
        n_iters = len(bins)
        colormap = pl.cm.coolwarm(np.linspace(0., 1, n_iters))
    elif condition == 'choice_x_prior':
        indx_trajs = common_cond & (df.special_trial == 2)
        bins_zt = [-1.01]
        percentiles = [1/num_bins_prior*i for i in range(1, num_bins_prior)]
        for perc in percentiles:
            bins_zt.append(df.loc[indx_trajs, 'choice_x_prior'].quantile(perc))
        bins_zt.append(1.01)
        bins = np.array(bins_zt)
        bintype = 'edges'
        n_iters = len(bins)-1
        colormap = pl.cm.copper(np.linspace(0., 1, n_iters))
    elif condition == 'origidx':
        bins = np.linspace(0, 1e3, num=6)
        bintype = 'edges'
        n_iters = len(bins) - 1
        indx_trajs = common_cond & (df.special_trial == 0)
        colormap = pl.cm.jet(np.linspace(0., 1, n_iters))
    return bins, bintype, indx_trajs, n_iters, colormap



def tachometric_data(coh, hit, sound_len, subjid, ax, label='Data'):
    rm_top_right_lines(ax)
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len, 'subjid': subjid})
    tachometric(df_plot_data, ax=ax, fill_error=True, cmap='gist_yarg')
    ax.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title(label)
    ax.set_ylim(0.24, 1.04)
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, 4))
    legendelements = [Line2D([0], [0], color=colormap[0], lw=2,
                             label='0'),
                      Line2D([0], [0], color=colormap[1], lw=2,
                             label='0.25'),
                      Line2D([0], [0], color=colormap[2], lw=2,
                             label='0.5'),
                      Line2D([0], [0], color=colormap[3], lw=2,
                             label='1')]
    ax.legend(handles=legendelements, fontsize=7)
    # ax.legend([1, 0.5, 0.25, 0])
    return ax.get_position()


def pdf_cohs(df, ax, bins=np.linspace(0, 200, 41), yaxis=True):
    # ev_vals = np.unique(np.abs(coh))
    sound_len = df.sound_len.values
    coh = df.coh2.values
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, 4))
    num_subjs = len(df.subjid.unique())
    for i_coh, ev in enumerate([0, 0.25, 0.5, 1]):
        counts_all_rats = np.zeros((len(bins)-1, num_subjs))
        for i_s, subj in enumerate(df.subjid.unique()):
            index = (np.abs(coh) == ev) & (df.subjid == subj)
            counts_coh, bins_coh = np.histogram(sound_len[index], bins=bins)
            norm_counts = counts_coh/sum(counts_coh)
            counts_all_rats[:, i_s] = norm_counts
        norm_counts = np.nanmean(counts_all_rats, axis=1)
        error = np.nanstd(counts_all_rats, axis=1)/np.sqrt(num_subjs)
        xvals = bins_coh[:-1]+(bins_coh[1]-bins_coh[0])/2
        ax.plot(xvals, norm_counts, color=colormap[i_coh], label=str(ev))
        ax.fill_between(xvals, norm_counts-error, norm_counts+error,
                        color=colormap[i_coh], alpha=0.4)
    ax.set_xlabel('Reaction time (ms)')
    if yaxis:
        ax.set_ylabel('RT density')
    ax.legend()


# function to add letters to panel
def add_text(ax, letter, x=-0.1, y=1.2, fontsize=16):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='right')


def groupby_binom_ci(x, method="beta"):
    # so we can plot groupby with errorbars in binomial vars in 2 lines
    return [abs(x.mean() - ci) for ci in
            proportion_confint(x.sum(), len(x), method=method)]


def tachs_values(df, evidence_bins=np.array([0, 0.15, 0.30, 0.60, 1.05]),
                 rtbins=np.arange(0, 151, 3), rt='sound_len',
                 evidence='avtrapz', hits='hithistory'):
    rtbinsize = rtbins[1]-rtbins[0]
    tmp_df = df
    tmp_df['rtbin'] = pd.cut(
        tmp_df[rt], rtbins, labels=np.arange(rtbins.size-1),
        retbins=False, include_lowest=True, right=True).astype(float)
    xvals = np.zeros((len(rtbins)-1, len(evidence_bins)-1))
    yvals = np.zeros((len(rtbins)-1, len(evidence_bins)-1))
    yerr = np.zeros((len(rtbins)-1, len(evidence_bins)-1))
    n_subjs = len(df.subjid.unique())
    vals_all_rats = np.zeros((len(rtbins)-1, n_subjs))
    for i in range(evidence_bins.size-1):
        for i_s, subj in enumerate(df.subjid.unique()):
            tmp = (tmp_df.loc[(tmp_df[evidence].abs() >= evidence_bins[i]) & (
                   tmp_df[evidence].abs() < evidence_bins[i+1]) &
                   (tmp_df.subjid == subj)]
                   .groupby('rtbin')[hits].agg(['mean',
                                                groupby_binom_ci]).reset_index())
            vals_all_rats[:len(tmp['mean'].values), i_s] = tmp['mean'].values
        xvals[:len(tmp.rtbin.values), i] =\
            tmp.rtbin.values * rtbinsize + 0.5 * rtbinsize
        yvals[:, i] = np.nanmean(vals_all_rats, axis=1)
        yerr[:, i] = np.nanstd(vals_all_rats, axis=1) / n_subjs
    xvals = xvals[:len(tmp['mean'].values), :]
    yvals = yvals[:len(tmp['mean'].values), :]
    return xvals, yvals, yerr


def tach_1st_2nd_choice(df, ax, model=False, tachometric=False):
    # TODO: average across rats
    choice = df.R_response.values * 2 - 1
    coh = df.coh2.values
    gt = df.rewside.values * 2 - 1
    hit = df.hithistory.values
    sound_len = df.sound_len.values
    subj = df.subjid.values
    if not model:
        com = df.CoM_sugg.values
    if model:
        com = df.com_detected.values
    choice_com = choice
    choice_com[com] = -choice[com]
    hit_com = choice_com == gt
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len, 'subjid': subj})
    if tachometric:
        xvals, yvals1, _ = tachs_values(df=df_plot_data,
                                        rtbins=np.arange(0, 151, 3))
        colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
        for j in range(4):
            ax.plot(xvals[:, j], yvals1[:, j], color=colormap[j], linewidth=1.5)
        df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit_com,
                                     'sound_len': sound_len, 'subjid': subj})
        xvals, yvals2, _ = tachs_values(df=df_plot_data,
                                        rtbins=np.arange(0, 151, 3))
        for j in range(4):
            ax.plot(xvals[:, j], yvals2[:, j], color=colormap[j], linestyle='--',
                    linewidth=1.5)
            ax.fill_between(xvals[:, j], yvals1[:, j], yvals2[:, j],
                            color=fig_3.COLOR_COM, alpha=0.8)
        ax.set_xlabel('RT (ms)')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.3, 1)
        legendelements = [Line2D([0], [0], linestyle='--', color='k', lw=2,
                                 label='initial trajectory'),
                          Line2D([0], [0], color='k', lw=2, label='final response')]
        ax.legend(handles=legendelements)
    else:
        mean_com = []
        mean_nocom = []
        err_com = []
        err_nocom = []
        nsubs = len(df.subjid.unique())
        ev_vals = [0, 0.25, 0.5, 1]
        pvals = [1e-2, 1e-3, 1e-4]
        for i_ev, ev in enumerate(ev_vals):
            mean_x_subj_com = []
            mean_x_subj_nocom = []
            # pv_per_sub = []
            for i_s, subj in enumerate(df.subjid.unique()):
                indx = (coh == ev) & (df.subjid == subj)
                h_nocom = hit_com[indx]
                h_com = hit[indx]
                mean_x_subj_com.append(np.nanmean(h_com))
                mean_x_subj_nocom.append(np.nanmean(h_nocom))
            _, pv = ttest_rel(mean_x_subj_com, mean_x_subj_nocom)
            if pv < pvals[0] and pv > pvals[1]:
                ax.text(ev-0.02, np.nanmean(h_com)+0.05, '*', fontsize=10)
            if pv < pvals[1] and pv > pvals[2]:
                ax.text(ev-0.02, np.nanmean(h_com)+0.05, '**', fontsize=10)
            if pv < pvals[2]:
                ax.text(ev-0.02, np.nanmean(h_com)+0.05, '***', fontsize=10)
            mean_com.append(np.nanmean(mean_x_subj_com))
            mean_nocom.append(np.nanmean(mean_x_subj_nocom))
            err_com.append(np.nanstd(mean_x_subj_com)/np.sqrt(nsubs))
            err_nocom.append(np.nanstd(mean_x_subj_nocom)/np.sqrt(nsubs))
        ax.errorbar(ev_vals, mean_com, yerr=err_com, marker='o', color=fig_3.COLOR_COM,
                    label='Final trajectory', markersize=5)
        ax.errorbar(ev_vals, mean_nocom, yerr=err_nocom, marker='o',
                    color=fig_3.COLOR_NO_COM, label='Initial trajectory', markersize=5)
        ax.set_xlabel('Stimulus evidence')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right')
        ax.set_xticks(ev_vals)


def supp_trajs_prior_cong(df_sim, ax=None):
    signed_response = df_sim.R_response.values
    nanidx = df_sim.loc[df_sim[['dW_trans',
                                'dW_lat']].isna().sum(axis=1) == 2].index
    df_sim['allpriors'] = np.nansum(df_sim[['dW_trans', 'dW_lat']].values, axis=1)
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
    ax.set_ylabel('Position (px)', fontsize=10)
    ax.set_xlabel('Time from movement onset (ms)', fontsize=10)


def basic_statistics(decision, resp_fin):
    mat = confusion_matrix(decision, resp_fin)
    print(mat)
    fpr, tpr, _ = roc_curve(resp_fin, decision)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()


def run_model(stim, zt, coh, gt, trial_index, human=False,
              subject=None, num_tr=None, load_params=True):
    # dt = 5e-3
    if num_tr is not None:
        num_tr = num_tr
    else:
        num_tr = int(len(zt))
    data_augment_factor = 10
    if not human:
        detect_CoMs_th = 8
    if human:
        detect_CoMs_th = 200
    if not load_params:
        p_t_aff = 5
        p_t_eff = 4
        p_t_a = 14  # 90 ms (18) PSIAM fit includes p_t_eff
        p_w_zt = 0.1
        p_w_stim = 0.08
        p_e_bound = .6
        p_com_bound = 0.1
        p_w_a_intercept = 0.056
        p_w_a_slope = 2e-5  # fixed
        p_a_bound = 2.6  # fixed
        p_1st_readout = 50
        p_2nd_readout = 90
        p_leak = 0.5
        p_mt_noise = 35
        p_MT_intercept = 320
        p_MT_slope = 0.07
        conf = [p_w_zt, p_w_stim, p_e_bound, p_com_bound, p_t_aff,
                p_t_eff, p_t_a, p_w_a_intercept, p_w_a_slope, p_a_bound, p_1st_readout,
                p_2nd_readout, p_leak, p_mt_noise, p_MT_intercept, p_MT_slope]
        jitters = len(conf)*[0]
    else:
        if human:
            conf = np.load(SV_FOLDER +
                           'parameters_MNLE_BADS_human_subj_' + str(subject) + '.npy')
        else:
            conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        jitters = len(conf)*[0]
    print('Number of trials: ' + str(stim.shape[1]))
    p_w_zt = conf[0]+jitters[0]*np.random.rand()
    p_w_stim = conf[1]+jitters[1]*np.random.rand()
    p_e_bound = conf[2]+jitters[2]*np.random.rand()
    p_com_bound = conf[3]*p_e_bound+jitters[3]*np.random.rand()
    p_t_aff = int(round(conf[4]+jitters[4]*np.random.rand()))
    p_t_eff = int(round(conf[5]++jitters[5]*np.random.rand()))
    p_t_a = int(round(conf[6]++jitters[6]*np.random.rand()))
    p_w_a_intercept = conf[7]+jitters[7]*np.random.rand()
    p_w_a_slope = -conf[8]+jitters[8]*np.random.rand()
    p_a_bound = conf[9]+jitters[9]*np.random.rand()
    p_1st_readout = conf[10]+jitters[10]*np.random.rand()
    p_2nd_readout = conf[11]+jitters[11]*np.random.rand()
    p_leak = conf[12]+jitters[12]*np.random.rand()
    p_mt_noise = conf[13]+jitters[13]*np.random.rand()
    p_MT_intercept = conf[14]+jitters[14]*np.random.rand()
    p_MT_slope = conf[15]+jitters[15]*np.random.rand()
    stim = edd2.data_augmentation(stim=stim.reshape(20, num_tr),
                                  daf=data_augment_factor)
    stim_res = 50/data_augment_factor
    compute_trajectories = True
    all_trajs = True

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
                                 p_MT_slope=p_MT_slope,
                                 p_MT_intercept=p_MT_intercept,
                                 p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                 p_e_bound=p_e_bound, p_com_bound=p_com_bound,
                                 p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                                 num_tr=num_tr, p_w_a_intercept=p_w_a_intercept,
                                 p_w_a_slope=p_w_a_slope,
                                 p_a_bound=p_a_bound,
                                 p_1st_readout=p_1st_readout,
                                 p_2nd_readout=p_2nd_readout, p_leak=p_leak,
                                 p_mt_noise=p_mt_noise,
                                 compute_trajectories=compute_trajectories,
                                 stim_res=stim_res, all_trajs=all_trajs,
                                 human=human)
    hit_model = resp_fin == gt
    reaction_time = (first_ind[tr_index]-int(300/stim_res) + p_t_eff)*stim_res
    detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt


def run_simulation_different_subjs(stim, zt, coh, gt, trial_index, subject_list,
                                   subjid, human=False, num_tr=None, load_params=True,
                                   simulate=True):
    hit_model = np.empty((0))
    reaction_time = np.empty((0))
    detected_com = np.empty((0))
    resp_fin = np.empty((0))
    com_model = np.empty((0))
    pro_vs_re = np.empty((0))
    total_traj = []
    x_val_at_updt = np.empty((0))
    for subject in subject_list:
        if subject_list[0] is not None:
            index = subjid == subject
        else:
            index = range(num_tr)
        if not human:
            sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_simulation.pkl'
        if human:
            sim_data = DATA_FOLDER + '/Human/' + str(subject) + '/sim_data/' + str(subject) + '_simulation.pkl'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(sim_data), exist_ok=True)
        if os.path.exists(sim_data) and not simulate:
            data_simulation = np.load(sim_data, allow_pickle=True)
            hit_model_tmp = data_simulation['hit_model_tmp']
            reaction_time_tmp = data_simulation['reaction_time_tmp']
            detected_com_tmp = data_simulation['detected_com_tmp']
            resp_fin_tmp = data_simulation['resp_fin_tmp']
            com_model_tmp = data_simulation['com_model_tmp']
            pro_vs_re_tmp = data_simulation['pro_vs_re_tmp']
            total_traj_tmp = data_simulation['total_traj_tmp']
            x_val_at_updt_tmp = data_simulation['x_val_at_updt_tmp']
        else:
            hit_model_tmp, reaction_time_tmp, detected_com_tmp, resp_fin_tmp,\
                com_model_tmp, pro_vs_re_tmp, total_traj_tmp, x_val_at_updt_tmp =\
                run_model(stim=stim[:, index], zt=zt[index], coh=coh[index],
                          gt=gt[index], trial_index=trial_index[index],
                          subject=subject, load_params=load_params, human=human)
            data_simulation = {'hit_model_tmp': hit_model_tmp, 'reaction_time_tmp': reaction_time_tmp,
                               'detected_com_tmp': detected_com_tmp, 'resp_fin_tmp': resp_fin_tmp,
                               'com_model_tmp': com_model_tmp, 'pro_vs_re_tmp': pro_vs_re_tmp,
                               'total_traj_tmp': total_traj_tmp, 'x_val_at_updt_tmp': x_val_at_updt_tmp}
            pd.to_pickle(data_simulation, sim_data)
        hit_model = np.concatenate((hit_model, hit_model_tmp))
        reaction_time = np.concatenate((reaction_time, reaction_time_tmp))
        detected_com = np.concatenate((detected_com, detected_com_tmp))
        resp_fin = np.concatenate((resp_fin, resp_fin_tmp))
        com_model = np.concatenate((com_model, com_model_tmp))
        pro_vs_re = np.concatenate((pro_vs_re, pro_vs_re_tmp))
        total_traj = total_traj + total_traj_tmp
        x_val_at_updt = np.concatenate((x_val_at_updt, x_val_at_updt_tmp))
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt


def fig_7(df, df_sim):
    zt = df.allpriors.values
    coh = df.coh2.values
    com = df.CoM_sugg.values
    com_model = df_sim['com_detected'].values
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




def plot_mt_matrix_different_rtbins(df, small_rt=40, big_rt=120):
    fig, ax = plt.subplots(ncols=2)
    fig_1.mt_matrix_ev_vs_zt(df, ax, rt_bin=big_rt, silent_comparison=True)
    fig, ax = plt.subplots(ncols=2)
    fig_1.mt_matrix_ev_vs_zt(df, ax, rt_bin=small_rt, silent_comparison=True)


def binning_mt(df):
    bins_zt = [-1.01]
    # TODO: fix issue with equipopulated bins
    for i_p, perc in enumerate([0.75, 0.5, 0.25, 0.25, 0.5, 0.75]):
        if i_p > 2:
            bins_zt.append(df.norm_allpriors.abs().quantile(perc))
        else:
            bins_zt.append(-df.norm_allpriors.abs().quantile(perc))
    bins_zt.append(1.01)
    # matrix with rows for subjects and columns for bins
    mat_mt = np.empty((len(df.subjid.unique()), len(bins_zt)-1))
    for i_s, subject in enumerate(df.subjid.unique()):
        df_sub = df.loc[df.subjid == subject]
        for i_zt, bin_zt in enumerate(bins_zt[:-1]):
            mt_sub = df_sub.loc[(df_sub.norm_allpriors >= bin_zt) &
                                (df_sub.norm_allpriors < bins_zt[i_zt+1]), 'resp_len']
            mat_mt[i_s, i_zt] = np.nanmean(mt_sub)
    return mat_mt  # if you want mean across subjects, np.nanmean(mat_mt, axis=0)

def norm_allpriors_per_subj(df):
    norm_allpriors = np.empty((0,))
    for subj in df.subjid.unique():
        df_1 = df.loc[df.subjid == subj]
        zt_tmp = df_1.allpriors.values
        norm_allpriors = np.concatenate((norm_allpriors,
                                         zt_tmp/np.nanmax(abs(zt_tmp))))
    return norm_allpriors


def different_com_thresholds(traj_y, time_trajs, decision, sound_len,
                             coh, zt, com_th_list=np.linspace(0.5, 10, 20)):
    fig, ax = plt.subplots(1)
    rm_top_right_lines(ax=ax)
    colormap = pl.cm.Reds(np.linspace(0.2, 1, len(com_th_list)))
    com_d = {}
    fig1_th, ax1 = plt.subplots(4, 2)
    ax1 = ax1.flatten()
    cont = 0
    for i_th, com_th in enumerate(com_th_list):
        print('Com threshold = ' + str(com_th))
        _, _, _, com = edd2.com_detection(trajectories=traj_y, decision=decision,
                                          time_trajs=time_trajs,
                                          com_threshold=com_th)
        df_plot = pd.DataFrame({'sound_len': sound_len, 'com': com})
        binned_curve(df_plot, 'com', 'sound_len', bins=BINS_RT, xpos=xpos_RT,
                     errorbar_kw={'color': colormap[i_th], 'label': str(com_th)},
                     ax=ax)
        com = np.array(com)
        com_d['com_'+str(com_th)] = com
        if com_th == 1 or com_th == 2.5 or com_th == 5 or com_th == 8:
            i_ax = cont
            df_1 = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                                 'norm_allpriors': zt/max(abs(zt)),
                                 'R_response': (decision+1)/2})
            nbins = 7
            matrix_side_0 = fig_3.com_heatmap_marginal_pcom_side_mat(df=df_1, side=0)
            matrix_side_1 = fig_3.com_heatmap_marginal_pcom_side_mat(df=df_1, side=1)
            # L-> R
            vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
            pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
            im = ax1[i_ax*2].imshow(matrix_side_1, vmin=0, vmax=vmax, cmap='magma')
            plt.sca(ax1[i_ax*2])
            plt.colorbar(im, fraction=0.04)
            # R -> L
            pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
            im = ax1[i_ax*2+1].imshow(matrix_side_0, vmin=0, vmax=vmax, cmap='magma')
            ax1[i_ax*2+1].yaxis.set_ticks_position('none')
            plt.sca(ax1[i_ax*2+1])
            plt.colorbar(im, fraction=0.04)
            if i_ax == 0:
                ax1[i_ax].set_title(pcomlabel_1)
                ax1[i_ax+1].set_title(pcomlabel_0)
            for ax_i in [ax1[i_ax*2], ax1[i_ax*2+1]]:
                ax_i.set_yticklabels(['']*nbins)
                ax_i.set_xticklabels(['']*nbins)
            ax1[i_ax*2].set_ylabel('stim, th = {} px'.format(com_th))
            if i_ax == len(ax1) - 1:
                ax1[i_ax*2].set_xlabel('Prior evidence')
                ax1[i_ax*2+1].set_xlabel('Prior evidence')
            cont += 1
    ax.set_xlabel('RT(ms)')
    ax.set_ylabel('P(CoM)')
    com_dframe = pd.DataFrame(com_d)
    com_dframe.to_csv(SV_FOLDER + 'com_diff_thresholds.csv')


def mt_vs_stim_cong(df, rtbins=np.linspace(0, 80, 9), matrix=False, vigor=True,
                    title=None):
    ev_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    nsubs = len(df.subjid.unique())
    nsubs_sil = len(df.loc[df.special_trial == 2].subjid.unique())
    colormap = pl.cm.coolwarm(np.linspace(0, 1, len(ev_vals)))
    mat_mt_rt = np.empty((len(rtbins)-1, len(ev_vals)+1))
    err_mt_rt = np.empty((len(rtbins)-1, len(ev_vals)+1))
    mat_mt_rt[:] = np.nan
    err_mt_rt[:] = np.nan
    for irt, rtbin in enumerate(rtbins[:-1]):
        mt_mat = np.empty((nsubs, len(ev_vals)))
        mt_sil = []
        for i_sub, subject in enumerate(df.subjid.unique()):
            df_sub = df.loc[(df.subjid == subject) * (df.soundrfail == 0) *
                            (df.sound_len >= rtbin)*(df.sound_len < rtbins[irt+1])]
            # * (df.norm_allpriors.abs() < 0.1)]
            coh_cong = (df_sub.R_response.values*2-1)*(df_sub.coh2.values) *\
                (df_sub.special_trial == 0)
            for i_ev, ev in enumerate(ev_vals):
                index = coh_cong == ev
                mt_mat[i_sub, i_ev] = np.nanmean(df_sub.resp_len.values[index])
            if sum(df_sub.special_trial == 2) > 0:
                mt_sil.append(np.nanmean(
                    df_sub.resp_len.values[df_sub.special_trial == 2]))
        mt_mat *= 1e3
        mt_sil = np.array(mt_sil) * 1e3
        mat_mt_rt[irt, -1] = np.nanmean(1/mt_sil)
        err_mt_rt[irt, -1] = np.nanstd(1/mt_sil)/np.sqrt(nsubs_sil)
        resp_len_mean = np.nanmean(1/mt_mat, axis=0)
        resp_len_err = np.nanstd(1/mt_mat, axis=0)
        mat_mt_rt[irt, :-1] = resp_len_mean  # -np.nanmean(mt_sil)
        err_mt_rt[irt, :-1] = resp_len_err/np.sqrt(nsubs)
    ev_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1, 'silent']
    if not matrix and vigor:
        fig, ax = plt.subplots(1)
        rm_top_right_lines(ax)
        for ev in reversed(range(mat_mt_rt.shape[1])):
            if ev == len(ev_vals)-1:
                color = 'k'
            else:
                color = colormap[ev]
            x = rtbins[:-1]+(rtbins[1]-rtbins[0])/2
            y = mat_mt_rt[:, ev]
            err = err_mt_rt[:, ev]
            ax.errorbar(x, y, err, color=color, marker='o',
                        label=ev_vals[ev])
        ax.legend(title='Stim. evidence')
        ax.set_ylabel('vigor ~ 1/MT (ms^-1)')
        ax.set_xlabel('RT (ms)')
        ax.set_ylim(0.0028, 0.0043)
    if matrix and vigor:
        fig, ax = plt.subplots(1)
        im = ax.imshow(mat_mt_rt, cmap='coolwarm')
        ax.set_xticks(np.arange(8), labels=ev_vals)
        ax.set_yticks(np.arange(8), labels=rtbins[:-1]+(rtbins[1]-rtbins[0])/2)
        ax.set_xlabel('Stim evidence')
        ax.set_ylabel('RT (ms)')
        plt.colorbar(im, label='vigor')
    if matrix and not vigor:
        fig, ax = plt.subplots(1)
        mat_silent = np.zeros((mat_mt_rt[:, :-1].shape))
        for j in range(len(ev_vals)-1):
            mat_silent[:, j] += 1/mat_mt_rt[:, -1]
        im = ax.imshow(1/mat_mt_rt[:, :-1] - mat_silent, cmap='coolwarm',
                       vmin=-50, vmax=75)
        ax.set_xticks(np.arange(len(ev_vals)-1), labels=ev_vals[:-1])
        ax.set_yticks(np.arange(len(rtbins)-1),
                      labels=rtbins[:-1]+(rtbins[1]-rtbins[0])/2)
        ax.set_xlabel('Stim evidence')
        ax.set_ylabel('RT (ms)')
        plt.colorbar(im, label='MT (ms) - MT silent (ms)')
    ax.set_title(title)




def supp_com_threshold_matrices(df):
    dfth = pd.read_csv(SV_FOLDER + 'com_diff_thresholds.csv')
    fig, ax = plt.subplots(nrows=3, ncols=10, figsize=(15, 6))
    ax = ax.flatten()
    thlist = np.linspace(1, 10, 10)
    zt = df.allpriors.values
    coh = df.coh2.values
    decision = df.R_response.values*2 - 1
    nbins = 7
    for i_th, threshold in enumerate(thlist):
        com = dfth['com_'+str(threshold)]
        df_1 = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                             'norm_allpriors': zt/max(abs(zt)),
                             'R_response': (decision+1)/2})
        matrix_side_0 = fig_3.com_heatmap_marginal_pcom_side_mat(df=df_1, side=0)
        matrix_side_1 = fig_3.com_heatmap_marginal_pcom_side_mat(df=df_1, side=1)
        # L-> R
        vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
        pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
        im = ax[i_th].imshow(matrix_side_1, vmin=0, vmax=vmax)
        plt.sca(ax[i_th])
        plt.colorbar(im, fraction=0.04)
        # R -> L
        pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
        im = ax[i_th+len(thlist)].imshow(matrix_side_0, vmin=0, vmax=vmax, cmap='magma')
        ax[i_th+len(thlist)].yaxis.set_ticks_position('none')
        plt.sca(ax[i_th+len(thlist)])
        plt.colorbar(im, fraction=0.04)
        ax[i_th].set_title('stim, th = {} px'.format(threshold))
        ax[i_th+len(thlist)].set_xlabel('Prior evidence')
        if i_th == 0:
            ax[i_th].set_ylabel(pcomlabel_1 + ', avg. stim.')
            ax[i_th+len(thlist)].set_ylabel(pcomlabel_0 + ', avg. stim.')
            ax[i_th + 2*len(thlist)].set_ylabel('Position (px)')
        for ax_i in [ax[i_th], ax[i_th+len(thlist)]]:
            ax_i.set_yticklabels(['']*nbins)
            ax_i.set_xticklabels(['']*nbins)
        cont = 1
        j = 1000
        while cont <= 10:
            if threshold < 10:
                if com[j] and df.trajectory_y.values[j][-1] > 1 and\
                  df.R_response.values[j] == 1 and\
                  not dfth['com_'+str(threshold+0.5)][j] and\
                  df.trajectory_y.values[j][-0] >= -2 and\
                  df.trajectory_y.values[j][-0] <= 10:
                    traj = df.trajectory_y.values[j]
                    time_trajs = df.time_trajs.values[j]
                    traj -= np.nanmean(traj[
                        (time_trajs >= -100)*(time_trajs <= 0)])
                    ax[i_th + 2*len(thlist)].plot(time_trajs,
                                                  traj,
                                                  color='k', alpha=0.7)
                    cont += 1
            if threshold == 10:
                if com[j] and df.trajectory_y.values[j][-1] > 1 and\
                  df.R_response.values[j] == 1 and\
                  df.trajectory_y.values[j][-0] >= -2 and\
                  df.trajectory_y.values[j][-0] <= 10:
                    traj = df.trajectory_y.values[j]
                    time_trajs = df.time_trajs.values[j]
                    traj -= np.nanmean(traj[
                        (time_trajs >= -100)*(time_trajs <= 0)])
                    ax[i_th + 2*len(thlist)].plot(time_trajs,
                                                  traj,
                                                  color='k', alpha=0.7)
                    cont += 1
            j += 1
        ax[i_th + 2*len(thlist)].set_xlabel('Time')
        ax[i_th + 2*len(thlist)].set_ylim(-25, 25)
        ax[i_th + 2*len(thlist)].set_xlim(-100, 500)
        ax[i_th + 2*len(thlist)].axhline(-threshold, color='r', linestyle='--',
                                         alpha=0.5)
        ax[i_th + 2*len(thlist)].axvline(0, color='r', linestyle='--',
                                         alpha=0.5)
    thlist = np.linspace(0.5, 10, 20)
    mean_com = []
    fig2, ax2 = plt.subplots(1)
    for i_th, threshold in enumerate(thlist):
        com = dfth['com_'+str(threshold)]
        mean_com.append(np.nanmean(com))
    ax2.plot(thlist, mean_com, color='k', marker='o')
    ax2.set_xlabel('Threshold (pixels)')
    ax2.set_ylabel('P(CoM)')


def model_vs_data_traj(trajs_model, df_data):
    """
    Plots trajectories of data vs model for close MTs.

    Parameters
    ----------
    trajs_model : array
        Array with the trajectories of the model sorted by trial index.
    df_data : TYPE
        Array with the trajectories of the data sorted by trial index.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(5, 5)
    ax = ax.flatten()
    j = 0
    for i, ax_i in enumerate(ax):
        if i % 5 == 0:
            ax_i.set_ylabel('y-coord (px)')
        ax_i.set_xlabel('Time (ms)')
    for t in range(1000):
        ind = np.random.randint(0, len(trajs_model)-1)
        time_traj = df_data.time_trajs.values[ind]
        traj_data = df_data.trajectory_y.values[ind]
        if abs(len(trajs_model[ind]) - time_traj[-1]) < 15 and\
                np.sign(traj_data[-1]) == np.sign(trajs_model[ind][-1]):
            ax[j].plot(np.arange(len(trajs_model[ind])), trajs_model[ind],
                       color='r')
            ax[j].plot(time_traj, traj_data, color='k')
            ax[j].set_xlim(-100, max(time_traj[-1], len(trajs_model[ind])))
            j += 1
        if j == len(ax):
            break


def fig_trajs_model_4(trajs_model, df, reaction_time):
    fig, ax = plt.subplots(4, 4)
    ax = ax.flatten()
    ev_vals = [0, 0.25, 0.5, 1]
    norm_zt_vals = [0, 0.1, 0.4, 0.7, 1]
    j = 0
    # trajs_model = np.array(trajs_model)
    for i_ev, ev in enumerate(ev_vals):
        for izt, ztbin in enumerate(norm_zt_vals):
            if ztbin == 1:
                break
            indx = (df.coh2.values == ev) &\
                (df.norm_allpriors.values > ztbin)\
                & (df.norm_allpriors.values < norm_zt_vals[izt+1])
            pl = True
            while pl:
                ind = np.random.randint(0, sum(indx)-1)
                time_traj = df.time_trajs.values[indx][ind]
                traj_data = df.trajectory_y.values[indx][ind]
                rt_rat = df.sound_len.values[indx][ind]
                if abs(rt_rat - reaction_time[indx][ind]) < 30:
                    ax[j].plot(np.arange(len(trajs_model[indx][ind])),
                               trajs_model[indx][ind], color='r')
                    ax[j].plot(time_traj, traj_data, color='k')
                    ax[j].set_title('ev: {}, {} < zt < {} '
                                    .format(ev, ztbin, norm_zt_vals[izt+1]))
                    ax[j].set_xlim(-10, max(len(trajs_model[indx][ind]),
                                            time_traj[-1]))
                    j += 1
                    pl = False


def plot_conf_bias_distro(subjects, sv_folder=SV_FOLDER):
    fig, ax = plt.subplots(1)
    conf_mat = np.empty((3, len(subjects)))
    for i_s, subject in enumerate(subjects):
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        conf_mat[:, i_s] = np.array((conf[1], conf[2], conf[3]))
    mu = 5 * conf_mat[1, :]*conf_mat[2, :] / conf_mat[0, :]
    sns.kdeplot(mu, ax=ax)
    ax.set_xlabel('Time to reach CoM bound (ms)')


def plot_params_all_subs(subjects, sv_folder=SV_FOLDER, diff_col=True):
    fig, ax = plt.subplots(4, 4)
    if diff_col:
        colors = pl.cm.jet(np.linspace(0., 1, len(subjects)))
    else:
        colors = ['k' for _ in range(len(subjects))]
    ax = ax.flatten()
    labels = ['prior weight', 'stim weight', 'EA bound', 'CoM bound',
              't aff', 't eff', 'tAction', 'intercept AI',
              'slope AI', 'AI bound', 'DV weight 1st readout',
              'DV weight 2nd readout', 'leak', 'MT noise std',
              'MT offset', 'MT slope T.I.']
    conf_mat = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        conf_mat[:, i_s] = conf
    for i in range(len(labels)):
        if i == 4 or i == 5 or i == 6:
            sns.violinplot(conf_mat[i, :]*5, ax=ax[i], orient='h')
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s]*5,
                           0.05*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=1.2)
            ax[i].set_xlabel(labels[i] + str(' (ms)'))
        else:
            sns.violinplot(conf_mat[i, :], ax=ax[i], orient='h')
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s],
                           0.1*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=1.2)
            ax[i].set_xlabel(labels[i] + str(' ms'))
            ax[i].set_xlabel(labels[i])


def mt_vs_ti_data_comparison(df, df_sim):
    df = df[:len(df_sim)]
    coh = np.array(df.coh2)[:len(df_sim)]
    mt_data = df.resp_len.values[:len(df_sim)]
    t_i_data = df.origidx[:len(df_sim)]
    mt_model = df_sim.resp_len
    t_i_model = df_sim.origidx
    plt.figure()
    plt.plot(t_i_data, mt_data*1e3, 'o', markersize=0.6, color='orange',
             label='Data')
    plt.plot(t_i_model, mt_model*1e3, 'o', markersize=0.8, color='blue',
             label='Model')
    plt.ylabel('MT (ms)')
    plt.xlabel('Trial index')
    plt.legend()
    mt_model_signed = np.copy(mt_model)*1e3
    mt_model_signed[df_sim.R_response.values == 0] *= -1
    mt_sign = np.copy(mt_data*1e3)
    mt_sign[df.R_response.values == 0] *= -1
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    ax = ax.flatten()
    sns.histplot(mt_model_signed, ax=ax[0])
    sns.histplot(mt_model*1e3, ax=ax[1])
    sns.histplot(mt_data*1e3, color='orange', ax=ax[2])
    sns.histplot(mt_sign, color='orange', ax=ax[3])
    sns.kdeplot(mt_data*1e3, color='orange', ax=ax[4], label='Data')
    sns.kdeplot(mt_model*1e3, color='blue', ax=ax[4], label='Model')
    ax[4].legend()
    ax[1].set_title('Model')
    ax[0].set_title('Model')
    ax[2].set_title('Data')
    ax[3].set_xlabel('MT (ms)')
    ax[4].set_xlabel('MT (ms)')
    for i in range(2):
        ax[i+1].set_xlim(0, 600)
    ax[3].set_xlim(-600, 600)
    ax[0].set_xlim(-600, 600)
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    for iev, ev in enumerate([0, 0.25, 0.5, 1]):
        index = np.abs(coh) == ev
        sns.kdeplot(mt_data[index]*1e3, color=colormap[iev], ax=ax[5])
        index = np.abs(df_sim.coh2) == ev
        sns.kdeplot(mt_model[index]*1e3, color=colormap[iev], ax=ax[5],
                    linestyle='--')
    ax[5].set_xlim(0, 600)
    # plt.show()
    fig, ax = plt.subplots(ncols=3)
    sns.kdeplot(df_sim.sound_len.values, color='blue', label='Model', ax=ax[0])
    sns.kdeplot(df.sound_len.values, color='orange', label='Data', ax=ax[0])
    ax[0].legend()
    ax[0].set_xlabel('RT (ms)')
    sns.histplot(df_sim.sound_len.values, color='blue', label='Model', ax=ax[1])
    sns.histplot(df.sound_len.values, color='orange', label='Data', ax=ax[2])
    ax[1].set_xlabel('RT (ms)')
    ax[1].set_title('Model')
    ax[2].set_xlabel('Data')


def plot_mt_vs_rt_model_comparison(df, df_sim, bins_rt=np.linspace(0, 300, 31)):
    fig, ax = plt.subplots(ncols=2)
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    ax1, ax2 = ax
    for iev, ev in enumerate([0, 0.25, 0.5, 1]):
        binned_curve(df.loc[df.coh2.abs() == ev], 'resp_len', 'sound_len',
                     bins=bins_rt, xpos=np.diff(bins_rt)[0], ax=ax1,
                     errorbar_kw={'label': 'ev: ' + str(ev),
                                  'color': colormap[iev]})
    ax1.set_xlabel('RT (ms)')
    ax1.set_ylabel('MT (s)')
    ax1.set_title('Data')
    ax1.set_ylim(0.23, 0.42)
    colormap = pl.cm.gist_gray_r(np.linspace(0.3, 1, 4))
    for iev, ev in enumerate([0, 0.25, 0.5, 1]):
        binned_curve(df_sim.loc[df_sim.coh2.abs() == ev], 'resp_len', 'sound_len',
                     bins=bins_rt, xpos=np.diff(bins_rt)[0], ax=ax2,
                     errorbar_kw={'label': 'ev: ' + str(ev),
                                  'color': colormap[iev]})
    ax2.set_xlabel('RT (ms)')
    ax2.set_ylabel('MT (s)')
    ax2.set_title('Model')
    ax2.set_ylim(0.23, 0.42)


def plot_rt_sim(df_sim):
    fig, ax = plt.subplots(nrows=5, ncols=3)
    ax = ax.flatten()
    colormap = pl.cm.gist_gray_r(np.linspace(0.2, 1, 4))
    for isub, subj in enumerate(df_sim.subjid.unique()):
        rm_top_right_lines(ax[isub])
        ax[isub].set_title(subj)
        for iev, ev in enumerate([0, 0.25, 0.5, 1]):
            rts = df_sim.loc[(df_sim.coh2.abs() == ev) &
                             (df_sim.subjid == subj), 'sound_len']
            sns.kdeplot(rts,
                        color=colormap[iev], ax=ax[isub])
            ax[isub].set_xlabel('RT (ms)')
            # ax[isub].set_title(subj + ' ' + str(np.round(np.mean(rts < 0), 4)))
            ax[isub].set_title(subj)


def plot_fb_per_subj_from_df(df):
    # plots the RT distros conditioning on coh
    fig, ax = plt.subplots(5, 3)
    ax = ax.flatten()
    colormap = pl.cm.gist_gray_r(np.linspace(0.2, 1, 4))
    subjects = df.subjid.unique()
    for i_s, subj in enumerate(subjects):
        rm_top_right_lines(ax[i_s])
        df_1 = df[df.subjid == subj]
        coh_vec = df_1.coh2.values
        for ifb, fb in enumerate(df_1.fb):
            for j in range(len(fb)):
                coh_vec = np.append(coh_vec, [df_1.coh2.values[ifb]])
        fix_breaks =\
            np.vstack(np.concatenate([df_1.sound_len/1000,
                                      np.concatenate(df_1.fb.values)-0.3]))
        for iev, ev in enumerate([0, 0.25, 0.5, 1]):
            index = np.abs(coh_vec) == ev
            fix_breaks_2 = fix_breaks[index]*1e3
            sns.kdeplot(fix_breaks_2.reshape(-1),
                        color=colormap[iev], ax=ax[i_s])
        # ax[i_s].set_title(subj + str(sum(fix_breaks < 0)/len(fix_breaks)))
        ax[i_s].set_title(subj)
        ax[i_s].set_xlabel('RT (ms)')


def sess_t_index_stats(df, subjects):
    subs_spec_trial = df.loc[df.special_trial == 2, 'subjid'].unique()
    subs_no_silent = list(set(subjects)-set(subs_spec_trial))
    ses_st = []
    ntr_sil = []
    ntr_only_sil = []
    for sub_sil in subs_spec_trial:
        ses_st.append(len(df.loc[df.subjid == sub_sil, 'sessid'].unique()))
        ntr_sil.append(len(df.loc[df.subjid == sub_sil]))
        ntr_only_sil.append(len(
            df.loc[(df.special_trial == 2) & (df.subjid == sub_sil)]))
    ses_n_st = []
    ntr_n_sil = []
    for sub_n_sil in subs_no_silent:
        ses_n_st.append(len(df.loc[df.subjid == sub_n_sil, 'sessid'].unique()))
        ntr_n_sil.append(len(df.loc[df.subjid == sub_n_sil]))
    print(', '.join(list(subs_spec_trial)))
    print(', '.join(list(subs_no_silent)))


def mean_traj_per_deflection_time(df, time_com, ax,
                                  bins=np.arange(100, 401, 100)):
    df_1 = df.loc[df.CoM_sugg]
    all_trajs = np.empty((len(df.subjid.unique()), 1700))
    all_trajs[:] = np.nan
    for ind_b, min_b in enumerate(bins[:-1]):
        index = (time_com >= min_b) * (time_com < bins[ind_b+1])
        df_2 = df_1.loc[index]


def mt_diff_rev_nonrev(df):
    mt_x_sub_rev = []
    for subj in df.subjid.unique():
        mt_x_sub_rev.append(
            np.nanmean(df.loc[(df.subjid == subj) & (df.CoM_sugg), 'resp_len']) -
            np.nanmean(df.loc[(~df.CoM_sugg) & (df.subjid == subj), 'resp_len']))
    print('Mean +- SEM of difference in MT rev vs non_rev')
    print(np.nanmean(mt_x_sub_rev)*1e3)
    print('+-')
    print(np.nanstd(mt_x_sub_rev)*1e3/np.sqrt(15))


def supp_mt_per_rat(df, title='Data'):
    fig, ax = plt.subplots(5, 3)
    ax = ax.flatten()
    for a in ax:
        rm_top_right_lines(a)
    subjects = df.subjid.unique()
    for i_s, subj in enumerate(subjects):
        df_1 = df[df.subjid == subj]
        if title == 'Data':
            mt_nocom = df_1.loc[~df_1['CoM_sugg'], 'resp_len']*1e3
            mt_com = df_1.loc[df_1['CoM_sugg'], 'resp_len']*1e3
        else:
            mt_nocom = df_1.loc[~df_1['com_detected'], 'resp_len']*1e3
            mt_com = df_1.loc[df_1['com_detected'], 'resp_len']*1e3
        sns.kdeplot(mt_com, color=COLOR_COM, ax=ax[i_s],
                    label='Rev.')
        sns.kdeplot(mt_nocom, color=COLOR_NO_COM, ax=ax[i_s],
                    label='No-Rev.')
        ax[i_s].set_xlabel('MT (ms)')
        ax[i_s].set_title(subj)
    ax[0].legend()
    fig.suptitle(title)


def plot_model_density(df_sim, df=None, offset=0, plot_data_trajs=False, n_trajs_plot=50,
                       pixel_precision=5, cmap='pink'):
    """
    Plots density of the position of the model, it can plot rat trajectories on top.

    Parameters
    ----------
    df_sim : data frame
        Data frame with simulations.
    df : data frame, optional
        Data frame with rat data. The default is None.
    offset : int, optional
        Padding. The default is 0.
    plot_data_trajs : bool, optional
        Whereas to plot rat trajectories on top or not. The default is False.
    n_trajs_plot : int, optional
        In case of plotting the trajectories, how many. The default is 50.
    pixel_precision : float, optional
        Pixel precision for the density (the smaller the cleaner the plot).
        The default is 5.
    cmap : str, optional
        Colormap. The default is 'pink'.

    Returns
    -------
    None.

    """
    fig2, ax2 = plt.subplots(nrows=5, ncols=5)
    np.random.seed(seed=5)  # set seed
    # fig2.tight_layout()
    ax2 = ax2.flatten()
    coh = df_sim.coh2.values
    zt = np.round(df_sim.normallpriors.values, 1)
    coh_vals = [-1, -0.25, 0, 0.25, 1]
    zt_vals = [-np.max(np.abs(zt)), -np.median(np.abs(zt)),
               -0.1, 0.1, np.median(np.abs(zt)), np.max(np.abs(zt))]
    i = 0
    ztlabs = [-1, -0.2, 0, 0.2, 1]
    gkde = scipy.stats.gaussian_kde  # we define gkde that will generate the kde
    if plot_data_trajs:
        bins = np.array([-1.1, 1.1])  # for data plotting
        bintype = 'edges'
        trajectory = 'trajectory_y'
        df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    for ie, ev in enumerate(coh_vals):
        for ip, pr in enumerate(zt_vals):
            if ip == 5:
                break
            index = (zt >= pr) & (zt < zt_vals[ip+1]) & (coh == ev)  # index of filtered
            max_len = max([len(t) for t in df_sim.traj[index].values])
            mat_fin = np.empty((sum(index), max_len+offset))
            mat_fin[:] = np.nan
            trajs = df_sim.traj[index].values
            for j in range(sum(index)):
                mat_fin[j, :len(trajs[j])] = trajs[j]  # mat_fin contains trajectories by rows
                mat_fin[j, len(trajs[j]):-1] = trajs[j][-1]  # set the last value (-75 or 75) until the end
            values = np.arange(-80, 81, pixel_precision)
            mat_final_density = np.empty((len(values), 50))  # matrix that will contain density by columns
            mat_final_density[:] = np.nan
            for j in range(2, 50):
                yvalues = np.nanmean(mat_fin[:, j*5:(j+1)*5], axis=1)  # we get the trajectory values
                kernel_1 = gkde(yvalues)  # we create the kernel using gkde
                vals_density = kernel_1(values)  # we evaluate the values defined before
                mat_final_density[:, j] = vals_density / np.nansum(vals_density)  # we normalize the density
            ax2[i].imshow(np.flipud(mat_final_density), cmap=cmap, aspect='auto',
                          vmin=0, vmax=0.4)  # plot the matrix
            ax2[i].set_xlim(0, 50)
            ax2[i].set_ylim(len(values), 0)
            ax2[i].set_xticks(np.arange(0, 50, 5), np.arange(0, 50, 5)*5)
            ax2[i].set_yticks(np.arange(0, len(values), int(20/pixel_precision)),
                              np.arange(80, -81, -20))
            ax2[i].set_title('coh = {}, zt = {}'.format(ev, ztlabs[ip]))
            if i == 0 or i % 5 == 0:
                ax2[i].set_ylabel('Position, pixels')
            if i >= 20:
                ax2[i].set_xlabel('Time (ms)')
            if plot_data_trajs:
                index = (zt >= pr) & (zt < zt_vals[ip+1]) & (coh == ev)  # & (t_index < np.median(t_index))
                # to extract interpolated trajs in mat --> these aren't signed
                _, _, _, mat, idx, _ =\
                trajectory_thr(df.loc[index], 'choice_x_prior', bins,
                               collapse_sides=True, thr=30, ax=None, ax_traj=None,
                               return_trash=True, error_kwargs=dict(marker='o'),
                               cmap=None, bintype=bintype,
                               trajectory=trajectory, plotmt=False, alpha_low=False)
                mat_0 = mat[0]
                # we multiply by response to have the sign
                mat_0 = mat_0*(df.loc[idx[0]].R_response.values*2-1).reshape(-1, 1)
                n_trajs = mat_0.shape[0]
                # we select the number of trajectories that we want
                index_trajs_plot = np.random.choice(np.arange(n_trajs), n_trajs_plot)
                for ind in index_trajs_plot:
                    traj = mat_0[ind, :]
                    # we do some filtering
                    if sum(np.abs(traj[700:950]) > 80) > 1:
                        continue
                    if np.abs(traj[700]) > 5:
                        continue
                    ax2[i].plot(np.arange(0, 50, 0.2), (-traj[700:950]+80)/160*len(values),
                                color='blue', linewidth=0.5)
            i += 1


def plot_data_trajs_density(df):
    fig, ax = plt.subplots(nrows=5, ncols=5)
    fig.tight_layout()
    # plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
    #                     hspace=0.2, wspace=0.1)
    ax = ax.flatten()
    coh = df.coh2.values
    zt = np.round(df.norm_allpriors.values, 1)
    coh_vals = [-1, -0.25, 0, 0.25, 1]
    zt_vals = [-np.max(np.abs(zt)), -np.median(np.abs(zt)),
               -0.1, 0.1, np.median(np.abs(zt)), np.max(np.abs(zt))]
    i = 0
    # t_index = df_sim.origidx.values
    bins = np.array([-1.1, 1.1])  # for data plotting
    bintype = 'edges'
    trajectory = 'trajectory_y'
    df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    gkde = scipy.stats.gaussian_kde
    ztlabs = [-1, -0.2, 0, 0.2, 1]
    for ie, ev in enumerate(coh_vals):
        for ip, pr in enumerate(zt_vals):
            if ip == 5:
                break
            index = (zt >= pr) & (zt < zt_vals[ip+1]) & (coh == ev)  # & (t_index < np.median(t_index))
            _, _, _, mat, idx, _ =\
            trajectory_thr(df.loc[index], 'choice_x_prior', bins,
                           collapse_sides=True, thr=30, ax=None, ax_traj=None,
                           return_trash=True, error_kwargs=dict(marker='o'),
                           cmap=None, bintype=bintype,
                           trajectory=trajectory, plotmt=False, alpha_low=False)
            mat_fin = mat[0]
            mat_fin = mat_fin*(df.loc[idx[0]].R_response.values*2-1).reshape(-1, 1)
            values = np.arange(-100, 101, 5)
            mat_final_density = np.empty((len(values), 50))
            mat_final_density[:] = np.nan
            for j in range(2, 50):
                yvalues = np.nanmean(mat_fin[:, 700+j*5:700+(j+1)*5], axis=1)
                kernel_1 = gkde(yvalues[~np.isnan(yvalues)])
                vals_density = kernel_1(values)
                mat_final_density[:, j] = vals_density / np.nansum(vals_density)
            ax[i].imshow(np.flipud(mat_final_density), cmap='hot', aspect='auto')
            ax[i].set_xticks(np.arange(0, 50, 5), np.arange(0, 50, 5)*5)
            ax[i].set_yticks(np.arange(0, len(values), 4), np.arange(100, -101, -20))
            ax[i].set_title('coh = {}, zt = {}'.format(ev, ztlabs[ip]))
            if i == 0 or i % 5 == 0:
                ax[i].set_ylabel('Position, pixels')
            if i >= 20:
                ax[i].set_xlabel('Time (ms)')
            i += 1


def plot_model_trajs(df_sim, df, model_alone=False, align_y_onset=False, offset=200):
    """
    Trajectories conditioned on median T.I. and changing coh/zt
    """
    fig, ax = plt.subplots(nrows=7, ncols=5)
    fig.tight_layout()
    # plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.98,
    #                     hspace=0.2, wspace=0.1)
    ax = ax.flatten()
    if model_alone:
        fig2, ax2 = plt.subplots(nrows=7, ncols=5)
        # fig2.tight_layout()
        ax2 = ax2.flatten()
    coh = df_sim.coh2.values
    zt = np.round(df_sim.normallpriors.values, 1)
    coh_vals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    zt_vals = [-np.max(np.abs(zt)), -np.median(np.abs(zt)),
               -0.1, 0.1, np.median(np.abs(zt)), np.max(np.abs(zt))]
    i = 0
    # t_index = df_sim.origidx.values
    ztlabs = [-1, -0.2, 0, 0.2, 1]
    err_mat = np.empty((len(coh_vals), len(zt_vals)-1))
    bins = np.array([-1.1, 1.1])  # for data plotting
    bintype = 'edges'
    trajectory = 'trajectory_y'
    df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    interpolatespace=np.linspace(-700000, 1000000, 1700)/1e3
    gkde = scipy.stats.gaussian_kde
    for ie, ev in enumerate(coh_vals):
        for ip, pr in enumerate(zt_vals):
            if ip == 5:
                break
            index = (zt >= pr) & (zt < zt_vals[ip+1]) & (coh == ev)  # & (t_index < np.median(t_index))
            max_len = max([len(t) for t in df_sim.traj[index].values])
            mat_fin = np.empty((sum(index), max_len+offset))
            time = np.arange(-offset, max_len)
            mat_fin[:] = np.nan
            trajs = df_sim.traj[index].values
            for j in range(sum(index)):
                if not align_y_onset:
                    mat_fin[j, :len(trajs[j])] = trajs[j]
                    mat_fin[j, len(trajs[j]):-1] = trajs[j][-1]
                else:
                    traj_model = trajs[j]
                    ind_model = np.where(np.abs(trajs[j]) >= 1)[0][0]
                    mat_fin[j, offset-ind_model:offset+len(trajs[j])-ind_model] = traj_model
            # traj = df_sim.traj[index].values[np.random.choice(np.arange(0, sum(index)))]
            # traj = df_sim.traj[index].values[1]
            # if traj[-1] < 0:
            #     traj = -traj
            if model_alone:
                values = np.arange(-80, 81, 5)
                mat_final_density = np.empty((len(values), 50))
                # y_trajs = {}
                # y_bins = []
                for j in range(1, 50):
                    yvalues = np.nanmean(mat_fin[:, j*5:(j+1)*5], axis=1)
                    kernel_1 = gkde(yvalues)
                    vals_density = kernel_1(values)
                    mat_final_density[:, j] = vals_density / np.sum(vals_density)
                    # y_trajs[str(j)] = yvalues
                    # y_bins.append((j*50 + (j+1)*50)/2)
                    # sns.kdeplot(yvalues, color=colormap[j], fill=True,
                    #             ax=ax2[i], label=str(j*10) + ' to ' + str((j+1)*10) + ' ms')
                # data = pd.DataFrame(y_trajs)
                # sns.kdeplot(data, palette='Blues', ax=ax2[i])
                # sns.violinplot(data=data, y='yvals', x='xvals',
                #                ax=ax[i], color='blue', alpha=0.3)  #, label=str(j*50) + ' to ' + str((j+1)*50) + ' ms')
                # norm_image = mat_final_density/np.max(mat_final_density)
                ax2[i].imshow(np.flipud(mat_final_density), cmap='hot', aspect='auto')
                ax2[i].set_xticks(np.arange(0, 50, 5), np.arange(0, 50, 5)*5)
                ax2[i].set_yticks(np.arange(0, len(values), 4), np.arange(80, -81, -20))
            if not model_alone:
                _, _, _, mat, idx, _ =\
                trajectory_thr(df.loc[index], 'choice_x_prior', bins,
                               collapse_sides=True, thr=30, ax=None, ax_traj=None,
                               return_trash=True, error_kwargs=dict(marker='o'),
                               cmap=None, bintype=bintype,
                               trajectory=trajectory, plotmt=False, alpha_low=False)
                mat_0 = mat[0]
                if align_y_onset:
                    for i_t, traj_d in enumerate(mat_0):
                        traj_d = traj_d - np.nanmean(
                            traj_d[500:700])
                        if np.nansum(traj_d) == 0:
                            continue
                        try:
                            ind_data = np.where(np.abs(traj_d) >= 1)[0][0]
                        except IndexError:
                            continue
                        mat_0[i_t, :] = np.roll(mat_0[i_t, :], -ind_data)
                # mat_0 = np.roll(mat[0], -30, axis=1)
                mat_0 = mat_0*(df.loc[idx[0]].R_response.values*2-1).reshape(-1, 1)
                traj_data = np.nanmedian(mat_0, axis=0)
                err_data = np.nanstd(mat_0, axis=0)
                ax[i].plot(interpolatespace, traj_data, color='k', label='Data')
                ax[i].fill_between(interpolatespace, traj_data-err_data,
                                   traj_data+err_data,
                                   alpha=0.2, color='k')
            traj = np.nanmedian(mat_fin, axis=0)
            err = np.nanstd(mat_fin, axis=0)  # / np.sqrt(sum(index))
            err_mat[ie, ip] = np.sum(err)
            ax[i].plot(time, traj, color='r', label='Model')
            ax[i].fill_between(time, traj-err, traj+err,
                               alpha=0.2, color='r')
            ax[i].set_xlim(-5, 255)
            ax[i].set_title('coh = {}, zt = {}'.format(ev, ztlabs[ip]))
            ax2[i].set_title('coh = {}, zt = {}'.format(ev, ztlabs[ip]))
            if i == 0 or i % 5 == 0:
                ax[i].set_ylabel('Position, pixels')
                ax2[i].set_ylabel('Position, pixels')
            if i >= 30:
                ax[i].set_xlabel('Time (ms)')
                ax2[i].set_xlabel('Time (ms)')
            rm_top_right_lines(ax[i])
            ax[i].set_ylim(-125, 125)
            ax[i].axhline(y=0, color='k', alpha=0.4, linestyle='--')
            i += 1
            ax[0].legend()
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(err_mat)
    ax1.set_yticks(np.arange(len(coh_vals)), coh_vals)
    ax1.set_xticks(np.arange(len(ztlabs)), ztlabs)
    ax1.set_xlabel('zt')
    ax1.set_ylabel('coh')
    plt.colorbar(im, label='SD')


def get_human_mt(df_data):
    motor_time = []
    times = df_data.times.values
    for tr in range(len(df_data)):
        ind_time = [True if t != '' else False for t in times[tr]]
        time_tr = np.array(times[tr])[np.array(ind_time)].astype(float)
        mt = time_tr[-1]
        if mt > 1:
            mt = 1
        motor_time.append(mt*1e3)
    return motor_time


def get_human_data(user_id, sv_folder=SV_FOLDER, nm='300'):
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
    # retrieve data
    df = ah.traj_analysis(data_folder=folder,
                          subjects=subj, steps=steps, name=nm,
                          sv_folder=sv_folder)
    return df


def simulate_model_humans(df_data):
    choice = df_data.R_response.values*2-1
    hit = df_data.hithistory.values*2-1
    subjects = df_data.subjid.unique()
    subjid = df_data.subjid.values
    gt = (choice*hit+1)/2
    coh = df_data.avtrapz.values*5
    stim = np.repeat(coh.reshape(-1, 1), 20, 1).T
    zt = df_data.norm_allpriors.values*3
    len_task = [len(df_data.loc[subjid == subject]) for subject in subjects]
    trial_index = np.empty((0))
    for j in range(len(len_task)):
        trial_index = np.concatenate((trial_index, np.arange(len_task[j])+1))
    num_tr = len(trial_index)
    hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt =\
        run_simulation_different_subjs(
            stim=stim, zt=zt, coh=coh, gt=gt,
            trial_index=trial_index, num_tr=num_tr, human=True,
            subject_list=subjects, subjid=subjid, simulate=True)
    
    return hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
        _, trajs, x_val_at_updt



def plot_params_all_subs_humans(subjects, sv_folder=SV_FOLDER, diff_col=True):
    fig, ax = plt.subplots(4, 4)
    if diff_col:
        colors = pl.cm.jet(np.linspace(0., 1, len(subjects)))
    else:
        colors = ['k' for _ in range(len(subjects))]
    ax = ax.flatten()
    labels = ['prior weight', 'stim weight', 'EA bound', 'CoM bound',
              't aff', 't eff', 'tAction', 'intercept AI',
              'slope AI', 'AI bound', 'DV weight 1st readout',
              'DV weight 2nd readout', 'leak', 'MT noise std',
              'MT offset', 'MT slope T.I.']
    conf_mat = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS_human_subj_' + str(subject) + '.npy')
        conf_mat[:, i_s] = conf
    for i in range(len(labels)):
        if i == 4 or i == 5 or i == 6:
            sns.violinplot(conf_mat[i, :]*5, ax=ax[i], orient='h')
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s]*5,
                           0.05*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=1.2)
            ax[i].set_xlabel(labels[i] + str(' (ms)'))
        else:
            sns.violinplot(conf_mat[i, :], ax=ax[i], orient='h')
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s],
                           0.1*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=1.2)
            ax[i].set_xlabel(labels[i] + str(' ms'))
            ax[i].set_xlabel(labels[i])
