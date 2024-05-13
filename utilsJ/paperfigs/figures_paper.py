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
from scipy.stats import ttest_rel, linregress
from matplotlib.lines import Line2D
from statsmodels.stats.proportion import proportion_confint
from matplotlib.colors import LogNorm
from skimage import exposure
from scipy.stats import pearsonr
import scipy
# from scipy import interpolate
# import shutil

# sys.path.append("/home/jordi/Repos/custom_utils/")  # alex idibaps
sys.path.append('C:/Users/alexg/Onedrive/Documentos/GitHub/custom_utils')  # Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append('C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/Documentos/GitHub/custom_utils')
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
# sys.path.append("/home/molano/custom_utils") # Cluster Manuel

# from utilsJ.Models import simul
from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.Models import different_models as model_variations
from utilsJ.Behavior.plotting import binned_curve, tachometric
from utilsJ.Behavior.plotting import trajectory_thr, interpolapply
from utilsJ.paperfigs import figure_1 as fig_1
from utilsJ.paperfigs import figure_2 as fig_2
from utilsJ.paperfigs import figure_3 as fig_3
# from utilsJ.paperfigs import figure_5 as fig_5
# from utilsJ.paperfigs import figure_6 as fig_6
from utilsJ.Models import analyses_humans as ah
import matplotlib
import matplotlib.pylab as pl



matplotlib.rcParams['font.size'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12
# matplotlib.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.markersize'] = 3

# ---GLOBAL VARIABLES
pc_name = 'alex'
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
elif pc_name == 'sara':
    SV_FOLDER = 'C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/' +\
        'Documentos/EBM/4t/IDIBAPS'
    DATA_FOLDER = 'C:/Users/Sara Fuentes/OneDrive - Universitat de Barcelona/'+\
        'Documentos/EBM/4t/IDIBAPS'

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
            mat_mt[i_s, bin] = np.nanmedian(mt_sub)
            if np.isnan(mat_mt[i_s, bin]):
                print(1)
    return mat_mt  # if you want mean across subjects, np.nanmean(mat_mt, axis=0)


def get_bin_info(df, condition, prior_limit=0.25, after_correct_only=True, rt_lim=50,
                 fpsmin=29, num_bins_prior=5, rtmin=0, silent=True):
    # after correct condition
    ac_cond = df.aftererror == False if after_correct_only else (df.aftererror*1) >= 0
    # common condition 
    # put together all common conditions: prior, reaction time and framerate
    common_cond = ac_cond & (df.norm_allpriors.abs() <= prior_limit) &\
        (df.sound_len < rt_lim) & (df.framerate >= fpsmin) & (df.sound_len >= rtmin)
    # define bins, bin type, trajectory index and colormap depending on condition
    if condition == 'choice_x_coh':
        bins = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
        bintype = 'categorical'
        indx_trajs = common_cond & (df.special_trial == 0) 
        n_iters = len(bins)
        colormap = pl.cm.coolwarm(np.linspace(0., 1, n_iters))
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
        colormap = colormap(np.linspace(0, 1, n_iters))
    elif condition == 'choice_x_prior':
        if silent:
            indx_trajs = common_cond & (df.special_trial == 2)
        if not silent:
            indx_trajs = common_cond & (df.special_trial == 0)
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



def tachometric_data(coh, hit, sound_len, subjid, ax, label='Data',
                     legend=True, rtbins=np.arange(0, 201, 3)):
    rm_top_right_lines(ax)
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len, 'subjid': subjid})
    tachometric(df_plot_data, ax=ax, fill_error=True, cmap='gist_yarg',
                rtbins=rtbins, evidence_bins=[0, 0.25, 0.5, 1])
    ax.axhline(y=0.5, linestyle='--', color='k', lw=0.5)
    ax.set_xlabel('Reaction time (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title(label)
    ax.set_ylim(0.24, 1.04)
    if legend:
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


def supp_pcom_teff_taff(stim, zt, coh, gt, trial_index, subjects,
                        subjid, sv_folder, idx=None):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    ax = ax.flatten()
    labs = ['a', 'b', 'c', 'd', '', '']
    for i_a, a in enumerate(ax):
        rm_top_right_lines(a)
        a.text(-0.1, 1.17, labs[i_a], transform=a.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    # ax[5].axis('off')
    # changing t_aff and t_eff
    params_to_explore_aff = [[4]] + [np.arange(7)]
    params_to_explore_eff = [[5]] + [np.arange(7)]
    params_to_explore = [params_to_explore_aff, params_to_explore_eff]
    plot_pcom_taff_teff(stim, zt, coh, gt, trial_index, subjects,
                        subjid, params_to_explore, ax=ax[0], com=True)
    plot_pcom_taff_teff(stim, zt, coh, gt, trial_index, subjects,
                        subjid, params_to_explore, ax=ax[1], com=False)
    fig.savefig(sv_folder+'/supp_model_b0b1.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_model_b0b1.png', dpi=400, bbox_inches='tight')



def plot_pcom_taff_teff(stim, zt, coh, gt, trial_index, subjects,
                         subjid, params_to_explore, ax, com=True):
    num_tr = int(len(coh))
    ind_stim = np.sum(stim, axis=0) != 0
    params_to_explore_aff, params_to_explore_eff = params_to_explore
    param_aff = params_to_explore_aff[0][0]
    param_eff = params_to_explore_eff[0][0]
    # colormap = pl.cm.BrBG(np.linspace(0.1, 1, len(params_to_explore_eff[1])))
    subject = str(np.unique(subjid))
    if com:
        sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_pcom_matrix_params_rt_under_20_v2.npy'
    else:
        sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_prev_matrix_params_rt_under_20_v2.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(sim_data), exist_ok=True)
    if os.path.exists(sim_data):
        mat_pcom = np.load(sim_data)
    else:
        mat_pcom = np.empty((7, 7))
        mat_pcom[:] = np.nan
        for ind_aff in range(len(params_to_explore_aff[1])):
            aff_value = params_to_explore_aff[1][ind_aff]
            for ind_eff in range(len(params_to_explore_eff[1])):
                eff_value = params_to_explore_aff[1][ind_eff]
                param_iter = str(param_aff)+'_'+str(aff_value)+'_'+str(param_eff)+'_'+str(eff_value)+'_res_1ms'
                hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
                    _, trajs, x_val_at_updt =\
                        run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                               trial_index=trial_index, num_tr=num_tr,
                                               subject_list=subjects, subjid=subjid,
                                               simulate=False,
                                               params_to_explore=[[param_aff, param_eff], [aff_value, eff_value]],
                                               change_param=True,
                                               param_iter=param_iter)
                MT = [len(t) for t in trajs]
                df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                                        'sound_len': reaction_time,
                                        'rewside': (gt + 1)/2,
                                        'R_response': (resp_fin+1)/2,
                                        'resp_len': np.array(MT)*1e-3,
                                        'subjid': subjid})
                df_sim = df_sim.loc[ind_stim]
                if com:
                    pcom_mean = np.nanmean(com_model[(reaction_time <= (20)) &
                                                     (reaction_time >= (0))])
                if not com:
                    pcom_mean = np.nanmean(com_model_detected[reaction_time <= (20) &
                                                              (reaction_time >= (0))])
                mat_pcom[ind_aff, ind_eff] = pcom_mean
        np.save(sim_data, mat_pcom)
    im = ax.imshow(np.flipud(mat_pcom), cmap='magma')
    cbar = plt.colorbar(im, ax=ax, fraction=0.08, aspect=14)
    if com:
        cbar.ax.set_title('p(CoM)')
    else:
        cbar.ax.set_title('p(reversal)')
    ax.set_xticks(np.arange(7), np.arange(7)*5)
    ax.set_yticks(np.arange(7)[::-1], np.arange(7)*5)
    ax.set_ylabel(r'$t_{aff}$ (ms)')
    ax.set_xlabel(r'$t_{eff}$ (ms)')


def plot_p_rev_vs_taff_teff(subject='LE42'):
    sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_prev_matrix_params_rt_under_20_v2.npy'
    mat_pcom = np.load(sim_data)
    fig, ax = plt.subplots(1)
    rm_top_right_lines(ax)
    colormap = pl.cm.BrBG(np.linspace(0.1, 1,7))
    for j in range(7):
        ax.plot(5*np.arange(7), mat_pcom[j], color=colormap[j], label=j*5)
        ax.plot(30-5*j, mat_pcom[j][6-j], color='k', marker='o', markersize=6)
        # if j > 0:
        #     ax.plot(35-5*j, mat_pcom[j][7-j], color='r', marker='o', markersize=6)
        # if j > 1:
        #     ax.plot(40-5*j, mat_pcom[j][8-j], color='b', marker='o', markersize=6)
    ax.set_xlabel(r'$t_{aff}$')
    ax.legend(title=r'$t_{eff}$ (ms)', frameon=False)
    ax.set_ylabel('p(reversal)')


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


def estimate_coef_linear_reg(x, y):
  # number of observations/points
  n = np.size(x)
 
  # mean of x and y vector
  m_x = np.mean(x)
  m_y = np.mean(y)
 
  # calculating cross-deviation and deviation about x
  SS_xy = np.sum(y*x) - n*m_y*m_x
  SS_xx = np.sum(x*x) - n*m_x*m_x
 
  # calculating regression coefficients
  b_1 = SS_xy / SS_xx
  b_0 = m_y - b_1*m_x
 
  return b_0, b_1


def real_minst_vs_bound(df, sv_folder, data_folder, param=3,
                       param2=None, sim=False):
    """
    Function to check real or simulated minimum splitting time vs a fitted parameter.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    sv_folder : TYPE
        DESCRIPTION.
    data_folder : TYPE
        DESCRIPTION.
    param : int, optional
        Index of the parameter in the params list. The default is 3.
    param2 : int, optional
        Index of a 2nd parameter which will be divided by the first one if
        it is not None. The default is None.
    sim : boolean, optional
        Wether to load splitting time data from data (False) or from
        simulations (True). The default is False.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(1)
    subjects = df.subjid.unique()
    # pcom = []
    bound = []
    min_data = []
    rm_top_right_lines(ax)
    for subject in df.subjid.unique():
        if not sim:
            split_data = data_folder + subject + '/traj_data/' + subject + '_traj_split_stim_005.npz'
        if sim:
            split_data = data_folder + subject + '/sim_data/' + subject + '_traj_split_stim_005_forward.npz'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(split_data), exist_ok=True)
        if os.path.exists(split_data):
            split_data = np.load(split_data, allow_pickle=True)
            out_data_sbj = split_data['out_data_sbj']
        min_data.append(np.nanmin(out_data_sbj))
    for i_s, subject in enumerate(subjects):
        conf = np.load(sv_folder + 'parameters_MNLE_BADS' + subject + '.npy')
        # pcom.append(np.nanmean(df.loc[df.subjid == subject, column]))
        if param2 is None:
            bound.append(conf[param])
        else:
            bound.append(conf[param]/conf[param2])
    slope, intercept, r, p, std_err = linregress(x=np.array(bound), y=np.array(min_data))
    x = np.linspace(min(bound), max(bound), 10)
    y = slope*x + intercept
    ax.plot(np.array(bound), min_data, color='k', marker='o', linestyle='',
            label='data')
    ax.plot(x, y, color='r', label='{} {}x'.format(round(intercept, 3), round(slope, 3)))
    ax.text(0.055, 140, r'$R^2$ = ' + str(round(r**2, 3)))
    ax.text(0.055, 135, r'$p$ = ' + str(round(p, 3)))
    ax.set_ylabel('Min. ST (ms)')
    ax.legend()


def run_model(stim, zt, coh, gt, trial_index, human=False,
              subject=None, num_tr=None, load_params=True, params_to_explore=[],
              extra_label=''):
    if extra_label == '':
        model = edd2.trial_ev_vectorized
    if type(extra_label) is not int:
        if 'neg' in extra_label:
            model = model_variations.trial_ev_vectorized_neg_starting_point
        if '_1_ro' in extra_label:  # only with 1st readout
            model = model_variations.trial_ev_vectorized_without_2nd_readout
        if '_1_ro' in extra_label and 'com_modulation_' in extra_label:  # only with 1st readout (RESULAJ)
            model = model_variations.trial_ev_vectorized_CoM_without_update
        if '_2_ro' in extra_label and 'rand' not in extra_label:  # only with 2nd readout
            model = model_variations.trial_ev_vectorized_without_1st_readout
        if '_2_ro' in extra_label and 'rand' in extra_label:
            model = model_variations.trial_ev_vectorized_without_1st_readout_random_1st_choice
        if 'virt' in extra_label:
            model = edd2.trial_ev_vectorized
        if '_prior_sign_1_ro_' in extra_label:
            model = model_variations.trial_ev_vectorized_only_prior_1st_choice
        if 'silent' in extra_label or 'redo' in extra_label:
            model = edd2.trial_ev_vectorized
        if 'continuous' in extra_label:
            model = model_variations.trial_ev_vectorized_n_readouts
    if type(extra_label) is int:
        model = edd2.trial_ev_vectorized
    # dt = 5e-3
    if num_tr is not None:
        num_tr = num_tr
    else:
        num_tr = int(len(zt))
    data_augment_factor = 10  # 50 for 1 ms precision
    if not human:
        detect_CoMs_th = 8
    if human:
        detect_CoMs_th = 100
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
            if 'virt' not in str(extra_label):
                conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
            if 'virt' in str(extra_label):
                conf = np.load(SV_FOLDER + 'virt_params/' + 'parameters_MNLE_BADS_prt_n50_' +
                               str(extra_label) + '.npy')
        jitters = len(conf)*[0]
        # check if there are params to explore
        if len(params_to_explore) != 0:
            # update conf with params to explore
            for i, index in enumerate(params_to_explore[0]):
                conf[index] = params_to_explore[1][i]

            
    print('Number of trials: ' + str(stim.shape[1]))
    factor = 10 / data_augment_factor  # to normalize parameters
    p_w_zt = conf[0]+jitters[0]*np.random.rand()
    p_w_stim = conf[1]*factor+jitters[1]*np.random.rand()
    p_e_bound = conf[2]+jitters[2]*np.random.rand()
    p_com_bound = conf[3]*p_e_bound+jitters[3]*np.random.rand()
    p_t_aff = int(round(conf[4]/factor+jitters[4]*np.random.rand()))
    p_t_eff = int(round(conf[5]/factor++jitters[5]*np.random.rand()))
    p_t_a = int(round(conf[6]/factor+jitters[6]*np.random.rand()))
    p_w_a_intercept = conf[7]*factor+jitters[7]*np.random.rand()
    p_w_a_slope = -conf[8]*factor+jitters[8]*np.random.rand()
    p_a_bound = conf[9]+jitters[9]*np.random.rand()
    p_1st_readout = conf[10]+jitters[10]*np.random.rand()
    p_2nd_readout = conf[11]+jitters[11]*np.random.rand()
    p_leak = conf[12]*factor+jitters[12]*np.random.rand()
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
        model(zt=zt, stim=stim_temp, coh=coh,
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
        pro_vs_re, total_traj, x_val_at_updt, frst_traj_motor_time


def check_mean_std_time_com(df, com, time_com):
    lsubs = pd.DataFrame(df.loc[com & (df.special_trial == 0), 'subjid'].reset_index())
    time_com = np.array(time_com)[lsubs.index]
    timecom_list = []
    for i_s, subj in enumerate(lsubs.subjid.unique()):
        index = (lsubs['subjid'] == subj).values
        timecom_list.append(np.nanmean(time_com[index]))


def trajs_splitting_stim_all(df, ax, color, threshold=300, par_value=None,
                             rtbins=np.linspace(0, 150, 16),
                             trajectory="trajectory_y", plot=True):

    # split time/subject by coherence
    splitfun = fig_2.get_splitting_mat_simul
    df['traj'] = df.trajectory_y.values
    out_data = []
    for subject in df.subjid.unique():
        out_data_sbj = []
        for i in range(rtbins.size-1):
            evs = [0, 0.25, 0.5, 1]
            for iev, ev in enumerate(evs):
                matatmp =\
                    splitfun(df=df.loc[(df.subjid == subject)],
                             side=0, rtbin=i, rtbins=rtbins, coh=ev,
                             align="sound")
                
                if iev == 0:
                    mat = matatmp
                    evl = np.repeat(0, matatmp.shape[0])
                else:
                    mat = np.concatenate((mat, matatmp))
                    evl = np.concatenate((evl, np.repeat(ev, matatmp.shape[0])))
            max_mt = 800
            current_split_index =\
                fig_2.get_split_ind_corr(mat, evl, pval=0.0001, max_MT=max_mt,
                                         startfrom=0)+5
            if current_split_index >= rtbins[i]:
                out_data_sbj += [current_split_index]
            else:
                out_data_sbj += [np.nan]
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

    # because we might want to plot each subject connecting lines, lets iterate
    # draw  datapoints

    error_kws = dict(ecolor=color, capsize=2,
                     color=color, marker='o', label=str(par_value*5))
    xvals = binsize/2 + binsize * np.arange(rtbins.size-1)
    if plot:
        ax.errorbar(
            xvals,
            # we do the mean across rtbin axis
            np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1),
            # other axes we dont care
            yerr=fig_2.sem(out_data.reshape(rtbins.size-1, -1),
                           axis=1, nan_policy='omit'),
            **error_kws)
    # if draw_line is not None:
    #     ax.plot(*draw_line, c='r', ls='--', zorder=0, label='slope -1')
    min_st = np.nanmin(np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1))
    rt_min_split = xvals[np.where(np.nanmean(out_data.reshape(rtbins.size-1, -1), axis=1) == min_st)[0]]
    # ax.arrow(rt_min_split+28, min_st, -12, 0, color=color, width=1, head_width=5)
    if rt_min_split.shape[0] > 1:
        rt_min_split = rt_min_split=[0]
    if sum(min_st.shape) > 1:
        min_st = min_st[0]
    return min_st, rt_min_split


def supp_parameter_analysis(stim, zt, coh, gt, trial_index, subjects,
                            subjid, sv_folder, idx=None):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.09, right=0.95,
                        hspace=0.4, wspace=0.45)
    ax = ax.flatten()
    labs = ['a', 'b', 'c', 'd', '', '']
    for i_a, a in enumerate(ax):
        rm_top_right_lines(a)
        a.text(-0.1, 1.17, labs[i_a], transform=a.transAxes, fontsize=16,
               fontweight='bold', va='top', ha='right')
    # ax[5].axis('off')
    # changing t_aff and t_eff
    params_to_explore_aff = [[4]] + [np.arange(7)]
    params_to_explore_eff = [[5]] + [np.arange(7)]
    params_to_explore = [params_to_explore_aff, params_to_explore_eff]
    plot_splitting_for_param(stim, zt, coh, gt, trial_index, subjects,
                             subjid, params_to_explore, ax=[ax[0], ax[2]])
    plot_min_st_vs_t_eff_cartoon(ax[1], offset=10)
    # # changing t_eff
    # plot_splitting_for_param(stim, zt, coh, gt, trial_index, subjects,
    #                          subjid, params_to_explore, ax=[ax[1], ax[4]])
    # changing bound action
    # params_to_explore = [[9]] + [np.round(np.logspace(0, 3, 11), 2)]
    # plot_mt_vs_coh_changing_action_bound(stim, zt, coh, gt, trial_index, subjects,
    #                                      subjid, params_to_explore, ax=ax[2])
    params_to_explore = [[9]] + [np.round(np.linspace(1, 14, 14), 2)]
    plot_pcom_vs_proac(stim, zt, coh, gt, trial_index, subjects,
                       subjid, params_to_explore, ax=ax[3],
                       param_title=r'$\theta_{AI}$', idx=idx)
    fig.savefig(sv_folder+'/supp_model_params.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'/supp_model_params.png', dpi=400, bbox_inches='tight')


def plot_min_st_vs_t_eff_cartoon(ax, offset=15):
    if ax is None:
        fig, ax = plt.subplots(1)
        rm_top_right_lines(ax)
    t_aff = np.arange(7)*5
    t_eff = np.copy(t_aff)
    colormap = pl.cm.BrBG(np.linspace(0.1, 1, len(t_eff)))[::-1]
    for i_taff, t_aff_val in enumerate(t_aff[::-1]):
        min_st = t_aff_val + t_eff + offset
        ax.plot(t_eff, min_st, color=colormap[i_taff], label=str(t_aff_val))
    ax.set_ylabel('Minimum splitting time (ms)')
    # ax.legend(title=r'$t_{aff} \;\; (ms)$', loc='upper right',
    #           frameon=False, bbox_to_anchor=(1.23, 1.17),
    #           labelspacing=0.1)
    ax.legend(title=r'$t_{aff} \;\; (ms)$', loc='upper right',
              frameon=False, bbox_to_anchor=(1.21, 1.145),
              labelspacing=0.35)
    ax.plot([0, 30], [0, 30], color='gray', linewidth=0.8)
    ax.set_xlabel(r'Efferent time $t_{eff} \;\; (ms)$')
    ax.annotate(text='', xy=(30, 31), xytext=(30, 39), arrowprops=dict(arrowstyle='<->'))
    ax.text(30.5, 31.5, 'offset')
    ax.text(15, 10, 'y=x', rotation=17, color='grey')
    # ax.text(0, 65, r'$min(ST) = t_{aff}+t_{eff}+offset$')
    ax.set_title(r'Prediction: $min(ST) = t_{aff}+t_{eff}+offset$',
                 fontsize=10)
    ax.set_ylim(-2, 71)


def plot_splitting_for_param(stim, zt, coh, gt, trial_index, subjects,
                             subjid, params_to_explore, ax=None):
    # fig, ax = plt.subplots(1)
    for a in ax:
        rm_top_right_lines(a)
    ax[0].plot([0, 155], [0, 155], color='k')
    ax[0].fill_between([0, 155], [0, 155], [0, 0],
                    color='grey', alpha=0.2)
    ax[0].set_xlim(-5, 155)
    ax[0].set_yticks([0, 100, 200, 300])
    ax[0].set_xlabel('Reaction time (ms)')
    ax[0].set_ylabel('Splitting time (ms)')
    change_params_exploration_and_plot(stim, zt, coh, gt, trial_index, subjects,
                                       subjid, params_to_explore, [ax[0], ax[1]])
    ax[0].set_ylim(-1, 275)
    labels = ['prior weight', 'stim weight', 'EA bound', 'CoM bound',
              r'$t_{aff}$', r'$t_{eff}$', 'tAction', 'intercept AI',
              'slope AI', 'AI bound', 'DV weight 1st readout',
              'DV weight 2nd readout', 'leak', 'MT noise std',
              'MT offset', 'MT slope T.I.']
    # ax[0].legend(title=labels[params_to_explore[0][0]], loc='upper center',
    #              bbox_to_anchor=(0.5, 1.2))



def plot_pcom_vs_param(stim, zt, coh, gt, trial_index, subjects,
                       subjid, params_to_explore, ax=None,
                       param_title=r'$\theta_{AI}$'):
    for a in ax:
        rm_top_right_lines(a)
    ax[1].set_xlabel(param_title)
    ax[1].set_ylabel('P(CoM)')
    ax[0].set_xlabel(param_title)
    ax[0].set_ylabel('P(proactive)')
    num_tr = int(len(coh))
    param_ind = params_to_explore[0][0]
    com_all = []
    reversals = []
    proac = []
    for ind in range(len(params_to_explore[1])):
        param_value = params_to_explore[1][ind]        
        param_iter = str(param_ind)+'_'+str(param_value)
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            pro_vs_re, trajs, x_val_at_updt =\
                run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                       trial_index=trial_index, num_tr=num_tr,
                                       subject_list=subjects, subjid=subjid,
                                       simulate=False,
                                       params_to_explore=[[param_ind], [param_value]],
                                       change_param=True,
                                       param_iter=param_iter)
        com_all.append(np.nanmean(com_model[reaction_time >= 0]))
        reversals.append(np.nanmean(com_model_detected[reaction_time >= 0]))
        proac.append(1-np.nanmean(pro_vs_re[reaction_time >= 0]))
    ax[1].plot(params_to_explore[1], com_all, color='k', marker='o')
    # ax[1].plot(params_to_explore[1], reversals, color='k', marker='o')
    ax[0].plot(params_to_explore[1], proac, color='k', marker='o')


def plot_pcom_vs_proac(stim, zt, coh, gt, trial_index, subjects,
                       subjid, params_to_explore, ax=None,
                       param_title=r'$\theta_{AI}$', idx=None):
    rm_top_right_lines(ax)
    ax.set_ylabel('P(CoM)')
    ax.set_xlabel('P(proactive)')
    num_tr = int(len(coh))
    param_ind = params_to_explore[0][0]
    com_all = []
    reversals = []
    proac = []
    acc_list = []
    rr_bound = []
    punishment = 2000
    colormap = pl.cm.magma(np.linspace(0., 0.9, len(params_to_explore[1])))
    for ind in range(len(params_to_explore[1])):
        param_value = params_to_explore[1][ind]        
        param_iter = str(param_ind)+'_'+str(param_value)
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            pro_vs_re, trajs, x_val_at_updt =\
                run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                       trial_index=trial_index, num_tr=num_tr,
                                       subject_list=subjects, subjid=subjid,
                                       simulate=False,
                                       params_to_explore=[[param_ind], [param_value]],
                                       change_param=True,
                                       param_iter=param_iter)
        sum_mt = np.sum([len(tr) for tr in trajs])
        sum_hit_model = np.sum(hit_model)
        sum_rt = np.sum(reaction_time)
        rr = 1000 * sum_hit_model / (
            sum_rt + sum_mt + punishment * (np.sum(hit_model == 0)))
        rr_bound.append(rr)
        if idx is None:
            com_all.append(np.nanmean(com_model[(reaction_time >= 0)]))
            acc_list.append(np.nanmean(hit_model[(reaction_time >= 0)]))
            reversals.append(np.nanmean(com_model_detected[(reaction_time >= 0)]))
            proac.append(1-np.nanmean(pro_vs_re[(reaction_time >= 0)]))
        else:
            com_all.append(np.nanmean(com_model[(reaction_time >= 0) & idx &
                                                (reaction_time < 1000)]))
            reversals.append(np.nanmean(com_model_detected[(reaction_time >= 0) &
                                                           idx & (reaction_time < 1000)]))
            acc_list.append(np.nanmean(hit_model[(reaction_time >= 0) & 
                                                 idx & (reaction_time < 1000)]))
            proac.append(1-np.nanmean(pro_vs_re[(reaction_time >= 0) &
                                                idx & (reaction_time < 1000)]))
    ax.plot(proac, com_all, color='k', linewidth=0.9)
    for i_val, pcom in enumerate(com_all):
        ax.plot(proac[i_val], pcom,
                color=colormap[i_val], marker='o', markersize=7)
    fig2, ax2 = plt.subplots(1)
    img = ax2.imshow(np.array([[min(params_to_explore[1]),
                                max(params_to_explore[1])]]), cmap="magma")
    img.set_visible(False)
    plt.close(fig2)
    cbar = plt.colorbar(img, orientation="vertical", ax=ax, cmap='magma', fraction=0.08,
                        aspect=12)
    cbar.ax.set_title(r'$\theta_{AI}$')
    ax.set_xticks([0, 0.4, 0.8])
    ind = np.where(params_to_explore[1] == 2)[0][0]
    ax.arrow(proac[ind]-0.16, com_all[ind], 0.1, 0, color='k', head_width=0.015)
    ax.text(proac[ind]-0.28, com_all[ind], r'$\theta_{AI}^*$')


def plot_mt_vs_coh_changing_action_bound(stim, zt, coh, gt, trial_index, subjects,
                                         subjid, params_to_explore, ax):
    rm_top_right_lines(ax)
    ax.set_xlabel('Stimulus evidence towards response')
    ax.set_ylabel('Movement time (ms)')
    action_bound_exploration_and_plot(stim, zt, coh, gt, trial_index, subjects,
                                      subjid, params_to_explore, ax)


def action_bound_exploration_and_plot(stim, zt, coh, gt, trial_index, subjects,
                                      subjid, params_to_explore, ax):
    num_tr = int(len(coh))
    colormap = pl.cm.magma(np.linspace(0., 1, len(params_to_explore[1])))
    param_ind = params_to_explore[0][0]
    ind_stim = np.sum(stim, axis=0) != 0
    for ind in range(len(params_to_explore[1])):
        param_value = params_to_explore[1][ind]        
        param_iter = str(param_ind)+'_'+str(param_value)
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            _, trajs, x_val_at_updt =\
                run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                       trial_index=trial_index, num_tr=num_tr,
                                       subject_list=subjects, subjid=subjid,
                                       simulate=False,
                                       params_to_explore=[[param_ind], [param_value]],
                                       change_param=True,
                                       param_iter=param_iter)
        MT = [len(t) for t in trajs]
        df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                                'sound_len': reaction_time,
                                'rewside': (gt + 1)/2,
                                'R_response': (resp_fin+1)/2,
                                'resp_len': np.array(MT),
                                'subjid': subjid, 'allpriors': zt})
        df_sim['choice_x_coh'] = np.round(coh * resp_fin, 2)
        df_sim['norm_allpriors'] = norm_allpriors_per_subj(df_sim)
        df_sim = df_sim.loc[ind_stim]
        mt_all = np.empty((len(subjects), 7))
        mt_all[:] = np.nan
        prior_lim = np.quantile(df_sim.norm_allpriors.abs(), 0.2)
        ev_vals = np.sort(np.unique(df_sim.choice_x_coh))
        for i_sub, subj in enumerate(subjects):
            for i_ev, ev in enumerate(ev_vals):
                index = (df_sim.choice_x_coh.values == ev) &\
                        (df_sim.sound_len.values >= 0) & (df_sim.subjid == subj) &\
                        (df_sim.norm_allpriors <= prior_lim)
                mt_all[i_sub, i_ev] = np.nanmean(df_sim.loc[index, 'resp_len'])
        mt_mean = np.nanmean(mt_all, axis=0)
        mt_err = np.nanstd(mt_all, axis=0) / np.sqrt(len(subjects))
        ax.errorbar(ev_vals, mt_mean, mt_err, color=colormap[ind], linewidth=1.4)
    fig2, ax2 = plt.subplots(1)
    img = ax2.imshow(np.array([[min(params_to_explore[1]),
                                max(params_to_explore[1])]]), cmap="magma", norm=LogNorm())
    img.set_visible(False)
    plt.close(fig2)
    cbar = plt.colorbar(img, orientation="vertical", ax=ax, cmap='magma', fraction=0.02)
    cbar.ax.set_title(r'$\theta_{AI}$')



def change_params_exploration_and_plot(stim, zt, coh, gt, trial_index, subjects,
                                       subjid, params_to_explore, ax, matrix=False):
    num_tr = int(len(coh))
    ind_stim = np.sum(stim, axis=0) != 0
    params_to_explore_aff, params_to_explore_eff = params_to_explore
    param_aff = params_to_explore_aff[0][0]
    param_eff = params_to_explore_eff[0][0]
    colormap = pl.cm.BrBG(np.linspace(0.1, 1, len(params_to_explore_eff[1])))
    subject = str(np.unique(subjid)[0])
    sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_splitting_matrix_params_2_5ms_bin.npy'
    # create folder if it doesn't exist
    os.makedirs(os.path.dirname(sim_data), exist_ok=True)
    if os.path.exists(sim_data):
        mat_min_st = np.load(sim_data)
    else:
        mat_min_st = np.empty((7, 7))
        mat_min_st[:] = np.nan
        for ind_aff in range(len(params_to_explore_aff[1])):
            min_st_list = []
            rt_min_sp = []
            aff_value = params_to_explore_aff[1][ind_aff]
            for ind_eff in range(len(params_to_explore_eff[1])):
                eff_value = params_to_explore_aff[1][ind_eff]
                param_iter = str(param_aff)+'_'+str(aff_value)+'_'+str(param_eff)+'_'+str(eff_value)+'_res_1ms'
                hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
                    _, trajs, x_val_at_updt =\
                        run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                               trial_index=trial_index, num_tr=num_tr,
                                               subject_list=subjects, subjid=subjid,
                                               simulate=False,
                                               params_to_explore=[[param_aff, param_eff], [aff_value, eff_value]],
                                               change_param=True,
                                               param_iter=param_iter)
                MT = [len(t) for t in trajs]
                df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                                        'sound_len': reaction_time,
                                        'rewside': (gt + 1)/2,
                                        'R_response': (resp_fin+1)/2,
                                        'resp_len': np.array(MT)*1e-3,
                                        'subjid': subjid})
                df_sim = df_sim.loc[ind_stim]
                min_st, rt_min_split = trajs_splitting_stim_all(df=df_sim, ax=ax[0], color=colormap[ind_eff], threshold=300,
                                                                rtbins=np.linspace(0, 150, 61),
                                                                trajectory="trajectory_y", par_value=param_eff, plot=False)
                min_st_list.append(min_st)
                rt_min_sp.append(rt_min_split)
                mat_min_st[ind_aff, ind_eff] = min_st
            i = 0
            for min_st, rt_min_split in zip(min_st_list, rt_min_sp):
                ax[0].plot(rt_min_split, min_st, marker='o', color=colormap[i],
                           markersize=8)
                i += 1
        np.save(sim_data, mat_min_st)
            # i = 0
    # example ST vs RT eff=3
    eff_value = 6
    min_st_list = []
    rt_min_sp = []
    for ind_aff in range(len(params_to_explore_aff[1])):
        aff_value = params_to_explore_aff[1][ind_aff]
        param_iter = str(param_aff)+'_'+str(aff_value)+'_'+str(param_eff)+'_'+str(eff_value)+'_res_1ms'
        hit_model, reaction_time, com_model_detected, resp_fin, com_model,\
            _, trajs, x_val_at_updt =\
                run_simulation_different_subjs(stim=stim, zt=zt, coh=coh, gt=gt,
                                        trial_index=trial_index, num_tr=num_tr,
                                        subject_list=subjects, subjid=subjid,
                                        simulate=False,
                                        params_to_explore=[[param_aff, param_eff], [aff_value, eff_value]],
                                        change_param=True,
                                        param_iter=param_iter)
        MT = [len(t) for t in trajs]
        df_sim = pd.DataFrame({'coh2': coh, 'avtrapz': coh, 'trajectory_y': trajs,
                                'sound_len': reaction_time,
                                'rewside': (gt + 1)/2,
                                'R_response': (resp_fin+1)/2,
                                'resp_len': np.array(MT)*1e-3,
                                'subjid': subjid})
        df_sim = df_sim.loc[ind_stim]
        min_st, rt_min_split = trajs_splitting_stim_all(df=df_sim, ax=ax[0], color=colormap[ind_aff], threshold=300,
                                                        rtbins=np.linspace(0, 150, 16),
                                                        trajectory="trajectory_y", par_value=aff_value, plot=True)
        min_st_list.append(min_st)
        rt_min_sp.append(rt_min_split)
    i = 0
    for min_st, rt_min_split in zip(min_st_list, rt_min_sp):
        ax[0].plot(rt_min_split, min_st, marker='o', color=colormap[i],
                    markersize=8)
        i += 1
    ax[0].text(70, 200, r'$t_{eff} = $ '+ str(eff_value*5) + ' ms')
    vals = (np.arange(7)*5)[::-1]
    legendelements = []
    for col, val in zip(colormap[::-1], vals):
        legendelements.append(Line2D([0], [0], color=col, lw=2,
                             label=val))
    ax[0].legend(handles=legendelements,
                 title=r'$t_{aff}$ (ms)', loc='upper right',
                 frameon=False, bbox_to_anchor=(1.24, 1.175))
    if matrix:
        im = ax[1].imshow(np.flipud(mat_min_st))
        cbar = plt.colorbar(im, ax=ax[1], fraction=0.04)
        cbar.ax.set_title('   Min. ST (ms)')
        ax[1].set_yticks([6, 3, 0], [0, 15, 30])
        ax[1].set_xticks([0, 3, 6], [0, 15, 30])
        ax[1].set_ylabel(r'$t_{aff}$ (ms)')
        ax[1].set_xlabel(r'$t_{eff}$ (ms)')
    else:
        for i_aff, min_st_list in enumerate(mat_min_st[::-1]):
            ax[1].plot(params_to_explore_eff[1]*5, min_st_list, color=colormap[::-1][i_aff],
                       label=str(params_to_explore_aff[1][::-1][i_aff]*5))
    ax[1].set_ylabel('Minimum splitting time (ms)')
    ax[1].annotate(text='', xy=(30, 31), xytext=(30, 43), arrowprops=dict(arrowstyle='<->'))
    ax[1].text(15, 10, 'y=x', rotation=17, color='grey')
    ax[1].text(30.5, 31.5, 'offset')
    # ax[1].set_yticks([20, 30, 40, 50, 60])
    # ax[1].set_xticks([20, 30, 40])
    ax[1].plot([0, 30], [0, 30], color='gray', linewidth=0.8)
    # ax[1].set_xlim([13, 47])
    # ax[1].set_ylim([14, 61])
    ax[1].set_xlabel(r'Efferent time $t_{eff} \;\; (ms)$')
    ax[1].legend(title=r'$t_{aff} \;\; (ms)$', loc='upper right',
                 frameon=False, bbox_to_anchor=(1.24, 1.175),
                 labelspacing=0.35)
    ax[1].set_ylim(-2, 71)


def run_simulation_different_subjs(stim, zt, coh, gt, trial_index, subject_list,
                                   subjid, human=False, num_tr=None, load_params=True,
                                   simulate=True, change_param=False, param_iter=None,
                                   params_to_explore=[], extra_label=''):
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
            if change_param:
                sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_simulation_' + str(param_iter) + '.pkl'
            else:
                sim_data = DATA_FOLDER + subject + '/sim_data/' + subject + '_simulation' + str(extra_label) + '.pkl'
        if human:
            if change_param:
                sim_data = DATA_FOLDER + '/Human/' + str(subject) + '/sim_data/' + str(subject) + '_simulation_' + str(param_iter) + '.pkl'
            else:
                sim_data = DATA_FOLDER + '/Human/' + str(subject) + '/sim_data/' + str(subject) + '_simulation.pkl'
        # create folder if it doesn't exist
        os.makedirs(os.path.dirname(sim_data), exist_ok=True)
        if os.path.exists(sim_data) and not simulate:
            print('Loading simulated data')
            data_simulation = np.load(sim_data, allow_pickle=True)
            hit_model_tmp = data_simulation['hit_model_tmp']
            reaction_time_tmp = data_simulation['reaction_time_tmp']
            detected_com_tmp = data_simulation['detected_com_tmp']
            resp_fin_tmp = data_simulation['resp_fin_tmp']
            com_model_tmp = data_simulation['com_model_tmp']
            pro_vs_re_tmp = data_simulation['pro_vs_re_tmp']
            total_traj_tmp = data_simulation['total_traj_tmp']
            x_val_at_updt_tmp = data_simulation['x_val_at_updt_tmp']
            # frst_traj_motor_time = data_simulation['frst_traj_motor_time']
        else:
            hit_model_tmp, reaction_time_tmp, detected_com_tmp, resp_fin_tmp,\
                com_model_tmp, pro_vs_re_tmp, total_traj_tmp, x_val_at_updt_tmp, frst_traj_motor_time_tmp=\
                run_model(stim=stim[:, index], zt=zt[index], coh=coh[index],
                          gt=gt[index], trial_index=trial_index[index],
                          subject=subject, load_params=load_params, human=human,
                          params_to_explore=params_to_explore, extra_label=extra_label)
            data_simulation = {'hit_model_tmp': hit_model_tmp, 'reaction_time_tmp': reaction_time_tmp,
                               'detected_com_tmp': detected_com_tmp, 'resp_fin_tmp': resp_fin_tmp,
                               'com_model_tmp': com_model_tmp, 'pro_vs_re_tmp': pro_vs_re_tmp,
                               'total_traj_tmp': total_traj_tmp, 'x_val_at_updt_tmp': x_val_at_updt_tmp,
                               'frst_traj_motor_time_tmp':frst_traj_motor_time_tmp}
            pd.to_pickle(data_simulation, sim_data)
        hit_model = np.concatenate((hit_model, hit_model_tmp))
        reaction_time = np.concatenate((reaction_time, reaction_time_tmp))
        detected_com = np.concatenate((detected_com, detected_com_tmp))
        resp_fin = np.concatenate((resp_fin, resp_fin_tmp))
        com_model = np.concatenate((com_model, com_model_tmp))
        pro_vs_re = np.concatenate((pro_vs_re, pro_vs_re_tmp))
        total_traj = total_traj + total_traj_tmp
        x_val_at_updt = np.concatenate((x_val_at_updt, x_val_at_updt_tmp))
        # frst_traj_motor_time = np.concatenate((frst_traj_motor_time, frst_traj_motor_time_tmp))
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt  # , frst_traj_motor_time


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


def supp_parameter_recovery_test(margin=.6):
    fig, ax = plt.subplots(4, 4, figsize=(14, 12))
    ax = ax.flatten()
    plot_param_recovery_test(fig, ax,
            subjects=['Virtual_rat_random_params' for _ in range(50)],
            sv_folder=SV_FOLDER, corr=True)
    pos = ax[12].get_position()
    cbar_ax = fig.add_axes([pos.x0+pos.width/2, pos.y0-pos.height-margin,
                            pos.width*5, pos.height*4])
    plot_corr_matrix_prt(cbar_ax,
            subjects=['Virtual_rat_random_params' for _ in range(50)],
            sv_folder=SV_FOLDER)
    fig.savefig(SV_FOLDER + 'supp_prt.png', dpi=500, bbox_inches='tight')


def supp_com_threshold_matrices(df):
    dfth = pd.read_csv(SV_FOLDER + 'com_diff_thresholds.csv')
    # fig, ax = plt.subplots(nrows=3, ncols=10, figsize=(15, 6))
    # ax = ax.flatten()
    # thlist = np.linspace(1, 10, 10)
    # zt = df.allpriors.values
    # coh = df.coh2.values
    # decision = df.R_response.values*2 - 1
    # nbins = 7
    # for i_th, threshold in enumerate(thlist):
    #     com = dfth['com_'+str(threshold)]
    #     df_1 = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
    #                          'norm_allpriors': zt/max(abs(zt)),
    #                          'R_response': (decision+1)/2})
    #     matrix_side_0 = fig_3.com_heatmap_marginal_pcom_side_mat(df=df_1, side=0)
    #     matrix_side_1 = fig_3.com_heatmap_marginal_pcom_side_mat(df=df_1, side=1)
    #     # L-> R
    #     vmax = max(np.max(matrix_side_0), np.max(matrix_side_1))
    #     pcomlabel_1 = 'Left to Right'   # r'$p(CoM_{L \rightarrow R})$'
    #     im = ax[i_th].imshow(matrix_side_1, vmin=0, vmax=vmax)
    #     plt.sca(ax[i_th])
    #     plt.colorbar(im, fraction=0.04)
    #     # R -> L
    #     pcomlabel_0 = 'Right to Left'  # r'$p(CoM_{L \rightarrow R})$'
    #     im = ax[i_th+len(thlist)].imshow(matrix_side_0, vmin=0, vmax=vmax, cmap='magma')
    #     ax[i_th+len(thlist)].yaxis.set_ticks_position('none')
    #     plt.sca(ax[i_th+len(thlist)])
    #     plt.colorbar(im, fraction=0.04)
    #     ax[i_th].set_title('stim, th = {} px'.format(threshold))
    #     ax[i_th+len(thlist)].set_xlabel('Prior evidence')
    #     if i_th == 0:
    #         ax[i_th].set_ylabel(pcomlabel_1 + ', avg. stim.')
    #         ax[i_th+len(thlist)].set_ylabel(pcomlabel_0 + ', avg. stim.')
    #         ax[i_th + 2*len(thlist)].set_ylabel('Position (px)')
    #     for ax_i in [ax[i_th], ax[i_th+len(thlist)]]:
    #         ax_i.set_yticklabels(['']*nbins)
    #         ax_i.set_xticklabels(['']*nbins)
    #     cont = 1
    #     j = 1000
    #     while cont <= 10:
    #         if threshold < 10:
    #             if com[j] and df.trajectory_y.values[j][-1] > 1 and\
    #               df.R_response.values[j] == 1 and\
    #               not dfth['com_'+str(threshold+0.5)][j] and\
    #               df.trajectory_y.values[j][-0] >= -2 and\
    #               df.trajectory_y.values[j][-0] <= 10:
    #                 traj = df.trajectory_y.values[j]
    #                 time_trajs = df.time_trajs.values[j]
    #                 traj -= np.nanmean(traj[
    #                     (time_trajs >= -100)*(time_trajs <= 0)])
    #                 ax[i_th + 2*len(thlist)].plot(time_trajs,
    #                                               traj,
    #                                               color='k', alpha=0.7)
    #                 cont += 1
    #         if threshold == 10:
    #             if com[j] and df.trajectory_y.values[j][-1] > 1 and\
    #               df.R_response.values[j] == 1 and\
    #               df.trajectory_y.values[j][-0] >= -2 and\
    #               df.trajectory_y.values[j][-0] <= 10:
    #                 traj = df.trajectory_y.values[j]
    #                 time_trajs = df.time_trajs.values[j]
    #                 traj -= np.nanmean(traj[
    #                     (time_trajs >= -100)*(time_trajs <= 0)])
    #                 ax[i_th + 2*len(thlist)].plot(time_trajs,
    #                                               traj,
    #                                               color='k', alpha=0.7)
    #                 cont += 1
    #         j += 1
    #     ax[i_th + 2*len(thlist)].set_xlabel('Time')
    #     ax[i_th + 2*len(thlist)].set_ylim(-25, 25)
    #     ax[i_th + 2*len(thlist)].set_xlim(-100, 500)
    #     ax[i_th + 2*len(thlist)].axhline(-threshold, color='r', linestyle='--',
    #                                      alpha=0.5)
    #     ax[i_th + 2*len(thlist)].axvline(0, color='r', linestyle='--',
    #                                      alpha=0.5)
    thlist = np.linspace(0.5, 10, 20)
    mean_com = []
    fig2, ax2 = plt.subplots(1)
    rm_top_right_lines(ax2)
    for i_th, threshold in enumerate(thlist):
        com = dfth['com_'+str(threshold)]
        mean_com.append(np.nanmean(com))
    ax2.plot(thlist, mean_com, color='k', marker='o')
    ax2.set_yscale('log')
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


def plot_params_all_subs_rats_humans(subjects, subjectsh, sv_folder=SV_FOLDER, diff_col=True):
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
    confh_mat = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        conf_mat[:, i_s] = conf
        confh = np.load(SV_FOLDER + 'parameters_MNLE_BADS_human_subj_' + str(subjectsh[i_s]) + '.npy')
        confh_mat[:, i_s] = confh
    for i in range(len(labels)):
        if i == 4 or i == 5 or i == 6:
            sns.kdeplot(conf_mat[i, :]*5, ax=ax[i], label='Rats')
            sns.kdeplot(confh_mat[i, :]*5, ax=ax[i], label='Humans')
            ax[i].set_xlabel(labels[i] + str(' (ms)'))
        else:
            sns.kdeplot(conf_mat[i, :], ax=ax[i], label='Rats')
            sns.kdeplot(confh_mat[i, :], ax=ax[i], label='Humans')
            ax[i].set_xlabel(labels[i] + str(' ms'))
            ax[i].set_xlabel(labels[i])
    ax[0].legend(fontsize=12)


def supp_plot_params_all_subs(subjects, sv_folder=SV_FOLDER, diff_col=False):
    fig, ax = plt.subplots(4, 4)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92,
                        hspace=0.5, wspace=0.4)
    if diff_col:
        colors = pl.cm.jet(np.linspace(0., 1, len(subjects)))
    else:
        colors = ['k' for _ in range(len(subjects))]
    ax = ax.flatten()
    for a in ax:
        rm_top_right_lines(a)
    labels = [r'Prior weight, $z_P$', r'Stimulus drift, $a_P$',
              r'EA bound, $\theta_{DV}$',
              r'CoM bound, $\theta_{COM}$',
              r'Afferent time, $t_{aff}$', r'Efferent time, $t_{eff}$',
              r'AI time offset, $t_{AI}$',
              r'AI drift offset, $v_{AI}$',
              r'AI drift slope, $w_{AI}$',
              r'AI bound, $\theta_{AI}$',
              r'DV weight 1st readout, $\beta_{DV}$',
              r'DV weight update, $\beta_u$', r'Leak, $\lambda$',
              # r'MT noise variance, $\sigma_{MT}$',
              r'MT offset, $\beta_0$', r'MT slope, $\beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        conf[1] = conf[1] / 5
        conf[7] = conf[7] / 5
        conf[8] = conf[8] / 5
        conf_mat[:, i_s] = np.delete(conf, -3)
    for i in range(len(labels)):
        if i == 4 or i == 5 or i == 6:
            sns.violinplot(conf_mat[i, :]*5, ax=ax[i], orient='h', color='lightskyblue',
                           fmt='g', linewidth=0)
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s]*5,
                           0.1*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=4)
            ax[i].set_xlabel(labels[i] + str(' (ms)'))
        else:
            sns.violinplot(conf_mat[i, :], ax=ax[i], orient='h', color='lightskyblue',
                           fmt='g', linewidth=0)
            for i_s in range(len(subjects)):
                ax[i].plot(conf_mat[i, i_s],
                           0.1*np.random.randn(),
                           color=colors[i_s], marker='o', linestyle='',
                           markersize=4)
            ax[i].set_xlabel(labels[i])
        ax[i].set_yticks([])
        ax[i].spines['left'].set_visible(False)
    ax[-1].axis('off')


def plot_corr_matrix_prt(ax,
        subjects=['Virtual_rat_random_params' for _ in range(50)],
        sv_folder=SV_FOLDER):
    labels = [r'Prior weight, $z_P$', r'Stimulus drift, $a_P$',
              r'EA bound, $\theta_{DV}$',
              r'CoM bound, $\theta_{COM}$',
              r'Afferent time, $t_{aff}$', r'Efferent time, $t_{eff}$',
              r'AI time offset, $t_{AI}$',
              r'AI drift offset, $v_{AI}$',
              r'AI drift slope, $w_{AI}$',
              r'AI bound, $\theta_{AI}$',
              r'DV weight 1st readout, $\beta_{DV}$',
              r'DV weight update, $\beta_u$', r'Leak, $\lambda$',
              r'MT noise variance, $\sigma_{MT}$',
              r'MT offset, $\beta_0$', r'MT slope, $\beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    conf_mat_rec = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        # conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        # conf_rec = np.load(SV_FOLDER + 'parameters_MNLE_BADSprt' + subject + '.npy')
        conf = np.load(SV_FOLDER + 'virt_params/' +
                       'parameters_MNLE_BADS_prt_n50_' + 'virt_sim_' + str(i_s) + '.npy')
        conf_rec =  np.load(SV_FOLDER + 'virt_params/' +
                            'parameters_MNLE_BADS_prt_n50_prt_' + str(i_s) + '.npy')
        conf_mat[:, i_s] = conf
        conf_mat_rec[:, i_s] = conf_rec
    corr_mat = np.empty((len(labels), len(labels)))
    corr_mat[:] = np.nan
    for i in range(len(labels)):
        for j in range(len(labels)):
            corr_mat[i, j] = np.corrcoef(conf_mat[i, :], conf_mat_rec[j, :])[1][0]
    im = ax.imshow(corr_mat.T, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xticks(np.arange(16), labels, rotation='270', fontsize=12)
    ax.set_yticks(np.arange(16), labels, fontsize=12)
    ax.set_ylabel('Inferred parameters', fontsize=14)
    ax.set_xlabel('Original parameters', fontsize=14)



def plot_param_recovery_test(fig, ax,
        subjects=['Virtual_rat_random_params' for _ in range(50)],
        sv_folder=SV_FOLDER, corr=True):
    # fig, ax = plt.subplots(4, 4, figsize=(14, 12))
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.06, right=0.94,
                        hspace=0.6, wspace=0.5)
    for a in ax:
        rm_top_right_lines(a)
        pos = a.get_position()
        a.set_position([pos.x0, pos.y0, pos.height, pos.height])
    labels = [r'Prior weight, $z_P$', r'Stimulus drift, $a_P$',
              r'EA bound, $\theta_{DV}$',
              r'CoM bound, $\theta_{COM}$',
              r'Afferent time, $t_{aff}$ (ms)', r'Efferent time, $t_{eff}$ (ms)',
              r'AI time offset, $t_{AI}$ (ms)',
              r'AI drift offset, $v_{AI}$',
              r'AI drift slope, $w_{AI}$',
              r'AI bound, $\theta_{AI}$',
              r'DV weight 1st readout, $\beta_{DV}$',
              r'DV weight update, $\beta_u$', r'Leak, $\lambda$',
              r'MT noise variance, $\sigma_{MT}$',
              r'MT offset, $\beta_0$', r'MT slope, $\beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    conf_mat_rec = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        # conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        # conf_rec = np.load(SV_FOLDER + 'parameters_MNLE_BADSprt' + subject + '.npy')
        conf = np.load(SV_FOLDER + 'virt_params/' +
                       'parameters_MNLE_BADS_prt_n50_' + 'virt_sim_' + str(i_s) + '.npy')
        conf_rec =  np.load(SV_FOLDER + 'virt_params/' +
                            'parameters_MNLE_BADS_prt_n50_prt_' + str(i_s) + '.npy')
        conf_mat[:, i_s] = conf
        conf_mat_rec[:, i_s] = conf_rec
    mlist = []
    rlist = []
    for i in range(len(labels)):
        max_val = max(conf_mat_rec[i, :])
        max_val_rec = max(conf_mat[i, :])
        max_total = max(max_val, max_val_rec)
        # min_val = min(conf_mat_rec[i, :])
        # min_val_rec = min(conf_mat[i, :])
        min_total = 0
        ax[i].set_title(labels[i])
        if i == 4 or i == 5 or i == 6:
            ax[i].plot(conf_mat[i, :]*5, conf_mat_rec[i, :]*5,
                       marker='o', color='k', linestyle='')
            # ax[i].set_xlabel(labels[i] + str(' (ms)'), fontsize=12)
            # ax[i].set_ylabel(labels[i] + str(' (ms),')  + ' PRT', fontsize=12)
            ax[i].plot([5*min_total*0.4,
                        5*max_total*1.6],
                       [5*min_total*0.4,
                        5*max_total*1.6])
            ax[i].set_xlim(5*min_total*0.4, 5*max_total*1.6)
            ax[i].set_ylim(5*min_total*0.4, 5*max_total*1.6)
            out = linregress(conf_mat[i, :]*5, 5*conf_mat_rec[i, :])
        else:
            ax[i].plot(conf_mat[i, :], conf_mat_rec[i, :],
                       marker='o', color='k', linestyle='')
            # ax[i].set_xlabel(labels[i], fontsize=12)
            # ax[i].set_ylabel(labels[i] + ' PRT', fontsize=12)
            ax[i].plot([min_total*0.4, max_total*1.6],
                       [min_total*0.4, max_total*1.6])
            ax[i].set_xlim(min_total*0.4, max_total*1.6)
            ax[i].set_ylim(min_total*0.4, max_total*1.6)
            out = linregress(conf_mat[i, :], conf_mat_rec[i, :])
        r = out.rvalue
        m = out.slope
        if i != 13:
            mlist.append(m)
            rlist.append(r)
        # p2 = out.pvalue
        # if corr:
        #     ax[i].set_title(r'$\rho=$' + str(round(r, 3)) + ', p-value = ' + str(round(p2, 3)),
        #                     pad=10)
        # else:
        #     ax[i].set_title('m=' + str(round(m, 3)) + ', p-value = ' + str(round(p2, 3)),
        #                     pad=10)
        xtcks = ax[i].get_yticks()
        if i == 8:
            ax[i].set_xticks(xtcks[:-1], [f'{x:.0e}' for x in xtcks[:-1]])
            ax[i].set_yticks(xtcks[:-1], [f'{x:.0e}' for x in xtcks[:-1]])
        else:
            ax[i].set_xticks(xtcks[:-1])    
        if i > 11:
            ax[i].set_xlabel('Original parameter')
        if i % 4 == 0:
            ax[i].set_ylabel('Inferred parameter')
        #     ax[i].ticklabel_format(style='sci', axis='both',
        #                            scilimits=(5, 1))
    # plt.figure()
    # plt.plot(mlist, rlist)
    # print(np.median(mlist))
    # print(np.mean(mlist))
    # plt.figure()
    # sns.violinplot(mlist, inner='point')
    # fig2, ax2 = plt.subplots(1)
    # t_aff_t_eff = (conf_mat[4, :] + conf_mat[5, :])*5
    # t_aff_t_eff_rec = (conf_mat_rec[4, :] + conf_mat_rec[5, :])*5
    # ax2.set_title(linregress(t_aff_t_eff, t_aff_t_eff_rec).rvalue)
    # ax2.plot(t_aff_t_eff_rec, t_aff_t_eff,
    #          marker='o', color='k', linestyle='')
    # ax2.set_xlabel('t_aff + t_eff, original')
    # ax2.set_ylabel('t_aff + t_eff, recovered')
    # min_total = 0
    # max_total = np.max((t_aff_t_eff, t_aff_t_eff_rec))
    # ax2.plot([min_total*0.4,
    #           max_total*1.6],
    #          [min_total*0.4,
    #           max_total*1.6])



def plot_param_recovery_test_boxplot_difference(subjects, sv_folder=SV_FOLDER):
    fig, ax = plt.subplots(1)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92,
                        hspace=0.5, wspace=0.4)
    rm_top_right_lines(ax)
    labels = [r'$\Delta z_P$', r'$\Delta a_P$',
              r'$\Delta \theta_{DV}$',
              r'$\Delta  \theta_{COM}$',
              r'$\Delta t_{aff}$', r'$\Delta t_{eff}$',
              r'$\Delta t_{AI}$',
              r'$\Delta v_{AI}$',
              r'$\Delta w_{AI}$',
              r'$\Delta \theta_{AI}$',
              r'$\Delta \beta_{DV}$',
              r'$\Delta \beta_u$', r'$\Delta \lambda$',
              r'$\Delta \sigma_{MT}$',
              r'$\Delta \beta_0$', r'$\Delta \beta_{TI}$']
    conf_mat = np.empty((len(labels), len(subjects)))
    conf_mat_rec = np.empty((len(labels), len(subjects)))
    for i_s, subject in enumerate(subjects):
        conf = np.load(SV_FOLDER + 'parameters_MNLE_BADS' + subject + '.npy')
        conf_rec = np.load(SV_FOLDER + 'parameters_MNLE_BADSprt' + subject + '.npy')
        conf_mat[:, i_s] = conf
        conf_mat_rec[:, i_s] = conf_rec
    df = pd.DataFrame()
    for i in range(len(labels)):
        if i == 4 or i == 5 or i == 6:
            conf_mat[i, :] *= 5
            conf_mat_rec[i, :] *= 5
        diffs = conf_mat_rec[i, :] - conf_mat[i, :]
        df[labels[i]] = diffs / np.mean(conf_mat_rec[i, :]) * 100
    ax.axvline(x=0, color='r')
    sns.boxplot(df, ax=ax, orient='horizontal')
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel('Relative difference (%)')


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


def supp_plot_rt_distros_data_model(df, df_sim, sv_folder):
    # plots the RT distros conditioning on coh
    fig, ax = plt.subplots(6, 5, figsize=(9, 10))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.06, right=0.95,
                        hspace=0.4, wspace=0.4)
    ax = ax.flatten()
    ev_vals = [0, 1]
    colormap = pl.cm.gist_gray_r(np.linspace(0.4, 1, len(ev_vals)))
    cmap_model = pl.cm.Reds(np.linspace(0.4, 1, len(ev_vals)))
    subjects = df.subjid.unique()
    labs_data = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    labs_model = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]
    for i_s, subj in enumerate(subjects):
        rm_top_right_lines(ax[labs_model[i_s]])
        rm_top_right_lines(ax[labs_data[i_s]])
        pos_ax_mod = ax[labs_model[i_s]].get_position()
        ax[labs_model[i_s]].set_position([pos_ax_mod.x0,
                                          pos_ax_mod.y0 + pos_ax_mod.height/12.5,
                                          pos_ax_mod.width,
                                          pos_ax_mod.height])
        pos_ax_dat = ax[labs_data[i_s]].get_position()
        ax[labs_data[i_s]].set_position([pos_ax_dat.x0,
                                          pos_ax_dat.y0 - pos_ax_dat.height/12.5,
                                          pos_ax_dat.width,
                                          pos_ax_dat.height])
        if (i_s+1) % 5 == 0:
            axmod = ax[labs_model[i_s]].twinx()
            axdat = ax[labs_data[i_s]].twinx()
            axdat.set_ylabel('Data')
            axmod.set_ylabel('Model')
            axmod.set_yticks([])
            axdat.set_yticks([])
            axmod.spines['bottom'].set_visible(False)
            axdat.spines['bottom'].set_visible(False)
            rm_top_right_lines(axdat)
            rm_top_right_lines(axmod)
        df_1 = df[df.subjid == subj]
        df_sim_1 = df_sim[df_sim.subjid == subj]
        coh_vec = df_1.coh2.values
        coh = df_sim_1.coh2.abs().values
        ax[labs_data[i_s]].set_ylim(-0.0001, 0.011)
        ax[labs_model[i_s]].set_ylim(-0.0001, 0.011)
        for ifb, fb in enumerate(df_1.fb):
            for j in range(len(fb)):
                coh_vec = np.append(coh_vec, [df_1.coh2.values[ifb]])
        fix_breaks =\
            np.vstack(np.concatenate([df_1.sound_len/1000,
                                      np.concatenate(df_1.fb.values)-0.3]))
        for iev, ev in enumerate(ev_vals):
            index = np.abs(coh_vec) == ev
            fix_breaks_2 = fix_breaks[index]*1e3
            rt_model = df_sim_1.sound_len.values[coh == ev]
            sns.kdeplot(fix_breaks_2.reshape(-1),
                        color=colormap[iev], ax=ax[labs_data[i_s]])
            sns.kdeplot(rt_model,
                        color=cmap_model[iev], ax=ax[labs_model[i_s]])
        ax[labs_data[i_s]].set_xticks([])        
        ax[labs_data[i_s]].set_title(subj)
        ax[labs_data[i_s]].set_xlim(-205, 410)
        ax[labs_model[i_s]].set_xlim(-205, 410)
        if (i_s) % 5 != 0:
            axmod = ax[labs_model[i_s]]
            axdat = ax[labs_data[i_s]]
            axdat.set_ylabel('')
            axmod.set_ylabel('')
            axdat.set_yticks([])
            axmod.set_yticks([])
        if i_s < 10:
            ax[labs_model[i_s]].set_xticks([])        
        if i_s >= 10:
            ax[labs_model[i_s]].set_xlabel('RT (ms)')
    fig.savefig(sv_folder+'supp_fig_8.svg', dpi=400, bbox_inches='tight')
    fig.savefig(sv_folder+'supp_fig_8.png', dpi=400, bbox_inches='tight')


def supp_plot_rt_data_vs_model_all(df, df_sim):
    # plots the RT distros of data vs model
    fig, ax = plt.subplots(3, 5)
    ax = ax.flatten()
    subjects = df.subjid.unique()
    for i_s, subj in enumerate(subjects):
        rm_top_right_lines(ax[i_s])
        df_1 = df[df.subjid == subj]
        df_sim_1 = df_sim[df_sim.subjid == subj]
        coh_vec = df_1.coh2.values
        for ifb, fb in enumerate(df_1.fb):
            for j in range(len(fb)):
                coh_vec = np.append(coh_vec, [df_1.coh2.values[ifb]])
        fix_breaks =\
            np.vstack(np.concatenate([df_1.sound_len/1000,
                                      np.concatenate(df_1.fb.values)-0.3]))
        sns.kdeplot(fix_breaks.reshape(-1)*1e3,
                    color='k', ax=ax[i_s], label='Rats')
        sns.kdeplot(df_sim_1.sound_len,
                    color='r', ax=ax[i_s], label='Model')
        ax[i_s].set_title(subj)
        ax[i_s].set_xlim(-205, 410)
        if i_s >= 10:
            ax[i_s].set_xlabel('RT (ms)')
    ax[0].legend()


def check_perc_silent(df):
    subs_spec_trial = df.loc[df.special_trial == 2, 'subjid'].unique()
    l=[]
    for sub_sil in subs_spec_trial:
        l.append(np.sum(df.loc[df.subjid == sub_sil, 'special_trial']==2)
                 / len(df.loc[df.subjid == sub_sil, 'special_trial']))
    mean_l = np.mean(l)*100
    std_l = np.std(l)*100
    print(str(np.round(mean_l, 1)) + ' % +- ' + str(np.round(std_l, 1)))


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


def supp_mt_per_rat(df, df_sim, title=''):
    fig, ax = plt.subplots(5, 3)
    ax = ax.flatten()
    for a in ax:
        rm_top_right_lines(a)
    subjects = df.subjid.unique()
    for i_s, subj in enumerate(subjects):
        df_1 = df[df.subjid == subj]
        df_sim_1 = df_sim[df_sim.subjid == subj]
        # mt_nocom_data = df_1.loc[~df_1['CoM_sugg'], 'resp_len']*1e3
        # mt_com_data = df_1.loc[df_1['CoM_sugg'], 'resp_len']*1e3
        # mt_nocom_sim = df_sim_1.loc[~df_sim_1['com_detected'], 'resp_len']*1e3
        # mt_com_sim = df_sim_1.loc[df_sim_1['com_detected'], 'resp_len']*1e3
        mt_rat = df_1.resp_len.values*1e3
        mt_model = df_sim_1.resp_len.values*1e3
        sns.kdeplot(mt_rat, color='k', ax=ax[i_s],
                    label='Rats')
        sns.kdeplot(mt_model, color='r', ax=ax[i_s],
                    label='Model')
        # sns.kdeplot(mt_com_sim, color=COLOR_COM, ax=ax[i_s],
        #             label='Model Rev.', linestyle='--')
        # sns.kdeplot(mt_nocom_sim, color=COLOR_NO_COM, ax=ax[i_s],
        #             label='Model No-Rev.', linestyle='--')
        if i_s >= 12:
            ax[i_s].set_xlabel('MT (ms)')
        else:
            ax[i_s].set_xticks([])
        if i_s % 3 != 0:
            ax[i_s].set_ylabel('')
            ax[i_s].set_yticks([])
        ax[i_s].set_title(subj)
        ax[i_s].set_xlim(-5, 725)
        ax[i_s].set_ylim(0, 0.0085)
    ax[0].legend()
    fig.suptitle(title)


def plot_model_density(df_sim, sv_folder=SV_FOLDER, df=None, offset=0,
                       plot_data_trajs=False, n_trajs_plot=150,
                       pixel_precision=1, cmap='Reds', max_ms=400):
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
        In case of plotting the trajectories, how many. The default is 150.
    pixel_precision : float, optional
        Pixel precision for the density (the smaller the cleaner the plot).
        The default is 5.
    cmap : str, optional
        Colormap. The default is 'Reds'.

    Returns
    -------
    None.

    """
    n_steps = int(max_ms/5)
    fig2, ax2 = plt.subplots(nrows=3, ncols=3, figsize=(10, 7))
    np.random.seed(seed=5)  # set seed
    fig2.tight_layout()
    ax2 = ax2.flatten()
    coh = df_sim.coh2.values
    zt = np.round(df_sim.norm_allpriors.values, 1)
    coh_vals = [-1, 0, 1]
    zt_vals = [-np.max(np.abs(zt)), -np.max(np.abs(zt))*0.4,
               -0.05, 0.05,
               np.max(np.abs(zt))*0.4, np.max(np.abs(zt))]
    i = 0
    ztlabs = [-1, 0, 1]
    gkde = scipy.stats.gaussian_kde  # we define gkde that will generate the kde
    if plot_data_trajs:
        bins = np.array([-1.1, 1.1])  # for data plotting
        bintype = 'edges'
        trajectory = 'trajectory_y'
        df['choice_x_prior'] = (df.R_response*2-1) * df.norm_allpriors
    for ie, ev in enumerate(coh_vals):
        for ip, pr in enumerate(zt_vals):
            ip2 = 2*ip
            if ip == 3:
                break
            index = (zt >= zt_vals[ip2]) & (zt < zt_vals[ip2+1]) & (coh == ev)  # index of filtered
            max_len = max([len(t) for t in df_sim.traj[index].values])
            mat_fin = np.empty((sum(index), max_len+offset))
            mat_fin[:] = np.nan
            trajs = df_sim.traj[index].values
            for j in range(sum(index)):
                mat_fin[j, :len(trajs[j])] = trajs[j]  # mat_fin contains trajectories by rows
                mat_fin[j, len(trajs[j]):-1] = trajs[j][-1]  # set the last value (-75 or 75) until the end
            values = np.arange(-80, 81, pixel_precision)
            mat_final_density = np.empty((len(values), n_steps))  # matrix that will contain density by columns
            mat_final_density[:] = np.nan
            for j in range(2, n_steps):
                yvalues = np.nanmean(mat_fin[:, j*5:(j+1)*5], axis=1)  # we get the trajectory values
                kernel_1 = gkde(yvalues)  # we create the kernel using gkde
                vals_density = kernel_1(values)  # we evaluate the values defined before
                mat_final_density[:, j] = vals_density / np.nansum(vals_density)  # we normalize the density
            ax2[i].imshow(np.flipud(mat_final_density), cmap=cmap, aspect='auto',
                          norm=LogNorm(vmin=0.001, vmax=0.6))  # plot the matrix
            ax2[i].set_xlim(0, n_steps+0.2)
            ax2[i].set_ylim(len(values), 0)
            if i == 2 or i == 5 or i == 8:
                ax1 = ax2[i].twinx()
                ax1.set_yticks([])
                ax1.set_ylabel('stimulus = {}'.format(ztlabs[int((i-2) // 3)]),
                               rotation=90, labelpad=5, fontsize=12)
            if i >= 6:
                ax2[i].set_xticks(np.arange(0, n_steps+1, 20), np.arange(0, n_steps+1, 20)*5)
            else:
                ax2[i].set_xticks([])
            if i % 3 == 0:
                ax2[i].set_yticks(np.arange(0, len(values), int(80/pixel_precision)),
                                  ['5.6', '0', '-5.6'])
                # np.arange(80, -81, -80)
            else:
                ax2[i].set_yticks([])
            if i % 3 == 0:
                ax2[i].set_ylabel('y position (cm)')
            if i >= 6:
                ax2[i].set_xlabel('Time (ms)')
            if plot_data_trajs:
                index = (zt >= zt_vals[ip2]) & (zt < zt_vals[ip2+1]) & (coh == ev)  # index of filtered
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
                mtime = df.loc[idx[0]].resp_len.values
                n_trajs = mat_0.shape[0]
                # we select the number of trajectories that we want
                index_trajs_plot = np.random.choice(np.arange(n_trajs), n_trajs_plot)
                for ind in index_trajs_plot:
                    traj = mat_0[ind, :]
                    traj = traj - np.nanmean(traj[500:700])
                    # we do some filtering
                    if sum(np.abs(traj[700:700+int(max_ms)]) > 80) > 1:
                        continue
                    if np.abs(traj[700]) > 5:
                        continue
                    if mtime[ind] > 0.4:
                        continue
                    if np.abs(traj[920]) < 10:
                        continue
                    ax2[i].plot(np.arange(0, n_steps, 0.2), (-traj[700:700+int(max_ms)]+80)/160*len(values),
                                color='blue', linewidth=0.3)
            i += 1
    ax2[0].set_title('prior = -1')
    ax2[1].set_title('prior = 0')
    ax2[2].set_title('prior = 1')
    fig2.savefig(sv_folder+'supp_fig_6.svg', dpi=400, bbox_inches='tight')
    fig2.savefig(sv_folder+'supp_fig_6.png', dpi=400, bbox_inches='tight')


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
        if mt > 2:
            mt = 2
        motor_time.append(mt*1e3)
    return motor_time


def plot_mt_humans_all(df_data):
    fig, ax = plt.subplots(ncols=5, nrows=4)
    ax = ax.flatten()
    subjects = df_data.subjid.unique()
    mt = get_human_mt(df_data)
    median_mt = np.median(mt)
    for i_s, sub in enumerate(subjects[:-1]):
        color = 'k'
        label='MT'
        rm_top_right_lines(ax[i_s])
        mt = get_human_mt(df_data.loc[df_data.subjid == sub])
        mean_acc = np.round(np.nanmean(df_data.loc[
                df_data.subjid == sub, 'hithistory']), 2)
        median_mt_sub = np.round(np.nanmedian(mt), 1)
        mean_mt_sub = np.round(np.nanmean(mt), 1)
        if mean_acc < 0.7:
            color = 'b'
            label = 'Acc. < 0.7'
        if median_mt_sub > 300:
            color = 'g'
            label = 'Median > 300'
        sns.kdeplot(mt, ax=ax[i_s], color=color, shade=True,
                    label=label)
        ax[i_s].set_xlim(50, 600)
        ax[i_s].set_title('Median = ' + str(median_mt_sub) +
                          ' ms\nMean = ' + str(mean_mt_sub) + ' ms\nAccuracy = ' +
                          str(mean_acc))
        ax[i_s].axvline(median_mt, color='r', label='Median all')
        if i_s >= 15:
            ax[i_s].set_xlabel('MT (ms)')
        if mean_acc < 0.7:
            ax[i_s].legend()
        if median_mt_sub > 300:
            ax[i_s].legend(loc='upper left')
    ax[0].legend()


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
    if user_id == 'sara':
        folder = 'C:\\Users\\Sara Fuentes\\OneDrive - Universitat de Barcelona\\Documentos\\EBM\\4t\\IDIBAPS\\80_20\\'+nm+'ms\\'
    subj = ['general_traj_all']
    steps = [None]
    # retrieve data
    df = ah.traj_analysis(data_folder=folder,
                          subjects=subj, steps=steps, name=nm,
                          sv_folder=sv_folder)
    return df


def simulate_model_humans(df_data, stim, load_params, params_to_explore=[]):
    choice = df_data.R_response.values*2-1
    hit = df_data.hithistory.values*2-1
    subjects = df_data.subjid.unique()
    subjid = df_data.subjid.values
    gt = (choice*hit+1)/2
    coh = df_data.avtrapz.values*5
    zt = df_data.norm_allpriors.values*3
    trial_index = df_data.origidx.values
    num_tr = len(trial_index)
    hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt =\
        run_simulation_different_subjs(
            stim=stim, zt=zt, coh=coh, gt=gt,
            trial_index=trial_index, num_tr=num_tr, human=True,
            subject_list=subjects, subjid=subjid, simulate=True,
            load_params=load_params, params_to_explore=params_to_explore)
    
    return hit_model, reaction_time, detected_com, resp_fin, com_model,\
        pro_vs_re, total_traj, x_val_at_updt



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


def simulate_random_rat(n_samples, sv_folder, stim, zt, coh, gt, trial_index, num_tr):
    subjid = np.array(['Virtual_rat_random_params' for _ in range(len(coh))])
    subjects = ['Virtual_rat_random_params']
    for n in range(n_samples):
        extra_label = 'virt_sim_' + str(n)
        parameter_string = sv_folder + 'virt_params/' + 'parameters_MNLE_BADS_prt_n50_' +\
            extra_label + '.npy'
        os.makedirs(os.path.dirname(parameter_string), exist_ok=True)
        if os.path.exists(parameter_string):
            pars = np.load(parameter_string)
            simulate = False
        else:
            pars = np.array([np.random.uniform(1e-2, .5),
                             np.random.uniform(1e-2, .15),
                             np.random.uniform(1e-1, 4),
                             np.random.uniform(1e-6, .3),
                             np.random.uniform(3, 6),
                             np.random.uniform(3, 6),
                             np.random.uniform(5, 16),
                             np.random.uniform(0.015, 0.06),
                             np.random.uniform(1e-6, 2e-5),
                             np.random.uniform(1.5, 3),
                             np.random.uniform(50, 200),
                             np.random.uniform(200, 400),
                             np.random.uniform(0.05, 0.15),
                             np.random.uniform(18, 22),
                             np.random.uniform(200, 300),
                             np.random.uniform(0.05, 0.2)])
            np.save(sv_folder + 'virt_params/' + 'parameters_MNLE_BADS_prt_n50_' +
                    extra_label + '.npy',
                    pars)
            simulate = True
        _, _, _, _, _,\
            _, _, _ =\
            run_simulation_different_subjs(
                stim=stim, zt=zt, coh=coh, gt=gt,
                trial_index=trial_index, num_tr=num_tr, human=False,
                subject_list=subjects, subjid=subjid, simulate=simulate,
                load_params=True, extra_label=extra_label)
    return


def plot_acc_express(df):
    df_filt = df.copy().loc[(df.sound_len >= 0) & (df.sound_len <= 50)]
    # cohs = df_filt.coh2.abs().unique().values
    hit = df_filt.groupby(df_filt.coh2.abs()).hithistory.mean()
    fig, ax = plt.subplots(1, figsize=(6, 5))
    rm_top_right_lines(ax)
    ax.plot(hit.index, hit, color='k', marker='o', markersize=8)
    ax.set_xlabel('Stimulus strength')
    ax.set_ylabel('Accuracy')
    ax.set_yticks([0.5, 0.75, 1])
    ax.set_xticks([0, 0.25, 0.5, 1])


def supp_silent(df):
    fig, ax = plt.subplots(ncols=3, figsize=(10, 4))
    ax = ax.flatten()
    subjects = df.subjid.unique()
    labels = ['RT (ms)', 'MT (ms)']
    for i_s, subj in enumerate(subjects):
        df_subj = df.copy().loc[(df.subjid == subj) & (df.special_trial == 2)]
        df_subj['resp_len'] = df_subj.resp_len*1e3
        # RT
        fix_breaks =\
            np.vstack(np.concatenate([df_subj.sound_len/1000,
                                      np.concatenate(df_subj.fb.values)-0.3]))
        sns.kdeplot(fix_breaks.T[0]*1e3, ax=ax[0], alpha=0.3, color='k')
        # MT
        mt = df_subj.resp_len.values
        sns.kdeplot(mt, ax=ax[1], alpha=0.3, color='k')
        # PCoM RT
    binned_curve(df.loc[(df.special_trial == 2)], 'sound_len', 'CoM_sugg',
                 bins=BINS_RT, xpos=np.diff(BINS_RT)[0], ax=ax[2],
                 errorbar_kw={'color': 'k'})
    for j in range(2):
        ax[j].set_xlabel(labels[j])
    ax[0].set_xlim(-105, 305)
    ax[1].set_xlim(45, 755)
    fig.tight_layout()


# def plot_rats_humans_model_mt(df, df_sim, pc_name, mt=True):
#     df_humans = get_human_data(user_id=pc_name, sv_folder=SV_FOLDER)
#     minimum_accuracy = 0.7  # 70%
#     max_median_mt = 400  # ms
#     df_humans = fig_6.acc_filt(df_humans, acc_min=minimum_accuracy, mt_max=max_median_mt)
#     mt_human = np.array(get_human_mt(df_humans))
#     df_humans['coh2'] = df_humans.avtrapz.values*5
#     df_humans['resp_len'] = mt_human
#     df_humans['choice_x_coh'] = np.round((df_humans.R_response*2-1) * df_humans.coh2, 2)
#     df_humans['choice_x_prior'] = (df_humans.R_response*2-1) * df_humans.norm_allpriors
#     df_humans['aftererror'] = False
#     df_humans['framerate'] = 200
#     df_humans['special_trial'] = 0
#     prior = df_humans.choice_x_prior.values
#     if mt:
#         fig, ax = plt.subplots(ncols=6, nrows=1, figsize=(16, 3))
#         ax[0].axvline(x=0, color='k', linestyle='--', linewidth=0.6)
#         ax[1].axvline(x=0, color='k', linestyle='--', linewidth=0.6)
#         fig.subplots_adjust(wspace=0.6, hspace=0.5, left=0.05, right=0.95, bottom=0.1)
#         ax = ax.flatten()
#         cohvals = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
#         # humans
#         times = df_humans.times.values
#         mov_time_list = []
#         colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["mediumblue","plum","firebrick"])
#         colormap = colormap(np.linspace(0, 1, len(cohvals)))
#         for i_ev, coh in enumerate(cohvals):
#             # humans
#             index = (df_humans.choice_x_coh == coh) &\
#                 (np.abs(prior) <= np.quantile(np.abs(prior), 0.25)) &\
#                     (df_humans.sound_len <= 300) & (df_humans.sound_len >= 0)
#             mts = np.array([float(t[-1]) for t in
#                                             times[index]
#                                             if t[-1] != ''])
#             mov_time = np.nanmean(mts)*1e3
#             err_traj = np.nanstd(mts) / np.sqrt(sum(index))*1e3
#             mov_time_list.append(mov_time)
#             ax[1].errorbar(coh, mov_time, err_traj, color=colormap[i_ev],
#                             marker='o')
#         ax[1].plot(cohvals, mov_time_list, ls='-', lw=0.5, color='k')
#         # prior
#         condition = 'choice_x_prior'
        
#         bins_zt, _, _, _, _ =\
#               get_bin_info(df=df_humans, condition=condition, prior_limit=1,
#                             after_correct_only=True, rt_lim=300, silent=False)
#         xvals_zt = (bins_zt[:-1] + bins_zt[1:]) / 2
#         colormap = pl.cm.copper_r(np.linspace(0., 1, len(bins_zt)-1))[::-1]
#         mov_time_list = []
#         for i_ev, zt in enumerate(bins_zt[:-1]):
#             index = (prior >= bins_zt[i_ev])*(prior < bins_zt[i_ev+1])
#             mov_time = np.nanmean(np.array(mt_human[index]))
#             err_traj = np.nanstd(np.array(mt_human[index])) / np.sqrt(sum(index))
#             ax[0].errorbar(xvals_zt[i_ev], mov_time, err_traj, color=colormap[i_ev],
#                             marker='o')
#             mov_time_list.append(mov_time)
#         ax[0].plot(xvals_zt, mov_time_list, ls='-', lw=0.5, color='k')
#         ax[0].set_xticks([-1, 0, 1])
#         # RATS
#         # MT VS PRIOR
#         df_mt = df.copy()
#         fig_1.plot_mt_vs_evidence(df=df_mt.loc[df_mt.special_trial == 2], ax=ax[2],
#                                   condition='choice_x_prior', prior_limit=1,
#                                   rt_lim=200)
#         del df_mt
#         # MT VS COH
#         df_mt = df.copy()
#         fig_1.plot_mt_vs_evidence(df=df_mt, ax=ax[3], prior_limit=0.1,  # 10% quantile
#                                   condition='choice_x_coh', rt_lim=50)
#         fig2, ax2 = plt.subplots(1)
#         ax_zts = np.array((ax2, ax2, ax[-2], ax2))
#         # MODEL
#         fig_5.traj_cond_coh_simul(df_sim=df_sim[df_sim.special_trial == 2], ax=ax_zts,
#                                   new_data=False, data_folder=DATA_FOLDER,
#                                   save_new_data=False,
#                                   median=True, prior=True, rt_lim=300, extra_label='')
#         ax_cohs = np.array((ax2, ax2, ax[-1], ax2))
#         fig_5.traj_cond_coh_simul(df_sim=df_sim, ax=ax_cohs, median=True, prior=False,
#                                   save_new_data=False,
#                                   new_data=False, data_folder=DATA_FOLDER,
#                                   prior_lim=np.quantile(df_sim.norm_allpriors.abs(), 0.1),
#                                   rt_lim=50, extra_label='')
#         for i_a, a in enumerate(ax):
#             rm_top_right_lines(a)
#             if i_a % 2 == 0:
#                 a.set_xlabel('Prior evidence\ntowards response')
#                 a.set_ylabel('Movement time (ms)')
#             else:
#                 a.set_xlabel('Stimulus evidence\ntowards response')
#                 a.set_ylabel('')
#             # a.set_ylim(180, 310)
#             # a.set_yticks([200, 250, 300])
        