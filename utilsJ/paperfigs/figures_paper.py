# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:24:12 2022
@author: Alex Garcia-Duran
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
# from scipy import interpolate
# sys.path.append("/home/jordi/Repos/custom_utils/")  # Jordi
# sys.path.append("C:/Users/Alexandre/Documents/GitHub/")  # Alex
sys.path.append("C:/Users/agarcia/Documents/GitHub/custom_utils")  # Alex CRM
# sys.path.append("/home/garciaduran/custom_utils")  # Cluster Alex
from utilsJ.Models import extended_ddm_v2 as edd2
from utilsJ.Behavior.plotting import binned_curve, tachometric, psych_curve,\
    com_heatmap_paper_marginal_pcom_side
from utilsJ.paperfigs import fig1, fig3, fig2

# SV_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/figures_python/'  # Alex
# DATA_FOLDER = 'C:/Users/Alexandre/Desktop/CRM/Alex/paper/data/'  # Alex
# DATA_FOLDER = '/home/molano/ChangesOfMind/data/'  # Manuel
# SV_FOLDER = '/home/molano/Dropbox/project_Barna/' +\
#     'ChangesOfMind/figures/from_python/'  # Manuel
SV_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/'  # Alex CRM
DATA_FOLDER = 'C:/Users/agarcia/Desktop/CRM/Alex/paper/data/'  # Alex CRM
BINS_RT = np.linspace(1, 301, 21)
xpos_RT = int(np.diff(BINS_RT)[0])


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


def tachometric_data(coh, hit, sound_len, ax):
    rm_top_right_lines(ax)
    df_plot_data = pd.DataFrame({'avtrapz': coh, 'hithistory': hit,
                                 'sound_len': sound_len})
    tachometric(df_plot_data, ax=ax, fill_error=True)
    ax.set_xlabel('RT (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Data')
    ax.set_ylim(0, 1.1)
    ax.legend()
    return ax.get_position()


def reaction_time_histogram(sound_len, label, ax, bins=np.linspace(1, 301, 61)):
    rm_top_right_lines(ax)
    if label == 'Data':
        color = 'k'
    if label == 'Model':
        color = 'red'
    ax.hist(sound_len, bins=bins, alpha=0.3, density=True, linewidth=0.,
            histtype='stepfilled', label=label, color=color)
    ax.set_xlabel("RT (ms)")
    ax.set_ylabel('Density')
    ax.set_xlim(0, max(BINS_RT))


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


def fig_1(coh, hit, sound_len, decision, supt=''):
    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax = ax.flatten()
    psych_curve((decision+1)/2, coh, ret_ax=ax[0])
    ax[0].set_xlabel('Coherence')
    ax[0].set_ylabel('Probability of right')
    pos_tach_ax = tachometric_data(coh=coh, hit=hit, sound_len=sound_len, ax=ax[1])
    reaction_time_histogram(sound_len=sound_len, ax=ax[2])
    express_performance(hit=hit, coh=coh, sound_len=sound_len,
                        pos_tach_ax=pos_tach_ax, ax=ax[3])
    fig.suptitle('')


def fig_5(coh, hit, sound_len, decision, hit_model, sound_len_model,
          decision_model, com, com_model, com_model_detected):
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
    hit_model = hit_model[reaction_time >= 0]
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
    reaction_time_histogram(sound_len=sound_len, label='Data', ax=ax[0])
    reaction_time_histogram(sound_len=sound_len_model, label='Model', ax=ax[0])
    ax[0].legend()
    express_performance(hit=hit, coh=coh, sound_len=sound_len,
                        pos_tach_ax=pos_tach_ax, ax=ax[4], label='Data')
    express_performance(hit=hit_model, coh=coh[sound_len_model >= 0],
                        sound_len=sound_len_model[sound_len_model >= 0],
                        pos_tach_ax=pos_tach_ax_model, ax=ax[4], label='Model')
    df_plot = pd.DataFrame({'com': com[sound_len_model >= 0],
                            'sound_len': sound_len[sound_len_model >= 0],
                            'rt_model': sound_len_model[sound_len_model >= 0],
                            'com_model': com_model[sound_len_model >= 0],
                            'com_model_detected':
                                com_model_detected[sound_len_model >= 0]})
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
    df_data = pd.DataFrame({'avtrapz': coh, 'CoM_sugg': com,
                            'norm_allpriors': zt/max(abs(zt)),
                            'R_response': decision})
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
                                 com_model_detected[sound_len_model >= 0],
                             'norm_allpriors':
                                 zt/max(abs(zt))[sound_len_model >= 0],
                             'R_response': (decision_model+1)/2})
    com_heatmap_paper_marginal_pcom_side(df_model, side=0)
    com_heatmap_paper_marginal_pcom_side(df_model, side=1)


def run_model(stim, zt, coh, gt):
    num_tr = int(len(zt))
    data_augment_factor = 10
    MT_slope = 0.123
    MT_intercep = 254
    detect_CoMs_th = 5
    p_t_aff = 6
    p_t_eff = 5
    p_t_a = 10
    p_w_zt = 0.25
    p_w_stim = 0.05
    p_e_noise = 0.01
    p_com_bound = 0.
    p_w_a = 0.035
    p_a_noise = np.sqrt(5e-3)
    p_1st_readout = 120
    p_2nd_readout = 180
    stim = edd2.data_augmentation(stim=stim.T, daf=data_augment_factor)
    stim_res = 50/data_augment_factor
    compute_trajectories = True
    all_trajs = True
    conf = [p_w_zt, p_w_stim, p_e_noise, p_com_bound, p_t_aff,
            p_t_eff, p_t_a, p_w_a, p_a_noise, p_1st_readout,
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
    p_w_a = conf[7]+jitters[7]*np.random.rand()
    p_a_noise = conf[8]+jitters[8]*np.random.rand()
    p_1st_readout = conf[9]+jitters[9]*np.random.rand()
    p_2nd_readout = conf[10]+jitters[10]*np.random.rand()
    stim_temp =\
        np.concatenate((stim, np.zeros((int(p_t_aff+p_t_eff),
                                        stim.shape[1]))))
    # TODO: get in a dict
    E, A, com_model, first_ind, second_ind, resp_first, resp_fin,\
        pro_vs_re, matrix, total_traj, init_trajs, final_trajs,\
        frst_traj_motor_time, x_val_at_updt, xpos_plot, median_pcom,\
        rt_vals, rt_bins, tr_index =\
        edd2.trial_ev_vectorized(zt=zt, stim=stim_temp, coh=coh,
                                 MT_slope=MT_slope, MT_intercep=MT_intercep,
                                 p_w_zt=p_w_zt, p_w_stim=p_w_stim,
                                 p_e_noise=p_e_noise, p_com_bound=p_com_bound,
                                 p_t_aff=p_t_aff, p_t_eff=p_t_eff, p_t_a=p_t_a,
                                 num_tr=num_tr, p_w_a=p_w_a,
                                 p_a_noise=p_a_noise,
                                 p_1st_readout=p_1st_readout,
                                 p_2nd_readout=p_2nd_readout,
                                 compute_trajectories=compute_trajectories,
                                 stim_res=stim_res, all_trajs=all_trajs)
    hit_model = resp_fin == gt
    reaction_time = (first_ind[tr_index]+p_t_eff -
                     int(300/stim_res))*stim_res
    detected_com = np.abs(x_val_at_updt) > detect_CoMs_th
    return hit_model, reaction_time, detected_com, resp_fin, com_model


# ---MAIN
if __name__ == '__main__':
    plt.close('all')
    df = edd2.get_data_and_matrix(dfpath=DATA_FOLDER + 'LE43_',
                                  return_df=True, sv_folder=SV_FOLDER)
    # if we want to use data from all rats, we must use dani_clean.pkl
    f1 = False
    f2 = False
    f3 = False
    f5 = True

    # fig 1
    if f1:
        fig1.d(df, savpath=SV_FOLDER, average=True)  # psychometrics
        zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        hit = np.array(df['hithistory'])
        stim = np.array([stim for stim in df.res_sound])
        coh = np.array(df.coh2)
        com = df.CoM_sugg.values
        decision = np.array(df.R_response) * 2 - 1
        sound_len = np.array(df.sound_len)
        gt = np.array(df.rewside) * 2 - 1

        # tachometrics, rt distribution, express performance
        fig_1(coh, hit, sound_len, decision, supt='data')

    # fig 2
    if f2:
        fig3.trajs_cond_on_prior(df, savpath=SV_FOLDER)
        fig3.trajs_cond_on_coh(df, savpath=SV_FOLDER)
        fig3.trajs_splitting(df, savpath=SV_FOLDER)
        fig3.trajs_splitting_point(df, savpath=SV_FOLDER)

    # fig 3
    if f3:
        fig2.bcd(df)
        fig2.e(df, savepath=SV_FOLDER)
        fig2.f(df, savepath=SV_FOLDER)
        fig2.g(df, savepath=SV_FOLDER)

    # fig 5 (model)
    if f5:
        zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)
        hit = np.array(df['hithistory'])
        stim = np.array([stim for stim in df.res_sound])
        coh = np.array(df.coh2)
        com = df.CoM_sugg.values
        decision = np.array(df.R_response) * 2 - 1
        sound_len = np.array(df.sound_len)
        gt = np.array(df.rewside) * 2 - 1
        hit_model, reaction_time, com_model_detected, resp_fin, com_model =\
            run_model(stim=stim, zt=zt, coh=coh, gt=gt)
        fig_5(coh=coh, hit=hit, sound_len=sound_len, decision=decision,
              hit_model=hit_model, sound_len_model=reaction_time,
              decision_model=resp_fin, com=com, com_model=com_model,
              com_model_detected=com_model_detected)
        fig1.d(df, savpath=SV_FOLDER, average=True)  # psychometrics data
        df_1 = df.copy()
        df_1['R_response'] = (resp_fin + 1)/2
        fig1.d(df_1, savpath=SV_FOLDER, average=True)  # psychometrics model
