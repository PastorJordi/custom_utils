# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

# ddm


def plotting(com, E):
    f, ax = plt.subplots(nrows=4, ncols=2)
    ax = ax.flatten()
    indx_no_com = np.where(~com)[0]
    # ax.plot(E[:, indx_no_com], 'k')
    # ax.plot(E[:first_ind[indx_no_com]+t_motor, indx_no_com], 'r')
    # ax.plot(E[:first_ind[indx_no_com]+1, indx_no_com], 'b')
    indx_com = np.where(com)[0]
    # for i_c in indx_com:
    # ax.plot(E[:, indx_com], '--k')
    ax[0].set_title('CoM')
    ax[1].set_title('No-CoM')
    ax[6].set_xlabel('Time (ms)')
    ax[7].set_xlabel('Time (ms)')
    ax[0].scatter(first_ind[indx_com[4]], E[first_ind[indx_com[4]],
                                            indx_com[4]],
                  color='green', linewidths=3, zorder=4)
    ax[0].scatter(first_ind[indx_com[4]]+t_motor,
                  E[first_ind[indx_com[4]]+t_motor, indx_com[4]],
                  color='purple', linewidths=3, zorder=4)
    ax[0].plot(E[:first_ind[indx_com[4]]+t_motor+1, indx_com[4]], 'r')
    ax[0].plot(E[:first_ind[indx_com[4]]+1, indx_com[4]], 'b')
    ax[2].plot(A[:first_ind[indx_com[4]]+t_motor+1, indx_com[4]], 'r')
    ax[2].plot(A[:first_ind[indx_com[4]]+1, indx_com[4]], 'b')
    ax[1].plot(E[:first_ind[indx_no_com[12]]+t_motor+1, indx_no_com[12]], 'r',
               label='After hitting')
    ax[1].plot(E[:first_ind[indx_no_com[12]]+1, indx_no_com[12]], 'b',
               label='Before hitting')
    ax[1].scatter(first_ind[indx_no_com[12]], E[first_ind[indx_no_com[12]],
                                                indx_no_com[12]],
                  color='green', linewidths=3, zorder=4)
    ax[1].scatter(first_ind[indx_no_com[12]]+t_motor,
                  E[first_ind[indx_no_com[12]]+t_motor, indx_no_com[12]],
                  color='green', linewidths=3, zorder=4)
    ax[1].legend()
    ax[3].plot(A[:first_ind[indx_no_com[12]]+t_motor+1, indx_no_com[12]], 'r')
    ax[3].plot(A[:first_ind[indx_no_com[12]]+1, indx_no_com[12]], 'b')
    ax[4].plot(E[:first_ind[indx_com[1]]+t_motor+1, indx_com[1]], 'r')
    ax[4].plot(E[:first_ind[indx_com[1]]+1, indx_com[1]], 'b')
    ax[4].scatter(first_ind[indx_com[1]], E[first_ind[indx_com[1]], indx_com[1]],
                  color='purple', linewidths=3, zorder=4)
    ax[4].scatter(first_ind[indx_com[1]]+t_motor,
                  E[first_ind[indx_com[1]]+t_motor, indx_com[1]],
                  color='green', linewidths=3, zorder=4)
    ax[6].plot(A[:first_ind[indx_com[1]]+t_motor+1, indx_com[1]], 'r')
    ax[6].plot(A[:first_ind[indx_com[1]]+1, indx_com[1]], 'b')
    ax[5].plot(E[:first_ind[indx_no_com[0]]+t_motor+1, indx_no_com[0]], 'r')
    ax[5].plot(E[:first_ind[indx_no_com[0]]+1, indx_no_com[0]], 'b')
    ax[5].scatter(first_ind[indx_no_com[0]], E[first_ind[indx_no_com[0]],
                                               indx_no_com[0]],
                  color='green', linewidths=3, zorder=4)
    ax[5].scatter(first_ind[indx_no_com[0]]+t_motor,
                  E[first_ind[indx_no_com[0]]+t_motor, indx_no_com[0]],
                  color='green', linewidths=3, zorder=4)
    ax[7].plot(A[:first_ind[indx_no_com[0]]+t_motor+1, indx_no_com[0]], 'r')
    ax[7].plot(A[:first_ind[indx_no_com[0]]+1, indx_no_com[0]], 'b')
    ax[0].set_ylabel('EA (Proactive)')
    ax[1].set_ylabel('EA (Proactive)')
    ax[2].set_ylabel('AI (Proactive)')
    ax[3].set_ylabel('AI (Proactive)')
    ax[4].set_ylabel('EA (Reactive)')
    ax[5].set_ylabel('EA (Reactive)')
    ax[6].set_ylabel('AI (Reactive)')
    ax[7].set_ylabel('AI (Reactive)')
    for i in range(ax.shape[0]):
        ax[i].axhline(y=1, color='purple', linewidth=2)
        ax[i].axhline(y=-1, color='green', linewidth=2)
        ax[i].axhline(y=0, linestyle='--', color='k', linewidth=0.7)


def trial_ev_vectorized(zt, stim, p_w_zt, p_w_stim, p_e_noise, p_com_bound,
                        p_t_m, p_t_e, p_t_a, num_tr, p_w_a, p_a_noise, plot=False):
    bound = 1
    bound_a = 1
    prior = zt*p_w_zt
    Ve = stim*p_w_stim
    Va = p_w_a
    trial_dur = 1  # trial duration (s)
    N = stim.shape[0]  # int(trial_dur/dt)  # number of timesteps
    dW = np.random.randn(N+p_t_e, num_tr)*p_e_noise+Ve
    dA = np.random.randn(N+p_t_a, num_tr)*p_a_noise+Va
    dW[:p_t_e+1, :] = 0
    dA[:p_t_a+1, :] = 0
    dW[0] += prior
    E = np.cumsum(dW, axis=0)
    A = np.cumsum(dA, axis=0)
    com = False
    # reaction_time = 300
    first_ind = []
    second_ind = []
    resp_first = np.ones(E.shape[1])
    resp_fin = np.ones(E.shape[1])
    for i_c in range(E.shape[1]):
        indx_hit_bound = np.abs(E[:, i_c]) >= bound
        indx_hit_action = np.abs(A[:, i_c]) >= bound_a
        hit_bound = E.shape[0]-1
        hit_action = E.shape[0]-1
        if (indx_hit_bound).any():
            hit_bound = np.where(indx_hit_bound)[0][0]
        if (indx_hit_action).any():
            hit_action = np.where(indx_hit_action)[0][0]
        hit_dec = min(hit_bound, hit_action)
        first_ind.append(hit_dec)
        # first response
        resp_first[i_c] *= (-1)**(E[hit_dec, i_c] < 0)
        com_bound_temp = (-resp_first[i_c])*p_com_bound
        # second response
        second_thought = hit_dec+p_t_e
        indx_fin_ch = min(second_thought, E.shape[0]-1)
        post_dec_integration = E[hit_bound:indx_fin_ch, i_c]-com_bound_temp
        indx_com =\
            np.where(np.sign(E[hit_dec, i_c]) != np.sign(post_dec_integration))[0]
        indx_fin_ch = E.shape[0]-1 if len(indx_com) == 0 else indx_com[0]
        second_ind.append(indx_fin_ch)
        resp_fin[i_c] *= (-1)**(E[indx_fin_ch, i_c] < com_bound_temp)
    com = resp_first != resp_fin
    first_ind = np.array(first_ind).astype(int)
    return E, A, com, first_ind, second_ind, resp_first, resp_fin





if __name__ == '__main__':
    plt.close('all')
    bound = 1
    fluc = 2
    bound = 1
    dt = 1e-3
    t_motor = 80*1e-3
    t_motor = int(round(t_motor/(dt)))
    num_tr = 100
    plot = False
    zt = np.random.randn(num_tr)*1e-2
    p_t_e = 40
    num_timesteps = 1000
    stim = np.random.randn(num_tr)*1e-3+np.random.randn(num_timesteps+p_t_e
                                                        , num_tr)
    p_w_zt = 0.1
    p_w_stim = 0.1
    p_e_noise = 0.1
    p_com_bound= 0.1
    Va = np.abs(np.random.randn(num_tr))*1e-1
    fluc_a = 0.5
    bound_a = 1
    E, A, com, first_ind, second_ind, resp_first, resp_fin =\
        trial_ev_vectorized(dt=dt, zt=zt, stim=stim, p_w_zt=p_w_zt,
                            p_w_stim=p_w_stim, p_e_noise=p_e_noise,
                            p_com_bound=p_com_bound,
                            p_t_m, p_t_e, p_t_a, num_tr, p_w_a, p_a_noise, plot=False)
    plotting(E=E, com=com)

    import sys
    sys.exit()
    flucs = np.linspace(0, 5, num=50)
    bounds = np.linspace(0, 1, num=50)
    com_mat = np.zeros((len(flucs), len(bounds)))
    rtmat = np.zeros((len(flucs), len(bounds)))
    for i in range(len(flucs)):
        for j in range(len(bounds)):
            com_list = []
            react_time = []
            for num in range(50):
                _, rt, com = evidence(
                    0, 0.2, flucs[i], bounds[j], 1e3, 80*1e-3)
                com_list.append(com)
                react_time.append(rt)
                com_mat[j, i] = np.mean(com_list)
                rtmat[j, i] = np.nanmean(react_time)
    np.save('CoM_state_space_fluc_drift_0off.npy', com_mat)
    plt.contourf(flucs, bounds, com_mat)
    plt.xlabel("Fluctuation, $\sigma_{E}$", fontsize=14)
    plt.ylabel('Bound, $\\theta_{E}$', fontsize=14)
    plt.colorbar()
    plt.title('pCoM', fontsize=16)
    plt.figure()
    plt.contourf(flucs, bounds, rtmat)
    plt.xlabel("Fluctuation, $\sigma_{E}$", fontsize=14)
    plt.ylabel('Bound, $\\theta_{E}$', fontsize=14)
    plt.colorbar()
    plt.title('RT(s)', fontsize=16)