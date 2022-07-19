# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

# ddm
<<<<<<< HEAD


def evidence(offset, Ve, fluc, bound, dt, t_motor, plot=False):
    trial_dur = 1  # trial duration (s)
    N = int(trial_dur/dt)  # number of timesteps
    dW = []
    W = []
    E = []
    dW[0] = offset + np.sqrt(dt)*(np.random.randn()*fluc+Ve)
    W[0] = dW[0]
    E[0] = W[0]
    factor = 1/N
    timesteps_motor = round(t_motor/(dt))
    first = False
    counter = 1
    com = False
    reaction_time = 300
    for i in range(1, N):
        dW[i] = np.sqrt(dt)*(np.random.randn()*fluc+Ve)
        W[i] = dW[i] + W[i-1]
        E[i] = W[i]
        if np.abs(E[i]) >= bound and not first:
            first = True
            stop_var = i
            if E[i] > 0:
                # print('initial: right')
                response_in = 1
            else:
                # print('initial: left')
                response_in = -1
            # if plot:
            #     plt.figure()
            #     plt.plot(np.linspace(0, stop_var, num=stop_var)*factor, E[0:i])
            #     plt.ylabel('Evidence Accumulation')
            #     plt.xlabel('Time (s)')
        if i == N-1:
            com = False
            return E, reaction_time, com
        if first:
            counter += 1
            if counter == timesteps_motor:
                # if plot:
                # plt.plot(np.linspace(stop_var-1, stop_var+timesteps_motor-1,
                #                   num=timesteps_motor)*factor,
                #       E[stop_var-1:stop_var+timesteps_motor-1])
                # plt.axhline(1, linestyle='--', color='k', linewidth=0.4)
                # plt.axhline(0, linestyle='--', color='red', linewidth=0.4)
                # plt.axhline(-1, linestyle='--', color='k', linewidth=0.4)
=======
>>>>>>> b294c5915a636a0e0147ff7c4c632f9524d84e1e


<<<<<<< HEAD

def trial_ev_vectorized(offset, Ve, fluc, bound, dt, t_motor, num_tr, plot=False):
    trial_dur = 1  # trial duration (s)
    N = int(trial_dur/dt)  # number of timesteps
    dW = np.sqrt(dt)*(np.random.randn(N, num_tr)*fluc+Ve)
    dW[0] += offset
    E = np.cumsum(dW, axis=0)
    com = False
    # reaction_time = 300
    first_ind = []
    resp_first = np.ones(E.shape[1])
    resp_fin = np.ones(E.shape[1])
    for i_c in range(E.shape[1]):
        if (np.abs(E[:, i_c]) >= bound).any():
            hit_bound = np.where(np.abs(E[:, i_c]) >= bound)[0][0]
            first_ind.append(hit_bound)
            resp_first[i_c] *= ((-1)**(E[hit_bound, i_c] < 0))
            indx_fin_ch = min(hit_bound+t_motor, E.shape[1])
            resp_fin[i_c] *= ((-1)**(E[indx_fin_ch, i_c] < 0))
        else:
            first_ind.append(E.shape[0])
            resp_first[i_c] *= resp_first[i_c]*((-1)**(E[-1, i_c] < 0))
            resp_fin[i_c] = resp_first[i_c]
    com = resp_first != resp_fin
    first_ind = np.array(first_ind).astype(int)
    return E, com, first_ind, resp_first, resp_fin

#


if __name__ == '__main__':
    plt.close('all')
    bound = 1
    fluc = 1
    bound = 1
    dt = 1e-3
    t_motor = 80*1e-3
    t_motor = int(round(t_motor/(dt)))
    num_tr = 100
    plot = False
    offset = np.random.randn(num_tr)*1e-2
    Ve = np.random.randn(num_tr)*1e-5
    E, com, first_ind, resp_first, resp_fin =\
        trial_ev_vectorized(offset=offset, Ve=Ve, fluc=fluc,
                            bound=bound, dt=dt, t_motor=t_motor,
                            num_tr=num_tr, plot=plot)
    f, ax = plt.subplots()
    indx_no_com = np.where(~com)[0][0]
    print(resp_first[indx_no_com])
    print(resp_fin[indx_no_com])
    ax.plot(E[:first_ind[indx_no_com]+t_motor, indx_no_com], 'r')
    ax.plot(E[:first_ind[indx_no_com]+1, indx_no_com], 'b')
    indx_com = np.where(com)[0][0]
    print(resp_first[indx_com])
    print(resp_fin[indx_com])
    ax.plot(E[:first_ind[indx_com]+t_motor, indx_com], '--r')
    ax.plot(E[:first_ind[indx_com]+1, indx_com], '--b')
=======
def plotting(com, E, second_ind, first_ind):
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
    if E[first_ind[indx_com[4]], indx_com[4]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[0].scatter(first_ind[indx_com[4]], E[first_ind[indx_com[4]],
                                            indx_com[4]],
                  color=col, linewidths=3, zorder=4)
    if E[second_ind[indx_com[4]], indx_com[4]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[0].scatter(second_ind[indx_com[4]],
                  E[second_ind[indx_com[4]], indx_com[4]],
                  color=col, linewidths=3, zorder=4)
    ax[0].plot(E[:first_ind[indx_com[4]]+t_motor+1, indx_com[4]], 'r')
    ax[0].plot(E[:first_ind[indx_com[4]]+1, indx_com[4]], 'b')
    ax[2].plot(A[:first_ind[indx_com[4]]+t_motor+1, indx_com[4]], 'r')
    ax[2].plot(A[:first_ind[indx_com[4]]+1, indx_com[4]], 'b')
    ax[1].plot(E[:first_ind[indx_no_com[12]]+t_motor+1, indx_no_com[12]], 'r',
               label='After hitting')
    ax[1].plot(E[:first_ind[indx_no_com[12]]+1, indx_no_com[12]], 'b',
               label='Before hitting')
    if E[first_ind[indx_no_com[12]], indx_no_com[12]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[1].scatter(first_ind[indx_no_com[12]], E[first_ind[indx_no_com[12]],
                                                indx_no_com[12]],
                  color=col, linewidths=3, zorder=4)
    if E[first_ind[indx_no_com[12]]+t_motor, indx_no_com[12]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[1].scatter(first_ind[indx_no_com[12]]+t_motor,
                  E[first_ind[indx_no_com[12]]+t_motor, indx_no_com[12]],
                  color=col, linewidths=3, zorder=4)
    ax[1].legend()
    ax[3].plot(A[:first_ind[indx_no_com[12]]+t_motor+1, indx_no_com[12]], 'r')
    ax[3].plot(A[:first_ind[indx_no_com[12]]+1, indx_no_com[12]], 'b')
    ax[4].plot(E[:first_ind[indx_com[1]]+t_motor+1, indx_com[1]], 'r')
    ax[4].plot(E[:first_ind[indx_com[1]]+1, indx_com[1]], 'b')
    if E[first_ind[indx_com[1]], indx_com[1]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[4].scatter(first_ind[indx_com[1]], E[first_ind[indx_com[1]], indx_com[1]],
                  color=col, linewidths=3, zorder=4)
    if E[second_ind[indx_com[1]], indx_com[1]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[4].scatter(second_ind[indx_com[1]],
                  E[second_ind[indx_com[1]], indx_com[1]],
                  color=col, linewidths=3, zorder=4)
    ax[6].plot(A[:first_ind[indx_com[1]]+t_motor+1, indx_com[1]], 'r')
    ax[6].plot(A[:first_ind[indx_com[1]]+1, indx_com[1]], 'b')
    ax[5].plot(E[:first_ind[indx_no_com[0]]+t_motor+1, indx_no_com[0]], 'r')
    ax[5].plot(E[:first_ind[indx_no_com[0]]+1, indx_no_com[0]], 'b')
    if E[first_ind[indx_no_com[0]], indx_no_com[0]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[5].scatter(first_ind[indx_no_com[0]], E[first_ind[indx_no_com[0]],
                                               indx_no_com[0]],
                  color=col, linewidths=3, zorder=4)
    if E[first_ind[indx_no_com[0]]+t_motor, indx_no_com[0]] < 0:
        col = 'green'
    else:
        col = 'purple'
    ax[5].scatter(first_ind[indx_no_com[0]]+t_motor,
                  E[first_ind[indx_no_com[0]]+t_motor, indx_no_com[0]],
                  color=col, linewidths=3, zorder=4)
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
        ax[i].axhline(y=0.5, color='purple', linewidth=1, linestyle='--')
        ax[i].axhline(y=-0.5, color='green', linewidth=1, linestyle='--')


def trial_ev_vectorized(zt, stim, p_w_zt, p_w_stim, p_e_noise, p_com_bound,
                        p_t_m, p_t_eff, p_t_a, num_tr, p_w_a, p_a_noise, plot=False):
    bound = 1
    bound_a = 1
    prior = zt*p_w_zt
    Ve = stim*p_w_stim
    Va = p_w_a
    trial_dur = 1  # trial duration (s)
    N = stim.shape[0]  # int(trial_dur/dt)  # number of timesteps
    dW = np.random.randn(N, num_tr)*p_e_noise+Ve
    dA = np.random.randn(N, num_tr)*p_a_noise+Va
    dW[:p_t_eff, :] = 0
    dA[:p_t_a, :] = 0
    dW[0, :] = prior  # +np.random.randn(p_t_eff, num_tr)*p_e_noise
    dA[0, :] = prior  # +np.random.randn(p_t_a, num_tr)*p_a_noise
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
        second_thought = hit_dec+p_t_eff
        indx_fin_ch = min(second_thought, E.shape[0]-1)
        post_dec_integration = E[hit_dec:indx_fin_ch, i_c]-com_bound_temp
        indx_com =\
            np.where(np.sign(E[hit_dec, i_c]) != np.sign(post_dec_integration))[0]
        indx_update_ch = E.shape[0]-1 if len(indx_com) == 0 else\
            (indx_com[0] + hit_dec)
        resp_fin[i_c] = resp_first[i_c] if len(indx_com) == 0 else -resp_first[i_c]
        second_ind.append(indx_update_ch)
    com = resp_first != resp_fin
    first_ind = np.array(first_ind).astype(int)
    return E, A, com, first_ind, second_ind, resp_first, resp_fin
>>>>>>> b294c5915a636a0e0147ff7c4c632f9524d84e1e

    ax.axhline(y=1, linestyle='--', color='k', linewidth=0.8)
    ax.axhline(y=-1, linestyle='--', color='k', linewidth=0.8)
    ax.axhline(y=0, linestyle='--', color='k', linewidth=0.4)

<<<<<<< HEAD
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
=======
if __name__ == '__main__':
    plt.close('all')
    bound = 1
    fluc = 2
    bound = 1
    dt = 1e-3
    t_motor = 40*1e-3
    t_motor = int(round(t_motor/(dt)))
    num_tr = 100
    plot = False
    zt = np.random.randn(num_tr)*1e-2
    p_t_eff = 40
    p_t_a = p_t_eff
    num_timesteps = 1000
    stim = np.random.randn(num_tr)*1e-3+np.random.randn(num_timesteps+p_t_eff
                                                        , num_tr)*1e-1
    p_t_m = 40
    p_w_zt = 0.2
    p_w_stim = 0.2
    p_e_noise = 0.2
    p_com_bound= 0.5
    Va = np.abs(np.random.randn(num_tr))*1e-1
    fluc_a = 0.5
    bound_a = 1
    p_w_a = 0.05
    p_a_noise = 0.05
    E, A, com, first_ind, second_ind, resp_first, resp_fin =\
        trial_ev_vectorized(zt=zt, stim=stim, p_w_zt=p_w_zt,
                            p_w_stim=p_w_stim, p_e_noise=p_e_noise,
                            p_com_bound=p_com_bound,
                            p_t_m=p_t_m, p_t_eff=p_t_eff, p_t_a=p_t_a,
                            num_tr=num_tr, p_w_a=p_w_a, p_a_noise=p_a_noise,
                            plot=False)
    plotting(E=E, com=com, second_ind=second_ind, first_ind=first_ind)
>>>>>>> b294c5915a636a0e0147ff7c4c632f9524d84e1e
