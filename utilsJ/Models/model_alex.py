# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

# ddm


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

                reaction_time = (stop_var + timesteps_motor)*factor
                if E[i] > 0:
                    # print('final: right')
                    response_fin = 1*response_in
                else:
                    # print('final: left')
                    response_fin = -1*response_in
                if response_fin == -1:
                    com = True
                return E, reaction_time, com


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
    num_tr = 1000
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

    ax.axhline(y=1, linestyle='--', color='k', linewidth=0.8)
    ax.axhline(y=-1, linestyle='--', color='k', linewidth=0.8)
    ax.axhline(y=0, linestyle='--', color='k', linewidth=0.4)

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
