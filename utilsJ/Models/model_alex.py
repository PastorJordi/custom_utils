# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:17:47 2022
@author: Alex Garcia-Duran
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# ddm
def evidence(offset, Ve, fluc, bound, N, t_motor,plot=False):
    N=int(N)
    T = 1
    dt = T/N
    dW = np.zeros((N))
    W = np.zeros((N))   
    E = np.zeros((N))
    dW[0] = offset + np.sqrt(dt)*(np.random.randn()*fluc+Ve)
    W[0] = dW[0]
    E[0] = W[0]
    factor = 1/N
    timesteps_motor = round(t_motor/(dt))
    first = False
    counter=1
    com = False
    reaction_time = 300
    for i in range(1,N):
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
            return E, reaction_time , com
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

# same but vectorized
# def evidence_vec(offset, Ve, fluc, bound, N, t_motor):
#     N=int(N)
#     T = 1
#     dt = T/N
#     dW = np.sqrt(dt)*(np.random.randn(N)*fluc+Ve)
#     dW[0] += offset
#     E = np.cumsum(dW)
#     factor = 1/N
#     timesteps_motor = round(t_motor/(dt))
#     first = False
#     counter=1
#     com = False
#     reaction_time = 300
#     first_ind = np.where(np.abs(E) >= bound)[0][0]
#     second_ind = first_ind + timesteps_motor
#     if E[first_ind] > 0:
#         resp_first = 1
#     else:
#         resp_first = -1
#     if E[second_ind] > 0:
#         resp_fin = 1*resp_first
#     else:
#         resp_fin = (-1)*resp_first
#     if resp_fin == -1:
#         com = True
#     return E[0:second_ind], com, first_ind

# E,com,first = evidence_vec(0.2, 0.005, 3, 1, 1e3, 80*1e-3)


flucs = np.linspace(0, 5, num=50)
bounds = np.linspace(0, 1, num=50)
com_mat = np.zeros((len(flucs), len(bounds)))
rtmat = np.zeros((len(flucs), len(bounds)))
for i in range(len(flucs)):
    for j in range(len(bounds)):
        com_list = []
        react_time = []
        for num in range(50):
            _, rt, com = evidence(0, 0.2 , flucs[i], bounds[j], 1e3, 80*1e-3)
            com_list.append(com)
            react_time.append(rt)
            com_mat[j,i] = np.mean(com_list)
            rtmat[j,i] = np.nanmean(react_time)
np.save('CoM_state_space_fluc_drift_0off.npy',com_mat)
plt.contourf(flucs,bounds,com_mat)
plt.xlabel("Fluctuation, $\sigma_{E}$", fontsize=14)
plt.ylabel('Bound, $\\theta_{E}$', fontsize=14)
plt.colorbar()
plt.title('pCoM', fontsize=16)
plt.figure()
plt.contourf(flucs,bounds,rtmat)
plt.xlabel("Fluctuation, $\sigma_{E}$", fontsize=14)
plt.ylabel('Bound, $\\theta_{E}$', fontsize=14)
plt.colorbar()
plt.title('RT(s)', fontsize=16)