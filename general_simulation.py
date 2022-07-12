# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:38:44 2022
@author: Alex Garcia-Duran
"""
from utilsJ.regularimports import *
import importlib
import pandas as pd
from utilsJ.Models import simul, traj
from utilsJ.Behavior import plotting

"""
Alex G-D: Since I am more familiar with Spyder I have created this general_simulation.py
to run the model/simulation, so I can debug easily
"""


for px in [5]:
    for extra_te in [.02]: # 20=too flat for subject 42
        params = {
            't_update': 20, #125 # 50 # 80  # 100
            'proact_deltaMT': 0.2,#0.35-thr*0.25, # decreased a bit due to synergy with confirm
            'reactMT_interc': 135, 
            'reactMT_slope':0.15,
            'com_gamma':250, # 175 #150 # 125 destroys matrix
            'glm2Ze_scaling': 0.03, # 0.25 / 0.05, 0.1 good results # as little as possible, else com is more likely in prior==0
            'x_e0_noise': .002,# this is huge!
            'naive_jerk':True, # wtf
            'confirm_thr':0.3,
            'proportional_confirm':True,
            'confirm_ae':True,
            "t_update_noise":0,
            "com_deltaMT":0.5,
            "jerk_lock_ms": 45
        }

        # actual simulation
        savpath = 'C:/Users/alexg/Desktop/CRM/Alex/group code/paperfigs/prova_alex/'
        try:
            del out
        except:
            pass
        df, out = simul.whole_simul(
            'LE42',
            savpath,
            dfpath='C:/Users/alexg/Desktop/CRM/Alex/paper/LE42_clean.pkl',
            #dfpath=df, # comment this if first run
            params=params,
            return_data=False, batches=20,
            vanishing_bounds=False, both_traj=False,trajMT_jerk_extension=0, com_height=px,
            sample_silent_only=False, silent_trials=False, return_matrices=False,
            mtnoise=0.5, drift_multiplier=np.array([1,1.5,2.5,2.5]),  extra_t_0_e=extra_te # since we use extra t_e, lets try first without increasing default drift
        )

        plt.show()
        
        df['norm_allpriors'] = df['allpriors']/df.allpriors.abs().max()
        out['norm_allpriors'] = out['allpriors']/out.allpriors.abs().max()

        for i, side in enumerate(['left', 'right']):
            for dat, lab in [[df, 'data'], [out, 'simul']]:
                f, _ = plotting.com_heatmap_paper_marginal_pcom_side(dat, side=i)
                f.suptitle(f'{lab} {side}')
                plt.show()
