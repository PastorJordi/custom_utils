# this will be used to simulate trajectories.
# import argparse
import numpy as np
import pandas as pd
from utilsJ.Models import alex_bayes_clean as ab
import os
import pickle
from utilsJ.regularimports import loadmat
import tqdm
import warnings
import matplotlib.pyplot as plt

# remember, silent factorlist sorted like this: [intercept!] + ['zidx', 'dW_trans', 'dW_lat']

# example paths
# testL='/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/silent_LE42_left_RT0.pkl'
# mtclfL = loadclf('/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/MTclf_LE42_left_RT0.pkl')

# writting a short class to make it easier to wrte in notebooks


def loadclf(path):
    with open(path, "rb") as f:
        clf = pickle.load(f)
    return clf


class def_traj:
    def __init__(self, subject, side):
        """rtbin from 0 to 5,
        if side==None it assumes there are no RTbins neither"""
        assert side in ["left", "right", None], "side needs to be a str (left or right)"
        self.subject = subject
        self.side = side
        self.B = None
        self.factor_mask = None

    def selectRT(self, RTbin):
        if self.side is None:
            abobject = ab.ab(
                loadpath=f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/silent_{self.subject}_full2.pkl"
            )
            self.clf = loadclf(
                f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/MTclf_{self.subject}_full.pkl"
            )
        else:
            abobject = ab.ab(
                loadpath=f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/silent_{self.subject}_{self.side}_RT{RTbin}.pkl"
            )
            self.clf = loadclf(
                f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/MTclf_{self.subject}_{self.side}_RT{RTbin}.pkl"
            )
        self.B = abobject.B
        self.factorlist = abobject.factorlist

    def loadRTs(self, RTbin):
        self.Bs = {}
        self.clfs = {}
        if self.side is None:
            abobject = ab.ab(
                loadpath=f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/silent_{self.subject}_full2.pkl"
            )
            self.B = abobject.B
            self.factor_mask = abobject.factor_mask
            warnings.warn(
                f"You must realign left choices with this mask={repr(self.factor_mask)} in factors={repr(self.factorlist)}"
            )
        else:
            for i in range(6):
                try:
                    abobject = ab.ab(
                        loadpath=f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/silent_{self.subject}_{self.side}_RT{i}.pkl"
                    )
                    self.Bs[i] = abobject.B
                    if not i:
                        self.factorlist = abobject.factorlist
                except:
                    warnings.warn(f"could not retrieve B for {self.subject} in RT {i}")
                try:
                    self.clfs[i] = loadclf(
                        f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/MTclf_{self.subject}_{self.side}_RT{i}.pkl"
                    )
                except:
                    warnings.warn(
                        f"could not retrieve MT clf for {self.subject} in RT {i}"
                    )

    def expected_mt(self, Fk, add_intercept=True):  # adapt so Fk can be a matrix
        # Fk each row = 1 trial.
        assert (
            Fk.ndim == 2
        ), "ensure Fk is 2 d, 0th dim (rows) = trial, 2nd (cols) = each factor"
        # if self.factor_msk is not None:
        #     Fk = Fk * self.factor_mask.reshape(1,-1)
        # this wont work because it is agnostic to final choice
        if add_intercept:
            Fk = np.insert(Fk, 0, 1, axis=1)
        self.Fk = Fk
        # clf.predict(Fk) would have been way simpler
        self.mt = np.c_[
            Fk[:, 0] * self.clf.intercept_, (Fk[:, 1:] * self.clf.coef_)
        ].sum(axis=1)

    def prior_traj(self, Fk=None, times=None, step=1):
        if Fk is None:
            Fk = self.Fk
        if times is None:
            times = self.mt
        priorlist = []
        # print(times)
        for i in range(times.size):

            mu = self.B @ Fk[i].reshape(-1, 1)
            tspace = np.arange(int(times[i] * 1000), step=step)
            # print(tspace)
            if tspace[-1] != int(times[i] * 1000):
                tspace = np.append(tspace, int(times[i] * 1000))
            N = ab.v_(tspace) @ np.linalg.inv(ab.get_Mt0te(tspace[0], tspace[-1]))
            priorlist += [(N @ mu).ravel()]

        return priorlist

    def return_mu(self, Fk=None):
        if Fk is None:
            Fk = self.Fk
        mu_list = []
        for i in range(Fk.shape[0]):
            mu_list += [self.B @ Fk[i].reshape(-1, 1)]

        return mu_list


def simul_psiam(
    df,
    psiam_params,
    t_c=1.3,
    StimOnset=0.3,
    dt=1e-4,
    seed=None,
    batches=30,
    batch_size=2000,
    priortoZt=None,
    silent_trials=False,
    sample_silent_only=False,
    x_e0_noise=0,
    confirm_thr=0,
    proportional_confirm=False,
    confirm_ae=False,
    drift_multiplier=1 
):
    """
    rewritting this so it is straightforward
    This is hell
    documenting as it gets harder and harder to read / rewritte later based on sorting of loaded params
    
    df : df to take entropy to simulate (prefiltered rows)
    t_c, # 1.3 time of truncation (quan truncar els RTs dels contaminants)
    c, # prob de que 1 trial sigui contaminant (1-p(PSIAM)
    b, # (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    d, # part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    v_u, # intercept del drift l'urgency
    a_u, # threshold de l'action initation
    t_0_u, # temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    v, #[constant * stim str, aka drift real] drift de l'evidence [] 
    # ~ constant multiplicativa (llavors es multiplica per cada stim str) si fas varies coherencies passa vector. 
    # ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    a_e, # threshold evidence (scalar) entre 0 i qualsevol bound
    z_e, # starting point evidence (de -1 a 1) ?
    t_0_e, # temporal onset evidence acumulator. usually ~  0.35s (300 fixation + 50)
    StimOnset, # when is the stimulus starting (0.3)
    v_trial : drift urgency x trial index (x1000 times bigger due numerical errors)
    priortoZt: if not none, scales prior to starting Zt (evidence integrator offset)
    confirm_thr: (inverse) fraction of Ze that needs to be surpassed in order to revert preplanned choice. E.g, with confirmthr==0.3 rat with zE 1 (right) will need to reach evidence
    value of at least -0.3 to choose left, if above -0.3 it will stick to preplanned choice (right)
    proportional_confirm: whether to make confirm threshold proportional to starting offset (scaled, right?)
    confirm_ae: whether to apply confirm threshold after errors
    drift_multiplier: multiplies drift given by lluis (psiam parameters). It can be an array (size=4) to multiply sstr. 0, 0.25, 0.5, 1
    # what if t_update gets delayed with trial index?
    """
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random

    if isinstance(psiam_params, str):
        psiam_params = loadmat(psiam_params)["freepar_hat"][0]

    (
        c,
        v_u,
        a_u,
        t_0_u,
        *v,
        a_e,
        z_e,
        t_0_e,
        t_0_e_silent,
        v_trial,
        b,
        d,
        _,
        _,
        _,
    ) = psiam_params
    v = np.sort(v) * drift_multiplier
    # print('v_trial',v_trial)
    v_trial /= 1000
    # print('v_trial',v_trial)
    s = 0.2  # 1e-3 #0.1
    mu = (z_e + 1) / 2  # 0~1 space (assuming z_e was in -1~1)
    a = (1 - mu) * (mu / (s / 2)) ** 2 - mu  # ; % parameter a for the beta distribution
    B = a * (1 / mu - 1)  # ; % parameter b for the beta distribution
    # df["rtbin"] = pd.cut(
    #     df.sound_len, np.linspace(0, 150, 7), include_lowest=True, labels=False
    # )
    ntrials = df.shape[0]
    # for i, n in enumerate([
    #    'c', 'v_u', 'a_u', 't_0_u', 'v1', 'v2', 'v3', 'v4', 'a_e', 'z_e', 't_0_e', 't_0_e_silent', 'v_trial', 'b', 'd'
    # ]):
    #    print(n, psiam_params[i])

    assert ntrials, "df has no rows"
    # simulate independently per session! # if it works create a worker later to paralelize
    # as it is done in compipe
    outdf = pd.DataFrame([])  # empty
    for i in range(batches):
        if not sample_silent_only:
            dat = df.loc[(df.special_trial == 0) & (df.origidx != 1)].sample(
                n=batch_size, random_state=seed + i
            )
        else:
            dat = df.loc[(df.special_trial == 2) & (df.origidx != 1)].sample(
                n=batch_size, random_state=seed + i
            )
        n = len(dat)  # we will simulate all but first!
        u_drift = v_u + v_trial * dat.origidx.values  # urgency drift in each trial
        u_mat = rng.normal(
            size=(  # fill matrix with unitary drift
                u_drift.size,
                int((t_c - t_0_u) / dt) + 1,
            )
        ) * np.sqrt(dt) + (u_drift * dt).reshape(-1, 1)
        u_mat = u_mat.astype(np.float32).cumsum(axis=1)  # cumsum in each trial

        categ_vec = dat.rewside.values * 2 - 1
        if priortoZt is None: #warning. TODO: doublecheck this *a_e is not in conflict with the one below (aka multiplyed twice#
            x_e0 = (
                rng.beta(a, B, size=n) * 2 - 1
            ) * a_e  # % initialize Decision Variable with std so that performance is around 60%
            # plt.hist(x_e0)
            # plt.title('x_e0')
            # plt.show()
            usefulprior = (
                np.nansum(dat[["dW_trans", "dW_lat"]].values, axis=1) * categ_vec
            ) > 0
            x_e0[usefulprior] = x_e0[usefulprior] * categ_vec[usefulprior]
            x_e0[~usefulprior] = x_e0[~usefulprior] * categ_vec[~usefulprior] * -1
            x_e0[dat.aftererror.values == 1] = 0  # aftererror set it to 0
        else:
            x_e0 = np.nansum(dat[["dW_trans", "dW_lat"]].values, axis=1) * priortoZt
        
        x_e0_prenoise = x_e0.copy()

        if x_e0_noise: # here it is scaled with priortoZt
            # add gaussian # up to .3
            #x_e0 += rng.normal(scale=x_e0_noise, size=x_e0.size)
            # replace it as beta? up to .015
            x_e0 = (x_e0+1)/2 # 0-1 space for np.random.beta
            alpha = ((1 - x_e0) / x_e0_noise - 1 / x_e0) * x_e0 ** 2
            beta = alpha * (1 / x_e0 - 1)

            x_e0 = rng.beta(alpha, beta)*2-1 # draw beta distr and put it back to -1 ~ 1

        if priortoZt is not None:
            x_e0 *= a_e  # scale it to bound height

        # doublecheck so it is not wide enough and there's no reactive peak (RT)

        # calc e_drift according to stimulus str/coherence
        e_drift = (
            v[
                dat.coh2.abs()
                .map({1.0: 3, 0.5: 2, 0.4: 2, 0.25: 1, 0.2: 1, 0.0: 0})
                .values
            ]
            * categ_vec 
        )
        if silent_trials:  # perhaps in the future, add some tiny noise
            e_mat = np.zeros((n, int(1 / dt)))
        else:
            e_mat = rng.normal(
                size=(n, int(1 / dt))  # fill matrix with unitary drift
            ) * np.sqrt(dt) + (e_drift * dt).reshape(-1, 1)
        e_mat[:, 0] = x_e0  # removed +=
        e_mat = e_mat.astype(np.float32).cumsum(axis=1)

        evmask = np.abs(e_mat) > a_e
        bu = np.argmax(u_mat > a_u, axis=1)  # when is reached urgency bound
        be = np.argmax(evmask, axis=1)  # same but evidence
        be[np.where(np.all(evmask == False, axis=1))[0]] = (
            evmask.shape[1] - 1
        )  # replace those 0 for max
        # value where evidence bound was not hit!
        u_time = t_0_u + bu * dt  # add offsets to compare
        e_time = t_0_e + be * dt
        reactive_i = np.where(e_time < u_time)[0]
        ebound = np.zeros(n)
        ebound[reactive_i] = 1
        proactive_i = np.intersect1d(
            np.where(e_time > u_time)[0],
            np.where(u_time >= 0.3)[0],  # we will dismiss fb
        )
        non_fb = np.concatenate([proactive_i, reactive_i])
        RT = u_time.copy()
        RT[reactive_i] = e_time[reactive_i]
        # get x_e
        x_e = x_e0.copy()
        x_e[non_fb] = e_mat[non_fb, np.round((RT[non_fb] - 0.3) / dt).astype(int)]
        # old-to-do: account here for those that listened less than t_e. ! should be fine with line above!

        pred_choice = np.zeros(n)
        # pred_choice[non_fb] = np.ceil(x_e[non_fb]/1000)*2-1
        # pred_choice[reactive_i] = np.ceil(x_e[reactive_i]/1000)*2-1
        # set new thr for proactive
        # x_e0 is in a_e scale already and has noise (if enabled in params)
        # actually reactive will always be beyond, calc them altogether
        conf_bias_x_e = np.zeros(n)

        if (
            proportional_confirm
        ):  # confirm thr is proportional to -Ze ; screws com matrix
            newconfirm_thr = x_e0_prenoise * -(
                confirm_thr
            )  # x_e0 os a;ready in a_e scale
            # pred_choice[non_fb] = np.ceil((x_e[non_fb]-newconfirm_thr[non_fb])/1000)*2-1
        else:  # confirm thr it is a fixed value happening only after correct!
            newconfirm_thr = np.repeat(-confirm_thr, x_e0.size)  # create rep vec
            # ae_mask = dat.aftererror.values==1 # get a mask to turn aftererror to 0
            # newconfirm_thr[ae_mask] =  0 # set them to 0 remove this because it was increasing pcom to 30%
            newconfirm_thr[
                np.nansum(dat[["dW_trans", "dW_lat"]].values, axis=1) < 0
            ] *= -1  # invert it where it happenned to have left bias
            newconfirm_thr *= a_e  # scale it to a_e

        if not confirm_ae:  # beware, aftererror from source data!(instead of simul)
            ae_mask = dat.aftererror.values == 1  # get a mask to turn aftererror to 0
            newconfirm_thr[
                ae_mask
            ] = 0  # set them to 0 remove this because it was increasing pcom to 30%

        conf_bias_x_e[non_fb] = (
            x_e[non_fb] - newconfirm_thr[non_fb]
        )  # we will use it later to calc DeltaMT
        pred_choice[non_fb] = (conf_bias_x_e[non_fb] > 0) * 2 - 1

        newdf = pd.DataFrame(
            {
                "zidx": dat.zidx.values[non_fb],
                "dW_trans": dat.dW_trans.values[non_fb],
                "dW_lat": dat.dW_lat.values[non_fb],
                "dW_fixedbias": dat.dW_fixedbias.values[non_fb],
                "origidx": dat.origidx.values[non_fb],
                "rewside": dat.rewside.values[non_fb],
                "coh2": dat.coh2.values[non_fb],
                "sound_len": 1000 * (RT[non_fb] - 0.3),
                "subjid": dat.subjid.values[non_fb],
                "reactive": ebound[non_fb],
                "x_e0": x_e0[non_fb],
                "x_e": x_e[non_fb],
                "R_response": (pred_choice[non_fb] + 1) / 2,
                "hithistory": (pred_choice == categ_vec)[non_fb],
                "u_time": u_time[non_fb] - 0.3,
                "e_time": e_time[non_fb] - 0.3,
                "avtrapz": dat.avtrapz.values[non_fb],
                "delta_ev": conf_bias_x_e[non_fb],
                "aftererror": dat.aftererror.values[non_fb],
                "decision_threshold": newconfirm_thr[non_fb]
            }
        )

        outdf = pd.concat([outdf, newdf], ignore_index=True)

    return outdf


# def simul_traj(row): # whats the difference with function below?
#     """load params from simulated df
#     however this will just work for proactive responses, filter before applying
#     both sides mu are precalculated"""
#     try:
#         resp_len = int(row.resp_len*1000)

#         if row.reactive:
#             if row.R_response:
#                 curr_mu = row.B_R
#             else:
#                 curr_mu = row.B_L

#             t_arr = np.arange(resp_len)
#             M = ab.get_Mt0te(0,resp_len)
#             M_1 = np.linalg.inv(M)
#             vt = ab.v_(t_arr)
#             N = vt @ M_1
#             prior0 = (N @ curr_mu).ravel()
#             return prior0
#             #return np.zeros(prior0.size), prior0
#         else:
#             if row.prechoice: # R
#                 initial_expected_span = row.expectedMT_R
#                 curr_mu = row.B_R
#             else: # L
#                 initial_expected_span = row.expectedMT_L
#                 curr_mu = row.B_L

#             if row.R_response: # R
#                 final_mu = row.B_R.ravel()
#             else:
#                 final_mu = row.B_L.ravel()

#             t_arr = np.arange(int(initial_expected_span))
#             M = ab.get_Mt0te(0,initial_expected_span)
#             M_1 = np.linalg.inv(M)
#             vt = ab.v_(t_arr)
#             N = vt @ M_1
#             prior0 = (N @ curr_mu).ravel()
#             tup = int(row.t_update)

#             d1 = np.gradient(prior0, t_arr)
#             d2 = np.gradient(d1, t_arr)
#             Mf = ab.get_Mt0te(tup, resp_len)
#             Mf_1 = np.linalg.inv(Mf)
#             # init conditions are [prior0[tup], d1[tup], d2[tup]]
#             mu_prime = np.array([prior0[tup], d1[tup], d2[tup], *final_mu[3:]]).reshape(6,1) # final mu contains choice*
#             t_arr_prime = np.arange(tup, resp_len)
#             N_prime = ab.v_(t_arr_prime) @ Mf_1
#             updated_fragment = (N_prime @ mu_prime).ravel()
#             final_traj = np.concatenate(
#                 [prior0[:tup], updated_fragment]
#                 )
#             return final_traj
#             #return prior0,final_traj
#     except Exception as e:
#         raise e
#         return np.empty(0)
#         #return np.empty(0), np.empty(0)


def simul_traj_single(
    row,
    fixed_reactive_mu=None,
    return_both=False,
    silent_trials=False,
    trajMT_jerk_extension=0,
    jerk_lock_ms=0
):
    """load params from simulated df
    however this will just work for proactive responses, filter before applying
    both sides mu are precalculated"""
    if not np.isnan(row.t_update):
        tup = int(row.t_update)
    else:
        tup = np.nan

    # avoid t_update happening before jerk_lock finishes
    if tup<jerk_lock_ms:
        tup = jerk_lock_ms

    if fixed_reactive_mu is None:
        fixed_reactive_mu = np.array([0, 0, 0, 75, 0, 0]).reshape(-1, 1)
        
    try:
        resp_len = int(trajMT_jerk_extension + row.resp_len * 1000)
        RLresp = row.R_response * 2 - 1  # in -1 ~ 1 space
        if row.reactive:
            t_arr = np.arange(jerk_lock_ms,resp_len)
            M = ab.get_Mt0te(jerk_lock_ms, resp_len)
            M_1 = np.linalg.inv(M)
            vt = ab.v_(t_arr)
            N = vt @ M_1
            prior0 = (N @ (fixed_reactive_mu * RLresp)).ravel()
            prior0 = np.concatenate([[0]*jerk_lock_ms,prior0])
            if return_both:
                return np.zeros(prior0.size), prior0
            else:
                return prior0  # no update at all!

        elif silent_trials:
            initial_mu = row.mu_boundary
            t_arr = np.arange(jerk_lock_ms,resp_len)
            M = ab.get_Mt0te(jerk_lock_ms, resp_len)
            M_1 = np.linalg.inv(M)
            vt = ab.v_(t_arr)
            N = vt @ M_1
            prior0 = (N @ (initial_mu * RLresp)).ravel()
            prior0 = np.concatenate([[0]*jerk_lock_ms,prior0])
            if return_both:
                return np.zeros(prior0.size), prior0
            else:
                return prior0  # no update at all!
        else:
            initial_expected_span = trajMT_jerk_extension + row.expectedMT
            initial_mu = row.mu_boundary

            final_mu = initial_mu.copy().flatten()
            if row.R_response == 0:  # flip it
                final_mu *= -1

            t_arr = np.arange(jerk_lock_ms,int(initial_expected_span))
            t_arr_from_z = np.arange(int(initial_expected_span))
            M = ab.get_Mt0te(jerk_lock_ms, initial_expected_span)
            M_1 = np.linalg.inv(M)
            vt = ab.v_(t_arr)
            N = vt @ M_1
            prior0 = (N @ initial_mu).flatten()
            prior0 = np.concatenate([[0]*jerk_lock_ms,prior0])
            if row.prechoice == 0:  # it was left, flip it back
                prior0 *= -1
            # if t_update is too slow tup might happen after prior traj ended
            if (
                tup - 1 > prior0.size
            ):  # should I substract trajMT_jerk_extension from prior0.size?
                print(f'fuckuck here, reached the end before updating\ntup={tup}, priortraj={prior0.size}')
                if return_both:
                    return prior0, prior0
                else:
                    return prior0
            d1 = np.gradient(prior0, t_arr_from_z)
            d2 = np.gradient(d1, t_arr_from_z)
            Mf = ab.get_Mt0te(tup, resp_len)
            Mf_1 = np.linalg.inv(Mf)
            # init conditions are [prior0[tup], d1[tup], d2[tup]]
            mu_prime = np.array(
                [prior0[tup], d1[tup], d2[tup], final_mu[3], final_mu[4], final_mu[5]]
            ).reshape(
                6, 1
            )  # final mu contains choice*
            t_arr_prime = np.arange(tup, resp_len)
            N_prime = ab.v_(t_arr_prime) @ Mf_1
            updated_fragment = N_prime @ mu_prime  # .flatten()
            final_traj = np.concatenate([prior0[:tup], updated_fragment.flatten()])
            if return_both:
                return prior0, final_traj.flatten()
            else:
                return final_traj.flatten()
    except Exception as e:
        print(e)
        print('expectedMT', row.expectedMT)
        print('resp_len',row.resp_len)
        print('tup index', tup)
        #print('prior0',prior0.shape)
        
        
        raise e
        if return_both:
            return np.empty(0), np.empty(0)
        else:
            return np.empty(0)


# then call it with parametergrid: https://stackoverflow.com/questions/13370570/elegant-grid-search-in-python-numpy
def report_simul(df, sdf, simulparams):
    """create a minimal figure to generate reports for to do a gridsearch"""
    pass


# fix psiam first!

# params
# motor_delay = np.nan
# t_0_e = np.nan # fit
# Tu =  RT + t_0_e+ motor_delay # update time

# psiam params/fit
# sensory delay should be here, right?

# load psiam params

# load B for base traj

# decr in  MT depends linearly with accumulated evidence.
