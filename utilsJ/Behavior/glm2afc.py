#!/usr/bin/env python
# coding: utf-8


"""
## PERHAPS SHOULD PORT IT AS A CLASS SO there's no need to pass noenv, dual etc- always

This script was being copied way too many times. Adding it to a general and united framework
[+BLACKED VERSION+]

several functions to get glm weights etc.
available fn: 
 - get_stim_trapz
 - preprocess
 - check_colin
 - exec_glm
 - plot_...
 - get_module_weight"""
#  ### create some functions to be imported elsewhere

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings 


# In[4]:

# create funct for this crp
def getmodel_cols(cols='all', lateralized=False, noenv=False):
    """returns list
    all: all model cols
    ac: after correct
    ae: after error"""
    if cols not in ['all', 'ae', 'ac']:
        raise ValueError('cols valueshould be in [all, ac, ae]')
    if lateralized and noenv:
        raise NotImplementedError('cannot use lateralized and noenv atm')
    model_cols = [
        "L+1",
        "L-1",
        "L+2",
        "L-2",
        "L+3",
        "L-3",
        "L+4",
        "L-4",
        "L+5",
        "L-5",
        "L+6-10",
        "L-6-10",
        "T++1",
        "T+-1",
        "T-+1",
        "T--1",
        "T++2",
        "T+-2",
        "T-+2",
        "T--2",
        "T++3",
        "T+-3",
        "T-+3",
        "T--3",
        "T++4",
        "T+-4",
        "T-+4",
        "T--4",
        "T++5",
        "T+-5",
        "T-+5",
        "T--5",
        "T++6-10",
        "T+-6-10",
        "T-+6-10",
        "T--6-10",
        "intercept",
    ]
    if lateralized:
        model_cols = [f'SR{x}' for x in range(1,9)] \
                    + [f'SL{x}' for x in range(1,9)] \
                    + [f'afterefR{x}' for x in range(1,11)] \
                    + [f'afterefL{x}' for x in range(1,11)] \
                    + model_cols
    else:
        if noenv:
            model_cols = ['S'] + [f'aftereff{x}' for x in range(1,11)] + model_cols
        else:
            model_cols = [f'S{x}' for x in range(1,9)] + [f'aftereff{x}' for x in range(1,11)] + model_cols

    if cols=='all':
        return model_cols
    elif cols=='ac':
        afterc_cols = [x for x in model_cols if x not in ["L+2", "L-1", "T-+1", "T+-1", "T--1"]]
        return afterc_cols
    elif cols=='ae':
        aftere_cols = [x for x in model_cols if x not in ["L+1", "T++1", "T-+1", "T+-1", "T--1"]]
        return aftere_cols



def get_stim_trapz2(envL, envR, time, fail=True, samplingR=1000):
    """to begin use double envs (nparray size=20 each)
    time: ms played stim
    fail = TRUE = buggy sound envelopes (40Hz rather than 20) """
    if fail:
        modwave = np.abs(
            1 * np.sin(2 * np.pi * (20) * np.arange(0, 1, step=1 / samplingR) + np.pi)
        )
    else:
        # correct modwave
        raise NotImplementedError
    if np.isnan(time):
        time = 0
    tot_steps = int(0.001 * time * samplingR)
    envLpoints = np.abs(np.repeat(envL, samplingR / 20))
    envRpoints = np.abs(np.repeat(envR, samplingR / 20))
    reltotev = np.trapz(modwave[:tot_steps])
    relunit = np.trapz(modwave[: int(0.05 * samplingR)])  # 1 envelope
    L_int = np.trapz(modwave[:tot_steps] * envLpoints[:tot_steps])
    R_int = np.trapz(modwave[:tot_steps] * envRpoints[:tot_steps])
    return L_int / relunit, R_int / relunit, (R_int - L_int) / reltotev


def preprocess(in_data, lateralized=True, noenv=False):  # perhaps use trapz to calc intensity.
    """input df object, since it will calculate history*, it must contain consecutive trials
    returns preprocessed dataframe
    noenv = adapted to noenv- sessions
    laterme: wtf does newaftereff means? (doubled?) # now called lateralized, more mnemonic
    """
    model_cols = getmodel_cols(cols='all',lateralized=lateralized, noenv=noenv)
    if lateralized & noenv:
        raise NotImplementedError('cannot get sided aftereffects in noenv sessions because cohs are paired')
    df = in_data  # .copy(deep=True)
    df.loc[:, "rep_response"] *= 1  # originally is True/False col
    # expand stim [Sensory module]
    if not lateralized and not noenv:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    df["res_sound"].values.tolist(),
                    columns=["S" + str(x + 1) for x in range(20)],
                    index=df.index,
                ).loc[:, "S1":"S8"],
            ],
            axis=1,
        )
        if "soundrfail" in df.columns.tolist():
            df.loc[df.soundrfail, ["S" + str(x + 1) for x in range(8)]] = 0
    elif noenv:
        df['S']= df['res_sound']
        df.loc[df.soundrfail, 'S'] = 0
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    df["renv"].values.tolist(),
                    columns=["SR" + str(x + 1) for x in range(20)],
                    index=df.index,
                ).loc[:, "SR1":"SR8"],
            ],
            axis=1,
        )
        if "soundrfail" in df.columns.tolist():
            df.loc[df.soundrfail, ["SR" + str(x + 1) for x in range(8)]] = 0
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    df["lenv"].values.tolist(),
                    columns=["SL" + str(x + 1) for x in range(20)],
                    index=df.index,
                ).loc[:, "SL1":"SL8"]
                * -1,
            ],
            axis=1,
        )
        if "soundrfail" in df.columns.tolist():
            df.loc[df.soundrfail, ["SL" + str(x + 1) for x in range(8)]] = 0

    # aftereffect regressors
    # now using trapz to calc them. BTW, trapz is using theoretical time. Adapt at somepoint considering real delay
    # envmatrix = pd.DataFrame(df['res_sound'].values.tolist()).values
    # nanmat = np.tile(np.array([1,np.nan]), df.shape[0]).reshape(-1,2)
    # tvec = df.frames_listened.values # beware we have fucked up trials and i've lost 1 hour with this
    # tvec[tvec>20]=20
    # wholeframes = tvec.astype(int)
    # partialframes = np.zeros(tvec.size)
    # partialframes[(tvec%1)>0] = 1
    # partialframes[tvec==20] = 0
    # frames_listened_vec = (wholeframes + partialframes).astype(int)
    # frames_mask = np.zeros((df.shape[0], 2))
    # frames_mask[:,0], frames_mask[:,1] = frames_listened_vec, 20-frames_listened_vec
    # fuk = frames_mask.flatten().astype(int)
    # frame_mat_mask = np.repeat(nanmat.flatten(), fuk).reshape(-1,20)
    # df['aftereff1'] = np.nansum((frame_mat_mask * envmatrix), axis=1)
    if not noenv:
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    df[["lenv", "renv", "sound_len"]]
                    .apply(
                        lambda x: get_stim_trapz2(x[0], x[1], x[2], samplingR=1000), axis=1
                    )
                    .values.tolist(),
                    columns=["afterL", "afterR", "av_trapz"],
                    index=df.index,
                ),
            ],
            axis=1,
        )
    if not lateralized:
        if noenv:
            df["aftereff1"] = df.res_sound.shift(1)
            df.loc[df.origidx == 1, "aftereff1"] = np.nan
            for i in range(2, 11):
                df["aftereff" + str(i)] = df["aftereff" + str(i - 1)].shift(1)
                df.loc[df.origidx == 1, "aftereff" + str(i)] = np.nan
        else:
            df["aftereff1"] = df["afterR"] - df["afterL"]
            df["aftereff1"] = df.aftereff1.shift(1)
            df.loc[df.origidx == 1, "aftereff1"] = np.nan
            for i in range(2, 11):
                df["aftereff" + str(i)] = df["aftereff" + str(i - 1)].shift(1)
                df.loc[df.origidx == 1, "aftereff" + str(i)] = np.nan
    else:
        df["afterefR1"] = df.afterR.shift(1)
        df["afterefL1"] = df.afterL.shift(1)
        for i in range(2, 11):
            df["afterefR" + str(i)] = df["afterefR" + str(i - 1)].shift(1)
            df["afterefL" + str(i)] = df["afterefL" + str(i - 1)].shift(1)
            df.loc[df.origidx == 1, "afterefR" + str(i)] = np.nan
            df.loc[df.origidx == 1, "afterefL" + str(i)] = np.nan

    # for obsoletevariable in [envmatrix, nanmat, tvec, wholeframes, partialframes, frames_listened_vec,
    #                          frames_mask, frame_mat_mask]:
    #     del obsoletevariable

    # Lateral module
    df["L+1"] = np.nan  # np.nan considering invalids as errors
    df.loc[(df.R_response == 1) & (df.hithistory == 1), "L+1"] = 1
    df.loc[(df.R_response == 0) & (df.hithistory == 1), "L+1"] = -1
    df.loc[df.hithistory == 0, "L+1"] = 0
    df["L+1"] = df["L+1"].shift(1)
    df.loc[df.origidx == 1, "L+1"] = np.nan
    # L-
    df["L-1"] = np.nan
    df.loc[(df.R_response == 1) & (df.hithistory == 0), "L-1"] = 1
    df.loc[(df.R_response == 0) & (df.hithistory == 0), "L-1"] = -1
    df.loc[df.hithistory == 1, "L-1"] = 0
    df["L-1"] = df["L-1"].shift(1)
    df.loc[df.origidx == 1, "L-1"] = np.nan
    # shifts
    for i, item in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
        df["L+" + str(item)] = df["L+" + str(item - 1)].shift(1)
        df["L-" + str(item)] = df["L-" + str(item - 1)].shift(1)
        df.loc[df.origidx == 1, "L+" + str(item)] = np.nan
        df.loc[df.origidx == 1, "L-" + str(item)] = np.nan

    # add from 6 to 10, assign them and drop prev cols cols
    cols_lp = ["L+" + str(x) for x in range(6, 11)]
    cols_ln = ["L-" + str(x) for x in range(6, 11)]

    df["L+6-10"] = np.nansum(df[cols_lp].values, axis=1)
    df["L-6-10"] = np.nansum(df[cols_ln].values, axis=1)
    df.drop(cols_lp + cols_ln, axis=1, inplace=True)
    df.loc[df.origidx <= 6, "L+6-10"] = np.nan
    df.loc[df.origidx <= 6, "L-6-10"] = np.nan

    # pre transition module
    df.loc[df.origidx == 1, "rep_response"] = np.nan
    df["rep_response_11"] = df.rep_response
    df.loc[df.rep_response == 0, "rep_response_11"] = -1
    df.rep_response_11.fillna(value=0, inplace=True)
    df.loc[df.origidx == 1, "aftererror"] = np.nan

    # transition module
    df["T++1"] = np.nan  # np.nan #
    df.loc[(df.aftererror == 0) & (df.hithistory == 1), "T++1"] = df.loc[
        (df.aftererror == 0) & (df.hithistory == 1), "rep_response_11"
    ]
    df.loc[(df.aftererror == 1) | (df.hithistory == 0), "T++1"] = 0
    df["T++1"] = df["T++1"].shift(1)

    df["T+-1"] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hithistory == 0), "T+-1"] = df.loc[
        (df.aftererror == 0) & (df.hithistory == 0), "rep_response_11"
    ]
    df.loc[(df.aftererror == 1) | (df.hithistory == 1), "T+-1"] = 0
    df["T+-1"] = df["T+-1"].shift(1)

    df["T-+1"] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hithistory == 1), "T-+1"] = df.loc[
        (df.aftererror == 1) & (df.hithistory == 1), "rep_response_11"
    ]
    df.loc[(df.aftererror == 0) | (df.hithistory == 0), "T-+1"] = 0
    df["T-+1"] = df["T-+1"].shift(1)

    df["T--1"] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hithistory == 0), "T--1"] = df.loc[
        (df.aftererror == 1) & (df.hithistory == 0), "rep_response_11"
    ]
    df.loc[(df.aftererror == 0) | (df.hithistory == 1), "T--1"] = 0
    df["T--1"] = df["T--1"].shift(1)

    # shifts now
    for i, item in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
        df["T++" + str(item)] = df["T++" + str(item - 1)].shift(1)
        df["T+-" + str(item)] = df["T+-" + str(item - 1)].shift(1)
        df["T-+" + str(item)] = df["T-+" + str(item - 1)].shift(1)
        df["T--" + str(item)] = df["T--" + str(item - 1)].shift(1)
        df.loc[df.origidx == 1, "T++" + str(item)] = np.nan
        df.loc[df.origidx == 1, "T+-" + str(item)] = np.nan
        df.loc[df.origidx == 1, "T-+" + str(item)] = np.nan
        df.loc[df.origidx == 1, "T--" + str(item)] = np.nan

    cols_tpp = ["T++" + str(x) for x in range(6, 11)]
    # cols_tpp = [x for x in df.columns if x.startswith('T++')]
    cols_tpn = ["T+-" + str(x) for x in range(6, 11)]
    # cols_tpn = [x for x in df.columns if x.startswith('T+-')]
    cols_tnp = ["T-+" + str(x) for x in range(6, 11)]
    # cols_tnp = [x for x in df.columns if x.startswith('T-+')]
    cols_tnn = ["T--" + str(x) for x in range(6, 11)]
    # cols_tnn = [x for x in df.columns if x.startswith('T--')]

    df["T++6-10"] = np.nansum(df[cols_tpp].values, axis=1)
    df["T+-6-10"] = np.nansum(df[cols_tpn].values, axis=1)
    df["T-+6-10"] = np.nansum(df[cols_tnp].values, axis=1)
    df["T--6-10"] = np.nansum(df[cols_tnn].values, axis=1)

    df.drop(cols_tpp + cols_tpn + cols_tnp + cols_tnn, axis=1, inplace=True)
    df.loc[df.origidx < 6, ["T++6-10", "T+-6-10", "T-+6-10", "T--6-10"]] = np.nan

    for col in [x for x in df.columns if x.startswith("T")]:  ## not working?
        df[col] = df[col] * (
            df.R_response.shift(1) * 2 - 1
        )  # {0 = Left; 1 = Right, nan=invalid}

    if not noenv:
        for i in range(8, 0, -1):
            if not lateralized:
                df.loc[df.frames_listened < (i - 1), "S" + str(i)] = 0
            else:
                df.loc[df.frames_listened < (i - 1), ["SR" + str(i), "SL" + str(i)]] = 0

    df["intercept"] = 1
    df.loc[:, model_cols].fillna(value=0, inplace=True)

    if "soundrfail" in df.columns.tolist() and not noenv:
        # just replace res_sound
        ent_series = df["res_sound"]
        soundrfailidx = df.loc[df.soundrfail == True, "res_sound"].index
        # df.loc[df.soundrfail==True, 'res_sound'] = np.zeros(df.soundrfail.sum(),20) #* df.soundrfail.sum()
        ent_series[soundrfailidx] = [np.zeros(20)] * soundrfailidx.size
        df["res_sound"] = ent_series

    return df  # resulting df with lateralized T+


def check_colin(df, lateralized=False ,dual=True, noenv=False):
    """plots matrix
    df is the preprocessed dframe,
    dual = aftercorrect/aftererror,
    noenv = single Stim val (=Coh)"""
    if dual:
        afterc_cols = getmodel_cols(cols='ac', lateralized=lateralized, noenv=noenv)
        aftere_cols = getmodel_cols(cols='ae', lateralized=lateralized, noenv=noenv)
        for j, (t, cols) in enumerate(
            zip(["after correct", "after error"], [afterc_cols, aftere_cols])
        ):
            fig, ax = plt.subplots(figsize=(16, 16))
            sns.heatmap(
                df.loc[df.aftererror == j, cols].fillna(value=0).corr(),
                vmin=-1,
                vmax=1,
                cmap="coolwarm",
                ax=ax,
            )
            ax.set_title(t)
            plt.show()
    else:
        model_cols = getmodel_cols(cols='all', lateralized=lateralized, noenv=noenv)

        fig, ax = plt.subplots(figsize=(16, 16))
        sns.heatmap(
            df.loc[:, model_cols].fillna(value=0).corr(),
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            ax=ax,
        )
        ax.set_title("single")
        plt.show()


def exec_glm(df, dual=True, split=False, lateralized=False, noenv=False, plot=True, savdir=''):
    """perhaps iwont implement the splitting but w/e idk atm
    futureme: wth is split
    # adapt this function so it can accept **kwargs (dual, lateralized, noenv :D)
    """
    warnings.filterwarnings("ignore")

    if lateralized and noenv:
        NotImplementedError('cannot lateralized AND noenv')
    afterc_cols = getmodel_cols(cols='ac', lateralized=lateralized, noenv=noenv)
    aftere_cols = getmodel_cols(cols='ae', lateralized=lateralized, noenv=noenv)
    
    if dual:
        X_df_ac, y_df_ac = (
            df.loc[
                (df.aftererror == 0) & (df["R_response"].notna()), afterc_cols
            ].fillna(value=0),
            df.loc[(df.aftererror == 0) & (df["R_response"].notna()), "R_response"],
        )
        X_df_ae, y_df_ae = (
            df.loc[
                (df.aftererror == 1) & (df["R_response"].notna()), aftere_cols
            ].fillna(value=0),
            df.loc[(df.aftererror == 1) & (df["R_response"].notna()), "R_response"],
        )

        Lreg_ac = LogisticRegression(
            C=1,
            fit_intercept=False,
            penalty="l2",
            solver="saga",
            random_state=123,
            max_iter=10000000,
            n_jobs=-1,
        )
        Lreg_ac.fit(X_df_ac.values, y_df_ac.values)
        Lreg_ae = LogisticRegression(
            C=1,
            fit_intercept=False,
            penalty="l2",
            solver="saga",
            random_state=123,
            max_iter=10000000,
            n_jobs=-1,
        )
        Lreg_ae.fit(X_df_ae.values, y_df_ae.values)
        sm_logit_ac = sm.Logit(y_df_ac.values, X_df_ac.values)
        result_ac = sm_logit_ac.fit(
            method="bfgs", maxiter=10 ** 8
        )  # start_params=Lreg_ac.coef_ # alpha=1.3 # start_params=Lreg_ac.coef_
        sm_logit_ae = sm.Logit(y_df_ae.values, X_df_ae.values)
        result_ae = sm_logit_ae.fit(
            method="bfgs", maxiter=10 ** 8
        )  # start_params=Lreg_ae.coef_ alpha=1.3 #start_params=Lreg_ae.coef_
        if plot:
            plot_sensory_dual(X_df_ac, X_df_ae, Lreg_ac, result_ac, Lreg_ae, result_ae, lateralized=lateralized, savpath=savdir+'sens.png')
            plot_lateral_dual(X_df_ac, X_df_ae, Lreg_ac, result_ac, Lreg_ae, result_ae, savpath=savdir+'lat.png')
            plot_transition_dual(X_df_ac, X_df_ae, Lreg_ac, result_ac, Lreg_ae, result_ae, savpath=savdir+'trans.png')
        LRresult_ac = pd.read_html(
            result_ac.summary(xname=X_df_ac.columns.tolist()).tables[1].as_html(),
            header=0,
            index_col=0,
        )[0]
        LRresult_ae = pd.read_html(
            result_ae.summary(xname=X_df_ae.columns.tolist()).tables[1].as_html(),
            header=0,
            index_col=0,
        )[0]

        df["proba"] = np.nan
        probac = Lreg_ac.predict_proba(X_df_ac)
        probae = Lreg_ae.predict_proba(X_df_ae)
        print(
            "pred vec",
            probac.shape,
            "being classes",
            Lreg_ac.classes_,
            "and returning [:,1] prob",
        )
        print(
            "frame shape",
            df.loc[(df.aftererror == 0) & (df["R_response"].notna()), "proba"].shape,
        )
        df.loc[(df.aftererror == 0) & (df["R_response"].notna()), "proba"] = probac[
            :, 1
        ]
        df.loc[(df.aftererror == 1) & (df["R_response"].notna()), "proba"] = probae[
            :, 1
        ]
        warnings.filterwarnings("default")
        return {
            "skl_ac": Lreg_ac,
            "sm_ac": result_ac,
            "mat_ac": LRresult_ac,
            "skl_ae": Lreg_ae,
            "sm_ae": result_ae,
            "mat_ae": LRresult_ae,
            "proba": df["proba"].values,
        }

    else:
        model_cols = getmodel_cols(cols='all', lateralized=lateralized, noenv=noenv)
        X_df, y_df = (
            df.loc[df["R_response"].notna(), model_cols].fillna(value=0),
            df.loc[df["R_response"].notna(), "R_response"],
        )
        Lreg = LogisticRegression(
            C=1,
            fit_intercept=False,
            penalty="l2",
            solver="saga",
            random_state=123,
            max_iter=10000000,
            n_jobs=-1,
        )
        Lreg.fit(X_df.values, y_df.values)
        model = sm.Logit(y_df.values, X_df.values)
        result = model.fit_regularized(
            start_params=Lreg.coef_, maxiter=10 ** 6, alpha=1
        )
        if plot:
            plot_sensory(X_df, Lreg, result, lateralized=lateralized, savpath=savdir+'sens.png')
            plot_lateral(X_df, Lreg, result, savpath=savdir+'lat.png')
            plot_transition(X_df, Lreg, result, savpath=savdir+'trans.png')
        LRresult = pd.read_html(
            result.summary(xname=X_df.columns.tolist()).tables[1].as_html(),
            header=0,
            index_col=0,
        )[0]
        df["proba"] = np.nan
        df.loc[df.R_response.notna(), 'proba'] = Lreg.predict_proba(X_df)[:,1]
        warnings.filterwarnings("default")
        return {"skl": Lreg, "sm": result, "mat": LRresult, 'proba': df['proba'].values} 


# In[5]:


# plotting section
# tune functions to plot sensory + aftereff; Lateral; Transition
def plot_sensory(targ_df, model1, model2, lateralized=False, savpath=''):
    """
    models should be fitted previously
    being model1 sklearn and model2 statsmodels
    targ_df is the dataframe, to get index values/colnames
    """
    LRresult = pd.read_html(
        model2.summary(xname=targ_df.columns.tolist()).tables[1].as_html(),
        header=0,
        index_col=0,
    )[0]
    interestcols = np.where(targ_df.columns.str.startswith("S"))[0]
    f, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))
    ax = ax.flatten()
    ax[0].plot(np.arange(interestcols.size), model1.coef_[0, interestcols], "-o", c="r")
    ax[0].errorbar(
        np.arange(interestcols.size),
        LRresult.loc[targ_df.columns[interestcols], "coef"],
        yerr=LRresult.loc[targ_df.columns[interestcols], "std err"],
        marker="o",
        c="b",
    )
    ax[0].set_xticks(np.arange(interestcols.size))
    ax[0].set_xticklabels(targ_df.columns[interestcols])
    ax[0].set_title("sensory")
    ax[0].set_ylabel("weight")
    ax[0].axhline(y=0, linestyle=":", c="k")
    # stars
    signifposition = np.arange(interestcols.size)[
        np.where(LRresult.loc[targ_df.columns[interestcols], "P>|z|"].values < 0.05)
    ]
    ax[0].scatter(
        signifposition,
        LRresult.loc[targ_df.columns[interestcols], "coef"].values[signifposition]
        + 0.4,
        marker="*",
        c="g",
    )

    interestcols = np.where(targ_df.columns.str.startswith("aftereff"))[0]
    ax[1].errorbar(
        np.arange(interestcols.size),
        LRresult.loc[targ_df.columns[interestcols], "coef"],
        yerr=LRresult.loc[targ_df.columns[interestcols], "std err"],
        marker="o",
        c="b",
    )
    ax[1].plot(np.arange(interestcols.size), model1.coef_[0, interestcols], "-o", c="r")

    signifposition = np.arange(interestcols.size)[
        np.where(LRresult.loc[targ_df.columns[interestcols], "P>|z|"].values < 0.05)
    ]
    ax[1].scatter(
        signifposition,
        LRresult.loc[targ_df.columns[interestcols], "coef"].values[signifposition]
        + 0.05,
        marker="*",
        c="g",
    )
    ax[1].set_title("aftereffect")
    ax[1].set_ylabel("weight")
    ax[1].axhline(y=0, linestyle=":", c="k")
    plt.savefig(savpath)
    plt.show()


def plot_lateral(targ_df, model1, model2, savpath=''):
    LRresult = pd.read_html(
        model2.summary(xname=targ_df.columns.tolist()).tables[1].as_html(),
        header=0,
        index_col=0,
    )[0]
    interestcols = np.where(targ_df.columns.str.startswith("L+"))[0]
    numcols = interestcols.size
    f, ax = plt.subplots(ncols=2, nrows=1, sharey=True, figsize=(16, 6))
    ax = ax.flatten()
    ax[0].plot(
        np.arange(numcols),
        model1.coef_[0, np.where(targ_df.columns.str.startswith("L+"))[0]],
        "-o",
        c="r",
    )
    ax[0].errorbar(
        np.arange(numcols),
        LRresult.loc[targ_df.columns[interestcols], "coef"],
        yerr=LRresult.loc[targ_df.columns[interestcols], "std err"],
        marker="o",
        c="b",
    )
    ax[0].set_ylabel("weight")
    ax[0].set_title("L+")
    signifposition = np.arange(numcols)[
        np.where(LRresult.loc[targ_df.columns[interestcols], "P>|z|"].values < 0.05)
    ]
    ax[0].scatter(
        signifposition,
        LRresult.loc[targ_df.columns[interestcols], "coef"].values[signifposition]
        + 0.05,
        marker="*",
        c="g",
    )
    ax[0].set_xticks(np.arange(numcols))
    ax[0].set_xticklabels(targ_df.columns[interestcols])
    ax[0].axhline(y=0, linestyle=":", c="k")

    interestcols = np.where(targ_df.columns.str.startswith("L-"))[0]
    numcols = interestcols.size
    ax[1].plot(
        np.arange(numcols),
        model1.coef_[0, np.where(targ_df.columns.str.startswith("L-"))[0]],
        "-o",
        c="r",
    )
    ax[1].errorbar(
        np.arange(numcols),
        LRresult.loc[targ_df.columns[interestcols], "coef"],
        yerr=LRresult.loc[targ_df.columns[interestcols], "std err"],
        marker="o",
        c="b",
    )

    ax[1].set_ylabel("weight")
    ax[1].set_title("L-")
    signifposition = np.arange(numcols)[
        np.where(LRresult.loc[targ_df.columns[interestcols], "P>|z|"].values < 0.05)
    ]
    ax[1].scatter(
        signifposition,
        LRresult.loc[targ_df.columns[interestcols], "coef"].values[signifposition]
        + 0.05,
        marker="*",
        c="g",
    )
    ax[1].set_xticks(np.arange(numcols))
    ax[1].set_xticklabels(targ_df.columns[interestcols])
    ax[1].axhline(y=0, linestyle=":", c="k")
    plt.savefig(savpath)
    plt.show()


def plot_transition(targ_df, model1, model2, savpath=''):
    LRresult = pd.read_html(
        model2.summary(xname=targ_df.columns.tolist()).tables[1].as_html(),
        header=0,
        index_col=0,
    )[0]
    f, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 9))
    ax = ax.flatten()
    for i, name in enumerate(["T++", "T+-", "T-+", "T--"]):
        interestcols = np.where(targ_df.columns.str.startswith(name))[0]
        numcols = interestcols.size
        ax[i].plot(
            np.arange(numcols),
            model1.coef_[0, np.where(targ_df.columns.str.startswith(name))[0]],
            "-o",
            c="r",
        )
        ax[i].errorbar(
            np.arange(numcols),
            LRresult.loc[targ_df.columns[interestcols], "coef"],
            yerr=LRresult.loc[targ_df.columns[interestcols], "std err"],
            marker="o",
            c="b",
        )

        ax[i].set_ylabel("weight")
        ax[i].set_title(name)
        signifposition = np.arange(numcols)[
            np.where(LRresult.loc[targ_df.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[i].scatter(
            signifposition,
            LRresult.loc[targ_df.columns[interestcols], "coef"].values[signifposition]
            + 0.25,
            marker="*",
            c="g",
            s=80,
        )

        ax[i].set_xticks(np.arange(numcols))
        ax[i].set_xticklabels(targ_df.columns[interestcols])
        ax[i].axhline(y=0, linestyle=":", c="k")
    plt.savefig(savpath)
    plt.show()


def plot_sensory_dual(targ_df1, targ_df2, model1a, model1b, model2a, model2b, lateralized=False, savpath=''):
    """
    models should be fitted previously
    being modelxa sklearn and modelxb statsmodels
    targ_df is the dataframe, to get index values/colnames
    # i assume targdf1 = aftercorrect, targdf2 = aftererror
    # being model1=aftercorrect
    """
    colors = sns.color_palette()
    LRresult1 = pd.read_html(
        model1b.summary(xname=targ_df1.columns.tolist()).tables[1].as_html(),
        header=0,
        index_col=0,
    )[0]
    if lateralized:
        interestcols = np.where(targ_df1.columns.str.startswith("SR"))[0]
    
        f, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))
        ax = ax.flatten()
        ax[0].plot(
            np.arange(interestcols.size),
            model1a.coef_[0, interestcols],
            "-o",
            c=colors[1],
            alpha=0.5,
        )
        ax[0].errorbar(
            np.arange(interestcols.size),
            LRresult1.loc[targ_df1.columns[interestcols], "coef"],
            yerr=LRresult1.loc[targ_df1.columns[interestcols], "std err"],
            marker="o",
            c=colors[1],
            label="SR a-corr",
        )
        ax[0].set_xticks(np.arange(interestcols.size))
        ax[0].set_xticklabels(["S" + str(x) for x in range(1, 9)])
        ax[0].set_title("sensory")
        ax[0].set_ylabel("weight")
        ax[0].axhline(y=0, linestyle=":", c="k")
        # stars
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult1.loc[targ_df1.columns[interestcols], "P>|z|"].values <= 0.05)
        ]
        ax[0].scatter(
            signifposition,
            LRresult1.loc[targ_df1.columns[interestcols], "coef"].values[signifposition]
            - 0.4,
            marker="*",
            c=np.array(colors[1]).reshape(1, -1),
        )

        interestcols = np.where(targ_df1.columns.str.startswith("SL"))[0]
        ax[0].plot(
            np.arange(interestcols.size),
            model1a.coef_[0, interestcols],
            "-o",
            c=colors[2],
            alpha=0.5,
        )
        ax[0].errorbar(
            np.arange(interestcols.size),
            LRresult1.loc[targ_df1.columns[interestcols], "coef"],
            yerr=LRresult1.loc[targ_df1.columns[interestcols], "std err"],
            marker="o",
            c=colors[2],
            label="SL a-corr",
        )
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult1.loc[targ_df1.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[0].scatter(
            signifposition,
            LRresult1.loc[targ_df1.columns[interestcols], "coef"].values[signifposition]
            - 0.4,
            marker="*",
            c=np.array(colors[2]).reshape(1, -1),
        )

        ax[0].scatter(
            8,
            model1a.coef_[0, np.where(targ_df1.columns == "intercept")[0][0]],
            c="k",
            alpha=0.5,
        )
        ax[0].errorbar(
            8,
            LRresult1.loc["intercept", "coef"],
            yerr=LRresult1.loc["intercept", "std err"],
            marker="o",
            c="k",
            label="intercept ac",
        )
        if LRresult1.loc["intercept", "P>|z|"] <= 0.05:
            ax[0].scatter(
                8, LRresult1.loc["intercept", "coef"] - 0.4, marker="*", color="k"
            )
    else:
        interestcols = np.where(targ_df1.columns.str.startswith("S"))[0]
        f, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 6))
        ax = ax.flatten()
        ax[0].plot(
            np.arange(interestcols.size),
            model1a.coef_[0, interestcols],
            "-o",
            c=colors[1],
            alpha=0.5,
        )
        ax[0].errorbar(
            np.arange(interestcols.size),
            LRresult1.loc[targ_df1.columns[interestcols], "coef"],
            yerr=LRresult1.loc[targ_df1.columns[interestcols], "std err"],
            marker="o",
            c=colors[1],
            label="S a-corr",
        )
        ax[0].set_xticks(np.arange(interestcols.size).tolist()+[8])
        ax[0].set_xticklabels([f'S{x+1}' for x in np.arange(interestcols.size)]+['intercept'])
        ax[0].set_title("sensory")
        ax[0].set_ylabel("weight")
        ax[0].axhline(y=0, linestyle=":", c="k")
        # stars
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult1.loc[targ_df1.columns[interestcols], "P>|z|"].values <= 0.05)
        ]
        ax[0].scatter(
            signifposition,
            LRresult1.loc[targ_df1.columns[interestcols], "coef"].values[signifposition]
            - 0.4,
            marker="*",
            c=np.array(colors[1]).reshape(1, -1),
        )
        ax[0].scatter(
            8,
            model1a.coef_[0, np.where(targ_df1.columns == "intercept")[0][0]],
            c="tab:orange",
            alpha=0.5,
        )
        ax[0].errorbar(
            8,
            LRresult1.loc["intercept", "coef"],
            yerr=LRresult1.loc["intercept", "std err"],
            marker="o",
            c="tab:orange",
            label="intercept ac",
        )
        if LRresult1.loc["intercept", "P>|z|"] <= 0.05:
            ax[0].scatter(
                8, LRresult1.loc["intercept", "coef"] - 0.4, marker="*", color="tab:orange"
            )


    # aftereffects correct R9ight
    if lateralized:
        interestcols = np.where(targ_df1.columns.str.startswith("afterefR"))[0]
        ax[1].errorbar(
            np.arange(interestcols.size),
            LRresult1.loc[targ_df1.columns[interestcols], "coef"],
            yerr=LRresult1.loc[targ_df1.columns[interestcols], "std err"],
            marker="o",
            c=colors[1],
            label="afterefR correct",
        )
        ax[1].plot(
            np.arange(interestcols.size),
            model1a.coef_[0, interestcols],
            "-o",
            c=colors[1],
            alpha=0.5,
        )

        signifposition = np.arange(interestcols.size)[
            np.where(LRresult1.loc[targ_df1.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[1].scatter(
            signifposition,
            LRresult1.loc[targ_df1.columns[interestcols], "coef"].values[signifposition]
            + 0.05,
            marker="*",
            c=colors[1],
        )

        # Left
        interestcols = np.where(targ_df1.columns.str.startswith("afterefL"))[0]
        ax[1].errorbar(
            np.arange(interestcols.size),
            LRresult1.loc[targ_df1.columns[interestcols], "coef"],
            yerr=LRresult1.loc[targ_df1.columns[interestcols], "std err"],
            marker="o",
            c=colors[2],
            label="afterefL correct",
        )
        ax[1].plot(
            np.arange(interestcols.size),
            model1a.coef_[0, interestcols],
            "-o",
            c=colors[2],
            alpha=0.5,
        )
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult1.loc[targ_df1.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[1].scatter(
            signifposition,
            LRresult1.loc[targ_df1.columns[interestcols], "coef"].values[signifposition]
            + 0.05,
            marker="*",
            c=colors[2],
        )
    else:
        interestcols = np.where(targ_df1.columns.str.startswith("aftereff"))[0]
        ax[1].errorbar(
            np.arange(interestcols.size),
            LRresult1.loc[targ_df1.columns[interestcols], "coef"],
            yerr=LRresult1.loc[targ_df1.columns[interestcols], "std err"],
            marker="o",
            c=colors[1],
            label="aftereff correct",
        )
        ax[1].plot(
            np.arange(interestcols.size),
            model1a.coef_[0, interestcols],
            "-o",
            c=colors[1],
            alpha=0.5,
        )

        signifposition = np.arange(interestcols.size)[
            np.where(LRresult1.loc[targ_df1.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[1].scatter(
            signifposition,
            LRresult1.loc[targ_df1.columns[interestcols], "coef"].values[signifposition]
            + 0.05,
            marker="*",
            c=colors[1],
        )
    ax[1].set_title("aftereffect")
    ax[1].set_ylabel("weight")
    ax[1].axhline(y=0, linestyle=":", c="k")

    # aftererror
    LRresult2 = pd.read_html(
        model2b.summary(xname=targ_df2.columns.tolist()).tables[1].as_html(),
        header=0,
        index_col=0,
    )[0]
    if lateralized:
        interestcols = np.where(targ_df2.columns.str.startswith("SR"))[0]
        ax[0].plot(
            np.arange(interestcols.size),
            model2a.coef_[0, interestcols],
            "-o",
            c=colors[0],
            alpha=0.5,
        )
        ax[0].errorbar(
            np.arange(interestcols.size),
            LRresult2.loc[targ_df2.columns[interestcols], "coef"],
            yerr=LRresult2.loc[targ_df2.columns[interestcols], "std err"],
            marker="o",
            c=colors[0],
            label="SR a-err",
        )
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult2.loc[targ_df2.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[0].scatter(
            signifposition,
            LRresult2.loc[targ_df2.columns[interestcols], "coef"].values[signifposition]
            + 0.4,
            marker="*",
            c=np.array(colors[0]).reshape(1, -1),
        )

        interestcols = np.where(targ_df2.columns.str.startswith("SL"))[0]
        ax[0].plot(
            np.arange(interestcols.size),
            model2a.coef_[0, interestcols],
            "-o",
            c=colors[3],
            alpha=0.5,
        )
        ax[0].errorbar(
            np.arange(interestcols.size),
            LRresult2.loc[targ_df2.columns[interestcols], "coef"],
            yerr=LRresult2.loc[targ_df2.columns[interestcols], "std err"],
            marker="o",
            c=colors[3],
            label="SL a-err",
        )
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult2.loc[targ_df2.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[0].scatter(
            signifposition,
            LRresult2.loc[targ_df2.columns[interestcols], "coef"].values[signifposition]
            + 0.4,
            marker="*",
            c=np.array(colors[3]).reshape(1, -1),
        )

        ax[0].scatter(
            8,
            model2a.coef_[0, np.where(targ_df2.columns == "intercept")[0][0]], # changed this to df2
            c="blue",
            alpha=0.5,
        )
        ax[0].errorbar(
            8,
            LRresult2.loc["intercept", "coef"],
            yerr=LRresult2.loc["intercept", "std err"],
            marker="o",
            c="blue",
            label="intercept ae",
        )
        if LRresult2.loc["intercept", "P>|z|"] <= 0.05:
            ax[0].scatter(
                8, LRresult2.loc["intercept", "coef"] + 0.4, marker="*", color="blue"
            )
    else:
        interestcols = np.where(targ_df2.columns.str.startswith("S"))[0]
        ax[0].plot(
            np.arange(interestcols.size),
            model2a.coef_[0, interestcols],
            "-o",
            c='tab:blue',
            alpha=0.5,
        )
        ax[0].errorbar(
            np.arange(interestcols.size),
            LRresult2.loc[targ_df2.columns[interestcols], "coef"],
            yerr=LRresult2.loc[targ_df2.columns[interestcols], "std err"],
            marker="o",
            c='tab:blue',
            label="S a-err",
        )
        signifposition = np.arange(interestcols.size)[
            np.where(LRresult2.loc[targ_df2.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[0].scatter(
            signifposition,
            LRresult2.loc[targ_df2.columns[interestcols], "coef"].values[signifposition]
            + 0.4,
            marker="*",
            c='tab:blue'
        )

        ax[0].scatter(
            8,
            model2a.coef_[0, np.where(targ_df2.columns == "intercept")[0][0]], # changed this to df2
            c="tab:blue",
            alpha=0.5,
        )
        ax[0].errorbar(
            8,
            LRresult2.loc["intercept", "coef"],
            yerr=LRresult2.loc["intercept", "std err"],
            marker="o",
            c="tab:blue",
            label="intercept ae",
        )
        if LRresult2.loc["intercept", "P>|z|"] <= 0.05:
            ax[0].scatter(
                8, LRresult2.loc["intercept", "coef"] + 0.4, marker="*", color="tab:blue"
            )
        interestcols = np.where(targ_df2.columns.str.startswith("aftereff"))[0]
        ax[1].errorbar(
            np.arange(interestcols.size),
            LRresult2.loc[targ_df2.columns[interestcols], "coef"],
            yerr=LRresult2.loc[targ_df2.columns[interestcols], "std err"],
            marker="o",
            c='tab:blue',
            label="aftereff error",
        )
        ax[1].plot(
            np.arange(interestcols.size),
            model2a.coef_[0, interestcols],
            "-o",
            c='tab:blue',
            alpha=0.5,
        )

        signifposition = np.arange(interestcols.size)[
            np.where(LRresult2.loc[targ_df2.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[1].scatter(
            signifposition,
            LRresult2.loc[targ_df2.columns[interestcols], "coef"].values[signifposition]
            + 0.05,
            marker="*",
            c=colors[1],
        )
    ax[1].set_title("aftereffect")
    ax[1].set_ylabel("weight")
    ax[1].axhline(y=0, linestyle=":", c="k")
        # now aftereff err
    if lateralized: # forgot to pack it with previous segment
        interestcols = np.where(targ_df2.columns.str.startswith("afterefR"))[0]
        ax[1].errorbar(
            np.arange(interestcols.size),
            LRresult2.loc[targ_df2.columns[interestcols], "coef"],
            yerr=LRresult2.loc[targ_df2.columns[interestcols], "std err"],
            marker="o",
            c=colors[0],
            label="afterefR error",
        )
        ax[1].plot(
            np.arange(interestcols.size),
            model2a.coef_[0, interestcols],
            "-o",
            c=colors[0],
            alpha=0.5,
        )

        signifposition = np.arange(interestcols.size)[
            np.where(LRresult2.loc[targ_df2.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[1].scatter(
            signifposition,
            LRresult2.loc[targ_df2.columns[interestcols], "coef"].values[signifposition]
            - 0.05,
            marker="*",
            c=colors[0],
        )
        # after L
        interestcols = np.where(targ_df2.columns.str.startswith("afterefL"))[0]
        ax[1].errorbar(
            np.arange(interestcols.size),
            LRresult2.loc[targ_df2.columns[interestcols], "coef"],
            yerr=LRresult2.loc[targ_df2.columns[interestcols], "std err"],
            marker="o",
            c=colors[3],
            label="afterefL error",
        )
        ax[1].plot(
            np.arange(interestcols.size),
            model2a.coef_[0, interestcols],
            "-o",
            c=colors[3],
            alpha=0.5,
        )

        signifposition = np.arange(interestcols.size)[
            np.where(LRresult2.loc[targ_df2.columns[interestcols], "P>|z|"].values < 0.05)
        ]
        ax[1].scatter(
            signifposition,
            LRresult2.loc[targ_df2.columns[interestcols], "coef"].values[signifposition]
            - 0.05,
            marker="*",
            c=colors[3],
        )

    ax[0].legend()
    ax[1].legend()
    plt.savefig(savpath)
    plt.show()


def plot_lateral_dual(targ_df1, targ_df2, model1a, model1b, model2a, model2b, savpath=''):
    Lp_labels = np.array(["L+1", "L+2", "L+3", "L+4", "L+5", "L+6-10"])
    Ln_labels = np.array(["L-1", "L-2", "L-3", "L-4", "L-5", "L-6-10"])
    f, ax = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=False, figsize=(16, 6))
    ax = ax.flatten()
    ax[0].set_ylabel("weight")
    for i, tit in enumerate(["L+", "L-"]):
        ax[i].set_title(tit)
        ax[i].axhline(y=0, linestyle=":", c="k")
        ax[i].set_xticks(np.arange(6))

    ax[0].set_xticklabels(Lp_labels)
    ax[1].set_xticklabels(Ln_labels)

    for i, batch in enumerate(
        [
            (targ_df1, model1a, model1b, np.array(sns.color_palette()[1]), 1),
            (targ_df2, model2a, model2b, np.array(sns.color_palette()[0]), -1),
        ]
    ):

        # L+
        LRresult = pd.read_html(
            batch[2].summary(xname=batch[0].columns.tolist()).tables[1].as_html(),
            header=0,
            index_col=0,
        )[0]
        interestcols = np.where(batch[0].columns.str.startswith("L+"))[0]  # review
        xpos = np.searchsorted(
            Lp_labels, batch[0].columns[batch[0].columns.str.startswith("L+")]
        )

        ax[0].plot(
            xpos,
            batch[1].coef_[0, np.where(batch[0].columns.str.startswith("L+"))[0]],
            "-o",
            c="r",
        )
        ax[0].errorbar(
            xpos,
            LRresult.loc[batch[0].columns[interestcols], "coef"],
            yerr=LRresult.loc[batch[0].columns[interestcols], "std err"],
            marker="o",
            c=batch[3],
        )

        signifposition = xpos[
            np.where(
                LRresult.loc[batch[0].columns[interestcols], "P>|z|"].values < 0.05
            )
        ]
        signifmask = np.where(
            LRresult.loc[batch[0].columns[interestcols], "P>|z|"].values < 0.05
        )
        ax[0].scatter(
            signifposition,
            LRresult.loc[batch[0].columns[interestcols], "coef"].values[signifmask]
            + (0.05 * batch[-1]),
            marker="*",
            c=batch[3],
        )
        # L-
        interestcols = np.where(batch[0].columns.str.startswith("L-"))[0]
        xpos = np.searchsorted(
            Ln_labels, batch[0].columns[batch[0].columns.str.startswith("L-")]
        )

        signifposition = xpos[
            np.where(
                LRresult.loc[batch[0].columns[interestcols], "P>|z|"].values < 0.05
            )
        ]
        signifmask = np.where(
            LRresult.loc[batch[0].columns[interestcols], "P>|z|"].values < 0.05
        )

        ax[1].plot(
            xpos,
            batch[1].coef_[0, np.where(batch[0].columns.str.startswith("L-"))[0]],
            "-o",
            c="r",
        )
        ax[1].errorbar(
            xpos,
            LRresult.loc[batch[0].columns[interestcols], "coef"],
            yerr=LRresult.loc[batch[0].columns[interestcols], "std err"],
            marker="o",
            c=batch[3],
        )

        ax[1].scatter(
            signifposition,
            LRresult.loc[batch[0].columns[interestcols], "coef"].values[signifmask]
            + (0.05 * batch[-1]),
            marker="*",
            c=batch[3].reshape(1, -1),
        )
    plt.savefig(savpath)
    plt.show()


def plot_transition_dual(targ_df1, targ_df2, model1a, model1b, model2a, model2b, savpath=''):
    Tpp_labels = np.array(["T++1", "T++2", "T++3", "T++4", "T++5", "T++6-10"])
    Tpn_labels = np.array(["T+-1", "T+-2", "T+-3", "T+-4", "T+-5", "T+-6-10"])
    Tnp_labels = np.array(["T-+1", "T-+2", "T-+3", "T-+4", "T-+5", "T-+6-10"])
    Tnn_labels = np.array(["T--1", "T--2", "T--3", "T--4", "T--5", "T--6-10"])
    label_list = [Tpp_labels, Tpn_labels, Tnp_labels, Tnn_labels]

    f, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 9), sharey=True)
    ax = ax.flatten()

    for j, name in enumerate(["T++", "T+-", "T-+", "T--"]):
        ax[j].set_xticks(np.arange(6))
        ax[j].set_xlim([-0.1, 5.1])
        ax[j].set_xticklabels(label_list[j])
        ax[j].axhline(y=0, linestyle=":", c="k")
        ax[j].set_ylabel("weight")
        ax[j].set_title(name)

        for i, batch in enumerate(
            [
                (targ_df1, model1a, model1b, np.array(sns.color_palette()[1]), 1),
                (targ_df2, model2a, model2b, np.array(sns.color_palette()[0]), -1),
            ]
        ):
            LRresult = pd.read_html(
                batch[2].summary(xname=batch[0].columns.tolist()).tables[1].as_html(),
                header=0,
                index_col=0,
            )[0]
            interestcols = np.where(batch[0].columns.str.startswith(name))[0]
            # numcols = interestcols.size
            interestcols = np.where(batch[0].columns.str.startswith(name))[0]  # review
            xpos = np.searchsorted(
                label_list[j], batch[0].columns[batch[0].columns.str.startswith(name)]
            )

            ax[j].plot(
                xpos,
                batch[1].coef_[0, np.where(batch[0].columns.str.startswith(name))[0]],
                "-o",
                c="r",
                alpha=0.5,
            )
            ax[j].errorbar(
                xpos,
                LRresult.loc[batch[0].columns[interestcols], "coef"],
                yerr=LRresult.loc[batch[0].columns[interestcols], "std err"],
                marker="o",
                c=batch[3],
            )

            signifposition = xpos[
                np.where(
                    LRresult.loc[batch[0].columns[interestcols], "P>|z|"].values < 0.05
                )
            ]
            signifmask = np.where(
                LRresult.loc[batch[0].columns[interestcols], "P>|z|"].values < 0.05
            )
            ax[j].scatter(
                signifposition,
                LRresult.loc[batch[0].columns[interestcols], "coef"].values[signifmask]
                + (0.25 * batch[-1]),
                marker="*",
                c=batch[3],
                s=80,
            )
    plt.savefig(savpath)
    plt.show()


def get_module_weight(df, dic, lateralized=False, noenv=False):
    """I FUCKOING DELETED IT because retarded cut icon + copy rather than pasting"""
    if lateralized and noenv:
        raise NotImplementedError('cannot use lateralized and noenv atm')
    if lateralized:
        stimcols = ["SR" + str(x) for x in range(1, 9)] + [
            "SL" + str(x) for x in range(1, 9)
        ]
        # aftereffcols = ['aftereff'+str(x) for x in range(1,11)]
        aftereffcols = ["afterefR" + str(x) for x in range(1, 11)] + [
            "afterefL" + str(x) for x in range(1, 11)
        ]
    else:
        aftereffcols = ['aftereff'+str(x) for x in range(1,11)]
        if not noenv:
            stimcols = ["S" + str(x) for x in range(1, 9)]
        else:
            stimcols = ['S']
    fcolnames = ["stim", "short_s", "lat", "trans"]


    if len(list(dic.keys())) == 4:  # single
        prefix = "sW_"
        # for newcol in [prefix+x for x in fcolnames if (prefix+x) not in df.columns]:
        #    df[newcol] = np.nan

        scomp = np.dot(
            dic["mat"].loc[stimcols, "coef"].values.reshape(1, -1),
            df[stimcols].fillna(value=0).values.T,
        )
        afcomp = np.dot(
            dic["mat"].loc[aftereffcols, "coef"].values.reshape(1, -1),
            df[aftereffcols].fillna(value=0).values.T,
        )
        latcols = [x for x in dic["mat"].index if x.startswith("L")]
        transcols = [x for x in dic["mat"].index if x.startswith("T")]
        latcomp = np.dot(
            dic["mat"].loc[latcols, "coef"].values.reshape(1, -1),
            df[latcols].fillna(value=0).values.T,
        )
        transcomp = np.dot(
            dic["mat"].loc[transcols, "coef"].values.reshape(1, -1),
            df[transcols].fillna(value=0).values.T,
        )
        for col, vec in zip(fcolnames, [scomp, afcomp, latcomp, transcomp]):
            df[prefix + col] = vec.flatten()

        return df

    elif len(list(dic.keys())) == 7:  # double
        prefix = "dW_"
        # for newcol in [prefix+x for x in fcolnames if (prefix+x) not in df.columns]:
        #    df[newcol] = np.nan

        for i, key in enumerate(["mat_ac", "mat_ae"]):
            scomp = np.dot(
                dic[key].loc[stimcols, "coef"].values.reshape(1, -1),
                df.loc[df.aftererror == i, stimcols].fillna(value=0).values.T,
            )
            afcomp = np.dot(
                dic[key].loc[aftereffcols, "coef"].values.reshape(1, -1),
                df.loc[df.aftererror == i, aftereffcols].fillna(value=0).values.T,
            )
            latcols = [x for x in dic[key].index if x.startswith("L")]
            transcols = [x for x in dic[key].index if x.startswith("T")]
            latcomp = np.dot(
                dic[key].loc[latcols, "coef"].values.reshape(1, -1),
                df.loc[df.aftererror == i, latcols].fillna(value=0).values.T,
            )
            transcomp = np.dot(
                dic[key].loc[transcols, "coef"].values.reshape(1, -1),
                df.loc[df.aftererror == i, transcols].fillna(value=0).values.T,
            )
            for col, vec in zip(fcolnames, [scomp, afcomp, latcomp, transcomp]):
                df.loc[df.aftererror == i, prefix + col] = vec.flatten()
        return df

    else:
        print(f"requires a dict with3or6 keys, this got {len(list(dic.keys()))}")
        return -1
