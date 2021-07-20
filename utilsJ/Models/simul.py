"""
the idea is to get simulations working with few filepaths and parameters
then simply save a multi-pannel figure report.

so the following function can be called to do kind of a grid-search
"""
import os
from sklearn.model_selection import ParameterGrid
from utilsJ.regularimports import *
from utilsJ.Behavior import plotting, ComPipe
from utilsJ.Models import traj
from concurrent.futures import as_completed, ThreadPoolExecutor
from scipy.stats import ttest_ind, sem
from matplotlib import cm
import swifter
import seaborn as sns


def get_when_t(a, b, startfrom=700, tot_iter=1000, pval=0.001, nan_policy="omit"):
    """a and b are traj matrices.
    returns ms after motor onset and median of the first one (to plot)
    startfrom: matrix index to start from (should be 0th position in ms
    tot_iter= remaining)"""
    for i in range(tot_iter):
        t2, p2 = ttest_ind(
            a[:, startfrom + i], b[:, startfrom + i], nan_policy=nan_policy
        )
        if p2 < pval:
            return i  # , np.nanmedian(a[:,startfrom+i])
    return np.nan  # , -1


def when_did_split_dat(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7), startfrom=700):
    """gets when they are statistically different by t_test"""
    # get matrices
    if side == 0:
        coh1 = -1
    else:
        coh1 = 1
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        & (df.resp_len)
    ]  # &(df.R_response==side)
    mata = np.vstack(
        dat.loc[dat.coh2 == coh1]
        .swifter.apply(lambda x: plotting.interpolapply(x), axis=1)
        .values.tolist()
    )
    matb = np.vstack(
        dat.loc[(dat.coh2 == 0) & (dat.rewside == side)]
        .swifter.apply(lambda x: plotting.interpolapply(x), axis=1)
        .values.tolist()
    )
    for a in [mata, matb]:  # discard all nan rows
        a = a[~np.isnan(a).all(axis=1)]

    ind = get_when_t(mata, matb, startfrom=startfrom)
    return ind  # mata, matb,


def shortpad(traj, upto=1000):
    missing = upto - traj.size
    return np.pad(traj, ((0, missing)), "constant", constant_values=np.nan)


def when_did_split_simul(df, side, rtbin=0, rtbins=np.linspace(0, 150, 7)):
    """gets when they are statistically different by t_test
    here df is simulated df"""
    # get matrices
    if side == 0:
        coh1 = -1
    else:
        coh1 = 1
    dat = df.loc[
        (df.sound_len < rtbins[rtbin + 1])
        & (df.sound_len >= rtbins[rtbin])
        & (df.resp_len)
    ]  # &(df.R_response==side) this goes out
    mata = np.vstack(
        dat.loc[(dat.traj.apply(len) > 0) & (dat.coh2 == coh1), "traj"]
        .apply(shortpad)
        .values.tolist()
    )
    matb = np.vstack(
        dat.loc[
            (dat.traj.apply(len) > 0) & (dat.coh2 == 0) & (dat.rewside == side), "traj"
        ]
        .apply(shortpad)
        .values.tolist()
    )

    for a in [mata, matb]:  # discard all nan rows
        a = a[~np.isnan(a).all(axis=1)]
    #     for mat in [mata, matb]:
    #         plt.plot(
    #         np.nanmedian(mat, axis=0))
    #     plt.show()
    ind = get_when_t(mata, matb, startfrom=0)
    return ind  # mata, matb,


def whole_splitting(df, rtbins=np.arange(0, 151, 25), simul=False):
    _index = [0, 1]  # side
    _columns = np.arange(rtbins.size - 1)  # rtbins
    tdf = pd.DataFrame(np.ones((2, _columns.size)) * -1, index=_index, columns=_columns)
    if simul:
        splitfun = when_did_split_simul
    else:
        splitfun = when_did_split_dat
    # tdf.columns = tdf.columns.set_names(['RTbin', 'side'])
    for b in range(rtbins.size - 1):
        for s, side in enumerate(["L", "R"]):
            split_time = splitfun(df, s, b, rtbins=rtbins)
            tdf.loc[s, b] = split_time

    return tdf


def splitplot(df, out, ax):
    tdf = whole_splitting(df)
    tdf2 = whole_splitting(out, simul=True)
    colors = ["green", "purple"]
    for i, (dat, name, marker) in enumerate([[tdf, "data", "o"], [tdf2, "simul", "x"]]):
        for j, side in enumerate(["L", "R"]):
            ax.scatter(
                dat.columns,
                dat.loc[j, :].values,
                marker=marker,
                color=colors[j],
                label=f"{name} {side}",
            )

    ax.set_xlabel("rtbin")
    ax.set_ylabel("time to diverge")
    ax.legend(fancybox=False, frameon=False)


def plot0(df, out, ax):
    df.loc[(df.sound_len < 250)].sound_len.hist(
        bins=np.linspace(0, 250, 101),
        ax=ax,
        label="data",
        density=True,
        alpha=0.5,
        grid=False,
    )
    out.sound_len.hist(
        bins=np.linspace(0, 250, 101),
        ax=ax,
        label="simul all",
        density=True,
        alpha=0.5,
        grid=False,
    )
    ax.set_xlabel("RT (ms)")
    ax.set_title("RT distr")
    ax.legend(frameon=False, fancybox=False)


def pcomRT(df, out, ax):
    plotting.binned_curve(
        df,
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="data", color="tab:blue"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls="-"),
    )
    plotting.binned_curve(
        out,
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="simul", color="tab:orange"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls=":"),
    )
    ax.set_ylabel("p(CoM)")
    ax.set_xlabel("RT(ms)")
    ax.set_title("pcom vs rt")
    ax.legend(frameon=False, fancybox=False)


def pcomRT_proactive_only(df, out, ax):
    plotting.binned_curve(
        out.loc[out.reactive == 0],
        "CoM_sugg",
        "sound_len",
        bins=np.linspace(0, 250, 26),
        xpos=10,
        xoffset=5,
        ax=ax,
        errorbar_kw=dict(label="simul", color="tab:orange"),
        legend=False,
        traces="sstr",
        traces_kw=dict(color="grey", alpha=0.3, ls=":"),
    )
    ax.set_ylabel("p(CoM)")
    ax.set_xlabel("RT(ms)")
    ax.set_title("pcom in proactive")
    ax.legend(frameon=False, fancybox=False)


def plot2(df, out, ax):
    df.resp_len.hist(
        bins=np.linspace(0, 1, 81),
        ax=ax,
        label="data",
        density=True,
        alpha=0.5,
        grid=False,
    )
    out.resp_len.hist(
        bins=np.linspace(0, 1, 81),
        ax=ax,
        label="simul all",
        density=True,
        alpha=0.5,
        grid=False,
    )
    ax.set_xlabel("MT (secs)")
    ax.set_title("MT distr")
    ax.legend(frameon=False, fancybox=False)


def plot3(df, out, ax):
    titles = ["data all", "simul all"]
    datacol = ["resp_len", "resp_len"]
    traces_ls = ["-", ":"]
    for i, dfobj in enumerate([df, out]):  # .loc[out.reactive==0]
        plotting.binned_curve(
            dfobj,
            datacol[i],
            "sound_len",
            ax=ax,
            bins=np.linspace(0, 150, 16),
            # xpos=np.arange(0,150,10), traces='sstr', traces_kw = dict(color='grey', alpha=0.3, ls=traces_ls[i]),
            xpos=10,
            traces="sstr",
            traces_kw=dict(color="grey", alpha=0.3, ls=traces_ls[i]),
            xoffset=5,
            errorbar_kw={"ls": "none", "marker": "o", "label": titles[i]},
        )
        ax.set_xlabel("RT (ms)")
        ax.set_ylabel("MT (s)")
        ax.set_title("MT vs RT")
        ax.legend(frameon=False, fancybox=False)


def plot4(df, out, ax):
    counts_t, xpos = np.histogram(out.sound_len, bins=np.linspace(0, 250, 26))
    counts_p, _ = np.histogram(
        out.loc[out.reactive == 0, "sound_len"], bins=np.linspace(0, 250, 26)
    )
    prop_pro = counts_p / counts_t
    ax.plot(xpos[:-1] + 5, prop_pro, marker="o")
    ax.set_ylabel("proportion proactive")
    ax.set_xlabel("RT")
    ax.set_ylim([-0.05, 1.05])


def plot5(df, out, ax):
    ax.set_title("stimuli split trajectories")
    ax.annotate("splitting time per rtbin", (0, 0))


def plot67(df, out, ax, ax2, rtbins=np.linspace(0, 150, 7)):
    markers = ["o", "x"]
    rtbins = np.linspace(0, 150, 7)
    priorbins = np.linspace(-2, 2, 6)
    cmap = cm.get_cmap("viridis_r")
    datres, simulres = pd.DataFrame([]), pd.DataFrame([])
    for i, (dat, store) in enumerate([[df, datres], [out, simulres]]):
        dat["priorbin"] = pd.cut(
            dat.choice_x_allpriors, priorbins, labels=False, include_lowest=True
        )
        for j in range(rtbins.size - 1):
            tmp = (
                dat[(dat.sound_len >= rtbins[j]) & (dat.sound_len < rtbins[j + 1])]
                .groupby("priorbin")["time_to_thr"]
                .agg(m="mean", e=sem)
            )
            store[f"rtbin{j}"] = tmp["m"]

            if j % 2 == 0:  # plot half of it
                kws = {
                    "ls": "none",
                    "marker": markers[i],
                    "color": cmap(j / (rtbins.size - 1)),
                    "capsize": 2,
                }
                if not i:
                    kws = {
                        "ls": "none",
                        "marker": markers[i],
                        "label": f"rtbin={j}",
                        "color": cmap(j / (rtbins.size - 1)),
                        "capsize": 2,
                    }
                ax.errorbar(tmp.index, tmp["m"], yerr=tmp["e"], **kws)
    ax.legend().remove()
    ax.set_ylabel("ms to threshold (30px)")
    ax.set_xlabel("congruence (choice * prior)")
    ax.set_title("prior congruence on MT (o=data, x=simul)")
    diffdf = datres - simulres
    ax2.axhline(0, c="gray", ls=":")
    ax2.set_xlabel("congruence (choice * prior)")
    ax2.set_ylabel("data - simul")
    for i in range(rtbins.size - 1):
        ax2.scatter(
            [-2, -1, 0, 1, 2],
            diffdf[f"rtbin{i}"].values,
            color=cmap(i / (rtbins.size - 1)),
            label=f"rtbin {i}",
        )
    ax2.set_title("prior congruence on MT (data - simul)")
    ax2.legend(frameon=False, fancybox=False)


def plot910(df, out, ax, ax2, rtbins=np.linspace(0, 150, 7)):
    markers = ["o", "x"]
    cmap = cm.get_cmap("viridis_r")
    datres, simulres = pd.DataFrame([]), pd.DataFrame([])
    for i, (dat, store) in enumerate([[df, datres], [out, simulres]]):
        kwargs = dict(ls="none", marker=markers[i], capsize=2)
        for j in range(rtbins.size - 1):
            tmp = (
                dat[(dat.sound_len >= rtbins[j]) & (dat.sound_len < rtbins[j + 1])]
                .groupby("choice_x_coh")["time_to_thr"]
                .agg(m="mean", e=sem)
            )
            store[f"rtbin{j}"] = tmp["m"]
            if j % 2 == 0:
                c = cmap(j / (rtbins.size - 1))
                ax.errorbar(tmp.index, tmp["m"], yerr=tmp["e"], **kwargs, c=c)
    ax.set_xlabel("coh * choice")
    ax.set_ylabel("ms to threshold (30px)")
    ax.set_title("coherence congruence on MT (o=data, x=simul)")
    diffdf = datres - simulres
    ax2.axhline(0, c="gray", ls=":")
    ax2.set_xlabel("coh * choice")
    ax2.set_ylabel("data - simul")
    ax2.set_title("coherence congruence on MT (data - simul)")
    for i in range(rtbins.size - 1):
        ax2.scatter(
            diffdf.index,
            diffdf[f"rtbin{i}"].values,
            color=cmap(i / (rtbins.size - 1)),
            label=f"rtbin {i}",
        )

    ax2.legend(frameon=False, fancybox=False)


def plot1112(df, out, ax, ax2):
    subset = df.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        flip=True,
        ax=ax,
        cmap="viridis",
        fmt=".0f",
    )
    ax.set_title(f"real p(CoM)")
    subset = out.dropna(subset=["avtrapz", "allpriors", "CoM_sugg"])
    plotting.com_heatmap(
        subset.allpriors,
        subset.avtrapz,
        subset.CoM_sugg,
        flip=True,
        ax=ax2,
        cmap="magma",
        fmt=".0f",
    )
    ax2.set_title(f" SIMULATIONS p(CoM)")


def _callsimul(args):
    return traj.simul_psiam(*args)


def safe_threshold(row, threshold):
    pass  # will this be implemented?


def whole_simul(
    subject,
    # grid,
    savpath=None,
    dfpath="/home/jordi/DATA/Documents/changes_of_mind/data/paper/dani_clean.pkl",  # parameter grid
    rtbins=np.linspace(0, 150, 7),
    params={
        "t_update": 80,
        "proact_deltaMT": 0.3,
        "reactMT_interc": 110,
        "reactMT_slope": 0.15,
        "com_gamma": 250,
        "glm2Ze_scaling": 0.25,
        "x_e0_noise": 0.001,
        "naive_jerk": False,
        "confirm_thr": 0,
        "proportional_confirm": False,
    },
    batches=10,
    batch_size=1500,
    return_data=False,
    vanishing_bounds=True,
    both_traj=False,
    silent_trials=False,
    sample_silent_only=False,
    trajMT_jerk_extension=0,
    mtnoise=True,
    com_height=5,
):
    # t_update: time it takes from bound hit to exert effect in movement
    # deltaMT: coef to reduce expected MT based on accumulated evidence
    # com_gamma:
    # vanishing bounds, horiz bounds disappear after AI
    # both_traj: whether to return prior and final or just final (updated) trajectory,
    # silent trials = no drift in evidence;
    # sample_silent = just sample silent trials to reproduce data
    if savpath is None:
        raise ValueError("provide save path")

    # load real data
    df = pd.read_pickle(dfpath)
    # this was giving issues, now restrict df usage to single subject
    df = df.loc[df.subjid == subject]
    df["sstr"] = df.coh2.abs()
    df["priorZt"] = np.nansum(
        df[["dW_lat", "dW_trans"]].values, axis=1
    )  # 'dW_fixedbias'
    df["prechoice"] = np.ceil(df.priorZt.values / 1000)
    df["prechoice"] = df.prechoice.astype(int)
    df["time_to_thr"] = np.nan
    # df.swifter.apply(lambda x: np.argmax(np.abs(plotting.interpolapply(x)[700:])>30), axis=1)
    # split lft and right now!
    df.loc[(df.R_response == 1) & (df.trajectory_y.apply(len) > 10), "time_to_thr"] = (
        df.loc[(df.R_response == 1) & (df.trajectory_y.apply(len) > 10)]
        .dropna(subset=["sound_len"])
        .swifter.apply(
            lambda x: np.argmax(plotting.interpolapply(x)[700:] > 30), axis=1
        )
    )  # axis arg not req. in series
    df.loc[(df.R_response == 0) & (df.trajectory_y.apply(len) > 10), "time_to_thr"] = (
        df.loc[(df.R_response == 0) & (df.trajectory_y.apply(len) > 10)]
        .dropna(subset=["sound_len"])
        .swifter.apply(
            lambda x: np.argmax(plotting.interpolapply(x)[700:] < -30), axis=1
        )
    )  # axis arg not req. in series
    df["rtbin"] = pd.cut(df.sound_len, rtbins, labels=False, include_lowest=True)
    df["choice_x_coh"] = (df.R_response * 2 - 1) * df.coh2
    df["allpriors"] = np.nansum(
        df[["dW_trans", "dW_lat"]].values, axis=1
    )  # , 'dW_fixedbias'
    df["choice_x_allpriors"] = (df.R_response * 2 - 1) * df.allpriors

    psiam_params = loadmat(
        f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/fits_psiam/{subject} D2Mconstrainedfit_fitonly.mat"
    )["freepar_hat"][0]
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

    out = pd.DataFrame([])
    print("psiam_simul began")
    if sample_silent_only:
        df = df[df.special_trial == 2]
        print(
            f"just sampling from silent trials in subject {subject}\nwhich is around {len(df.loc[df.subjid==subject])}"
        )
    with ThreadPoolExecutor(max_workers=7) as executor:
        jobs = [
            executor.submit(
                _callsimul,
                [
                    df.loc[df.subjid == subject],
                    f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/fits_psiam/{subject} D2Mconstrainedfit_fitonly.mat",
                    1.3,
                    0.3,
                    1e-4,
                    x,
                    batches,
                    batch_size,
                    params["glm2Ze_scaling"],
                    silent_trials,
                    sample_silent_only,
                    params["x_e0_noise"],
                    params["confirm_thr"],
                    params["proportional_confirm"],
                    params["confirm_ae"],
                ],
            )
            for x in np.arange(7) * 50  # +1050#4200
        ]
        for job in tqdm.tqdm_notebook(as_completed(jobs), total=7):
            out = out.append(job.result(), ignore_index=True)

    tr = traj.def_traj(subject, None)
    out["expectedMT"] = np.nan
    out["mu_boundary"] = np.nan
    out["mu_boundary"] = out["mu_boundary"].astype(object)

    # WARNING using fixed boas here...
    out["priorZt"] = np.nansum(
        out[["dW_lat", "dW_trans"]].values, axis=1
    )  # 'dW_fixedbias',
    out["prechoice"] = np.ceil(out.priorZt.values / 1000)
    out["prechoice"] = out.prechoice.astype(int)

    for col in ["dW_trans", "dW_lat"]:  # invert those factors in left choices
        # out[f'{col}_i'] = out[col] * (out['R_response']*2-1)
        out[f"{col}_i"] = out[col] * (out["prechoice"] * 2 - 1)

    try:  ### PROACTIVE RESPONSES
        sdf = out.loc[out.reactive == 0]
        tr.selectRT(0)
        fkmat = sdf[["zidx", "dW_trans_i", "dW_lat_i"]].fillna(0).values
        fkmat = np.insert(fkmat, 0, 1, axis=1)
        # fkmat = sdf[['zidx', 'dW_trans', 'dW_lat']].fillna(0).values
        # fkmat = np.insert(fkmat, 0, 1,axis=1)
        tr.expected_mt(fkmat, add_intercept=False)
        out.loc[sdf.index, "expectedMT"] = tr.mt * 1000
        if mtnoise:  # load error
            with open(
                f"/home/jordi/DATA/Documents/changes_of_mind/data/paper/trajectory_fit/MTmse.pkl",
                "rb",
            ) as handle:
                msedict = pickle.load(handle)

            err = msedict[subject] ** 0.5
            out.loc[sdf.index, "expectedMT"] += np.random.normal(
                scale=err, size=out.loc[sdf.index, "expectedMT"].values.size
            )

        if params["naive_jerk"]:
            naive_jerk = np.array([0, 0, 0, 75, 0, 0]).reshape(
                -1, 1
            )  # broadcast final position to port # since it is aligned to final choice, that-s it.
            # It will be aligned later when generating trajectories
            out.loc[sdf.index, f"mu_boundary"] = len(sdf) * [naive_jerk]
        else:
            out.loc[sdf.index, f"mu_boundary"] = tr.return_mu(Fk=fkmat)
            # out.loc[sdf.index, f'priortraj{side}'] = bo.prior_traj(Fk=fkmat,times=bo.mt+0.05,step=10)

        ### REACTIVE ONES
        sdf = out.loc[out.reactive == 1]
        fkmat = sdf[["zidx", "dW_trans_i", "dW_lat_i"]].fillna(0).values
        fkmat = np.insert(fkmat, 0, 1, axis=1)
        times = (
            sdf.loc[sdf.index, "origidx"].values * params["reactMT_slope"]
            + params["reactMT_interc"]
        )
        out.loc[sdf.index, f"expectedMT"] = times
        if params["naive_jerk"]:
            naive_jerk = np.array([0, 0, 0, 75, 0, 0]).reshape(
                -1, 1
            )  # broadcast final position to port # since it is aligned to final choice, that-s it.
            # It will be aligned later when generating trajectories
            out.loc[sdf.index, f"mu_boundary"] = len(sdf) * [naive_jerk]
        else:
            out.loc[sdf.index, f"mu_boundary"] = tr.return_mu(
                Fk=fkmat
            )  # this one is not used later, right?
            # out.loc[sdf.index, f'priortraj{side}'] = bo.prior_traj(step=10, Fk=fkmat, times=(times+50)/1000)
    except Exception as e:
        raise e

    out["e_after_u"] = 0
    remaining_sensory = (t_0_e - 0.3) * 1000
    out["remaining_sensory"] = remaining_sensory
    out.loc[out.sound_len < remaining_sensory, "remaining_sensory"] = out.loc[
        out.sound_len < remaining_sensory, "sound_len"
    ]
    out.loc[
        (out.e_time * 1000 < out.sound_len + out.remaining_sensory)
        & (out.reactive == 0),
        "e_after_u",
    ] = 1  # control for those which have listened less than the delay # done
    # TODO is accu ev really taken into account?
    out.prechoice = out.prechoice.astype(int)
    out["sstr"] = out.coh2.abs()
    out["t_update"] = np.nan
    # out.loc[out.reactive==0, 't_update'] = params['t_update'] + out.loc[out.reactive==0, 'sound_len'] # a big bug lied here*
    out.loc[out.reactive == 0, "t_update"] = params[
        "t_update"
    ]  # now it happen relative to movement onset!
    # how defek? shouldnt it be respective movement onset + te + tupdate? check model scketch?
    # TODO: probably this is a bug as well

    # UNCOMMENT LINE BELOW IF UPDATE CAN HAPPEN EARLIER WHEN EV-BOUND IS REACHED
    if not vanishing_bounds:
        # out.loc[out.e_after_u==1, 't_update'] = out.loc[out.e_after_u==1, 'e_time']
        out.loc[out.e_after_u == 1, "t_update"] = (
            (out.loc[out.e_after_u == 1, "e_time"] - remaining_sensory)
            + params["t_update"]
        )  # e_time is already respective sound onset
        # sns.distplot(out.t_update.dropna())
        # plt.show()

    # add speed up for confirmed choices
    out["resp_len"] = np.nan
    out["resp_len"] = out["expectedMT"]  # this include reactive
    out["base_mt"] = out["expectedMT"]  # this include reactive
    if not silent_trials:
        if params["confirm_thr"] > 0:
            out.loc[
                (out.R_response == out.prechoice) & (out.reactive == 0), "resp_len"
            ] = (
                (
                    1
                    - params["proact_deltaMT"]
                    * out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "delta_ev",
                    ].abs()
                    / a_e
                )
                * (
                    out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "resp_len",
                    ]
                    - out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "t_update",
                    ]
                )
                + out.loc[
                    (out.R_response == out.prechoice) & (out.reactive == 0), "t_update"
                ]
            )  # prechoice==final choice aka confirm
        else:
            out.loc[
                (out.R_response == out.prechoice) & (out.reactive == 0), "resp_len"
            ] = (
                (
                    1
                    - params["proact_deltaMT"]
                    * out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0), "x_e"
                    ].abs()
                    / a_e
                )
                * (
                    out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "resp_len",
                    ]
                    - out.loc[
                        (out.R_response == out.prechoice) & (out.reactive == 0),
                        "t_update",
                    ]
                )
                + out.loc[
                    (out.R_response == out.prechoice) & (out.reactive == 0), "t_update"
                ]
            )  # prechoice==final choice aka confirm
        #     out.loc[(out.R_response==i)&(out.prechoice==abs(i-1))&(out.reactive==0),  'resp_len'] = (1-deltaMT *out.loc[(out.R_response==i)&(out.prechoice==abs(i-1))&(out.reactive==0), 'x_e'].abs()/a_e ) * (out.loc[(out.R_response==i)&(out.prechoice==abs(i-1))&(out.reactive==0), 'resp_len'] - out.loc[(out.R_response==i)&(out.prechoice==abs(i-1))&(out.reactive==0),  't_update']) +\
        #                                                                     out.loc[(out.R_response==i)&(out.prechoice==abs(i-1))&(out.reactive==0),  't_update'] * com_handicap   # prechoice!=final choice aka com
        out.loc[(out.R_response != out.prechoice) & (out.reactive == 0), "resp_len"] = (
            out.loc[(out.R_response != out.prechoice) & (out.reactive == 0), "t_update"]
            + params["com_gamma"]
            + params["reactMT_slope"]
            * out.loc[
                (out.R_response != out.prechoice) & (out.reactive == 0), "origidx"
            ]
        )  # tri_ind is new

    out["resp_len"] /= 1000
    out["base_mt"] /= 1000
    for col in ["zidx", "origidx", "expectedMT", "resp_len", "base_mt"]:
        out[col] = out[col].astype(float)

    out["allpriors"] = np.nansum(out[["dW_trans", "dW_lat"]].values, axis=1)
    out["choice_x_allpriors"] = (out.R_response * 2 - 1) * out.allpriors
    out["traj"] = np.nan
    out.traj = out.traj.astype(object)
    print("psiam done, generating trajectories + derivatives")
    if both_traj:
        out["pretraj"] = np.nan
        out.pretraj = out.pretraj.astype(object)
        out[["pretraj", "traj"]] = out.swifter.apply(
            lambda x: traj.simul_traj_single(
                x,
                return_both=True,
                silent_trials=silent_trials,
                trajMT_jerk_extension=trajMT_jerk_extension,
            ),
            axis=1,
            result_type="expand",
        )
    else:
        out["traj"] = out.swifter.apply(
            lambda x: traj.simul_traj_single(
                x,
                silent_trials=silent_trials,
                trajMT_jerk_extension=trajMT_jerk_extension,
            ),
            axis=1,
        )

    tmp = out.apply(
        lambda x: plotting.gradient_np_simul(x), axis=1, result_type="expand"
    )  # add swifter
    tmp.columns = ["traj_d1", "traj_d2", "traj_d3"]
    out = pd.concat([out, tmp], axis=1)

    print("getting CoM etc.")
    out["time_to_thr"] = np.nan
    for i, thr in enumerate([30, 30]):
        out.loc[
            (out.R_response == i) & (out.traj.apply(len) > 0), "time_to_thr"
        ] = out.loc[
            (out.R_response == i) & (out.traj.apply(len) > 0), "traj"
        ].swifter.apply(
            lambda x: np.argmax(np.abs(x) > thr)
        )
    out["rtbin"] = pd.cut(out.sound_len, rtbins, labels=False, include_lowest=True)
    out["choice_x_coh"] = (out.R_response * 2 - 1) * out.coh2
    out[["Hesitation", "CoM_sugg", "CoM_peakf"]] = out.apply(
        lambda x: ComPipe.chom.did_he_hesitate(
            x, simul=True, positioncol="traj", speedcol="traj_d1", height=com_height
        ),
        axis=1,
        result_type="expand",
    )  # return 1 or more peak frames?

    # not saving data, its 2.9 GB each
    # print('saving data')
    # out.to_pickle(f'{savpath}.pkl')

    # plotting
    if silent_trials:
        pref_title = "silent_"
        a, b = df.loc[(df.special_trial == 2) & (df.subjid == subject)], out
    else:
        pref_title = ""
        a, b = df.loc[(df.special_trial == 0) & (df.subjid == subject)], out
    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(24, 15))
    plot0(a, b, ax[0, 0])
    # plot1(a,b, ax[1,0])
    plot2(a, b, ax[1, 0])
    pcomRT(a, b, ax[1, 1])
    pcomRT_proactive_only(a, b, ax[1, 2])
    plot3(a, b, ax[0, 1])  # ushape
    plot4(a, b, ax[2, 0])
    # plot5(a,b,ax[2,1])
    # try:
    #     splitplot(a,b,ax[2,1])
    # except:
    #     print("splitplot typically crashes with silent trials\nbecause in dani's tasks they all have the same coh")
    # plot67(a,b,ax[0,2], ax[1,2])
    # plot910(a,b,ax[0,3], ax[1,3])
    plot1112(a, b, ax[2, 2], ax[2, 3])

    fig.suptitle(pref_title + subject + " " + str(params))
    # suptitle
    fname = f"{pref_title}{subject}-"
    for k, i in params.items():
        fname += f"{k}-{i}-"
    fname = fname[:-1] + ".png"
    # if not return_data:
    fig.savefig(f"{savpath}{fname}")
    if return_data:
        plt.show()
        return df, out
