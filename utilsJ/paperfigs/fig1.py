# the idea here is to stack all code for all "figures" so they can be reproduced
# making this code modular/versatile is out of scope (most of the times i'll reuse)
# code already written in custom_utils (https://github.com/PastorJordi/custom_utils)

# all functions shoul take an unfiltered (all subjects) dataframe, and or
# likewise for simulations

from utilsJ.regularimports import * 
from utilsJ.Behavior import plotting
from scipy.stats import norm, sem
from scipy.optimize import minimize
from matplotlib import cm


SAVPATH = '/home/jordi/Documents/changes_of_mind/changes_of_mind_scripts/PAPER/paperfigs/test_fig_scripts/' # where to save figs
SAVEPLOTS = True # whether to save plots or just show them


### ACCESSORY FUNCTIONS

def sigmoid_sens(fit_params, x, y):
    # params = sens, bias
    sens, bias = fit_params
    y_hat = 1/ (1+np.exp(-sens*(x+bias)))
    return -np.sum(norm.logpdf(y, loc=y_hat))

def opt_sig(y, x):
    loglike = minimize(
        sigmoid_sens,
        [0,0],
        (x,y)
    )
    params = loglike['x']
    return params

def exp_decay(t, N0=1, tau=1.5):
    return N0 * np.exp(-t / tau)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


### FIG1

def b(StimLen = 100, offset=150, maxval=1000, savpath=SAVPATH):
    """trial scheme"""
    LED, LEDKw = (
        np.array([0, offset, offset, 300+offset, 300+offset, maxval]), 
        np.array([1,1,1,1,0,0])+10), dict(color='#8b8000')
    CenterPort, CenterPortKw = (
        np.array([0,offset, offset, 300+offset+StimLen, 300+offset+StimLen, maxval]),
        np.array([0,0,1,1,0,0]) + 8) , dict(color='k')
    Stim, StimKw = (
        np.array([0,offset+300, offset+300, offset+300+StimLen, offset+300+StimLen, maxval]), 
        np.array([0, 0, 1, 1, 0, 0])+6), dict(color='gray')
    SidePort, SidePortKw = (
        np.array([0, 850, 850, maxval]), np.array([0,0,1,1])+4
    ), dict(color='k')
    Reward, RewardKw = (
        np.array([0,850, 850, 900, 900, maxval]), 
        np.array([0, 0, 1, 1, 0, 0])+2), dict(color='tab:blue')

    f, ax = plt.subplots(figsize=(8,4))
    f.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_ylim(1,11.5)
    ax.plot(*LED, **LEDKw)
    ax.text(0,10.25, 'Center LED', **LEDKw)
    ax.plot(*CenterPort, **CenterPortKw)
    ax.text(0, 8.25, 'Center Port', **CenterPortKw)
    ax.plot(*Stim, **StimKw)
    ax.text(0, 6.25, 'Stimulus', **StimKw)
    ax.plot(*SidePort, **SidePortKw)
    ax.text(0, 4.25, 'Side Port' ,**SidePortKw)
    ax.plot(*Reward, **RewardKw)
    ax.text(0, 2.25, 'Reward', **RewardKw)
    # anotate stuff

    ax.fill_between([offset,offset+300], [11, 11], y2=[2,2], color='maroon', alpha=0.3)
    ax.fill_between([offset+300, offset+300+StimLen], [11, 11], y2=[2,2], color='teal', alpha=0.3)
    # fixation and RT below fill between
    h = 1.5 # arrow height
    th = 1 # text height
    s = offset
    distances = [300, StimLen, 850-(offset+300+StimLen)]
    text = ['Fixation', 'RT', 'MT']
    for d, t in zip(distances,text):
        e = s+d
        ax.annotate(text='', xytext=(s, 1.5), xy=(e, 1.5), arrowprops=dict(arrowstyle='<->', color='k'), zorder=3)
        ax.text((s+e)/2, th, t, horizontalalignment='center')
        s = e

    if SAVEPLOTS:    
        f.savefig(f'{savpath}1b_trial_scheme.svg')
    plt.show()


def d(df, average=False, bins=np.linspace(-1,1,8), savpath=SAVPATH):
    # here we will stack priors and scale them
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1)==2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values,axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan

    df['norm_allpriors'] = (
        df.allpriors
        / df.groupby('subjid').allpriors.transform(lambda x: np.max(np.abs(x)))
    )
    # transform to rep-alt space
    df['prev_response'] = df.R_response.shift(1)*2 -1
    df.loc[df.origidx==1, 'prev_response'] = np.nan
    df['norm_allpriors_repalt'] = df.prev_response * df.norm_allpriors
    df['rep_response'] = (df.prev_response * (df.R_response*2-1)+1)/2
    df['ev_repeat'] = df.avtrapz * df.prev_response
    df['coh2repalt'] = df.coh2 * df.prev_response
    cmap = cm.get_cmap('coolwarm')

    if average:
        f, ax = plt.subplots(figsize=(6,6))
        f.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_ylabel('p(Repeat)')
        ax.set_xlabel('evidence to repeat')
        curves = np.zeros(
            (df.subjid.unique().size, bins.size-1, 50) # dimensions: subject, curve/serie/prior, npoints/curve
        ) * np.nan
        dots = np.zeros(
            (df.subjid.unique().size, bins.size-1, 7)
        ) * np.nan
        for i, subject in enumerate(df.subjid.unique()):
            for j in range(bins.size-1): # prior bins
                test = df.loc[
                    (df.subjid==subject)
                    &(df.special_trial==0)
                    &(df.norm_allpriors_repalt>=bins[j])
                    &(df.norm_allpriors_repalt<=bins[j+1])
                    ].dropna(subset=['ev_repeat', 'norm_allpriors_repalt', 'rep_response'])
                xpoints, ypoints, _, (xcurve, ycurve), _=plotting.psych_curve(
                    test.rep_response,
                    test.coh2repalt,
                    kwargs_plot=dict(color=cmap(np.linspace(0,1,7)[j])),
                    kwargs_error=dict(color=cmap(np.linspace(0,1,7)[j])),
                    ret_ax=None
                )
                if xpoints.size==7:
                    try:
                        curves[i,j,:] = ycurve
                        dots[i, j, :] = ypoints
                    except Exception as e:
                        print(subject)
                        print(ypoints)
                        raise(e)

        # average
        curves = np.nanmean(curves, axis=0)
        dots_sem = sem(dots, axis=0, nan_policy='omit')
        dots = np.nanmean(dots, axis=0)

        for b in range(bins.size-1):
            ccolor = cmap(b/(bins.size-1))
            ax.plot(xcurve, curves[b], color=ccolor)
            ax.errorbar(
                xpoints, dots[b], yerr=dots_sem[i], color=ccolor,
                marker='o', capsize=2
            )

    else:
        f, ax = plt.subplots(nrows=6, ncols=3, figsize=(15,30), sharey=True, sharex='col')
        f.patch.set_facecolor('white')
        ax =ax.flatten()
        
        
        for i, subject in enumerate(df.subjid.unique()):
            ax[i].set_title(subject)
            ax[i].set_ylabel('p(Repeat)')
            ax[i].set_xlabel('evidence to repeat')
            for j in range(bins.size-1): # prior bins
                test = df.loc[
                    (df.subjid==subject)
                    &(df.special_trial==0)
                    &(df.norm_allpriors_repalt>=bins[j])
                    &(df.norm_allpriors_repalt<=bins[j+1])
                    ].dropna(subset=['ev_repeat', 'norm_allpriors_repalt', 'rep_response'])
                plotting.psych_curve(
                    test.rep_response,
                    test.coh2repalt,
                    kwargs_plot=dict(color=cmap(np.linspace(0,1,7)[j])),
                    kwargs_error=dict(color=cmap(np.linspace(0,1,7)[j])),
                    ret_ax=ax[i]
                )
        if SAVEPLOTS:
            f.savefig(f'{savpath}1d_individual_psychometric_curves2d.svg')
        plt.show()



def d_3d(df, average=False, savpath=SAVPATH):
    """tachometric 3d, this splits all subjects. adjust subplots and figsize if single"""
    # here we will stack priors and scale them
    nanidx = df.loc[df[['dW_trans', 'dW_lat']].isna().sum(axis=1)==2].index
    df['allpriors'] = np.nansum(df[['dW_trans', 'dW_lat']].values,axis=1)
    df.loc[nanidx, 'allpriors'] = np.nan

    df['norm_allpriors'] = (
        df.allpriors
        / df.groupby('subjid').allpriors.transform(lambda x: np.max(np.abs(x)))
    )
    # transform to rep-alt space
    df['prev_response'] = df.R_response.shift(1)*2 -1
    df.loc[df.origidx==1, 'prev_response'] = np.nan
    df['norm_allpriors_repalt'] = df.prev_response * df.norm_allpriors
    df['rep_response'] = (df.prev_response * (df.R_response*2-1)+1)/2
    df['ev_repeat'] = df.avtrapz * df.prev_response

    # get x and y coords
    x = (np.linspace(-1, 1, 8)[1:] + np.linspace(-1, 1, 8)[:-1])/2
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # create fig
    if not average:
        fig=plt.figure(figsize=(15,30))
        fig.patch.set_facecolor('white')
        ax = []
        for i, subject in enumerate(df.subjid.unique()): # works fine for 18 subjects
            ax += [fig.add_subplot(6,3,i+1, projection='3d')]
            test = (
                df.loc[(df.subjid==subject)&(df.special_trial==0)] # just regular trials
                .dropna(subset=['rep_response', 'norm_allpriors_repalt', 'ev_repeat', 'avtrapz'])
            )
            mat, _ = plotting.com_heatmap( # this is simply to calc p event_i / n events in 2D
                test.ev_repeat.values,
                test.norm_allpriors_repalt.values,
                test.rep_response.values,
                annotate=False,
                return_mat=True
            )
            ax[i].set_title(subject)
            ax[i].plot_wireframe(X, Y, mat, rstride=1, cstride=1)
            ax[i].set_xlabel('evidence repeat')
            ax[i].set_ylabel('prior')
            ax[i].set_zlabel('p(repeat)')
    else: # compute average # never tested, plain average in "subject dimension" of the cube
        fig =plt.figure(figsize=(7,7))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(projection='3d')
        matrices = []
        for i, subject in enumerate(df.subjid.unique()): # works fine for 18 subjects
            tmp = (
                df.loc[(df.subjid==subject)&(df.special_trial==0)] # just regular trials
                .dropna(subset=['rep_response', 'norm_allpriors_repalt', 'ev_repeat', 'avtrapz'])
            )
            mat, _ = plotting.com_heatmap( # this is simply to calc p event_i / n events in 2D
                tmp.ev_repeat.values,
                tmp.norm_allpriors_repalt.values,
                tmp.rep_response.values,
                annotate=False,
                return_mat=True
            )
            matrices += [mat]

        ax.set_title("average across rats")
        ax.plot_wireframe(
            X, Y, np.stack(matrices).mean(axis=0), 
            rstride=1, cstride=1
            )
        ax.set_xlabel('evidence repeat')
        ax.set_ylabel('prior')
        ax.set_zlabel('p(repeat)')
    
    if SAVEPLOTS:
        fig.savefig(f'{SAVPATH}1d_psychometric_wiremeshes_repalt.svg')
    plt.show()


def e(
    df, 
    session='LE37_p4_20190309-151130',
    wlen=50,
    accu_axis=False,
    tau=1.5,
    accu_window=20,
    savpath=SAVPATH
):
    """main idea here is to show block structure and that they follow it"""
    # precalc / adjust to rep/alt space
    s = df[df.sessid==session] # session
    if accu_axis:
        f, (ax, ax2) = plt.subplots(nrows=2, sharex=True,figsize=(20,6))
    else:
        f, ax = plt.subplots(figsize=(20,4))
    s['ev_repeat'] = s.coh2 * (s.R_response*2-1).shift(1)
    s['rewside_rep'] = (s.rewside*2-1) * (s.R_response*2-1).shift(1)
    s['t++'] = (s.rep_response*2-1)*s.hithistory * s.hithistory.shift(-1) # 1
    s['t++'] = s['t++'].shift(1) # so the estimate is previous to the response and outcome!
    out = []
    prob_rep_estimate = []
    for i in range(len(s)-wlen):
        c = s.iloc[i:i+wlen]
        if len(c)!=wlen:
            print(f'wrong length in iter {i}')
        out += [opt_sig(c.rep_response.values*1, c.ev_repeat)]
        prob_rep_estimate += [(exp_decay(np.arange(1,wlen+1)[::-1], tau=tau) * c['t++']).sum()]

    RB = np.vstack(out)[:,1]
    RB = RB/np.abs(RB).max()
    # prob_rep_estimate = moving_average(
    #     np.array(prob_rep_estimate), wlen
    # )
    prob_rep_estimate = np.array(prob_rep_estimate) / np.abs(np.array(prob_rep_estimate)).max()

    for i, (side, offset) in enumerate([['left', -1], ['right', 1]]):
        tmp = s.loc[(s.rewside_rep==offset)&(s.hithistory>=0)]
        ax.eventplot(
            tmp.loc[tmp.hithistory==0,['ev_repeat','origidx']].values, 
            linelengths=0.2, 
            lineoffsets=tmp.loc[tmp.hithistory==0,'ev_repeat'].values+(offset*1.15), 
            colors='tab:red'
        )
        ax.eventplot(
            tmp.loc[tmp.hithistory==1,['ev_repeat','origidx']].values, 
            linelengths=0.2, 
            lineoffsets=tmp.loc[tmp.hithistory==1,'ev_repeat'].values+(offset*1.15), 
            colors='tab:green'
        )
    c_trial = 0
    c_color = 1
    colors = {-1: 'purple', 1: 'violet'}
    while c_trial< len(s):
        n_edge = c_trial + 80
        if n_edge>len(s):
            n_edge=len(s)
        ax.axvspan(c_trial, n_edge, alpha=0.2, color=colors[c_color])
        c_color *= -1
        c_trial=n_edge



    
    ax.plot(np.arange(RB.size)+wlen, RB, label='Repeating bias')
    ax.plot(np.arange(RB.size)+wlen, prob_rep_estimate, label='local estimate p(rep)')
    ax.set_facecolor('white')
    sns.despine(ax=ax, left=True, bottom=True)
    if accu_axis:
        sns.despine(ax=ax2, left=True, bottom=True)
        ax2.set_facecolor('white')
        for (i,label, color) in [[-1, 'alt accu', 'b'],[1, 'rep accu', 'r']]:
            cdat = s.loc[s.rewside_rep==i]
            ax2.plot(
                cdat.origidx,
                (
                    cdat.hithistory
                    .rolling(accu_window, min_periods=1)
                    .mean().values
                ),
                c=color, marker='o', label=label, markersize=2)
    
    ax2.set_ylim(.4, 1.05)
    ax2.legend(frameon=False, fancybox=False)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(0,s.origidx.max())
    ax.axhline(0, ls=':', color='gray', zorder=0)
    ax.legend(loc='upper left')
    ax.set_xlabel('trial index')

    if SAVPATH:
        f.savefig(f'{savpath}1e_task_blocks_n_biases.svg')
    plt.show()