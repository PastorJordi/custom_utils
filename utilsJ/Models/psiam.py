"""attempting to port Lluís Hernández code"""
# TODO: add variant using native random module!
import numpy as np
import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
from numba import jit # testing this for the first time :)
import random

def D2M_simulation(
    t_c,c,b,d,v_u,a_u,t_0_u,v,a_e,z_e,t_0_e,StimOnset,prop,dt,n,x_cont_inf, seed=None,
    block_len=80, prep_rep=.8, prep_alt=.2
):
    """this one should be direct transcript, later perhaps try to tune performance
    ---------------------------------
    %D2M simulations under parameters
    %"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
    %for the computation of pdfs is "err", and the time-sep and the number of
    %realizations for the simulation of the urgency integrator's accuracy are
    %respectively "dt" and "n".
    %
    %t_c: time of truncation (quan truncar els RTs dels contaminants)
    %c : scalar; prob de que 1 trial sigui contaminant (1-p(PSIAM))
    %b : scalar; (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    %d : scalar: part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    %v_u:  scalar, intercept de l'urgency
    %a_u: threshold de l'action initation
    %t_0_u: temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    %v = [constant * stim str, aka drift real] drift de l'evidence [pot no ser scalar] ~ constant multiplicativa (llavors es multiplica per cada stim str)
    % si fas varies coherencies passa vector. ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    %a_e: threshold evidence (scalar) entre 0 i qualsevol bound
    %z_e: starting point evidence (de -1 a 1)
    %t_0_e: temporal onset evidence acumulator sol tenir valors de 0.35s (300 fixation + 50)
    %StimOnset: quna comença l'stimulus, 0.3
    %prop: proportion [struct] conté v_trial (weight del glm pel trial index), v_rewardsize, v_cumrew [weigths] -> scalar
    % i també els X_t [trial, rewardsize, cumrew] -> vectors
    %dt
    %n: # simulacions
    %x_cont_inf
    % accu [0~1] correcte o incorrecte
    % bound[0~1]: 0=proactive, 1: reactive
    % s: SD del starting point Z_e


    returns RT, v, Accu, bound, s

    """
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    RT = np.empty((n)) * np.nan
    Accu = RT.copy()
    bound = np.zeros(n)
    # Block structure (for stimulus-coding of transition bias in starting offset z_e)
    # prep_rep = .8
    # prep_alt = .2
    # block_len = 80 
    # added them as kwargs

    Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2).astype(int)
    if not rng.choice([0,1]): # if true, flips and first block is alternating!
        Block_vec = (Block_vec-1)**2
    Rep_vec = Block_vec*(rng.binomial(1, prep_rep, n)*2 - 1) + \
        (Block_vec-1)**2 * (rng.binomial(1, prep_alt, n)*2 - 1)
    Block_vec = 2*Block_vec-1 # to -1~1 space

    # Markovian stimulus
    Categ_vec = [rng.choice([-1,1])]
    for i in range(1,n):
        #Categ_vec(i) = Categ_vec(i-1)*Rep_vec(i);
        Categ_vec += [Categ_vec[-1]*Rep_vec[i]]

    Categ_vec = np.array(Categ_vec)
    # % Difficulty
    if len(v)>1:
        v[v==v.min()]=0 #; %to force zero coherence as random stimulus.
    # end
    Dif = rng.choice([0,1,2,3], size=n) # idk which is which # consider switching to 0, .25, .5, 1

    # % Response category
    # Resp_vec=zeros(size(Block_vec)); Resp_vec(1)=2*binornd(1,.5)-1;
    Resp_vec = np.zeros(Block_vec.size)
    Resp_vec[0] = rng.choice([-1,1])

    # % Modulation with trial index
    # rr = rand(1,n-1);
    rr = rng.uniform(size=n-1)
    ss = rng.uniform(size=n-1)
    uu = rng.uniform(size=n-1)

    # % D2M
    mu = (z_e+1)/2 # 0~1 space (assuming z_e was in -1~1) 
    #s = 0.06 # std of beta distribution
    var = 0.015
    # a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
    # B = a*(1/mu-1); % parameter b for the beta distribution
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    B = a* (1 / mu - 1)

    if z_e == 0:
        zz = 0
    else:
        zz=1

    # timer
    for i in tqdm.tqdm_notebook(range(1,n)):
        vv_u = v_u + prop['v_trial'] * prop['trial'][1, (prop['trial'][0,:]<rr[i-1]).sum()]
        vv_u += prop['v_rewsize'] * prop['rewardsize'][1, (prop['rewardsize'][0,:]<ss[i-1]).sum()]
        vv_u += prop['v_cumrew'] * prop['cumreward'][1, (prop['cumreward'][0,:]<uu[i-1]).sum()]

        l = Dif[i]
        v_e = v[l]*Categ_vec[i]
        x_u = 0
        x_e = zz*(rng.beta(a,B)*2-1)*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = np.floor(t_0_u/dt) #; % initialize DDM iteration
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + vv_u*dt + dt**0.5*rng.normal(size=1)[0] #; % urgency integrator
            k = k+1 
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + vv_u*dt + dt**0.5*rng.normal(size=1)[0] # % urgency integrator
            x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # % evidence integrator
            k = k+1 # ; % time step counter
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        bound[i] = x_u<a_u # ; % whether the trial hit the evidence bound or the urgency bound
        if (not bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # ; % evidence integrator
                k = k+1 #; % time step counter
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = (Resp_vec[i]*Categ_vec[i]+1)/2 # ;

    return RT, v, Accu, bound, var**0.5

def D2M_simulation_parallel(
    args, kwargs={}
    
):
    """this one should be direct transcript, later perhaps try to tune performance
    ---------------------------------
    %D2M simulations under parameters
    %"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
    %for the computation of pdfs is "err", and the time-sep and the number of
    %realizations for the simulation of the urgency integrator's accuracy are
    %respectively "dt" and "n".
    %
    %t_c: time of truncation (quan truncar els RTs dels contaminants)
    %c : scalar; prob de que 1 trial sigui contaminant (1-p(PSIAM))
    %b : scalar; (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    %d : scalar: part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    %v_u:  scalar, intercept de l'urgency
    %a_u: threshold de l'action initation
    %t_0_u: temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    %v = [constant * stim str, aka drift real] drift de l'evidence [pot no ser scalar] ~ constant multiplicativa (llavors es multiplica per cada stim str)
    % si fas varies coherencies passa vector. ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    %a_e: threshold evidence (scalar) entre 0 i qualsevol bound
    %z_e: starting point evidence (de -1 a 1)
    %t_0_e: temporal onset evidence acumulator sol tenir valors de 0.35s (300 fixation + 50)
    %StimOnset: quna comença l'stimulus, 0.3
    %prop: proportion [struct] conté v_trial (weight del glm pel trial index), v_rewardsize, v_cumrew [weigths] -> scalar
    % i també els X_t [trial, rewardsize, cumrew] -> vectors
    %dt
    %n: # simulacions
    %x_cont_inf
    % accu [0~1] correcte o incorrecte
    % bound[0~1]: 0=proactive, 1: reactive
    % s: SD del starting point Z_e


    returns RT, v, Accu, bound, s

    """
    t_c,c,b,d,v_u,a_u,t_0_u,v,a_e,z_e,t_0_e,StimOnset,prop,dt,n,x_cont_inf, seed = args
    # unpack prop

    block_len= kwargs.get('block_len', 80)
    prep_rep = kwargs.get('prep_rep', .8)
    prep_alt= kwargs.get('prep_alt', .2)

    
    rng = np.random.RandomState(seed=seed)
    
    RT = np.empty((n)) * np.nan
    Accu = RT.copy()
    bound = np.zeros(n)


    Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2).astype(int)
    if not rng.choice([0,1]): # if true, flips and first block is alternating!
        Block_vec = (Block_vec-1)**2
    Rep_vec = Block_vec*(rng.binomial(1, prep_rep, n)*2 - 1) + \
        (Block_vec-1)**2 * (rng.binomial(1, prep_alt, n)*2 - 1)
    Block_vec = 2*Block_vec-1 # to -1~1 space

    # Markovian stimulus
    Categ_vec = [rng.choice([-1,1])]
    for i in range(1,n):
        #Categ_vec(i) = Categ_vec(i-1)*Rep_vec(i);
        Categ_vec += [Categ_vec[-1]*Rep_vec[i]]

    Categ_vec = np.array(Categ_vec)
    # % Difficulty
    if len(v)>1:
        v[v==v.min()]=0 #; %to force zero coherence as random stimulus.
    # end
    Dif = rng.choice([0,1,2,3], size=n) # idk which is which # consider switching to 0, .25, .5, 1

    # % Response category
    # Resp_vec=zeros(size(Block_vec)); Resp_vec(1)=2*binornd(1,.5)-1;
    Resp_vec = np.zeros(Block_vec.size)
    Resp_vec[0] = rng.choice([-1,1])

    # % Modulation with trial index
    # rr = rand(1,n-1);
    rr = rng.uniform(size=n-1)
    ss = rng.uniform(size=n-1)
    uu = rng.uniform(size=n-1)

    # % D2M
    mu = (z_e+1)/2 # 0~1 space (assuming z_e was in -1~1) 
    #s = 0.06 # std of beta distribution
    var = kwargs.get('var',0.015)
    # a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
    # B = a*(1/mu-1); % parameter b for the beta distribution
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    B = a* (1 / mu - 1)

    if z_e == 0:
        zz = 0
    else:
        zz=1


    # precalc those random normal as matrices for speed
    max_step_u_alone = int((t_0_e - t_0_u)/dt)
    # afterwards try to mke it vectorized with cumsum!
    offset_k = int(t_0_u/dt)

    # timer
    # unpacking into scalars arrays so we avoid using dicts with numba
    v_trial = prop['v_trial']
    trial=prop['trial']
    v_rewsize = prop['v_rewsize']
    rewardsize = prop['rewardsize']
    v_cumrew = prop['v_cumrew']
    cumreward = prop['cumreward']
    for i in range(1,n):
        vv_u = v_u + v_trial * trial[1, (trial[0,:]<rr[i-1]).sum()]
        vv_u += v_rewsize * rewardsize[1, (rewardsize[0,:]<ss[i-1]).sum()]
        vv_u += v_cumrew * cumreward[1, (cumreward[0,:]<uu[i-1]).sum()]

        l = Dif[i]
        v_e = v[l]*Categ_vec[i]
        x_u = 0
        x_e = zz*(rng.beta(a,B)*2-1)*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = offset_k #; % initialize DDM iteration
        precomp_normal= rng.normal(size=max_step_u_alone)
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[k-offset_k] # rng.normal(size=1)[0] #; % urgency integrator
            k = k+1 
        precomp_normal = rng.normal(size=20000).reshape(2,-1)
        ii = 0
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[0,ii] # rng.normal(size=1)[0] # % urgency integrator
            x_e = x_e + v_e*dt + dt**0.5*precomp_normal[1,ii] # rng.normal(size=1)[0] # % evidence integrator
            k = k+1 # ; % time step counter
            ii += 1
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        bound[i] = x_u<a_u # ; % whether the trial hit the evidence bound or the urgency bound
        if (not bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # ; % evidence integrator
                k = k+1 #; % time step counter
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = (Resp_vec[i]*Categ_vec[i]+1)/2 # ;

    # some shit missing
    return RT, v, Accu, bound, np.repeat(var**0.5, RT.size)

#@jit(nopython=True)
@jit()
def D2M_simulation_numba(
    t_c,c,b,d,v_u,a_u,t_0_u,v,a_e,z_e,t_0_e,StimOnset,prop,dt,n,x_cont_inf, seed    
):
    """this one should be direct transcript, later perhaps try to tune performance
    ---------------------------------
    %D2M simulations under parameters
    %"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
    %for the computation of pdfs is "err", and the time-sep and the number of
    %realizations for the simulation of the urgency integrator's accuracy are
    %respectively "dt" and "n".
    %
    %t_c: time of truncation (quan truncar els RTs dels contaminants)
    %c : scalar; prob de que 1 trial sigui contaminant (1-p(PSIAM))
    %b : scalar; (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    %d : scalar: part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    %v_u:  scalar, intercept de l'urgency
    %a_u: threshold de l'action initation
    %t_0_u: temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    %v = [constant * stim str, aka drift real] drift de l'evidence [pot no ser scalar] ~ constant multiplicativa (llavors es multiplica per cada stim str)
    % si fas varies coherencies passa vector. ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    %a_e: threshold evidence (scalar) entre 0 i qualsevol bound
    %z_e: starting point evidence (de -1 a 1)
    %t_0_e: temporal onset evidence acumulator sol tenir valors de 0.35s (300 fixation + 50)
    %StimOnset: quna comença l'stimulus, 0.3
    %prop: proportion [struct] conté v_trial (weight del glm pel trial index), v_rewardsize, v_cumrew [weigths] -> scalar
    % i també els X_t [trial, rewardsize, cumrew] -> vectors
    %dt
    %n: # simulacions
    %x_cont_inf
    % accu [0~1] correcte o incorrecte
    % bound[0~1]: 0=proactive, 1: reactive
    % s: SD del starting point Z_e


    returns RT, v, Accu, bound, s

    """

    block_len= 80
    prep_rep = .8
    prep_alt= .2

    
    rng = np.random.RandomState(seed=seed)
    
    RT = np.empty((n)) * np.nan
    Accu = RT.copy()
    bound = np.zeros(n)


    Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2).astype(int)
    if not rng.choice([0,1]): # if true, flips and first block is alternating!
        Block_vec = (Block_vec-1)**2
    Rep_vec = Block_vec*(rng.binomial(1, prep_rep, n)*2 - 1) + \
        (Block_vec-1)**2 * (rng.binomial(1, prep_alt, n)*2 - 1)
    Block_vec = 2*Block_vec-1 # to -1~1 space

    # Markovian stimulus
    Categ_vec = [rng.choice([-1,1])]
    for i in range(1,n):
        #Categ_vec(i) = Categ_vec(i-1)*Rep_vec(i);
        Categ_vec += [Categ_vec[-1]*Rep_vec[i]]

    Categ_vec = np.array(Categ_vec)
    # % Difficulty
    if len(v)>1:
        v[v==v.min()]=0 #; %to force zero coherence as random stimulus.
    # end
    Dif = rng.choice([0,1,2,3], size=n) # idk which is which # consider switching to 0, .25, .5, 1

    # % Response category
    # Resp_vec=zeros(size(Block_vec)); Resp_vec(1)=2*binornd(1,.5)-1;
    Resp_vec = np.zeros(Block_vec.size)
    Resp_vec[0] = rng.choice([-1,1])

    # % Modulation with trial index
    # rr = rand(1,n-1);
    rr = rng.uniform(size=n-1)
    ss = rng.uniform(size=n-1)
    uu = rng.uniform(size=n-1)

    # % D2M
    mu = (z_e+1)/2 # 0~1 space (assuming z_e was in -1~1) 
    #s = 0.06 # std of beta distribution
    var = 0.015
    # a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
    # B = a*(1/mu-1); % parameter b for the beta distribution
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    B = a* (1 / mu - 1)

    if z_e == 0:
        zz = 0
    else:
        zz=1


    # precalc those random normal as matrices for speed
    max_step_u_alone = int((t_0_e - t_0_u)/dt)
    # afterwards try to mke it vectorized with cumsum!
    offset_k = int(t_0_u/dt)

    # timer
    # unpacking into scalars arrays so we avoid using dicts with numba
    v_trial = prop['v_trial']
    trial=prop['trial']
    v_rewsize = prop['v_rewsize']
    rewardsize = prop['rewardsize']
    v_cumrew = prop['v_cumrew']
    cumreward = prop['cumreward']
    for i in range(1,n):
        vv_u = v_u + v_trial * trial[1, (trial[0,:]<rr[i-1]).sum()]
        vv_u += v_rewsize * rewardsize[1, (rewardsize[0,:]<ss[i-1]).sum()]
        vv_u += v_cumrew * cumreward[1, (cumreward[0,:]<uu[i-1]).sum()]

        l = Dif[i]
        v_e = v[l]*Categ_vec[i]
        x_u = 0
        x_e = zz*(rng.beta(a,B)*2-1)*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = offset_k #; % initialize DDM iteration
        precomp_normal= rng.normal(size=max_step_u_alone)
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[k-offset_k] # rng.normal(size=1)[0] #; % urgency integrator
            k = k+1 
        precomp_normal = rng.normal(size=20000).reshape(2,-1)
        ii = 0
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[0,ii] # rng.normal(size=1)[0] # % urgency integrator
            x_e = x_e + v_e*dt + dt**0.5*precomp_normal[1,ii] # rng.normal(size=1)[0] # % evidence integrator
            k = k+1 # ; % time step counter
            ii += 1
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        bound[i] = x_u<a_u # ; % whether the trial hit the evidence bound or the urgency bound
        if (not bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # ; % evidence integrator
                k = k+1 #; % time step counter
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = (Resp_vec[i]*Categ_vec[i]+1)/2 # ;

    # some shit missing
    return RT, v, Accu, bound, np.repeat(var**0.5, RT.size)

def parallel_wrapper(params, workers=8, seeds=np.arange(80), jobsperworker=10):

    # adapt params to n
    original_n = params[-2]
    params[-2] =  1 + int(params[-2]/(workers*jobsperworker))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = [
            executor.submit(
                D2M_simulation_parallel, params+[x]
                ) for x in seeds]
        for job in tqdm.tqdm_notebook(as_completed(jobs), total=jobsperworker*workers):
            out += [job.result()]

    RT = np.concatenate(
            [out[x][0] for x in range(len(out))]
    )
    v = np.concatenate(
            [out[x][1] for x in range(len(out))]
    )
    Accu = np.concatenate(
            [out[x][2] for x in range(len(out))]
    )
    bound = np.concatenate(
            [out[x][3] for x in range(len(out))]
    )
    s = np.concatenate(
        [out[x][4] for x in range(len(out))]
    )
    return RT, v, Accu, bound, s



###
@jit(nopython=True)
def internal_loop(
    v_trial, trial, v_rewsize, rewardsize, v_cumrew, cumreward, rng, Dif,
    v, Categ_vec, zz, a, B, a_e, Resp_vec, Block_vec
):
    for i in range(1,n):
        vv_u = v_u + v_trial * trial[1, (trial[0,:]<rr[i-1]).sum()]
        vv_u += v_rewsize * rewardsize[1, (rewardsize[0,:]<ss[i-1]).sum()]
        vv_u += v_cumrew * cumreward[1, (cumreward[0,:]<uu[i-1]).sum()]

        l = Dif[i]
        v_e = v[l]*Categ_vec[i]
        x_u = 0
        x_e = zz*(rng.beta(a,B)*2-1)*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = offset_k #; % initialize DDM iteration
        precomp_normal= rng.normal(size=max_step_u_alone)
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[k-offset_k] # rng.normal(size=1)[0] #; % urgency integrator
            k = k+1 
        precomp_normal = rng.normal(size=20000).reshape(2,-1)
        ii = 0
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[0,ii] # rng.normal(size=1)[0] # % urgency integrator
            x_e = x_e + v_e*dt + dt**0.5*precomp_normal[1,ii] # rng.normal(size=1)[0] # % evidence integrator
            k = k+1 # ; % time step counter
            ii += 1
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        bound[i] = x_u<a_u # ; % whether the trial hit the evidence bound or the urgency bound
        if (not bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # ; % evidence integrator
                k = k+1 #; % time step counter
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = (Resp_vec[i]*Categ_vec[i]+1)/2 # ;

        return RT, v, Accu, bound, s


def D2M_simulation_numba2(
    t_c,c,b,d,v_u,a_u,t_0_u,v,a_e,z_e,t_0_e,StimOnset,prop,dt,n,x_cont_inf, seed    
):
    """this one should be direct transcript, later perhaps try to tune performance
    ---------------------------------
    %D2M simulations under parameters
    %"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
    %for the computation of pdfs is "err", and the time-sep and the number of
    %realizations for the simulation of the urgency integrator's accuracy are
    %respectively "dt" and "n".
    %
    %t_c: time of truncation (quan truncar els RTs dels contaminants)
    %c : scalar; prob de que 1 trial sigui contaminant (1-p(PSIAM))
    %b : scalar; (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    %d : scalar: part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    %v_u:  scalar, intercept de l'urgency
    %a_u: threshold de l'action initation
    %t_0_u: temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    %v = [constant * stim str, aka drift real] drift de l'evidence [pot no ser scalar] ~ constant multiplicativa (llavors es multiplica per cada stim str)
    % si fas varies coherencies passa vector. ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    %a_e: threshold evidence (scalar) entre 0 i qualsevol bound
    %z_e: starting point evidence (de -1 a 1)
    %t_0_e: temporal onset evidence acumulator sol tenir valors de 0.35s (300 fixation + 50)
    %StimOnset: quna comença l'stimulus, 0.3
    %prop: proportion [struct] conté v_trial (weight del glm pel trial index), v_rewardsize, v_cumrew [weigths] -> scalar
    % i també els X_t [trial, rewardsize, cumrew] -> vectors
    %dt
    %n: # simulacions
    %x_cont_inf
    % accu [0~1] correcte o incorrecte
    % bound[0~1]: 0=proactive, 1: reactive
    % s: SD del starting point Z_e


    returns RT, v, Accu, bound, s

    """

    block_len= 80
    prep_rep = .8
    prep_alt= .2

    
    rng = np.random.RandomState(seed=seed)
    
    RT = np.empty((n)) * np.nan
    Accu = RT.copy()
    bound = np.zeros(n)


    Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2).astype(int)
    if not rng.choice([0,1]): # if true, flips and first block is alternating!
        Block_vec = (Block_vec-1)**2
    Rep_vec = Block_vec*(rng.binomial(1, prep_rep, n)*2 - 1) + \
        (Block_vec-1)**2 * (rng.binomial(1, prep_alt, n)*2 - 1)
    Block_vec = 2*Block_vec-1 # to -1~1 space

    # Markovian stimulus
    Categ_vec = [rng.choice([-1,1])]
    for i in range(1,n):
        #Categ_vec(i) = Categ_vec(i-1)*Rep_vec(i);
        Categ_vec += [Categ_vec[-1]*Rep_vec[i]]

    Categ_vec = np.array(Categ_vec)
    # % Difficulty
    if len(v)>1:
        v[v==v.min()]=0 #; %to force zero coherence as random stimulus.
    # end
    Dif = rng.choice([0,1,2,3], size=n) # idk which is which # consider switching to 0, .25, .5, 1

    # % Response category
    # Resp_vec=zeros(size(Block_vec)); Resp_vec(1)=2*binornd(1,.5)-1;
    Resp_vec = np.zeros(Block_vec.size)
    Resp_vec[0] = rng.choice([-1,1])

    # % Modulation with trial index
    # rr = rand(1,n-1);
    rr = rng.uniform(size=n-1)
    ss = rng.uniform(size=n-1)
    uu = rng.uniform(size=n-1)

    # % D2M
    mu = (z_e+1)/2 # 0~1 space (assuming z_e was in -1~1) 
    #s = 0.06 # std of beta distribution
    var = 0.015
    # a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
    # B = a*(1/mu-1); % parameter b for the beta distribution
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    B = a* (1 / mu - 1)

    if z_e == 0:
        zz = 0
    else:
        zz=1


    # precalc those random normal as matrices for speed
    max_step_u_alone = int((t_0_e - t_0_u)/dt)
    # afterwards try to mke it vectorized with cumsum!
    offset_k = int(t_0_u/dt)

    # timer
    # unpacking into scalars arrays so we avoid using dicts with numba
    v_trial = prop['v_trial']
    trial=prop['trial']
    v_rewsize = prop['v_rewsize']
    rewardsize = prop['rewardsize']
    v_cumrew = prop['v_cumrew']
    cumreward = prop['cumreward']
    for i in range(1,n):
        vv_u = v_u + v_trial * trial[1, (trial[0,:]<rr[i-1]).sum()]
        vv_u += v_rewsize * rewardsize[1, (rewardsize[0,:]<ss[i-1]).sum()]
        vv_u += v_cumrew * cumreward[1, (cumreward[0,:]<uu[i-1]).sum()]

        l = Dif[i]
        v_e = v[l]*Categ_vec[i]
        x_u = 0
        x_e = zz*(rng.beta(a,B)*2-1)*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = offset_k #; % initialize DDM iteration
        precomp_normal= rng.normal(size=max_step_u_alone)
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[k-offset_k] # rng.normal(size=1)[0] #; % urgency integrator
            k = k+1 
        precomp_normal = rng.normal(size=20000).reshape(2,-1)
        ii = 0
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + vv_u*dt + dt**0.5*precomp_normal[0,ii] # rng.normal(size=1)[0] # % urgency integrator
            x_e = x_e + v_e*dt + dt**0.5*precomp_normal[1,ii] # rng.normal(size=1)[0] # % evidence integrator
            k = k+1 # ; % time step counter
            ii += 1
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        bound[i] = x_u<a_u # ; % whether the trial hit the evidence bound or the urgency bound
        if (not bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # ; % evidence integrator
                k = k+1 #; % time step counter
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = (Resp_vec[i]*Categ_vec[i]+1)/2 # ;

    # some shit missing
    return RT, v, Accu, bound, np.repeat(var**0.5, RT.size)




def D2M_simulation_native(
    t_c,c,b,d,v_u,a_u,t_0_u,v,a_e,z_e,t_0_e,StimOnset,prop,dt,n,x_cont_inf, seed=None,
    block_len=80, prep_rep=.8, prep_alt=.2):
    """try using native random despite no seed
    ---------------------------------
    %D2M simulations under parameters
    %"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
    %for the computation of pdfs is "err", and the time-sep and the number of
    %realizations for the simulation of the urgency integrator's accuracy are
    %respectively "dt" and "n".
    %
    %t_c: time of truncation (quan truncar els RTs dels contaminants)
    %c : scalar; prob de que 1 trial sigui contaminant (1-p(PSIAM))
    %b : scalar; (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    %d : scalar: part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    %v_u:  scalar, intercept de l'urgency
    %a_u: threshold de l'action initation
    %t_0_u: temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    %v = [constant * stim str, aka drift real] drift de l'evidence [pot no ser scalar] ~ constant multiplicativa (llavors es multiplica per cada stim str)
    % si fas varies coherencies passa vector. ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    %a_e: threshold evidence (scalar) entre 0 i qualsevol bound
    %z_e: starting point evidence (de -1 a 1)
    %t_0_e: temporal onset evidence acumulator sol tenir valors de 0.35s (300 fixation + 50)
    %StimOnset: quna comença l'stimulus, 0.3
    %prop: proportion [struct] conté v_trial (weight del glm pel trial index), v_rewardsize, v_cumrew [weigths] -> scalar
    % i també els X_t [trial, rewardsize, cumrew] -> vectors
    %dt
    %n: # simulacions
    %x_cont_inf
    % accu [0~1] correcte o incorrecte
    % bound[0~1]: 0=proactive, 1: reactive
    % s: SD del starting point Z_e


    returns RT, v, Accu, bound, s
    """
    raise NotImplemented
    if seed is not None:
        rng = np.random.RandomState(seed=seed)
    else:
        rng = np.random.RandomState()

    RT = np.empty((n)) * np.nan
    Accu = RT.copy()
    bound = np.zeros(n)
    # Block structure (for stimulus-coding of transition bias in starting offset z_e)
    # prep_rep = .8
    # prep_alt = .2
    # block_len = 80 
    # added them as kwargs

    Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2).astype(int)
    if not rng.choice([0,1]): # if true, flips and first block is alternating!
        Block_vec = (Block_vec-1)**2
    Rep_vec = Block_vec*(rng.binomial(1, prep_rep, n)*2 - 1) + \
        (Block_vec-1)**2 * (rng.binomial(1, prep_alt, n)*2 - 1)
    Block_vec = 2*Block_vec-1 # to -1~1 space

    # Markovian stimulus
    Categ_vec = [rng.choice([-1,1])]
    for i in range(1,n):
        #Categ_vec(i) = Categ_vec(i-1)*Rep_vec(i);
        Categ_vec += [Categ_vec[-1]*Rep_vec[i]]

    Categ_vec = np.array(Categ_vec)
    # % Difficulty
    if len(v)>1:
        v[v==v.min()]=0 #; %to force zero coherence as random stimulus.
    # end
    Dif = rng.choice([0,1,2,3], size=n) # idk which is which # consider switching to 0, .25, .5, 1

    # % Response category
    # Resp_vec=zeros(size(Block_vec)); Resp_vec(1)=2*binornd(1,.5)-1;
    Resp_vec = np.zeros(Block_vec.size)
    Resp_vec[0] = rng.choice([-1,1])

    # % Modulation with trial index
    # rr = rand(1,n-1);
    rr = rng.uniform(size=n-1)
    ss = rng.uniform(size=n-1)
    uu = rng.uniform(size=n-1)

    # % D2M
    mu = (z_e+1)/2 # 0~1 space (assuming z_e was in -1~1) 
    #s = 0.06 # std of beta distribution
    var = 0.015
    # a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
    # B = a*(1/mu-1); % parameter b for the beta distribution
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    B = a* (1 / mu - 1)

    if z_e == 0:
        zz = 0
    else:
        zz=1

    # timer
    for i in tqdm.tqdm_notebook(range(1,n)):
        vv_u = v_u + prop['v_trial'] * prop['trial'][1, (prop['trial'][0,:]<rr[i-1]).sum()]
        vv_u += prop['v_rewsize'] * prop['rewardsize'][1, (prop['rewardsize'][0,:]<ss[i-1]).sum()]
        vv_u += prop['v_cumrew'] * prop['cumreward'][1, (prop['cumreward'][0,:]<uu[i-1]).sum()]

        l = Dif[i]
        v_e = v[l]*Categ_vec[i]
        x_u = 0
        x_e = zz*(rng.beta(a,B)*2-1)*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = np.floor(t_0_u/dt) #; % initialize DDM iteration
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + vv_u*dt + dt**0.5*rng.normal(size=1)[0] #; % urgency integrator
            k = k+1 
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + vv_u*dt + dt**0.5*rng.normal(size=1)[0] # % urgency integrator
            x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # % evidence integrator
            k = k+1 # ; % time step counter
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        bound[i] = x_u<a_u # ; % whether the trial hit the evidence bound or the urgency bound
        if (not bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + v_e*dt + dt**0.5*rng.normal(size=1)[0] # ; % evidence integrator
                k = k+1 #; % time step counter
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = (Resp_vec[i]*Categ_vec[i]+1)/2 # ;

    return RT, v, Accu, bound, var**0.5