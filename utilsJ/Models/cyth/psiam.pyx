#cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython

## TODO: remove double declarations of vars (py+c) when they are not returned just use cdef 

## check this for clarificatio in dtypes https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython
# adapt inputs, for efficient indexing! docu here! https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
# like def naive_convolve(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
ctypedef cnp.int_t DTYPE_int
ctypedef cnp.float_t DTYPE_float

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def D2M_simulation_c(
    double t_c, # 1.3 time of truncation (quan truncar els RTs dels contaminants)
    double c, # prob de que 1 trial sigui contaminant (1-p(PSIAM)
    double b, # (invers temps característic / factor multiplicatiu exponent contaminant) e^(-b*t) ~> t:RT
    double d, # part llarga dels contaminants (proporció contaminants 1: exponencial 0: tots uniformes)
    double v_u, # intercept de l'urgency
    double a_u, # threshold de l'action initation
    double t_0_u, # temporal onset del action initiation (s) # respecte fixation onset (-0.1 to 1)
    cnp.ndarray[DTYPE_float, ndim=1] v, #[constant * stim str, aka drift real] drift de l'evidence [pot no ser scalar] 
    # ~ constant multiplicativa (llavors es multiplica per cada stim str) si fas varies coherencies passa vector. 
    # ~ valor més petit es força a 0 per defecte # promig(coh0) => en absolut potser no cau a 0!
    double a_e, # threshold evidence (scalar) entre 0 i qualsevol bound
    double z_e, # starting point evidence (de -1 a 1) ?
    double t_0_e, # temporal onset evidence acumulator. usually ~  0.35s (300 fixation + 50)
    double StimOnset, # when is the stimulus starting (0.3)
    # unpacking prop: prop: proportion [struct] conté v_trial (weight del glm pel trial index), v_rewardsize, v_cumrew [weigths] -> scalar
    # % i també els X_t [trial, rewardsize, cumrew] -> vectors
    cnp.ndarray[DTYPE_float, ndim=2] trial, # removing prop from naming 
    double v_trial,
    cnp.ndarray[DTYPE_float, ndim=2] rewardsize,
    double v_rewsize,
    cnp.ndarray[DTYPE_float, ndim=2] cumreward,
    double v_cumrew,
    double dt, # step time
    long n, # ntrials to simulate
    double x_cont_inf, # ?
    long seed, # seed for random generator
    long block_len, 
    double prep_rep, 
    double prep_alt
):
    """this one should be direct transcript, later perhaps try to tune performance
    ---------------------------------
    %D2M simulations under parameters
    %"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
    %for the computation of pdfs is "err", and the time-sep and the number of
    %realizations for the simulation of the urgency integrator's accuracy are
    %respectively "dt" and "n".

    % accu [0~1] correcte o incorrecte
    % bound[0~1]: 0=proactive, 1: reactive
    % s: SD del starting point Z_e


    returns RT, v, Accu, bound, s

    """
    # set seed in random generator
    rng = np.random.RandomState(seed=seed)

    # init vars
    cdef cnp.ndarray[DTYPE_float, ndim=1] RT = np.repeat([-1.], n) # float
    cdef cnp.ndarray[DTYPE_int, ndim=1] Accu = np.zeros((n), dtype=np.int_)
    cdef cnp.ndarray[DTYPE_int, ndim=1] Bound = np.zeros((n), dtype=np.int_)
    cdef long i, k, ii
    cdef int k_0 = int(np.floor(t_0_u/dt)) # we store this in mem so we avoid 1 useless calc
    cdef double vv_u, v_e, zz, mu, a ,B, x_u, x_e
    cdef double var = 0.015
    cdef long max_steps = int((t_c+abs(t_0_u))/dt)
    cdef double curr_vvu, curr_vve, T



    # vectorized python
    cdef cnp.ndarray[DTYPE_int, ndim=1] Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len),2).astype(np.int_)
    #Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2)
    # Block_vec = np.mod(np.ceil(np.arange(1,n+1)/block_len), 2).astype(np.int_t)
    if rng.choice([True,False]): # if true, flips and first block is alternating!
        Block_vec = (Block_vec-1)**2
    Rep_vec = (Block_vec*(rng.binomial(1, prep_rep, n)*2 - 1) + \
        (Block_vec-1)**2 * (rng.binomial(1, prep_alt, n)*2 - 1)).astype(np.int_)
    Block_vec = (2*Block_vec-1).astype(np.int_) # to -1~1 space
    cdef int[:] Rep_view = Rep_vec

    # Markovian stimulus
    cdef cnp.ndarray[DTYPE_int, ndim=1 ]Categ_vec = np.zeros(n, dtype=np.int_)
    Categ_vec[0] = rng.choice(np.array([-1,1], dtype=np.int_)) # 0th trial side
    
    for i in range(1,n):
        Categ_vec[i] = Categ_vec[i-1]*Rep_view[i]

    # % Difficulty
    if len(v)>1:
        v[v==v.min()]=0 #; %to force zero coherence as random stimulus.
    # end
    cdef cnp.ndarray[DTYPE_int] Dif = rng.choice([0,1,2,3], size=n) # idk which is which # consider switching to 0, .25, .5, 1

    # % Response category
    cdef cnp.ndarray[DTYPE_int, ndim=1] Resp_vec = np.zeros(Block_vec.size, dtype=np.int_)
    Resp_vec[0] = rng.choice([-1,1])

    # % Modulation with trial index
    cdef cnp.ndarray[DTYPE_float, ndim=1] rr = rng.uniform(size=n-1)
    cdef cnp.ndarray[DTYPE_float, ndim=1] ss = rng.uniform(size=n-1)
    cdef cnp.ndarray[DTYPE_float, ndim=1] uu = rng.uniform(size=n-1)

    # % D2M
    mu = (z_e+1)/2 # 0~1 space (assuming z_e was in -1~1) 
    #s = 0.06 # std of beta distribution
    # var = 0.015 # defined above, when initializing variables
    # a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
    # B = a*(1/mu-1); % parameter b for the beta distribution
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    B = a* (1 / mu - 1)

    if z_e == 0:
        zz = 0
    else:
        zz=1

    ### precompute beta and normals randomness here
    cdef cnp.ndarray[DTYPE_float, ndim=1] betas = rng.beta(a,B, size=n)*2-1 # one per trial (n)
    cdef cnp.ndarray[DTYPE_float, ndim=2] normals = rng.normal(size=(2,max_steps)) * (dt**0.5) # scaling here sqrt(dt)
    
    # trial loop
    for i in range(1,n):
        # cdef np.ndarray nrandomness = rng.normal(size=()) # decide how many random numbers to precompute
        vv_u = v_u + v_trial * trial[1, (trial[0,:]<rr[i-1]).sum()] # need to transform these
        vv_u += v_rewsize * rewardsize[1, (rewardsize[0,:]<ss[i-1]).sum()]
        vv_u += v_cumrew * cumreward[1, (cumreward[0,:]<uu[i-1]).sum()]

        v_e = v[Dif[i]]*Categ_vec[i]
        x_u = 0 # initialize urgency
        x_e = zz*betas[i]*a_e # ; # initialize Decision Variable with std so that performance is around 60%
        x_e = Resp_vec[i-1]*Block_vec[i]*x_e # ; %aftercorrect fit, so that it is really categ
        k = k_0 #; % initialize DDM iteration
        ii=0
        curr_vvu = vv_u*dt
        curr_vve = v_e*dt
        while (x_u<a_u) & (k*dt<t_0_e): # % for FBs and Express Responses (purely urgency-triggered responses)
            x_u = x_u + curr_vvu + normals[0,ii] #; % urgency integrator
            k = k+1
            ii = ii+1
        while (x_u<a_u) & (abs(x_e)<a_e): # % for Non-Express Responses
            x_u = x_u + curr_vvu + normals[0,ii] # % urgency integrator
            x_e = x_e + curr_vve + normals[1,ii] # % evidence integrator
            k = k+1 # ; % time step counter
            ii = ii+1
        # end
        RT[i] = k*dt #; % the RT is already set, but the choice is not set yet for urgency-triggered trials
        if x_u<a_u: # ; % whether the trial hit the evidence bound or the urgency bound
            Bound[i] = 1 # if true replace to 1
        if (not Bound[i]) & (StimOnset<RT[i]): # % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
            T  = RT[i] # ; % we save the hitting time
            if T<t_0_e: # % for express responses
                k = int(t_0_e/dt) # ; % we cannot integrate evidence until "t_0_e" # should I int or floor'?
            while k*dt < T+(t_0_e-StimOnset): # % keep integrating for "(t_0_e-StimOnset)" seconds more
                x_e = x_e + curr_vvu + normals[1,ii] # ; % evidence integrator
                k = k+1 #; % time step counter
                k = ii+1
        #    end
        #end
        # sign function does not exist
        if x_e<0:
            Resp_vec[i] = -1
        elif x_e>0:
            Resp_vec[i] = 1
        else:
            Resp_vec[i] = 0
        Accu[i] = ((Resp_vec[i]*Categ_vec[i]+1)/2).astype(np.int_) # ;

    return RT, v, Accu, Bound, var**0.5



    # obsolete shit
    # bound = np.zeros((n), dtype=np.intc)
    # cdef double[:] RT_view = RT
    # cdef int[:] Accu_view = Accu
    # cdef int[:] bound_view = bound