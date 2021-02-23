import pandas as pd
import swifter
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from concurrent.futures import as_completed, ThreadPoolExecutor

# still need to decide whether to work single trial or whole subjecct level
# start with single trial
class ab: # mnemonic short: alex bayes.
    default_vary = np.array([
        [9.06948728e+01],
        [1.03033116e-03],
        [1.67140557e-05],
        [2.24947226e-09],
        [3.77947579e-14],
        [7.35611508e-20]]
    )
    default_varx = np.array([
        [1.32727322e+02],
        [1.82784201e-02],
        [2.46233988e-04],
        [3.10683547e-08],
        [4.23700961e-13],
        [6.73214905e-19]
        ])
    meanvar = (default_varx+default_vary)/2
    
    default_mu = np.array([[-25,0],[0,0],[0,0],[-35,85],[0,0],[0,0]])

    def __init__(self, row, kwargs={'inv_method':'pinv'}, sigma=5, var=None, mu=default_mu, l_sub_y=200, l_sub_x=75):
        t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
        T = row.resp_len*1000 + 50 # total span
        t = t.astype(int)/1000_000 - 250 - row.sound_len #  we take 50ms earlier CportOut
        pose = np.c_[row.trajectory_x,row.trajectory_y]
        fp = np.argmax(t>=0)
        lastp = np.argmax(t>T) # first frame after lateral poke in 
        tsegment = np.append(t[fp:lastp],T)
        tsegment = np.insert(tsegment,0,0)
        boundaries = [[0,0]]*6
        for i, traj in enumerate([pose, row.traj_d1, row.traj_d2]):
            f = interp1d(t, traj, axis=0)
            initial = f(0)
            last = f(T)
            boundaries[i] = initial
            boundaries[i+3] = last
        pose = np.insert(pose[fp:lastp], 0, boundaries[0], axis=0)
        pose = np.append(pose, boundaries[3].reshape(-1,2), axis=0)
        self.kwargs = kwargs
        self.x_b = np.vstack(boundaries)
        self.coords = pose
        self.t = tsegment
        self.vt = tsegment.reshape(-1,1) ** np.arange(6)
        self.Mt0te = ab.get_Mt0te(tsegment[0], tsegment[-1])
        self.theta = ab.get_theta(tsegment, pose, inv_method=kwargs.get('inv_method', 'pinv'))
        self.N = ab.get_N(self.vt, self.Mt0te, inv_method=kwargs.get('inv_method', 'pinv'))
        self.theta_b = np.linalg.pinv(self.Mt0te) @ self.x_b
        self.sigma = sigma
        self.mu = mu # beware it is 6,2
        if var is None:
            self.SIGMA = ab.meanvar * np.identity(6)
        else:
            self.SIGMA = var * np.identity(6) # this attribute can be changed afterwards, and recompute following parameters right?

        self.W, self.mx, self.my, self.L = ab.get_WmL(self.sigma, self.N, self.SIGMA, self.mu, self.coords)
        self.l_sub_y = l_sub_y
        self.l_sub_x = l_sub_x

        
    
    @staticmethod
    def v_(t):
        return t.reshape(-1,1)**np.arange(6)

    @staticmethod
    def get_theta(t, coords, inv_method='pinv'):
        # extracts coef by inverting
        X = ab.v_(t)
        XTX = X.T @ X
        inv_method = getattr(np.linalg, inv_method)
        inv = inv_method(XTX)
        theta = (inv@X.T)@coords
        return theta

    @staticmethod
    def get_Mt0te(t0, te):
        Mt0te=np.array([
            [1, t0, t0**2, t0**3, t0**4, t0**5],
            [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
            [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
            [1, te, te**2, te**3, te**4, te**5],
            [0, 1, 2*te, 3*te**2, 4*te**3, 5*te**4],
            [0, 0, 2, 6*te, 12*te**2, 20*te**3]
        ])
        return Mt0te

    @staticmethod
    def get_N(vt, M, inv_method='pinv'):
        # Basically the matrix N is the linear transformation from the boundary conditions to the predicted x value at observation times
        inv_method = getattr(np.linalg, inv_method)
        return vt @ inv_method(M)
        
    @staticmethod
    def retrieve_coords(t, coefs):
        vt = ab.v_(t)
        return np.dot(vt, coefs)

    @staticmethod
    def get_WmL(sigma, N, SIGMA, mu, coords):
        #mu = mu.reshape(-1,1)
        SIGMA_1 = np.linalg.pinv(SIGMA)
        W = sigma**2 * np.identity(N.shape[0]) + N @ SIGMA @ N.T # shape n x n
        L = np.linalg.pinv( # shape 6,6
            SIGMA_1 + N.T @ N/sigma**2 
        )
        mx = L @ (N.T @ coords[:,0].reshape(-1,1)/sigma**2 + SIGMA_1 @ mu[:,0].reshape(-1,1)) # shape 6,1
        my = L @ (N.T @ coords[:,1].reshape(-1,1)/sigma**2 + SIGMA_1 @ mu[:,1].reshape(-1,1))
        return W, mx, my, L

    @staticmethod
    def n_(t, M):
        if t.size==1: # scalaR
            vt = (np.array([t]*6) ** np.arange(6)).reshape(-1,1)
            return vt.T @ np.linalg.inv(M)
        else:
            vt = t.reshape(1,-1) ** np.arange(6).reshape(-1,1)
            return vt.T @ np.linalg.inv(M)
        #return np.dot()

    # @staticmethod
    # def loglikelihood(coordi, lsub,eps=0.05):
    #     """
    #     coordi = x or y coordinate vector
    #     lsub = space in given dimension for dummy model"""
    #     LLH = np.log(
    #             eps/lsub** + (1-eps)
    #         )
        # sum afterwards

    @staticmethod
    def get_zk():
        
        return None

    # N es la matrix n x 6 con los valores de t con observación
    # n(t) es un vector de length 6 que es para un t dado (así te da toda la trayectoría continua)
    # et n(t)m en la eq 11 te da el polinomio de la mean trajectory

    # perdón, era n(t) y no m(t) # final de eq 11


def get_data(row, dim=0, sigma=None, SIGMA=None, SIGMA_1=None, invert=False, factors=False, twoD=False):
    """dim= 0 = x; dim=1=y,
    twoD overrides and take both"""
    try:
        t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
        T = row.resp_len*1000 + 50 # total span
        t = t.astype(int)/1000_000 - 250 - row.sound_len #  we take 50ms earlier CportOut
        if t.size<10:
            raise ValueError('custom exception to discard trial')
        
        fp = np.argmax(t>=0)
        lastp = np.argmax(t>T) # first frame after lateral poke in 
        tsegment = np.append(t[fp:lastp],T)
        tsegment = np.insert(tsegment,0,0)
        if invert and row.R_response==0:
            pose = np.c_[row.trajectory_x,row.trajectory_y.copy()*-1]
        else:
            pose = np.c_[row.trajectory_x,row.trajectory_y]
        boundaries = [[0,0]]*6
        for i, traj in enumerate([pose, row.traj_d1, row.traj_d2]):
            f = interp1d(t, traj, axis=0)
            initial = f(0)
            last = f(T)
            boundaries[i] = initial
            boundaries[i+3] = last
        pose = np.insert(pose[fp:lastp], 0, boundaries[0], axis=0)
        pose = np.append(pose, boundaries[3].reshape(-1,2), axis=0)

        # precompute and keep it in memory so we avoid recalculaiting through EM
        # we can do it until unless we update sigmas hyperparams
        M = ab.get_Mt0te(tsegment[0], tsegment[-1])
        M_1 = np.linalg.inv(M)
        vt = ab.v_(tsegment)
        N = vt @ M_1
        if not twoD:
            W = sigma**2 * np.identity(N.shape[0]) + N @ SIGMA @ N.T # shape n x n
            pose = pose[:,dim] # 0dim if not twoD
        else:
            W = (
                sigma**2 * np.identity(N.shape[0]) + N @ SIGMA[0] @ N.T,
                sigma**2 * np.identity(N.shape[0]) + N @ SIGMA[1] @ N.T
            )
    
        if not factors:
            return True, pose, N, W
        else:
            extra = [row.coh *2 -1, row.zidx, row.dW_trans] #! remember order
            for i in range(3):
                if np.isnan(extra[i]): # replace nans for extra
                    extra[i] = 0
            # mount F
            Fk = np.array([1]+extra)
            return True, pose, N, W, Fk
    except Exception as e:
        print(f'exception while extracting base data @ index {row.name}\n{e}')
        # raise e
        if not factors:
            return False, None, None, None
        else:
            return [False] + [None]*4

def preprocess_df(
    df, dim=0, sigma=None,SIGMA=None, SIGMA_1=None, invert=False, factors=False,
    twoD=False):
    """ preprocess df so tsegment and coords are adapted"""
    print('preprocess_df function called')
    colnames = ['OK', 'coords', 'N', 'W']
    if factors:
        colnames += ['F'] # coherence, trial index, transition bias
    print('get_data called')
    tmp = df.swifter.apply( ## TODO: reuse swifter
        lambda x: get_data(x, dim=dim, sigma=sigma, SIGMA=SIGMA, SIGMA_1=SIGMA_1, invert=invert, factors=factors, twoD=twoD), 
        axis=1, result_type='expand')
    tmp.columns = colnames
    out = {}
    for cname in colnames[1:]: # all but OK
        out[cname] = tmp.loc[tmp.OK==True,cname].tolist()

    return out

def E_step(args):
    """
    coords = 1d coords, mu=1d mu, dummyspan span for dummy model in that dim"""
    try:
        coords, N, W, mu, eps, dummy_span = args

        marg_gauss = multivariate_normal( (N @ mu).ravel(), W) # eq 9
        pmk = marg_gauss.pdf(coords) # prob under ballistic model

        alt_prob = eps/((1-eps)*dummy_span**coords.size)
        zk = pmk / (pmk + alt_prob)
        # loglikelihod for trial k
        llh = np.log(
            eps/dummy_span**N.shape[0] + (1-eps) * pmk
        )

        return (pmk, zk, llh)
    except Exception as e:
        print(f'exception in E_step, \n{e}')
        #raise e
        return (1e-10,1e-10,1e-10)



def E_step_factors(args):
    """
    coords = 1d coords, mu=1d mu, dummyspan span for dummy model in that dim"""
    try:
        coords, N, W, F, B, eps, dummy_span = args

        mu = B @ F.reshape(4,1) #F.reshape(1,4) @ B 
        marg_gauss = multivariate_normal( (N @ mu.reshape(-1,1)).ravel(), W) # eq 9
        pmk = marg_gauss.pdf(coords) # prob under ballistic model

        alt_prob = eps/((1-eps)*dummy_span**coords.size) # sometimes runtime warning div 0
        zk = pmk / (pmk + alt_prob)
        # loglikelihod for trial k
        llh = np.log(
            eps/dummy_span**N.shape[0] + (1-eps) * pmk
        )

        return (pmk, zk, llh)
    except Exception as e:
        print(f'exception in E_step_factors, \n{e}')
        # print(B.shape)
        # print(F.shape)
        #raise e
        return (1e-10,1e-10,1e-10)

def M_step(args):
    coords, N, W, mu, eps, dummy_span = args
    marg_gauss = multivariate_normal( (N @ mu).ravel(), W) # eq 9
    pmk = marg_gauss.pdf(coords) # prob under ballistic model

    alt_prob = eps/((1-eps)*dummy_span**coords.size)
    zk = pmk / (pmk + alt_prob)

    W_1 = np.linalg.inv(W)

    first_term = zk * N.T @ W_1 @ N # (6,6)
    second_term = zk * coords @ W_1 @ N # (6,)

    return (first_term, second_term)

def M_step_factors(args):
    """this one works with beta instead of mu (eq. 19)"""
    coords, N, W, F, Beta, eps, dummy_span  = args
    #mu = F.reshape(1,4) @ Beta
    mu = Beta @ F.reshape(4,1)

    marg_gauss = multivariate_normal( (N @ mu.reshape(-1,1)).ravel(), W) # eq 9
    pmk = marg_gauss.pdf(coords) # prob under ballistic model

    alt_prob = eps/((1-eps)*dummy_span**coords.size)
    zk = pmk / (pmk + alt_prob)

    W_1 = np.linalg.inv(W)

    # Alex:
    # a la izquiera: F (4,6) x NT (6,n) x W-1 (n,n) x N (n,6) x FT (6,4) -> 4 x 4
    # a la derecha: x (1,n) x W-1(n,n) x N (n,6) x FT (6,4 ) -> 1x4 
    # # -> hay que tomar el transpose de todo esto para que de (4,1)
    # Fk = np.repeat(F.reshape(-1,1),6, axis=1) # (4,6)
    # first_term = Fk * zk @ N.T @ W_1 @ N @ Fk.T# (6,6)
    # second_term = zk * coords @ W_1 @ N @ Fk.T # (6,)

    # happy new year
    #B_hat = Beta.flatten('F') # fortran style (columnwise)
    F_hat = np.zeros((6,24))
    for i, j in enumerate(np.arange(24,step=4)):
        F_hat[i, j:j+4] = F

    first_term = zk * F_hat.T @ N.T @ W_1 @ N @ F_hat # (24,24)
    second_term = zk * F_hat.T @ N.T @ W_1 @ coords.reshape(-1,1) # (24,1)

    return (first_term, second_term)

def estimate_init6(row):
    """requires precomputed d1 and d2"""
    t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
    T = row.resp_len*1000 + 50 # total span
    t = t.astype(int)/1000_000 - 250 - row.sound_len #  we take 50ms earlier CportOut
    pose = np.c_[row.trajectory_x,row.trajectory_y]
    out = [0]*6
    for i, traj in enumerate([pose, row.traj_d1, row.traj_d2]):
        f = interp1d(t, traj, axis=0)
        initial = f(0)
        last = f(T)
        out[i] = initial
        out[i+3] = last
    #return pose, tsegment
    xb = np.vstack(out)
    M = ab.get_Mt0te(0, T)
    res = np.linalg.pinv(M)@xb
    return res


def EM(
    df, dim=0, mu=None, eps=0.05, sigma=5, SIGMA=None, dummy_span=None, 
    threshold=1e-3, workers=7, invert=False, factors=False
): 
    """perform expectation maximization
    dim= 0=x; 1=y, invert=flip y coords"""
    
    # TODO: track -n likelihoods
    if not factors:
        if mu is None and SIGMA is None:
            print('using default mu and SIGMA')
            mu = ab.default_mu[:,dim]
            if dim==0:
                var = ab.default_varx
                dummy_span = 75
            else:
                var=ab.default_vary
                dummy_span = 200
            SIGMA = var * np.identity(6)
        
    assert dummy_span is not None, 'set dummy span when defining mu or SIGMA'

    SIGMA_1 = np.linalg.inv(SIGMA)
    # get data to work with
    print('preprocessing df')
    d = preprocess_df(
        df, dim=dim, sigma=sigma ,SIGMA=SIGMA, SIGMA_1=SIGMA_1, 
        invert=invert, factors=factors
    )
    if factors:
        # init Beta:
        B = np.zeros((6,4)) # remember ,4 = intercept, coherence, trial_ind and transition bias
        # B = np.random.normal(scale=0.1, size=(6,4))
        #bigF = np.concatenate(d['F']).reshape(-1,4)
        #mu = F @ B

    loglikelihood = -np.inf # init var
    llh_increase = 1. # init var
    # main loop:
    itercount = 0
    print('main loop started')
    while llh_increase > threshold:
        #### expectation step:
        # compute the probability under the ballistic model for each trial using eq 9
        # compute Zk using eq 13
        allres = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if not factors:
                jobs = [
                    executor.submit(
                        E_step, 
                        (d['coords'][i], d['N'][i], d['W'][i],mu, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            else:
                jobs = [
                    executor.submit(
                        E_step_factors, 
                        (d['coords'][i], d['N'][i], d['W'][i], d['F'][i], B, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            for job in as_completed(jobs):
                allres += [job.result()]
        res = np.concatenate(allres).reshape(-1,3).sum(axis=0) # sum prob ballistic, zk and llh
        # calc new Loglike
        llh_increase = res[2] - loglikelihood
        print('llh_increase',llh_increase)
        if llh_increase < threshold:
            break
        loglikelihood = res[2]
        assert llh_increase > 0, f'llh increase below 0 in iter={itercount}'

        #### maximization step
        # update eps using eq 16 
        eps = 1 - res[1]/len(d['N'])
        # update the mean over boundary conditions mu using eq 18
        allres = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if not factors:
                jobs = [
                    executor.submit(
                        M_step, 
                        (d['coords'][i], d['N'][i], d['W'][i],mu, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            else:
                jobs = [
                    executor.submit(
                        M_step_factors, 
                        (d['coords'][i], d['N'][i], d['W'][i],d['F'][i], B, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            for job in as_completed(jobs):
                allres += [job.result()]

        first_term, second_term = [],[]
        for i in range(len(allres)): # split in two lists
            first_term += [allres[i][0]]
            second_term += [allres[i][1]]
        if not factors:
            try:
                mu = np.linalg.inv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0)
            except:
                print('using linalg.pinv for mu update due to singular matrix')
                mu = np.linalg.pinv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0)
        else:
            try: # TODO: fix
                print('eps',eps)
                B_hat = np.linalg.inv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0)
                
            except:
                print('using linalg.pinv for B update due to singular matrix')
                B_hat = np.linalg.pinv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0)

            #B = np.repeat(b.reshape(-1,1), 6, axis=1) # TODO: fix
            B = B_hat.reshape((6,4), order='C') # this was the issue


        itercount += 1
        if itercount%10==0:
            print(f'{itercount} iters done...')

    print(f'"converged" after {itercount} iterations; llh = {loglikelihood} ')
    if not factors:
        return eps, mu, sigma, SIGMA
    else:
        return eps, B, sigma, SIGMA


def get_zk(row, eps, mu, sigma, SIGMA, dummy_span,dim=0, invert=False):
    """evaluates whether it was or not (through pandas apply), copypasting 
    preprocess + expand to loglike"""
    try:
        t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
        T = row.resp_len*1000 + 50 # total span
        t = t.astype(int)/1000_000 - 250 - row.sound_len #  we take 50ms earlier CportOut
        if t.size<10:
            raise ValueError('custom exception to discard trial')
        
        fp = np.argmax(t>=0)
        lastp = np.argmax(t>T) # first frame after lateral poke in 
        tsegment = np.append(t[fp:lastp],T)
        tsegment = np.insert(tsegment,0,0)
        if invert and row.R_response==0:
            pose = np.c_[row.trajectory_x,row.trajectory_y.copy()*-1]
        else:
            pose = np.c_[row.trajectory_x,row.trajectory_y]
        boundaries = [[0,0]]*6
        for i, traj in enumerate([pose, row.traj_d1, row.traj_d2]):
            f = interp1d(t, traj, axis=0)
            initial = f(0)
            last = f(T)
            boundaries[i] = initial
            boundaries[i+3] = last
        pose = np.insert(pose[fp:lastp], 0, boundaries[0], axis=0)
        pose = np.append(pose, boundaries[3].reshape(-1,2), axis=0)

        # precompute and keep it in memory so we avoid recalculaiting through EM
        # we can do it until unless we update sigmas hyperparams
        M = ab.get_Mt0te(tsegment[0], tsegment[-1])
        M_1 = np.linalg.inv(M)
        vt = ab.v_(tsegment)
        N = vt @ M_1
        W = sigma**2 * np.identity(N.shape[0]) + N @ SIGMA @ N.T # shape n x n
        args = pose[:,dim], N, W, mu, eps, dummy_span
        _, zk, _ = E_step(args)
        return zk
    except:
        print(f'exception in while extracting base data @ index {row.name}')
        return np.nan

def get_zk_factors(row, eps, B, sigma, SIGMA, dummy_span,dim=0, invert=False):
    """evaluates whether it was or not (through pandas apply), copypasting 
    preprocess + expand to loglike"""
    try:
        SIGMA_1 = np.linalg.inv(SIGMA)
        ret, pose, N, W, Fk = get_data(row, dim=dim, sigma=sigma, SIGMA=SIGMA, SIGMA_1=SIGMA_1,
            invert=invert, factors=True, twoD=False            
        )
        
        #assert not ret, "wrong returned value"
        args = pose, N, W, Fk, B, eps, dummy_span
        # return True, pose, N, W, Fk
        _, zk, _ = E_step_factors(args)
        return zk
    except Exception as e:
        print(f'exception in get_zk_factors @ index {row.name}\n{e}')
        raise e
        return np.nan



def E_step2D(args):
    """
    coords = 2d coords (n, 2) arr, mu= array(6,2), 
    dummyspan span for dummy model in each dim, tuple
    W = tuple (2 matrices, x and y)"""
    try:
        coords, N, W, mu, eps, dummy_span = args
        #xdim
        marg_gauss_x = multivariate_normal( (N @ mu[:,0]).ravel(), W[0]) # eq 9
        pmkx = marg_gauss_x.pdf(coords[:,0]) # prob under ballistic model
        #ydim
        marg_gauss_y = multivariate_normal( (N @ mu[:,1]).ravel(), W[1]) # eq 9
        pmky = marg_gauss_y.pdf(coords[:,1]) # prob under ballistic model

        alt_prob = eps/((1-eps)*(dummy_span[0]*dummy_span[1])**coords.shape[0])
        zk = (pmkx*pmky) / (pmkx*pmky + alt_prob)
        # loglikelihod for trial k
        llh = np.log(
            eps/(dummy_span[0]*dummy_span[1])**N.shape[0] + (1-eps) * pmkx*pmky
        )

        return ((pmkx, pmky), zk, llh)
    except Exception as e:
        print(f'exception in E_step2D, \n{e}')
        #raise e
        return ((1e-10, 1e-10),1e-10,1e-10)

def M_step2D(args):
    """like other but coords, W , mu and dummy_span have extra dim"""
    coords, N, W, mu, eps, dummy_span = args
    marg_gauss_x = multivariate_normal( (N @ mu[:,0]).ravel(), W[0]) # eq 9
    pmkx = marg_gauss_x.pdf(coords[:,0]) # prob under ballistic model
    #ydim
    marg_gauss_y = multivariate_normal( (N @ mu[:,1]).ravel(), W[1]) # eq 9
    pmky = marg_gauss_y.pdf(coords[:,1]) # prob under ballistic model

    alt_prob = eps/((1-eps)*(dummy_span[0]*dummy_span[1])**coords.shape[0])
    zk = (pmkx*pmky) / (pmkx*pmky + alt_prob)

    W_1_x = np.linalg.inv(W[0])
    W_1_y = np.linalg.inv(W[1])

    first_term_x = zk * N.T @ W_1_x @ N # (6,6)
    second_term_x = zk * coords[:,0] @ W_1_x @ N # (6,)

    first_term_y = zk * N.T @ W_1_y @ N # (6,6)
    second_term_y = zk * coords[:,1] @ W_1_y @ N # (6,)

    return (first_term_x, second_term_x, first_term_y, second_term_y)


def EM2D(
    df, dim=0, mu=None, eps=0.05, sigma=5, SIGMA=None,
    threshold=1e-3, workers=7, invert=False, factors=False, dummy_span=(75,150)
): 
    """perform expectation maximization
    using both dimensions, invert=flip y coords"""
    
    # TODO: track -n likelihoods
    if not factors:
        if mu is None and SIGMA is None:
            print('using default mu and SIGMA')
            mu = ab.default_mu
            SIGMA = (
                np.concatenate([ab.default_varx[:3]]*2) * np.identity(6), 
                np.concatenate([ab.default_vary[:3]]*2) * np.identity(6)
            )
        
    assert dummy_span is not None, 'set dummy span when defining mu or SIGMA'

    SIGMA_1 = (np.linalg.inv(SIGMA[0]), np.linalg.inv(SIGMA[1]))
    # get data to work with
    print('preprocessing df')
    d = preprocess_df(
        df, dim=dim, sigma=sigma ,SIGMA=SIGMA, SIGMA_1=SIGMA_1, 
        invert=invert, factors=factors, twoD=True
    )
    if factors:
        # init Beta:
        B = np.ones((4,6))
        #bigF = np.concatenate(d['F']).reshape(-1,4)
        #mu = F @ B

    loglikelihood = -np.inf # init var
    llh_increase = 1. # init var
    # main loop:
    itercount = 0
    while llh_increase > threshold:
        #### expectation step:
        # compute the probability under the ballistic model for each trial using eq 9
        # compute Zk using eq 13
        allres = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if not factors:
                jobs = [
                    executor.submit(
                        E_step2D, 
                        (d['coords'][i], d['N'][i], d['W'][i],mu, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            else:
                jobs = [
                    executor.submit(
                        E_step_factors, 
                        (d['coords'][i], d['N'][i], d['W'][i], d['F'][i], B, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            for job in as_completed(jobs):
                allres += [job.result()]
        res = np.concatenate(allres).reshape(-1,3).sum(axis=0) # sum prob ballistic, zk and llh
        # calc new Loglike
        llh_increase = res[2] - loglikelihood
        if llh_increase < threshold:
            break
        loglikelihood = res[2]
        assert llh_increase > 0, f'llh increase below 0 in iter={itercount}'

        #### maximization step
        # update eps using eq 16 
        eps = 1 - res[1]/len(d['N'])
        print('eps', eps)
        # update the mean over boundary conditions mu using eq 18
        allres = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            if not factors:
                jobs = [
                    executor.submit(
                        M_step2D, 
                        (d['coords'][i], d['N'][i], d['W'][i],mu, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            else:
                jobs = [
                    executor.submit(
                        M_step_factors, 
                        (d['coords'][i], d['N'][i], d['W'][i],d['F'][i], B, eps, dummy_span)
                        ) for i in range(len(d['N']))]
            for job in as_completed(jobs):
                allres += [job.result()]

        first_term_x, second_term_x, first_term_y, second_term_y = [],[], [], []
        for i in range(len(allres)): # split in two lists
            first_term_x += [allres[i][0]]
            second_term_x += [allres[i][1]]
            first_term_y += [allres[i][2]]
            second_term_y += [allres[i][3]]

        if not factors:
            try:
                mu_x = np.linalg.inv(np.sum(first_term_x, axis=0)) @ np.sum(second_term_x, axis=0)
                mu_y = np.linalg.inv(np.sum(first_term_y, axis=0)) @ np.sum(second_term_y, axis=0)
                mu = np.stack([mu_x, mu_y], axis=-1)
            except:
                print('using linalg.pinv for mu update due to singular matrix')
                mu_x = np.linalg.pinv(np.sum(first_term_x, axis=0)) @ np.sum(second_term_x, axis=0)
                mu_y = np.linalg.pinv(np.sum(first_term_y, axis=0)) @ np.sum(second_term_y, axis=0)
                mu = np.stack([mu_x, mu_y], axis=-1)
        else:
            try: # TODO: fix
                b = np.linalg.inv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0)
            except:
                print('using linalg.pinv for mu update due to singular matrix')
                b = np.linalg.pinv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0)

            B = np.repeat(b.reshape(-1,1), 6, axis=1) # TODO: fix


        itercount += 1
        if itercount%10==0:
            print(f'{itercount} iters done...')

    print(f'"converged" after {itercount} iterations')
    if not factors:
        return eps, mu, sigma, SIGMA
    else:
        return eps, B, sigma, SIGMA