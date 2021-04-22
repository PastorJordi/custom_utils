import pandas as pd
import swifter
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from concurrent.futures import as_completed, ThreadPoolExecutor
import os
import pickle
import warnings


# rewritting everything before switching towards a simpler strategy
# I tried to integrate it as a class (will make it simpler in terms of repeating kwargs)
# 2 dims at once is still missing, also the roundabout to avoid numerical issues quen calculating zk
# fitting mu or B should work. Still have to add a function to retrieve Zks with a given instance of the class

# this code assume you have certain naming in the dataframe columns, i'll list them just in case you want
# to search and replace them.
# 'trajecotry_stamps': col with trajectory timestamps (numpy datetim64)
# fix_onset_dt:  fixation onset (pandas datetime)
# resp_len : motor response length, in seconds (from c-port out to lateral port-in)
# sound_len: stimulus length in ms
# trajectory_? : snout/above/snout coordinates in ? axis. same sithe than traj stamps
# R_response: whether trial had a left (0) or right (1) response
# origidx = trial index in the session (did not stack sessions in the same day)
# dW_trans : transition bias in that trial

# re-writting few functions scattered elsewhere

def gradient_np(row):
    """returns speed accel and jerk by np.gradient"""
    try:
        t = row['trajectory_stamps'].astype(float)/1000
        #dt1 = np.repeat(np.diff(row['trajectory_stamps'].astype(float)/1000),2).reshape(-1,2)
        t = t-t[0]
        coords = np.c_[row['trajectory_x'], row['trajectory_y']]
        d1 = np.gradient(coords, t, axis=0)
        d2 = np.gradient(d1, t,axis=0)
        d3 = np.gradient(d2, t,axis=0)
        return d1, d2, d3, t
    except: #just in case something goes wrong, we avoid losing everything when using .apply
        return np.nan, np.nan, np.nan, np.nan

def v_(t):
    return t.reshape(-1,1)**np.arange(6)

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


class tinf: # (trial info)
    '''given that same code is rewritten and messy, ill try to compact access to trial info (row) in a class
    call it to access info later *(do not preprocess everything by default to save computing power when possible)'''
    def __init__(self, row, M=False, othermatrices=False, factors=False, dim=1,twoD=False, boundary_cond=False, verbose=False,
        sigma=5, SIGMA=None, ab_instance=None, collapse_sides=False, factorlist=['coh2', 'zidx', 'dW_trans'], 
        factor_kw={'add_intercept':True}, collapsing_factors = ['coh2', 'dW_trans'],time_pre_portout=50, 
    ):
        # BL and BR are betas to use interactive plotting function. [size should match with factorlist + itnercept]
        # row is a row from a df. Ideally this is will be instantiated when using long apply
        # time_pre_portout: get some extra ms before port out (trajectory starts slightly before)
        t = row.trajectory_stamps - row.fix_onset_dt.to_datetime64()
        T = row.resp_len*1000 + time_pre_portout # total span
        t = t.astype(int)/1000_000 - (300-time_pre_portout) - row.sound_len #  align to Cportout
        if collapse_sides and row.R_response==0:
            pose = np.c_[row.trajectory_x,row.trajectory_y.copy()*-1]
        else:
            pose = np.c_[row.trajectory_x,row.trajectory_y]
        self.original_pose = pose
        f = interp1d(t, pose, axis=0)
        initpose = f(0)
        lastpose = f(T)
        fp = np.argmax(t>=0)
        lastp = np.argmax(t>T) # first frame after lateral poke in 
        tsegment = np.append(t[fp:lastp],T)
        tsegment = np.insert(tsegment,0,0)
        pose = np.insert(pose[fp:lastp], 0, initpose, axis=0)
        pose = np.append(pose, lastpose.reshape(-1,2), axis=0) # 2D!
        self.row = row
        self.pose = pose
        self.tsegment = tsegment
        self.boundary_cond = None
        self.Fk = None
        self.M = None
        self.M_1 = None
        self.vt = None
        self.N = None
        self.W = None
        self.okstatus = True
        self.t = t # original t vector
        self.T = T

        if ab_instance is not None: # reduce code by looping with setattr and getattr
            for at in ['sigma', 'SIGMA', 'twoD', 'factors', 'dim', 'invert', 'factorlist', 'factor_kw']:
                setattr(
                    self, 
                    at,
                    getattr(ab_instance, at, None)
                )
        else:
            self.sigma = sigma
            self.SIGMA = SIGMA
            self.twoD = twoD
            self.factors = factors
            self.collapse_sides = collapse_sides
            self.factorlist = factorlist
            self.factor_kw = factor_kw
            if twoD:
                self.dim = np.array([0,1])
            else:
                self.dim = dim

        try:
            if factors:
                self.get_factors(**factor_kw)
            if M:
                self.get_M()
            if boundary_cond:
                self.get_boundarycond()
            if othermatrices:
                self.get_othermatrices()
        except Exception as e:
            self.okstatus=False
            if verbose:
                print(f'err while extracting info in trial {row.name}\n{e}')

    def get_factors(self, add_intercept = True):
        factors = self.row[self.factorlist].fillna(0).values
        if add_intercept:
            self.Fk = np.array([1]+factors.tolist()) # prepended intercept
        else:
            self.Fk = np.array(factors)
    
    def get_boundarycond(self):
        # if self.M is None:
        #     self.get_M()

        out = [0]*6
        for i, traj in enumerate([self.original_pose, self.row.traj_d1, self.row.traj_d2]):
            f = interp1d(self.t, traj, axis=0)
            initial = f(0)
            last = f(self.T)
            out[i] = initial
            out[i+3] = last
        # self.boundary_cond = self.M_1 @ np.vstack(out)
        self.boundary_cond = np.vstack(out)


    def get_M(self):
        self.M = get_Mt0te(self.tsegment[0], self.tsegment[-1])
        self.M_1 = np.linalg.inv(self.M)
    
    def get_othermatrices(self):
        if self.M is None:
            self.get_M()
        self.vt = v_(self.tsegment)
        self.N = self.vt @ self.M_1
        if not self.twoD:
            self.W = self.sigma**2 * np.identity(self.N.shape[0]) + self.N @ self.SIGMA @ self.N.T # shape n x n
        else:
            self.W = (
                self.sigma**2 * np.identity(self.N.shape[0]) + self.N @ self.SIGMA[0] @ self.N.T,
                self.sigma**2 * np.identity(self.N.shape[0]) + self.N @ self.SIGMA[1] @ self.N.T
            )

    def return_preprocessed(self):
        # this should check whether N W and Fk are processed, spent many time with this one
        toreturn = [self.okstatus, self.pose[:,self.dim], self.N, self.W] + self.factors*[self.Fk]
        return toreturn
    


# TODO: implement both dims at once
# TODO: auto invert is not handling default mu

class ab:
    """
    class to implement Alex Hyafil's suggestion.
    His equations and disclosure here:
    https://www.overleaf.com/project/5fae56a2878ad0dd1ca7472d

    I do not know whether to make it flexible or simply use my feature naming (latter, probably)

    the idea is to follow the same flow used in the notebook but saving hyperparams/kwargs in class vars
    """

    default_mu = np.array([[-25,0],[0,0],[0,0],[-35,85],[0,0],[0,0]])

    def __init__(
        self, df=None, sigma=5, SIGMA=None, mu=None, factors=False, dim=1, dummy_span=150, workers=7, twoD=False,
        min_frames_traj=15, random_state=123, response_side=0, eps = 0.05, collapse_sides=False,
        factorlist=['coh2', 'zidx', 'dW_trans'], factor_kw={'add_intercept':True}, collapsing_factors=['coh2', 'dW_trans'],
        loadpath=None, time_pre_portout=50
    ):
        """
        df = filtered dataframe (side, invalids, rt/mt bins)
        mu = if None (and B=None as well) ues default initial values
        factors = whether to calc mu from coef matrix B
        dim = 0,1; which dim to use (x or y). twoD overrides this
        dummy_span = span in px for alternative model (75 for x and 150 for y should work fine when fitting single dim)
        workers = number of threads when using concurrent.futures
        twoD = whether to fit both dims at once or not
        response_side = 0-left or 1-right
        min_frames_traj: minimum frames to consider (discard trajectories with less datapoints)
        eps = initial epsilon
        collapse_sides : whether to invert left trial trajectories & their factors (so we align it to final choice* and congruent/incongruent priors)
        factorlist: which cols/factors/features from DF will be used to find B
        time_pre_portout: ms to consider as trajectory before photogate going off
        collapsing_factors : if collapse_sides, these factorlist should be inverted (to align with final choice  hence regression makes sense)

        briefly, this will work as follows>
        - init( with params)
        - preprocess (pulls required data)
        - fit_EM
        """
        
        # apply filtering so we avoid to rewrite it every single time
        if loadpath is None:
            assert isinstance(df, pd.DataFrame), 'df does not look as a dataframe'
            df = df.loc[
                (df.trajectory_x.apply(len)>=min_frames_traj) & (df.R_response==response_side)
                ]
            assert df.size, 'beware filtering, (response=side?), df is empty'
            
            if twoD:
                dim=np.array([0,1])
                raise NotImplementedError('not yet')
            

            # calc some columns if not precomputed
            if 'traj_d1' not in df.columns:
                print('traj derivatives not found, calculating...')
                tmp = df.swifter.apply(lambda x: gradient_np(x)[:3], axis=1, result_type='expand')
                tmp.columns = ['traj_d1', 'traj_d2', 'traj_d3']
                df = pd.concat([df, tmp], axis=1)

            if factors and 'zidx' not in df.columns:
                print('calculating zscore for trial index')
                df['zidx'] = (df.origidx - df.origidx.mean())/df.origidx.std()

            if factors:
                for f in factorlist:
                    assert f in df.columns, f'factor "{f}" not found among df.cols'

            self.df=df#.copy()
            self.subject = df.subjid.unique()
            self.B = None # we-ll define it later according to Fk.size
            self.eps = eps
            self.sigma=sigma
            
            self.dim = dim
            self.random_state = random_state
            self.factors = factors
            self.twoD = twoD
            self.workers=workers
            self.response_side = response_side
            self.dummy_span = dummy_span
            self.collapse_sides = collapse_sides
            self.factorlist = factorlist
            self.factor_kw = factor_kw
            self.time_pre_portout = time_pre_portout

            if mu is None and factors==False:
                print('using default mu, else provide it as an arg')
                self.mu = self.default_mu.copy()[:,dim]
            else:
                self.mu=mu

            if SIGMA is not None:
                self.SIGMA=SIGMA
            else:
                self.estimate_SIGMA_()

            self.SIGMA_1 = np.linalg.inv(self.SIGMA)
            self.ntrials = None # perhaps after discarding some!*(bad trajectories etc.)
            self.llh = -np.inf
            self.prep_data = None
            self.history = {} # fit history in short
        else:
            self.subject = None
            self.load_model(loadpath)


    def estimate_SIGMA_(self, sample_size=1000):
        """samples trajectories to get an average as a prior"""
        print('SIGMA was not provided, estimating it')
        subset = self.df.loc[(self.df.hithistory>=0)&(self.df.Hesitation==False)].sample(sample_size, replace=True, random_state=self.random_state)
        subset['est_params'] = subset.swifter.apply(lambda x: tinf(x, dim=self.dim, boundary_cond=True, time_pre_portout=self.time_pre_portout).boundary_cond, axis=1)
        if not self.twoD:
            sigma_b = np.stack(subset['est_params'].tolist(), axis=2).std(axis=2)
            var = (sigma_b[:,self.dim]**2).reshape(-1,1)
            # SIGMA = np.concatenate([var[:3], var[:3]]) * np.identity(6)
            SIGMA = var * np.identity(6)
        self.SIGMA=SIGMA



    def preprocess(
            self
        ):
        """colkw - kwords for tinf class (clolumn mapping)"""
        colnames = ['OK', 'coords', 'N', 'W'] + ['F'] * self.factors
        tmp = self.df.swifter.apply( 
            #lambda row: tinf(row, factors=self.factors, othermatrices=True, SIGMA=self.SIGMA, verbose=True, factorlist=self.factorlist).return_preprocessed(), 
            lambda row: tinf(row, factors=self.factors, othermatrices=True, ab_instance=self, time_pre_portout=self.time_pre_portout).return_preprocessed(), 
            axis=1, result_type='expand')
        tmp.columns = colnames

        out = {}
        for cname in colnames[1:]: # all but OK
            out[cname] = tmp.loc[tmp.OK==True,cname].tolist()

        self.ntrials = tmp.shape[0] # number of trials which will run through EM
        self.prep_data = out

    @staticmethod
    def E_step_worker_(args):
        if len(args)==6:
            coords, N, W, mu, eps, dummy_span = args
        else:
            coords, N, W, F, B, eps, dummy_span = args
            mu = B @ F.reshape(-1,1) # usually (4,1)

        try:
            marg_gauss = multivariate_normal( (N @ mu.reshape(-1,1)).ravel(), W) # eq 9
            pmk = marg_gauss.pdf(coords) # prob under ballistic model

            alt_prob = eps/((1-eps)*dummy_span**coords.size) # sometimes runtime warning div 0
            zk = pmk / (pmk + alt_prob)
            # loglikelihod for trial k
            llh = np.log(eps/dummy_span**N.shape[0] + (1-eps) * pmk)
            return (pmk, zk, llh)
        except Exception as e:
            print(f'exception in E_step_worker, \n{e}')
            return (1e-10,1e-10,1e-10) # is this tiny enough?
    
    @staticmethod
    def M_step_worker_(args):
        if len(args)==6:
            coords, N, W, mu, eps, dummy_span = args
            factors_flag = False
        else:
            coords, N, W, F, B, eps, dummy_span = args
            mu = B @ F.reshape(-1,1)
            factors_flag = True

        marg_gauss = multivariate_normal( (N @ mu.reshape(-1,1)).ravel(), W) # eq 9
        pmk = marg_gauss.pdf(coords) # prob under ballistic model
        alt_prob = eps/((1-eps)*dummy_span**coords.size)
        zk = pmk / (pmk + alt_prob)
        W_1 = np.linalg.inv(W)
        
        if factors_flag:
            F_hat = np.zeros((6,6*F.size))
            for i, j in enumerate(np.arange(6*F.size,step=F.size)):
                F_hat[i, j:j+F.size] = F
            first_term =  F_hat.T @ N.T @ W_1 @ N @ F_hat # usually (24,24)
            second_term = zk * F_hat.T @ N.T @ W_1 @ coords.reshape(-1,1) # (24,1)
        else:
            first_term = zk * N.T @ W_1 @ N # (6,6)
            second_term = zk * coords @ W_1 @ N # (6,)

        return (first_term, second_term)

    def get_worker_list(self, i=None):
        '''avoiding typing twice same endless list
        selects args based on fitting mu or B
        if row is passed as arg, it extract single trial info'''
        ugly_arg_list = [
                            self.prep_data['coords'][i], # arg list # this could be cleaner.
                            self.prep_data['N'][i],
                            self.prep_data['W'][i]
                        ] + [self.mu] * (not self.factors) +\
                        [ self.prep_data['F'][i], self.B ] * self.factors +\
                        [self.eps, self.dummy_span] # remaining parameters

        return ugly_arg_list


    def fit_EM(self, threshold=1e-3, streak=2, verbose=0):
        '''
        expectation maximization loop
        threshold  =  loglikelihood increase below this will make exit the loop
        streak = times that threshold rule must be violated to exit the loop
        all vars and params should be already initialized in __init__

        ## we'll track fitting history since this takes few steps to converge (5~30)
        '''
        assert streak>=1, f'min streak value is 1, and attempted to use {streak}'
        
        self.history = {
            'llh' : [self.llh],
            'eps' : [self.eps]
        }
        if self.factors: # init B now that stuff is preprocessed
            self.B = np.zeros((6,self.prep_data['F'][0].size))


        if self.factors:
            self.history['tofit'] = [self.B.copy()] 
        else:
            self.history['tofit'] = [self.mu.copy()]

        
        itercounter = 0
        break_flag = False
        while True: # we'll break it explicitly
            allres = []

            # E step
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                jobs = [
                    executor.submit(
                        ab.E_step_worker_,  # static function 
                        self.get_worker_list(i=i) # util function to forge args
                    ) for i in range(self.ntrials)
                ]
                for job in as_completed(jobs):
                    allres += [job.result()]

            res = np.concatenate(allres).reshape(-1,3).sum(axis=0) # sum prob ballistic, zk and llh

            # update LLH & eps
            self.history['llh'] += [res[2]]
            self.llh = res[2]
            self.history['eps'] += [ 1 - res[1]/self.ntrials ] # this goes actually in M step. Eq 16
            self.eps = 1 - res[1]/self.ntrials

            # break or not based in llh history
            if itercounter>=streak: # avoids getting evaluated before we have not enough iters
                if (np.diff(self.history['llh'])[-streak:]<threshold).all():
                    break_flag = True
            
            # M step
            # update the mean over boundary conditions mu using eq 18
            allres = []
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                jobs = [
                    executor.submit(
                        ab.M_step_worker_,  # static function 
                        self.get_worker_list(i=i) # util function to forge args
                    ) for i in range(self.ntrials)
                ]
                for job in as_completed(jobs):
                    allres += [job.result()]

            first_term, second_term = [],[]
            for i in range(len(allres)): # split in two lists
                first_term += [allres[i][0]]
                second_term += [allres[i][1]]

            if self.factors: # fitting B
                B_hat = np.linalg.inv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0) # eq 23
                B = B_hat.reshape((6,-1), order='C') # this was the issue!
                self.history['tofit'] += [B.copy()]
                self.B = B
            else: # fitting mu
                mu = np.linalg.inv(np.sum(first_term, axis=0)) @ np.sum(second_term, axis=0) # former eq 20
                self.history['tofit'] += [mu.copy()]
                self.mu = mu

            itercounter += 1
            if verbose:
                print(
                    f'finnished iter {itercounter}\ncurrently eps={self.eps} and llh increase={np.diff(self.history["llh"])[-1]}'
                )

            if break_flag:
                break

        # once the loop finishes keep best params in self.
        best_index = np.argmax(self.history['llh'])
        self.llh = self.history['llh'][best_index]
        if self.factors:
            self.B = self.history['tofit'][best_index]
        else:
            self.mu = self.history['tofit'][best_index]
        self.eps = self.history['eps'][best_index]
        print('all fine?')

    # TODO: add function to save and load previously fitted model from pickle
    def save_model(self, path):
        assert isinstance(str(path),str), 'path is not string-like convertable' # handles posix
        if not path.endswith('.pkl'):
            path += '.csv'

        tosave = self.__dict__.copy()
        for k in ['df', 'prep_data', 'loadpath']: # we do not want to pickle these
            _ = tosave.pop(k, None)
        with open(path, 'wb') as handle:
            pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, path): # initialize with df
        with open(path, 'rb') as handle:
            in_dic = pickle.load(handle)

        if in_dic['subject'] != self.subject:
            warnings.warn(f'instantiazed subject (from df) {self.subject} will be overrided by loaded model subject {in_dic["subject"]}')
        for k, val in in_dic.items():
            setattr(self, k, val)
        print(f'model loaded') # \nmu={self.mu},\n B={self.B}')

    def retrieve_trial_info(self, row, zkonly=True):
        """given a row (trial), retrieve, zk, prior trajectory error and posterior+ from the fitted params
        row is a df row so it can be applied with a lambda function"""
        # perhaps this would be more useful/natural in the other class but as it is auxiliary ideally
        # end user will just interact with ab class | that's no excuse

        r = tinf(row, othermatrices=True, ab_instance=self, time_pre_portout=self.time_pre_portout)
        args =  [
                    r.pose[:,self.dim], # arg list # this could be cleaner.
                    r.N,
                    r.W
                ] + [self.mu] * (not self.factors) +\
                [ r.Fk, self.B ] * self.factors +\
                [self.eps, self.dummy_span] # remaining parameters
        pmk, zk, llh = ab.E_step_worker_(args)
        if zkonly:
            return zk
            
        L = np.linalg.pinv( # shape 6,6
                self.SIGMA_1 + r.N.T @ r.N/self.sigma**2 # args[1] = N
        )
        if not self.factors:
            mu=self.mu  
        else:
            mu = self.B @ r.Fk.reshape(-1,1) # args[3] = F if using factors
        m = L @ (r.N.T @ r.pose[:,self.dim].reshape(-1,1)/self.sigma**2 + self.SIGMA_1 @ mu.reshape(-1,1))
        posterior = r.N @ m
        prior = r.N @ mu 
        se = np.sqrt(np.diag(r.N @ L @ r.N.T))
        return r.tsegment, r.pose[self.dim], posterior, se, prior


def viz_traj_B(self, row, Lobjectpath=None, Robjectpath=None, time=None):
    raise NotImplementedError
    qL=ab(loadpath=Lobjectpath)
    qR=ab(loadpath=Robjectpath)
    assert qL.factorlist==qR.factorlist, f'different factorlists in objects {qL.factorlist} and {qR.factorlist}'
    
    # BL and BR are betas to use interactive plotting function. [size should match with factorlist + itnercept]
    # def retrieve_trial_info(self, row, zkonly=True):
    # """given a row (trial), retrieve, zk, prior trajectory error and posterior+ from the fitted params
    # row is a df row so it can be applied with a lambda function"""
    # # perhaps this would be more useful/natural in the other class but as it is auxiliary ideally
    # end user will just interact with ab class | that's no excuse

    r = tinf(row, othermatrices=True, ab_instance=self, time_pre_portout=self.time_pre_portout)
    argsL =  [
                r.pose[:,self.dim], # arg list # this could be cleaner.
                r.N,
                r.W
            ] + [self.mu] * (not self.factors) +\
            [ r.Fk, self.B ] * self.factors +\
            [self.eps, self.dummy_span] # remaining parameters
    pmk, zk, llh = ab.E_step_worker_(argsL)
    # if zkonly:
    #     return zk
        
    # L = np.linalg.pinv( # shape 6,6
    #         self.SIGMA_1 + r.N.T @ r.N/self.sigma**2 # args[1] = N
    # )
    # if not self.factors:
    #     mu=self.mu  
    # else:
    #     mu = self.B @ r.Fk.reshape(-1,1) # args[3] = F if using factors
    # m = L @ (r.N.T @ r.pose[:,self.dim].reshape(-1,1)/self.sigma**2 + self.SIGMA_1 @ mu.reshape(-1,1))
    # posterior = r.N @ m
    # prior = r.N @ mu 
    # se = np.sqrt(np.diag(r.N @ L @ r.N.T))
    # return r.tsegment, r.pose[self.dim], posterior, se, prior