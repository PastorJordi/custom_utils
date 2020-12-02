import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

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

    def __init__(self, row, kwargs={'inv_method':'pinv'}, sigma=5, var=None, mu=default_mu):
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
        mx = L @ (N.T @ coords[:,0].reshape(-1,1)/sigma**2 + SIGMA @ mu[:,0].reshape(-1,1)) # shape 6,1
        my = L @ (N.T @ coords[:,1].reshape(-1,1)/sigma**2 + SIGMA @ mu[:,1].reshape(-1,1))
        return W, mx, my, L

    @staticmethod
    def n_(t, M):
        # t is a scalar, right?
        vt = (np.array([t]*6) ** np.arange(6)).reshape(1,-1)
        return vt @ np.linalg.inv(M)

    # @staticmethod
    # def hyperparams_llh(eps=0.05):
    #     LLH = np.sum( # across k trials
    #         np.log()
    #     )

    # N es la matrix n x 6 con los valores de t con observación
    # n(t) es un vector de length 6 que es para un t dado (así te da toda la trayectoría continua)
    # et n(t)m en la eq 11 te da el polinomio de la mean trajectory

    # perdón, era n(t) y no m(t) # final de eq 11
