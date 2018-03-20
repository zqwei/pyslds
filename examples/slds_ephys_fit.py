from __future__ import division
import scipy.io as sio
import numpy as np
from copy import deepcopy
import sys
import os

from sklearn.decomposition import FactorAnalysis as factan
from pybasicbayes.util.text import progprint_xrange
from pyslds.models import HMMSLDS
from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression

def loadfile(fileName, nSession):
    mat_contents = sio.loadmat(fileName)
    nDataSet = mat_contents['nDataSet']
    params = mat_contents['params']

    nSession = 17 # matlab index
    nSession = nSession - 1
    totTargets = nDataSet[0, nSession][7].flatten()
    unit_yes_trial = nDataSet[0, nSession][5]
    unit_no_trial = nDataSet[0, nSession][6]
    unit_trial = np.concatenate((unit_yes_trial, unit_no_trial))

    return unit_trial


def getData(unit_trial):
    numTrial, _, _ = unit_trial.shape
    trainIndex = np.random.rand(numTrial)<0.9
    Ytrain = unit_trial[trainIndex, :, :]
    Yvalid = unit_trial[~trainIndex, :, :]
    return Ytrain, Yvalid

def trainModel(fileName, unit_trial, K=8, xDim=5):
    # unit_trial -- training data
    # randomization
    np.random.seed()

    numTrial, numUnit, numTime = unit_trial.shape

    # factor analysis for initialization
    factor_unit_trial = unit_trial.transpose([0, 2, 1])
    factor_unit_trial = factor_unit_trial.reshape([-1, factor_unit_trial.shape[2]])
    yDim = numUnit
    inputDim = 1 # some constants
    inputs = np.ones((numTime, inputDim))
    estimator = factan(n_components=xDim, tol=0.00001, copy=True,
                       max_iter=10000, noise_variance_init=None,
                       svd_method='randomized', iterated_power=3,
                       random_state=None)
    estimator.fit(factor_unit_trial)
    C_init = estimator.components_.T
    D_init = estimator.mean_.reshape([-1, 1])

    # SLDS fit
    init_dynamics_distns = [Gaussian(nu_0=xDim+3,
                                     sigma_0=3.*np.eye(xDim),
                                     mu_0=np.zeros(xDim),
                                     kappa_0=0.01)
                            for _ in range(K)]

    dynamics_distns = [Regression(nu_0=xDim + 1,
                                  S_0=xDim * np.eye(xDim),
                                  M_0=np.hstack((.99 * np.eye(xDim), np.zeros((xDim, inputDim)))),
                                  K_0=xDim * np.eye(xDim + inputDim))
                       for _ in range(K)]

    As = [np.eye(xDim) for _ in range(K)]
    if inputDim > 0:
        As = [np.hstack((A, np.zeros((xDim, inputDim))))
              for A in As]
    for dd, A in zip(dynamics_distns, As):
        dd.A = A

    sigma_statess = [np.eye(xDim) for _ in range(K)]
    for dd, sigma in zip(dynamics_distns, sigma_statess):
        dd.sigma = sigma

    emission_distns = [DiagonalRegression(yDim, xDim + inputDim,
                                          mu_0=None, Sigma_0=None,
                                          alpha_0=3.0, beta_0=2.0,
                                          A=np.hstack((C_init.copy(), D_init.copy())),
                                          sigmasq=None, niter=1)
                       for _ in range(K)]

    train_model = HMMSLDS(
        init_dynamics_distns= init_dynamics_distns,
        dynamics_distns= dynamics_distns,
        emission_distns= emission_distns,
        alpha=3., init_state_distn='uniform')

    # Adding training data
    for trial in range(numTrial):
        train_model.add_data(unit_trial[trial].T, inputs=inputs)

    print("Initializing with Gibbs")
    N_gibbs_samples = 2000
    def initialize(model):
        model.resample_model()
        return model.log_likelihood()

    gibbs_lls = [initialize(train_model) for _ in progprint_xrange(N_gibbs_samples)]

    print("Fitting with VBEM")
    N_vbem_iters = 100
    def update(model):
        model.VBEM_step()
        return model.log_likelihood()

    train_model._init_mf_from_gibbs()
    vbem_lls = [update(train_model) for _ in progprint_xrange(N_vbem_iters)]

    np.save(fileName + '_gibbs_lls', gibbs_lls)
    np.save(fileName + '_vbem_lls', vbem_lls)
    np.save(fileName + '_train_model', train_model)

    return train_model


def LONOresults(fileName, train_model, unit_trial):
    # unit_trial -- test data
    test_model = deepcopy(train_model)
    test_model.states_list = []

    numTrial, numUnit, numTime = unit_trial.shape
    y_est = unit_trial.copy()
    inputDim = 1 # some constants
    inputs = np.ones((numTime, inputDim))

    for nUnit in range(numUnit):
        mask = np.ones((numTime, numUnit), dtype=bool)
        # leave out the last neuron
        mask[:,nUnit] = False
        # add the test data
        for trial in range(numTrial):
            test_model.add_data(unit_trial[trial].T, inputs=inputs, mask=mask)
            test_states = test_model.states_list[trial]
            for intr in range(100):
                test_states.resample()
            # get the latent states
            z_test = test_states.stateseq
            x_test = test_states.gaussian_states
            Cs = test_states.Cs
            Ds = test_states.Ds
            y_est_ = np.array([Cs[_,:,:].dot(x_test[_,:]).reshape(-1, 1)+Ds[_,:,:] for _ in range(numTime)]).squeeze()
            y_est[trial, nUnit, :] = y_est_[:, nUnit]
    np.save(fileName + '_test_y_est', y_est)
    np.save(fileName + '_test_y', unit_trial)



def main():
    fileName = sys.argv[1]
    nSession = int(sys.argv[2])
    K = int(sys.argv[3])
    xDim = int(sys.argv[4])
    numFold = 10
    unit_trial = loadfile(fileName=fileName, nSession=nSession)
    for nFold in range(numFold):
        Ytrain, Yvalid = getData(unit_trial=unit_trial)
        saveFileName = os.path.basename(fileName)
        saveFileName = os.path.splitext(saveFileName)[0]
        saveFileName = saveFileName + '_Session_%02d_K_%02d_xDim_%02d_nFold_%02d'%(nSession, K, xDim, nFold)
        print('save to file --- %s'%(saveFileName))
        train_model = trainModel(saveFileName, Ytrain, K=K, xDim=xDim)
        LONOresults(saveFileName, train_model, Yvalid)

if __name__ == '__main__':
    main()
