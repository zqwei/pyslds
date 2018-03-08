import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from __future__ import division
from sklearn.decomposition import FactorAnalysis as factan
from pybasicbayes.util.text import progprint_xrange
from pylds.util import random_rotation
from pyslds.models import DefaultSLDS


# get trial
def getData(mat_contents, nSession):
    nSession = nSession - 1
    totTargets = nDataSet[0, nSession][7].flatten()
    unit_yes_trial = nDataSet[0, nSession][5]
    unit_no_trial = nDataSet[0, nSession][6]
    unit_trial = np.concatenate((unit_yes_trial, unit_no_trial))
    print("Loading data for Session #%d".{nSession})
    return unit_trial, totTargets


def getSLDSFit(unit_trial, K, xDim):
    # K : number of discrete state of z
    # xDim : number of latent dimensions
    numTrial, numUnit, numTime = unit_trial.shape
    factor_unit_trial = unit_trial.transpose([0, 2, 1])
    factor_unit_trial = factor_unit_trial.reshape([-1, factor_unit_trial.shape[2]])
    yDim = numUnit
    inputDim = 1 # some constants
    inputs = np.ones((numTime, inputDim))

    # factor analysis as initialization
    # Use FA or PCA to initialize C_init and D_init
    estimator = factan(n_components=xDim, tol=0.000001, copy=True,
                       max_iter=1000, noise_variance_init=None,
                       svd_method='randomized', iterated_power=3,
                       random_state=None)

    estimator.fit(factor_unit_trial)
    C_init = estimator.components_.T
    D_init = estimator.mean_

    # To support different emission matrices for each discrete state, pass in
    Cs=[C_init.copy() for _ in range(numTrial)]
    Ds=[D_init.copy() for _ in range(numTrial)]

    # SLDS model fit
    test_model = DefaultSLDS(K, yDim, xDim, inputDim, Cs=Cs, Ds=Ds)
    for trial in range(numTrial):
        test_model.add_data(unit_trial[trial].T, inputs=inputs)

    print("Initializing with Gibbs")




if __name__ == '__main__':
    main()
