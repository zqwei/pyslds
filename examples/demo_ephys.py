from __future__ import division
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FactorAnalysis as factan
from pybasicbayes.util.text import progprint_xrange
from pylds.util import random_rotation
from pyslds.models import DefaultSLDS

mat_contents = sio.loadmat('/Users/dalin/Documents/My_Research/Dataset/ALM_Svoboda_Lab_Data/Code/TLDS/TempDat/Simultaneous_Spikes.mat')
nDataSet = mat_contents['nDataSet']
params = mat_contents['params']

nSession = 17 # matlab index
nSession = nSession - 1
totTargets = nDataSet[0, nSession][7].flatten()
unit_yes_trial = nDataSet[0, nSession][5]
unit_no_trial = nDataSet[0, nSession][6]


unit_trial = np.concatenate((unit_yes_trial, unit_no_trial))
numTrial, numUnit, numTime = unit_trial.shape

factor_unit_trial = unit_trial.transpose([0, 2, 1])
factor_unit_trial = factor_unit_trial.reshape([-1, factor_unit_trial.shape[2]])
np.random.seed(12345678)
# transition
K = 8
yDim = numUnit
xDim = 5
inputDim = 1 # some constants
inputs = np.ones((numTime, inputDim))

estimator = factan(n_components=xDim, tol=0.000001, copy=True, 
                   max_iter=1000, noise_variance_init=None, 
                   svd_method='randomized', iterated_power=3, 
                   random_state=None)


estimator.fit(factor_unit_trial)
C_init = estimator.components_.T
D_init = estimator.mean_

Cs=[C_init.copy() for _ in range(K)]
Ds=[D_init.copy().reshape([-1, 1]) for _ in range(K)]

test_model = DefaultSLDS(K, yDim, xDim, inputDim,
                         Cs=Cs,
                         Ds=Ds)
for trial in range(numTrial):
    test_model.add_data(unit_trial[trial].T, inputs=inputs)
    
print("Initializing with Gibbs")
N_gibbs_samples = 10
def initialize(model):
    model.resample_model()
    return model.log_likelihood()

gibbs_lls = [initialize(test_model) for _ in progprint_xrange(N_gibbs_samples)]

# Fit with VBEM
print("Fitting with VBEM")
N_vbem_iters = 10
def update(model):
    model.VBEM_step()
    return model.log_likelihood()

test_model.states_list[0]._init_mf_from_gibbs()
vbem_lls = [update(test_model) for _ in progprint_xrange(N_vbem_iters)]    


