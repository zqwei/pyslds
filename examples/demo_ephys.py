from __future__ import division
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.decomposition import FactorAnalysis as factan
from pybasicbayes.util.text import progprint_xrange
from pyslds.models import HMMSLDS
from pybasicbayes.distributions import DiagonalRegression, Gaussian, Regression

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

np.random.seed()
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
D_init = estimator.mean_.reshape([-1, 1])


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

for trial in range(numTrial):
    train_model.add_data(unit_trial[trial].T, inputs=inputs)


print("Initializing with Gibbs")
N_gibbs_samples = 1000
def initialize(model):
    model.resample_model()
    return model.log_likelihood()

gibbs_lls = [initialize(train_model) for _ in progprint_xrange(N_gibbs_samples)]


print("Fitting with VBEM")
N_vbem_iters = 1000
def update(model):
    model.VBEM_step()
    return model.log_likelihood()

train_model._init_mf_from_gibbs()
vbem_lls = [update(train_model) for _ in progprint_xrange(N_vbem_iters)]


test_model = deepcopy(train_model)
test_model.states_list = []

mask = np.ones((numTime, numUnit), dtype=bool)
# leave out the last neuron
mask[:,-1] = False
# add the test data
for trial in range(numTrial):
    test_model.add_data(unit_trial[trial].T, inputs=inputs, mask=mask)

test_states = test_model.states_list[0]


for intr in range(100):
    test_states.resample()
# get the latent states
z_test = test_states.stateseq
x_test = test_states.gaussian_states
Cs = test_states.Cs
Ds = test_states.Ds
y_est = np.array([Cs[_,:,:].dot(x_test[_,:]).reshape(-1, 1)+Ds[_,:,:] for _ in range(numTime)]).squeeze()