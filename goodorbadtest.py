# constraintless again but with one GP please help
## import libs
#common libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
import pickle
#torch libs
import torch
import torch.nn as nn

#gpytorch libs
import gpytorch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel, Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood 

# BoTorch
from botorch.models import SingleTaskGP # No longer used
from botorch.fit import fit_gpytorch_mll # No longer used
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms import Normalize, Standardize


#scipy
from scipy.interpolate import griddata
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


#tcpython

# user settings for TC
HEAT_SOURCE_NAME = "Double ellipsoidal - 316L - beam d 15um"
MATERIAL_NAME = "SS316L"
POWDER_THICKNESS = 10.0  # in micro-meter
HATCH_DISTANCE = 10.0  # in micro-meter. Only single track experiment but let's use a hatch spacing for the printability map
AMBIENT_TEMPERATURE = 353  # in K, as given in the paper
USE_BALLING = True
USE_EDENSITY   = False  
USE_EDGE_PENALTY = True
dtype = torch.double

# active learning settings
ntrain = 10
niter = 20
plotiter = 5
ngrid = 100
nmc = 128
restarts = 10
edensity_high =10000000
SEED = 12
np.random.seed(SEED)
torch.manual_seed(SEED)
device= torch.device("cpu")

#import data
filename = "results_progress.csv"
data = pd.read_csv(filename)

power = data['Power'].values.reshape(-1, 1)
speed = data['Speed'].values.reshape(-1, 1)

depth  = data["Depth"].values.reshape(-1, 1)
width  = data["Width"].values.reshape(-1, 1)
length = data["Length"].values.reshape(-1, 1)
power  = data["Power"].values.reshape(-1, 1)
speed  = data["Speed"].values.reshape(-1, 1)

# training data
n_total = len(depth)

initial_indices = np.random.choice(range(n_total), size=ntrain, replace=False)
train_x = np.hstack((power[initial_indices], speed[initial_indices]))

depth_init = depth[initial_indices]
width_init = width[initial_indices]
length_init = length[initial_indices]

#classify defects/good based on criteria
def classify_samples(depth, width, length, power, speed):
    # 0 is good 1 is bad
    keyholing = 1.5
    lof = 1.9
    balling = 0.23
    thickness = POWDER_THICKNESS
    jitter = 1e-9
    if depth == 0 or (width/(depth + jitter)) < keyholing or ((depth)/(thickness)) < lof or (width/(length + jitter)) < balling:
        return 1  # lack of fusion
    else:
        return 0  # good
train_y = []
for i in range(ntrain):
    label = classify_samples(depth_init[i], width_init[i], length_init[i], train_x[i, 0], train_x[i, 1])
    train_y.append(label)
train_y = np.array(train_y).reshape(-1, 1)


xtrain = torch.tensor(train_x, dtype=dtype)
ytrain = torch.tensor(train_y, dtype=dtype)
# train GP
def FitGP (xtrain, ytrain, restarts):
    # convert to torch tensors
    train_x = torch.tensor(xtrain, dtype=dtype)
    train_y = torch.tensor(ytrain.flatten(), dtype=dtype)

    # define likelihood and model
    kernel = ScaleKernel(MaternKernel(nu=2.5), ard_num_dims=2)
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1), covar_module=kernel,
                         input_transform=Normalize(train_x.shape[1]),
        outcome_transform=Standardize(1))

    #fit gp
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={'maxiter': 1000}, restarts=restarts)
    return model

#evaluate GP
def evaluate_gp_models(gp, X, Y_targets, n_samples=10, label=""):
    preds, rmses = {}, {}

    with torch.no_grad():
        posterior = gp.posterior(X)
        y_pred = posterior.mean.cpu().numpy().ravel()
        y_true = Y_targets.cpu().numpy().ravel()
        preds = y_pred
        rmses = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"\nEvaluation {label}")
    print("RMSE per task:")
    print ("RMSE:", rmses)

    print("\nSample predictions:")
    for i in range(min(n_samples, len(X))):
        true_vals = [f"{Y_targets[i].item():.2f}"]
        pred_vals = [f"{preds[i]:.2f}"]
        print(f"  True: {true_vals} | Pred: {pred_vals}")

    return preds, rmses


## def acq func
def entropy_sigma (Xgrid, gp, nmc, Xtrain, topk=10):
    posterior = gp.posterior(Xgrid)
    mean = posterior.mean
    std = posterior.variance.sqrt()

    # monte carlo samples
    with torch.no_grad():
        samples = posterior.rsample(torch.Size([nmc])).squeeze(-1)
    
    # calculate entropy sigma
    H = -(samples.mean(0) * torch.log(samples.mean(0) + 1e-9) + (1 - samples.mean(0)) * torch.log(1 - samples.mean(0) + 1e-9))
    sigma = std
    J = H * sigma

    # distance penalty
    if USE_EDGE_PENALTY:
        dist_penalty = torch.zeros(Xgrid.shape[0], dtype=dtype)
        for i in range(Xgrid.shape[0]):
            dists = torch.norm(Xtrain - Xgrid[i, :], dim=1)
            min_dist = torch.min(dists)
            dist_penalty[i] = 1.0 / (min_dist + 1e-9)
        J = J - 0.1 * dist_penalty
        return J
    else:
        return J

# good or bad test main loop
powermin, powermax = xtrain[:, 0].min().item(), xtrain[:, 0].max().item()
velomin, velomax = xtrain[:, 1].min().item(), xtrain[:, 1].max().item()

powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij")

gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
Xgrid = torch.tensor(gridpoints, dtype=dtype, device=device)

#init TC python

