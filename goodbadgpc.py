# constraintless again but with one GP please help
## import libs
#common libs
import gc
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
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO

#tcpython
from tc_python import *
# user settings for TC
HEAT_SOURCE_NAME = "Double ellipsoidal - 316L - beam d 15um"
MATERIAL_NAME = "SS316L"
POWDER_THICKNESS = 10.0  # in micro-meter
HATCH_DISTANCE = 10.0  # in micro-meter. Only single track experiment but let's use a hatch spacing for the printability map
AMBIENT_TEMPERATURE = 353  # in K, as given in the paper
USE_BALLING = True
dtype = torch.double

# active learning settings
ntrain = 10
niter = 20
plotiter = 5
ngrid = 100
restarts = 10
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

# training data
n_total = len(depth)
initial_indices = np.random.choice(range(n_total), size=ntrain, replace=False)
train_x = np.hstack((power[initial_indices], speed[initial_indices]))
depth_init = depth[initial_indices]
width_init = width[initial_indices]
length_init = length[initial_indices]

#classify defects/good based on criteria
def classify_samples(depth, width, length, power, speed):
    keyholing = 1.5
    lof = 1.9
    balling = 0.23
    thickness = POWDER_THICKNESS
    jitter = 1e-9
    if depth == 0 or (width/(depth + jitter)) < keyholing or ((depth)/(thickness)) < lof or (width/(length + jitter)) < balling:
        return 1
    else:
        return 0
train_y = np.array([classify_samples(d, w, l, train_x[i,0], train_x[i,1]) for i, (d,w,l) in enumerate(zip(depth_init, width_init, length_init))]).reshape(-1, 1)

xtrain = torch.tensor(train_x, dtype=dtype)
ytrain = torch.tensor(train_y, dtype=dtype).flatten() # GPC expects a 1D tensor

# --- GPC Model Definition ---
class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=False)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

# --- REVISED: GPC Fitting Function with Manual Normalization ---
def FitGPC(xtrain, ytrain):
    # Manually normalize inputs
    train_mean = xtrain.mean(dim=0, keepdim=True)
    train_std = xtrain.std(dim=0, keepdim=True)
    train_x_normalized = (xtrain - train_mean) / train_std

    model = GPClassificationModel(train_x_normalized)
    likelihood = BernoulliLikelihood()

    model.to(xtrain.dtype)
    likelihood.to(xtrain.dtype)

    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = VariationalELBO(likelihood, model, num_data=ytrain.size(0))

    # Convert targets from {0, 1} to {-1, 1} for training
    ytrain_transformed = ytrain.clone()
    ytrain_transformed[ytrain_transformed == 0] = -1

    for i in range(100): # Training loop
        optimizer.zero_grad()
        output = model(train_x_normalized)
        loss = -mll(output, ytrain_transformed)
        loss.backward()
        optimizer.step()
        
    model.eval()
    likelihood.eval()
    
    return model, likelihood, train_mean, train_std

#evaluate GP
def evaluate_gp_models(model, likelihood, train_mean, train_std, X, Y_targets, n_samples=10, label=""):
    with torch.no_grad():
        X_normalized = (X - train_mean) / train_std
        posterior = likelihood(model(X_normalized))
        y_pred = posterior.mean.cpu().numpy().ravel()
        y_true = Y_targets.cpu().numpy().ravel()
        
        pred_labels = (y_pred > 0.5).astype(int)
        accuracy = np.mean(pred_labels == y_true) * 100

    print(f"\nEvaluation {label}")
    print (f"Accuracy: {accuracy:.2f}%")

    print("\nSample predictions (Probability of Defect):")
    for i in range(min(n_samples, len(X))):
        print(f"  True: {int(y_true[i])} | Pred Prob: {y_pred[i]:.2f}")

# Acquisition Function for GPC
def max_uncertainty_acq(model, likelihood, train_mean, train_std, Xgrid, topk=10):
    print("Calculating Maximum Uncertainty acquisition function")
    with torch.no_grad():
        Xgrid_normalized = (Xgrid - train_mean) / train_std
        posterior = likelihood(model(Xgrid_normalized))
        p_mean = posterior.mean.clamp(1e-6, 1.0 - 1e-6)
        
        uncertainty = 1.0 - torch.abs(p_mean - 0.5) * 2.0
        
        J = uncertainty.squeeze(-1).cpu().numpy()
        top_indices = np.argsort(-J)[:topk]
        return J, top_indices

#plotting func 
def plot_gp_and_acq(model, likelihood, train_mean, train_std, Xgrid, xtrain_np, powergrid, velogrid, J=None, iteration=None):
    if J is not None: fig, axes = plt.subplots(1, 2, figsize=(16, 7)); ax_gp, ax_acq = axes
    else: fig, ax_gp = plt.subplots(figsize=(8, 7)); ax_acq = None
    ngrid = len(powergrid)
    with torch.no_grad():
        Xgrid_normalized = (Xgrid - train_mean) / train_std
        posterior = likelihood(model(Xgrid_normalized))
        p_mean = posterior.mean.cpu().numpy().reshape(ngrid, ngrid)
    im_gp = ax_gp.imshow(p_mean, origin="lower", extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()], cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    if xtrain_np.shape[0] > 1: ax_gp.scatter(xtrain_np[:-1, 1], xtrain_np[:-1, 0], c='black', marker='x', s=50, label='Previous Samples')
    ax_gp.scatter(xtrain_np[-1, 1], xtrain_np[-1, 0], c='lime', marker='*', s=150, edgecolor='black', label='Latest Sample')
    ax_gp.set_xlabel("Scan Velocity (mm/s)"); ax_gp.set_ylabel("Laser Power (W)"); title_gp = f"GPC Predicted Probability of Defect - Iter {iteration}"; ax_gp.set_title(title_gp); ax_gp.legend(loc="best"); fig.colorbar(im_gp, ax=ax_gp, label='P(Defect)')
    if J is not None and ax_acq is not None:
        grid_J = J.reshape(ngrid, ngrid)
        im_acq = ax_acq.imshow(grid_J, origin="lower", extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()], cmap='viridis', aspect='auto')
        if xtrain_np.shape[0] > 1: ax_acq.scatter(xtrain_np[:-1, 1], xtrain_np[:-1, 0], c='black', marker='x', s=50, label='Previous Samples')
        ax_acq.scatter(xtrain_np[-1, 1], xtrain_np[-1, 0], c='lime', marker='*', s=150, edgecolor='black', label='Latest Sample')
        ax_acq.set_xlabel("Scan Velocity (mm/s)"); ax_acq.set_ylabel("Laser Power (W)"); title_acq = f"Acquisition Function - Iter {iteration}"; ax_acq.set_title(title_acq); ax_acq.legend(loc="best"); fig.colorbar(im_acq, ax=ax_acq, label='Acquisition Value')
    plt.tight_layout(); plt.savefig(f"iteration_{iteration}.png"); print(f"Saved plot to iteration_{iteration}.png"); plt.close(fig)
    
#train initial GPC
model, likelihood, train_mean, train_std = FitGPC(xtrain, ytrain)
evaluate_gp_models(model, likelihood, train_mean, train_std, xtrain, ytrain, n_samples=10, label="Initial GPC on training data")

# Main loop setup
powermin, powermax = xtrain[:, 0].min().item(), xtrain[:, 0].max().item()
velomin, velomax = xtrain[:, 1].min().item(), xtrain[:, 1].max().item()
powergrid = np.linspace(powermin, powermax, ngrid); velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij"); gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
Xgrid = torch.tensor(gridpoints, dtype=dtype, device=device)

# --- Initial Plot (Iteration 0) ---
print("\n--- Generating initial plot (Iteration 0) ---")
J_initial, _ = max_uncertainty_acq(model, likelihood, train_mean, train_std, Xgrid, topk=3)
plot_gp_and_acq(model, likelihood, train_mean, train_std, Xgrid, xtrain.cpu().numpy(), powergrid, velogrid, J=J_initial, iteration=0)

# Active learning loop
success_it = 0
while success_it < niter:
    J, top_candidates = max_uncertainty_acq(model, likelihood, train_mean, train_std, Xgrid, topk=3)
    x_next, d_next, w_next, l_next = None, None, None, None
    with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
        start.set_cache_folder("cache")
        mp = MaterialProperties.from_library(MATERIAL_NAME)
        for ind in top_candidates:
            try:
                candidate_x = Xgrid[ind:ind + 1]
                am_calculator = start.with_additive_manufacturing().with_steady_state_calculation().with_numerical_options(NumericalOptions().set_number_of_cores(20)).disable_fluid_flow_marangoni().with_material_properties(mp).with_mesh(Mesh().coarse())
                am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE); am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)
                heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
                heat_source.set_power(float(candidate_x[0, 0].item())); heat_source.set_scanning_speed(float(candidate_x[0, 1].item()) / 1e3)
                am_calculator.with_heat_source(heat_source)
                result: SteadyStateResult = am_calculator.calculate()
                d_next = float(result.get_meltpool_depth()) * 1e6; w_next = float(result.get_meltpool_width()) * 1e6; l_next = float(result.get_meltpool_length()) * 1e6; x_next = candidate_x
                del result, am_calculator, heat_source; gc.collect(); break 
            except tc_python.exceptions.CalculationException: print(f"[trial] Thermo-Calc failed at candidate {ind}, trying next-best..."); continue
    if x_next is None: print(f"[trial] Top 3 candidates failed, retrying without incrementing iteration."); continue
    
    y_label = classify_samples(d_next, w_next, l_next, x_next[0, 0].item(), x_next[0, 1].item())
    xtrain = torch.cat([xtrain, x_next.to(device)], dim=0)
    ytrain = torch.cat([ytrain, torch.tensor([y_label], dtype=dtype, device=device)], dim=0)
    
    model, likelihood, train_mean, train_std = FitGPC(xtrain, ytrain)
    success_it += 1

    print(f"\n--- Iteration {success_it}/{niter} Complete ---")
    evaluate_gp_models(model, likelihood, train_mean, train_std, xtrain, ytrain, n_samples=10, label=f"GPC after iteration {success_it}")
    print(f"Sampled: P={float(x_next[0,0]):.2f}, V={float(x_next[0,1]):.2f}, Resulting Status={y_label}\n" + "-"*35)
    
    if success_it % plotiter == 0 or success_it == niter:
        print(f"\n--- Generating plot for iteration {success_it} ---")
        J_for_plot, _ = max_uncertainty_acq(model, likelihood, train_mean, train_std, Xgrid, topk=3)
        plot_gp_and_acq(model, likelihood, train_mean, train_std, Xgrid, xtrain.cpu().numpy(), powergrid, velogrid, J=J_for_plot, iteration=success_it)

print("Active learning loop finished.")