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
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

# BoTorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize

#tcpython
from tc_python import *
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
    train_x = torch.tensor(xtrain, dtype=dtype)
    train_y = torch.tensor(ytrain.flatten(), dtype=dtype)
    kernel = ScaleKernel(MaternKernel(nu=2.5), ard_num_dims=2)
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1), covar_module=kernel,
                         input_transform=Normalize(train_x.shape[1]),
        outcome_transform=Standardize(1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, options={'maxiter': 1000}, restarts=restarts)
    return model

#evaluate GP
def evaluate_gp_models(gp, X, Y_targets, n_samples=10, label=""):
    with torch.no_grad():
        posterior = gp.posterior(X)
        y_pred = posterior.mean.cpu().numpy().ravel()
        y_true = Y_targets.cpu().numpy().ravel()
        rmses = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"\nEvaluation {label}")
    print ("RMSE:", rmses)
    print("\nSample predictions:")
    for i in range(min(n_samples, len(X))):
        true_vals = [f"{Y_targets[i].item():.2f}"]
        pred_vals = [f"{y_pred[i]:.2f}"]
        print(f"  True: {true_vals} | Pred: {pred_vals}")

## def acq func
def entropy_sigma_single_task(Xgrid, gp, Xtrain, topk=10, alpha_dist=0.1):
    print("Calculating acquisition function")
    with torch.no_grad():
        posterior = gp.posterior(Xgrid)
        p_mean = posterior.mean.clamp(1e-6, 1.0 - 1e-6)
        sigma_p = posterior.variance.sqrt().clamp(min=1e-6)
        H = -(p_mean * torch.log2(p_mean) + (1.0 - p_mean) * torch.log2(1.0 - p_mean))
        J_tensor = (H * sigma_p).squeeze(-1)

        if USE_EDGE_PENALTY and Xtrain is not None and Xtrain.shape[0] > 0:
            min_dists = torch.cdist(Xgrid, Xtrain).min(dim=1).values
            dist_reward = min_dists
            J_tensor = J_tensor * (1.0 + alpha_dist * dist_reward)
                
        J = J_tensor.cpu().numpy()
        top_indices = np.argsort(-J)[:topk]
        return J, top_indices

#plotting func 
def plot_gp_and_acq(gp, Xgrid, xtrain_np, powergrid, velogrid, J=None, iteration=None):
    if J is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax_gp, ax_acq = axes
    else:
        fig, ax_gp = plt.subplots(figsize=(8, 7))
        ax_acq = None

    ngrid = len(powergrid)
    
    with torch.no_grad():
        posterior = gp.posterior(Xgrid)
        p_mean = posterior.mean.cpu().numpy().reshape(ngrid, ngrid)

    im_gp = ax_gp.imshow(p_mean, origin="lower",
                       extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()],
                       cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    
    if xtrain_np.shape[0] > 1:
        ax_gp.scatter(xtrain_np[:-1, 1], xtrain_np[:-1, 0], c='black', marker='x', s=50, label='Previous Samples')
    ax_gp.scatter(xtrain_np[-1, 1], xtrain_np[-1, 0], c='lime', marker='*', s=150, edgecolor='black', label='Latest Sample')

    ax_gp.set_xlabel("Scan Velocity (mm/s)")
    ax_gp.set_ylabel("Laser Power (W)")
    title_gp = "GP Predicted Probability of Defect"
    if iteration is not None:
        title_gp += f" - Iter {iteration}"
    ax_gp.set_title(title_gp)
    ax_gp.legend(loc="best")
    fig.colorbar(im_gp, ax=ax_gp, label='P(Defect)')

    if J is not None and ax_acq is not None:
        grid_J = J.reshape(ngrid, ngrid)
        im_acq = ax_acq.imshow(grid_J, origin="lower",
                           extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()],
                           cmap='viridis', aspect='auto')
        
        if xtrain_np.shape[0] > 1:
            ax_acq.scatter(xtrain_np[:-1, 1], xtrain_np[:-1, 0], c='black', marker='x', s=50, label='Previous Samples')
        ax_acq.scatter(xtrain_np[-1, 1], xtrain_np[-1, 0], c='lime', marker='*', s=150, edgecolor='black', label='Latest Sample')

        ax_acq.set_xlabel("Scan Velocity (mm/s)")
        ax_acq.set_ylabel("Laser Power (W)")
        title_acq = "Acquisition Function"
        if iteration is not None:
            title_acq += f" - Iter {iteration}"
        ax_acq.set_title(title_acq)
        ax_acq.legend(loc="best")
        fig.colorbar(im_acq, ax=ax_acq, label='Acquisition Value')

    plt.tight_layout()
    if iteration is not None:
        plt.savefig(f"iteration_{iteration}.png")
        print(f"Saved plot to iteration_{iteration}.png")
    plt.close(fig)
    
#train initial GP
model = FitGP(xtrain, ytrain, restarts)
evaluate_gp_models(model, xtrain, ytrain, n_samples=10, label="Initial GP on training data")

# good or bad test main loop
powermin, powermax = xtrain[:, 0].min().item(), xtrain[:, 0].max().item()
velomin, velomax = xtrain[:, 1].min().item(), xtrain[:, 1].max().item()

powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij")
gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
Xgrid = torch.tensor(gridpoints, dtype=dtype, device=device)

# --- Initial Plot (Iteration 0) ---
print("\n--- Generating initial plot (Iteration 0) ---")
J_initial, _ = entropy_sigma_single_task(Xgrid, model, xtrain, topk=3, alpha_dist=0.1)
plot_gp_and_acq(
    gp=model, Xgrid=Xgrid, xtrain_np=xtrain.cpu().numpy(), 
    powergrid=powergrid, velogrid=velogrid, J=J_initial, iteration=0
)

# Active learning loop
success_it = 0
while success_it < niter:
    J, top_candidates = entropy_sigma_single_task(Xgrid, model, xtrain, topk=3, alpha_dist=0.1)
    
    x_next = None
    d_next, w_next, l_next = None, None, None

    with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
        start.set_cache_folder("cache")
        mp = MaterialProperties.from_library(MATERIAL_NAME)
        for ind in top_candidates:
            try:
                candidate_x = Xgrid[ind:ind + 1]
                am_calculator = (
                    start.with_additive_manufacturing().with_steady_state_calculation()
                    .with_numerical_options(NumericalOptions().set_number_of_cores(20))
                    .disable_fluid_flow_marangoni().with_material_properties(mp)
                    .with_mesh(Mesh().coarse())
                )
                am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE)
                am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)
                heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
                heat_source.set_power(float(candidate_x[0, 0].item()))
                heat_source.set_scanning_speed(float(candidate_x[0, 1].item()) / 1e3)
                am_calculator.with_heat_source(heat_source)
                result: SteadyStateResult = am_calculator.calculate()

                d_next = float(result.get_meltpool_depth()) * 1e6
                w_next = float(result.get_meltpool_width()) * 1e6
                l_next = float(result.get_meltpool_length()) * 1e6
                x_next = candidate_x
                
                del result, am_calculator, heat_source
                gc.collect()
                break 
            except tc_python.exceptions.CalculationException:
                print(f"[trial] Thermo-Calc failed at candidate {ind}, trying next-best...")
                continue

    if x_next is None:
        print(f"[trial] Top 3 candidates failed, retrying without incrementing iteration.")
        continue

    y_label = classify_samples(d_next, w_next, l_next, x_next[0, 0].item(), x_next[0, 1].item())
    
    xtrain = torch.cat([xtrain, x_next.to(device)], dim=0)
    ytrain = torch.cat([ytrain, torch.tensor([[y_label]], dtype=dtype, device=device)], dim=0)

    model = FitGP(xtrain.cpu().numpy(), ytrain.cpu().numpy(), restarts)
    success_it += 1
    
    print(f"\n--- Iteration {success_it}/{niter} Complete ---")
    evaluate_gp_models(model, xtrain, ytrain, n_samples=10, label=f"GP after iteration {success_it}")
    print(f"Sampled: P={float(x_next[0,0]):.2f}, V={float(x_next[0,1]):.2f}, Resulting Status={y_label}")
    print("---------------------------------\n")

    # --- Plotting logic at the END of the iteration ---
    if success_it % plotiter == 0 or success_it == niter:
        print(f"\n--- Generating plot for iteration {success_it} ---")
        J_for_plot, _ = entropy_sigma_single_task(Xgrid, model, xtrain, topk=3, alpha_dist=0.1)
        plot_gp_and_acq(
            gp=model, Xgrid=Xgrid, xtrain_np=xtrain.cpu().numpy(), 
            powergrid=powergrid, velogrid=velogrid, J=J_for_plot, iteration=success_it
        )

print("Active learning loop finished.")