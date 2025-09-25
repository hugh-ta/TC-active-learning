# --- ACS Debug Version: Fast Execution, Parallelized with Metrics --- #
import pandas as pd
import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from joblib import Parallel, delayed

# ---------------- Setup ---------------- #
dtype = torch.double
DEVICE = torch.device("cpu")
torch.set_default_dtype(dtype)
SEED = 8
torch.manual_seed(SEED)
np.random.seed(SEED)

# Debug parameters
ngrid = 100
Ntrain = 10
Niter = 40
nmc = 64
maternmu = 2.5
restarts = 2
thickness = 50
keyholing = 1.5
lof = 1.5
balling = 3.8
constraints = [keyholing, lof, balling]
n_cores = 16

# ---------------- Load & Prepare Data ---------------- #
df = pd.read_csv("SS417 data.csv")
X_np = df[["Power", "Velocity"]].values
X_all = torch.tensor(X_np, dtype=dtype, device=DEVICE)
Yw_all = torch.tensor(df["width of melt pool"].values.reshape(-1, 1), dtype=dtype, device=DEVICE)
Yl_all = torch.tensor(df["length of melt pool"].values.reshape(-1, 1), dtype=dtype, device=DEVICE)
Yd_all = torch.tensor(df["depth of meltpool"].values.reshape(-1, 1), dtype=dtype, device=DEVICE)

bounds = torch.tensor([[X_all[:, 0].min(), X_all[:, 0].max()],
                       [X_all[:, 1].min(), X_all[:, 1].max()]], dtype=dtype, device=DEVICE)

# ---------------- GP Utilities ---------------- #
def TrainGP(X, Y, nu=maternmu):
    kernel = ScaleKernel(MaternKernel(nu=nu, ard_num_dims=X.shape[-1]))
    model = SingleTaskGP(
        train_X=X, train_Y=Y,
        covar_module=kernel,
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=Y.shape[-1])
    )
    return model

def fitmodel(model, restarts=restarts):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, num_restarts=restarts)
    return model

def diagnostics(gp, label):
    ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    outscale = float(gp.covar_module.outputscale.detach().cpu())
    noise = float(gp.likelihood.noise.detach().cpu()) if hasattr(gp.likelihood, "noise") else float("nan")
    print(f"[{label}] Lengthscales: {ls}, Outputscale: {outscale:.4g}, Noise: {noise:.4g}")

# ---------------- Metrics Helper ---------------- #
def evaluate_gp_quality(gp, X_test, Y_true, label=""):
    with torch.no_grad():
        posterior = gp.posterior(X_test)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
    
    y_true = Y_true.squeeze(-1).cpu().numpy()

    mae = np.mean(np.abs(mean - y_true))
    rmse = np.sqrt(np.mean((mean - y_true) ** 2))
    mape = np.mean(np.abs((mean - y_true) / (y_true + 1e-12))) * 100
    lower, upper = mean - 1.96 * std, mean + 1.96 * std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    nll = -np.mean(norm.logpdf(y_true, loc=mean, scale=std))

    print(f"[{label}] MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%, Coverage={coverage:.2f}, NLL={nll:.3f}")
    return dict(mae=mae, rmse=rmse, mape=mape, coverage=coverage, nll=nll)

# ---------------- Train ground-truth GPs on the CSV (full data) ---------------- #
gpw_gt = TrainGP(X_all, Yw_all); gpw_gt = fitmodel(gpw_gt, restarts=restarts); diagnostics(gpw_gt, "GT Width GP")
gpl_gt = TrainGP(X_all, Yl_all); gpl_gt = fitmodel(gpl_gt, restarts=restarts); diagnostics(gpl_gt, "GT Length GP")
gpd_gt = TrainGP(X_all, Yd_all); gpd_gt = fitmodel(gpd_gt, restarts=restarts); diagnostics(gpd_gt, "GT Depth GP")

# Evaluate baseline metrics for ground-truth models
evaluate_gp_quality(gpw_gt, X_all, Yw_all, label="GT Width")
evaluate_gp_quality(gpl_gt, X_all, Yl_all, label="GT Length")
evaluate_gp_quality(gpd_gt, X_all, Yd_all, label="GT Depth")

# ---------------- Acquisition helpers (top-level worker) ---------------- #
def _evaluate_point_worker(i, meanw_np, stdw_np, meanl_np, stdl_np, meand_np, stdd_np,
                           samplew_np, samplel_np, sampled_np, c1, c2, c3, thickness,
                           mode):
    if mode == "MC":
        sw = samplew_np[:, i]; sl = samplel_np[:, i]; sd = sampled_np[:, i]
        keyholing = (sw / sd) < c1
        lof = (sd / thickness) < c2
        balling = (sl / sw) > c3
        p1, p2, p3 = float(np.mean(keyholing)), float(np.mean(lof)), float(np.mean(balling))
        p4 = max(1.0 - (p1 + p2 + p3), 1e-12)
    elif mode == "Blind":
        p1, p2, p3, p4 = 0.0, 0.0, 0.0, 1.0
    else:
        raise ValueError("Unknown mode")

    probs = np.array([p1, p2, p3, p4], dtype=float)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    H = -(probs * np.log(probs)).sum()
    joint_std = float(stdw_np[i] * stdl_np[i] * stdd_np[i])
    return float(H * joint_std)

def entropy_sigma(X, gps, constraints, thickness, mode="MC", nmc=8, n_cores=1):
    gpw, gpl, gpd = gps
    c1, c2, c3 = constraints
    N = X.shape[0]

    with torch.no_grad():
        postw = gpw.posterior(X); meanw = postw.mean.squeeze(-1); stdw = postw.variance.sqrt().squeeze(-1).clamp(min=1e-6)
        postl = gpl.posterior(X); meanl = postl.mean.squeeze(-1); stdl = postl.variance.sqrt().squeeze(-1).clamp(min=1e-6)
        postd = gpd.posterior(X); meand = postd.mean.squeeze(-1); stdd = postd.variance.sqrt().squeeze(-1).clamp(min=1e-6)

    meanw_np, stdw_np = meanw.cpu().numpy(), stdw.cpu().numpy()
    meanl_np, stdl_np = meanl.cpu().numpy(), stdl.cpu().numpy()
    meand_np, stdd_np = meand.cpu().numpy(), stdd.cpu().numpy()

    with torch.no_grad():
        samplew = gpw.posterior(X).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
        samplel = gpl.posterior(X).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
        sampled = gpd.posterior(X).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()

    results = Parallel(n_jobs=n_cores)(
        delayed(_evaluate_point_worker)(i,
            meanw_np, stdw_np, meanl_np, stdl_np, meand_np, stdd_np,
            samplew, samplel, sampled,
            c1, c2, c3, thickness, mode
        ) for i in range(N)
    )
    return np.array(results)

# ---------------- Plotting ---------------- #
def plot_history(Xgrid, J_history, gps_history, Xtrain_history, constraints, ngrid, iterations_to_plot):
    cmap_defects = colors.ListedColormap(['green', 'red', 'blue', 'orange'])
    labels_defects = ['Printable', 'Keyholing', 'LOF', 'Balling']
    c1, c2, c3 = constraints

    fig, axes = plt.subplots(len(iterations_to_plot), 2, figsize=(14, 5 * len(iterations_to_plot)))
    if len(iterations_to_plot) == 1:
        axes = axes[np.newaxis, :]

    for idx, it in enumerate(iterations_to_plot):
        gps = gps_history[it - 1]
        Xtrain = Xtrain_history[it - 1]
        gpw, gpl, gpd = gps

        with torch.no_grad():
            meanw = gpw.posterior(Xgrid).mean.squeeze(-1)
            meanl = gpl.posterior(Xgrid).mean.squeeze(-1)
            meand = gpd.posterior(Xgrid).mean.squeeze(-1)

        p1 = ((meanw / meand) < c1).float().cpu().numpy()
        p2 = ((meand / thickness) < c2).float().cpu().numpy()
        p3 = ((meanl / meanw) > c3).float().cpu().numpy()
        p4 = 1 - (p1 + p2 + p3)
        probs = np.stack([p4, p1, p2, p3], axis=-1)
        category = np.argmax(probs, axis=-1).reshape(ngrid, ngrid)
        grid_x = Xgrid[:, 1].reshape(ngrid, ngrid).cpu().numpy()
        grid_y = Xgrid[:, 0].reshape(ngrid, ngrid).cpu().numpy()
        grid_J = J_history[it - 1].reshape(ngrid, ngrid)

        ax = axes[idx, 0]
        ax.imshow(category, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                  cmap=cmap_defects, alpha=0.6, aspect='auto')
        ax.scatter(Xtrain[:, 1].cpu(), Xtrain[:, 0].cpu(), color='k', s=25)
        ax.set_title(f"Iteration {it}: Defect Boundaries")
        ax.set_xlabel("Velocity"); ax.set_ylabel("Power")

        ax = axes[idx, 1]
        im = ax.imshow(grid_J, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                       cmap='viridis', aspect='auto')
        ax.scatter(Xtrain[:, 1].cpu(), Xtrain[:, 0].cpu(), color='k', s=25)
        ax.set_title(f"Iteration {it}: Acquisition Function")
        ax.set_xlabel("Velocity"); ax.set_ylabel("Power")
        fig.colorbar(im, ax=ax, label='Acquisition Value')

    plt.tight_layout()
    plt.show()

# ---------------- Grid for ACS ---------------- #
powergrid = torch.linspace(bounds[0, 0], bounds[0, 1], ngrid, device=DEVICE, dtype=dtype)
velogrid = torch.linspace(bounds[1, 0], bounds[1, 1], ngrid, device=DEVICE, dtype=dtype)
PP, VV = torch.meshgrid(powergrid, velogrid, indexing="ij")
Xgrid = torch.stack([PP.flatten(), VV.flatten()], dim=-1)

# ---------------- Initial Training Points ---------------- #
Xtrain_raw = draw_sobol_samples(bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=dtype, device=DEVICE),
                                n=1, q=Ntrain, seed=SEED).squeeze(0)
Xtrain = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * Xtrain_raw

with torch.no_grad():
    Ytrainw = gpw_gt.posterior(Xtrain).mean.detach()
    Ytrainl = gpl_gt.posterior(Xtrain).mean.detach()
    Ytraind = gpd_gt.posterior(Xtrain).mean.detach()

Xinit, Yinitw, Yinitl, Yinitd = Xtrain.clone(), Ytrainw.clone(), Ytrainl.clone(), Ytraind.clone()

gps_history, Xtrain_history, J_history, metrics_history = [], [], [], []

import os

output_csv = "ACS_quality_metrics.csv"
# Remove previous CSV if you want to start fresh
if os.path.isfile(output_csv):
    os.remove(output_csv)

for it in tqdm(range(1, Niter + 1)):
    # --- Fit surrogate GPs on current training data ---
    gpw = TrainGP(Xtrain, Ytrainw); gpw = fitmodel(gpw, restarts=restarts)
    gpl = TrainGP(Xtrain, Ytrainl); gpl = fitmodel(gpl, restarts=restarts)
    gpd = TrainGP(Xtrain, Ytraind); gpd = fitmodel(gpd, restarts=restarts)
    gps = [gpw, gpl, gpd]

    # --- Compute acquisition function (surrogates) ---
    J = entropy_sigma(Xgrid, gps, constraints, thickness, mode="MC", nmc=nmc, n_cores=n_cores)

    # --- Select next candidate ---
    ind = int(np.argmax(J))
    x_next = Xgrid[ind:ind + 1]

    # --- Query ground-truth GPs for labels ---
    with torch.no_grad():
        w_next = gpw_gt.posterior(x_next).mean.detach()
        l_next = gpl_gt.posterior(x_next).mean.detach()
        d_next = gpd_gt.posterior(x_next).mean.detach()

    # --- Update training data ---
    Xtrain = torch.cat([Xtrain, x_next], dim=0)
    Ytrainw = torch.cat([Ytrainw, w_next], dim=0)
    Ytrainl = torch.cat([Ytrainl, l_next], dim=0)
    Ytraind = torch.cat([Ytraind, d_next], dim=0)

    # --- Metrics computation ---
    metrics = {}
    for name, gp, Ytrue in zip(["Width", "Length", "Depth"],
                               [gpw, gpl, gpd],
                               [Yw_all, Yl_all, Yd_all]):
        print(f"\n--- Iteration {it} Metrics for {name} ---")
        metrics[name.lower()] = evaluate_gp_quality(gp, X_all, Ytrue, label=f"Iter {it} - {name}")

    metrics_history.append(metrics)

    # --- Export metrics immediately to CSV ---
    metrics_list = []
    for output_name, vals in metrics.items():
        row = {"Iteration": it, "Output": output_name}
        row.update(vals)
        metrics_list.append(row)

    metrics_df = pd.DataFrame(metrics_list)
    if not os.path.isfile(output_csv):
        metrics_df.to_csv(output_csv, index=False)
    else:
        metrics_df.to_csv(output_csv, index=False, mode='a', header=False)

    # --- Store history for plotting ---
    gps_history.append(gps)
    Xtrain_history.append(Xtrain.clone())
    J_history.append(J.copy())

    # --- PLOT every 5 iterations ---
    if it % 5 == 0 or it == Niter:
        plot_history(Xgrid, J_history, gps_history, Xtrain_history, constraints, ngrid, [it])