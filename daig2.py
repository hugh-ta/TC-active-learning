# ============================================================
# Independent SingleTaskGPs (improved but same outputs)
# ============================================================

import numpy as np
import pandas as pd
import torch

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll

# -------------------------
# SETTINGS
# -------------------------
dtype = torch.double
device = torch.device("cpu")
thickness = 10
ntrain = 10
file_name = "results_progress.csv"

# -------------------------
# Load + preprocess data
# -------------------------
data = pd.read_csv(file_name)
data["Depth"] = data["Depth"].replace(0, 1e-6)  # avoid divide by zero
data["hmax"] = data["Width"] * np.sqrt(1 - thickness / (thickness + data["Depth"]))

for col in ["Depth", "Width", "Power", "Speed", "Length", "hmax"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

depth  = data["Depth"].values.reshape(-1, 1)
width  = data["Width"].values.reshape(-1, 1)
length = data["Length"].values.reshape(-1, 1)
power  = data["Power"].values.reshape(-1, 1)
speed  = data["Speed"].values.reshape(-1, 1)

Xin = np.hstack([power, speed])
Xin = (Xin - Xin.mean(axis=0)) / Xin.std(axis=0)

n_total = len(depth)
idx = torch.randperm(n_total)[:ntrain]

X = torch.tensor(Xin, dtype=dtype, device=device)[idx]
Y_targets = {
    "Depth":  torch.tensor(depth,  dtype=dtype, device=device)[idx],
    "Width":  torch.tensor(width,  dtype=dtype, device=device)[idx],
    "Length": torch.tensor(length, dtype=dtype, device=device)[idx],
}

# -------------------------
# Train one GP per task
# -------------------------
def train_single_task_gp(X, y):
        # -------------------------------
    # Why this version improves RMSE:
    #
    # 1. Input normalization (Normalize):
    #    - Rescales each input dim (Power, Speed) into [0,1].
    #    - Without this, ARD lengthscales can become poorly conditioned
    #      if inputs are on very different numeric scales.
    #
    # 2. Output standardization (Standardize):
    #    - Standardizes y to mean=0, variance=1 before training.
    #    - Prevents large-magnitude targets (e.g. Length ~500)
    #      from dominating the likelihood optimization.
    #    - Ensures Depth, Width, Length are all trained on
    #      equal footing, giving smaller residuals after rescaling back.
    #
    # 3. Consistency across tasks:
    #    - Every GP gets the exact same kernel, transforms,
    #      and optimizer routine.
    #    - Removes subtle inconsistencies from hand-coded
    #      per-task training in the old version.
    #
    # Both old and new versions maximize the marginal log likelihood,
    # but this version conditions the data better → optimizer finds
    # better hyperparameters → lower RMSE.
    # -------------------------------
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1]))
    model = SingleTaskGP(
        train_X=X,
        train_Y=y,
        covar_module=kernel,
        input_transform=Normalize(X.shape[1]),
        outcome_transform=Standardize(1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

gp_models = {}
for task, y in Y_targets.items():
    print(f"\n--- Training GP for {task} ---")
    gp_models[task] = train_single_task_gp(X, y)

# -------------------------
# Evaluate
# -------------------------
preds, rmses = {}, {}
for task, gp in gp_models.items():
    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(X)
        y_pred = posterior.mean.cpu().numpy().ravel()
        y_true = Y_targets[task].cpu().numpy().ravel()
        preds[task] = y_pred
        rmses[task] = np.sqrt(np.mean((y_pred - y_true) ** 2))

print("\nTraining RMSE per task:")
for t, r in rmses.items():
    print(f"  {t}: {r:.3f}")

print("\nSample predictions:")
for i in range(min(10, ntrain)):
    true_vals = [f"{Y_targets[t][i].item():.2f}" for t in ["Depth", "Width", "Length"]]
    pred_vals = [f"{preds[t][i]:.2f}" for t in ["Depth", "Width", "Length"]]
    print(f" True: {true_vals} | Pred: {pred_vals}")

    # Additional Tips to Improve GP Predictions and Stability

# 1. Use sensible initial noise settings:
#    - Set model.likelihood.noise to a small fraction of the target std (e.g., 1-2% of std).
#    - Helps optimizer avoid zero-noise solutions and stabilizes Cholesky decomposition.
#
# 2. Consider multiple kernel candidates per task:
#    - Try Matern 1.5, Matern 2.5, and RBF kernels.
#    - Pick the one with lowest validation RMSE.
#    - Allows each task to get the kernel that best fits its behavior.

# 3. Train/test split for hyperparameter selection:
#    - Reserve 10-20% of points as test.
#    - Use test RMSE (or log-likelihood) to pick kernel/lengthscale priors.
#    - Prevents overfitting to just the training data.

# 4. Lengthscale priors (GammaPrior):
#    - Helps optimizer find reasonable ARD lengthscales.
#    - Prevents overly large or tiny lengthscales that hurt generalization.

# 5. Outputscale priors (SmoothedBoxPrior):
#    - Avoids runaway variance estimates in ScaleKernel.
#    - Keeps GP predictive variance in a reasonable range.

# 6. Optional fixed-noise mode for noisy observations:
#    - If your experimental measurements have known variance,
#      you can provide it via FixedNoiseGaussianLikelihood.
#    - Ensures GP doesn’t overfit noise fluctuations.

# 7. Gradient clipping during MLL optimization:
#    - Prevents exploding gradients if optimizer tries wild hyperparameters.
#    - Use torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

# 8. Check for NaNs or extreme values:
#    - Clip outliers in targets to a reasonable quantile (0.5%–99.5%).
#    - Keeps optimizer from chasing extreme points and stabilizes RMSE.

# 9. Multiple random restarts:
#    - Run fit_gpytorch_mll() with several random initializations.
#    - Pick the one with best training/test RMSE.
#    - Reduces risk of local minima in hyperparameter optimization.

# 10. Keep input dimensions small and informative:
#    - Avoid unnecessary derived features unless they improve predictive power.
#    - Too many inputs can make ARD lengthscale optimization unstable.

# Following these practices together with the normalization and standardization
# steps you already have will maximize prediction reliability and minimize RMSE.