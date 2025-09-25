import numpy as np
import pandas as pd
import torch

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from botorch.models import KroneckerMultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll

# -------------------------
# SETTINGS
# -------------------------
dtype = torch.double
device = torch.device("cpu")
ntrain = 99
file_name = "results_progress.csv"

# -------------------------
# LOAD AND PREPROCESS DATA
# -------------------------
data = pd.read_csv(file_name)
data["Depth"] = data["Depth"].replace(0, 1e-6)  # avoid divide-by-zero
thickness = 10
data["hmax"] = data["Width"] * np.sqrt(1 - thickness/(thickness + data["Depth"]))

# Keep only rows where all outputs exist
data = data.dropna(subset=["Depth", "Width", "Length", "Power", "Speed"])

# extract inputs
X_in = np.column_stack([
    data["Power"].values,
    data["Speed"].values
])
# normalize
X_in = (X_in - X_in.mean(axis=0)) / X_in.std(axis=0)

# extract outputs
Y_out = np.column_stack([
    data["Depth"].values,
    data["Width"].values,
    data["Length"].values
])

# training subset
n_total = X_in.shape[0]
idx = torch.randperm(n_total)[:ntrain]

X = torch.tensor(X_in[idx], dtype=dtype, device=device)
Y = torch.tensor(Y_out[idx], dtype=dtype, device=device)

n, m = Y.shape
d = X.shape[1]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# -------------------------
# TRAINING FUNCTION
# -------------------------
def TrainKroneckerMultiTaskGP(X, Y):
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    likelihood = MultitaskGaussianLikelihood(num_tasks=Y.shape[1], rank=1)
    model = KroneckerMultiTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=kernel,
        input_transform=Normalize(d),
        outcome_transform=Standardize(Y.shape[1]),
        likelihood=likelihood
    )
    return model, likelihood

def fitmodel(model, likelihood):
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    print("Model fitting complete.")

# -------------------------
# TRAIN MODEL
# -------------------------
model, likelihood = TrainKroneckerMultiTaskGP(X, Y)
fitmodel(model, likelihood)

# -------------------------
# PREDICTIONS AND RMSE
# -------------------------
model.eval()
with torch.no_grad():
    posterior = model(X)[0]
    Y_pred_standardized = posterior.mean.cpu()
    Y_pred, _ = model.outcome_transform.untransform(Y_pred_standardized)
    Y_pred = Y_pred.numpy()

# RMSE per output
rmse = np.sqrt(np.mean((Y_pred - Y.cpu().numpy())**2, axis=0))
print("\nTraining RMSE (Depth, Width, Length):", rmse)

# sample predictions safely
n_sample = min(10, Y.shape[0], Y_pred.shape[0])
print("\nSample predictions:")
for i in range(n_sample):
    print(
        f"True: {Y[i,0]:.3f}, {Y[i,1]:.3f}, {Y[i,2]:.3f} | "
        f"Pred: {Y_pred[i,0]:.3f}, {Y_pred[i,1]:.3f}, {Y_pred[i,2]:.3f}"
    )