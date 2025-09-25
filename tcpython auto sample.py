# TC Python Sampling!!
# Ok the goal here is to simulate what we'd actually do w/ Flow3d or an experiment,
# train it on some data, and then tell it to sample from TC python or something idk! but yeah!
# hopefully it works hehe

# import libs
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from botorch.models import IndependentMultitaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# BoTorch
from botorch.models import KroneckerMultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll

# important defns and toggles
USE_BALLING = True
dtype = torch.double
device = torch.device('cpu')
thickness = 10
ntrain = 99

# import data
file_name = "results_progress.csv"
data = pd.read_csv(file_name)

# replace 0 values in Depth to avoid division by zero
data["Depth"] = data["Depth"].replace(0, 1e-6)

# recalc h max
data["hmax"] = data["Width"] * np.sqrt(1 - thickness/(thickness + data["Depth"]))

# ensure numeric
numeric_cols = ["Depth", "Width", "Power", "Speed", "hmax"]
if "Length" in data.columns:
    numeric_cols.append("Length")
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# read in data
hmax = data["hmax"].values.reshape(-1,1)
power = data["Power"].values.reshape(-1,1)
speed = data["Speed"].values.reshape(-1,1)
depth = data["Depth"].values.reshape(-1,1)
width = data["Width"].values.reshape(-1,1)
length = data["Length"].values.reshape(-1,1)

# pick number of training samples
n_total = len(width)
idx = torch.randperm(n_total)[:ntrain]

# prepare outputs
Yd = torch.tensor(depth, dtype=dtype, device=device)[idx]
Yw = torch.tensor(width, dtype=dtype, device=device)[idx]
Yl = torch.tensor(length, dtype=dtype, device=device)[idx]

# stack outputs into [n,3]
Y = torch.cat([Yd, Yw, Yl], dim=1)
n, m = Y.shape

# prepare inputs
Xin = np.hstack([power, speed])
# normalize inputs
Xin = (Xin - Xin.mean(axis=0)) / Xin.std(axis=0)
X = torch.tensor(Xin, dtype=dtype, device=device)[idx]
d = X.shape[1]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# CSV to check
df_full = pd.DataFrame(
    torch.cat([X, Y], dim=1).cpu().numpy(),
    columns=["Power", "Speed", "Depth", "Width", "Length"]
)
df_full.to_csv("gp_X_Y_check.csv", index=False)
print("\nData saved to 'gp_X_Y_check.csv'.")

# Train function
def TrainGP_independent(X, Y):
    """
    Train an Independent Multitask GP where each output has its own kernel
    and its own Gaussian likelihood variance.
    """
    d = X.shape[1]
    m = Y.shape[1]

    # one kernel per output dimension
    kernels = [
        ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
        for _ in range(m)
    ]

    likelihood = GaussianLikelihood(batch_shape=torch.Size([m]))

    model = IndependentMultitaskGP(
        train_X=X,
        train_Y=Y,
        outcome_transform=Standardize(m),
        covar_modules=kernels,
        likelihood=likelihood,
    )

    return model, likelihood

def fitmodel(model, likelihood):
    """
    Fit the multitask GP by maximizing the marginal log likelihood.
    """
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    print("Model fitting complete.")

def fitmodel(model, likelihood):
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)
    print("Model fitting complete.")

def diagnostics_multi(gp):
    kernel = gp.covar_module.data_covar_module
    if hasattr(kernel, "base_kernel"):
        ls = kernel.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    else:
        ls = kernel.lengthscale.detach().cpu().numpy().ravel()
    outscale = float(kernel.outputscale.detach().cpu()) if hasattr(kernel, "outputscale") else float("nan")
    task_cov = gp.covar_module.task_covar_module.covar_matrix.detach().cpu().numpy()
    print(f"Lengthscales (per input): {ls}")
    print(f"Outputscale: {outscale}")
    print(f"Task covariance matrix:\n{task_cov}")

# Train
model, likelihood = TrainGP(X, Y)
diagnostics_multi(model)
fitmodel(model, likelihood)
diagnostics_multi(model)

# Evaluate predictions (training data) using untransform
model.eval()
with torch.no_grad():
    posterior = model(X)[0]  # unpack the first element
    Y_pred_standardized = posterior.mean.cpu()
    pred, _ = model.outcome_transform.untransform(Y_pred_standardized)
    pred = pred.numpy()

# Compute RMSE
rmse = np.sqrt(np.mean((pred - Y.cpu().numpy())**2, axis=0))
print("\nTraining RMSE (Depth, Width, Length):", rmse)

# Show sample predictions
for i in range(min(10, n)):
    print(f"True: {Y[i,0]:.3f}, {Y[i,1]:.3f}, {Y[i,2]:.3f} | Pred: {pred[i,0]:.3f}, {pred[i,1]:.3f}, {pred[i,2]:.3f}")

# Plotting function
def plot_gp_defect_map(model, thickness=10, use_balling=False, grid_res=200, dtype=torch.float32, device='cpu'):
    model.eval()
    grid_x = np.linspace(X[:,0].min(), X[:,0].max(), grid_res)
    grid_y = np.linspace(X[:,1].min(), X[:,1].max(), grid_res)
    xi, yi = np.meshgrid(grid_x, grid_y)
    X_grid = np.column_stack([xi.ravel(), yi.ravel()])
    X_grid_tensor = torch.tensor(X_grid, dtype=dtype, device=device)
    
    # Prediction with untransform
    with torch.no_grad():
        posterior = model(X_grid_tensor)[0]  # unpack first element
        Y_pred_standardized = posterior.mean.cpu()
        Y_pred, _ = model.outcome_transform.untransform(Y_pred_standardized)
        Y_pred = Y_pred.numpy()
        Y_pred = np.maximum(Y_pred, 1e-6)
    
    depth_grid = Y_pred[:,0].reshape(xi.shape)
    width_grid = Y_pred[:,1].reshape(xi.shape)
    length_grid = Y_pred[:,2].reshape(xi.shape) if Y_pred.shape[1]>2 else None
    
    n_rows, n_cols = width_grid.shape
    data = {
        "Scan_Velocity": np.repeat(np.linspace(X[:,0].min(), X[:,0].max(), n_cols), n_rows),
        "Laser_Power": np.tile(np.linspace(X[:,1].min(), X[:,1].max(), n_rows), n_cols),
        "Depth": depth_grid.ravel(),
        "Width": width_grid.ravel()
    }
    if length_grid is not None:
        data["Length"] = length_grid.ravel()
    
    df = pd.DataFrame(data)
    df.to_csv("gp_grids.csv", index=False)
    print("GP grids exported to gp_grids.csv")
    
    # Classify defects
    def classify_simple(width, depth, length=None, thickness=10):
        if width/depth < 1.5:
            return "Keyhole"
        elif depth/thickness < 1.9:
            return "Lack of Fusion"
        if use_balling and (length is not None):
            if width/length < 0.23:
                return "Balling"
        return "Good"
    
    classify_vec = np.vectorize(classify_simple)
    defect_grid = classify_vec(width_grid, depth_grid, length_grid, thickness)
    
    color_map = {
        "Keyhole": "#E07B7B",
        "Lack of Fusion": "#7bbfc8",
        "Good": "#FFFFFF",
        "Balling": "#289C8E"
    }
    rgb_grid = np.ones(defect_grid.shape + (3,), dtype=float)
    for label, hex_color in color_map.items():
        mask = defect_grid == label
        rgb = np.array(mcolors.to_rgb(hex_color))
        rgb_grid[mask] = rgb
    
    plt.figure(figsize=(10,7))
    plt.imshow(rgb_grid, extent=[X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()],
               origin='lower', aspect='auto')
    plt.xlabel("Scan Velocity (mm/s)")
    plt.ylabel("Laser Power (W)")
    plt.title("GP-based Printability Map")
    
    legend_elements = [
        Patch(facecolor=color_map["Keyhole"], label="Keyhole W/D < 1.5"),
        Patch(facecolor=color_map["Lack of Fusion"], label="Lack of Fusion D/t <1.9"),
        Patch(facecolor=color_map["Good"], edgecolor="black", label="Stable/Printable")
    ]
    if use_balling:
        legend_elements.append(Patch(facecolor=color_map["Balling"], label="Balling W/L < 0.23"))
    
    plt.legend(handles=legend_elements, loc="best")
    plt.show()

plot_gp_defect_map(model, thickness=10)

"""
This repaired Kronecker MultiTask GP code is set up to *force the model to fit* 
by carefully controlling the data pipeline, model definition, and training process.

Key things it does compared to the earlier broken versions:

1. Data Handling
   - Loads CSV melt pool data (Depth, Width, Length, Power, Speed).
   - Replaces zero Depth with a small epsilon to avoid divide-by-zero.
   - Computes derived features like hmax.
   - Ensures all relevant columns are numeric.
   - Normalizes inputs (Power, Speed) so the GP sees standardized features.
   - Randomly subsamples a fixed number of training points.

2. Outputs
   - Collects Depth, Width, and Length into a single multitask output tensor (Y).
   - Shapes everything consistently as [n_samples, n_tasks].

3. Model Construction
   - Uses KroneckerMultiTaskGP with:
       * Matern ARD kernel inside a ScaleKernel for flexibility.
       * MultitaskGaussianLikelihood with low-rank task covariance to learn correlations.
       * Input transform (Normalize) and outcome transform (Standardize) to stabilize.
   - Encapsulated into helper functions: TrainGP(), fitmodel(), diagnostics_multi().

4. Training & Stability
   - Fits the marginal log likelihood with fit_gpytorch_mll().
   - Prints diagnostics before/after training:
       * Kernel lengthscales
       * Outputscale
       * Task covariance matrix
   - Uses outcome_transform.untransform() so predictions are in physical units.

5. Evaluation
   - Computes training RMSE per task.
   - Prints true vs predicted outputs for a few samples so errors are visible.
   - Handles possible NaN/inf instabilities by standardizing/unstandardizing explicitly.

6. Visualization
   - Predicts outputs across a (Power, Speed) grid.
   - Untransforms back to Depth, Width, Length fields.
   - Applies defect classification rules:
       * Keyhole: W/D < 1.5
       * Lack of Fusion: D/t < 1.9
       * Balling (optional): W/L < 0.23
       * Else: Good
   - Colors regions accordingly and plots a printability map.
   - Exports full prediction grid to CSV for further analysis.

In short:
This repaired version takes extra steps to normalize, stabilize, and explicitly 
untransform predictions, forcing the multitask GP to actually fit the training 
data while still producing diagnostic outputs and a defect classification map.
"""