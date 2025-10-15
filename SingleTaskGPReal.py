#ok ok single task gp 
# Ok the goal here is to simulate what we'd actually do w/ Flow3d or an experiment,
# train it on some data, and then tell it to sample from TC python or something idk! but yeah!
# hopefully it works hehe fr fr this time not like last time ahaha

## import libs
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

# BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll

#scipy
from scipy.interpolate import griddata


# user settings and toggles

USE_BALLING = True
USE_EDENSITY   = False  
dtype = torch.double
device = torch.device("cpu")
thickness = 10
ntrain = 10
restarts = 5
ngrid= 50
nmc = 64
edensity_low = 0
edensity_high =10000000
file_name = "results_progress.csv"\


#import the data 
data = pd.read_csv(file_name)

depth  = data["Depth"].values.reshape(-1, 1)
width  = data["Width"].values.reshape(-1, 1)
length = data["Length"].values.reshape(-1, 1)
power  = data["Power"].values.reshape(-1, 1)
speed  = data["Speed"].values.reshape(-1, 1)

# pick number of training samples
n_total = len(width)
idx = torch.randperm(n_total)[:ntrain]

#set values into tensors
Xin = np.hstack([power,speed])
X = torch.tensor(Xin[idx], dtype=dtype, device=device)
d = X.shape[1]
#think about including zscore normalization

Yd = torch.tensor(depth[idx], dtype=dtype, device=device)
Yw = torch.tensor(width[idx], dtype=dtype, device=device)
Yl = torch.tensor(length[idx], dtype=dtype, device=device)
m = Yd.shape[1]

Y_targets = {
    "Depth":  Yd,
    "Width":  Yw,
    "Length": Yl,
}

#def funcs
def FitGP(X,Y):
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims= d))
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=kernel,
        input_transform=Normalize(X.shape[1]),
        outcome_transform=Standardize(1),
    )
    return model
def fitGP(model, restarts=restarts):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_cls=None, options={"maxiter": 200}, num_restarts=restarts)
    return model
def evaluate_gp_models(gp_models, X, Y_targets, n_samples=10, label=""):
    preds, rmses = {}, {}

    for task, gp in gp_models.items():
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(X)
            y_pred = posterior.mean.cpu().numpy().ravel()
            y_true = Y_targets[task].cpu().numpy().ravel()
            preds[task] = y_pred
            rmses[task] = np.sqrt(np.mean((y_pred - y_true) ** 2))

    print(f"\n=== Evaluation {label} ===")
    print("RMSE per task:")
    for t, r in rmses.items():
        print(f"  {t}: {r:.3f}")

    print("\nSample predictions:")
    for i in range(min(n_samples, len(X))):
        true_vals = [f"{Y_targets[t][i].item():.2f}" for t in ["Depth", "Width", "Length"]]
        pred_vals = [f"{preds[t][i]:.2f}" for t in ["Depth", "Width", "Length"]]
        print(f"  True: {true_vals} | Pred: {pred_vals}")

    return preds, rmses
def classify_defect(width, depth, e, length=None, thickness=10, ed_low=edensity_low, ed_high=edensity_high):
    if e is not None:
        if e < ed_low:
            return "Lack of Fusion"
        elif e > ed_high:
            return "Keyhole"
    if width/depth < 1.5:
        return "Keyhole"
    elif depth/thickness < 1.9:
        return "Lack of Fusion"
    if USE_BALLING and (length is not None):
        if width/length < 0.23:
            return "Balling"
    return "Good"

def classify_grid(width_grid, depth_grid, thickness, ed_grid, length_grid=None):
    result = np.empty_like(width_grid, dtype=object)
    for i in range(width_grid.shape[0]):
        for j in range(width_grid.shape[1]):
            w = width_grid[i,j]
            d = depth_grid[i,j]
            e = ed_grid[i,j] if ed_grid is not None else None
            l = length_grid[i,j] if length_grid is not None else None
            if np.isnan(w) or np.isnan(d) or (USE_EDENSITY and (e is None or np.isnan(e))) or (USE_BALLING and length_grid is not None and np.isnan(l)):
                result[i,j] = "Good"
            else:
                result[i,j] = classify_defect(w, d, e, l, thickness)
    return result

def classify_defect_mc(width_samples, depth_samples, e_samples=None, length_samples=None, thickness=10):

    n_mc, n_points = width_samples.shape
    labels = np.empty(n_points, dtype=object)

    for i in range(n_points):
        mc_labels = []
        w_s = width_samples[:, i]
        d_s = depth_samples[:, i]
        e_s = e_samples[:, i] if e_samples is not None else None
        l_s = length_samples[:, i] if length_samples is not None else None

        for j in range(n_mc):
            l = l_s[j] if l_s is not None else None
            e = e_s[j] if e_s is not None else None
            # classify defect
            if USE_EDENSITY:
                mc_labels.append(classify_defect(w_s[j], d_s[j], e, l, thickness))
            else:
                mc_labels.append(classify_defect(w_s[j], d_s[j], None, l, thickness))
        # most frequent label
        labels[i] = max(set(mc_labels), key=mc_labels.count)

    return labels

#train train train!
gp_models = {}
for task, Y in Y_targets.items():
    gp = FitGP(X, Y)
    gp = fitGP(gp, restarts=5)
    gp_models[task] = gp

evaluate_gp_models(gp_models, X, Y_targets, n_samples=10, label="After training")


## ok graph graph graph to see the before!
#make grid
#grid
powermin, powermax = X[:,0].min().item(), X[:,0].max().item()
velomin, velomax = X[:,1].min().item(), X[:,1].max().item()
bounds = torch.tensor([[powermin, powermax],   
                       [velomin, velomax]], dtype=dtype, device=device)
powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP,VV = np.meshgrid(powergrid, velogrid, indexing="ij")

gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
Xgrid = torch.tensor(gridpoints, dtype=dtype, device=device)
# gp posteriors
depth_posterior  = gp_models["Depth"].posterior(Xgrid)
width_posterior  = gp_models["Width"].posterior(Xgrid)
length_posterior = gp_models["Length"].posterior(Xgrid)

# sampleing for montecarlo
width_samples  = gp_models["Width"].posterior(Xgrid).rsample(torch.Size([nmc])).detach().cpu().numpy().reshape(nmc,-1)
depth_samples  = gp_models["Depth"].posterior(Xgrid).rsample(torch.Size([nmc])).detach().cpu().numpy().reshape(nmc,-1)
length_samples = gp_models["Length"].posterior(Xgrid).rsample(torch.Size([nmc])).detach().cpu().numpy().reshape(nmc,-1)

# classify defect
defect_labels = classify_defect_mc(
    width_samples, 
    depth_samples, 
    e_samples=None, 
    length_samples=length_samples, 
    thickness=thickness
)
defect_grid_mc = defect_labels.reshape(ngrid, ngrid)

# map defects to numeric values
mapping = {"Keyhole": 0, "Lack of Fusion": 1, "Good": 2, "Balling":3}
defect_numeric_grid = np.vectorize(mapping.get)(defect_grid_mc)

# create RGB grid
rgb_grid = np.ones(defect_numeric_grid.shape + (3,), dtype=float)  # shape: (rows, cols, 3)
red   = np.array(mcolors.to_rgb("#E07B7B"))
blue  = np.array(mcolors.to_rgb("#7bbfc8"))
green = np.array(mcolors.to_rgb("#289C8E"))
alpha_defects = 1.0

# fill RGB channels
for i in range(3):
    rgb_grid[defect_numeric_grid == 0, i] = alpha_defects*red[i] + (1-alpha_defects)*1
    rgb_grid[defect_numeric_grid == 1, i] = alpha_defects*blue[i] + (1-alpha_defects)*1
    rgb_grid[defect_numeric_grid == 3, i] = alpha_defects*green[i] + (1-alpha_defects)*1

if rgb_grid.shape[0] == 3:  # channel-first
    rgb_grid = np.transpose(rgb_grid, (1, 2, 0))  # -> (rows, cols, 3)

# plot MC defect map
plt.figure(figsize=(10,7))
plt.imshow(rgb_grid, extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()],
           origin="lower", aspect="auto", zorder=1)

plt.xlabel("Scan Velocity (mm/s)")
plt.ylabel("Laser Power (W)")
plt.title("316L Printability Map (Monte Carlo)")

# legend setup
legend_elements = [
    Patch(facecolor="#E07B7B", label="Keyhole W/D < 1.5"),
    Patch(facecolor="#7bbfc8", label="Lack of Fusion D/t <1.9"),
]
if USE_BALLING:
    legend_elements.append(Patch(facecolor="#289C8E", label="Balling W/L < 0.23"))
legend_elements.append(Patch(facecolor="#FFFFFF", edgecolor="black", label="Stable/Printable"))

plt.legend(handles=legend_elements, loc="best")
plt.show()