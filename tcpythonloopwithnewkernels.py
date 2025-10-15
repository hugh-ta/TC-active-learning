## import libs
# common libs
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import tqdm
from matplotlib import colors
import gc, time
import joblib # <-- Added to load the scaler objects

# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel, Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from torch import nn

# BoTorch
from botorch.fit import fit_gpytorch_mll

# scipy
from scipy.interpolate import griddata
from scipy.stats import norm

# tcpython
from tc_python import *


# user settings and toggles
HEAT_SOURCE_NAME = "Double ellipsoidal - 316L - beam d 15um"
MATERIAL_NAME = "SS316L"
POWDER_THICKNESS = 10.0  # in micro-meter
HATCH_DISTANCE = 10.0  # in micro-meter. Only single track experiment but let's use a hatch spacing for the printability map
AMBIENT_TEMPERATURE = 353  # in K, as given in the paper
USE_BALLING = True
USE_EDENSITY   = False  
USE_EDGE_PENALTY = True
dtype = torch.double
device = torch.device("cpu")
thickness = 10
ntrain = 10
niter = 20 #how many AL loops after  training are allowed
restarts = 10
ngrid= 100
nmc = 64
edensity_low = 0
edensity_high =10000000
file_name = "results_progress.csv"
SEED = 8
np.random.seed(SEED)
torch.manual_seed(SEED)
keyholing = 1.9
lof = 1.5
balling = 4.35

constraints = [keyholing, lof, balling]

#store the runs and stuff!
CSV_FILE = "active_learning_runs.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=["power", "speed", "Depth", "Width", "Length"])
    df.to_csv(CSV_FILE, index=False)

# import the data
data = pd.read_csv(file_name)

depth  = data["Depth"].values.reshape(-1, 1)
width  = data["Width"].values.reshape(-1, 1)
length = data["Length"].values.reshape(-1, 1)
power  = data["Power"].values.reshape(-1, 1)
speed  = data["Speed"].values.reshape(-1, 1)

# total number of data points
n_total = len(data)

# pick ntrain indices randomly from the actual data
initial_idx = np.random.choice(n_total, size=ntrain, replace=False)

# training inputs (kept in raw, unscaled format)
X = torch.tensor(np.column_stack([power[initial_idx], speed[initial_idx]]), dtype=dtype, device=device)

# training outputs
Yd = torch.tensor(depth[initial_idx], dtype=dtype, device=device)
Yw = torch.tensor(width[initial_idx], dtype=dtype, device=device)
Yl = torch.tensor(length[initial_idx], dtype=dtype, device=device)

# make dict for easier access
Y_targets = {
    "Depth": Yd,
    "Width": Yw,
    "Length": Yl,
}
d = X.shape[1]
m = Yd.shape[1]


# Define DKL Model Components
class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', nn.Linear(2, 100))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(100, 50))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(50, 2))


class DeepGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        # Squeeze train_y for the ExactGP super constructor
        super(DeepGPModel, self).__init__(train_x, train_y.squeeze(-1), likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

# def funcs
def evaluate_gp_models(gp_models, X_raw, Y_targets, scalers, n_samples=10, label=""):
    preds, rmses = {}, {}

    for task, gp in gp_models.items():
        gp.eval()
        with torch.no_grad():
            # Manually scale the input data before prediction
            scaler = scalers[task]
            X_scaled = torch.from_numpy(scaler.transform(X_raw.numpy()))
            
            posterior = gp.likelihood(gp(X_scaled))
            y_pred = posterior.mean.cpu().numpy().ravel()
            if Y_targets is not None:
                y_true = Y_targets[task].cpu().numpy().ravel()
                rmses[task] = np.sqrt(np.mean((y_pred - y_true) ** 2))
            preds[task] = y_pred

    print(f"\n=== Evaluation {label} ===")
    if rmses:
        print("RMSE per task:")
        for t, r in rmses.items():
            print(f"  {t}: {r:.3f}")

    print("\nSample predictions:")
    for i in range(min(n_samples, len(X_raw))):
        if Y_targets is not None:
            true_vals = [f"{Y_targets[t][i].item():.2f}" for t in ["Depth", "Width", "Length"]]
            pred_vals = [f"{preds[t][i]:.2f}" for t in ["Depth", "Width", "Length"]]
            print(f"  True: {true_vals} | Pred: {pred_vals}")
        else:
            pred_vals = [f"{preds[t][i]:.2f}" for t in ["Depth", "Width", "Length"]]
            print(f"  Probe Point {i + 1} Pred: {pred_vals}")
    return preds, rmses

def classify_defect(width, depth, e, length=None, thickness=10, ed_low=edensity_low, ed_high=edensity_high):
    jitter = 1e-9
    if e is not None:
        if e < ed_low:
            return "Lack of Fusion"
        elif e > ed_high:
            return "Keyhole"
    if (depth + jitter) == 0:
        return "Lack of Fusion"
    if width / (depth + jitter) < 1.5:
        return "Keyhole"
    elif (depth + jitter) / (thickness + jitter) < 1.9:
        return "Lack of Fusion"
    if USE_BALLING and (length is not None):
        if width / (length + jitter) < 0.23:
            return "Balling"
    return "Good"

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
            if USE_EDENSITY:
                mc_labels.append(classify_defect(w_s[j], d_s[j], e, l, thickness))
            else:
                mc_labels.append(classify_defect(w_s[j], d_s[j], None, l, thickness))
        labels[i] = max(set(mc_labels), key=mc_labels.count)
    return labels

# fit initial GP models
gp_models = {}
scalers = {} # <-- Store the loaded scalers
print("--- Loading pre-trained DKL kernels and their original scalers ---")
for task, Y_target in Y_targets.items():
    # Load the correct scaler for this task
    scaler_path = f"scaler_for_{task}.save"
    print(f"  Loading scaler for '{task}' from '{scaler_path}'...")
    scaler = joblib.load(scaler_path)
    scalers[task] = scaler
    
    # Scale the initial training data using the loaded scaler
    X_scaled = torch.from_numpy(scaler.transform(X.numpy()))

    # Build the DeepGPModel with the scaled data
    feature_extractor = FeatureExtractor()
    likelihood = GaussianLikelihood()
    model = DeepGPModel(X_scaled, Y_target, likelihood, feature_extractor)
    
    model_path = f"dkl_kernel_for_{task}.pth"
    print(f"  Loading model for '{task}' from '{model_path}'...")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Freeze parameters
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    for param in model.covar_module.parameters():
        param.requires_grad = False
    
    gp_models[task] = model

evaluate_gp_models(gp_models, X, Y_targets, scalers, n_samples=10, label="After loading pre-trained kernels")

# def acquisiton func and helpers
def entropy_sigma_improved(X_grid_raw, gps, scalers, constraints, thickness, Xtrain_raw=None, nmc=64, alpha_dist=0.1, top_k=5):
    gpw, gpl, gpd = gps['Width'], gps['Length'], gps['Depth']
    scaler_w, scaler_l, scaler_d = scalers['Width'], scalers['Length'], scalers['Depth']
    c1, c2, c3 = constraints
    N = X_grid_raw.shape[0]
    jitter = 1e-9

    with torch.no_grad():
        # Scale inputs for each model before getting posterior
        X_grid_w_scaled = torch.from_numpy(scaler_w.transform(X_grid_raw.numpy()))
        postw = gpw(X_grid_w_scaled); meanw = postw.mean; stdw = postw.variance.sqrt().clamp(min=1e-6)
        
        X_grid_l_scaled = torch.from_numpy(scaler_l.transform(X_grid_raw.numpy()))
        postl = gpl(X_grid_l_scaled); meanl = postl.mean; stdl = postl.variance.sqrt().clamp(min=1e-6)
        
        X_grid_d_scaled = torch.from_numpy(scaler_d.transform(X_grid_raw.numpy()))
        postd = gpd(X_grid_d_scaled); meand = postd.mean; stdd = postd.variance.sqrt().clamp(min=1e-6)

    meanw_np, stdw_np = meanw.cpu().numpy(), stdw.cpu().numpy()
    meanl_np, stdl_np = meanl.cpu().numpy(), stdl.cpu().numpy()
    meand_np, stdd_np = meand.cpu().numpy(), stdd.cpu().numpy()

    with torch.no_grad():
        samplew = gpw.likelihood(gpw(X_grid_w_scaled)).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
        samplel = gpl.likelihood(gpl(X_grid_l_scaled)).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
        sampled = gpd.likelihood(gpd(X_grid_d_scaled)).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()

    J = np.zeros(N)
    for i in range(N):
        sw, sl, sd = samplew[:, i], samplel[:, i], sampled[:, i]
        keyholing = (sw / (sd + jitter)) < c1
        lof = ((sd + jitter) / (thickness + jitter)) < c2
        balling = (sl / (sw + jitter)) > c3
        p1, p2, p3 = float(np.mean(keyholing)), float(np.mean(lof)), float(np.mean(balling))
        p4 = max(1.0 - (p1 + p2 + p3), 1e-12)
        probs = np.array([p1, p2, p3, p4]); probs = np.clip(probs, 1e-12, 1-1e-12)
        H = -(probs * np.log(probs)).sum()
        total_std = float(stdw_np[i] + stdl_np[i] + stdd_np[i])
        J[i] = H * total_std
        
    if Xtrain_raw is not None:
        dists = np.min(np.linalg.norm(X_grid_raw.numpy()[:, None, :] - Xtrain_raw.numpy()[None, :, :], axis=-1), axis=1)
        J *= (1 + alpha_dist * dists)
        
    top_indices = np.argsort(-J)[:top_k]
    return J, top_indices


def plot_mc_defect_map(labels_grid, powergrid, velogrid, use_balling=USE_BALLING, alpha_defects=0.7, J=None, iteration=None):
    if J is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_defect, ax_acq = axes
    else:
        fig, ax_defect = plt.subplots(figsize=(10,7))
        ax_acq = None
    label_to_num = {"Good": 0, "Keyhole": 1, "Lack of Fusion": 3, "Balling": 2}
    defect_numeric_grid = np.vectorize(label_to_num.get)(labels_grid)
    red = np.array([224, 123, 123]) / 255; blue = np.array([123, 191, 200]) / 255; green = np.array([40, 156, 142]) / 255
    rgb_grid = np.ones((labels_grid.shape[0], labels_grid.shape[1], 3))
    for i in range(3):
        rgb_grid[..., i][defect_numeric_grid == 1] = alpha_defects*red[i] + (1-alpha_defects)*1
        rgb_grid[..., i][defect_numeric_grid == 3] = alpha_defects*blue[i] + (1-alpha_defects)*1
        rgb_grid[..., i][defect_numeric_grid == 2] = alpha_defects*green[i] + (1-alpha_defects)*1
    ax_defect.imshow(rgb_grid, extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()], origin="lower", aspect="auto", zorder=1)
    ax_defect.set_xlabel("Scan Velocity (mm/s)"); ax_defect.set_ylabel("Laser Power (W)")
    title_defect = "316L Printability Map (Monte Carlo)"
    if iteration is not None: title_defect += f" - Iter {iteration}"
    ax_defect.set_title(title_defect)
    legend_elements = [ Patch(facecolor="#E07B7B", label="Keyhole W/D < 1.5"), Patch(facecolor="#7bbfc8", label="Lack of Fusion D/t <1.9"), ]
    if use_balling: legend_elements.append(Patch(facecolor="#289C8E", label="Balling W/L < 0.23"))
    legend_elements.append(Patch(facecolor="#FFFFFF", edgecolor="black", label="Stable/Printable"))
    ax_defect.legend(handles=legend_elements, loc="best")
    if J is not None and ax_acq is not None:
        grid_J = J.reshape(labels_grid.shape)
        im = ax_acq.imshow(grid_J, origin="lower", extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()], cmap='viridis', aspect='auto')
        ax_acq.set_xlabel("Scan Velocity (mm/s)"); ax_acq.set_ylabel("Laser Power (W)")
        title_acq = "Acquisition Function"
        if iteration is not None: title_acq += f" - Iter {iteration}"
        ax_acq.set_title(title_acq)
        fig.colorbar(im, ax=ax_acq, label='Acquisition Value')
    plt.tight_layout(); plt.show()


# main active learning loop
with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
    start.set_cache_folder("cache"); mp = MaterialProperties.from_library(MATERIAL_NAME)
    am_calculator = (start.with_additive_manufacturing().with_steady_state_calculation().with_numerical_options(NumericalOptions().set_number_of_cores(20)).disable_fluid_flow_marangoni().with_material_properties(mp).with_mesh(Mesh().coarse()))
    am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE); am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)
    heat_source = HeatSource.from_library(HEAT_SOURCE_NAME); am_calculator.with_heat_source(heat_source)

#make grid
powermin = data["Power"].min(); powermax = data["Power"].max()
velomin = data["Speed"].min(); velomax = data["Speed"].max()
powergrid = np.linspace(powermin, powermax, ngrid); velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij")
Xgrid = torch.from_numpy(np.column_stack([PP.ravel(), VV.ravel()])) # Grid is in raw units

# Active learning loop
success_it = 0
while success_it < niter:
    with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
        start.set_cache_folder("cache"); mp = MaterialProperties.from_library(MATERIAL_NAME)

        J, top_candidates = entropy_sigma_improved(
            Xgrid,
            gps=gp_models,
            scalers=scalers,
            constraints=constraints,
            thickness=thickness,
            Xtrain_raw=X,
            nmc=nmc,
            alpha_dist=0.1,
            top_k=5
        )

        sorted_idx = top_candidates[:3]
        d_next = w_next = l_next = None
        x_next = None

        for ind in sorted_idx:
            try:
                x_next = Xgrid[ind:ind + 1]
                
                # ... (TC-Python calculation logic is unchanged) ...
                am_calculator = (start.with_additive_manufacturing().with_steady_state_calculation().with_numerical_options(NumericalOptions().set_number_of_cores(20)).disable_fluid_flow_marangoni().with_material_properties(mp).with_mesh(Mesh().coarse()))
                am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE); am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)
                heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
                heat_source.set_power(float(x_next[0, 0].item())); heat_source.set_scanning_speed(float(x_next[0, 1].item()) / 1e3)
                am_calculator.with_heat_source(heat_source)
                result: SteadyStateResult = am_calculator.calculate()
                d_next = float(result.get_meltpool_depth()) * 1e6
                w_next = float(result.get_meltpool_width()) * 1e6
                l_next = float(result.get_meltpool_length()) * 1e6
                break
            except tc_python.exceptions.CalculationException:
                print(f"[trial] Thermo-Calc failed at candidate {ind}, trying next-best...")
                continue
        if x_next is None or d_next is None:
            print(f"[trial] Top 3 candidates failed, retrying without incrementing iteration.")
            continue
        del result, am_calculator, heat_source; gc.collect(); time.sleep(0.05)

    # Add new data point (use raw, unscaled X)
    X = torch.cat([X, x_next], dim=0)
    Yd = torch.cat([Yd, torch.tensor([[d_next]], dtype=dtype, device=device)], dim=0)
    Yw = torch.cat([Yw, torch.tensor([[w_next]], dtype=dtype, device=device)], dim=0)
    Yl = torch.cat([Yl, torch.tensor([[l_next]], dtype=dtype, device=device)], dim=0)
    Y_targets = {"Depth": Yd, "Width": Yw, "Length": Yl}

    # Fine-tune models
    print("--- Fine-tuning model likelihoods with new data point ---")
    for task, model in gp_models.items():
        scaler = scalers[task]
        X_scaled = torch.from_numpy(scaler.transform(X.numpy()))
        
        # Re-instantiate a clean model with the full, updated dataset
        new_model = DeepGPModel(X_scaled, Y_targets[task], model.likelihood, model.feature_extractor)
        for param in new_model.feature_extractor.parameters():
            param.requires_grad = False
        for param in new_model.covar_module.parameters():
            param.requires_grad = False
        
        new_model.train()
        mll = ExactMarginalLogLikelihood(new_model.likelihood, new_model)
        fit_gpytorch_mll(mll)
        gp_models[task] = new_model

    success_it += 1
    print(f"[iter {success_it}/{niter}] DONE: P={x_next[0,0]:.2f}, V={x_next[0,1]:.2f}, W={w_next:.2f}, D={d_next:.2f}, L={l_next:.2f}")

    if (success_it) % 5 == 0 or success_it == niter:
        with torch.no_grad():
            gpw, gpd, gpl = gp_models["Width"], gp_models["Depth"], gp_models["Length"]
            
            # Manually scale the grid for each model before prediction
            Xgrid_w_scaled = torch.from_numpy(scalers['Width'].transform(Xgrid.numpy()))
            Xgrid_d_scaled = torch.from_numpy(scalers['Depth'].transform(Xgrid.numpy()))
            Xgrid_l_scaled = torch.from_numpy(scalers['Length'].transform(Xgrid.numpy()))

            width_samples  = gpw.likelihood(gpw(Xgrid_w_scaled)).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
            depth_samples  = gpd.likelihood(gpd(Xgrid_d_scaled)).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
            length_samples = gpl.likelihood(gpl(Xgrid_l_scaled)).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()

        labels_grid_mc = classify_defect_mc(width_samples, depth_samples, length_samples=length_samples, thickness=thickness)
        labels_grid_mc = labels_grid_mc.reshape(ngrid, ngrid)
        plot_mc_defect_map(labels_grid_mc, powergrid, velogrid, J=J, iteration=success_it)
        plt.close('all')

# Final evaluation uses raw X data
preds, rmses = evaluate_gp_models(gp_models, X, Y_targets, scalers, n_samples=len(X), label="After Active Learning")
X_probe = torch.tensor([[300., 1200.], [400., 800.], [250., 1800.]], dtype=dtype, device=device)

# Evaluate
probe_preds, _ = evaluate_gp_models(gp_models, X_probe, None, scalers, label="Probe Points")