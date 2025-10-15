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
import joblib

# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel, Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from torch import nn

# BoTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples

#scipy
from scipy.interpolate import griddata
from scipy.stats import norm

#tcpython
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
ntrain = 5
niter = 15#how many AL loops after  training are allowed
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
plotiter = 2  # plot every 'plotiter' iterations

constraints = [keyholing, lof, balling]

#store the runs and stuff!
CSV_FILE = "active_learning_runs.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = pd.DataFrame(columns=["power", "speed", "Depth", "Width", "Length"])
    df.to_csv(CSV_FILE, index=False)

#import the data 
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

# training inputs
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
m= Yd.shape[1]

## -------------------------------------------------------------------------- ##
## DEFINE DKL MODEL COMPONENTS (FIX: Added the missing class definitions here)
## -------------------------------------------------------------------------- ##
class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', nn.Linear(2, 100))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(100, 50))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(50, 2))

class PreTrainedDKLKernel(Kernel):
    def __init__(self, feature_extractor, base_kernel):
        super(PreTrainedDKLKernel, self).__init__()
        self.feature_extractor = feature_extractor
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        projected_x1 = self.feature_extractor(x1)
        projected_x2 = self.feature_extractor(x2)
        return self.base_kernel.forward(projected_x1, projected_x2, diag=diag, **params)

#def funcs
def FitDKL_GP(X, Y, task, model_path_template="dkl_kernel_for_{}.pth", scaler_path="scaler.pkl"):
    # 1. Instantiate components AND SET THEIR DTYPE
    feature_extractor = FeatureExtractor().to(device=device, dtype=dtype)
    base_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)).to(device=device, dtype=dtype)
    likelihood = GaussianLikelihood().to(device=device, dtype=dtype)

    # 2. Load the state dict (which is already in torch.double)
    model_path = model_path_template.format(task)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained DKL model not found: {model_path}")
    full_state_dict = torch.load(model_path, map_location=device)

    # 3. Load parameters into each component (robust key matching)
    fe_state = {k.replace('feature_extractor.', ''): v for k, v in full_state_dict.items() if k.startswith('feature_extractor.')}
    cov_state = {k.replace('covar_module.', ''): v for k, v in full_state_dict.items() if k.startswith('covar_module.')}
    lik_state = {k.replace('likelihood.', ''): v for k, v in full_state_dict.items() if k.startswith('likelihood.')}

    # apply to modules (use strict=False to be robust to minor naming differences)
    if len(fe_state) > 0:
        feature_extractor.load_state_dict(fe_state, strict=False)
    if len(cov_state) > 0:
        base_kernel.load_state_dict(cov_state, strict=False)
    if len(lik_state) > 0:
        likelihood.load_state_dict(lik_state, strict=False)

    # 4. Freeze parameters
    feature_extractor.eval()
    base_kernel.eval()
    likelihood.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    for param in base_kernel.parameters():
        param.requires_grad = False
    for param in likelihood.parameters():
        param.requires_grad = False

    # 5. Create the custom DKL kernel
    dkl_kernel = PreTrainedDKLKernel(feature_extractor, base_kernel)

    # 6. Ensure X and Y are on same device & dtype
    X = X.to(device=device, dtype=dtype)
    Y = Y.to(device=device, dtype=dtype)

    # 7. Create the final SingleTaskGP model
    # IMPORTANT: Do NOT add extra Normalize/Standardize that will change the input scale
    # (we expect inputs to already be scaled the same way as during training).
    # We'll still construct BoTorch transforms with care: wrap only if you intentionally want them.
    try:
        model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=dkl_kernel, likelihood=likelihood,
                             input_transform=None, outcome_transform=None)
    except Exception:
        # fallback in case of API differences
        model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=dkl_kernel, likelihood=likelihood)

    # ensure model on device
    model = model.to(device)

    return model

def fitGP(model, restarts=restarts):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # Use fit_gpytorch_mll with default optimizer; small maxiter to avoid long runs
    fit_gpytorch_mll(mll, optimizer_cls=None, options={"maxiter": 200}, num_restarts=restarts)
    model = model.to(device)
    return model

def evaluate_gp_models(gp_models, X, Y_targets, n_samples=10, label=""):
    preds, rmses = {}, {}

    for task, gp in gp_models.items():
        gp.eval()
        with torch.no_grad():
            posterior = gp.posterior(X)
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
    for i in range(min(n_samples, len(X))):
        if Y_targets is not None:
            true_vals = [f"{Y_targets[t][i].item():.2f}" for t in ["Depth", "Width", "Length"]]
            pred_vals = [f"{preds[t][i]:.2f}" for t in ["Depth", "Width", "Length"]]
            print(f"  True: {true_vals} | Pred: {pred_vals}")
        else:
            pred_vals = [f"{preds[t][i]:.2f}" for t in ["Depth", "Width", "Length"]]
            print(f"  Probe Point {i+1} Pred: {pred_vals}")

    return preds, rmses

# ... (The rest of your script is unchanged) ...

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
# attempt to load scaler used during training if present; otherwise create and save one
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    # Fit MinMax on full data (to preserve the same scaling that the training script used)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.column_stack([power, speed]))
    joblib.dump(scaler, scaler_path)

# Transform initial X to the training scale used by the original DKL trainer
# NOTE: original DKL script used MinMaxScaler on inputs before training.
X_np_for_scaling = np.column_stack([power[initial_idx], speed[initial_idx]])
X_scaled_np = scaler.transform(X_np_for_scaling)
X = torch.tensor(X_scaled_np, dtype=dtype, device=device)

# training outputs remain raw (as in training script)
Yd = torch.tensor(depth[initial_idx], dtype=dtype, device=device)
Yw = torch.tensor(width[initial_idx], dtype=dtype, device=device)
Yl = torch.tensor(length[initial_idx], dtype=dtype, device=device)

Y_targets = {"Depth": Yd, "Width": Yw, "Length": Yl}

for task, Y in Y_targets.items():
    gp = FitDKL_GP(X, Y, task)
    gp = fitGP(gp, restarts=restarts)
    gp_models[task] = gp

evaluate_gp_models(gp_models, X, Y_targets, n_samples=10, label="After initial DKL training")

#def acquisiton func and helpers

def entropy_sigma_improved(X, gps, constraints, thickness, Xtrain=None, mode="MC", nmc=64, alpha_dist=0.1, top_k=5):

    gpw, gpd, gpl = gps['Width'], gps['Depth'], gps['Length']
    c1, c2, c3 = constraints
    N = X.shape[0]

    jitter = 1e-9

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

    J = np.zeros(N)
    for i in range(N):
        sw, sl, sd = samplew[:, i], samplel[:, i], sampled[:, i]

        keyholing = (sw / (sd + jitter)) < c1
        lof = ((sd + jitter) / (thickness + jitter)) < c2
        balling = (sl / (sw + jitter)) > c3

        p1, p2, p3 = float(np.mean(keyholing)), float(np.mean(lof)), float(np.mean(balling))
        p4 = max(1.0 - (p1 + p2 + p3), 1e-12)
        probs = np.array([p1, p2, p3, p4])
        probs = np.clip(probs, 1e-12, 1-1e-12)
        H = -(probs * np.log(probs)).sum()

        total_std = float(stdw_np[i] + stdl_np[i] + stdd_np[i])
        J[i] = H * total_std

    if Xtrain is not None:
        Xtrain_np = Xtrain.cpu().numpy()
        dists = np.min(np.linalg.norm(X[:, None, :].cpu().numpy() - Xtrain_np[None, :, :], axis=-1), axis=1)
        J *= (1 + alpha_dist * dists)

    top_indices = np.argsort(-J)[:top_k]
    return J, top_indices

def plot_mc_defect_map(labels_grid, powergrid, velogrid, use_balling=USE_BALLING, alpha_defects=0.7, J=None, iteration=None, Xtrain=None, newest_point=None):

    if J is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_defect, ax_acq = axes
    else:
        fig, ax_defect = plt.subplots(figsize=(10,7))
        ax_acq = None

    label_to_num = {"Good": 0, "Keyhole": 1, "Lack of Fusion": 3, "Balling": 2}
    defect_numeric_grid = np.vectorize(label_to_num.get)(labels_grid)

    red = np.array([224, 123, 123]) / 255
    blue = np.array([123, 191, 200]) / 255
    green = np.array([40, 156, 142]) / 255

    rgb_grid = np.ones((labels_grid.shape[0], labels_grid.shape[1], 3))
    for i in range(3):
        rgb_grid[..., i][defect_numeric_grid == 1] = alpha_defects*red[i] + (1-alpha_defects)*1
        rgb_grid[..., i][defect_numeric_grid == 3] = alpha_defects*blue[i] + (1-alpha_defects)*1
        rgb_grid[..., i][defect_numeric_grid == 2] = alpha_defects*green[i] + (1-alpha_defects)*1

    ax_defect.imshow(rgb_grid, extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()],
                       origin="lower", aspect="auto", zorder=1)
    ax_defect.set_xlabel("Scan Velocity (mm/s)")
    ax_defect.set_ylabel("Laser Power (W)")
    title_defect = "316L Printability Map (Monte Carlo)"
    if iteration is not None:
        title_defect += f" - Iter {iteration}"
    ax_defect.set_title(title_defect)

    # If training points provided, plot them: previous points as black dots, newest point as a star
    if Xtrain is not None:
        try:
            # Convert to numpy float array safely (supports torch tensors)
            if hasattr(Xtrain, 'detach'):
                Xtrain_np = Xtrain.detach().cpu().numpy().astype(float)
            elif hasattr(Xtrain, 'cpu'):
                Xtrain_np = Xtrain.cpu().numpy().astype(float)
            else:
                Xtrain_np = np.asarray(Xtrain, dtype=float)
        except Exception:
            Xtrain_np = np.asarray(Xtrain)

        # Normalize shape to (n,2)
        if Xtrain_np.ndim == 1 and Xtrain_np.size == 2:
            Xtrain_np = Xtrain_np.reshape(1, 2)
        elif Xtrain_np.ndim > 2:
            Xtrain_np = Xtrain_np.reshape(Xtrain_np.shape[0], -1)
            if Xtrain_np.shape[1] >= 2:
                Xtrain_np = Xtrain_np[:, :2]
            else:
                Xtrain_np = np.empty((0, 2))

        # If a scaler is available (we scaled inputs before modeling), inverse-transform
        try:
            if 'scaler' in globals() and Xtrain_np.size != 0:
                Xtrain_np = scaler.inverse_transform(Xtrain_np)
        except Exception:
            # If inverse transform fails, continue with original values
            pass

        if Xtrain_np.size != 0 and Xtrain_np.shape[1] >= 2:
            # Diagnostic prints: show ranges and sample values before plotting
            try:
                print(f"[plot dbg] Xtrain_np.shape={Xtrain_np.shape}")
                if Xtrain_np.size != 0:
                    print(f"[plot dbg] col0(min,max)={Xtrain_np[:,0].min():.3f},{Xtrain_np[:,0].max():.3f} | col1(min,max)={Xtrain_np[:,1].min():.3f},{Xtrain_np[:,1].max():.3f}")
                    print(f"[plot dbg] sample rows:\n{Xtrain_np[:5]}")
            except Exception as e:
                print(f"[plot dbg] failed diagnostics: {e}")

            # Plot previous points (all but last) as black dots
            if Xtrain_np.shape[0] > 1:
                prev = Xtrain_np[:-1]
                ax_defect.scatter(prev[:, 1], prev[:, 0], color='k', s=30, marker='o', zorder=5, label='Previous')
            # Plot newest point as a star
            newest_np = Xtrain_np[-1]
            try:
                print(f"[plot dbg] newest (power,vel) = ({newest_np[0]:.3f}, {newest_np[1]:.3f})")
            except Exception:
                pass
            ax_defect.scatter(newest_np[1], newest_np[0], color='red', s=120, marker='*', edgecolor='k', zorder=6, label='Newest')
    else:
        # If newest_point provided separately, plot it
        if newest_point is not None:
            try:
                if hasattr(newest_point, 'detach'):
                    new_np = newest_point.detach().cpu().numpy().astype(float).reshape(-1)
                elif hasattr(newest_point, 'cpu'):
                    new_np = newest_point.cpu().numpy().reshape(-1)
                else:
                    new_np = np.asarray(newest_point, dtype=float).reshape(-1)
                if new_np.size >= 2:
                    try:
                        print(f"[plot dbg] newest_point (power,vel) = ({new_np[0]:.3f}, {new_np[1]:.3f})")
                    except Exception:
                        pass
                    ax_defect.scatter(new_np[1], new_np[0], color='red', s=120, marker='*', edgecolor='k', zorder=6, label='Newest')
            except Exception:
                pass

    # Add legend entries for plotted training points if present
    handles, labels = ax_defect.get_legend_handles_labels()
    # ensure defect legend remains by adding custom legend elements
    custom = [Patch(facecolor="#E07B7B", label="Keyhole W/D < 1.5"),
              Patch(facecolor="#7bbfc8", label="Lack of Fusion D/t <1.9")]
    if use_balling:
        custom.append(Patch(facecolor="#289C8E", label="Balling W/L < 0.23"))
    custom.append(Patch(facecolor="#FFFFFF", edgecolor="black", label="Stable/Printable"))
    # extend with training handles if they exist
    if handles:
        custom.extend(handles)
    ax_defect.legend(handles=custom, loc='best')

    legend_elements = [
        Patch(facecolor="#E07B7B", label="Keyhole W/D < 1.5"),
        Patch(facecolor="#7bbfc8", label="Lack of Fusion D/t <1.9"),
    ]
    if use_balling:
        legend_elements.append(Patch(facecolor="#289C8E", label="Balling W/L < 0.23"))
    legend_elements.append(Patch(facecolor="#FFFFFF", edgecolor="black", label="Stable/Printable"))
    ax_defect.legend(handles=legend_elements, loc="best")

    if J is not None and ax_acq is not None:
        grid_J = J.reshape(labels_grid.shape)
        im = ax_acq.imshow(grid_J, origin="lower",
                              extent=[velogrid.min(), velogrid.max(), powergrid.min(), powergrid.max()],
                              cmap='viridis', aspect='auto')
        ax_acq.set_xlabel("Scan Velocity (mm/s)")
        ax_acq.set_ylabel("Laser Power (W)")
        title_acq = "Acquisition Function"
        if iteration is not None:
            title_acq += f" - Iter {iteration}"
        ax_acq.set_title(title_acq)
        fig.colorbar(im, ax=ax_acq, label='Acquisition Value')

    plt.tight_layout()
    plt.show()

# plug that into tc and stuff or whatever
## main active learning loop
# initialized TC
with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
    start.set_cache_folder("cache")

    mp = MaterialProperties.from_library(MATERIAL_NAME)

    am_calculator = (
        start.with_additive_manufacturing()
        .with_steady_state_calculation()
        .with_numerical_options(NumericalOptions().set_number_of_cores(20))
        .disable_fluid_flow_marangoni()
        .with_material_properties(mp)
        .with_mesh(Mesh().coarse())
    )
    am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE)
    am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)
    heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
    am_calculator.with_heat_source(heat_source)

#make grid
# make grid for plotting and acquisition
powermin = data["Power"].min()
powermax = data["Power"].max()
velomin = data["Speed"].min()
velomax = data["Speed"].max()

powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij")

gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
# scale the gridpoints using the same scaler used during training
gridpoints_scaled = scaler.transform(gridpoints)
Xgrid = torch.tensor(gridpoints_scaled, dtype=dtype, device=device)

# store active learning history
# ... (rest of your script remains unchanged)

# Active learning loop with top-3 fallback and only counting successful iterations
success_it = 0

while success_it < niter:
    with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
        start.set_cache_folder("cache")
        mp = MaterialProperties.from_library(MATERIAL_NAME)

        J, top_candidates = entropy_sigma_improved(
            Xgrid,
            gps=gp_models,
            constraints=constraints,
            thickness=thickness,
            Xtrain=X,
            mode="MC",
            nmc=nmc,
            alpha_dist=0.1,
            top_k=5
        )

        sorted_idx = top_candidates[:3]
        d_next = w_next = l_next = None
        x_next = None

        for ind in sorted_idx:
            try:
                # When selecting a candidate, remember Xgrid is scaled; convert back to real values for TC
                x_next_scaled = Xgrid[ind:ind+1].cpu().numpy()
                # inverse transform to real power/speed for TC
                x_next_real = scaler.inverse_transform(x_next_scaled)
                x_next = torch.tensor(x_next_scaled, dtype=dtype, device=device)

                am_calculator = (
                    start.with_additive_manufacturing()
                    .with_steady_state_calculation()
                    .with_numerical_options(NumericalOptions().set_number_of_cores(20))
                    .disable_fluid_flow_marangoni()
                    .with_material_properties(mp)
                    .with_mesh(Mesh().coarse())
                )
                am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE)
                am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)

                heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
                heat_source.set_power(float(x_next_real[0, 0]))
                # your original script divides speed by 1e3 for TC
                heat_source.set_scanning_speed(float(x_next_real[0, 1]) / 1e3)
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

        del result, am_calculator, heat_source
        gc.collect()
        time.sleep(0.05)

    # no more tc hehe
    d_next_t = torch.tensor([[d_next]], dtype=dtype, device=device)
    w_next_t = torch.tensor([[w_next]], dtype=dtype, device=device)
    l_next_t = torch.tensor([[l_next]], dtype=dtype, device=device)

    # Append the scaled x_next (already scaled) to X
    X = torch.cat([X, x_next], dim=0)
    Yd = torch.cat([Yd, d_next_t], dim=0)
    Yw = torch.cat([Yw, w_next_t], dim=0)
    Yl = torch.cat([Yl, l_next_t], dim=0)
    
    Y_targets = {"Depth": Yd, "Width": Yw, "Length": Yl}

    # THIS IS THE FINAL, ROBUST RETRAINING LOOP
    print("--- Re-fitting models from scratch with new data point ---")
    for task, Y in Y_targets.items():
        gp = FitDKL_GP(X, Y, task)  # Build a fresh, clean model
        gp = fitGP(gp, restarts=restarts)  # Fine-tune its noise
        gp_models[task] = gp
    
    success_it += 1

    print(f"[iter {success_it}/{niter}] DONE: "
          f"P={float(scaler.inverse_transform(x_next.cpu().numpy())[0,0]):.2f}, V={float(scaler.inverse_transform(x_next.cpu().numpy())[0,1]):.2f}, "
          f"W={w_next:.2f}, D={d_next:.2f}, L={l_next:.2f}")

    if (success_it) % plotiter == 0 or success_it == niter:
        with torch.no_grad():
            gpw, gpd, gpl = gp_models["Width"], gp_models["Depth"], gp_models["Length"]
            width_samples  = gpw.posterior(Xgrid).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
            depth_samples  = gpd.posterior(Xgrid).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
            length_samples = gpl.posterior(Xgrid).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()

        labels_grid_mc = classify_defect_mc(
            width_samples,
            depth_samples,
            e_samples=None,
            length_samples=length_samples,
            thickness=thickness
        )
        labels_grid_mc = labels_grid_mc.reshape(ngrid, ngrid)

        plot_mc_defect_map(
            labels_grid_mc,
            powergrid,
            velogrid,
            use_balling=USE_BALLING,
            alpha_defects=0.7,
            J=J,
            iteration=success_it,
            Xtrain=X,
            newest_point=x_next
        )
        plt.close('all')

preds, rmses = evaluate_gp_models(gp_models, X, Y_targets, n_samples=len(X), label="After Active Learning")
X_probe_real = np.array([
    [300, 1200],
    [400, 800],
    [250, 1800]
])
# scale probes before sending into GPs
X_probe_scaled = scaler.transform(X_probe_real)
X_probe = torch.tensor(X_probe_scaled, dtype=dtype, device=device)

# Evaluate
probe_preds, _ = evaluate_gp_models(gp_models, X_probe, Y_targets=None, label="Probe Points")
#grrrr