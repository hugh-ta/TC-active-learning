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


# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

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

# fit initial GP models
gp_models = {}
for task, Y in Y_targets.items():
    gp = FitGP(X, Y)
    gp = fitGP(gp, restarts=restarts)
    gp_models[task] = gp

evaluate_gp_models(gp_models, X, Y_targets, n_samples=10, label="After Sobol training")

evaluate_gp_models(gp_models, X, Y_targets, n_samples=10, label="After training")

#def acquisiton func and helpers
def _evaluate_point_worker(i, meanw_np, stdw_np, meanl_np, stdl_np, meand_np, stdd_np,
                           samplew_np, samplel_np, sampled_np, c1, c2, c3, thickness,
                           mode):
    jitter = 1e-9
    if mode == "MC":
        sw = samplew_np[:, i]; sl = samplel_np[:, i]; sd = sampled_np[:, i]
        keyholing = (sw / (sd + jitter)) < c1
        lof = ((sd + jitter) / (thickness + jitter)) < c2
        balling = (sl / (sw + jitter)) > c3
        p1, p2, p3 = float(np.mean(keyholing)), float(np.mean(lof)), float(np.mean(balling))
        p4 = max(1.0 - (p1 + p2 + p3), 1e-12)
    elif mode == "Blind":
        p1, p2, p3, p4 = 0.0, 0.0, 0.0, 1.0
    else:
        raise ValueError("Unknown mode")

    probs = np.array([p1, p2, p3, p4], dtype=float)
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    H = -(probs * np.log(probs)).sum()

    # Include joint standard deviation like ACS debug version
    joint_std = float(stdw_np[i] * stdl_np[i] * stdd_np[i])
    return float(H * joint_std)
# JOINT STD IS GONE FOR NOW!!!
def entropy_sigma_improved(X, gps, constraints, thickness, Xtrain=None, mode="MC", nmc=64, alpha_dist=0.1, top_k=5):

    gpw, gpl, gpd = gps
    c1, c2, c3 = constraints
    N = X.shape[0]

    jitter = 1e-9

    # GP posteriors
    with torch.no_grad():
        postw = gpw.posterior(X); meanw = postw.mean.squeeze(-1); stdw = postw.variance.sqrt().squeeze(-1).clamp(min=1e-6)
        postl = gpl.posterior(X); meanl = postl.mean.squeeze(-1); stdl = postl.variance.sqrt().squeeze(-1).clamp(min=1e-6)
        postd = gpd.posterior(X); meand = postd.mean.squeeze(-1); stdd = postd.variance.sqrt().squeeze(-1).clamp(min=1e-6)

    meanw_np, stdw_np = meanw.cpu().numpy(), stdw.cpu().numpy()
    meanl_np, stdl_np = meanl.cpu().numpy(), stdl.cpu().numpy()
    meand_np, stdd_np = meand.cpu().numpy(), stdd.cpu().numpy()

    # MC sampling
    with torch.no_grad():
        samplew = gpw.posterior(X).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
        samplel = gpl.posterior(X).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()
        sampled = gpd.posterior(X).rsample(torch.Size([nmc])).squeeze(-1).cpu().numpy()

    # Compute acquisition per point
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

        # sum of per-output std instead of product
        total_std = float(stdw_np[i] + stdl_np[i] + stdd_np[i])
        J[i] = H * total_std

    # Distance penalty to encourage exploration
    if Xtrain is not None:
        Xtrain_np = Xtrain.cpu().numpy()
        dists = np.min(np.linalg.norm(X[:, None, :].cpu().numpy() - Xtrain_np[None, :, :], axis=-1), axis=1)
        J *= (1 + alpha_dist * dists)

    # Return sorted top-k candidates for diversified AL
    top_indices = np.argsort(-J)[:top_k]  # top-k
    return J, top_indices

def plot_mc_defect_map(labels_grid, powergrid, velogrid, use_balling=USE_BALLING, alpha_defects=0.7, J=None, iteration=None):

    if J is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_defect, ax_acq = axes
    else:
        fig, ax_defect = plt.subplots(figsize=(10,7))
        ax_acq = None

    #print map yurrr
    label_to_num = {"Good": 0, "Keyhole": 1, "Lack of Fusion": 3, "Balling": 2}
    defect_numeric_grid = np.vectorize(label_to_num.get)(labels_grid)

    red = np.array([224, 123, 123]) / 255  # Keyhole
    blue = np.array([123, 191, 200]) / 255  # Lack of Fusion
    green = np.array([40, 156, 142]) / 255  # Balling

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

    legend_elements = [
        Patch(facecolor="#E07B7B", label="Keyhole W/D < 1.5"),
        Patch(facecolor="#7bbfc8", label="Lack of Fusion D/t <1.9"),
    ]
    if use_balling:
        legend_elements.append(Patch(facecolor="#289C8E", label="Balling W/L < 0.23"))
    legend_elements.append(Patch(facecolor="#FFFFFF", edgecolor="black", label="Stable/Printable"))
    ax_defect.legend(handles=legend_elements, loc="best")

    #acq funq
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
def run_thermocalc(power, speed, am_calculator, heat_source, mp, wait_time=5, max_retries=5):
    heat_source.set_power(float(power))
    heat_source.set_scanning_speed(float(speed) / 1e3)  # mm/s → m/s

    retries = 0
    while True:
        try:
            result: SteadyStateResult = am_calculator.calculate()  # blocks until done
            break
        except UnrecoverableCalculationException:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"TC failed after {max_retries} retries.")
            print(f"TC busy or internal server error, retrying in {wait_time}s... (Attempt {retries})")
            time.sleep(wait_time)

    return {
        "power": power,
        "speed": speed,
        "Depth": result.get_meltpool_depth() * 1e6,   # µm
        "Width": result.get_meltpool_width() * 1e6,
        "Length": result.get_meltpool_length() * 1e6,
    }
## main active learning loop
# initialized TC
with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
    start.set_cache_folder("cache")  # optional, keeps results cached

    # Load material properties
    mp = MaterialProperties.from_library(MATERIAL_NAME)

    # Setup AM steady-state calculator
    am_calculator = (
        start.with_additive_manufacturing()
        .with_steady_state_calculation()
        .with_numerical_options(NumericalOptions().set_number_of_cores(20))
        .disable_fluid_flow_marangoni()
        .with_material_properties(mp)
        .with_mesh(Mesh().coarse())
    )

    # Set ambient/base temperatures
    am_calculator.set_ambient_temperature(AMBIENT_TEMPERATURE)
    am_calculator.set_base_plate_temperature(AMBIENT_TEMPERATURE)

    # Load heat source
    heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
    am_calculator.with_heat_source(heat_source)

#make grid
# make grid for plotting and acquisition
powermin, powermax = X[:, 0].min().item(), X[:, 0].max().item()
velomin, velomax = X[:, 1].min().item(), X[:, 1].max().item()

powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij")

gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
Xgrid = torch.tensor(gridpoints, dtype=dtype, device=device)

# store active learning history
gps_history = []
Xtrain_history = []
J_history = []

# Active learning loop with top-3 fallback and only counting successful iterations
success_it = 0  # counts successful iterations

while success_it < niter:
    with TCPython(logging_policy=LoggingPolicy.SCREEN) as start:
        start.set_cache_folder("cache")
        mp = MaterialProperties.from_library(MATERIAL_NAME)

        # Pick next x with top-3 fallback
        gps = [gp_models["Width"], gp_models["Depth"], gp_models["Length"]]
        J, top_candidates = entropy_sigma_improved(
            Xgrid,
            gps=[gp_models["Width"], gp_models["Length"], gp_models["Depth"]],
            constraints=constraints,
            thickness=thickness,
            Xtrain=X,
            mode="MC",
            nmc=nmc,
            alpha_dist=0.1,
            top_k=5
        )

        # pick candidates in order for top-3 fallback
        sorted_idx = top_candidates[:3]
        d_next = w_next = l_next = None
        x_next = None

        for ind in sorted_idx:
            try:
                x_next = Xgrid[ind:ind+1]

                # New calculator each time
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

                # New heat source each time
                heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
                heat_source.set_power(float(x_next[0, 0].item()))
                heat_source.set_scanning_speed(float(x_next[0, 1].item()) / 1e3)
                am_calculator.with_heat_source(heat_source)

                result: SteadyStateResult = am_calculator.calculate()

                d_next = float(result.get_meltpool_depth()) * 1e6
                w_next = float(result.get_meltpool_width()) * 1e6
                l_next = float(result.get_meltpool_length()) * 1e6

                # success, stop fallback loop
                break

            except tc_python.exceptions.CalculationException:
                print(f"[trial] Thermo-Calc failed at candidate {ind}, trying next-best...")
                continue  # try next-best

        if x_next is None or d_next is None:
            print(f"[trial] Top 3 candidates failed, retrying without incrementing iteration.")
            continue  # do NOT increment success_it, retry

        # Cleanup immediately
        del result, am_calculator, heat_source
        gc.collect()
        time.sleep(0.05)

    # no more tc hehe
    d_next_t = torch.tensor([[d_next]], dtype=dtype, device=device)
    w_next_t = torch.tensor([[w_next]], dtype=dtype, device=device)
    l_next_t = torch.tensor([[l_next]], dtype=dtype, device=device)

    X = torch.cat([X, x_next], dim=0)
    Yd = torch.cat([Yd, d_next_t], dim=0)
    Yw = torch.cat([Yw, w_next_t], dim=0)
    Yl = torch.cat([Yl, l_next_t], dim=0)

    # retrain models
    for task, Y in zip(["Depth", "Width", "Length"], [Yd, Yw, Yl]):
        gp = FitGP(X, Y)
        gp = fitGP(gp, restarts=restarts)
        gp_models[task] = gp

    success_it += 1  # increment only on success

    print(f"[iter {success_it}/{niter}] DONE: "
          f"P={float(x_next[0,0]):.2f}, V={float(x_next[0,1]):.2f}, "
          f"W={w_next:.2f}, D={d_next:.2f}, L={l_next:.2f}")

    # plot every 5 successful iterations
    if (success_it) % 5 == 0 or success_it == niter:
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
            iteration=success_it
        )
        plt.close('all')  # erase figure to free memory

preds, rmses = evaluate_gp_models(gp_models, X[:len(Y_targets["Depth"])], Y_targets, n_samples=10, label="After Active Learning")
X_probe = torch.tensor([
    [60, 1200],  # point 1: [power, speed]
    [80, 2800.0],  # point 2
    [100, 400]   # point 3
], dtype=dtype, device=device)

# Evaluate
probe_preds, _ = evaluate_gp_models(gp_models, X_probe, label="Probe Points")