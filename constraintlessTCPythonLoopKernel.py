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
# --- New Imports for DKL Kernel ---
import os
import pickle
from torch import nn
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
# --- End New Imports ---

# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

# BoTorch
from botorch.models import SingleTaskGP
# No longer using BoTorch transforms, as we use a pre-trained scaler
# from botorch.models.transforms import Normalize, Standardize 
from botorch.fit import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples

#scipy
from scipy.interpolate import griddata
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


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
dtype = torch.double # Main script uses double precision
device = torch.device("cpu")

# training  settings
ntrain = 10
niter = 25 #how many AL loops after  training are allowed
restarts = 10 # This is for fit_gpytorch_mll
ngrid= 100
nmc = 64
edensity_low = 0
edensity_high =10000000
SEED = 8
np.random.seed(SEED)
torch.manual_seed(SEED)

# file settings
trainfile = "results_progress.csv"
SCALER_FILE = "scaler.pkl" # From script 2
MODEL_PATH_TEMPLATE = "dkl_kernel_for_{}.pth" # From script 2

# import data
data = pd.read_csv(trainfile)

# --- Create and Fit the Scaler ---
# We re-create the scaler by fitting it to the full dataset,
# which is what the DKL models were originally trained on.
print(f"Fitting new scaler from {trainfile}...")
from sklearn.preprocessing import MinMaxScaler

power_all = data["Power"].values.reshape(-1, 1)
speed_all = data["Speed"].values.reshape(-1, 1)
X_all = np.column_stack([power_all, speed_all])

scaler = MinMaxScaler()
scaler.fit(X_all)
print("Scaler successfully fitted.")
# --- End Scaler Fit ---


depth  = data["Depth"].values.reshape(-1, 1)
width  = data["Width"].values.reshape(-1, 1)
length = data["Length"].values.reshape(-1, 1)
power  = data["Power"].values.reshape(-1, 1)
speed  = data["Speed"].values.reshape(-1, 1)

# total number of data points
n_total = len(depth)

#inital defect classification function
def classify_defect(width, depth, e, length=None, ed_low=edensity_low, ed_high=edensity_high):
    keyholing = 1.5
    lof = 1.9
    balling = 0.23
    thickness = POWDER_THICKNESS
    jitter = 1e-9

    if e is not None:
        if e < edensity_low:
            return 3  # lack of fusion
        elif e > edensity_high:
            return 1  # keyholing
    if (depth) == 0:
        return 3  # lack of fusion
    elif (width/(depth + jitter)) < keyholing:
        return 1  # keyholing
    elif ((depth)/(thickness)) < lof:
        return 3  #lack of fusion
    if USE_BALLING and (length is not None):
        if (width/(length + jitter)) < balling:
            return 2  # balling
    return 0  # good

# classify all data points
defects = np.zeros((n_total, 1))
for i in range(n_total):
    if USE_EDENSITY:
        defects[i] = classify_defect(width[i], depth[i], data["Energy density"].values[i], length[i])
    else:
        defects[i] = classify_defect(width[i], depth[i], None, length[i])
# pick ntrain indices randomly from the actual data
initial_idx = np.array([
    1,   # Lack of Fusion (Low Power)
    5,   # Lack of Fusion (Low Power)
    25,  # Likely on a boundary
    45,  # Likely Good/LoF boundary
    65,  # Likely Good
    75,  # Keyhole
    85,  # Keyhole
    90,  # Keyhole
    95,  # Keyhole (High Power, High Speed)
    50   # Likely Good
])

# training inputs
X = torch.tensor(np.column_stack([power[initial_idx], speed[initial_idx]]), dtype=dtype, device=device)

# training outputs
d = torch.tensor(depth[initial_idx], dtype=dtype, device=device)
w = torch.tensor(width[initial_idx], dtype=dtype, device=device)
l = torch.tensor(length[initial_idx], dtype=dtype, device=device)

initial_labels = []
for i in range(ntrain):
    # Pass the individual physical results for point 'i' to the function
    label_code = classify_defect(
        width=w[i],
        depth=d[i],
        e=None, # Assuming no energy density for now
        length=l[i]
    )
    initial_labels.append(label_code)

labels_array = np.array(initial_labels)
Y_good = torch.tensor((labels_array == 0).astype(float), dtype=dtype, device=device).reshape(-1, 1)
Y_keyhole = torch.tensor((labels_array == 1).astype(float), dtype=dtype, device=device).reshape(-1, 1)
Y_balling = torch.tensor((labels_array == 2).astype(float), dtype=dtype, device=device).reshape(-1, 1)
Y_lof = torch.tensor((labels_array == 3).astype(float), dtype=dtype, device=device).reshape(-1, 1)


# --- DKL KERNEL DEFINITIONS (From User) ---
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
        # x1 is double, and the feature_extractor will also be cast to double.
        projected_x1 = self.feature_extractor(x1)
        projected_x2 = self.feature_extractor(x2)
        return self.base_kernel.forward(projected_x1, projected_x2, diag=diag, **params)
# --- END DKL KERNEL DEFINITIONS ---


#GP Helpers
def fitGP(xtrain, ytrain, scaler, kernel_target_name, restarts=restarts):
    """
    Fits a SingleTaskGP using the pre-trained DKL kernel.
    Only the Likelihood and Mean parameters are optimized.
    """
    
    # 1. Instantiate components
    # All components will be in the script's default dtype (double)
    feature_extractor = FeatureExtractor().to(device=device, dtype=dtype) 
    base_kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)).to(device=device, dtype=dtype)
    
    # 2. Load the state dict
    model_path = MODEL_PATH_TEMPLATE.format(kernel_target_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained DKL model not found: {model_path}")
    
    # Load weights (which are likely float32)
    full_state_dict = torch.load(model_path, map_location=device)

    # 3. Load parameters into each component
    # --- THIS IS THE FIX ---
    # The weights are in a *nested* dictionary.
    
    # Get the nested OrderedDict of float weights
    fe_state_float = full_state_dict['feature_extractor_state']
    # Create a new dict, casting each tensor to double
    fe_state_double = {k: v.to(dtype=dtype) for k, v in fe_state_float.items()}

    # Get the nested OrderedDict of float weights
    cov_state_float = full_state_dict['covar_module_state']
    # Create a new dict, casting each tensor to double
    cov_state_double = {k: v.to(dtype=dtype) for k, v in cov_state_float.items()}

    if len(fe_state_double) > 0:
        feature_extractor.load_state_dict(fe_state_double, strict=True)
    if len(cov_state_double) > 0:
        base_kernel.load_state_dict(cov_state_double, strict=True)
    
    print(f"Loaded DKL components from {model_path} for {kernel_target_name}")

    # 4. Freeze parameters
    feature_extractor.eval()
    base_kernel.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    for param in base_kernel.parameters():
        param.requires_grad = False

    # 5. Create the custom DKL kernel
    dkl_kernel = PreTrainedDKLKernel(feature_extractor, base_kernel)

    # 6. Scale inputs using the pre-trained scaler
    xtrain_scaled = torch.tensor(
        scaler.transform(xtrain.cpu().numpy()), dtype=dtype, device=device
    )

    # 7. Create the final SingleTaskGP model
    # This model creates its own GaussianLikelihood and ConstantMean
    gp = SingleTaskGP(
        train_X=xtrain_scaled,
        train_Y=ytrain,
        covar_module=dkl_kernel,
        # No input/outcome transforms, scaling is done manually
    ).to(dtype=dtype, device=device)
    
    # --- ADD THIS BLOCK TO FIX THE NOISE ---
    # Set the noise to a small, fixed value.
    # This prevents the model from "smudging" the 0/1 data.
    gp.likelihood.noise = 1e-4 
    # Freeze the noise parameter so fit_gpytorch_mll can't optimize it.
    gp.likelihood.noise_covar.raw_noise.requires_grad = False
    # --- END OF FIX ---
    
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(dtype=dtype, device=device)
    
    # 8. fit the model
    # This will ONLY optimize parameters with requires_grad=True
    # (i.e., mean_module.constant)
    fit_gpytorch_mll(mll, options={"maxiter": 1000, "disp": False})
    
    return gp

def evaluate_gp_classifiers(gp_models, X, Y_targets, scaler, n_samples=10, label=""):
    """
    Evaluates the DKL-GP models.
    Requires the scaler to transform data before prediction.
    """
    print(f"\n=== Evaluation {label} ===")
    class_names = list(Y_targets.keys())
    predictions = {}
    
    # Scale the input data
    X_scaled = torch.tensor(
        scaler.transform(X.cpu().numpy()), dtype=dtype, device=device
    )

    with torch.no_grad():
        for name, model in gp_models.items():
            model.eval()
            posterior = model.posterior(X_scaled)
            predictions[name] = posterior.mean.cpu().numpy().ravel()

    print("Sample Predictions (Probabilities):")
    for i in range(min(n_samples, len(X))):
        true_label = "Unknown"
        for name in class_names:
            if Y_targets[name][i].item() == 1.0:
                true_label = name
                break
        
        prob_strs = [f"{name}: {predictions[name][i]:.2f}" for name in class_names]
        print(f"  Point {i+1:>2} | True Label: {true_label:<15} | Preds: [ {' | '.join(prob_strs)} ]")

# fit initial GPs
print("--- Fitting initial models using pre-trained DKL kernels ---")
gp_good = fitGP(X, Y_good, scaler, "Good")
gp_keyhole = fitGP(X, Y_keyhole, scaler, "Keyhole")
gp_balling = fitGP(X, Y_balling, scaler, "Balling")
gp_lof = fitGP(X, Y_lof, scaler, "Lack_of_Fusion")
# --- END MODIFICATION ---

gp_models = {
    "Good": gp_good,
    "Keyhole": gp_keyhole,
    "Balling": gp_balling,
    "Lack of Fusion": gp_lof
}

Y_targets = {
    "Good": Y_good,
    "Keyhole": Y_keyhole,
    "Balling": Y_balling,
    "Lack of Fusion": Y_lof,
}

evaluate_gp_classifiers(
    gp_models,
    X,
    Y_targets,
    scaler, # Pass the scaler
    n_samples=ntrain,
    label="Initial Training Data"
)

# def acquisition function
def entropy_sigma(X_grid, gp_models, X_train, scaler, top_k=5, alpha_dist=0.1):
    """
    Calculates acquisition score using a robust Monte Carlo (MC) estimation of
    the posterior entropy, combined with an Upper Confidence Bound (UCB)
    for a principled exploration/exploitation trade-off.

    The score is J = E[H(y|x)] + alpha * Std[H(y|x)],
    where H is the entropy of the class probabilities.

    NOTE: The 'X_train' and 'alpha_dist' parameters are no longer used
    by this more advanced method, but are kept in the signature
    for drop-in compatibility.
    """
    print("Calculating acquisition function (using Monte Carlo UCB Entropy)...")

    # This is the trade-off parameter for the UCB (replaces alpha_dist)
    # 1.0 is a good default, balances exploitation and exploration.
    UCB_alpha = 1.0

    # 1. Ensure models are in evaluation mode
    for model in gp_models.values():
        model.eval()

    # 2. Scale inputs
    X_grid_scaled = torch.tensor(
        scaler.transform(X_grid.cpu().numpy()), dtype=dtype, device=device
    )
    n_grid = X_grid_scaled.shape[0]

    # 3. Get posterior samples from all 4 models
    class_names = ["Good", "Keyhole", "Balling", "Lack of Fusion"]
    samples_list = []
    
    # We use the global `nmc` variable defined at the top of your script
    with torch.no_grad():
        for name in class_names:
            # Use rsample for reparameterization, shape [nmc, n_grid]
            posterior = gp_models[name].posterior(X_grid_scaled)
            samples = posterior.rsample(torch.Size([nmc])).squeeze(-1)
            
            # Clamp samples to be valid probabilities (0-1)
            # This is important since the regression GP can predict < 0 or > 1
            samples = samples.clamp(1e-6, 1.0)
            samples_list.append(samples)

    # 4. Stack samples into one tensor: [nmc, n_grid, 4]
    prob_samples = torch.stack(samples_list, dim=2)

    # 5. Normalize samples to create valid probability distributions
    # Sum across the 4 classes (dim=2), shape [nmc, n_grid]
    prob_sum = prob_samples.sum(dim=2, keepdim=True).clamp(min=1e-9)
    prob_samples_normalized = prob_samples / prob_sum # shape [nmc, n_grid, 4]

    # 6. Calculate entropy for EACH sample: H = -sum(p * log2(p))
    log_probs = torch.log2(prob_samples_normalized.clamp(min=1e-9))
    # Sum across 4 classes (dim=2), shape [nmc, n_grid]
    H_samples = -(prob_samples_normalized * log_probs).sum(dim=2)

    # 7. Calculate the mean and std of the entropy across the samples
    # Aggregate across the nmc samples (dim=0)
    H_mean = H_samples.mean(dim=0) # Exploitation term
    H_std = H_samples.std(dim=0)   # Exploration term

    # 8. Final acquisition score: J = Mean(H) + alpha * Std(H)
    J_tensor = H_mean + UCB_alpha * H_std
    J = J_tensor.cpu().numpy()

    # 9. Find top candidates
    # The distance penalty is no longer needed.
    top_indices = np.argsort(-J)[:top_k]

    print("Acquisition function calculation complete.")
    return J, top_indices
def plot_printability_map(gp_models, X_grid, power_grid, velo_grid, scaler, J=None, X_train=None, iteration=None):
    """
    Generates a printability map by predicting the most likely class from the GP
    classifiers and plots the acquisition function alongside it.
    Requires the scaler.
    """
    print(f"Generating plot for iteration {iteration}...")
    
    # --- Scale grid for prediction ---
    X_grid_scaled = torch.tensor(
        scaler.transform(X_grid.cpu().numpy()), dtype=dtype, device=device
    )

    # --- 1. Predict the most likely class for each grid point ---
    with torch.no_grad():
        predictions = {name: model.posterior(X_grid_scaled).mean for name, model in gp_models.items()}
    
    class_names = ["Good", "Keyhole", "Balling", "Lack of Fusion"]
    prob_tensor = torch.cat([predictions[name] for name in class_names], dim=1)
    pred_indices = torch.argmax(prob_tensor, dim=1).cpu().numpy()
    
    index_to_label = {0: "Good", 1: "Keyhole", 2: "Balling", 3: "Lack of Fusion"}
    labels_grid = np.array([index_to_label[i] for i in pred_indices]).reshape(len(power_grid), len(velo_grid))

    # --- 2. Plotting ---
    fig, (ax_defect, ax_acq) = plt.subplots(1, 2, figsize=(15, 6))

    label_to_num = {"Good": 0, "Keyhole": 1, "Balling": 2, "Lack of Fusion": 3}
    defect_numeric_grid = np.vectorize(label_to_num.get)(labels_grid)

    colors = {
        "Good": np.array([255, 255, 255]) / 255, # Explicitly define white
        "Keyhole": np.array([224, 123, 123]) / 255,
        "Lack of Fusion": np.array([123, 191, 200]) / 255,
        "Balling": np.array([40, 156, 142]) / 255,
    }
    
    rgb_grid = np.zeros((*labels_grid.shape, 3))
    alpha = 0.7 

    for label, num in label_to_num.items():
        if label in colors:
            mask = (defect_numeric_grid == num)
            if label == "Good":
                rgb_grid[mask] = colors[label]
            else:
                rgb_grid[mask] = alpha * colors[label] + (1 - alpha) * 1.0

    # Plot the defect map
    # Use original (unscaled) grid limits
    ax_defect.imshow(
        rgb_grid,
        extent=[velo_grid.min(), velo_grid.max(), power_grid.min(), power_grid.max()],
        origin="lower", aspect="auto", zorder=1
    )
    ax_defect.set_xlabel("Scan Velocity (mm/s)")
    ax_defect.set_ylabel("Laser Power (W)")
    ax_defect.set_title(f"Predicted Printability Map (Iter {iteration})" if iteration is not None else "Predicted Printability Map")

    # X_train is the original (unscaled) data
    if X_train is not None:
        X_train_np = X_train.cpu().numpy()
        ax_defect.scatter(X_train_np[:-1, 1], X_train_np[:-1, 0], c='black', marker='o', s=30, label='Previous Points', zorder=5)
        ax_defect.scatter(X_train_np[-1, 1], X_train_np[-1, 0], c='red', marker='*', s=150, edgecolor='black', label='Newest Point', zorder=6)

    legend_elements = [
        Patch(facecolor=colors["Keyhole"], label="Keyhole"),
        Patch(facecolor=colors["Lack of Fusion"], label="Lack of Fusion"),
        Patch(facecolor=colors["Balling"], label="Balling"),
        Patch(facecolor="white", edgecolor="black", label="Good"),
    ]
    ax_defect.legend(handles=legend_elements, loc="best")

    if J is not None:
        grid_J = J.reshape(len(power_grid), len(velo_grid))
        im = ax_acq.imshow(grid_J, origin="lower", extent=[velo_grid.min(), velo_grid.max(), power_grid.min(), power_grid.max()], cmap='viridis', aspect='auto')
        ax_acq.set_title(f"Acquisition Function (Iter {iteration})")
        ax_acq.set_xlabel("Scan Velocity (mm/s)")
        fig.colorbar(im, ax=ax_acq, label="Acquisition Score")
        max_idx = np.unravel_index(np.argmax(J), grid_J.shape)
        # Plot candidate on original (unscaled) grid
        ax_acq.scatter(velo_grid[max_idx[1]], powergrid[max_idx[0]], c='red', marker='*', s=150, edgecolor='white', label='Next Candidate')
        ax_acq.legend()
    
    plt.tight_layout()
    plt.show()

# active learning loop
# initialize TC python
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
powermin, powermax = data["Power"].min(), data["Power"].max()
velomin, velomax = data["Speed"].min(), data["Speed"].max()
powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP, VV = np.meshgrid(powergrid, velogrid, indexing="ij")
Xgrid = torch.tensor(np.column_stack([PP.ravel(), VV.ravel()]), dtype=dtype, device=device)
# Note: We do NOT scale Xgrid here. Scaling is handled by the functions that use it.


success_it = 0 # counts successful iterations
plotiter = 5 # plot every 5 iterations

while success_it < niter:
    # calculate acquisition function to find the next best point
    J, top_candidates = entropy_sigma(
        X_grid=Xgrid, # Pass unscaled grid
        gp_models=gp_models,
        X_train=X,    # Pass unscaled training data
        scaler=scaler,
        top_k=5,
        alpha_dist=0.1
    )

    # try top 3 candidates in case of simulation failure
    sorted_idx = top_candidates[:3]
    d_next = w_next = l_next = None
    x_next = None

    for ind in sorted_idx:
        try:
            # get the next point to sample (from the unscaled grid)
            x_next = Xgrid[ind:ind+1]

            # create a new calculator for the simulation
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

                # set heat source parameters for the chosen point
                heat_source = HeatSource.from_library(HEAT_SOURCE_NAME)
                heat_source.set_power(float(x_next[0, 0].item()))
                heat_source.set_scanning_speed(float(x_next[0, 1].item()) / 1e3) # TC uses m/s
                am_calculator.with_heat_source(heat_source)

                # run the simulation
                result: SteadyStateResult = am_calculator.calculate()

                # get melt pool dimensions
                d_next = float(result.get_meltpool_depth()) * 1e6 # convert to um
                w_next = float(result.get_meltpool_width()) * 1e6 # convert to um
                l_next = float(result.get_meltpool_length()) * 1e6 # convert to um
                
                # if simulation is successful, break the fallback loop
                break

        except tc_python.exceptions.CalculationException:
            print(f"[trial] Thermo-Calc failed at candidate {ind}, trying next-best...")
            continue # try next-best candidate

    # if all top 3 candidates failed, restart the loop without updating
    if x_next is None or d_next is None:
        print(f"[trial] Top 3 candidates failed, retrying without incrementing iteration.")
        continue # do NOT increment success_it, retry

    # cleanup tc-python objects
    del result, am_calculator, heat_source
    gc.collect()

    # 1. Get the final label for the new point using our "black box" function
    new_label_code = classify_defect(width=w_next, depth=d_next, e=None, length=l_next)
    
    # 2. Create the four binary (0/1) labels for the new point
    y_good_next = torch.tensor([1.0 if new_label_code == 0 else 0.0], dtype=dtype, device=device).reshape(1, 1)
    y_keyhole_next = torch.tensor([1.0 if new_label_code == 1 else 0.0], dtype=dtype, device=device).reshape(1, 1)
    y_balling_next = torch.tensor([1.0 if new_label_code == 2 else 0.0], dtype=dtype, device=device).reshape(1, 1)
    y_lof_next = torch.tensor([1.0 if new_label_code == 3 else 0.0], dtype=dtype, device=device).reshape(1, 1)

    # 3. Append the new data to our training tensors (unscaled)
    X = torch.cat([X, x_next], dim=0)
    Y_good = torch.cat([Y_good, y_good_next], dim=0)
    Y_keyhole = torch.cat([Y_keyhole, y_keyhole_next], dim=0)
    Y_balling = torch.cat([Y_balling, y_balling_next], dim=0)
    Y_lof = torch.cat([Y_lof, y_lof_next], dim=0)
    
    # Update the Y_targets dictionary
    Y_targets = {"Good": Y_good, "Keyhole": Y_keyhole, "Balling": Y_balling, "Lack of Fusion": Y_lof}

    # 4. Retrain all four GP classifiers from scratch with the new point
    # The fitGP function will handle re-scaling the new, larger X tensor
    print("--- Re-fitting models with new data point (using pre-trained DKL kernels) ---")
    gp_models["Good"] = fitGP(X, Y_good, scaler, "Good")
    gp_models["Keyhole"] = fitGP(X, Y_keyhole, scaler, "Keyhole")
    gp_models["Balling"] = fitGP(X, Y_balling, scaler, "Balling")
    gp_models["Lack of Fusion"] = fitGP(X, Y_lof, scaler, "Lack_of_Fusion")
    
    # increment success counter
    success_it += 1

    print(f"[iter {success_it}/{niter}] DONE: "
          f"P={float(x_next[0,0]):.2f}, V={float(x_next[0,1]):.2f}, "
          f"W={w_next:.2f}, D={d_next:.2f}, L={l_next:.2f}, Label={new_label_code}")

    # plot the results periodically
    if (success_it) % plotiter == 0 or success_it == niter:
        plot_printability_map(
            gp_models=gp_models,
            X_grid=Xgrid,     # Pass unscaled grid
            power_grid=powergrid,
            velo_grid=velogrid,
            scaler=scaler,
            J=J,
            X_train=X,        # Pass unscaled training data
            iteration=success_it
        )
        plt.close('all') # free up memory

# --- Final evaluation after the loop is finished ---
print("\n--- Final Evaluation on All Sampled Points ---")
evaluate_gp_classifiers(
    gp_models, 
    X, # Pass unscaled data
    Y_targets,
    scaler,
    n_samples=len(X), 
    label="After Active Learning"
)