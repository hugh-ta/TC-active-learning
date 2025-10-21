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
# --- LIKELIHOOD AND MLL CHANGE ---
import gpytorch  # <-- ADD THIS LINE
from gpytorch.kernels import MaternKernel, ScaleKernel, Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
# --- END LIKELIHOOD CHANGE ---

# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood # No longer used, but kept for context

# BoTorch
from botorch.models import SingleTaskGP # No longer used
from botorch.fit import fit_gpytorch_mll # No longer used
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
# --- DKL (ApproximateGP) MUST USE FLOAT ---
dtype = torch.float
device = torch.device("cpu")
torch.set_default_dtype(dtype)
# --- END DTYPE CHANGE ---

# training  settings
ntrain = 20
niter = 25 #how many AL loops after  training are allowed
DKL_FIT_EPOCHS = 500 # Epochs to fit the variational parameters
ngrid= 100
nmc = 128
edensity_low = 0
edensity_high =10000000
SEED = 12
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
# initial_idx = np.array([
#     1,   # Lack of Fusion (Low Power)
#     5,   # Lack of Fusion (Low Power)
#     25,  # Likely on a boundary
#     45,  # Likely Good/LoF boundary
#     65,  # Likely Good
#     75,  # Keyhole
#     85,  # Keyhole
#     90,  # Keyhole
#     95,  # Keyhole (High Power, High Speed)
#     50   # Likely Good
# ])
initial_idx = np.random.choice(n_total, ntrain, replace=False)
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

# --- NEW MODEL: APPROXIMATE GP FOR CLASSIFICATION ---
class DKLGPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, feature_extractor, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
# --- END NEW MODEL ---


#GP Helpers
def fitGP(xtrain, ytrain, scaler, kernel_target_name, restarts=None):
    """
    Fits an ApproximateGP (DKL) classifier using the pre-trained kernel.
    This fine-tunes the variational parameters AND the GP kernel's
    hyperparameters (lengthscale, outputscale) on the new data.
    """
    
    # 1. Instantiate components
    feature_extractor = FeatureExtractor().to(device=device, dtype=dtype) 
    # This is the "base kernel" component of the DKL
    base_kernel = ScaleKernel(MaternKernel(nu=2.5)).to(device=device, dtype=dtype)
    
    # 2. Load the state dict
    model_path = MODEL_PATH_TEMPLATE.format(kernel_target_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained DKL model not found: {model_path}")
    
    full_state_dict = torch.load(model_path, map_location=device)

    # 3. Load parameters into each component
    fe_state = full_state_dict['feature_extractor_state']
    cov_state = full_state_dict['covar_module_state']

    feature_extractor.load_state_dict(fe_state, strict=True)
    base_kernel.load_state_dict(cov_state, strict=True)
    
    print(f"Loaded DKL components from {model_path} for {kernel_target_name}")

    # 4. Freeze parameters for the pre-trained parts
    
    # --- FREEZE THE "DEEP" PART ---
    # We keep the feature transform static
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
        
    # --- UNFREEZE THE "KERNEL" PART ---
    # We MUST allow the kernel's lengthscale and outputscale
    # to be fine-tuned to the new training data.
    base_kernel.train()
    for param in base_kernel.parameters():
        param.requires_grad = True  # Make sure this is True
    # --- END FIX ---

    # 5. Scale inputs
    xtrain_scaled = torch.tensor(
        scaler.transform(xtrain.cpu().numpy()), dtype=dtype, device=device
    )
    
    # 6. Create the DKL Classification Model
    
    # --- (RECOMMENDED FIX 2) ---
    # Use ALL training data as inducing points for better accuracy at low N
    # n_inducing = min(100, xtrain_scaled.size(0))
    # inducing_pts = xtrain_scaled[:n_inducing]
    inducing_pts = xtrain_scaled.clone()
    # --- END RECOMMENDED FIX ---
    
    # We need to manually put the DKL kernel *inside* the DKLGPClassifier
    model = DKLGPClassifier(feature_extractor, inducing_pts).to(dtype=dtype, device=device)
    
    # Here we assign the loaded, UN-FROZEN kernel to the model
    model.covar_module = base_kernel 
    
    # USE THE CORRECT LIKELIHOOD
    likelihood = BernoulliLikelihood().to(dtype=dtype, device=device)

    # 7. Fit the model by optimizing variational parameters
    model.train()
    likelihood.train()
    
    # This optimizer will now train the variational parameters
    # AND the parameters of model.covar_module (the base_kernel)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # (Optional) Use differential learning rates
    # optimizer = torch.optim.Adam([
    #    {'params': model.mean_module.parameters()},
    #    {'params': model.variational_strategy.parameters()},
    #    {'params': likelihood.parameters()},
    #    # Give the kernel parameters a smaller LR
    #    {'params': model.covar_module.parameters(), 'lr': 0.001}, 
    # ], lr=0.01) # Default LR for other params
    
    # Squeeze ytrain from [N, 1] to [N] for BernoulliLikelihood
    ytrain_squeezed = ytrain.squeeze(-1)
    mll = VariationalELBO(likelihood, model, num_data=xtrain_scaled.size(0))

    for epoch in range(DKL_FIT_EPOCHS):
        optimizer.zero_grad()
        output = model(xtrain_scaled)
        loss = -mll(output, ytrain_squeezed)
        loss.backward()
        optimizer.step()

    return model, likelihood # RETURN BOTH

def evaluate_gp_classifiers(gp_models, likelihoods, X, Y_targets, scaler, n_samples=10, label=""):
    """
    Evaluates the DKL-GP models.
    Requires the scaler and likelihoods.
    """
    print(f"\n=== Evaluation {label} ===")
    class_names = list(Y_targets.keys())
    predictions = {}
    
    # Scale the input data
    X_scaled = torch.tensor(
        scaler.transform(X.cpu().numpy()), dtype=dtype, device=device
    )

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for name, model in gp_models.items():
            model.eval()
            likelihoods[name].eval()
            # Get probability from likelihood
            posterior = model(X_scaled)
            predictions[name] = likelihoods[name](posterior).mean.cpu().numpy().ravel()

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
gp_good, like_good = fitGP(X, Y_good, scaler, "Good")
gp_keyhole, like_keyhole = fitGP(X, Y_keyhole, scaler, "Keyhole")
gp_balling, like_balling = fitGP(X, Y_balling, scaler, "Balling")
gp_lof, like_lof = fitGP(X, Y_lof, scaler, "Lack_of_Fusion")

gp_models = {
    "Good": gp_good,
    "Keyhole": gp_keyhole,
    "Balling": gp_balling,
    "Lack of Fusion": gp_lof
}
# --- STORE LIKELIHOODS ---
likelihoods = {
    "Good": like_good,
    "Keyhole": like_keyhole,
    "Balling": like_balling,
    "Lack of Fusion": like_lof
}

Y_targets = {
    "Good": Y_good,
    "Keyhole": Y_keyhole,
    "Balling": Y_balling,
    "Lack of Fusion": Y_lof,
}

evaluate_gp_classifiers(
    gp_models,
    likelihoods, # Pass likelihoods
    X,
    Y_targets,
    scaler, # Pass the scaler
    n_samples=ntrain,
    label="Initial Training Data"
)

# def acquisition function
# def entropy_sigma(X_grid, gp_models, likelihoods, X_train, scaler, top_k=5, alpha_dist=0.1):
#     """
#     Implements the "Entropy-Sigma" acquisition function, inspired by the paper.
#     J = H * (1 + sigma_total)
    
#     - H: The mean posterior entropy of the 4-class distribution (Exploitation).
#          This is high at the class boundaries.
#     - sigma_total: The sum of the standard deviations of the 4 individual
#                    model probability predictions (Exploration/Uncertainty).
#                    This is high where the models are "shaky".
    
#     This function is greedier and focuses on boundaries with high model uncertainty.
#     """
#     print("Calculating acquisition function (using Entropy-Sigma H * (1+σ) method)...")

#     # 1. Ensure models are in evaluation mode
#     for model in gp_models.values():
#         model.eval()
#     for like in likelihoods.values():
#         like.eval()

#     # 2. Scale inputs
#     X_grid_scaled = torch.tensor(
#         scaler.transform(X_grid.cpu().numpy()), dtype=dtype, device=device
#     )
#     n_grid = X_grid_scaled.shape[0]

#     # 3. Get posterior samples from all 4 models
#     class_names = ["Good", "Keyhole", "Balling", "Lack of Fusion"]
#     samples_list = []
    
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         for name in class_names:
#             posterior_f = gp_models[name](X_grid_scaled)
#             prob_samples_p = likelihoods[name](posterior_f).sample(torch.Size([nmc]))
#             samples_list.append(prob_samples_p.clamp(1e-6, 1.0))

#     # 4. Stack samples into one tensor: [nmc, n_grid, 4]
#     prob_samples = torch.stack(samples_list, dim=2)

#     # 5. --- CALCULATE "H" (Entropy) ---
#     # Normalize samples to create valid probability distributions
#     prob_sum = prob_samples.sum(dim=2, keepdim=True).clamp(min=1e-9)
#     prob_samples_normalized = prob_samples / prob_sum # shape [nmc, n_grid, 4]

#     # Calculate entropy for EACH sample
#     log_probs = torch.log2(prob_samples_normalized.clamp(min=1e-9))
#     H_samples = -(prob_samples_normalized * log_probs).sum(dim=2) # shape [nmc, n_grid]

#     # Get the MEAN entropy (our "H" term)
#     H_mean = H_samples.mean(dim=0) # shape [n_grid]

#     # 6. --- CALCULATE "SIGMA" (Prediction Uncertainty) ---
#     # Get the standard deviation of each model's raw probability prediction
#     std_good = samples_list[0].std(dim=0)
#     std_keyhole = samples_list[1].std(dim=0)
#     std_balling = samples_list[2].std(dim=0)
#     std_lof = samples_list[3].std(dim=0)
    
#     # Our "Sigma" term is the total predictive uncertainty
#     sigma_total = std_good + std_keyhole + std_balling + std_lof # shape [n_grid]

#     # 7. Final acquisition score: J = H * (1 + sigma)
#     # We use (1 + sigma) to prevent J=0 if sigma is very small
#     J_tensor = H_mean * (1.0 + sigma_total)
#     J = J_tensor.cpu().numpy()

#     # 8. Find top candidates
#     top_indices = np.argsort(-J)[:top_k]

#     print("Acquisition function calculation complete.")
#     return J, top_indices
# 
def entropy_sigma(X_grid, gp_models, likelihoods, X_train, scaler, top_k=5, alpha_dist=0.1):
    """
    Calculates acquisition score based on the probability of misclassification,
    which is maximized on the decision boundaries between the top two classes.
    
    J = 1.0 - (p_top1 - p_top2)
    
    This is high (near 1.0) when the top two class probabilities are very close,
    indicating a decision boundary.
    """
    print("Calculating acquisition function (using Boundary-Targeting 1-(p1-p2) method)...")

    # 1. Ensure models are in evaluation mode
    for model in gp_models.values():
        model.eval()
    for like in likelihoods.values():
        like.eval()

    # 2. Scale inputs
    X_grid_scaled = torch.tensor(
        scaler.transform(X_grid.cpu().numpy()), dtype=dtype, device=device
    )
    
    # 3. Get mean probability predictions from all models
    predictions = {}
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for name, model in gp_models.items():
            # Get latent posterior
            posterior_f = model(X_grid_scaled)
            # Get mean probability from likelihood
            prob_p = likelihoods[name](posterior_f).mean.unsqueeze(-1)
            # Clamp predictions to be safe
            predictions[name] = prob_p.clamp(1e-6, 1.0)

    # --- 4. Boundary-Finding Method (Probability of Misclassification) ---
    class_names = ["Good", "Keyhole", "Balling", "Lack of Fusion"]
    prob_tensor = torch.cat([predictions[name] for name in class_names], dim=1)
    
    # 4a. Normalize the probabilities to ensure they sum to 1
    # This makes the boundary-finding logic more robust
    prob_sum = prob_tensor.sum(dim=1, keepdim=True).clamp(min=1e-9)
    prob_tensor_normalized = prob_tensor / prob_sum
    
    # 4b. For each point, find the top two probabilities
    top_two_probs, _ = torch.topk(prob_tensor_normalized, 2, dim=1)
    
    p1 = top_two_probs[:, 0]  # The highest probability
    p2 = top_two_probs[:, 1]  # The second-highest probability
    
    # 4c. Calculate the acquisition score: 1 - (p1 - p2)
    # This value is highest (close to 1) when p1 and p2 are nearly equal.
    J_tensor = 1.0 - (p1 - p2)
    J = J_tensor.cpu().numpy()

    # --- 5. Distance Penalty (to avoid re-sampling) ---
    # This uses the original, UN-SCALED X_grid and X_train
    if X_train is not None and X_train.shape[0] > 0:
        dists = np.min(
            np.linalg.norm(
                X_grid.cpu().numpy()[:, None, :] - X_train.cpu().numpy()[None, :, :],
                axis=-1
            ), 
            axis=1
        )
        J *= (1 + alpha_dist * dists)

    # --- 6. Find top candidates ---
    top_indices = np.argsort(-J)[:top_k]

    print("Acquisition function calculation complete.")
    return J, top_indices
def plot_printability_map(gp_models, likelihoods, X_grid, power_grid, velo_grid, scaler, J=None, X_train=None, iteration=None):
    """
    Generates a printability map by predicting the most likely class from the DKL
    classifiers and plots the acquisition function alongside it.
    """
    print(f"Generating plot for iteration {iteration}...")
    
    # --- Scale grid for prediction ---
    X_grid_scaled = torch.tensor(
        scaler.transform(X_grid.cpu().numpy()), dtype=dtype, device=device
    )

    # --- 1. Predict the most likely class for each grid point ---
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = {}
        for name, model in gp_models.items():
            model.eval()
            likelihoods[name].eval()
            posterior = model(X_grid_scaled)
            # Get the mean probability
            predictions[name] = likelihoods[name](posterior).mean.unsqueeze(-1)
    
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
        likelihoods=likelihoods, # Pass likelihoods
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
    gp_models["Good"], likelihoods["Good"] = fitGP(X, Y_good, scaler, "Good")
    gp_models["Keyhole"], likelihoods["Keyhole"] = fitGP(X, Y_keyhole, scaler, "Keyhole")
    gp_models["Balling"], likelihoods["Balling"] = fitGP(X, Y_balling, scaler, "Balling")
    gp_models["Lack of Fusion"], likelihoods["Lack of Fusion"] = fitGP(X, Y_lof, scaler, "Lack_of_Fusion")
    
    # increment success counter
    success_it += 1

    print(f"[iter {success_it}/{niter}] DONE: "
          f"P={float(x_next[0,0]):.2f}, V={float(x_next[0,1]):.2f}, "
          f"W={w_next:.2f}, D={d_next:.2f}, L={l_next:.2f}, Label={new_label_code}")

    # plot the results periodically
    if (success_it) % plotiter == 0 or success_it == niter:
        plot_printability_map(
            gp_models=gp_models,
            likelihoods=likelihoods, # Pass likelihoods
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
    likelihoods, # Pass likelihoods
    X, # Pass unscaled data
    Y_targets,
    scaler,
    n_samples=len(X), 
    label="After Active Learning"
)