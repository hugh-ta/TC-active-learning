import torch
import gpytorch
import math
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import griddata
from torch import nn
from tqdm import tqdm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal


# ==============================================================================
# 0. SETTINGS & CONFIGURATION
# ==============================================================================
# --- Data Settings ---
FILEPATH = 'results_progress.csv'
TEST_SIZE = 0.1
RANDOM_STATE = 42

# --- Bayesian Optimization Settings ---
N_ITERATIONS = 40
N_INITIAL_CUSTOM = 15     # Initial random points for the custom kernel's BO
N_INITIAL_MATERN = 7      # Initial random points for the baseline kernel's BO
BO_NUM_RESTARTS = 20      # Number of restarts for the acquisition function optimizer

# --- GP Model Training Settings ---
TRAINING_EPOCHS = 200 # More epochs needed for DKL
FINAL_TRAINING_EPOCHS = 300 

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Set default tensor type to float64 (double precision) for numerical stability
torch.set_default_dtype(torch.double)

# Suppress GPyTorch warnings
warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)

# ==============================================================================
# 1. Load and Process Real Data
# ==============================================================================
def load_and_process_data(filepath, test_size, random_state):
    df = pd.read_csv(filepath)
    X = df[['Power', 'Speed']].values
    y = df['Depth'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_x = torch.from_numpy(X_train_scaled)
    train_y = torch.from_numpy(y_train)
    test_x = torch.from_numpy(X_test_scaled)
    test_y_true = torch.from_numpy(y_test)
    
    return train_x, train_y, test_x, test_y_true, scaler, X, y

TRAIN_X, TRAIN_Y, TEST_X, TEST_Y_TRUE, SCALER, FULL_X, FULL_Y = load_and_process_data(FILEPATH, TEST_SIZE, RANDOM_STATE)

# ==============================================================================
# 2. Deep Kernel Learning (DKL) Model Definition
# ==============================================================================
# This is the neural network part of our model.
# It learns to transform the input data into a better feature space for the GP.
class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', nn.Linear(2, 100))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(100, 50))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(50, 2))

# This is the GP part of our model. It operates on the features from the NN.
class DeepGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(DeepGPModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        # The GP kernel is simpler now, as the NN does the heavy lifting
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))

    def forward(self, x):
        # Pass the input through the neural network feature extractor
        projected_x = self.feature_extractor(x)
        
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

# Standard GP model for baseline comparison
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
# ==============================================================================
# 3. Bayesian Optimization Objective Functions
# ==============================================================================
def evaluate_deep_kernel(parameterization):
    """Objective function for the Deep Kernel Learning model."""
    # The BO now tunes the NN learning rate and the GP kernel's hyperparameters
    lr, weight, ls_1, ls_2, noise = parameterization
    
    feature_extractor = FeatureExtractor()
    likelihood = GaussianLikelihood()
    model = DeepGPModel(TRAIN_X, TRAIN_Y, likelihood, feature_extractor)

    # Set GP hyperparameters from the optimizer
    model.covar_module.outputscale = weight
    model.covar_module.base_kernel.lengthscale = torch.tensor([ls_1, ls_2])
    model.likelihood.noise = noise

    # Train the whole DKL model (NN weights and GP params)
    model.train()
    likelihood.train()
    # Use Adam optimizer for the NN and GP parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Adding a progress bar for the internal training loop
    for _ in tqdm(range(TRAINING_EPOCHS), desc="  Training DKL instance", leave=False): 
        optimizer.zero_grad()
        output = model(TRAIN_X)
        loss = -mll(output, TRAIN_Y)
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(TEST_X))
        pred_y = observed_pred.mean
    
    rmse = torch.sqrt(torch.mean((pred_y - TEST_Y_TRUE)**2))
    return -rmse.unsqueeze(-1)

def evaluate_matern52_kernel(parameterization):
    """Objective function for the standard ANISOTROPIC Matern 5/2 kernel."""
    matern_weight, matern_ls_p, matern_ls_v, noise = parameterization
    
    matern_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    matern_kernel.outputscale = matern_weight
    matern_kernel.base_kernel.lengthscale = torch.tensor([matern_ls_p, matern_ls_v])
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(TRAIN_X, TRAIN_Y, likelihood, matern_kernel)
    model.likelihood.noise = noise

    model.train()
    likelihood.train()
    training_params = [{'params': model.mean_module.parameters()}, {'params': model.likelihood.parameters()}]
    optimizer = torch.optim.Adam(training_params, lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(TRAINING_EPOCHS): 
        optimizer.zero_grad()
        output = model(TRAIN_X)
        loss = -mll(output, TRAIN_Y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(TEST_X))
        pred_y = observed_pred.mean
        
    rmse = torch.sqrt(torch.mean((pred_y - TEST_Y_TRUE)**2))
    return -rmse.unsqueeze(-1)

# ==============================================================================
# 4. Main Dual Bayesian Optimization Loop
# ==============================================================================
print("--- Starting Initial Evaluations for Deep Kernel ---")
# --- DKL Kernel BO Setup ---
bounds_custom = torch.tensor([
    # lr, weight, ls_1, ls_2, noise
    [0.001, 0.1, 0.05, 0.05, 0.01], # Lower
    [0.1,   2.0, 1.0,  1.0,  0.2]   # Upper
])
train_x_custom = torch.rand(N_INITIAL_CUSTOM, bounds_custom.shape[1]) * (bounds_custom[1] - bounds_custom[0]) + bounds_custom[0]
train_y_custom_list = [evaluate_deep_kernel(x) for x in tqdm(train_x_custom, desc="Initial DKL Evals")]
train_y_custom = torch.cat(train_y_custom_list)


print("\n--- Starting Initial Evaluations for Baseline Matern Kernel ---")
# --- Standard Matern 5/2 Kernel BO Setup ---
bounds_matern = torch.tensor([[0.1, 0.05, 0.05, 0.01], [2.0, 1.0, 1.0, 0.2]]) # w, ls_p, ls_v, noise
train_x_matern = torch.rand(N_INITIAL_MATERN, bounds_matern.shape[1]) * (bounds_matern[1] - bounds_matern[0]) + bounds_matern[0]
train_y_matern_list = [evaluate_matern52_kernel(x) for x in tqdm(train_x_matern, desc="Initial Matern Evals")]
train_y_matern = torch.cat(train_y_matern_list)


# --- Tracking Progress ---
best_rmse_custom = [-train_y_custom.max().item()]
best_rmse_matern = [-train_y_matern.max().item()]

print(f"\nInitial best RMSE (Deep Kernel): {best_rmse_custom[-1]:.4f}")
print(f"Initial best RMSE (Standard Matern): {best_rmse_matern[-1]:.4f}")

# --- Main BO loop ---
main_bo_loop = tqdm(range(N_ITERATIONS), desc="Main Bayesian Optimization Race")
for i in main_bo_loop:
    main_bo_loop.set_description(f"Main BO Race [Iteration {i+1}/{N_ITERATIONS}]")
    
    # --- 1. Optimize Deep Kernel ---
    train_x_custom_norm = normalize(train_x_custom, bounds_custom)
    gp_custom = SingleTaskGP(train_x_custom_norm, train_y_custom.unsqueeze(-1))
    mll_custom = ExactMarginalLogLikelihood(gp_custom.likelihood, gp_custom)
    fit_gpytorch_mll(mll_custom)
    
    acq_custom = UpperConfidenceBound(model=gp_custom, beta=2.5)
    candidate_custom_norm, _ = optimize_acqf(
        acq_function=acq_custom,
        bounds=torch.tensor([[0.0] * bounds_custom.shape[1], [1.0] * bounds_custom.shape[1]]),
        q=1, num_restarts=BO_NUM_RESTARTS, raw_samples=512,
    )
    new_x_custom = unnormalize(candidate_custom_norm, bounds_custom).detach()
    new_y_custom = evaluate_deep_kernel(new_x_custom.squeeze(0))
    
    train_x_custom = torch.cat([train_x_custom, new_x_custom])
    train_y_custom = torch.cat([train_y_custom, new_y_custom])
    best_rmse_custom.append(-train_y_custom.max().item())
    

    # --- 2. Optimize Standard Matern Kernel ---
    train_x_matern_norm = normalize(train_x_matern, bounds_matern)
    gp_matern = SingleTaskGP(train_x_matern_norm, train_y_matern.unsqueeze(-1))
    mll_matern = ExactMarginalLogLikelihood(gp_matern.likelihood, gp_matern)
    fit_gpytorch_mll(mll_matern)
    
    acq_matern = UpperConfidenceBound(model=gp_matern, beta=2.5)
    candidate_matern_norm, _ = optimize_acqf(
        acq_function=acq_matern,
        bounds=torch.tensor([[0.0] * bounds_matern.shape[1], [1.0] * bounds_matern.shape[1]]),
        q=1, num_restarts=BO_NUM_RESTARTS, raw_samples=512,
    )
    new_x_matern = unnormalize(candidate_matern_norm, bounds_matern).detach()
    new_y_matern = evaluate_matern52_kernel(new_x_matern.squeeze(0))
    
    train_x_matern = torch.cat([train_x_matern, new_x_matern])
    train_y_matern = torch.cat([train_y_matern, new_y_matern])
    best_rmse_matern.append(-train_y_matern.max().item())

    # Update progress bar with the latest scores
    main_bo_loop.set_postfix({
        'Best DKL RMSE': f"{best_rmse_custom[-1]:.4f}", 
        'Best Matern RMSE': f"{best_rmse_matern[-1]:.4f}"
    })


# ==============================================================================
# 5. Final Model Training and Prediction for Plotting
# ==============================================================================
print("\n=============================================")
print("Dual Bayesian Optimization Finished")
print("=============================================")

# --- Find best hyperparameters found during BO ---
best_params_custom = train_x_custom[train_y_custom.argmax()]
best_params_matern = train_x_matern[train_y_matern.argmax()]

# --- Create final models using the best hyperparameters ---
final_feature_extractor = FeatureExtractor()
final_likelihood_custom = GaussianLikelihood()
final_model_custom = DeepGPModel(TRAIN_X, TRAIN_Y, final_likelihood_custom, final_feature_extractor)
                                  
final_model_matern = ExactGPModel(TRAIN_X, TRAIN_Y, gpytorch.likelihoods.GaussianLikelihood(),
                                  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)))

# Set the hyperparameters on the final models
final_model_custom.covar_module.outputscale = best_params_custom[1]
final_model_custom.covar_module.base_kernel.lengthscale = torch.tensor([best_params_custom[2], best_params_custom[3]])
final_model_custom.likelihood.noise = best_params_custom[4]

final_model_matern.covar_module.outputscale = best_params_matern[0]
final_model_matern.covar_module.base_kernel.lengthscale = torch.tensor([best_params_matern[1], best_params_matern[2]])
final_model_matern.likelihood.noise = best_params_matern[3]

# Train the final models
print("\n--- Training final Deep Kernel model ---")
final_model_custom.train()
final_likelihood_custom.train()
optimizer_custom = torch.optim.Adam(final_model_custom.parameters(), lr=best_params_custom[0])
mll_custom = ExactMarginalLogLikelihood(final_likelihood_custom, final_model_custom)
for _ in tqdm(range(FINAL_TRAINING_EPOCHS), desc="Final DKL Training"):
    optimizer_custom.zero_grad()
    output = final_model_custom(TRAIN_X)
    loss = -mll_custom(output, TRAIN_Y)
    loss.backward()
    optimizer_custom.step()
final_model_custom.eval()
final_likelihood_custom.eval()

print("\n--- Training final Matern model ---")
final_model_matern.train()
optimizer_matern = torch.optim.Adam([{'params': final_model_matern.mean_module.parameters()}, {'params': final_model_matern.likelihood.parameters()}], lr=0.1)
mll_matern = gpytorch.mlls.ExactMarginalLogLikelihood(final_model_matern.likelihood, final_model_matern)
for _ in tqdm(range(FINAL_TRAINING_EPOCHS), desc="Final Matern Training"):
    optimizer_matern.zero_grad()
    output = final_model_matern(TRAIN_X)
    loss = -mll_matern(output, TRAIN_Y)
    loss.backward()
    optimizer_matern.step()
final_model_matern.eval()

# --- Create grid for plotting that covers the full data range ---
power_min, speed_min = FULL_X.min(axis=0)
power_max, speed_max = FULL_X.max(axis=0)
power_grid = np.linspace(power_min, power_max, 50)
speed_grid = np.linspace(speed_min, speed_max, 50)
PP, VV = np.meshgrid(power_grid, speed_grid)

# --- Interpolate Ground Truth data from the full dataset for a complete surface ---
Z_depth_grid = griddata(FULL_X, FULL_Y, (PP, VV), method='cubic')

# --- Make predictions on the same grid for a direct comparison ---
grid_np = np.vstack([PP.ravel(), VV.ravel()]).T
grid_scaled = SCALER.transform(grid_np)
grid_torch = torch.from_numpy(grid_scaled)
with torch.no_grad():
    Y_pred_tuned_mean = final_model_custom(grid_torch).mean.reshape(PP.shape)
    Y_pred_baseline_mean = final_model_matern(grid_torch).mean.reshape(PP.shape)

# ==============================================================================
# 6. Plotting Results
# ==============================================================================

# --- Plot 1: Convergence Race ---
plt.figure(figsize=(12, 7))
plt.plot(range(len(best_rmse_custom)), best_rmse_custom, marker='o', linestyle='-', label='Optimized Deep Kernel')
plt.plot(range(len(best_rmse_matern)), best_rmse_matern, marker='s', linestyle='--', color='r', label='Optimized Standard Matern 5/2')
plt.xlabel("BO Iteration")
plt.ylabel("Best RMSE Found")
plt.title("Convergence Race (RMSE): Deep Kernel vs. Standard Matern 5/2")
plt.grid(True, which='both', linestyle='--')
plt.xticks(range(len(best_rmse_custom)))
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: 3D Surface Comparison ---
fig = plt.figure(figsize=(20, 7))

# Determine shared Z-axis limits for a fair comparison
z_min = min(np.nanmin(Z_depth_grid), np.nanmin(Y_pred_tuned_mean.numpy()), np.nanmin(Y_pred_baseline_mean.numpy()))
z_max = max(np.nanmax(Z_depth_grid), np.nanmax(Y_pred_tuned_mean.numpy()), np.nanmax(Y_pred_baseline_mean.numpy()))

# Ground Truth
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(PP, VV, Z_depth_grid, cmap='viridis', alpha=0.9, linewidth=0)
ax1.set_title('Ground Truth (Interpolated Full Dataset)')
ax1.set_xlabel('Power'); ax1.set_ylabel('Speed'); ax1.set_zlabel('Depth')
ax1.view_init(elev=20, azim=-60)
ax1.set_zlim(z_min, z_max)

# GP Prediction (Tuned Kernel)
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(PP, VV, Y_pred_tuned_mean.numpy(), cmap='plasma', alpha=0.9)
custom_kernel_title = 'GP Prediction (Deep Kernel Learning)'
ax2.set_title(custom_kernel_title)
ax2.set_xlabel('Power'); ax2.set_ylabel('Speed'); ax2.set_zlabel('Depth')
ax2.view_init(elev=20, azim=-60)
ax2.set_zlim(z_min, z_max)

# GP Prediction (Baseline Matern)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(PP, VV, Y_pred_baseline_mean.numpy(), cmap='coolwarm', alpha=0.9)
ax3.set_title('GP Prediction (Anisotropic Baseline Matern)')
ax3.set_xlabel('Power'); ax3.set_ylabel('Speed'); ax3.set_zlabel('Depth')
ax3.view_init(elev=20, azim=-60)
ax3.set_zlim(z_min, z_max)

plt.tight_layout()
plt.show()