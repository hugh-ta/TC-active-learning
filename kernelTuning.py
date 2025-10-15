import torch
import gpytorch
import math
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.interpolate import griddata

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 0. SETTINGS & CONFIGURATION
# ==============================================================================
# --- Data Settings ---
FILEPATH = 'results_progress.csv'
TEST_SIZE = 0.1
RANDOM_STATE = 42

# --- Bayesian Optimization Settings ---
N_ITERATIONS = 40
N_INITIAL_CUSTOM = 15     # More initial points for the larger search space
N_INITIAL_MATERN = 7      # More initial points for the larger search space
BO_NUM_RESTARTS = 20      # Number of restarts for the acquisition function optimizer

# --- GP Model Training Settings ---
TRAINING_EPOCHS = 150       # Epochs for models inside the BO loop
FINAL_TRAINING_EPOCHS = 200 # Epochs for the final models used for plotting

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
    """
    Loads data from the specified CSV, selects relevant columns, normalizes
    the features, splits into training/testing sets, and returns PyTorch tensors
    as well as the scaler and the original full dataset for later use.
    """
    df = pd.read_csv(filepath)
    X = df[['Power', 'Speed']].values
    y = df['Depth'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Normalize the features (X) to be between 0 and 1
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
# 2. Gaussian Process Model Definition
# ==============================================================================
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
def evaluate_custom_kernel(parameterization):
    """Objective function for the custom ANISOTROPIC composite kernel."""
    (rbf_weight, m52_weight, m15_weight, 
     rbf_ls_p, rbf_ls_v, 
     m52_ls_p, m52_ls_v, 
     m15_ls_p, m15_ls_v, 
     noise) = parameterization
    
    # Anisotropic kernels have a separate lengthscale for each input dimension (ard_num_dims=2)
    rbf_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
    m52_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))
    m15_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2))
    
    combined_kernel = rbf_kernel + m52_kernel + m15_kernel

    # Set hyperparameters from the optimizer
    rbf_kernel.outputscale = rbf_weight
    m52_kernel.outputscale = m52_weight
    m15_kernel.outputscale = m15_weight
    
    # Set the two lengthscales for each kernel
    rbf_kernel.base_kernel.lengthscale = torch.tensor([rbf_ls_p, rbf_ls_v])
    m52_kernel.base_kernel.lengthscale = torch.tensor([m52_ls_p, m52_ls_v])
    m15_kernel.base_kernel.lengthscale = torch.tensor([m15_ls_p, m15_ls_v])
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(TRAIN_X, TRAIN_Y, likelihood, combined_kernel)
    model.likelihood.noise = noise

    # Train the non-kernel parameters (mean, noise)
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

    # Evaluate the model on the test set
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(TEST_X))
        pred_y = observed_pred.mean
    
    rmse = torch.sqrt(torch.mean((pred_y - TEST_Y_TRUE)**2))
    return -rmse.unsqueeze(-1) # Minimize RMSE (BoTorch maximizes)

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
    return -rmse.unsqueeze(-1) # Minimize RMSE (BoTorch maximizes)


# ==============================================================================
# 4. Main Dual Bayesian Optimization Loop
# ==============================================================================
# --- Custom Kernel BO Setup ---
bounds_custom = torch.tensor([
    # rbf_w, m52_w, m15_w, rbf_ls_p, rbf_ls_v, m52_ls_p, m52_ls_v, m15_ls_p, m15_ls_v, noise
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01], # Lower
    [2.0,  2.0,  2.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.2]   # Upper
])
train_x_custom = torch.rand(N_INITIAL_CUSTOM, bounds_custom.shape[1]) * (bounds_custom[1] - bounds_custom[0]) + bounds_custom[0]
train_y_custom = torch.cat([evaluate_custom_kernel(x) for x in train_x_custom])

# --- Standard Matern 5/2 Kernel BO Setup ---
bounds_matern = torch.tensor([[0.1, 0.05, 0.05, 0.01], [2.0, 1.0, 1.0, 0.2]]) # w, ls_p, ls_v, noise
train_x_matern = torch.rand(N_INITIAL_MATERN, bounds_matern.shape[1]) * (bounds_matern[1] - bounds_matern[0]) + bounds_matern[0]
train_y_matern = torch.cat([evaluate_matern52_kernel(x) for x in train_x_matern])

# --- Tracking Progress ---
best_rmse_custom = [-train_y_custom.max().item()]
best_rmse_matern = [-train_y_matern.max().item()]

print(f"Initial best RMSE (Custom Kernel): {best_rmse_custom[-1]:.4f}")
print(f"Initial best RMSE (Standard Matern): {best_rmse_matern[-1]:.4f}")

# --- Main BO loop ---
for i in range(N_ITERATIONS):
    print(f"\n--- Iteration {i+1}/{N_ITERATIONS} ---")
    
    # --- 1. Optimize Custom Kernel ---
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
    new_y_custom = evaluate_custom_kernel(new_x_custom.squeeze(0))
    
    train_x_custom = torch.cat([train_x_custom, new_x_custom])
    train_y_custom = torch.cat([train_y_custom, new_y_custom])
    best_rmse_custom.append(-train_y_custom.max().item())
    
    print(f"Candidate RMSE (Custom Kernel): {-new_y_custom.item():.4f}")
    print(f"Best RMSE so far (Custom Kernel): {best_rmse_custom[-1]:.4f}")

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

    print(f"Candidate RMSE (Standard Matern): {-new_y_matern.item():.4f}")
    print(f"Best RMSE so far (Standard Matern): {best_rmse_matern[-1]:.4f}")

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
custom_kernel_composition = (gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)) + 
                             gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)) + 
                             gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2)))
final_model_custom = ExactGPModel(TRAIN_X, TRAIN_Y, gpytorch.likelihoods.GaussianLikelihood(), custom_kernel_composition)
                                  
final_model_matern = ExactGPModel(TRAIN_X, TRAIN_Y, gpytorch.likelihoods.GaussianLikelihood(),
                                  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)))

# Set the hyperparameters on the final models
final_model_custom.covar_module.kernels[0].outputscale = best_params_custom[0]
final_model_custom.covar_module.kernels[1].outputscale = best_params_custom[1]
final_model_custom.covar_module.kernels[2].outputscale = best_params_custom[2]
final_model_custom.covar_module.kernels[0].base_kernel.lengthscale = torch.tensor([best_params_custom[3], best_params_custom[4]])
final_model_custom.covar_module.kernels[1].base_kernel.lengthscale = torch.tensor([best_params_custom[5], best_params_custom[6]])
final_model_custom.covar_module.kernels[2].base_kernel.lengthscale = torch.tensor([best_params_custom[7], best_params_custom[8]])
final_model_custom.likelihood.noise = best_params_custom[9]

final_model_matern.covar_module.outputscale = best_params_matern[0]
final_model_matern.covar_module.base_kernel.lengthscale = torch.tensor([best_params_matern[1], best_params_matern[2]])
final_model_matern.likelihood.noise = best_params_matern[3]

# Train the final models (mean and likelihood)
for model in [final_model_custom, final_model_matern]:
    model.train()
    optimizer = torch.optim.Adam([{'params': model.mean_module.parameters()}, {'params': model.likelihood.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    for _ in range(FINAL_TRAINING_EPOCHS):
        optimizer.zero_grad()
        output = model(TRAIN_X)
        loss = -mll(output, TRAIN_Y)
        loss.backward()
        optimizer.step()
    model.eval()

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
plt.plot(range(len(best_rmse_custom)), best_rmse_custom, marker='o', linestyle='-', label='Optimized Custom Kernel')
plt.plot(range(len(best_rmse_matern)), best_rmse_matern, marker='s', linestyle='--', color='r', label='Optimized Standard Matern 5/2')
plt.xlabel("BO Iteration")
plt.ylabel("Best RMSE Found")
plt.title("Convergence Race (RMSE): Custom Kernel vs. Standard Matern 5/2")
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
custom_kernel_title = 'GP Prediction (Anisotropic RBF + Matern5/2 + Matern1/5)'
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