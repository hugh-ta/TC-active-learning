# deepkernel.py
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
import json

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
# 0. GLOBAL SETTINGS & CONFIGURATION
# ==============================================================================
# --- Data Settings ---
FILEPATH = 'results_progress.csv'
INPUT_FEATURES = ['Power', 'Speed']
TARGET_VARIABLES = ['Depth', 'Width', 'Length'] # A list of the different outputs we will model sequentially

# --- Universal Settings ---
TEST_SIZE = 0.1
RANDOM_STATE = 42
N_ITERATIONS = 40
N_INITIAL_CUSTOM = 15
N_INITIAL_MATERN = 7
BO_NUM_RESTARTS = 20
TRAINING_EPOCHS = 200
FINAL_TRAINING_EPOCHS = 300

# --- Setup ---
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_default_dtype(torch.double)
warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)

# ==============================================================================
# 1. REUSABLE MODEL & DATA CLASSES
# ==============================================================================
def load_and_process_data(filepath, feature_cols, target_col):
    """Loads data for a specific set of inputs and one target output."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"FATAL: The file '{filepath}' was not found.")
        print("Creating a dummy file to allow the script to run for demonstration.")
        dummy_data = {
            'Power': np.random.rand(100) * 500 + 200,
            'Speed': np.random.rand(100) * 1500 + 500,
            'Depth': np.random.rand(100) * 0.5 + 0.1,
            'Width': np.random.rand(100) * 0.2 + 0.05,
            'Length': np.random.rand(100) * 5 + 1,
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(filepath, index=False)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_x = torch.from_numpy(X_train_scaled)
    train_y = torch.from_numpy(y_train)
    test_x = torch.from_numpy(X_test_scaled)
    test_y_true = torch.from_numpy(y_test)

    return train_x, train_y, test_x, test_y_true, scaler, X, y

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
        super(DeepGPModel, self).__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

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
# 2. MAIN PROCESSING LOOP
# ==============================================================================
# This loop will run the entire analysis for each target variable.
for target_output in TARGET_VARIABLES:

    # MODIFIED: Removed emojis from this print block
    print("\n" + "#"*80)
    print(f"## STARTING FULL ANALYSIS FOR TARGET: '{target_output}' ##")
    print("#"*80 + "\n")

    # --- Step 1: Load data for the current target ---
    TRAIN_X, TRAIN_Y, TEST_X, TEST_Y_TRUE, SCALER, FULL_X, FULL_Y = load_and_process_data(
        FILEPATH, INPUT_FEATURES, target_output
    )

    # --- Step 2: Define Objective functions that use the current data ---
    def evaluate_deep_kernel(parameterization):
        lr, weight, ls_1, ls_2, noise = parameterization
        model = DeepGPModel(TRAIN_X, TRAIN_Y, GaussianLikelihood(), FeatureExtractor())
        model.covar_module.outputscale = weight
        model.covar_module.base_kernel.lengthscale = torch.tensor([ls_1, ls_2])
        model.likelihood.noise = noise
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        for _ in range(TRAINING_EPOCHS):
            optimizer.zero_grad(); loss = -mll(model(TRAIN_X), TRAIN_Y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): pred_y = model(TEST_X).mean
        return -torch.sqrt(torch.mean((pred_y - TEST_Y_TRUE)**2)).unsqueeze(-1)

    def evaluate_matern52_kernel(parameterization):
        matern_weight, matern_ls_p, matern_ls_v, noise = parameterization
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))
        kernel.outputscale = matern_weight
        kernel.base_kernel.lengthscale = torch.tensor([matern_ls_p, matern_ls_v])
        model = ExactGPModel(TRAIN_X, TRAIN_Y, GaussianLikelihood(), kernel)
        model.likelihood.noise = noise
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        for _ in range(TRAINING_EPOCHS):
            optimizer.zero_grad(); loss = -mll(model(TRAIN_X), TRAIN_Y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): pred_y = model(TEST_X).mean
        return -torch.sqrt(torch.mean((pred_y - TEST_Y_TRUE)**2)).unsqueeze(-1)

    # --- Step 3: Run the Bayesian Optimization Race ---
    print(f"--- Starting BO Race for '{target_output}' ---")
    bounds_custom = torch.tensor([[0.001, 0.1, 0.05, 0.05, 0.01], [0.1, 2.0, 1.0, 1.0, 0.2]])
    bounds_matern = torch.tensor([[0.1, 0.05, 0.05, 0.01], [2.0, 1.0, 1.0, 0.2]])

    train_x_custom = torch.rand(N_INITIAL_CUSTOM, 5) * (bounds_custom[1] - bounds_custom[0]) + bounds_custom[0]
    train_y_custom = torch.cat([evaluate_deep_kernel(p) for p in tqdm(train_x_custom, desc="Initial DKL Evals")])

    train_x_matern = torch.rand(N_INITIAL_MATERN, 4) * (bounds_matern[1] - bounds_matern[0]) + bounds_matern[0]
    train_y_matern = torch.cat([evaluate_matern52_kernel(p) for p in tqdm(train_x_matern, desc="Initial Matern Evals")])

    best_rmse_custom = [-train_y_custom.max().item()]
    best_rmse_matern = [-train_y_matern.max().item()]

    main_bo_loop = tqdm(range(N_ITERATIONS), desc=f"BO Race for '{target_output}'")
    for i in main_bo_loop:
        # DKL Step
        gp_custom = SingleTaskGP(normalize(train_x_custom, bounds_custom), train_y_custom.unsqueeze(-1))
        mll_c = ExactMarginalLogLikelihood(gp_custom.likelihood, gp_custom); fit_gpytorch_mll(mll_c)
        cand_c, _ = optimize_acqf(UpperConfidenceBound(gp_custom, 2.5), torch.tensor([[0.0]*5, [1.0]*5]), 1, BO_NUM_RESTARTS, 512)
        new_x_c = unnormalize(cand_c, bounds_custom); new_y_c = evaluate_deep_kernel(new_x_c.squeeze(0))
        train_x_custom = torch.cat([train_x_custom, new_x_c]); train_y_custom = torch.cat([train_y_custom, new_y_c])
        best_rmse_custom.append(-train_y_custom.max().item())

        # Matern Step
        gp_matern = SingleTaskGP(normalize(train_x_matern, bounds_matern), train_y_matern.unsqueeze(-1))
        mll_m = ExactMarginalLogLikelihood(gp_matern.likelihood, gp_matern); fit_gpytorch_mll(mll_m)
        cand_m, _ = optimize_acqf(UpperConfidenceBound(gp_matern, 2.5), torch.tensor([[0.0]*4, [1.0]*4]), 1, BO_NUM_RESTARTS, 512)
        new_x_m = unnormalize(cand_m, bounds_matern); new_y_m = evaluate_matern52_kernel(new_x_m.squeeze(0))
        train_x_matern = torch.cat([train_x_matern, new_x_m]); train_y_matern = torch.cat([train_y_matern, new_y_m])
        best_rmse_matern.append(-train_y_matern.max().item())

        main_bo_loop.set_postfix({'DKL RMSE': f"{best_rmse_custom[-1]:.4f}", 'Matern RMSE': f"{best_rmse_matern[-1]:.4f}"})

    # --- Step 4: Final Model Training ---
    print(f"\n--- Finalizing models for '{target_output}' ---")
    best_params_custom = train_x_custom[train_y_custom.argmax()]
    best_params_matern = train_x_matern[train_y_matern.argmax()]

    # Final DKL
    final_model_custom = DeepGPModel(TRAIN_X, TRAIN_Y, GaussianLikelihood(), FeatureExtractor())
    final_model_custom.covar_module.outputscale = best_params_custom[1]
    final_model_custom.covar_module.base_kernel.lengthscale = torch.tensor([best_params_custom[2], best_params_custom[3]])
    final_model_custom.likelihood.noise = best_params_custom[4]
    opt_c = torch.optim.Adam(final_model_custom.parameters(), lr=best_params_custom[0])
    mll_c = ExactMarginalLogLikelihood(final_model_custom.likelihood, final_model_custom)
    final_model_custom.train()
    for _ in tqdm(range(FINAL_TRAINING_EPOCHS), desc="Final DKL Training"):
        opt_c.zero_grad(); loss = -mll_c(final_model_custom(TRAIN_X), TRAIN_Y); loss.backward(); opt_c.step()

    # Final Matern
    kernel_m = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))
    final_model_matern = ExactGPModel(TRAIN_X, TRAIN_Y, GaussianLikelihood(), kernel_m)
    final_model_matern.covar_module.outputscale = best_params_matern[0]
    final_model_matern.covar_module.base_kernel.lengthscale = torch.tensor([best_params_matern[1], best_params_matern[2]])
    final_model_matern.likelihood.noise = best_params_matern[3]
    opt_m = torch.optim.Adam(final_model_matern.parameters(), lr=0.1)
    mll_m = ExactMarginalLogLikelihood(final_model_matern.likelihood, final_model_matern)
    final_model_matern.train()
    for _ in tqdm(range(FINAL_TRAINING_EPOCHS), desc="Final Matern Training"):
        opt_m.zero_grad(); loss = -mll_m(final_model_matern(TRAIN_X), TRAIN_Y); loss.backward(); opt_m.step()

    # --- Step 5: Save the results for this target ---
    dkl_filename = f"dkl_kernel_for_{target_output}"
    matern_filename = f"matern_kernel_for_{target_output}"

    torch.save(final_model_custom.state_dict(), f"{dkl_filename}.pth")
    torch.save(final_model_matern.state_dict(), f"{matern_filename}.pth")
    print(f"\nSaved model states for '{target_output}'")

    # --- Step 6: Plot the results for this target ---
    plt.figure(figsize=(12, 7))
    plt.plot(range(len(best_rmse_custom)), best_rmse_custom, marker='o', label='Optimized Deep Kernel')
    plt.plot(range(len(best_rmse_matern)), best_rmse_matern, marker='s', linestyle='--', label='Optimized Matern 5/2')
    plt.xlabel("BO Iteration"); plt.ylabel("Best RMSE Found")
    plt.title(f"Convergence Race for Target: '{target_output}'")
    plt.grid(True, which='both', linestyle='--'); plt.xticks(range(len(best_rmse_custom))); plt.legend(); plt.tight_layout()
    plt.savefig(f"convergence_race_{target_output}.png")
    plt.show()

    fig = plt.figure(figsize=(20, 7))
    x1_grid = np.linspace(FULL_X[:, 0].min(), FULL_X[:, 0].max(), 50)
    x2_grid = np.linspace(FULL_X[:, 1].min(), FULL_X[:, 1].max(), 50)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z_truth = griddata(FULL_X, FULL_Y, (X1, X2), method='cubic')
    grid_torch = torch.from_numpy(SCALER.transform(np.vstack([X1.ravel(), X2.ravel()]).T))

    final_model_custom.eval(); final_model_matern.eval()
    with torch.no_grad():
        Z_dkl = final_model_custom(grid_torch).mean.reshape(X1.shape)
        Z_matern = final_model_matern(grid_torch).mean.reshape(X1.shape)

    z_min = np.nanmin([Z_truth, Z_dkl.numpy(), Z_matern.numpy()])
    z_max = np.nanmax([Z_truth, Z_dkl.numpy(), Z_matern.numpy()])

    ax1 = fig.add_subplot(1, 3, 1, projection='3d'); ax1.plot_surface(X1, X2, Z_truth, cmap='viridis'); ax1.set_title('Ground Truth'); ax1.set_xlabel(INPUT_FEATURES[0]); ax1.set_ylabel(INPUT_FEATURES[1]); ax1.set_zlabel(target_output); ax1.set_zlim(z_min, z_max)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d'); ax2.plot_surface(X1, X2, Z_dkl.numpy(), cmap='plasma'); ax2.set_title('DKL Prediction'); ax2.set_xlabel(INPUT_FEATURES[0]); ax2.set_ylabel(INPUT_FEATURES[1]); ax2.set_zlabel(target_output); ax2.set_zlim(z_min, z_max)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d'); ax3.plot_surface(X1, X2, Z_matern.numpy(), cmap='coolwarm'); ax3.set_title('Matern Prediction'); ax3.set_xlabel(INPUT_FEATURES[0]); ax3.set_ylabel(INPUT_FEATURES[1]); ax3.set_zlabel(target_output); ax3.set_zlim(z_min, z_max)

    plt.suptitle(f"Surface Plots for Target: '{target_output}'", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"surface_plots_{target_output}.png")
    plt.show()

print("\n" + "#"*80)
print("## ALL TARGET VARIABLES PROCESSED SUCCESSFULLY! ##")
print("#"*80)