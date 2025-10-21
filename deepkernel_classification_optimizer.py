import torch
import gpytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings

FILEPATH = 'results_progress.csv'
INPUT_FEATURES = ['Power', 'Speed']
TARGET_VARIABLES = ['Good', 'Keyhole', 'Balling', 'Lack_of_Fusion']
TEST_SIZE = 0.1
RANDOM_STATE = 42
TRAIN_EPOCHS = 200

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.set_default_dtype(torch.float)
warnings.filterwarnings("ignore")

# ------------------------- Defect Classification -------------------------
def classify_defect(width, depth, length):
    keyholing_boundary = 1.9
    lof_boundary = 1.5
    balling_boundary = 0.23
    thickness = 10.0
    jitter = 1e-9
    if depth <= 0:
        return "Lack_of_Fusion"
    if (width / (depth + jitter)) < keyholing_boundary:
        return "Keyhole"
    if (depth / thickness) < lof_boundary:
        return "Lack_of_Fusion"
    if (width / (length + jitter)) < balling_boundary:
        return "Balling"
    return "Good"

# ------------------------- Models -------------------------
class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('linear1', nn.Linear(2, 100))
        self.add_module('relu1', nn.ReLU())
        self.add_module('linear2', nn.Linear(100, 50))
        self.add_module('relu2', nn.ReLU())
        self.add_module('linear3', nn.Linear(50, 2))

class DKLGPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, feature_extractor, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BaseGPClassifier(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ------------------------- Data -------------------------
def load_data(filepath, target_class):
    df = pd.read_csv(filepath)
    labels = df.apply(lambda row: classify_defect(row['Width'], row['Depth'], row['Length']), axis=1)
    df['target'] = (labels == target_class).astype(float)
    X = df[INPUT_FEATURES].values
    y = df['target'].values

    print(f"\nDiagnose: {target_class} - locations for class=1:\n", df[df['target'] == 1][INPUT_FEATURES])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        torch.tensor(X_train_scaled, dtype=torch.float),
        torch.tensor(y_train, dtype=torch.float),
        torch.tensor(X_test_scaled, dtype=torch.float),
        torch.tensor(y_test, dtype=torch.float),
        scaler,
        X,
        y
    )

# ------------------------- Main Loop for Each Target -------------------------
for target_output in TARGET_VARIABLES:
    print("\n" + "=" * 60)
    print(f"Processing: {target_output}")
    print("=" * 60)

    TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, SCALER, FULL_X, FULL_Y = load_data(FILEPATH, target_output)
    FULL_X_SCALED = torch.tensor(SCALER.transform(FULL_X), dtype=torch.float)

    # ---- Deep Kernel GP ----
    feature_extractor = FeatureExtractor()
    inducing_pts_dkl = TRAIN_X[:100]
    dkl_gp = DKLGPClassifier(feature_extractor, inducing_pts_dkl)
    dkl_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    dkl_gp.train()
    dkl_likelihood.train()
    optimizer_dkl = torch.optim.Adam([
        {'params': dkl_gp.parameters()},
        {'params': dkl_likelihood.parameters()}
    ], lr=0.01)
    mll_dkl = gpytorch.mlls.VariationalELBO(dkl_likelihood, dkl_gp, num_data=TRAIN_X.size(0))

    for epoch in tqdm(range(TRAIN_EPOCHS), desc=f"DKL ({target_output}) Epoch"):
        optimizer_dkl.zero_grad()
        output = dkl_gp(TRAIN_X)
        loss = -mll_dkl(output, TRAIN_Y)
        loss.backward()
        optimizer_dkl.step()

    dkl_gp.eval()
    dkl_likelihood.eval()

    # ---- Base GP (Matern) ----
    inducing_pts_base = TRAIN_X[:100]
    base_gp = BaseGPClassifier(inducing_pts_base)
    base_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    base_gp.train()
    base_likelihood.train()
    optimizer_base = torch.optim.Adam([
        {'params': base_gp.parameters()},
        {'params': base_likelihood.parameters()}
    ], lr=0.01)
    mll_base = gpytorch.mlls.VariationalELBO(base_likelihood, base_gp, num_data=TRAIN_X.size(0))

    for epoch in tqdm(range(TRAIN_EPOCHS), desc=f"Matern ({target_output}) Epoch"):
        optimizer_base.zero_grad()
        output_base = base_gp(TRAIN_X)
        loss_base = -mll_base(output_base, TRAIN_Y)
        loss_base.backward()
        optimizer_base.step()

    base_gp.eval()
    base_likelihood.eval()

    # ------------------------- Meshgrid (FIXED) -------------------------
    x1_grid = np.linspace(FULL_X[:, 0].min(), FULL_X[:, 0].max(), 100)  # Power (Y)
    x2_grid = np.linspace(FULL_X[:, 1].min(), FULL_X[:, 1].max(), 100)  # Speed (X)

    # FIX: swap order so rows = Power, cols = Speed
    X2, X1 = np.meshgrid(x2_grid, x1_grid)
    grid_points = np.vstack([X1.ravel(), X2.ravel()]).T  # [Power, Speed]
    grid_scaled = torch.tensor(SCALER.transform(grid_points), dtype=torch.float)

    # ------------------------- Predictions -------------------------
    with torch.no_grad():
        Z_dkl = dkl_likelihood(dkl_gp(grid_scaled)).mean.numpy().reshape(X1.shape)
        Z_matern = base_likelihood(base_gp(grid_scaled)).mean.numpy().reshape(X1.shape)

    # ------------------------- Plots -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
    ax1, ax2, ax3 = axes

    true_points = FULL_X[FULL_Y == 1]
    false_points = FULL_X[FULL_Y == 0]

    # Scatter: x = Speed, y = Power
    ax1.scatter(false_points[:, 1], false_points[:, 0], c='gray', alpha=0.3, label='Class = 0')
    ax1.scatter(true_points[:, 1], true_points[:, 0], c='green', marker='x', label='Class = 1')
    ax1.set_title(f"Ground Truth for '{target_output}'")
    ax1.set_xlabel('Speed')
    ax1.set_ylabel('Power')
    ax1.legend()

    # DKL plot
    im2 = ax2.imshow(
        Z_dkl,
        origin='lower',
        extent=[x2_grid.min(), x2_grid.max(), x1_grid.min(), x1_grid.max()],
        aspect='auto', cmap='viridis', vmin=0, vmax=1
    )
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("DKL Predicted Probability")
    ax2.set_xlabel('Speed')
    ax2.set_ylabel('Power')

    # Matern plot
    im3 = ax3.imshow(
        Z_matern,
        origin='lower',
        extent=[x2_grid.min(), x2_grid.max(), x1_grid.min(), x1_grid.max()],
        aspect='auto', cmap='viridis', vmin=0, vmax=1
    )
    fig.colorbar(im3, ax=ax3)
    ax3.set_title("Matern Predicted Probability")
    ax3.set_xlabel('Speed')
    ax3.set_ylabel('Power')

    fig.suptitle(f"GP Classifier Maps for Target: '{target_output}'", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"gpcompare_{target_output}.png")
    plt.show()

    # Save models
    torch.save(dkl_gp.state_dict(), f"dkl_gp_for_{target_output}.pth")
    torch.save(base_gp.state_dict(), f"base_gp_for_{target_output}.pth")

print("\n" + "=" * 60)
print("All targets processed and plotted.")
print("=" * 60)
