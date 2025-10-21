# constraintlessTCPythonLoopKernel.py
# A TCPython Loop with Custom Kernels without constraints for iterative loops
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
from torch import nn
import gc, time

# GPyTorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

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

# training  settings
ntrain = 10
niter = 25 #how many AL loops after  training are allowed
restarts = 10
ngrid= 100
nmc = 64
edensity_low = 0
edensity_high =10000000
SEED = 8
np.random.seed(SEED)
torch.manual_seed(SEED)

# file settings
trainfile = "results_progress.csv"

# import data
data = pd.read_csv(trainfile)

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
    try:
        model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=dkl_kernel, likelihood=likelihood,
                             input_transform=None, outcome_transform=None)
    except Exception:
        # fallback in case of API differences
        model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=dkl_kernel, likelihood=likelihood)

    # ensure model on device
    model = model.to(device)

    return model
def evaluate_gp_classifiers(gp_models, X, Y_targets, n_samples=10, label=""):

    print(f"\n=== Evaluation {label} ===")
    class_names = list(Y_targets.keys())
    predictions = {}
    with torch.no_grad():
        for name, model in gp_models.items():
            model.eval()
            posterior = model.posterior(X)
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