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
from tcpython import *

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
initial_idx = np.random.choice(n_total, size=ntrain, replace=False)

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

#GP Helpers
def fitGP(xtrain, ytrain, restarts=restarts):
    #kernel
    kernel = ScaleKernel(MaternKernel(nu=2.5), ard_num_dims=2)
    # initialize the model
    gp = SingleTaskGP(
        train_X=xtrain,
        train_Y=ytrain,
        covar_module=kernel,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=2),
    ).to(dtype=dtype, device=device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(dtype=dtype, device=device)
    # fit the model
    fit_gpytorch_mll(mll, options={"maxiter": 1000, "disp": False})
    return gp

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

# fit initial GPs
gp_good = fitGP(X, Y_good)
gp_keyhole = fitGP(X, Y_keyhole)
gp_balling = fitGP(X, Y_balling)
gp_lof = fitGP(X, Y_lof)

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
    Y_targets,  # The dictionary with Y_good, Y_keyhole, etc.
    n_samples=ntrain,
    label="Initial Training Data"
)

# def acquisition function
def entropy_sigma(X_grid, gp_models, X_train, top_k=5, alpha_dist=0.1):
    print("Calculating acquisition function...")

    for model in gp_models.values():
        model.eval()
    
    predictions = {}
    # variances = {} # << VARIANCE PART COMMENTED OUT
    with torch.no_grad():
        for name, model in gp_models.items():
            posterior = model.posterior(X_grid)
            predictions[name] = posterior.mean.clamp(1e-6, 1 - 1e-6)
            # variances[name] = posterior.variance.clamp(min=1e-9) # << VARIANCE PART COMMENTED OUT

    # --- Entropy Calculation (The Core of the Acquisition) ---
    class_names = ["Good", "Keyhole", "Balling", "Lack of Fusion"]
    prob_tensor = torch.cat([predictions[name] for name in class_names], dim=1)
    
    # Normalize probabilities to ensure they sum to 1 for a clean entropy calculation
    prob_tensor_normalized = prob_tensor / torch.sum(prob_tensor, dim=1, keepdim=True)
    entropy = -torch.sum(prob_tensor_normalized * torch.log2(prob_tensor_normalized), dim=1)

    # --- Variance Term (Deactivated) ---
    # total_variance = torch.sum(torch.cat([variances[name] for name in class_names], dim=1), dim=1) # << VARIANCE PART COMMENTED OUT

    # --- J value (Now driven by Entropy) ---
    # J_tensor = entropy * total_variance # << OLD CALCULATION COMMENTED OUT
    J_tensor = entropy # << NEW CALCULATION
    J = J_tensor.cpu().numpy()

    # --- Distance Penalty (Still useful for diversification) ---
    if X_train is not None and X_train.shape[0] > 0:
        dists = np.min(np.linalg.norm(X_grid.cpu().numpy()[:, None, :] - X_train.cpu().numpy()[None, :, :], axis=-1), axis=1)
        J *= (1 + alpha_dist * dists)

    # --- Find top candidates ---
    top_indices = np.argsort(-J)[:top_k]

    print("Acquisition function calculation complete.")
    return J, top_indices

def plot_printability_map(gp_models, X_grid, power_grid, velo_grid, J=None, X_train=None, iteration=None):
    """
    Generates a printability map by predicting the most likely class from the GP
    classifiers and plots the acquisition function alongside it.
    """
    print(f"Generating plot for iteration {iteration}...")
    
    # --- 1. Predict the most likely class for each grid point ---
    with torch.no_grad():
        predictions = {name: model.posterior(X_grid).mean for name, model in gp_models.items()}
    
    class_names = ["Good", "Keyhole", "Balling", "Lack of Fusion"]
    prob_tensor = torch.cat([predictions[name] for name in class_names], dim=1)
    pred_indices = torch.argmax(prob_tensor, dim=1).cpu().numpy()
    
    index_to_label = {0: "Good", 1: "Keyhole", 2: "Balling", 3: "Lack of Fusion"}
    labels_grid = np.array([index_to_label[i] for i in pred_indices]).reshape(len(power_grid), len(velo_grid))

    # --- 2. Plotting ---
    fig, (ax_defect, ax_acq) = plt.subplots(1, 2, figsize=(15, 6))

    label_to_num = {"Good": 0, "Keyhole": 1, "Balling": 2, "Lack of Fusion": 3}
    defect_numeric_grid = np.vectorize(label_to_num.get)(labels_grid)

    # --- THE FIX IS HERE ---
    # We explicitly define white as the color for the "Good" class.
    colors = {
        "Good": np.array([255, 255, 255]) / 255, # Explicitly define white
        "Keyhole": np.array([224, 123, 123]) / 255,
        "Lack of Fusion": np.array([123, 191, 200]) / 255,
        "Balling": np.array([40, 156, 142]) / 255,
    }
    
    # Create an RGB image for the plot background
    # It starts as black, and we will color EVERY region.
    rgb_grid = np.zeros((*labels_grid.shape, 3))
    alpha = 0.7 # Using a fixed alpha for consistency

    # Now this loop will color all four regions, including "Good"
    for label, num in label_to_num.items():
        if label in colors:
            mask = (defect_numeric_grid == num)
            # For the "Good" region, we don't apply alpha blending
            if label == "Good":
                rgb_grid[mask] = colors[label]
            else:
                # For defects, blend with white for a lighter look
                rgb_grid[mask] = alpha * colors[label] + (1 - alpha) * 1.0

    # Plot the defect map
    ax_defect.imshow(
        rgb_grid,
        extent=[velo_grid.min(), velo_grid.max(), power_grid.min(), power_grid.max()],
        origin="lower", aspect="auto", zorder=1
    )
    # ... (rest of the plotting function is unchanged) ...
    ax_defect.set_xlabel("Scan Velocity (mm/s)")
    ax_defect.set_ylabel("Laser Power (W)")
    ax_defect.set_title(f"Predicted Printability Map (Iter {iteration})" if iteration is not None else "Predicted Printability Map")

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
        ax_acq.scatter(velo_grid[max_idx[1]], power_grid[max_idx[0]], c='red', marker='*', s=150, edgecolor='white', label='Next Candidate')
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



success_it = 0 # counts successful iterations
plotiter = 5 # plot every 5 iterations

while success_it < niter:
    # calculate acquisition function to find the next best point
    J, top_candidates = entropy_sigma(
        X_grid=Xgrid,
        gp_models=gp_models,
        X_train=X,
        top_k=5,
        alpha_dist=0.1
    )

    # try top 3 candidates in case of simulation failure
    sorted_idx = top_candidates[:3]
    d_next = w_next = l_next = None
    x_next = None

    for ind in sorted_idx:
        try:
            # get the next point to sample from our grid
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

    # 3. Append the new data to our training tensors
    X = torch.cat([X, x_next], dim=0)
    Y_good = torch.cat([Y_good, y_good_next], dim=0)
    Y_keyhole = torch.cat([Y_keyhole, y_keyhole_next], dim=0)
    Y_balling = torch.cat([Y_balling, y_balling_next], dim=0)
    Y_lof = torch.cat([Y_lof, y_lof_next], dim=0)
    
    # Update the Y_targets dictionary
    Y_targets = {"Good": Y_good, "Keyhole": Y_keyhole, "Balling": Y_balling, "Lack of Fusion": Y_lof}

    # 4. Retrain all four GP classifiers from scratch with the new point
    print("--- Re-fitting models with new data point ---")
    gp_models["Good"] = fitGP(X, Y_good)
    gp_models["Keyhole"] = fitGP(X, Y_keyhole)
    gp_models["Balling"] = fitGP(X, Y_balling)
    gp_models["Lack of Fusion"] = fitGP(X, Y_lof)
    
    # increment success counter
    success_it += 1

    print(f"[iter {success_it}/{niter}] DONE: "
          f"P={float(x_next[0,0]):.2f}, V={float(x_next[0,1]):.2f}, "
          f"W={w_next:.2f}, D={d_next:.2f}, L={l_next:.2f}, Label={new_label_code}")

    # plot the results periodically
    if (success_it) % plotiter == 0 or success_it == niter:
        plot_printability_map(
            gp_models=gp_models,
            X_grid=Xgrid,
            power_grid=powergrid,
            velo_grid=velogrid,
            J=J,
            X_train=X,
            iteration=success_it
        )
        plt.close('all') # free up memory

# --- Final evaluation after the loop is finished ---
print("\n--- Final Evaluation on All Sampled Points ---")
evaluate_gp_classifiers(gp_models, X, Y_targets, n_samples=len(X), label="After Active Learning")