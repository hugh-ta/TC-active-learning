## ACS test with irl data 
# import libs
import pandas as pd 
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from pyDOE import lhs
from scipy.interpolate import griddata
from tqdm import tqdm
from botorch.utils.sampling import draw_sobol_samples
from scipy.stats import norm
from matplotlib import colors



df = pd.read_csv("results_progress.csv")

# testing column names
#print ("Columns:", df.columns.to_list())
# select LBPF and In718 just cause idk
#data = df [(df["Process"] == "PBF") & (df['Material'] == "IN625") & (df["Sub-process"] == "SLM") & (df["layer thickness"] == 40)]
# print (data)

#setup
dtype = torch.double
SEED = 8
DEVICE = torch.device('cpu')
torch.set_default_dtype(dtype)
torch.manual_seed(SEED)
np.random.seed(SEED)

# params
maternmu = 2.5  #mu =1.5 -> matern 3/2, nu 2.5=5/2
restarts = 10 # make sure we dont get stuck in a local min

ngrid = 100 # for our grid search
padding = 0.05 # so we dont sample right up against the edge

# Optional: if you have per-point measured noise std (same shape as Y),
# provide them here; otherwise leave as None and we’ll learn noise.
# Yw_noise_std = None
# Yl_noise_std = None
# Yd_noise_std = None

# Now train GP on the data to create psuedo-ground truth
# process variables/dimensions
dfpower = df["Power"].values.reshape(-1,1)
dfvelo = df["Speed"].values.reshape(-1,1)

X_np = np.hstack([dfpower, dfvelo])  # shape (N,2)
X = torch.tensor(X_np, dtype=dtype, device=DEVICE)

dfwidth = df["Width"].values.reshape(-1,1)
dflength = df["Length"].values.reshape(-1,1)
dfdepth = df["Depth"].values.reshape(-1,1)

Yw = torch.tensor(dfwidth, dtype=dtype, device=DEVICE)
Yl = torch.tensor (dflength, dtype=dtype, device=DEVICE)
Yd= torch.tensor (dfdepth, dtype=dtype, device=DEVICE)

#train GPs
def TrainGP(X,Y, nu = 2.5): #nu =1.5 -> matern 3/2, nu 2.5=5/2 #def make_model(X, Y, nu=2.5, fixed_noise_std=True) if we wanted to include noise
    kernel = ScaleKernel(base_kernel= MaternKernel(nu=nu, ard_num_dims= 2)) # 2 because we have 2 inputs, ard allows for different length scales for different parameters
    model = SingleTaskGP(train_X=X, train_Y=Y, covar_module=kernel, input_transform= Normalize(d=X.shape[-1]), outcome_transform= Standardize(m=Y.shape[-1])) #ARD = automated revelance detection
    return model
 
def fitmodel(model, restarts =10 ):
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll,num_restarts = restarts)
    return model

def diagnostics(gp,label):
    ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    outscale = float(gp.covar_module.outputscale.detach().cpu())
    # For FixedNoiseGP, likelihood noise is not learned; for SingleTaskGP, it is.
    noise = float(gp.likelihood.noise.detach().cpu()) if hasattr(gp.likelihood, "noise") else float("nan")
    print(f"\n[{label}]")
    print(f"  Lengthscales (per input): {ls}  (ARD)")
    print(f"  Outputscale (signal var): {outscale:.6g}")
    print(f"  Noise variance:           {noise:.6g}")

#training time
gpw = TrainGP(X, Yw, nu=maternmu)
gpl = TrainGP(X,Yl, nu = maternmu)
gpd = TrainGP(X,Yd, nu=maternmu)

gpw = fitmodel(gpw, restarts=restarts); diagnostics(gpw,"Width GP")
gpl = fitmodel(gpl, restarts=restarts); diagnostics(gpl, "length GP")
gpd = fitmodel (gpd, restarts=restarts); diagnostics(gpd, "depth GP")

## Training Ground Truth Printability Map
#grid
powermin, powermax = X[:,0].min().item(), X[:,0].max().item()
velomin, velomax = X[:,1].min().item(), X[:,1].max().item()
bounds = torch.tensor([[powermin, powermax],   
                       [velomin, velomax]], dtype=dtype, device=DEVICE)
powergrid = np.linspace(powermin, powermax, ngrid)
velogrid = np.linspace(velomin, velomax, ngrid)
PP,VV = np.meshgrid(powergrid, velogrid, indexing="ij")

gridpoints = np.column_stack([PP.ravel(), VV.ravel()])
Xgrid = torch.tensor(gridpoints, dtype=dtype, device=DEVICE)

#predict on grid
with torch.no_grad():
    postw = gpw.posterior(Xgrid)
    meanw = postw.mean.view(ngrid, ngrid).cpu().numpy()
    stdw  = postw.variance.sqrt().view(ngrid, ngrid).cpu().numpy()

    postl = gpl.posterior(Xgrid)
    meanl = postl.mean.view(ngrid, ngrid).cpu().numpy()
    stdl  = postl.variance.sqrt().view(ngrid, ngrid).cpu().numpy()

    postd = gpd.posterior(Xgrid)
    meand = postd.mean.view(ngrid, ngrid).cpu().numpy()
    stdd  = postd.variance.sqrt().view(ngrid, ngrid).cpu().numpy()

# printability criteria

thickness = 50  # microns

# Criteria
keyholing = 1.5# W/D < 1.5
lof = 1.9       # D/t < 1.5
balling = 4.35 # L/W > 2.67

# Defect maps
keyholingmap = (meanw / meand) < keyholing
lofmap = (meand / thickness) < lof
ballingmap = (meanl / meanw) > balling

# Printable region = none of the defects triggered
printablemap = ~(keyholingmap | lofmap | ballingmap)

# Flatten grid and masks for plotting
power_flat = PP.ravel()
velo_flat = VV.ravel()
keyholing_flat = keyholingmap.ravel()
lof_flat = lofmap.ravel()
balling_flat = ballingmap.ravel()
printable_flat = printablemap.ravel()

# Assign each point to a single defect type (priority order)
category = np.empty(power_flat.shape, dtype=object)   # start unassigned
category[keyholing_flat] = 'Keyholing'
category[lof_flat & ~keyholing_flat] = 'Lack of Fusion'
category[balling_flat & ~keyholing_flat & ~lof_flat] = 'Balling'
category[~(keyholing_flat | lof_flat | balling_flat)] = 'Printable'  # assign last

# Debug counts
unique, counts = np.unique(category, return_counts=True)
print("Category distribution:")
for u, c in zip(unique, counts):
    print(f"  {u:12s}: {c} points ({100*c/len(category):.2f}%)")

# Color map for each category
color_map = {
    'Printable': 'green',
    'Keyholing': 'red',
    'Lack of Fusion': 'blue',
    'Balling': 'orange'
}

# Plotting
plt.figure(figsize=(8, 6))
for cat, color in color_map.items():
    mask = category == cat
    plt.scatter(velo_flat[mask], power_flat[mask], color=color, s=5, label=cat)

plt.xlabel('Velocity')
plt.ylabel('Power')
plt.title('Simulated Ground Truth (even though its ugly)')
plt.legend(markerscale=3)
plt.show()


## Compare ACS vs Grid search vs HLS or something

#grid search
def evaluate_printability(sampledim, gpw, gpl, gpd, 
                          powermin, powermax, velomin, velomax, padding, 
                          thickness=50, DEVICE="cpu", dtype=torch.float32):
    # Build grid
    samplepower = np.linspace(powermin + padding*(powermax - powermin),
                              powermax - padding*(powermax - powermin), sampledim)
    samplevelo = np.linspace(velomin + padding*(velomax - velomin),
                             velomax - padding*(velomax - velomin), sampledim)
    SP, SV = np.meshgrid(samplepower, samplevelo, indexing="ij")
    samplepoints = np.column_stack([SP.ravel(), SV.ravel()])
    Xsample = torch.tensor(samplepoints, dtype=dtype, device=DEVICE)

    # GP predictions
    with torch.no_grad():
        postw = gpw.posterior(Xsample)
        meanw = postw.mean.view(sampledim, sampledim).cpu().numpy()

        postl = gpl.posterior(Xsample)
        meanl = postl.mean.view(sampledim, sampledim).cpu().numpy()

        postd = gpd.posterior(Xsample)
        meand = postd.mean.view(sampledim, sampledim).cpu().numpy()

    # Criteria
    keyholing = 1.5# W/D < 1.5
    lof = 1.5        # D/t < 1.5
    balling = 3.8 # L/W > 2.67


    keyholingmap = (meanw / meand) < keyholing
    lofmap       = (meand / thickness) < lof
    ballingmap   = (meanl / meanw) > balling

    # Priority order: Keyholing > LOF > Balling > Printable
    category = np.full_like(meanw, 3, dtype=int)   # start as Balling (lowest priority)
    category[lofmap] = 2                           # overwrite with LOF
    category[keyholingmap] = 1                     # overwrite with Keyholing
    category[~(keyholingmap | lofmap | ballingmap)] = 0  # finally Printable (highest priority)

    return SP, SV, category

# ---- Run at multiple grid resolutions ----
grid_sizes = [5, 10, 20, 40]
fig, axes = plt.subplots(1, len(grid_sizes), figsize=(4*len(grid_sizes), 4))

# Colormap and labels
cmap = {0: "green", 1: "red", 2: "blue", 3: "orange"}
labels = {0: "Printable", 1: "Keyholing", 2: "Lack of Fusion", 3: "Balling"}

for ax, sdim in zip(axes, grid_sizes):
    SP, SV, category = evaluate_printability(
        sdim, gpw, gpl, gpd, powermin, powermax, velomin, velomax, padding
    )

    # Plot colored regions
    for val, color in cmap.items():
        mask = category == val
        ax.contourf(SV, SP, mask, levels=[0.5, 1.5], colors=[color], alpha=0.4)

    # Overlay black contour around printable region
    ax.contour(SV, SP, category == 0, levels=[0.5], colors="black", linewidths=1.2)

    # Show sampled grid points
    ax.scatter(SV, SP, color="k", s=10, marker="o", alpha=0.8)

    # Title with grid size and number of points
    npoints = sdim * sdim
    ax.set_title(f"{sdim}×{sdim} grid ({npoints} points)")

    ax.set_xlabel("Velocity")
    ax.set_ylabel("Power")

# Single legend
handles = [plt.Rectangle((0,0),1,1, color=c) for c in cmap.values()]
fig.legend(handles, labels.values(), loc="upper center", ncol=4)

plt.tight_layout()
plt.show()

#lhs
def evaluate_printability_lhs(sampledim, gpw, gpl, gpd,
                              powermin, powermax, velomin, velomax, padding,
                              thickness=50, DEVICE="cpu", dtype=torch.float32):
    nsamples = sampledim**2
    lhs_points = lhs(2, samples=nsamples)

    lhs_points[:,0] = powermin + padding*(powermax - powermin) + lhs_points[:,0] * ((1-2*padding)*(powermax-powermin))
    lhs_points[:,1] = velomin + padding*(velomax - velomin) + lhs_points[:,1] * ((1-2*padding)*(velomax-velomin))

    Xsample = torch.tensor(lhs_points, dtype=dtype, device=DEVICE)

    with torch.no_grad():
        meanw = gpw.posterior(Xsample).mean.cpu().numpy()
        meanl = gpl.posterior(Xsample).mean.cpu().numpy()
        meand = gpd.posterior(Xsample).mean.cpu().numpy()

    # Criteria
    keyholing = 1.5# W/D < 1.5
    lof = 1.5        # D/t < 1.5
    balling = 3.8 # L/W > 2.67


    keyholingmap = (meanw / meand) < keyholing
    lofmap       = (meand / thickness) < lof
    ballingmap   = (meanl / meanw) > balling

    # Priority order: Keyholing > LOF > Balling > Printable
    category = np.full_like(meanw, 3, dtype=int)   # start as Balling (lowest priority)
    category[lofmap] = 2                           # overwrite with LOF
    category[keyholingmap] = 1                     # overwrite with Keyholing
    category[~(keyholingmap | lofmap | ballingmap)] = 0  # finally Printable (highest priority)

    return lhs_points, category


# ---- Run at multiple LHS sample sizes ----
lhs_sizes = [5, 10, 20, 40]
fig, axes = plt.subplots(1, len(lhs_sizes), figsize=(4*len(lhs_sizes), 4))

# Colormap and labels
cmap = {0: "green", 1: "red", 2: "blue", 3: "orange"}
labels = {0: "Printable", 1: "Keyholing", 2: "Lack of Fusion", 3: "Balling"}

for ax, sdim in zip(axes, lhs_sizes):
    lhs_points, category = evaluate_printability_lhs(
        sdim, gpw, gpl, gpd, powermin, powermax, velomin, velomax, padding
    )

    # Interpolate scattered categories onto a regular grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(lhs_points[:,1].min(), lhs_points[:,1].max(), 200),  # velocity
        np.linspace(lhs_points[:,0].min(), lhs_points[:,0].max(), 200)   # power
    )
    grid_cat = griddata(
        lhs_points[:, [1, 0]],   # velocity, power
        category.ravel(),        # ensure 1D
        (grid_x, grid_y),
        method="nearest"
    )

    # Plot filled regions
    for val, color in cmap.items():
        if np.any(grid_cat == val):  # only plot if region exists
            mask = (grid_cat == val).astype(float).squeeze()  # ensure 2D
            ax.contourf(grid_x, grid_y, mask, levels=[0.5, 1.5],
                        colors=[color], alpha=0.4)

    # Outline printable region
    ax.contour(grid_x, grid_y, (grid_cat == 0).astype(float),
               levels=[0.5], colors="black", linewidths=1.2)

    # Sample points
    ax.scatter(lhs_points[:,1], lhs_points[:,0], color="k", s=10, marker="o", alpha=0.8)

    # Titles
    npoints = sdim * sdim
    ax.set_title(f"LHS {sdim}×{sdim} ({npoints} points)")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Power")

# Single legend
handles = [plt.Rectangle((0,0),1,1, color=cmap[k]) for k in sorted(cmap.keys())]
fig.legend(handles, [labels[k] for k in sorted(labels.keys())], loc="upper center", ncol=4)

plt.tight_layout()
plt.show()

# ACS time in two primary ways
# 1.) Constraint Aware
# 2.) Constraint Unaware
# For both we will use the same acquistion function, and do one determisitic (assuming constraints are gaussian, which they arent) and one stocastic (MC sampling of constraints)

# define our acquistion function
def entropy_sigma(
        X, #all potential points to evaluate
        gps, #our gp1 gp2 gp2
        constraints, #list of constraints
        thickness, # thickness parameter
        mode = "Gaussian", # mode we're solving
        nmc = 64, #number of monte carlo samples if using MC mode
        device = "cpu",
        dtype = torch.double):
    import torch
    import numpy as np
    from tqdm import tqdm
    from scipy.stats import norm

    #unpack gps and constraints, 
    gpw, gpl, gpd = gps
    c1,c2,c3 = constraints
    numpoints = X.shape[0]

    #get posteriors
    #gp1 posteriors
    postw = gpw.posterior(X)
    meanw = postw.mean.squeeze(-1)
    stdw = postw.variance.sqrt().squeeze(-1)

    #gp2 posteriors
    postl = gpl.posterior(X)
    meanl = postl.mean.squeeze(-1)
    stdl = postl.variance.sqrt().squeeze(-1)

    #gp3 posteriors
    postd=  gpd.posterior(X)
    meand = postd.mean.squeeze(-1)
    stdd = postd.variance.sqrt().squeeze(-1)

    #clip std to avoid numerical issues
    stdw = torch.clamp(stdw, min=1e-6)
    stdl = torch.clamp(stdl, min=1e-6)
    stdd = torch.clamp(stdd, min=1e-6)

    # setup probability tensors
    p1 = torch.zeros(numpoints, device=device, dtype=dtype) #keyholing
    p2 = torch.zeros(numpoints, device=device,dtype=dtype) # lof
    p3 = torch.zeros(numpoints, device=device, dtype=dtype) # balling
    p4 = torch.zeros(numpoints, device = device, dtype=dtype)

    for i in tqdm(range(numpoints), desc ="Acquisition func eval"):
        #ok for the gaussian acquistion functions we need the probabilit of satisfying the constraints: assume that gp/gp are gaussian and that uncertainty is small and all that
        if mode == "Gaussian":
            meanw_i = meanw[i].detach().cpu().numpy()
            stdw_i = stdw[i].detach().cpu().numpy()
            meanl_i = meanl[i].detach().cpu().numpy()
            stdl_i = stdl[i].detach().cpu().numpy()
            meand_i = meand[i].detach().cpu().numpy()
            stdd_i = stdd[i].detach().cpu().numpy()
            meand_i = np.clip(meand_i, 1e-6, None)

            #probability of satisfying the constraints:
            # keyholing 
            p1[i] = torch.tensor(norm.cdf((c1 - meanw_i/meand_i) / np.sqrt((stdw_i/meand_i)**2 + (meanw_i*stdd_i/meand_i**2)**2)), device=device, dtype=dtype)
            #LOF
            p2[i] = torch.tensor(norm.cdf((c2 - meand_i/thickness) / (stdd_i/thickness)), device=device, dtype=dtype)
            #balling
            p3[i] = torch.tensor(norm.cdf((c3 - meanl_i/meanw_i) / np.sqrt((stdl_i/meanw_i)**2 + (meanl_i*stdw_i/meanw_i**2)**2)), device=device, dtype=dtype)
            # printable
            p4[i] = 1 - (p1[i] + p2[i] +p3[i]) # BIG ASSUMPTION!!!!!!! Asuming independence here

        elif mode == "Blind":
            # This mode doesnt know about our constraints, just trys to reduce overall uncertainty
            p1[i] = 0
            p2[i] = 0
            p3[i] = 0
            # It goes in completely blind, and we're telling it that EVERYTHING is printable to make entropy 0, just relying on uncertainty reduction
            p4[i] = 1

        elif mode == "MC":
            #now we're doing montecarlo sampling to estimate the probabilites of satisfying the constraints
            #unlike the gaussian mode, this can handle non-gaussian constraints, which is what we have, so theoretically should solve in less iterations
            samplew = gpw.posterior(X).rsample(torch.Size([nmc])) #shape (nmc, numpoints)
            samplel = gpl.posterior(X).rsample(torch.Size([nmc]))
            sampled = gpd.posterior(X).rsample(torch.Size([nmc]))
            #evaluate constraints
            keyholing = (samplew / sampled) < c1
            lof = sampled / thickness < c2
            balling = (samplel / samplew) > c3

            p1[i] = keyholing[:, i].float().mean()
            p2[i] = lof[:, i].float().mean()
            p3[i] = balling[:, i].float().mean()
            p4[i] = 1 - (p1[i] + p2[i] + p3[i])
            p4[i] = torch.clamp(p4[i], 1e-6, 1-1e-6) # avoid numerical issues

        else:
            print ("typo loser")

    # now we have the probabilities, now to calculate our acquisition function!
    H = -(p1 * torch.log(p1 + 1e-12) + p2 * torch.log(p2 + 1e-12) + p3 * torch.log(p3 + 1e-12) + p4 * torch.log(p4 + 1e-12))
    J = H * torch.stack([stdw, stdl, stdd], dim=-1).prod(dim=-1) # joint uncertainty

    return J.detach().cpu().numpy()


def plot_active_learning_history_fixed(
        Xgrid, J_history, gps_history, Xtrain_history,
        constraints, thickness, ngrid, iterations_to_plot
    ):
    """
    Plots the defect boundaries and acquisition function for selected iterations.
    Uses the GPs trained at each iteration (gps_history), not ground truth.
    """
    num_iters = len(iterations_to_plot)
    fig, axes = plt.subplots(num_iters, 2, figsize=(12, 4*num_iters))

    if num_iters == 1:
        axes = axes[np.newaxis, :]  # ensure 2D array

    cmap_defects = colors.ListedColormap(['green','red','blue','orange'])
    labels_defects = ['Printable','Keyholing','LOF','Balling']

    c1, c2, c3 = constraints

    for idx, iter_num in enumerate(iterations_to_plot):
        gps = gps_history[iter_num-1]
        Xtrain = Xtrain_history[iter_num-1]
        J = J_history[iter_num-1]

        gpw, gpl, gpd = gps

        # compute GP posteriors
        postw = gpw.posterior(Xgrid)
        meanw = postw.mean.squeeze(-1).detach()
        stdw = postw.variance.sqrt().squeeze(-1).detach()

        postl = gpl.posterior(Xgrid)
        meanl = postl.mean.squeeze(-1).detach()
        stdl = postl.variance.sqrt().squeeze(-1).detach()

        postd = gpd.posterior(Xgrid)
        meand = postd.mean.squeeze(-1).detach()
        stdd = postd.variance.sqrt().squeeze(-1).detach()

        stdw = torch.clamp(stdw, min=1e-6)
        stdl = torch.clamp(stdl, min=1e-6)
        stdd = torch.clamp(stdd, min=1e-6)

        # compute probabilities
        p1 = torch.zeros(Xgrid.shape[0])
        p2 = torch.zeros_like(p1)
        p3 = torch.zeros_like(p1)
        p4 = torch.zeros_like(p1)

        for i in range(Xgrid.shape[0]):
            meanw_i, stdw_i = meanw[i].item(), stdw[i].item()
            meanl_i, stdl_i = meanl[i].item(), stdl[i].item()
            meand_i, stdd_i = meand[i].item(), stdd[i].item()
            meand_i = max(meand_i, 1e-6)

            p1[i] = norm.cdf((c1 - meanw_i / meand_i) /
                              np.sqrt((stdw_i / meand_i)**2 + (meanw_i * stdd_i / meand_i**2)**2))
            p2[i] = norm.cdf((c2 - meand_i / thickness) / (stdd_i / thickness))
            p3[i] = norm.cdf((c3 - meanl_i / meanw_i) /
                              np.sqrt((stdl_i / meanw_i)**2 + (meanl_i * stdw_i / meanw_i**2)**2))
            p4[i] = 1 - (p1[i] + p2[i] + p3[i])

        probs = torch.stack([p4, p1, p2, p3], dim=-1)
        category = torch.argmax(probs, dim=-1).numpy()

        # reshape for plotting
        grid_x = Xgrid[:,1].reshape(ngrid, ngrid).cpu().numpy()  # velocity
        grid_y = Xgrid[:,0].reshape(ngrid, ngrid).cpu().numpy()  # power
        grid_cat = category.reshape(ngrid, ngrid)
        grid_J = J.reshape(ngrid, ngrid)

        # top: defect boundaries
        ax = axes[idx, 0]
        ax.imshow(grid_cat, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                  cmap=cmap_defects, alpha=0.6, aspect='auto')
        ax.scatter(Xtrain[:,1].cpu(), Xtrain[:,0].cpu(), color='k', s=25, label='Training points')
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Power")
        ax.set_title(f"Iteration {iter_num}: Boundary estimates")
        handles = [plt.Rectangle((0,0),1,1,color=cmap_defects(i)) for i in range(4)]
        ax.legend(handles + [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='k', markersize=5)],
                  labels_defects + ['Training points'], loc='upper right')

        # bottom: acquisition function
        ax = axes[idx, 1]
        im = ax.imshow(grid_J, origin='lower', extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                       cmap='viridis', aspect='auto')
        ax.scatter(Xtrain[:,1].cpu(), Xtrain[:,0].cpu(), color='k', s=25)
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Power")
        ax.set_title(f"Iteration {iter_num}: Acquisition Function")
        fig.colorbar(im, ax=ax, label='Acquisition Value')

    plt.tight_layout()
    plt.show()



# the params
Ntrain = 10  # initial training points
Niter = 30  # number of iterations
thickness = 50
keyholing = 1.5
lof = 1.5
balling = 3.8
constraints = [keyholing, lof, balling]
model = "MC"
nmc = 16

# --- Initial training data ---
Xtrain_raw = draw_sobol_samples(bounds=torch.tensor([[0.0,1.0],[0.0,1.0]], dtype=dtype, device=DEVICE),
                                n=1, q=Ntrain, seed=SEED).squeeze(0)  # in [0,1]
Xtrain = bounds[:,0] + (bounds[:,1] - bounds[:,0]) * Xtrain_raw  # scale to actual bounds
Ytrainw = gpw.posterior(Xtrain).mean.detach()  # (Ntrain, 1)
Ytrainl = gpl.posterior(Xtrain).mean.detach()
Ytraind = gpd.posterior(Xtrain).mean.detach()

# repeatgrid
powergrid = torch.linspace(bounds[0,0], bounds[0,1], ngrid, device=DEVICE, dtype=dtype)
velogrid = torch.linspace(bounds[1,0], bounds[1,1], ngrid, device=DEVICE, dtype=dtype)
PP, VV = torch.meshgrid(powergrid, velogrid, indexing="ij")
Xgrid = torch.stack([PP.flatten(), VV.flatten()], dim=-1)  # shape (ngrid**2, 2)

# Keep track of initial data
Xinit = Xtrain.clone()
Yinitw = Ytrainw.clone()
Yinitl = Ytrainl.clone()
Yinitd = Ytraind.clone()

#history for plotting
gps_history = []
Xtrain_history = []
J_history = []

# --- Active Learning Loop ---
for i in tqdm(range(1, Niter + 1)):
    # Fit GPs
    gp1 = TrainGP(Xtrain, Ytrainw, nu=maternmu)
    gp2 = TrainGP(Xtrain, Ytrainl, nu=maternmu)
    gp3 = TrainGP(Xtrain, Ytraind, nu=maternmu)

    # Tune hyperparameters
    gp1 = fitmodel(gp1, restarts=restarts); diagnostics(gp1, "Width GP")
    gp2 = fitmodel(gp2, restarts=restarts); diagnostics(gp2, "Length GP")
    gp3 = fitmodel(gp3, restarts=restarts); diagnostics(gp3, "Depth GP")
    gps = [gp1, gp2, gp3]

    # Evaluate acquisition function
    J = entropy_sigma(
        Xgrid,
        gps,
        constraints=constraints,
        thickness=thickness,
        mode=model,
        nmc=nmc,
        device=DEVICE,
        dtype=dtype
    )

    # Pick next point
    ind = np.argmax(J)
    x_next = Xgrid[ind:ind+1]  # shape (1,2)

    # Evaluate GP means at next point (detached to avoid autograd issues)
    w_next = gpw.posterior(x_next).mean.detach()  # shape (1,1)
    l_next = gpl.posterior(x_next).mean.detach()
    d_next = gpd.posterior(x_next).mean.detach()

    # Add new point to training data
    Xtrain = torch.cat([Xtrain, x_next], dim=0)
    Ytrainw = torch.cat([Ytrainw, w_next], dim=0)
    Ytrainl = torch.cat([Ytrainl, l_next], dim=0)
    Ytraind = torch.cat([Ytraind, d_next], dim=0)

    #history for plooting
    gps_history.append(gps)
    Xtrain_history.append(Xtrain.clone())
    J_history.append(J.copy())
iterations_to_plot = list(range(5, Niter+1, 5))  # e.g., [5,10,15,20,25,30]
plot_active_learning_history_fixed(Xgrid, J_history, gps_history, Xtrain_history, constraints, thickness, ngrid, iterations_to_plot)
