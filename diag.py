# stable_kmtgp_stage_train.py
"""
Stronger-stabilized training for KroneckerMultiTaskGP.

Key additions:
 - staged training: (A) freeze outputscale + task covar for N_FREEZE iters, (B) unfreeze and continue
 - automatic param clamping & NaN/Inf repair by name
 - explicit per-task noise init and larger INIT_NOISE
 - lower LR, more jitter, gradient clipping
"""

import math
import numpy as np
import pandas as pd
import torch
import gpytorch

from botorch.models import KroneckerMultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import SmoothedBoxPrior, GammaPrior
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------- USER TUNABLES ----------------
DATA_CSV = "results_progress.csv"
USE_LOG_Y = False
CLAMP_OUTLIERS = True
LOWER_Q, UPPER_Q = 0.005, 0.995

dtype = torch.double
device = torch.device("cpu")

N_TRAIN = 10
TRAIN_ITERS = 800
LR = 5e-5               # even smaller LR for stability
WEIGHT_DECAY = 1e-6
MIN_POS = 1e-9
PRINT_EVERY = 60

OUTPUTSCALE_PRIOR_LO = 1e-2
OUTPUTSCALE_PRIOR_HI = 1e2
LENGTH_GAMMA_A = 3.0
LENGTH_GAMMA_B = 1.5
INIT_OUTPUTSCALE = 1.0
INIT_NOISE = 1.0        # stronger initial noise
JITTER = 1e-2
GRAD_CLIP = 0.8

# staged training
N_FREEZE_ITERS = 120    # keep scale & task-covar frozen for this many iters
SEED = 42
# -----------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------- data load ----------------
df = pd.read_csv(DATA_CSV)
req = ["Depth", "Width", "Length", "Power", "Speed"]
for c in req:
    if c not in df.columns:
        raise RuntimeError(f"Missing required column: {c}")

df["Depth"] = df["Depth"].replace(0, 1e-6)
for c in req:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=req).reset_index(drop=True)

power = df["Power"].values.reshape(-1, 1).astype(float)
speed = df["Speed"].values.reshape(-1, 1).astype(float)
depth = df["Depth"].values.reshape(-1, 1).astype(float)
width = df["Width"].values.reshape(-1, 1).astype(float)
length = df["Length"].values.reshape(-1, 1).astype(float)

Y_np = np.hstack([depth, width, length])
X_in = np.hstack([power, speed])

if CLAMP_OUTLIERS:
    for j in range(Y_np.shape[1]):
        lo = np.quantile(Y_np[:, j], LOWER_Q)
        hi = np.quantile(Y_np[:, j], UPPER_Q)
        Y_np[:, j] = np.clip(Y_np[:, j], lo, hi)
    print("[data] clamped outliers")

if USE_LOG_Y:
    Y_np_tr = np.log1p(np.maximum(Y_np, MIN_POS))
else:
    Y_np_tr = Y_np.copy()

X_norm = (X_in - X_in.mean(axis=0)) / (X_in.std(axis=0) + 1e-12)
n_total = X_norm.shape[0]
N_TRAIN = min(N_TRAIN, n_total)
idx = np.random.default_rng(SEED).permutation(n_total)[:N_TRAIN]

X = torch.tensor(X_norm[idx], dtype=dtype, device=device)
Y_sel = Y_np_tr[idx, :].astype(float)
Y = torch.tensor(Y_sel, dtype=dtype, device=device)

n, d = X.shape
_, m = Y.shape
print(f"[data] n={n}, d={d}, tasks={m}")

# --------------- priors & kernel ----------------
length_prior = GammaPrior(LENGTH_GAMMA_A, LENGTH_GAMMA_B)
output_prior = SmoothedBoxPrior(OUTPUTSCALE_PRIOR_LO, OUTPUTSCALE_PRIOR_HI)

base_k = MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_prior=length_prior)
covar = ScaleKernel(base_k, outputscale_prior=output_prior)

with torch.no_grad():
    try:
        covar.outputscale = torch.tensor(INIT_OUTPUTSCALE, dtype=dtype, device=device)
    except Exception:
        try:
            covar._set_outputscale(torch.tensor(INIT_OUTPUTSCALE, dtype=dtype, device=device))
        except Exception:
            pass

# ------------- likelihood ----------------------
likelihood = MultitaskGaussianLikelihood(num_tasks=m, rank=1)
with torch.no_grad():
    try:
        likelihood.noise = torch.full((m,), INIT_NOISE, dtype=dtype, device=device)
    except Exception:
        try:
            # a fallback attempt to tune raw_noise if needed
            rn = torch.tensor(INIT_NOISE, dtype=dtype, device=device).repeat(m)
            likelihood.noise_covar.register_parameter("raw_noise", torch.nn.Parameter(torch.log(torch.expm1(rn))))
        except Exception:
            print("[warn] couldn't set per-task noise explicitly; check gpytorch version")

model = KroneckerMultiTaskGP(
    train_X=X,
    train_Y=Y,
    covar_module=covar,
    input_transform=Normalize(d),
    outcome_transform=Standardize(m),
    likelihood=likelihood
)

def diag(prefix=""):
    try:
        kern = model.covar_module.data_covar_module
        ls = kern.base_kernel.lengthscale.detach().cpu().numpy().ravel() if hasattr(kern, "base_kernel") else kern.lengthscale.detach().cpu().numpy().ravel()
        outscale = float(kern.outputscale.detach().cpu()) if hasattr(kern, "outputscale") else float("nan")
    except Exception:
        ls, outscale = "?", "?"
    try:
        task_cov = model.covar_module.task_covar_module.covar_matrix.detach().cpu().numpy()
    except Exception:
        task_cov = "?"
    try:
        noise = likelihood.noise.detach().cpu().numpy() if hasattr(likelihood, "noise") else "?"
    except Exception:
        noise = "?"
    print(prefix, "lengthscales:", ls, "outputscale:", outscale, "task_cov:", task_cov, "noise:", noise)

diag("INIT:")

# ---------------- helper functions ----------------
def clamp_named_params(model, min_val=1e-6, max_val=1e6):
    """Search parameters by name and clamp likely-outputscale/lengthscale raw params or simple outputscale tensors."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None:
                continue
            # clamp raw_outputscale-like params or outputscale-like parameters
            ln = name.lower()
            if ("raw_outputscale" in ln) or ("raw_scale" in ln) or ("outputscale" in ln):
                # clamp in unconstrained space: if raw_ prefix likely unconstrained (we use log clamp safeguard)
                try:
                    p.data.clamp_(math.log(min_val), math.log(max_val))
                except Exception:
                    try:
                        p.data.clamp_(min_val, max_val)
                    except Exception:
                        pass
            # lengthscale raw clamping
            if "raw_lengthscale" in ln or "lengthscale" in ln:
                try:
                    p.data.clamp_(math.log(min_val), math.log(max_val))
                except Exception:
                    try:
                        p.data.clamp_(min_val, max_val)
                    except Exception:
                        pass

def repair_nan_params(model):
    """If essential params go NaN, try safe re-init: outputscale -> INIT_OUTPUTSCALE, lengthscale -> 1.0"""
    repaired = False
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                print("[repair] non-finite detected in", name)
                ln = name.lower()
                if "raw_outputscale" in ln or "outputscale" in ln:
                    # attempt to locate covar and set outputscale
                    try:
                        ks = getattr(model.covar_module, "data_covar_module", None)
                        if ks is None:
                            ks = model.covar_module
                        if hasattr(ks, "outputscale"):
                            ks.outputscale = torch.tensor(INIT_OUTPUTSCALE, dtype=p.dtype, device=p.device)
                            print("[repair] reset outputscale to INIT_OUTPUTSCALE")
                            repaired = True
                    except Exception:
                        pass
                if "raw_lengthscale" in ln or "lengthscale" in ln:
                    try:
                        k = getattr(model.covar_module, "data_covar_module", None)
                        if k is None:
                            k = model.covar_module
                        # set base lengthscale to moderate value
                        if hasattr(k, "base_kernel") and hasattr(k.base_kernel, "lengthscale"):
                            k.base_kernel.lengthscale = torch.ones_like(k.base_kernel.lengthscale)
                        elif hasattr(k, "lengthscale"):
                            k.lengthscale = torch.ones_like(k.lengthscale)
                        print("[repair] reset lengthscale to 1.0")
                        repaired = True
                    except Exception:
                        pass
    return repaired

def freeze_params_by_keyword(model, keywords):
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in keywords):
            p.requires_grad = False

def unfreeze_params_by_keyword(model, keywords):
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in keywords):
            p.requires_grad = True

# ---------------- training setup ----------------
model.train(); likelihood.train()
mll = ExactMarginalLogLikelihood(likelihood, model)
opt = Adam(list(model.parameters()) + list(likelihood.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=60, min_lr=1e-10)

# freeze outputscale & task covar raw params initially
freeze_keywords = ["outputscale", "raw_outputscale", "task_covar", "task_covar_module", "covar_matrix"]
freeze_params_by_keyword(model, freeze_keywords)
print("[train] frozen params containing keywords:", freeze_keywords)

loss_hist = []
with gpytorch.settings.cholesky_jitter(JITTER):
    for it in range(1, TRAIN_ITERS + 1):
        opt.zero_grad()
        out = model(X)
        loss = -mll(out, Y)

        if not torch.isfinite(loss):
            print(f"[train] non-finite loss at iter {it}: {loss}")
            # attempt repair once and continue; if still bad, break
            repaired = repair_nan_params(model)
            if not repaired:
                break
            else:
                print("[train] repaired NaNs; continuing")
                continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(likelihood.parameters()), GRAD_CLIP)
        opt.step()

        # After opt step: clamp named params and repair if needed
        clamp_named_params(model, min_val=1e-6, max_val=1e6)
        repaired = repair_nan_params(model)
        if repaired:
            print(f"[train] param repair after iter {it}")

        # Unfreeze after warm-start period
        if it == N_FREEZE_ITERS:
            unfreeze_params_by_keyword(model, freeze_keywords)
            print(f"[train] unfreezing scale/task-covar params at iter {it}")

        # Occasionally reduce LR on plateau
        if it % 10 == 0:
            old = opt.param_groups[0]["lr"]
            sched.step(float(loss.detach().cpu().numpy()))
            new = opt.param_groups[0]["lr"]
            if new < old:
                print(f"[train] LR reduced {old:.2e} -> {new:.2e}")

        loss_val = float(loss.item())
        loss_hist.append(loss_val)

        if it % PRINT_EVERY == 0 or it == 1:
            print(f"[train] iter {it} loss {loss_val:.6g}")
            diag("   ")

print("[train] tail losses:", loss_hist[-8:])

# ---------------- evaluation ----------------
model.eval(); likelihood.eval()
with gpytorch.settings.cholesky_jitter(JITTER):
    with torch.no_grad():
        post = model(X)
        mean_std = post.mean.cpu()
        try:
            Y_untrans, _ = model.outcome_transform.untransform(mean_std)
            Y_untrans = Y_untrans.numpy()
        except Exception:
            Y_untrans = mean_std.numpy()

Y_pred_orig = Y_untrans.copy()
Y_true_orig = Y_sel.copy()

if USE_LOG_Y:
    Y_pred_final = np.expm1(Y_pred_orig)
    Y_true_final = np.expm1(Y_true_orig)
else:
    Y_pred_final = Y_pred_orig.copy()
    Y_true_final = Y_true_orig.copy()

# safety: replace non-finite predictions
nan_mask = ~np.isfinite(Y_pred_final)
if nan_mask.any():
    print("[eval] WARNING: NaNs in preds; replacing with MIN_POS")
    Y_pred_final = np.where(nan_mask, MIN_POS, Y_pred_final)

Y_pred_final = np.maximum(Y_pred_final, MIN_POS)

rmse = np.sqrt(np.mean((Y_pred_final - Y_true_final) ** 2, axis=0))
print("[eval] Post-train RMSE per task:", rmse)
diag("FINAL:")

for i in range(min(12, n)):
    t = Y_true_final[i]; p = Y_pred_final[i]
    print(f" True: {t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f} | Pred: {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}")
