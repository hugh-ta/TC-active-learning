## Printability Map Creator (yes balling)

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors

# toggles
PLOT_MODE = "OVERLAY_POINTS"  # options: "INTERPOLATED", "RAW_POINTS", "OVERLAY_POINTS"
USE_BALLING = True            # toggle Balling 
USE_INTERPOLATION = False      # toggle interpolation
sigma = 1                     # Gaussian smoothing

dtype = 'float32'
thickness = 10  # microns

# import data
data = pd.read_csv('25test.csv')

# replace 0 values in Depth to avoid division by zero
data["Depth"] = data["Depth"].replace(0, 1e-6)

# recalc h max
data["hmax"] = data["Width"] * np.sqrt(1 - thickness/(thickness + data["Depth"]))

# ensure numeric
numeric_cols = ["Depth", "Width", "Power", "Speed", "hmax"]
if "Length" in data.columns:
    numeric_cols.append("Length")
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# read in data
hmax = data["hmax"].to_numpy(dtype=dtype)
power = data["Power"].to_numpy(dtype=dtype)
speed = data["Speed"].to_numpy(dtype=dtype)
edensity = data["P/v"].to_numpy(dtype=dtype)
depth = data["Depth"].to_numpy(dtype=dtype)
width = data["Width"].to_numpy(dtype=dtype)
length = data["Length"].to_numpy(dtype=dtype) if "Length" in data.columns else None

# coordinates
x = data["Speed"].to_numpy(dtype=dtype)
y = data["Power"].to_numpy(dtype=dtype)
z_hmax = data["hmax"].to_numpy(dtype=dtype)

# energy density thresholds
edensity_low = 0
edensity_high = 1e8

# defect classification
def classify_defect(width, depth, e, length=None, thickness=10, ed_low=edensity_low, ed_high=edensity_high):
    if e < ed_low:
        return "Lack of Fusion"
    elif e > ed_high:
        return "Keyhole"
    if width/depth < 1.5:
        return "Keyhole"
    elif depth/thickness < 1.9:
        return "Lack of Fusion"
    if USE_BALLING and (length is not None):
        if width/length < 0.23:
            return "Balling"
    return "Good"

def classify_grid(width_grid, depth_grid, thickness, ed_grid, length_grid=None):
    result = np.empty_like(width_grid, dtype=object)
    for i in range(width_grid.shape[0]):
        for j in range(width_grid.shape[1]):
            w = width_grid[i,j]
            d = depth_grid[i,j]
            e = ed_grid[i,j]
            l = length_grid[i,j] if length_grid is not None else None
            if np.isnan(w) or np.isnan(d) or np.isnan(e) or (USE_BALLING and length_grid is not None and np.isnan(l)):
                result[i,j] = "Good"
            else:
                result[i,j] = classify_defect(w, d, e, l, thickness)
    return result

# make grids
def make_grids():
    if USE_INTERPOLATION:
        grid_x = np.linspace(speed.min(), speed.max(), 200)
        grid_y = np.linspace(power.min(), power.max(), 200)
        xi, yi = np.meshgrid(grid_x, grid_y)
        width_grid = griddata((x, y), width, (xi, yi), method='cubic')
        depth_grid = griddata((x, y), depth, (xi, yi), method='cubic')
        ed_grid = griddata((x, y), edensity, (xi, yi), method='cubic')
        length_grid = griddata((x, y), length, (xi, yi), method='cubic') if length is not None else None
        z_hmax_grid = griddata((x, y), z_hmax, (xi, yi), method='cubic')
        width_grid = gaussian_filter(width_grid, sigma=sigma)
        depth_grid = gaussian_filter(depth_grid, sigma=sigma)
        ed_grid = gaussian_filter(ed_grid, sigma=sigma)
        if length_grid is not None:
            length_grid = gaussian_filter(length_grid, sigma=sigma)
    else:
        # build a raw rectangular grid from data
        grid_x = np.sort(data["Speed"].unique())
        grid_y = np.sort(data["Power"].unique())
        xi, yi = np.meshgrid(grid_x, grid_y)

        pivot = data.pivot(index="Power", columns="Speed", values="Width")
        width_grid = pivot.values

        pivot = data.pivot(index="Power", columns="Speed", values="Depth")
        depth_grid = pivot.values

        pivot = data.pivot(index="Power", columns="Speed", values="P/v")
        ed_grid = pivot.values

        if length is not None:
            pivot = data.pivot(index="Power", columns="Speed", values="Length")
            length_grid = pivot.values
        else:
            length_grid = None

        pivot = data.pivot(index="Power", columns="Speed", values="hmax")
        z_hmax_grid = pivot.values

    return xi, yi, width_grid, depth_grid, ed_grid, length_grid, z_hmax_grid

xi, yi, width_grid, depth_grid, ed_grid, length_grid, z_hmax_grid = make_grids()

# label grid
defect_grid = classify_grid(width_grid, depth_grid, thickness, ed_grid, length_grid)

# map defects to colors
mapping = {"Keyhole": 0, "Lack of Fusion": 1, "Good": 2, "Balling":3}
defect_numeric_grid = np.vectorize(mapping.get)(defect_grid)

# RGB grid
rgb_grid = np.ones(defect_numeric_grid.shape + (3,), dtype=float)
red = np.array(mcolors.to_rgb("#E07B7B"))
blue = np.array(mcolors.to_rgb("#7bbfc8"))
green = np.array(mcolors.to_rgb("#289C8E"))
alpha_defects = 1.0
for i in range(3):
    rgb_grid[defect_numeric_grid == 0, i] = alpha_defects*red[i] + (1-alpha_defects)*1
    rgb_grid[defect_numeric_grid == 1, i] = alpha_defects*blue[i] + (1-alpha_defects)*1
    rgb_grid[defect_numeric_grid == 3, i] = alpha_defects*green[i] + (1-alpha_defects)*1

# plot
plt.figure(figsize=(10,7))
if PLOT_MODE == "INTERPOLATED":
    plt.imshow(rgb_grid, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", aspect="auto", zorder=1)
    levels = [5] + list(range(25, int(np.ceil(z_hmax.max())) + 1, 25))
    contours = plt.contour(xi, yi, z_hmax_grid, levels=levels, colors='black', linewidths=1, zorder=2)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%d')
elif PLOT_MODE == "RAW_POINTS":
    color_map = {"Keyhole": "#E07B7B", "Lack of Fusion": "#7bbfc8", "Good": "#FFFFFF", "Balling": "#289C8E"}
    colors = [color_map[d[0]] for d in defect_grid] if defect_grid.ndim==2 else [color_map[d] for d in defect_grid.flatten()]
    plt.scatter(x, y, c=colors, s=50, edgecolors='k')
elif PLOT_MODE == "OVERLAY_POINTS":
    # always plot background map
    plt.imshow(rgb_grid,
               extent=[xi.min(), xi.max(), yi.min(), yi.max()],
               origin="lower",
               aspect="auto",
               zorder=1)

    # always add contours (interpolated or pivot-based)
    levels = [5] + list(range(25, int(np.ceil(z_hmax_grid[~np.isnan(z_hmax_grid)].max())) + 1, 25))
    contours = plt.contour(xi, yi, z_hmax_grid, levels=levels, colors='black', linewidths=1, zorder=2)
    plt.clabel(contours, inline=True, fontsize=8, fmt='%d')

    # then scatter raw experimental points
    color_map = {"Keyhole": "#E07B7B",
                 "Lack of Fusion": "#7bbfc8",
                 "Good": "#FFFFFF",
                 "Balling": "#289C8E"}
    raw_defects = classify_grid(width.reshape(-1,1), depth.reshape(-1,1), thickness,
                                edensity.reshape(-1,1),
                                length.reshape(-1,1) if length is not None else None)
    colors = [color_map[d[0]] for d in raw_defects]
    plt.scatter(x, y, c=colors, s=50, edgecolors='k', zorder=3)

plt.xlabel("Scan Velocity (mm/s)")
plt.ylabel("Laser Power (W)")
plt.title("316L Printability Map")
legend_elements = [
    Patch(facecolor="#E07B7B", label="Keyhole W/D < 1.5"),
    Patch(facecolor="#7bbfc8", label="Lack of Fusion D/t <1.9"),
]

if USE_BALLING:
    legend_elements.append(Patch(facecolor="#289C8E", label="Balling W/L<0.23"))

legend_elements.append(Patch(facecolor="#FFFFFF", edgecolor="black", label="Stable/Printable"))
plt.legend(handles=legend_elements, loc="best")
plt.legend(handles=legend_elements, loc="best")
plt.show()