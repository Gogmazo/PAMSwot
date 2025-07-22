import math, pathlib, numpy as np, xarray as xr, matplotlib.pyplot as plt
from tqdm.auto import tqdm

DATA_ROOT = pathlib.Path("/scratch/tm3076/pytorch_learning_tiles/HRS_SST_tiles")
SUBDIRS   = ["agg_cloud_masks", "agg_cloud_percentages"]
SST_VAR   = "sst_filtered_q5"

for sub in SUBDIRS:
    files = sorted((DATA_ROOT / sub).glob("*.nc"))
    if not files:
        continue

    cols = 5
    rows = math.ceil(len(files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.ravel()

    for idx, nc in enumerate(tqdm(files, desc=sub)):
        with xr.open_dataset(nc, engine="netcdf4") as ds:
            da = ds[SST_VAR].isel(time=0) if "time" in ds.dims else ds[SST_VAR]
            cm = plt.cm.viridis.copy()
            cm.set_bad("lightgray")
            axes[idx].imshow(np.ma.masked_invalid(da.values), cmap=cm, origin="lower")
            axes[idx].set_title(nc.stem, fontsize=7)
            axes[idx].axis("off")

    for ax in axes[len(files):]:
        ax.remove()

    plt.tight_layout()
    out_png = DATA_ROOT.parent / f"{sub}_mosaic.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("saved →", out_png)
