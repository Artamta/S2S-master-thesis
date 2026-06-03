import logging
import os
import sys
from typing import List, Optional

import xarray as xr

# ==== USER CONFIGURATION ====
ROOT = "/home/bedartha/"
STORE = "public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

# Train / Val / Test splits (edit dates as needed)
SPLITS = {
    "train_data_full.nc": ("1979-01-01T00:00", "2015-12-31T18:00"),  # 37 years
    "val_data_full.nc":   ("2016-01-01T00:00", "2019-12-31T18:00"),  # 4 years
    "test_data_full.nc":  ("2020-01-01T00:00", "2022-12-31T18:00"),  # 3 years
}

TIME_SKIP: Optional[int] = None   # e.g., 2 keeps every 2nd 6h step; None keeps all
LAT_SKIP: Optional[int] = None    # e.g., 2 subsamples latitude; None keeps all
LON_SKIP: Optional[int] = None    # e.g., 2 subsamples longitude; None keeps all
LOG_LEVEL = "INFO"
CHUNKS = {"time": 128}            # xarray chunking; adjust or set {} for auto
# ============================

PRESSURE_VARS: List[str] = [
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
PRESSURE_LEVELS = [850, 500, 250]
SURFACE_VARS: List[str] = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> None:
    configure_logging(LOG_LEVEL)

    store_path = STORE if os.path.isabs(STORE) else os.path.join(ROOT, STORE)
    if not os.path.exists(store_path):
        logging.error("Zarr store not found: %s", store_path)
        sys.exit(1)

    logging.info("Opening Zarr store: %s", store_path)
    try:
        ds = xr.open_zarr(store_path, consolidated=True, chunks=CHUNKS)
    except Exception:
        logging.exception("Failed to open Zarr store.")
        sys.exit(1)

    rename_dims = {}
    if "latitude" in ds.dims:
        rename_dims["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_dims["longitude"] = "lon"
    ds = ds.rename(rename_dims)

    missing_pressure = [v for v in PRESSURE_VARS if v not in ds]
    missing_surface = [v for v in SURFACE_VARS if v not in ds]
    if missing_pressure or missing_surface:
        logging.warning("Missing vars -> pressure:%s surface:%s", missing_pressure, missing_surface)

    present_vars = [v for v in PRESSURE_VARS + SURFACE_VARS if v in ds]
    if not present_vars:
        logging.error("No requested variables available; aborting.")
        sys.exit(1)

    ds = ds[present_vars]

    if "level" in ds.dims:
        available_levels = set(ds.coords["level"].values.tolist())
        desired_levels = [lvl for lvl in PRESSURE_LEVELS if lvl in available_levels]
        missing_levels = sorted(set(PRESSURE_LEVELS) - available_levels)
        if missing_levels:
            logging.warning("Dropping unavailable levels: %s", missing_levels)
        if not desired_levels:
            logging.error("No requested pressure levels available; aborting.")
            sys.exit(1)
        ds = ds.sel(level=desired_levels)

    ds = ds.isel(
        lat=slice(None, None, LAT_SKIP),
        lon=slice(None, None, LON_SKIP),
    )

    for output_file, (start, end) in SPLITS.items():
        logging.info("Processing split: %s (%s → %s)", output_file, start, end)
        subset = ds.sel(time=slice(start, end))

        if TIME_SKIP and TIME_SKIP > 1:
            subset = subset.isel(time=slice(None, None, TIME_SKIP))

        subset = subset.astype("float32")

        try:
            subset.to_netcdf(output_file)
            logging.info(
                "Saved: %s (time=%d, level=%d, lat=%d, lon=%d, vars=%d)",
                output_file,
                subset.sizes.get("time", 0),
                subset.sizes.get("level", 0),
                subset.sizes.get("lat", 0),
                subset.sizes.get("lon", 0),
                len(subset.data_vars),
            )
        except Exception:
            logging.exception("Failed to write %s.", output_file)
            sys.exit(1)

    logging.info("All splits saved successfully.")


if __name__ == "__main__":
    main()