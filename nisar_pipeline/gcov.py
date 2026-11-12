"""
NISAR L2 GCOV science processing:
- Linear power to dB conversion (strictly after multilooking/reprojection)
- Co/cross polarization selection and HH-only cycle fallback
- Dual-pol Radar Vegetation Index (RVI = 4 * HV / (HH + HV))
- Stretched RGB composite generation
- Same-track/frame/direction pairwise change detection
- Incremental GeoTIFF (COG), Zarr stack, point time series, and catalog generation
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
import xarray as xr

from nisar_pipeline.window import GCOVWindow

logger = logging.getLogger(__name__)


def linear_to_db(linear_arr: np.ndarray) -> np.ndarray:
    """
    Convert linear power array to decibels: 10 * log10(linear).
    Non-finite and non-positive values (<= 0) are mapped to NaN.
    Output is always float32.
    """
    arr = np.asarray(linear_arr, dtype=np.float32)
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0.0)
    out[valid] = 10.0 * np.log10(arr[valid])
    return out


def resolve_co_cross_pols(
    pols: Dict[str, np.ndarray],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify primary co-pol (HHHH preferred over VVVV) and cross-pol (HVHV preferred over VHVH).
    Do NOT raise if cross-pol is missing; return (co_key, None) so callers can fall back
    to HH-only products and record the polarization gap.
    """
    co = "HHHH" if "HHHH" in pols else ("VVVV" if "VVVV" in pols else None)
    cross = "HVHV" if "HVHV" in pols else ("VHVH" if "VHVH" in pols else None)
    return co, cross


def compute_rvi(
    co_linear: np.ndarray,
    cross_linear: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute Dual-Pol Radar Vegetation Index in linear power:
        RVI = clip(4 * HV / (HH + HV), 0.0, 1.0)
    where 0 = bare surface and 1 = dense volume scattering.
    If cross_linear is None (e.g., HH-only cycle), returns an all-NaN float32 array.
    """
    co = np.asarray(co_linear, dtype=np.float32)
    out = np.full(co.shape, np.nan, dtype=np.float32)
    if cross_linear is None:
        return out

    cr = np.asarray(cross_linear, dtype=np.float32)
    denom = co + cr
    valid = np.isfinite(co) & np.isfinite(cr) & (co > 0.0) & (cr > 0.0) & (denom > 0.0)
    out[valid] = np.clip((4.0 * cr[valid]) / denom[valid], 0.0, 1.0)
    return out


def compute_pol_ratio_db(
    co_linear: np.ndarray,
    cross_linear: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute cross-pol to co-pol ratio in linear power (HV / HH), then convert to dB.
    Returns all-NaN float32 if cross_linear is None.
    """
    co = np.asarray(co_linear, dtype=np.float32)
    if cross_linear is None:
        return np.full(co.shape, np.nan, dtype=np.float32)
    cr = np.asarray(cross_linear, dtype=np.float32)
    ratio_lin = np.full(co.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(co) & np.isfinite(cr) & (co > 0.0) & (cr > 0.0)
    ratio_lin[valid] = cr[valid] / co[valid]
    return linear_to_db(ratio_lin)


def _stretch_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Linearly scale float array into [1, 255] uint8 with 0 reserved for NaN nodata."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    scaled = (arr[valid] - vmin) / max(1e-6, (vmax - vmin))
    clipped = np.clip(scaled, 0.0, 1.0)
    out[valid] = np.round(clipped * 254.0 + 1.0).astype(np.uint8)
    return out


def compute_rgb_composite(
    co_db: np.ndarray,
    cross_db: Optional[np.ndarray],
    rvi_arr: Optional[np.ndarray] = None,
    co_range: Tuple[float, float] = (-22.0, 0.0),
    cross_range: Tuple[float, float] = (-30.0, -8.0),
    rvi_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Build a 3-band (3, H, W) uint8 SAR RGB composite:
      R = Co-pol dB (HH)
      G = Cross-pol dB (HV)  [or Co-pol if single-pol cycle]
      B = RVI (or Cross - Co dB difference)
    """
    r_band = _stretch_uint8(co_db, co_range[0], co_range[1])
    if cross_db is None or not np.any(np.isfinite(cross_db)):
        return np.stack([r_band, r_band, r_band], axis=0)

    g_band = _stretch_uint8(cross_db, cross_range[0], cross_range[1])
    if rvi_arr is not None and np.any(np.isfinite(rvi_arr)):
        b_band = _stretch_uint8(rvi_arr, rvi_range[0], rvi_range[1])
    else:
        diff_db = cross_db - co_db
        b_band = _stretch_uint8(diff_db, -15.0, -2.0)
    return np.stack([r_band, g_band, b_band], axis=0)


def compute_single_date_products(window: GCOVWindow) -> Dict[str, Any]:
    """
    Compute all single-date GCOV derived layers from a multilooked linear-power window.
    Never interpolates missing polarizations.
    """
    co_key, cross_key = resolve_co_cross_pols(window.pols)
    if co_key is None:
        raise RuntimeError(
            f"Cannot compute GCOV products without a co-pol layer; available={window.available_pols}"
        )

    co_lin = window.pols[co_key]
    cross_lin = window.pols.get(cross_key) if cross_key else None

    hh_db = linear_to_db(co_lin)
    if cross_lin is not None:
        hv_db = linear_to_db(cross_lin)
        rvi_arr = compute_rvi(co_lin, cross_lin)
        pol_ratio_db = compute_pol_ratio_db(co_lin, cross_lin)
        polarization_flag = "DUAL_POL"
    else:
        logger.warning(
            "Granule %s (%s) is missing cross-pol (%s); recording NaN gap and falling back to %s-only.",
            window.file_name,
            window.time,
            window.missing_pols,
            co_key,
        )
        hv_db = np.full(co_lin.shape, np.nan, dtype=np.float32)
        rvi_arr = np.full(co_lin.shape, np.nan, dtype=np.float32)
        pol_ratio_db = np.full(co_lin.shape, np.nan, dtype=np.float32)
        polarization_flag = f"{co_key}_ONLY_GAP"

    rgb = compute_rgb_composite(hh_db, hv_db if cross_lin is not None else None, rvi_arr)

    return {
        "co_pol_name": co_key,
        "cross_pol_name": cross_key,
        "polarization_flag": polarization_flag,
        "has_dual_pol": cross_lin is not None,
        "layers": {
            "hh_db": hh_db,
            "hv_db": hv_db,
            "rvi": rvi_arr,
            "pol_ratio_db": pol_ratio_db,
        },
        "rgb": rgb,
        "x": window.x,
        "y": window.y,
        "epsg": window.epsg,
        "time": window.time,
        "date_str": window.date_str,
        "track": window.track,
        "frame": window.frame,
        "direction": window.direction,
    }


def validate_same_geometry(pre: GCOVWindow, post: GCOVWindow) -> None:
    """
    Enforce science rule: comparisons and stacks are ONLY allowed within the
    same (pathNumber, frameNumber, flightDirection) and UTM EPSG code.
    """
    if (pre.track, pre.frame, pre.direction) != (post.track, post.frame, post.direction):
        raise ValueError(
            f"Cannot compare granules from different track/frame/direction groups: "
            f"pre=({pre.track}, {pre.frame}, {pre.direction}) vs "
            f"post=({post.track}, {post.frame}, {post.direction})."
        )
    if pre.epsg != post.epsg:
        raise ValueError(
            f"Cannot compare granules with different EPSG codes: {pre.epsg} vs {post.epsg}."
        )


def align_windows(
    pre: GCOVWindow, post: GCOVWindow
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Align two same-track GCOVWindows onto their shared UTM (x, y) coordinate grid.
    """
    validate_same_geometry(pre, post)

    # Intersect 1D coordinate arrays by rounding to 0.01 m to avoid float jitter
    pre_x_r = np.round(pre.x, 2)
    post_x_r = np.round(post.x, 2)
    common_x, ix_pre, ix_post = np.intersect1d(
        pre_x_r, post_x_r, return_indices=True
    )

    pre_y_r = np.round(pre.y, 2)
    post_y_r = np.round(post.y, 2)
    common_y, iy_pre, iy_post = np.intersect1d(
        pre_y_r, post_y_r, return_indices=True
    )

    if len(common_x) == 0 or len(common_y) == 0:
        raise RuntimeError("Pre and post windows have zero overlapping UTM coordinates.")

    # Preserve original Y ordering (typically North-to-South descending in UTM)
    if pre.y[0] > pre.y[-1]:
        order_y = np.argsort(pre.y[iy_pre])[::-1]
    else:
        order_y = np.argsort(pre.y[iy_pre])
    iy_pre = iy_pre[order_y]
    iy_post = iy_post[order_y]

    order_x = np.argsort(pre.x[ix_pre])
    ix_pre = ix_pre[order_x]
    ix_post = ix_post[order_x]

    pre_aligned = {
        k: v[np.ix_(iy_pre, ix_pre)] for k, v in pre.pols.items()
    }
    post_aligned = {
        k: v[np.ix_(iy_post, ix_post)] for k, v in post.pols.items()
    }
    return pre_aligned, post_aligned, pre.x[ix_pre], pre.y[iy_pre]


def compute_pairwise_change(
    pre: GCOVWindow, post: GCOVWindow
) -> Dict[str, Any]:
    """
    Compute pairwise GCOV change between two dates on the same track/frame/direction:
      - `hv_db_change` = post_hv_db - pre_hv_db (negative = vegetation / volume loss)
      - `hh_db_change` = post_hh_db - pre_hh_db
      - `rvi_change`   = post_rvi - pre_rvi
    Falls back cleanly to `hh_db_change` if either acquisition is missing cross-pol.
    """
    pre_pols, post_pols, x_common, y_common = align_windows(pre, post)
    co_pre, cr_pre = resolve_co_cross_pols(pre_pols)
    co_post, cr_post = resolve_co_cross_pols(post_pols)

    if co_pre is None or co_post is None:
        raise RuntimeError("Both pre and post windows require a valid co-pol layer.")

    hh_db_pre = linear_to_db(pre_pols[co_pre])
    hh_db_post = linear_to_db(post_pols[co_post])
    hh_db_change = (hh_db_post - hh_db_pre).astype(np.float32)

    if cr_pre is not None and cr_post is not None:
        hv_db_pre = linear_to_db(pre_pols[cr_pre])
        hv_db_post = linear_to_db(post_pols[cr_post])
        hv_db_change = (hv_db_post - hv_db_pre).astype(np.float32)

        rvi_pre = compute_rvi(pre_pols[co_pre], pre_pols[cr_pre])
        rvi_post = compute_rvi(post_pols[co_post], post_pols[cr_post])
        rvi_change = (rvi_post - rvi_pre).astype(np.float32)
        cross_pol_gap = False
    else:
        logger.warning(
            "Cross-pol missing in pair (%s -> %s); setting hv_db_change and rvi_change "
            "to NaN and falling back to hh_db_change.",
            pre.date_str,
            post.date_str,
        )
        hv_db_change = np.full(hh_db_change.shape, np.nan, dtype=np.float32)
        rvi_change = np.full(hh_db_change.shape, np.nan, dtype=np.float32)
        cross_pol_gap = True

    return {
        "layers": {
            "hv_db_change": hv_db_change,
            "hh_db_change": hh_db_change,
            "rvi_change": rvi_change,
        },
        "x": x_common,
        "y": y_common,
        "epsg": pre.epsg,
        "pre_date": pre.date_str,
        "post_date": post.date_str,
        "cross_pol_gap": cross_pol_gap,
    }


def compute_affine_and_orient(
    arr: np.ndarray, x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Affine]:
    """
    Ensure raster array rows are ordered North-to-South (top-down) and columns
    West-to-East (left-to-right), returning `(arr_oriented, x_asc, y_desc, transform)`.
    """
    out = arr
    x_out = x.copy()
    y_out = y.copy()

    if len(x_out) > 1 and x_out[0] > x_out[-1]:
        x_out = x_out[::-1]
        out = out[:, ::-1]
    if len(y_out) > 1 and y_out[0] < y_out[-1]:
        y_out = y_out[::-1]
        out = out[::-1, :]

    dx = float(abs(x_out[1] - x_out[0])) if len(x_out) > 1 else 20.0
    dy = float(abs(y_out[1] - y_out[0])) if len(y_out) > 1 else 20.0
    west = float(x_out.min() - dx / 2.0)
    north = float(y_out.max() + dy / 2.0)
    transform = from_origin(west, north, dx, dy)
    return np.ascontiguousarray(out, dtype=np.float32), x_out, y_out, transform


def write_float32_cog(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    out_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a 2D float32 array with NaN nodata as a Cloud-Optimized / tiled GeoTIFF
    in native UTM projection.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data, _, _, transform = compute_affine_and_orient(arr, x, y)
    h, w = data.shape

    block_h = min(256, max(16, (h // 16) * 16)) if h >= 16 else h
    block_w = min(256, max(16, (w // 16) * 16)) if w >= 16 else w
    tiled = h >= 16 and w >= 16

    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": np.nan,
        "compress": "DEFLATE",
    }
    if tiled:
        profile.update({"tiled": True, "blockxsize": block_w, "blockysize": block_h})

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        if metadata:
            dst.update_tags(**{k: str(v) for k, v in metadata.items()})
        if h >= 64 and w >= 64:
            dst.build_overviews([2, 4], Resampling.bilinear)
            dst.update_tags(ns="rio_overview", resampling="bilinear")

    return out_path


def append_to_zarr_stack(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    date_str: str,
    layer_name: str,
    zarr_path: Path,
    attrs: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Incrementally write or append a single date's 2D float32 array to a
    `time x y x x` Zarr stack (`out/<site>/stack/<layer>.zarr`).
    Compatible with both Zarr v2 and Zarr v3 on Python 3.13.
    """
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    data, x_c, y_c, _ = compute_affine_and_orient(arr, x, y)

    # Use object dtype for string time coordinates so both Zarr v2 and Zarr v3 accept it
    time_coord = np.array([str(date_str)], dtype=object)

    ds_new = xr.Dataset(
        {
            layer_name: (
                ("time", "y", "x"),
                data[np.newaxis, :, :].astype(np.float32),
            )
        },
        coords={
            "time": time_coord,
            "y": y_c.astype(np.float64),
            "x": x_c.astype(np.float64),
        },
        attrs={"epsg": int(epsg), **(attrs or {})},
    )

    if not zarr_path.exists():
        ds_new.to_zarr(zarr_path, mode="w")
    else:
        existing = xr.open_zarr(zarr_path)
        existing_times = [str(t) for t in existing["time"].values.tolist()]
        if str(date_str) in existing_times:
            logger.info(
                "Date %s already present in %s; skipping append.", date_str, zarr_path
            )
            return zarr_path
        ds_aligned = ds_new.reindex(
            y=existing["y"].values,
            x=existing["x"].values,
            method="nearest",
            tolerance=50.0,
        )
        ds_aligned.to_zarr(zarr_path, mode="a", append_dim="time")

    return zarr_path


def sample_point_value(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    lon: float,
    lat: float,
    radius_px: int = 1,
) -> Optional[float]:
    """
    Extract median value around a (lon, lat) point from a native-UTM grid.
    Returns None if outside bounds or all-NaN (serialized as `null` in JSON).
    """
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    utmx, utmy = tr.transform(float(lon), float(lat))

    ix = int(np.argmin(np.abs(x - utmx)))
    iy = int(np.argmin(np.abs(y - utmy)))

    dx = abs(float(x[1] - x[0])) if len(x) > 1 else 20.0
    dy = abs(float(y[1] - y[0])) if len(y) > 1 else 20.0
    if abs(x[ix] - utmx) > 3 * dx or abs(y[iy] - utmy) > 3 * dy:
        return None

    y0 = max(0, iy - radius_px)
    y1 = min(arr.shape[0], iy + radius_px + 1)
    x0 = max(0, ix - radius_px)
    x1 = min(arr.shape[1], ix + radius_px + 1)
    patch = arr[y0:y1, x0:x1]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return None
    return round(float(np.median(finite)), 4)


def update_points_timeseries_json(
    points_cfg: Sequence[Dict[str, Any]],
    date_iso: str,
    layers: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    out_json_path: Path,
) -> Path:
    """
    Incrementally update `out/<site>/timeseries/points.json` with exact schema:
    [{"id","lon","lat","dates":[],"hv_db":[],"hh_db":[],"coh":[],"disp_mm":[],"sm":[]}]
    """
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    records_by_id: Dict[str, Dict[str, Any]] = {}

    if out_json_path.exists():
        try:
            existing_list = json.loads(out_json_path.read_text(encoding="utf-8"))
            for item in existing_list:
                records_by_id[str(item["id"])] = item
        except Exception as exc:
            logger.warning("Could not parse existing %s (%s); recreating.", out_json_path, exc)

    for pt in points_cfg:
        pid = str(pt["id"])
        lon = float(pt["lon"])
        lat = float(pt["lat"])
        if pid not in records_by_id:
            records_by_id[pid] = {
                "id": pid,
                "lon": lon,
                "lat": lat,
                "dates": [],
                "hv_db": [],
                "hh_db": [],
                "coh": [],
                "disp_mm": [],
                "sm": [],
            }
        rec = records_by_id[pid]
        if date_iso in rec["dates"]:
            idx = rec["dates"].index(date_iso)
        else:
            rec["dates"].append(date_iso)
            rec["hv_db"].append(None)
            rec["hh_db"].append(None)
            rec["coh"].append(None)
            rec["disp_mm"].append(None)
            rec["sm"].append(None)
            idx = len(rec["dates"]) - 1

        if "hv_db" in layers:
            rec["hv_db"][idx] = sample_point_value(layers["hv_db"], x, y, epsg, lon, lat)
        if "hh_db" in layers:
            rec["hh_db"][idx] = sample_point_value(layers["hh_db"], x, y, epsg, lon, lat)
        if "coherence" in layers:
            rec["coh"][idx] = sample_point_value(layers["coherence"], x, y, epsg, lon, lat)
        if "los_disp_mm" in layers:
            rec["disp_mm"][idx] = sample_point_value(layers["los_disp_mm"], x, y, epsg, lon, lat)
        if "soil_moisture" in layers:
            rec["sm"][idx] = sample_point_value(layers["soil_moisture"], x, y, epsg, lon, lat)

    out_list = list(records_by_id.values())
    out_json_path.write_text(json.dumps(out_list, indent=2), encoding="utf-8")
    return out_json_path