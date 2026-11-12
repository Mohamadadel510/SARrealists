"""
AOI-to-UTM window transformation, windowed HDF5 reads, layover/shadow masking,
linear-power boxcar multilooking, coordinate alignment, and reference/chaining math.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from pyproj import Transformer

from nisar_pipeline.io_nisar import UnsupportedGranuleError
from nisar_pipeline.layers import (
    GCOV_GRID_GROUP,
    GCOV_INCIDENCE_ANGLE,
    GCOV_MASK,
    GCOV_POLARIZATIONS,
    GCOV_PROJECTION,
    GCOV_RADAR_GRID_X,
    GCOV_RADAR_GRID_Y,
    GCOV_X,
    GCOV_Y,
)

logger = logging.getLogger(__name__)


@dataclass
class GCOVWindow:
    """Container for a multilooked GCOV AOI window in native UTM coordinates."""

    pols: Dict[str, np.ndarray]  # Linear power (float32, NaN nodata)
    x: np.ndarray  # 1D UTM Easting centers (float64)
    y: np.ndarray  # 1D UTM Northing centers (float64)
    epsg: int
    mask: Optional[np.ndarray] = None  # Multilooked validity mask (uint8)
    incidence_angle: Optional[np.ndarray] = None  # Degrees (float32)
    track: Optional[int] = None
    frame: Optional[int] = None
    direction: Optional[str] = None
    time: Optional[str] = None
    date_str: Optional[str] = None
    file_name: Optional[str] = None
    available_pols: List[str] = field(default_factory=list)
    missing_pols: List[str] = field(default_factory=list)
    has_dual_pol: bool = False


def read_epsg_from_projection(proj_ds: h5py.Dataset) -> int:
    """
    Read the UTM EPSG code from a NISAR `projection` dataset,
    falling back to its `epsg_code` attribute.
    """
    try:
        val = int(np.array(proj_ds[()]).item())
        if val > 0:
            return val
    except Exception:
        pass
    if "epsg_code" in proj_ds.attrs:
        return int(np.array(proj_ds.attrs["epsg_code"]).item())
    raise RuntimeError(
        "Unable to determine EPSG code from projection dataset or its 'epsg_code' attribute."
    )


def bbox_to_utm_slices(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    bbox_lonlat: Sequence[float],
    epsg: int,
) -> Tuple[slice, slice, Tuple[float, float, float, float]]:
    """
    Convert an EPSG:4326 bounding box (W, S, E, N) into pixel index slices
    `(slice_y, slice_x)` on the granule's 1D UTM coordinate arrays.
    """
    w, s, e, n = [float(v) for v in bbox_lonlat]
    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    sample_pts = [
        (w, s),
        (e, s),
        (e, n),
        (w, n),
        (0.5 * (w + e), s),
        (0.5 * (w + e), n),
        (w, 0.5 * (s + n)),
        (e, 0.5 * (s + n)),
    ]
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in sample_pts])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    ix = np.where((x_coords >= xmin) & (x_coords <= xmax))[0]
    iy = np.where((y_coords >= ymin) & (y_coords <= ymax))[0]

    if len(ix) == 0 or len(iy) == 0:
        raise UnsupportedGranuleError(
            f"AOI bbox {bbox_lonlat} (UTM [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}] "
            f"in EPSG:{epsg}) is outside this granule's footprint "
            f"(X=[{x_coords.min():.1f}, {x_coords.max():.1f}], "
            f"Y=[{y_coords.min():.1f}, {y_coords.max():.1f}])."
        )

    slice_y = slice(int(iy.min()), int(iy.max()) + 1)
    slice_x = slice(int(ix.min()), int(ix.max()) + 1)
    return slice_y, slice_x, (xmin, ymin, xmax, ymax)


def block_mean(arr: np.ndarray, looks: int) -> np.ndarray:
    """
    2D boxcar multilook (nanmean) in linear power.
    Returns float32 array with NaN where all pixels in a look window are NaN.
    """
    arr_f32 = arr.astype(np.float32, copy=False)
    if looks <= 1:
        return arr_f32
    h = (arr_f32.shape[0] // looks) * looks
    w = (arr_f32.shape[1] // looks) * looks
    if h == 0 or w == 0:
        raise ValueError(
            f"Window shape {arr_f32.shape} is smaller than multilook factor {looks}."
        )
    reshaped = arr_f32[:h, :w].reshape(h // looks, looks, w // looks, looks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(reshaped, axis=(1, 3)).astype(np.float32)
    return out


def block_mask_nearest(mask_arr: np.ndarray, looks: int) -> np.ndarray:
    """
    Downsample a discrete uint8 mask using top-left / nearest-neighbor sampling.
    """
    if looks <= 1:
        return mask_arr.astype(np.uint8, copy=False)
    h = (mask_arr.shape[0] // looks) * looks
    w = (mask_arr.shape[1] // looks) * looks
    return mask_arr[:h:looks, :w:looks].astype(np.uint8, copy=False)


def coord_mean(coords: np.ndarray, looks: int) -> np.ndarray:
    """Downsample 1D coordinate center array to match `block_mean` dimensions."""
    c = np.asarray(coords, dtype=np.float64)
    if looks <= 1:
        return c
    k = (len(c) // looks) * looks
    return c[:k].reshape(-1, looks).mean(axis=1)


def is_valid_mask_array(
    mask_arr: np.ndarray,
    invalid_values: Sequence[int] = (0, 255),
) -> np.ndarray:
    """
    Build a boolean validity mask from NISAR's integer `mask` dataset.
    Pixels matching `invalid_values` (default 0=outside swath, 255=fill/layover/shadow)
    are marked False.
    """
    valid = np.ones(mask_arr.shape, dtype=bool)
    for val in invalid_values:
        valid &= mask_arr != val
    return valid


def _read_interpolated_incidence_angle(
    h5: h5py.File,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Read `/science/LSAR/GCOV/metadata/radarGrid/incidenceAngle` and bilinearly
    interpolate the near-surface height slice onto `(target_y, target_x)`.
    Returns None if the dataset is absent.
    """
    if GCOV_INCIDENCE_ANGLE not in h5:
        return None
    try:
        inc_ds = h5[GCOV_INCIDENCE_ANGLE]
        rx = h5[GCOV_RADAR_GRID_X][:]
        ry = h5[GCOV_RADAR_GRID_Y][:]
        raw = inc_ds[:]
        if raw.ndim == 3:
            slice_2d = np.asarray(raw[raw.shape[0] // 2, :, :], dtype=np.float32)
        elif raw.ndim == 2:
            slice_2d = np.asarray(raw, dtype=np.float32)
        else:
            return None

        if rx[0] > rx[-1]:
            rx = rx[::-1]
            slice_2d = slice_2d[:, ::-1]
        if ry[0] > ry[-1]:
            ry = ry[::-1]
            slice_2d = slice_2d[::-1, :]

        interp_x = np.empty((slice_2d.shape[0], len(target_x)), dtype=np.float32)
        for row_idx in range(slice_2d.shape[0]):
            interp_x[row_idx, :] = np.interp(
                target_x, rx, slice_2d[row_idx, :], left=np.nan, right=np.nan
            )
        out = np.empty((len(target_y), len(target_x)), dtype=np.float32)
        for col_idx in range(len(target_x)):
            out[:, col_idx] = np.interp(
                target_y, ry, interp_x[:, col_idx], left=np.nan, right=np.nan
            )
        return out
    except Exception as exc:
        logger.debug("Could not interpolate incidenceAngle metadata cube: %s", exc)
        return None


def load_gcov_window(
    h5: h5py.File,
    bbox_lonlat: Sequence[float],
    looks: int = 2,
    invalid_mask_values: Sequence[int] = (0, 255),
    track: Optional[int] = None,
    frame: Optional[int] = None,
    direction: Optional[str] = None,
    time: Optional[str] = None,
    date_str: Optional[str] = None,
    file_name: Optional[str] = None,
) -> GCOVWindow:
    """
    Stream only the AOI window from a NISAR GCOV HDF5 file, mask invalid/nonpositive
    and layover/shadow pixels in linear power, and multilook.

    Falls back to `/science/LSAR/GCOV/grids/frequencyB` if `frequencyA` is absent, or
    raises `UnsupportedGranuleError` immediately without retrying.
    """
    grid_path = GCOV_GRID_GROUP
    if grid_path not in h5:
        alt_grid_path = "/science/LSAR/GCOV/grids/frequencyB"
        if alt_grid_path in h5:
            logger.warning(
                "Granule %s is missing '%s'; falling back to '%s'.",
                file_name,
                grid_path,
                alt_grid_path,
            )
            grid_path = alt_grid_path
        else:
            raise UnsupportedGranuleError(
                f"Granule {file_name} has neither '{GCOV_GRID_GROUP}' nor '{alt_grid_path}'."
            )

    g = h5[grid_path]
    x_full = g[GCOV_X.split("/")[-1]][:]
    y_full = g[GCOV_Y.split("/")[-1]][:]
    epsg = read_epsg_from_projection(g[GCOV_PROJECTION.split("/")[-1]])

    slice_y, slice_x, _ = bbox_to_utm_slices(x_full, y_full, bbox_lonlat, epsg)

    valid_pixel_mask: Optional[np.ndarray] = None
    mask_multilooked: Optional[np.ndarray] = None
    mask_name = GCOV_MASK.split("/")[-1]
    if mask_name in g:
        raw_mask = np.asarray(g[mask_name][slice_y, slice_x], dtype=np.uint8)
        valid_pixel_mask = is_valid_mask_array(raw_mask, invalid_mask_values)
        mask_multilooked = block_mask_nearest(raw_mask, looks)

    pols: Dict[str, np.ndarray] = {}
    available_pols: List[str] = []
    missing_pols: List[str] = []

    for pol_name in GCOV_POLARIZATIONS:
        if pol_name in g:
            arr = np.asarray(g[pol_name][slice_y, slice_x], dtype=np.float32)
            bad = ~np.isfinite(arr) | (arr <= 0.0)
            if valid_pixel_mask is not None:
                bad |= ~valid_pixel_mask
            arr[bad] = np.nan
            ml_arr = block_mean(arr, looks)
            if np.any(np.isfinite(ml_arr)):
                pols[pol_name] = ml_arr
                available_pols.append(pol_name)
            else:
                missing_pols.append(pol_name)
        else:
            missing_pols.append(pol_name)

    if not pols:
        raise UnsupportedGranuleError(
            f"No valid GCOV polarization layers found in AOI window for {file_name}. "
            f"Group keys: {list(g.keys())}"
        )

    x_win = coord_mean(x_full[slice_x], looks)
    y_win = coord_mean(y_full[slice_y], looks)
    inc_win = _read_interpolated_incidence_angle(h5, x_win, y_win)

    has_co = ("HHHH" in pols) or ("VVVV" in pols)
    has_cross = ("HVHV" in pols) or ("VHVH" in pols)

    return GCOVWindow(
        pols=pols,
        x=x_win,
        y=y_win,
        epsg=epsg,
        mask=mask_multilooked,
        incidence_angle=inc_win,
        track=track,
        frame=frame,
        direction=direction,
        time=time,
        date_str=date_str,
        file_name=file_name,
        available_pols=available_pols,
        missing_pols=missing_pols,
        has_dual_pol=bool(has_co and has_cross),
    )


def subtract_reference_window(
    arr: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    ref_bbox_lonlat: Sequence[float],
    epsg: int,
) -> Tuple[np.ndarray, float]:
    """
    Reference a displacement or phase grid to the median of a stable reference box
    `(west, south, east, north)` in EPSG:4326.
    """
    slice_y, slice_x, _ = bbox_to_utm_slices(x_coords, y_coords, ref_bbox_lonlat, epsg)
    patch = arr[slice_y, slice_x]
    finite_vals = patch[np.isfinite(patch)]
    if finite_vals.size == 0:
        raise ValueError(
            f"Reference box {ref_bbox_lonlat} contains only NaN values; "
            "choose a higher-coherence reference region."
        )
    ref_val = float(np.median(finite_vals))
    referenced = (arr - ref_val).astype(np.float32)
    return referenced, ref_val


def chain_pairwise_displacements(
    pair_displacements: Sequence[np.ndarray],
    propagate_nan: bool = False,
) -> np.ndarray:
    """
    Compute a cumulative displacement stack from nearest-neighbor sequential pairs
    (e.g., 12-day interferograms: [d(t0->t1), d(t1->t2), ..., d(t_{k-1}->t_k)]).
    """
    if not pair_displacements:
        raise ValueError("pair_displacements cannot be empty.")

    stack = np.stack([np.asarray(p, dtype=np.float32) for p in pair_displacements], axis=0)
    n_pairs, h, w = stack.shape
    cum = np.zeros((n_pairs + 1, h, w), dtype=np.float32)

    all_nan_mask = np.all(~np.isfinite(stack), axis=0)
    cum[0, all_nan_mask] = np.nan

    if propagate_nan:
        for i in range(n_pairs):
            cum[i + 1] = cum[i] + stack[i]
    else:
        running = np.zeros((h, w), dtype=np.float32)
        for i in range(n_pairs):
            step = stack[i]
            valid = np.isfinite(step)
            running = np.where(valid, running + step, running)
            out_slice = running.copy()
            out_slice[~valid] = np.nan
            cum[i + 1] = out_slice

    return cum