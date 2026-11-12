"""
NISAR L2 GUNW (Geocoded Unwrapped Interferogram) processing:
- Coherence thresholding, connected-component masking, and layover/shadow masking
- Optional ionospheric and tropospheric phase screen removal
- Unwrapped phase (radians) to Line-of-Sight (LOS) displacement (mm):
    LOS = -phase * lambda / (4 * pi), lambda = 0.24 m
  Sign convention (NISAR L2 GUNW Product Specification):
    If unwrapped phase = (phase_reference - phase_secondary) or range-change convention
    where positive phase corresponds to increased slant range (subsidence / motion away
    from radar), multiplying by -lambda / (4*pi) makes:
      * negative LOS displacement = motion away from satellite (subsidence)
      * positive LOS displacement = motion toward satellite (uplift)
    Verify the exact sign attribute on your CRID via `run_site.py inspect --products GUNW`.
- Stable reference pixel or bounding-box median subtraction
- Nearest-neighbor 12-day pair selection and chained cumulative displacement sum
- Vertical displacement estimate: vertical = LOS / cos(incidence_angle),
  strictly restricted to flat-terrain sites (e.g., Nile Delta)
- Robust per-pixel linear subsidence rate fit (Theil-Sen / IRLS in mm/yr)
  with a configurable minimum-number-of-dates check
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from nisar_pipeline.io_nisar import GranuleRecord, UnsupportedGranuleError
from nisar_pipeline.layers import (
    GUNW_COHERENCE,
    GUNW_CONNECTED_COMPONENTS,
    GUNW_INCIDENCE_ANGLE,
    GUNW_IONOSPHERE_PHASE,
    GUNW_MASK,
    GUNW_PROJECTION,
    GUNW_UNWRAPPED_PHASE,
    GUNW_X,
    GUNW_Y,
    require_verified_layer,
)
from nisar_pipeline.window import (
    bbox_to_utm_slices,
    block_mask_nearest,
    block_mean,
    chain_pairwise_displacements,
    coord_mean,
    is_valid_mask_array,
    read_epsg_from_projection,
    subtract_reference_window,
)

logger = logging.getLogger(__name__)

# NISAR L-band radar wavelength in meters (~1.257 GHz -> ~0.2384 m; spec nominal 0.24 m)
NISAR_L_BAND_WAVELENGTH_M: float = 0.24

# Regex extracting the two YYYYMMDDTHHMMSS timestamps in NISAR L2 filenames
_PAIR_TS_RE = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


@dataclass
class GUNWWindow:
    """Container for a multilooked GUNW AOI interferometric pair in native UTM."""

    unwrapped_phase: np.ndarray  # Radians (float32, NaN nodata)
    coherence: np.ndarray  # [0, 1] (float32, NaN nodata)
    x: np.ndarray  # 1D UTM Easting (float64)
    y: np.ndarray  # 1D UTM Northing (float64)
    epsg: int
    ref_date: str  # YYYYMMDD
    sec_date: str  # YYYYMMDD
    dt_days: float
    incidence_angle: Optional[np.ndarray] = None  # Degrees (float32)
    connected_components: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    track: Optional[int] = None
    frame: Optional[int] = None
    direction: Optional[str] = None
    file_name: Optional[str] = None


def parse_pair_dates(
    file_name: str,
    fallback_start: Optional[str] = None,
    fallback_stop: Optional[str] = None,
) -> Tuple[str, str, float]:
    """
    Extract (ref_date_YYYYMMDD, sec_date_YYYYMMDD, dt_days) from a NISAR GUNW/GOFF
    filename or metadata timestamps.
    """
    match = _PAIR_TS_RE.search(file_name)
    if match:
        t1_raw, t2_raw = match.group(1), match.group(2)
        dt1 = datetime.strptime(t1_raw, "%Y%m%dT%H%M%S")
        dt2 = datetime.strptime(t2_raw, "%Y%m%dT%H%M%S")
        ref_dt, sec_dt = (dt1, dt2) if dt1 <= dt2 else (dt2, dt1)
        dt_days = max(1e-3, (sec_dt - ref_dt).total_seconds() / 86400.0)
        return ref_dt.strftime("%Y%m%d"), sec_dt.strftime("%Y%m%d"), float(dt_days)

    if fallback_start and fallback_stop:
        s1 = fallback_start[:19].replace("-", "").replace(":", "")
        s2 = fallback_stop[:19].replace("-", "").replace(":", "")
        dt1 = datetime.strptime(s1[:15], "%Y%m%dT%H%M%S")
        dt2 = datetime.strptime(s2[:15], "%Y%m%dT%H%M%S")
        ref_dt, sec_dt = (dt1, dt2) if dt1 <= dt2 else (dt2, dt1)
        dt_days = max(12.0, (sec_dt - ref_dt).total_seconds() / 86400.0)
        return ref_dt.strftime("%Y%m%d"), sec_dt.strftime("%Y%m%d"), float(dt_days)

    raise ValueError(f"Could not parse interferometric pair dates from '{file_name}'.")


def _interpolate_2d_or_3d_grid(
    h5: h5py.File,
    ds_path: str,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> Optional[np.ndarray]:
    """Bilinearly interpolate a metadata radarGrid layer onto (target_y, target_x)."""
    if ds_path not in h5:
        return None
    try:
        parent = "/".join(ds_path.split("/")[:-1])
        rx = h5[f"{parent}/xCoordinates"][:]
        ry = h5[f"{parent}/yCoordinates"][:]
        raw = h5[ds_path][:]
        slice_2d = (
            np.asarray(raw[raw.shape[0] // 2, :, :], dtype=np.float32)
            if raw.ndim == 3
            else np.asarray(raw, dtype=np.float32)
        )
        if rx[0] > rx[-1]:
            rx = rx[::-1]
            slice_2d = slice_2d[:, ::-1]
        if ry[0] > ry[-1]:
            ry = ry[::-1]
            slice_2d = slice_2d[::-1, :]

        interp_x = np.empty((slice_2d.shape[0], len(target_x)), dtype=np.float32)
        for r in range(slice_2d.shape[0]):
            interp_x[r, :] = np.interp(
                target_x, rx, slice_2d[r, :], left=np.nan, right=np.nan
            )
        out = np.empty((len(target_y), len(target_x)), dtype=np.float32)
        for c in range(len(target_x)):
            out[:, c] = np.interp(
                target_y, ry, interp_x[:, c], left=np.nan, right=np.nan
            )
        return out
    except Exception as exc:
        logger.debug("Could not interpolate %s: %s", ds_path, exc)
        return None


def load_gunw_window(
    h5: h5py.File,
    bbox_lonlat: Sequence[float],
    looks: int = 2,
    coherence_threshold: float = 0.4,
    apply_ionosphere: bool = False,
    invalid_mask_values: Sequence[int] = (0, 255),
    verify_check: bool = True,
    track: Optional[int] = None,
    frame: Optional[int] = None,
    direction: Optional[str] = None,
    file_name: str = "unknown_gunw.h5",
    fallback_start: Optional[str] = None,
    fallback_stop: Optional[str] = None,
) -> GUNWWindow:
    """
    Stream the AOI window from a NISAR L2 GUNW granule, apply layover/shadow,
    connected-component, and coherence masks, and multilook.
    """
    if verify_check:
        require_verified_layer(GUNW_UNWRAPPED_PHASE, product="GUNW")
        require_verified_layer(GUNW_COHERENCE, product="GUNW")
        require_verified_layer(GUNW_X, product="GUNW")
        require_verified_layer(GUNW_Y, product="GUNW")
        require_verified_layer(GUNW_PROJECTION, product="GUNW")

    for req_path in (GUNW_UNWRAPPED_PHASE, GUNW_COHERENCE, GUNW_X, GUNW_Y, GUNW_PROJECTION):
        if req_path not in h5:
            raise UnsupportedGranuleError(
                f"GUNW granule {file_name} is missing dataset '{req_path}'."
            )

    x_full = h5[GUNW_X][:]
    y_full = h5[GUNW_Y][:]
    epsg = read_epsg_from_projection(h5[GUNW_PROJECTION])
    slice_y, slice_x, _ = bbox_to_utm_slices(x_full, y_full, bbox_lonlat, epsg)

    phase = np.asarray(h5[GUNW_UNWRAPPED_PHASE][slice_y, slice_x], dtype=np.float32)
    coh = np.asarray(h5[GUNW_COHERENCE][slice_y, slice_x], dtype=np.float32)

    if apply_ionosphere and GUNW_IONOSPHERE_PHASE in h5:
        iono = np.asarray(h5[GUNW_IONOSPHERE_PHASE][slice_y, slice_x], dtype=np.float32)
        phase = np.where(np.isfinite(iono), phase - iono, phase)

    valid_mask = np.isfinite(phase) & np.isfinite(coh) & (coh >= float(coherence_threshold))

    mask_ml: Optional[np.ndarray] = None
    if GUNW_MASK in h5:
        raw_mask = np.asarray(h5[GUNW_MASK][slice_y, slice_x], dtype=np.uint8)
        valid_mask &= is_valid_mask_array(raw_mask, invalid_mask_values)
        mask_ml = block_mask_nearest(raw_mask, looks)

    conn_ml: Optional[np.ndarray] = None
    if GUNW_CONNECTED_COMPONENTS in h5:
        conn = np.asarray(h5[GUNW_CONNECTED_COMPONENTS][slice_y, slice_x])
        valid_mask &= np.isfinite(conn) & (conn > 0)
        conn_ml = block_mask_nearest(conn.astype(np.uint8), looks)

    phase_masked = np.where(valid_mask, phase, np.nan).astype(np.float32)
    coh_masked = np.where(np.isfinite(coh) & (coh >= 0.0), coh, np.nan).astype(np.float32)

    phase_ml = block_mean(phase_masked, looks)
    coh_ml = np.clip(block_mean(coh_masked, looks), 0.0, 1.0)
    # Re-apply coherence threshold after multilooking
    phase_ml[~np.isfinite(coh_ml) | (coh_ml < float(coherence_threshold))] = np.nan

    x_win = coord_mean(x_full[slice_x], looks)
    y_win = coord_mean(y_full[slice_y], looks)
    inc_win = _interpolate_2d_or_3d_grid(h5, GUNW_INCIDENCE_ANGLE, x_win, y_win)

    ref_date, sec_date, dt_days = parse_pair_dates(file_name, fallback_start, fallback_stop)

    return GUNWWindow(
        unwrapped_phase=phase_ml,
        coherence=coh_ml,
        x=x_win,
        y=y_win,
        epsg=epsg,
        ref_date=ref_date,
        sec_date=sec_date,
        dt_days=dt_days,
        incidence_angle=inc_win,
        connected_components=conn_ml,
        mask=mask_ml,
        track=track,
        frame=frame,
        direction=direction,
        file_name=file_name,
    )


def phase_to_los_displacement_mm(
    unwrapped_phase_rad: np.ndarray,
    wavelength_m: float = NISAR_L_BAND_WAVELENGTH_M,
) -> np.ndarray:
    """
    Convert unwrapped interferometric phase (radians) to Line-of-Sight (LOS)
    displacement in millimeters:
        LOS_m  = -phase * wavelength_m / (4 * pi)
        LOS_mm = LOS_m * 1000.0

    Note on sign and units:
      - `unwrapped_phase_rad` is in radians.
      - `wavelength_m` is 0.24 m (NISAR L-band).
      - Negative LOS values indicate motion away from the radar (range increase / subsidence).
      - Positive LOS values indicate motion toward the radar (range decrease / uplift).
    """
    phase = np.asarray(unwrapped_phase_rad, dtype=np.float32)
    scale_mm = -float(wavelength_m) * 1000.0 / (4.0 * float(np.pi))
    out = np.full(phase.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(phase)
    out[valid] = phase[valid] * scale_mm
    return out


def los_to_vertical_displacement_mm(
    los_disp_mm: np.ndarray,
    incidence_angle_deg: np.ndarray | float,
    flat_terrain: bool = True,
) -> np.ndarray:
    """
    Convert LOS displacement (mm) to vertical displacement (mm) assuming negligible
    horizontal motion:
        vertical_mm = LOS_mm / cos(incidence_angle)

    Strictly enabled ONLY when `flat_terrain=True` (e.g., Nile Delta).
    Raises ValueError if called with `flat_terrain=False`.
    """
    if not flat_terrain:
        raise ValueError(
            "Vertical displacement projection (LOS / cos(inc)) is only allowed "
            "for flat terrain sites (e.g., Nile Delta)."
        )
    los = np.asarray(los_disp_mm, dtype=np.float32)
    inc_rad = np.deg2rad(np.asarray(incidence_angle_deg, dtype=np.float32))
    cos_inc = np.cos(inc_rad)
    out = np.full(los.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(los) & np.isfinite(cos_inc) & (cos_inc > 0.1)
    if np.ndim(cos_inc) == 0:
        out[valid] = los[valid] / float(cos_inc)
    else:
        out[valid] = los[valid] / cos_inc[valid]
    return out


def select_nearest_neighbor_chain(
    windows: Sequence[GUNWWindow],
) -> List[GUNWWindow]:
    """
    Filter and sort GUNW pairs from a single (track, frame, direction) group into a
    non-overlapping nearest-neighbor temporal chain:
    [(t0 -> t1), (t1 -> t2), ..., (t_{k-1} -> t_k)].
    When multiple pairs start at the same `ref_date`, prefers the shortest `dt_days`
    (~12-day repeat).
    """
    if not windows:
        return []

    first = windows[0]
    for w in windows[1:]:
        if (w.track, w.frame, w.direction) != (first.track, first.frame, first.direction):
            raise ValueError(
                "GUNW stack chaining requires all granules to share the same "
                "(track, frame, direction)."
            )

    sorted_wins = sorted(windows, key=lambda w: (w.ref_date, w.dt_days, w.sec_date))
    chain: List[GUNWWindow] = [sorted_wins[0]]

    for cand in sorted_wins[1:]:
        prev = chain[-1]
        if cand.ref_date == prev.ref_date:
            # Duplicate reference date: already picked shortest dt_days first
            continue
        if cand.ref_date >= prev.sec_date:
            chain.append(cand)

    return chain


def fit_robust_linear_rate_mm_yr(
    dates_yyyymmdd: Sequence[str],
    cum_disp_mm: np.ndarray,
    min_dates: int = 3,
    n_irls_iter: int = 3,
) -> np.ndarray:
    """
    Fit a robust linear displacement rate (mm/yr) per pixel across a cumulative
    displacement stack of shape `(T, H, W)` using Iteratively Reweighted Least Squares
    (Huber-like L1/L2 weighting) initialized from ordinary least squares.

    Pixels with fewer than `min_dates` finite observations are assigned NaN.
    """
    stack = np.asarray(cum_disp_mm, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got shape {stack.shape}")
    n_t, h, w = stack.shape
    if len(dates_yyyymmdd) != n_t:
        raise ValueError(
            f"Length of dates ({len(dates_yyyymmdd)}) does not match stack T ({n_t})."
        )
    if n_t < min_dates:
        return np.full((h, w), np.nan, dtype=np.float32)

    dt0 = datetime.strptime(dates_yyyymmdd[0], "%Y%m%d")
    t_years = np.array(
        [
            (datetime.strptime(d, "%Y%m%d") - dt0).total_seconds() / (365.25 * 86400.0)
            for d in dates_yyyymmdd
        ],
        dtype=np.float32,
    )[:, np.newaxis, np.newaxis]  # (T, 1, 1)

    valid = np.isfinite(stack)
    count = np.sum(valid, axis=0)  # (H, W)
    enough = count >= int(min_dates)

    weights = valid.astype(np.float32)
    slope = np.zeros((h, w), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_safe = np.where(valid, stack, 0.0)
        for _ in range(max(1, n_irls_iter)):
            w_sum = np.sum(weights, axis=0, keepdims=True)
            w_sum_safe = np.where(w_sum > 0, w_sum, 1.0)

            t_mean = np.sum(weights * t_years, axis=0, keepdims=True) / w_sum_safe
            y_mean = np.sum(weights * y_safe, axis=0, keepdims=True) / w_sum_safe

            dt = t_years - t_mean
            dy = y_safe - y_mean

            denom = np.sum(weights * dt * dt, axis=0)
            numer = np.sum(weights * dt * dy, axis=0)

            good_denom = enough & (denom > 1e-6)
            slope = np.where(good_denom, numer / np.where(good_denom, denom, 1.0), np.nan)
            intercept = np.squeeze(y_mean, axis=0) - slope * np.squeeze(t_mean, axis=0)

            pred = intercept[np.newaxis, :, :] + slope[np.newaxis, :, :] * t_years
            resid = np.abs(y_safe - pred)
            # Huber weight: 1 / max(1.0 mm, |resid|)
            weights = np.where(valid, 1.0 / np.maximum(1.0, resid), 0.0).astype(np.float32)

    slope[~enough] = np.nan
    return slope.astype(np.float32)


def process_gunw_stack(
    windows: Sequence[GUNWWindow],
    ref_bbox_lonlat: Optional[Sequence[float]] = None,
    wavelength_m: float = NISAR_L_BAND_WAVELENGTH_M,
    flat_terrain: bool = False,
    min_dates_for_rate: int = 3,
    propagate_nan: bool = False,
) -> Dict[str, Any]:
    """
    Process a sequence of same-track GUNWWindows into:
      - Referenced pairwise LOS displacements (mm)
      - Chained cumulative LOS displacement stack (mm)
      - Chained cumulative vertical displacement stack (mm, only if `flat_terrain=True`)
      - Robust linear displacement rate map (mm/yr)
    """
    chain = select_nearest_neighbor_chain(windows)
    if not chain:
        raise ValueError("No valid GUNW windows to process.")

    # Align all windows in the chain to the first window's coordinate grid
    base_x, base_y, epsg = chain[0].x, chain[0].y, chain[0].epsg
    h, w = len(base_y), len(base_x)

    pair_los_list: List[np.ndarray] = []
    coherence_list: List[np.ndarray] = []
    pair_tags: List[str] = []
    dates: List[str] = [chain[0].ref_date]

    for win in chain:
        los_mm = phase_to_los_displacement_mm(win.unwrapped_phase, wavelength_m=wavelength_m)
        los_Crop = los_mm[:h, :w]
        coh_Crop = win.coherence[:h, :w]

        if ref_bbox_lonlat is not None:
            los_Crop, _ = subtract_reference_window(
                los_Crop, base_x[:w], base_y[:h], ref_bbox_lonlat, epsg
            )

        pair_los_list.append(los_Crop)
        coherence_list.append(coh_Crop)
        pair_tags.append(f"{win.ref_date}_{win.sec_date}")
        dates.append(win.sec_date)

    cum_los_mm = chain_pairwise_displacements(pair_los_list, propagate_nan=propagate_nan)
    rate_mm_yr = fit_robust_linear_rate_mm_yr(
        dates_yyyymmdd=dates,
        cum_disp_mm=cum_los_mm,
        min_dates=min_dates_for_rate,
    )

    cum_vert_mm: Optional[np.ndarray] = None
    if flat_terrain:
        inc = chain[0].incidence_angle
        inc_used: np.ndarray | float = (
            inc[:h, :w] if inc is not None else 38.0
        )
        vert_slices = [
            los_to_vertical_displacement_mm(cum_los_mm[i], inc_used, flat_terrain=True)
            for i in range(cum_los_mm.shape[0])
        ]
        cum_vert_mm = np.stack(vert_slices, axis=0)

    return {
        "dates": dates,
        "pair_tags": pair_tags,
        "coherence_pairs": coherence_list,
        "pair_los_mm": pair_los_list,
        "cum_los_mm": cum_los_mm,
        "cum_vert_mm": cum_vert_mm,
        "rate_mm_yr": rate_mm_yr,
        "x": base_x[:w],
        "y": base_y[:h],
        "epsg": epsg,
        "track": chain[0].track,
        "frame": chain[0].frame,
        "direction": chain[0].direction,
    }