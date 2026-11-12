"""
NISAR L2 GOFF (Geocoded Pixel Offsets) glacier surface velocity processing:
- Quality masking using correlation peak / SNR threshold and layover/shadow mask
- Offset-to-velocity conversion: velocity (m/yr) = offset_m / (dt_days / 365.25)
- 2D horizontal velocity vector combination (East, North) from slant-range and
  along-track offsets:
    Physical assumptions:
      1. Surface-parallel / horizontal flow approximation (vertical velocity << horizontal
         velocity over the glacier tongue), converting slant-range velocity v_sr into
         ground-range velocity: v_gr = v_sr / sin(incidence_angle).
      2. Satellite track heading `heading_deg` (measured clockwise from North:
         typically ~350 deg for ascending, ~190 deg for descending) and radar look
         azimuth `look_az_deg = heading_deg - 90` for NISAR's left-looking L-band SAR
         (or `heading_deg + 90` if right-looking).
      3. Projection into UTM Easting / Northing:
         v_east  = v_az * sin(heading) + v_gr * sin(look_az)
         v_north = v_az * cos(heading) + v_gr * cos(look_az)
- Speed (m/yr) and flow direction (degrees clockwise from North, [0, 360))
- Optional glacier-outline polygon clipping from a user-supplied RGI GeoJSON
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import Affine

from nisar_pipeline.gcov import compute_affine_and_orient
from nisar_pipeline.gunw import _interpolate_2d_or_3d_grid, parse_pair_dates
from nisar_pipeline.io_nisar import UnsupportedGranuleError
from nisar_pipeline.layers import (
    GOFF_ALONG_TRACK_OFFSET,
    GOFF_CORRELATION_PEAK,
    GOFF_INCIDENCE_ANGLE,
    GOFF_MASK,
    GOFF_PROJECTION,
    GOFF_SLANT_RANGE_OFFSET,
    GOFF_SNR,
    GOFF_X,
    GOFF_Y,
    require_verified_layer,
)
from nisar_pipeline.window import (
    bbox_to_utm_slices,
    block_mask_nearest,
    block_mean,
    coord_mean,
    is_valid_mask_array,
    read_epsg_from_projection,
)

logger = logging.getLogger(__name__)


@dataclass
class GOFFWindow:
    """Container for a multilooked GOFF AOI offset window in native UTM."""

    slant_range_offset: np.ndarray  # Meters (float32, NaN nodata)
    along_track_offset: np.ndarray  # Meters (float32, NaN nodata)
    quality: np.ndarray  # Correlation peak or SNR (float32, NaN nodata)
    x: np.ndarray  # 1D UTM Easting (float64)
    y: np.ndarray  # 1D UTM Northing (float64)
    epsg: int
    ref_date: str  # YYYYMMDD
    sec_date: str  # YYYYMMDD
    dt_days: float
    incidence_angle: Optional[np.ndarray] = None  # Degrees (float32)
    mask: Optional[np.ndarray] = None
    track: Optional[int] = None
    frame: Optional[int] = None
    direction: Optional[str] = None
    file_name: Optional[str] = None


def load_goff_window(
    h5: h5py.File,
    bbox_lonlat: Sequence[float],
    looks: int = 2,
    quality_threshold: float = 0.3,
    invalid_mask_values: Sequence[int] = (0, 255),
    verify_check: bool = True,
    track: Optional[int] = None,
    frame: Optional[int] = None,
    direction: Optional[str] = None,
    file_name: str = "unknown_goff.h5",
    fallback_start: Optional[str] = None,
    fallback_stop: Optional[str] = None,
) -> GOFFWindow:
    """
    Stream only the AOI window from a NISAR L2 GOFF HDF5 file, mask low-correlation
    and layover/shadow pixels, and multilook.
    """
    if verify_check:
        require_verified_layer(GOFF_SLANT_RANGE_OFFSET, product="GOFF")
        require_verified_layer(GOFF_ALONG_TRACK_OFFSET, product="GOFF")
        require_verified_layer(GOFF_X, product="GOFF")
        require_verified_layer(GOFF_Y, product="GOFF")
        require_verified_layer(GOFF_PROJECTION, product="GOFF")

    for req_path in (
        GOFF_SLANT_RANGE_OFFSET,
        GOFF_ALONG_TRACK_OFFSET,
        GOFF_X,
        GOFF_Y,
        GOFF_PROJECTION,
    ):
        if req_path not in h5:
            raise UnsupportedGranuleError(
                f"GOFF granule {file_name} is missing dataset '{req_path}'."
            )

    x_full = h5[GOFF_X][:]
    y_full = h5[GOFF_Y][:]
    epsg = read_epsg_from_projection(h5[GOFF_PROJECTION])
    slice_y, slice_x, _ = bbox_to_utm_slices(x_full, y_full, bbox_lonlat, epsg)

    sr_off = np.asarray(h5[GOFF_SLANT_RANGE_OFFSET][slice_y, slice_x], dtype=np.float32)
    az_off = np.asarray(h5[GOFF_ALONG_TRACK_OFFSET][slice_y, slice_x], dtype=np.float32)

    if GOFF_CORRELATION_PEAK in h5:
        qual = np.asarray(h5[GOFF_CORRELATION_PEAK][slice_y, slice_x], dtype=np.float32)
    elif GOFF_SNR in h5:
        qual = np.asarray(h5[GOFF_SNR][slice_y, slice_x], dtype=np.float32)
    else:
        qual = np.ones(sr_off.shape, dtype=np.float32)

    valid = (
        np.isfinite(sr_off)
        & np.isfinite(az_off)
        & np.isfinite(qual)
        & (qual >= float(quality_threshold))
    )

    mask_ml: Optional[np.ndarray] = None
    if GOFF_MASK in h5:
        raw_mask = np.asarray(h5[GOFF_MASK][slice_y, slice_x], dtype=np.uint8)
        valid &= is_valid_mask_array(raw_mask, invalid_mask_values)
        mask_ml = block_mask_nearest(raw_mask, looks)

    sr_masked = np.where(valid, sr_off, np.nan).astype(np.float32)
    az_masked = np.where(valid, az_off, np.nan).astype(np.float32)
    qual_masked = np.where(np.isfinite(qual), qual, np.nan).astype(np.float32)

    sr_ml = block_mean(sr_masked, looks)
    az_ml = block_mean(az_masked, looks)
    qual_ml = block_mean(qual_masked, looks)

    low_q = ~np.isfinite(qual_ml) | (qual_ml < float(quality_threshold))
    sr_ml[low_q] = np.nan
    az_ml[low_q] = np.nan

    x_win = coord_mean(x_full[slice_x], looks)
    y_win = coord_mean(y_full[slice_y], looks)
    inc_win = _interpolate_2d_or_3d_grid(h5, GOFF_INCIDENCE_ANGLE, x_win, y_win)

    ref_date, sec_date, dt_days = parse_pair_dates(file_name, fallback_start, fallback_stop)

    return GOFFWindow(
        slant_range_offset=sr_ml,
        along_track_offset=az_ml,
        quality=qual_ml,
        x=x_win,
        y=y_win,
        epsg=epsg,
        ref_date=ref_date,
        sec_date=sec_date,
        dt_days=dt_days,
        incidence_angle=inc_win,
        mask=mask_ml,
        track=track,
        frame=frame,
        direction=direction,
        file_name=file_name,
    )


def compute_velocity_vectors(
    slant_range_offset_m: np.ndarray,
    along_track_offset_m: np.ndarray,
    dt_days: float,
    incidence_angle_deg: np.ndarray | float = 38.0,
    flight_direction: str = "A",
    heading_deg: Optional[float] = None,
    left_looking: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert slant-range and along-track pixel offsets (meters) over `dt_days`
    into horizontal velocity vectors (v_east, v_north), speed (m/yr), and
    azimuth direction (degrees clockwise from North in [0, 360)).

    Assumptions:
      1. `slant_range_offset_m` and `along_track_offset_m` are in meters (NISAR L2 GOFF
         geocoded grid convention). Velocity in m/yr is `offset_m / (dt_days / 365.25)`.
      2. Surface flow is approximately horizontal (`v_gr = v_sr / sin(incidence_angle)`).
      3. Orbit track heading defaults to 350.0 deg (Ascending) or 190.0 deg (Descending)
         unless overridden by `heading_deg`. Look direction is `heading_deg - 90` for
         NISAR's left-looking antenna (`left_looking=True`).
    """
    if dt_days <= 0:
        raise ValueError(f"dt_days must be positive, got {dt_days}")

    dt_yr = float(dt_days) / 365.25
    v_sr = np.asarray(slant_range_offset_m, dtype=np.float32) / dt_yr
    v_az = np.asarray(along_track_offset_m, dtype=np.float32) / dt_yr

    inc_rad = np.deg2rad(np.asarray(incidence_angle_deg, dtype=np.float32))
    sin_inc = np.maximum(0.2, np.sin(inc_rad))
    v_gr = (v_sr / sin_inc).astype(np.float32)

    if heading_deg is None:
        dir_char = (flight_direction or "A").strip()[0].upper()
        heading_deg = 350.0 if dir_char == "A" else 190.0

    look_deg = (heading_deg - 90.0) if left_looking else (heading_deg + 90.0)
    head_rad = np.deg2rad(float(heading_deg))
    look_rad = np.deg2rad(float(look_deg))

    v_east = (v_az * np.sin(head_rad) + v_gr * np.sin(look_rad)).astype(np.float32)
    v_north = (v_az * np.cos(head_rad) + v_gr * np.cos(look_rad)).astype(np.float32)

    speed = np.hypot(v_east, v_north).astype(np.float32)
    direction = (np.rad2deg(np.arctan2(v_east, v_north)) + 360.0) % 360.0
    direction = direction.astype(np.float32)

    invalid = ~np.isfinite(v_east) | ~np.isfinite(v_north)
    speed[invalid] = np.nan
    direction[invalid] = np.nan

    return {
        "v_east_m_yr": v_east,
        "v_north_m_yr": v_north,
        "speed_m_yr": speed,
        "direction_deg": direction,
    }


def _reproject_geojson_coords(geom: Dict[str, Any], tr: Transformer) -> Dict[str, Any]:
    """Transform GeoJSON Polygon / MultiPolygon coordinates from EPSG:4326 to target EPSG."""
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        new_rings = [
            [list(tr.transform(float(pt[0]), float(pt[1]))) for pt in ring]
            for ring in coords
        ]
        return {"type": "Polygon", "coordinates": new_rings}
    if gtype == "MultiPolygon":
        new_polys = [
            [
                [list(tr.transform(float(pt[0]), float(pt[1]))) for pt in ring]
                for ring in poly
            ]
            for poly in coords
        ]
        return {"type": "MultiPolygon", "coordinates": new_polys}
    raise ValueError(f"Unsupported glacier outline geometry type: {gtype}")


def clip_to_glacier_outlines(
    layers: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    rgi_geojson_path: Optional[Path | str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Mask velocity arrays outside Randolph Glacier Inventory (RGI) polygons supplied
    in an EPSG:4326 GeoJSON file. If `rgi_geojson_path` is None, returns oriented arrays unchanged.
    """
    first_arr = next(iter(layers.values()))
    _, x_oriented, y_oriented, transform = compute_affine_and_orient(first_arr, x, y)

    oriented_layers: Dict[str, np.ndarray] = {}
    for k, arr in layers.items():
        data_o, _, _, _ = compute_affine_and_orient(arr, x, y)
        oriented_layers[k] = data_o

    if not rgi_geojson_path:
        return oriented_layers, x_oriented, y_oriented

    geojson_file = Path(rgi_geojson_path)
    if not geojson_file.exists():
        raise FileNotFoundError(f"RGI glacier outline GeoJSON not found: {geojson_file}")

    doc = json.loads(geojson_file.read_text(encoding="utf-8"))
    features = doc.get("features", [doc] if "geometry" in doc else [])
    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    utm_geoms = []
    for feat in features:
        geom = feat.get("geometry") or feat
        if geom.get("type") in ("Polygon", "MultiPolygon"):
            utm_geoms.append(_reproject_geojson_coords(geom, tr))

    if not utm_geoms:
        logger.warning("No Polygon/MultiPolygon geometries in %s; skipping clip.", geojson_file)
        return oriented_layers, x_oriented, y_oriented

    outside_mask = geometry_mask(
        utm_geoms,
        out_shape=first_arr.shape,
        transform=transform,
        invert=False,
    )

    clipped: Dict[str, np.ndarray] = {}
    for k, arr in oriented_layers.items():
        out = arr.copy()
        out[outside_mask] = np.nan
        clipped[k] = out

    return clipped, x_oriented, y_oriented