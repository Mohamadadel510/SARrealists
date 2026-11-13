"""
Web visualization asset exporter:
- Vectorized single-warp per zoom level from native UTM to EPSG:3857 (Web Mercator):
  * Reprojects linear backscatter power BEFORE converting to dB
  * Uses bilinear resampling for continuous layers and nearest-neighbor for masks/RGB
- Slippy-map 8-bit RGBA colormapped XYZ PNG tiles (`out/<site>/tiles/<layer>/<YYYYMMDD>/{z}/{x}/{y}.png`)
  and 3-band Polarimetric RGB composite XYZ tiles (`out/<site>/tiles/rgb_composite/<YYYYMMDD>/{z}/{x}/{y}.png`)
- GeoJSON vector exporters in EPSG:4326 (`out/<site>/vectors/<name>.geojson`)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.image as mpimg
import numpy as np
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

from nisar_pipeline.gcov import compute_affine_and_orient, linear_to_db

logger = logging.getLogger(__name__)

WEB_MERCATOR_HALF_WORLD: float = 20037508.342789244


def reproject_array_to_3857(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    src_epsg: int,
    is_mask: bool = False,
    convert_linear_to_db_after: bool = False,
) -> Tuple[np.ndarray, Affine, Tuple[float, float, float, float]]:
    """
    Reproject a 2D array from native UTM (`EPSG:src_epsg`) to Web Mercator (`EPSG:3857`).
    """
    src_data, x_asc, y_desc, src_transform = compute_affine_and_orient(arr, x, y)
    h, w = src_data.shape
    dx = float(abs(x_asc[1] - x_asc[0])) if len(x_asc) > 1 else 20.0
    dy = float(abs(y_desc[1] - y_desc[0])) if len(y_desc) > 1 else 20.0
    left = float(x_asc.min() - dx / 2.0)
    right = float(x_asc.max() + dx / 2.0)
    bottom = float(y_desc.min() - dy / 2.0)
    top = float(y_desc.max() + dy / 2.0)

    src_crs = CRS.from_epsg(int(src_epsg))
    dst_crs = CRS.from_epsg(3857)

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, left=left, bottom=bottom, right=right, top=top
    )
    dst_data = np.full((int(dst_h), int(dst_w)), np.nan, dtype=np.float32)

    resampling_mode = Resampling.nearest if is_mask else Resampling.bilinear
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling_mode,
    )

    if convert_linear_to_db_after:
        dst_data = linear_to_db(dst_data)

    dst_left = float(dst_transform.c)
    dst_top = float(dst_transform.f)
    dst_right = dst_left + float(dst_w) * float(dst_transform.a)
    dst_bottom = dst_top + float(dst_h) * float(dst_transform.e)
    return dst_data, dst_transform, (dst_left, dst_bottom, dst_right, dst_top)


def apply_fixed_colormap_rgba(
    arr: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_name: str = "viridis",
) -> np.ndarray:
    """
    Map a 2D float32 array with NaN nodata into an 8-bit `(H, W, 4)` RGBA uint8 image
    using fixed `[vmin, vmax]` bounds. NaN pixels receive alpha = 0 (transparent).
    """
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return rgba

    denom = max(1e-6, float(vmax) - float(vmin))
    normed = np.clip((arr[valid] - float(vmin)) / denom, 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    mapped = cmap(normed)
    rgba_valid = np.round(mapped * 255.0).astype(np.uint8)
    rgba_valid[:, 3] = 255
    rgba[valid] = rgba_valid
    return rgba


def _lonlat_to_tile_xy(lon: float, lat: float, z: int) -> Tuple[int, int]:
    """Convert WGS84 (lon, lat) to slippy-map XYZ tile indices at zoom `z`."""
    lat_clamped = max(-85.05112878, min(85.05112878, float(lat)))
    n = 2**z
    xtile = int(math.floor((float(lon) + 180.0) / 360.0 * n))
    lat_rad = math.radians(lat_clamped)
    ytile = int(
        math.floor(
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        )
    )
    xtile = max(0, min(n - 1, xtile))
    ytile = max(0, min(n - 1, ytile))
    return xtile, ytile


def _tile_xy_to_3857_bounds(xtile: int, ytile: int, z: int) -> Tuple[float, float, float, float]:
    """Return (left, bottom, right, top) bounds in EPSG:3857 meters for tile (z, x, y)."""
    n = float(2**z)
    tile_span = (2.0 * WEB_MERCATOR_HALF_WORLD) / n
    left = -WEB_MERCATOR_HALF_WORLD + xtile * tile_span
    right = left + tile_span
    top = WEB_MERCATOR_HALF_WORLD - ytile * tile_span
    bottom = top - tile_span
    return left, bottom, right, top


def _choose_adaptive_zoom_levels(
    w_lon: float, s_lat: float, e_lon: float, n_lat: float, requested: Sequence[int]
) -> Sequence[int]:
    """Scale default zoom levels to regional extent so large AOIs tile quickly."""
    max_span_deg = max(e_lon - w_lon, n_lat - s_lat)
    if tuple(requested) == (10, 11, 12) and max_span_deg > 0.6:
        return (7, 8, 9) if max_span_deg > 1.5 else (8, 9, 10)
    return requested


def export_xyz_png_tiles(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    src_epsg: int,
    tiles_dir: Path,
    vmin: float,
    vmax: float,
    cmap_name: str = "viridis",
    zoom_levels: Sequence[int] = (10, 11, 12),
    is_mask: bool = False,
    convert_linear_to_db_after: bool = False,
    tile_size: int = 256,
) -> int:
    """
    Generate 8-bit colormapped XYZ PNG tiles in `EPSG:3857` using a single vectorized
    reproject call per zoom level (`out/<site>/tiles/<layer>/<YYYYMMDD>/{z}/{x}/{y}.png`).
    """
    src_data, x_asc, y_desc, src_transform = compute_affine_and_orient(arr, x, y)
    if not np.any(np.isfinite(src_data)):
        return 0

    src_crs = CRS.from_epsg(int(src_epsg))
    dst_crs = CRS.from_epsg(3857)
    resampling_mode = Resampling.nearest if is_mask else Resampling.bilinear

    tr_to_4326 = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lons, lats = zip(
        *[
            tr_to_4326.transform(cx, cy)
            for cx, cy in [
                (float(x_asc.min()), float(y_desc.min())),
                (float(x_asc.max()), float(y_desc.min())),
                (float(x_asc.max()), float(y_desc.max())),
                (float(x_asc.min()), float(y_desc.max())),
            ]
        ]
    )
    w_lon, s_lat, e_lon, n_lat = min(lons), min(lats), max(lons), max(lats)
    zoom_levels = _choose_adaptive_zoom_levels(w_lon, s_lat, e_lon, n_lat, zoom_levels)

    written_count = 0
    for z in zoom_levels:
        x_min, y_max = _lonlat_to_tile_xy(w_lon, s_lat, z)
        x_max, y_min = _lonlat_to_tile_xy(e_lon, n_lat, z)

        n_tx = x_max - x_min + 1
        n_ty = y_max - y_min + 1
        grid_left, _, _, grid_top = _tile_xy_to_3857_bounds(x_min, y_min, z)
        _, grid_bottom, grid_right, _ = _tile_xy_to_3857_bounds(x_max, y_max, z)

        grid_w = n_tx * tile_size
        grid_h = n_ty * tile_size
        grid_transform = from_bounds(
            grid_left, grid_bottom, grid_right, grid_top, grid_w, grid_h
        )
        grid_buf = np.full((grid_h, grid_w), np.nan, dtype=np.float32)

        reproject(
            source=src_data,
            destination=grid_buf,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=grid_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling_mode,
        )

        if not np.any(np.isfinite(grid_buf)):
            continue

        if convert_linear_to_db_after:
            grid_buf = linear_to_db(grid_buf)

        grid_rgba = apply_fixed_colormap_rgba(
            grid_buf, vmin=vmin, vmax=vmax, cmap_name=cmap_name
        )

        for col_idx, tx in enumerate(range(x_min, x_max + 1)):
            c0 = col_idx * tile_size
            c1 = c0 + tile_size
            tx_dir = tiles_dir / str(z) / str(tx)
            for row_idx, ty in enumerate(range(y_min, y_max + 1)):
                r0 = row_idx * tile_size
                r1 = r0 + tile_size
                tile_rgba = grid_rgba[r0:r1, c0:c1]
                if not np.any(tile_rgba[:, :, 3] > 0):
                    continue
                tx_dir.mkdir(parents=True, exist_ok=True)
                mpimg.imsave(tx_dir / f"{ty}.png", tile_rgba)
                written_count += 1

    return written_count


def export_rgb_xyz_png_tiles(
    rgb_arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    src_epsg: int,
    tiles_dir: Path,
    zoom_levels: Sequence[int] = (10, 11, 12),
    tile_size: int = 256,
) -> int:
    """
    Generate 8-bit RGBA XYZ PNG tiles in `EPSG:3857` for the 3-band `(3, H, W)`
    Polarimetric RGB composite (`0` = transparent nodata) using a single warp per zoom level.
    """
    src_data, x_asc, y_desc, src_transform = compute_affine_and_orient(rgb_arr, x, y)
    src_crs = CRS.from_epsg(int(src_epsg))
    dst_crs = CRS.from_epsg(3857)

    tr_to_4326 = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lons, lats = zip(
        *[
            tr_to_4326.transform(cx, cy)
            for cx, cy in [
                (float(x_asc.min()), float(y_desc.min())),
                (float(x_asc.max()), float(y_desc.min())),
                (float(x_asc.max()), float(y_desc.max())),
                (float(x_asc.min()), float(y_desc.max())),
            ]
        ]
    )
    w_lon, s_lat, e_lon, n_lat = min(lons), min(lats), max(lons), max(lats)
    zoom_levels = _choose_adaptive_zoom_levels(w_lon, s_lat, e_lon, n_lat, zoom_levels)

    written_count = 0
    for z in zoom_levels:
        x_min, y_max = _lonlat_to_tile_xy(w_lon, s_lat, z)
        x_max, y_min = _lonlat_to_tile_xy(e_lon, n_lat, z)

        n_tx = x_max - x_min + 1
        n_ty = y_max - y_min + 1
        grid_left, _, _, grid_top = _tile_xy_to_3857_bounds(x_min, y_min, z)
        _, grid_bottom, grid_right, _ = _tile_xy_to_3857_bounds(x_max, y_max, z)

        grid_w = n_tx * tile_size
        grid_h = n_ty * tile_size
        grid_transform = from_bounds(
            grid_left, grid_bottom, grid_right, grid_top, grid_w, grid_h
        )
        grid_rgb = np.zeros((3, grid_h, grid_w), dtype=np.uint8)

        for b_idx in range(3):
            reproject(
                source=src_data[b_idx],
                destination=grid_rgb[b_idx],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=grid_transform,
                dst_crs=dst_crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

        valid_mask = (grid_rgb[0] > 0) | (grid_rgb[1] > 0) | (grid_rgb[2] > 0)
        if not np.any(valid_mask):
            continue

        grid_rgba = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        grid_rgba[:, :, 0] = grid_rgb[0]
        grid_rgba[:, :, 1] = grid_rgb[1]
        grid_rgba[:, :, 2] = grid_rgb[2]
        grid_rgba[:, :, 3] = np.where(valid_mask, 255, 0).astype(np.uint8)

        for col_idx, tx in enumerate(range(x_min, x_max + 1)):
            c0 = col_idx * tile_size
            c1 = c0 + tile_size
            tx_dir = tiles_dir / str(z) / str(tx)
            for row_idx, ty in enumerate(range(y_min, y_max + 1)):
                r0 = row_idx * tile_size
                r1 = r0 + tile_size
                tile_rgba = grid_rgba[r0:r1, c0:c1]
                if not np.any(tile_rgba[:, :, 3] > 0):
                    continue
                tx_dir.mkdir(parents=True, exist_ok=True)
                mpimg.imsave(tx_dir / f"{ty}.png", tile_rgba)
                written_count += 1

    return written_count


def export_velocity_vectors_geojson(
    v_east_m_yr: np.ndarray,
    v_north_m_yr: np.ndarray,
    speed_m_yr: np.ndarray,
    direction_deg: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    out_geojson_path: Path,
    stride: int = 4,
    min_speed_m_yr: float = 0.5,
) -> Path:
    """
    Export subsampled glacier velocity vectors as an EPSG:4326 GeoJSON FeatureCollection
    (`out/<site>/vectors/glacier_velocity.geojson`) with properties:
    `speed`, `dir`, `ve_m_yr`, `vn_m_yr`.
    """
    out_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)

    ve, x_c, y_c, _ = compute_affine_and_orient(v_east_m_yr, x, y)
    vn, _, _, _ = compute_affine_and_orient(v_north_m_yr, x, y)
    spd, _, _, _ = compute_affine_and_orient(speed_m_yr, x, y)
    ang, _, _, _ = compute_affine_and_orient(direction_deg, x, y)

    features: List[Dict[str, Any]] = []
    step = max(1, int(stride))

    for iy in range(0, spd.shape[0], step):
        for ix in range(0, spd.shape[1], step):
            s_val = float(spd[iy, ix])
            if not np.isfinite(s_val) or s_val < float(min_speed_m_yr):
                continue
            lon, lat = tr.transform(float(x_c[ix]), float(y_c[iy]))
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(lon), 6), round(float(lat), 6)],
                    },
                    "properties": {
                        "speed": round(s_val, 3),
                        "dir": round(float(ang[iy, ix]), 2),
                        "ve_m_yr": round(float(ve[iy, ix]), 3),
                        "vn_m_yr": round(float(vn[iy, ix]), 3),
                    },
                }
            )

    fc = {"type": "FeatureCollection", "features": features}
    out_geojson_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return out_geojson_path


def export_subsidence_points_geojson(
    rate_mm_yr: np.ndarray,
    cum_disp_mm: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
    out_geojson_path: Path,
    stride: int = 4,
) -> Path:
    """
    Export subsampled land subsidence grid points as an EPSG:4326 GeoJSON FeatureCollection
    (`out/<site>/vectors/subsidence_rates.geojson`) with properties:
    `rate_mm_yr`, `disp_mm`.
    """
    out_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)

    rate_o, x_c, y_c, _ = compute_affine_and_orient(rate_mm_yr, x, y)
    disp_o, _, _, _ = compute_affine_and_orient(cum_disp_mm, x, y)

    features: List[Dict[str, Any]] = []
    step = max(1, int(stride))

    for iy in range(0, rate_o.shape[0], step):
        for ix in range(0, rate_o.shape[1], step):
            r_val = float(rate_o[iy, ix])
            d_val = float(disp_o[iy, ix])
            if not np.isfinite(r_val) and not np.isfinite(d_val):
                continue
            lon, lat = tr.transform(float(x_c[ix]), float(y_c[iy]))
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(lon), 6), round(float(lat), 6)],
                    },
                    "properties": {
                        "rate_mm_yr": round(r_val, 3) if np.isfinite(r_val) else None,
                        "disp_mm": round(d_val, 3) if np.isfinite(d_val) else None,
                    },
                }
            )

    fc = {"type": "FeatureCollection", "features": features}
    out_geojson_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return out_geojson_path