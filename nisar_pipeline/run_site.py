#!/usr/bin/env python3
"""
CLI entrypoint for NISAR Space Apps preprocessing pipeline:
  python -m nisar_pipeline.run_site search  --site california --products GCOV GUNW
  python -m nisar_pipeline.run_site inspect --site nile_delta --products GUNW --idx 0
  python -m nisar_pipeline.run_site build   --site nile_delta --products GCOV
  python -m nisar_pipeline.run_site plot    --site nile_delta
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pyproj import Transformer
import yaml

from nisar_pipeline.export_web import (
    export_rgb_xyz_png_tiles,
    export_subsidence_points_geojson,
    export_velocity_vectors_geojson,
    export_xyz_png_tiles,
)
from nisar_pipeline.gcov import (
    append_to_zarr_stack,
    compute_pairwise_change,
    compute_single_date_products,
    resolve_co_cross_pols,
    update_points_timeseries_json,
    write_float32_cog,
    write_rgb_cog,
)
from nisar_pipeline.goff import (
    GOFFWindow,
    clip_to_glacier_outlines,
    compute_velocity_vectors,
    load_goff_window,
)
from nisar_pipeline.gunw import (
    GUNWWindow,
    load_gunw_window,
    process_gunw_stack,
)
from nisar_pipeline.io_nisar import (
    GranuleRecord,
    get_edl_token,
    group_by_track_frame,
    inspect_granule,
    print_search_table,
    run_with_h5_retry,
    search_nisar,
    select_primary_group,
)
from nisar_pipeline.layers import (
    SME2_PROJECTION,
    SME2_QUALITY_FLAG,
    SME2_SOIL_MOISTURE,
    SME2_X,
    SME2_Y,
    require_verified_layer,
)
from nisar_pipeline.plots import (
    enrich_catalog_with_human_labels,
    plot_human_friendly_guide,
    plot_pol_and_freq_explorer,
)
from nisar_pipeline.window import (
    GCOVWindow,
    align_grid_to_target_coords,
    bbox_to_utm_slices,
    block_mean,
    coord_mean,
    load_gcov_window,
    read_epsg_from_projection,
)

logger = logging.getLogger("nisar_pipeline")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML pipeline configuration."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def utm_bounds_to_lonlat(x: np.ndarray, y: np.ndarray, epsg: int) -> List[float]:
    """Compute [west, south, east, north] in EPSG:4326 from 1D UTM coordinate arrays."""
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    lons, lats = zip(*[tr.transform(cx, cy) for cx, cy in corners])
    return [
        round(float(min(lons)), 6),
        round(float(min(lats)), 6),
        round(float(max(lons)), 6),
        round(float(max(lats)), 6),
    ]


def _merge_same_date_gcov_windows(windows: List[GCOVWindow]) -> GCOVWindow:
    """
    If multiple frames are acquired on the same date in the same EPSG zone,
    merge their linear-power grids onto a combined UTM coordinate canvas so
    adjacent swaths fill the AOI without overwriting one another.
    """
    if len(windows) == 1:
        return windows[0]

    base = max(windows, key=lambda w: w.x.size * w.y.size)
    same_epsg = [w for w in windows if w.epsg == base.epsg]
    if len(same_epsg) == 1:
        return base

    dx = float(abs(base.x[1] - base.x[0])) if len(base.x) > 1 else 20.0
    dy = float(abs(base.y[1] - base.y[0])) if len(base.y) > 1 else 20.0
    xmin = min(float(w.x.min()) for w in same_epsg)
    xmax = max(float(w.x.max()) for w in same_epsg)
    ymin = min(float(w.y.min()) for w in same_epsg)
    ymax = max(float(w.y.max()) for w in same_epsg)

    merged_x = np.arange(xmin, xmax + 0.5 * dx, dx, dtype=np.float64)
    merged_y = np.arange(ymax, ymin - 0.5 * dy, -dy, dtype=np.float64)

    all_pols = sorted({p for w in same_epsg for p in w.pols})
    merged_pols: Dict[str, np.ndarray] = {}
    for pol in all_pols:
        stack = []
        for w in same_epsg:
            if pol in w.pols:
                aligned = align_grid_to_target_coords(w.pols[pol], w.x, w.y, merged_x, merged_y)
                out_bounds = (
                    (merged_x[np.newaxis, :] < float(w.x.min()) - dx)
                    | (merged_x[np.newaxis, :] > float(w.x.max()) + dx)
                    | (merged_y[:, np.newaxis] < float(w.y.min()) - dy)
                    | (merged_y[:, np.newaxis] > float(w.y.max()) + dy)
                )
                aligned = aligned.copy()
                aligned[out_bounds] = np.nan
                stack.append(aligned)
        if stack:
            with np.errstate(invalid="ignore"):
                merged_pols[pol] = np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)

    all_fb_pols = sorted({p for w in same_epsg for p in w.freq_b_pols})
    merged_fb: Dict[str, np.ndarray] = {}
    for pol in all_fb_pols:
        stack_b = []
        for w in same_epsg:
            if pol in w.freq_b_pols:
                aligned_b = align_grid_to_target_coords(
                    w.freq_b_pols[pol], w.x, w.y, merged_x, merged_y
                )
                out_bounds = (
                    (merged_x[np.newaxis, :] < float(w.x.min()) - dx)
                    | (merged_x[np.newaxis, :] > float(w.x.max()) + dx)
                    | (merged_y[:, np.newaxis] < float(w.y.min()) - dy)
                    | (merged_y[:, np.newaxis] > float(w.y.max()) + dy)
                )
                aligned_b = aligned_b.copy()
                aligned_b[out_bounds] = np.nan
                stack_b.append(aligned_b)
        if stack_b:
            with np.errstate(invalid="ignore"):
                merged_fb[pol] = np.nanmean(np.stack(stack_b, axis=0), axis=0).astype(np.float32)

    has_co = ("HHHH" in merged_pols) or ("VVVV" in merged_pols)
    has_cross = ("HVHV" in merged_pols) or ("VHVH" in merged_pols)
    return GCOVWindow(
        pols=merged_pols,
        x=merged_x,
        y=merged_y,
        epsg=base.epsg,
        mask=base.mask,
        incidence_angle=base.incidence_angle,
        track=base.track,
        frame=base.frame,
        direction=base.direction,
        time=base.time,
        date_str=base.date_str,
        file_name=base.file_name,
        available_pols=list(merged_pols.keys()),
        missing_pols=base.missing_pols,
        has_dual_pol=bool(has_co and has_cross),
        freq_b_pols=merged_fb,
        has_dual_freq=bool(len(merged_fb) > 0),
        freq_a_mhz=base.freq_a_mhz,
        freq_b_mhz=base.freq_b_mhz,
        pixel_spacing_a_m=base.pixel_spacing_a_m,
        pixel_spacing_b_m=base.pixel_spacing_b_m,
        active_bands=base.active_bands,
    )


def update_catalog_json(
    catalog_path: Path,
    site_name: str,
    site_title: str,
    layer_entries: Dict[str, Dict[str, Any]],
) -> Path:
    """Create or update `out/<site>/catalog.json` and enrich with plain-English legend labels."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog: Dict[str, Any] = {
        "site": site_name,
        "title": site_title,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestamps": [],
        "default_timestamp": None,
        "layers": {},
    }
    if catalog_path.exists():
        try:
            existing = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(existing.get("layers"), dict):
                catalog["layers"] = existing["layers"]
        except Exception as exc:
            logger.warning("Overwriting unreadable catalog %s (%s)", catalog_path, exc)

    for layer_id, entry in layer_entries.items():
        catalog["layers"][layer_id] = entry

    all_dates = sorted(
        {
            d
            for meta in catalog["layers"].values()
            for d in meta.get("dates", [])
            if isinstance(d, str) and len(d) == 8 and d.isdigit()
        }
    )
    catalog["timestamps"] = all_dates
    catalog["default_timestamp"] = all_dates[-1] if all_dates else None

    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    enrich_catalog_with_human_labels(catalog_path)
    return catalog_path


def build_gcov_for_site(
    site_name: str,
    cfg: Dict[str, Any],
    records: Sequence[GranuleRecord],
    token: str,
    download: bool = False,
    max_dates: Optional[int] = None,
) -> None:
    """
    Build GCOV float32 COGs, 3-band Polarimetric RGB COGs, Dual-Frequency (`Freq A - Freq B`)
    COGs, Zarr stacks, EPSG:3857 XYZ PNG tiles, point time series, catalog, and diagnostic plots
    across all available dates from 2025 to present.
    """
    site_cfg = cfg["sites"][site_name]
    bbox = site_cfg["bbox"]
    looks = int(site_cfg.get("looks", 2))
    invalid_mask_vals = cfg.get("gcov_defaults", {}).get("invalid_mask_values", [0, 255])
    stream_cfg = cfg.get("streaming", {})
    raw_dir = Path(cfg.get("raw_dir", "/content/nisar_raw"))
    site_out_dir = Path(cfg.get("out_dir", "out")) / site_name
    color_cfg = cfg.get("color_ranges", {})

    # Identify primary (track, frame, direction) group for strict same-geometry pairwise change
    primary_key, _ = select_primary_group(records)
    track, frame, direction = primary_key

    # Group all valid non-commissioning granules by acquisition date (YYYYMMDD) from 2025 to today
    by_date: Dict[str, List[GranuleRecord]] = defaultdict(list)
    for rec in records:
        if "_0005_NADV_" in rec.file_name:
            logger.warning(
                "Skipping commissioning granule %s (Frequency A disabled: 0005_NADV).",
                rec.file_name,
            )
            continue
        by_date[rec.date_str].append(rec)

    sorted_dates = sorted(by_date.keys())
    if max_dates is not None and max_dates > 0:
        sorted_dates = sorted_dates[:max_dates]

    logger.info(
        "Building GCOV time series for site='%s' across %d date(s) (%s to %s).",
        site_name,
        len(sorted_dates),
        sorted_dates[0] if sorted_dates else "none",
        sorted_dates[-1] if sorted_dates else "none",
    )

    first_primary_win: Optional[GCOVWindow] = None
    last_primary_win: Optional[GCOVWindow] = None
    last_window: Optional[GCOVWindow] = None
    last_prod: Optional[Dict[str, Any]] = None
    bounds_lonlat: List[float] = list(bbox)
    epsg_code: int = int(site_cfg.get("expected_epsg") or 4326)

    layer_dates: Dict[str, List[str]] = {
        "hh_db": [],
        "hv_db": [],
        "rvi": [],
        "pol_ratio_db": [],
        "freq_diff_db": [],
        "rgb_composite": [],
    }
    layer_cog_paths: Dict[str, List[str]] = {k: [] for k in layer_dates}
    layer_tile_templates: Dict[str, List[str]] = {k: [] for k in layer_dates}
    pol_flags_by_date: Dict[str, str] = {}

    for d_idx, date_str in enumerate(sorted_dates):
        day_records = sorted(
            by_date[date_str], key=lambda r: (r.group_key == primary_key, r.aoi_overlap_pct), reverse=True
        )
        day_windows: List[GCOVWindow] = []

        for rec in day_records:
            logger.info(
                "[Date %d/%d: %s] Streaming GCOV window for %s (track=%d, frame=%d, overlap=%.1f%%)",
                d_idx + 1,
                len(sorted_dates),
                date_str,
                rec.file_name,
                rec.track,
                rec.frame,
                rec.aoi_overlap_pct,
            )
            try:
                w_item: GCOVWindow = run_with_h5_retry(
                    url_or_path=rec.url,
                    token=token,
                    download=download,
                    raw_dir=raw_dir,
                    max_tries=int(stream_cfg.get("max_retries", 6)),
                    sleep_sec=float(stream_cfg.get("retry_sleep_sec", 10)),
                    block_size_mb=int(stream_cfg.get("block_size_mb", 4)),
                    reader_fn=lambda h5, r=rec: load_gcov_window(
                        h5,
                        bbox_lonlat=bbox,
                        looks=looks,
                        invalid_mask_values=invalid_mask_vals,
                        track=r.track,
                        frame=r.frame,
                        direction=r.direction,
                        time=r.time,
                        date_str=r.date_str,
                        file_name=r.file_name,
                    ),
                )
                day_windows.append(w_item)
                if rec.group_key == primary_key:
                    if first_primary_win is None:
                        first_primary_win = w_item
                    last_primary_win = w_item
            except Exception as exc:
                logger.error(
                    "Skipping granule %s on %s due to error: %s",
                    rec.file_name,
                    rec.time,
                    exc,
                )

        if not day_windows:
            continue

        win = _merge_same_date_gcov_windows(day_windows)
        prod = compute_single_date_products(win)
        co_key, cr_key = resolve_co_cross_pols(win.pols)
        epsg_code = win.epsg
        bounds_lonlat = utm_bounds_to_lonlat(win.x, win.y, win.epsg)
        pol_flags_by_date[date_str] = prod["polarization_flag"]

        for layer_name, arr in prod["layers"].items():
            if layer_name == "freq_diff_db" and not prod["has_dual_freq"]:
                continue

            cog_rel = Path("cog") / layer_name / f"{date_str}.tif"
            write_float32_cog(
                arr=arr,
                x=win.x,
                y=win.y,
                epsg=win.epsg,
                out_path=site_out_dir / cog_rel,
                metadata={
                    "date": date_str,
                    "time": win.time,
                    "track": win.track,
                    "frame": win.frame,
                    "direction": win.direction,
                    "polarization_flag": prod["polarization_flag"],
                    "freq_a_mhz": win.freq_a_mhz,
                    "freq_b_mhz": win.freq_b_mhz,
                },
            )
            append_to_zarr_stack(
                arr=arr,
                x=win.x,
                y=win.y,
                epsg=win.epsg,
                date_str=date_str,
                layer_name=layer_name,
                zarr_path=site_out_dir / "stack" / f"{layer_name}.zarr",
                attrs={"track": win.track, "frame": win.frame, "direction": win.direction},
            )

            c_meta = color_cfg.get(layer_name, {})
            tiles_rel = Path("tiles") / layer_name / date_str
            if layer_name == "hh_db" and co_key:
                export_xyz_png_tiles(
                    arr=win.pols[co_key],
                    x=win.x,
                    y=win.y,
                    src_epsg=win.epsg,
                    tiles_dir=site_out_dir / tiles_rel,
                    vmin=float(c_meta.get("vmin", -22.0)),
                    vmax=float(c_meta.get("vmax", 0.0)),
                    cmap_name=str(c_meta.get("cmap", "gray")),
                    convert_linear_to_db_after=True,
                )
            elif layer_name == "hv_db" and cr_key:
                export_xyz_png_tiles(
                    arr=win.pols[cr_key],
                    x=win.x,
                    y=win.y,
                    src_epsg=win.epsg,
                    tiles_dir=site_out_dir / tiles_rel,
                    vmin=float(c_meta.get("vmin", -30.0)),
                    vmax=float(c_meta.get("vmax", -8.0)),
                    cmap_name=str(c_meta.get("cmap", "YlGn")),
                    convert_linear_to_db_after=True,
                )
            else:
                export_xyz_png_tiles(
                    arr=arr,
                    x=win.x,
                    y=win.y,
                    src_epsg=win.epsg,
                    tiles_dir=site_out_dir / tiles_rel,
                    vmin=float(c_meta.get("vmin", -4.0 if layer_name == "freq_diff_db" else 0.0)),
                    vmax=float(c_meta.get("vmax", 4.0 if layer_name == "freq_diff_db" else 1.0)),
                    cmap_name=str(c_meta.get("cmap", "coolwarm" if layer_name == "freq_diff_db" else "YlGn")),
                    convert_linear_to_db_after=False,
                )

            layer_dates[layer_name].append(date_str)
            layer_cog_paths[layer_name].append(str(cog_rel))
            layer_tile_templates[layer_name].append(f"{tiles_rel}/{{z}}/{{x}}/{{y}}.png")

        # Export 3-band Polarimetric RGB Composite COG + XYZ PNG tiles
        rgb_cog_rel = Path("cog") / "rgb_composite" / f"{date_str}.tif"
        write_rgb_cog(
            rgb_arr=prod["rgb"],
            x=win.x,
            y=win.y,
            epsg=win.epsg,
            out_path=site_out_dir / rgb_cog_rel,
            metadata={"date": date_str, "polarization_flag": prod["polarization_flag"]},
        )
        rgb_tiles_rel = Path("tiles") / "rgb_composite" / date_str
        export_rgb_xyz_png_tiles(
            rgb_arr=prod["rgb"],
            x=win.x,
            y=win.y,
            src_epsg=win.epsg,
            tiles_dir=site_out_dir / rgb_tiles_rel,
        )
        layer_dates["rgb_composite"].append(date_str)
        layer_cog_paths["rgb_composite"].append(str(rgb_cog_rel))
        layer_tile_templates["rgb_composite"].append(f"{rgb_tiles_rel}/{{z}}/{{x}}/{{y}}.png")

        sample_pts = site_cfg.get("sample_points", [])
        if sample_pts:
            date_iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            update_points_timeseries_json(
                points_cfg=sample_pts,
                date_iso=date_iso,
                layers=prod["layers"],
                x=win.x,
                y=win.y,
                epsg=win.epsg,
                out_json_path=site_out_dir / "timeseries" / "points.json",
            )

        plot_pol_and_freq_explorer(
            win=win,
            prod=prod,
            site_out_dir=site_out_dir,
            site_title=site_cfg.get("title", site_name),
        )
        last_window = win
        last_prod = prod

    catalog_entries: Dict[str, Dict[str, Any]] = {}
    for layer_name, dates in layer_dates.items():
        if not dates:
            continue
        c_meta = color_cfg.get(layer_name, {})
        entry: Dict[str, Any] = {
            "id": layer_name,
            "title": c_meta.get("title", layer_name),
            "units": c_meta.get("units", ""),
            "dates": dates,
            "bounds": list(bbox),
            "crs": f"EPSG:{epsg_code}",
            "color_range": [c_meta.get("vmin", -25.0), c_meta.get("vmax", 0.0)],
            "colormap": c_meta.get("cmap", "viridis"),
            "cog_files": layer_cog_paths[layer_name],
            "tile_templates": layer_tile_templates[layer_name],
            "track": track,
            "frame": frame,
            "direction": direction,
            "polarization_flags": pol_flags_by_date,
        }
        if layer_name != "rgb_composite":
            entry["zarr_stack"] = f"stack/{layer_name}.zarr"
        catalog_entries[layer_name] = entry

    if (
        first_primary_win is not None
        and last_primary_win is not None
        and first_primary_win.date_str != last_primary_win.date_str
    ):
        try:
            change_res = compute_pairwise_change(first_primary_win, last_primary_win)
            pair_tag = f"{first_primary_win.date_str}_{last_primary_win.date_str}"
            for ch_name, ch_arr in change_res["layers"].items():
                cog_rel = Path("cog") / ch_name / f"{pair_tag}.tif"
                write_float32_cog(
                    arr=ch_arr,
                    x=change_res["x"],
                    y=change_res["y"],
                    epsg=change_res["epsg"],
                    out_path=site_out_dir / cog_rel,
                    metadata={
                        "pre_date": first_primary_win.date_str,
                        "post_date": last_primary_win.date_str,
                        "track": track,
                        "frame": frame,
                        "direction": direction,
                        "cross_pol_gap": change_res["cross_pol_gap"],
                    },
                )
                c_meta = color_cfg.get(ch_name, {})
                tiles_rel = Path("tiles") / ch_name / pair_tag
                export_xyz_png_tiles(
                    arr=ch_arr,
                    x=change_res["x"],
                    y=change_res["y"],
                    src_epsg=change_res["epsg"],
                    tiles_dir=site_out_dir / tiles_rel,
                    vmin=float(c_meta.get("vmin", -6.0)),
                    vmax=float(c_meta.get("vmax", 6.0)),
                    cmap_name=str(c_meta.get("cmap", "RdBu")),
                )
                catalog_entries[ch_name] = {
                    "id": ch_name,
                    "title": c_meta.get("title", ch_name),
                    "units": c_meta.get("units", ""),
                    "dates": [pair_tag],
                    "bounds": bounds_lonlat,
                    "crs": f"EPSG:{epsg_code}",
                    "color_range": [c_meta.get("vmin", -6.0), c_meta.get("vmax", 6.0)],
                    "colormap": c_meta.get("cmap", "RdBu"),
                    "cog_files": [str(cog_rel)],
                    "tile_templates": [f"{tiles_rel}/{{z}}/{{x}}/{{y}}.png"],
                    "track": track,
                    "frame": frame,
                    "direction": direction,
                    "polarization_flags": {
                        pair_tag: "CROSS_POL_GAP" if change_res["cross_pol_gap"] else "DUAL_POL"
                    },
                }
        except Exception as exc:
            logger.warning("Skipping pairwise change export: %s", exc)

    update_catalog_json(
        catalog_path=site_out_dir / "catalog.json",
        site_name=site_name,
        site_title=site_cfg.get("title", site_name),
        layer_entries=catalog_entries,
    )

    if last_window is not None and last_prod is not None:
        plot_human_friendly_guide(site_out_dir, show_inline=False)


def build_gunw_for_site(
    site_name: str,
    cfg: Dict[str, Any],
    records: Sequence[GranuleRecord],
    token: str,
    download: bool = False,
    max_dates: Optional[int] = None,
) -> None:
    """Build GUNW coherence, cumulative LOS/vertical displacement, subsidence rate, and vectors."""
    site_cfg = cfg["sites"][site_name]
    bbox = site_cfg["bbox"]
    looks = int(site_cfg.get("looks", 2))
    gunw_cfg = cfg.get("gunw_defaults", {})
    coh_thresh = float(
        site_cfg.get("coherence_threshold", gunw_cfg.get("coherence_threshold", 0.4))
    )
    wavelength_m = float(gunw_cfg.get("wavelength_m", 0.24))
    min_dates_rate = int(gunw_cfg.get("min_dates_for_rate", 3))
    flat_terrain = bool(site_cfg.get("flat_terrain", False))
    ref_box = site_cfg.get("reference_box")
    stream_cfg = cfg.get("streaming", {})
    raw_dir = Path(cfg.get("raw_dir", "/content/nisar_raw"))
    site_out_dir = Path(cfg.get("out_dir", "out")) / site_name
    color_cfg = cfg.get("color_ranges", {})

    group_key, group_records = select_primary_group(records)
    track, frame, direction = group_key
    if max_dates is not None and max_dates > 0:
        group_records = group_records[:max_dates]

    windows: List[GUNWWindow] = []
    for idx, rec in enumerate(group_records):
        logger.info("[%d/%d] Streaming GUNW window for %s", idx + 1, len(group_records), rec.file_name)
        win: GUNWWindow = run_with_h5_retry(
            url_or_path=rec.url,
            token=token,
            download=download,
            raw_dir=raw_dir,
            max_tries=int(stream_cfg.get("max_retries", 6)),
            sleep_sec=float(stream_cfg.get("retry_sleep_sec", 10)),
            block_size_mb=int(stream_cfg.get("block_size_mb", 4)),
            reader_fn=lambda h5, r=rec: load_gunw_window(
                h5,
                bbox_lonlat=bbox,
                looks=looks,
                coherence_threshold=coh_thresh,
                track=r.track,
                frame=r.frame,
                direction=r.direction,
                file_name=r.file_name,
                fallback_start=r.time,
                fallback_stop=r.secondary_time,
            ),
        )
        windows.append(win)

    stack_res = process_gunw_stack(
        windows=windows,
        ref_bbox_lonlat=ref_box,
        wavelength_m=wavelength_m,
        flat_terrain=flat_terrain,
        min_dates_for_rate=min_dates_rate,
    )

    x, y, epsg = stack_res["x"], stack_res["y"], stack_res["epsg"]
    dates: List[str] = stack_res["dates"]
    bounds_lonlat = utm_bounds_to_lonlat(x, y, epsg)
    catalog_entries: Dict[str, Dict[str, Any]] = {}

    los_cogs: List[str] = []
    los_tiles: List[str] = []
    c_los = color_cfg.get("los_disp_mm", {})
    for i, d_str in enumerate(dates):
        arr = stack_res["cum_los_mm"][i]
        cog_rel = Path("cog") / "los_disp_mm" / f"{d_str}.tif"
        write_float32_cog(arr, x, y, epsg, site_out_dir / cog_rel, metadata={"date": d_str})
        append_to_zarr_stack(arr, x, y, epsg, d_str, "los_disp_mm", site_out_dir / "stack" / "los_disp_mm.zarr")
        tiles_rel = Path("tiles") / "los_disp_mm" / d_str
        export_xyz_png_tiles(
            arr=arr,
            x=x,
            y=y,
            src_epsg=epsg,
            tiles_dir=site_out_dir / tiles_rel,
            vmin=float(c_los.get("vmin", -100.0)),
            vmax=float(c_los.get("vmax", 100.0)),
            cmap_name=str(c_los.get("cmap", "RdBu_r")),
        )
        los_cogs.append(str(cog_rel))
        los_tiles.append(f"{tiles_rel}/{{z}}/{{x}}/{{y}}.png")

        sample_pts = site_cfg.get("sample_points", [])
        if sample_pts:
            date_iso = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
            coh_slice = stack_res["coherence_pairs"][min(i, len(stack_res["coherence_pairs"]) - 1)]
            update_points_timeseries_json(
                points_cfg=sample_pts,
                date_iso=date_iso,
                layers={"los_disp_mm": arr, "coherence": coh_slice},
                x=x,
                y=y,
                epsg=epsg,
                out_json_path=site_out_dir / "timeseries" / "points.json",
            )

    catalog_entries["los_disp_mm"] = {
        "id": "los_disp_mm",
        "title": c_los.get("title", "Cumulative LOS Displacement"),
        "units": "mm",
        "dates": dates,
        "bounds": bounds_lonlat,
        "crs": f"EPSG:{epsg}",
        "color_range": [c_los.get("vmin", -100.0), c_los.get("vmax", 100.0)],
        "colormap": c_los.get("cmap", "RdBu_r"),
        "cog_files": los_cogs,
        "tile_templates": los_tiles,
        "zarr_stack": "stack/los_disp_mm.zarr",
        "track": track,
        "frame": frame,
        "direction": direction,
    }

    if stack_res["cum_vert_mm"] is not None:
        vert_cogs: List[str] = []
        vert_tiles: List[str] = []
        c_vert = color_cfg.get("vert_disp_mm", {})
        for i, d_str in enumerate(dates):
            v_arr = stack_res["cum_vert_mm"][i]
            cog_rel = Path("cog") / "vert_disp_mm" / f"{d_str}.tif"
            write_float32_cog(v_arr, x, y, epsg, site_out_dir / cog_rel, metadata={"date": d_str})
            tiles_rel = Path("tiles") / "vert_disp_mm" / d_str
            export_xyz_png_tiles(
                arr=v_arr,
                x=x,
                y=y,
                src_epsg=epsg,
                tiles_dir=site_out_dir / tiles_rel,
                vmin=float(c_vert.get("vmin", -120.0)),
                vmax=float(c_vert.get("vmax", 120.0)),
                cmap_name=str(c_vert.get("cmap", "RdBu_r")),
            )
            vert_cogs.append(str(cog_rel))
            vert_tiles.append(f"{tiles_rel}/{{z}}/{{x}}/{{y}}.png")
        catalog_entries["vert_disp_mm"] = {
            "id": "vert_disp_mm",
            "title": c_vert.get("title", "Cumulative Vertical Displacement"),
            "units": "mm",
            "dates": dates,
            "bounds": bounds_lonlat,
            "crs": f"EPSG:{epsg}",
            "color_range": [c_vert.get("vmin", -120.0), c_vert.get("vmax", 120.0)],
            "colormap": c_vert.get("cmap", "RdBu_r"),
            "cog_files": vert_cogs,
            "tile_templates": vert_tiles,
            "track": track,
            "frame": frame,
            "direction": direction,
        }

    rate_tag = f"{dates[0]}_{dates[-1]}"
    rate_cog_rel = Path("cog") / "disp_rate_mm_yr" / f"{rate_tag}.tif"
    write_float32_cog(stack_res["rate_mm_yr"], x, y, epsg, site_out_dir / rate_cog_rel)
    vec_rel = Path("vectors") / "subsidence_rates.geojson"
    export_subsidence_points_geojson(
        rate_mm_yr=stack_res["rate_mm_yr"],
        cum_disp_mm=stack_res["cum_los_mm"][-1],
        x=x,
        y=y,
        epsg=epsg,
        out_geojson_path=site_out_dir / vec_rel,
    )
    c_rate = color_cfg.get("disp_rate_mm_yr", {})
    rate_tiles_rel = Path("tiles") / "disp_rate_mm_yr" / rate_tag
    export_xyz_png_tiles(
        arr=stack_res["rate_mm_yr"],
        x=x,
        y=y,
        src_epsg=epsg,
        tiles_dir=site_out_dir / rate_tiles_rel,
        vmin=float(c_rate.get("vmin", -50.0)),
        vmax=float(c_rate.get("vmax", 50.0)),
        cmap_name=str(c_rate.get("cmap", "RdBu_r")),
    )
    catalog_entries["disp_rate_mm_yr"] = {
        "id": "disp_rate_mm_yr",
        "title": c_rate.get("title", "Linear Displacement Rate"),
        "units": "mm/yr",
        "dates": [rate_tag],
        "bounds": bounds_lonlat,
        "crs": f"EPSG:{epsg}",
        "color_range": [c_rate.get("vmin", -50.0), c_rate.get("vmax", 50.0)],
        "colormap": c_rate.get("cmap", "RdBu_r"),
        "cog_files": [str(rate_cog_rel)],
        "tile_templates": [f"{rate_tiles_rel}/{{z}}/{{x}}/{{y}}.png"],
        "vector_file": str(vec_rel),
        "track": track,
        "frame": frame,
        "direction": direction,
    }

    update_catalog_json(site_out_dir / "catalog.json", site_name, site_cfg.get("title", site_name), catalog_entries)
    plot_human_friendly_guide(site_out_dir, show_inline=False)


def build_goff_for_site(
    site_name: str,
    cfg: Dict[str, Any],
    records: Sequence[GranuleRecord],
    token: str,
    download: bool = False,
    max_dates: Optional[int] = None,
) -> None:
    """Build GOFF glacier surface speed, direction, RGI-clipped rasters, and vector arrows GeoJSON."""
    site_cfg = cfg["sites"][site_name]
    bbox = site_cfg["bbox"]
    looks = int(site_cfg.get("looks", 2))
    rgi_geojson = site_cfg.get("rgi_geojson")
    stream_cfg = cfg.get("streaming", {})
    raw_dir = Path(cfg.get("raw_dir", "/content/nisar_raw"))
    site_out_dir = Path(cfg.get("out_dir", "out")) / site_name
    color_cfg = cfg.get("color_ranges", {})

    group_key, group_records = select_primary_group(records)
    track, frame, direction = group_key
    if max_dates is not None and max_dates > 0:
        group_records = group_records[:max_dates]

    speed_cogs: List[str] = []
    speed_tiles: List[str] = []
    pair_dates: List[str] = []
    bounds_lonlat: List[float] = list(bbox)
    epsg_code: int = int(site_cfg.get("expected_epsg") or 4326)
    vec_rel = Path("vectors") / "glacier_velocity.geojson"
    c_vel = color_cfg.get("velocity_m_yr", {})

    for idx, rec in enumerate(group_records):
        logger.info("[%d/%d] Streaming GOFF window for %s", idx + 1, len(group_records), rec.file_name)
        win: GOFFWindow = run_with_h5_retry(
            url_or_path=rec.url,
            token=token,
            download=download,
            raw_dir=raw_dir,
            max_tries=int(stream_cfg.get("max_retries", 6)),
            sleep_sec=float(stream_cfg.get("retry_sleep_sec", 10)),
            block_size_mb=int(stream_cfg.get("block_size_mb", 4)),
            reader_fn=lambda h5, r=rec: load_goff_window(
                h5,
                bbox_lonlat=bbox,
                looks=looks,
                track=r.track,
                frame=r.frame,
                direction=r.direction,
                file_name=r.file_name,
                fallback_start=r.time,
                fallback_stop=r.secondary_time,
            ),
        )
        inc_used: np.ndarray | float = win.incidence_angle if win.incidence_angle is not None else 38.0
        vel = compute_velocity_vectors(
            slant_range_offset_m=win.slant_range_offset,
            along_track_offset_m=win.along_track_offset,
            dt_days=win.dt_days,
            incidence_angle_deg=inc_used,
            flight_direction=win.direction or direction,
        )
        clipped, x_c, y_c = clip_to_glacier_outlines(vel, win.x, win.y, win.epsg, rgi_geojson)
        epsg_code = win.epsg
        bounds_lonlat = utm_bounds_to_lonlat(x_c, y_c, epsg_code)
        pair_tag = f"{win.ref_date}_{win.sec_date}"

        spd_cog_rel = Path("cog") / "velocity_m_yr" / f"{pair_tag}.tif"
        write_float32_cog(clipped["speed_m_yr"], x_c, y_c, epsg_code, site_out_dir / spd_cog_rel)
        dir_cog_rel = Path("cog") / "velocity_dir_deg" / f"{pair_tag}.tif"
        write_float32_cog(clipped["direction_deg"], x_c, y_c, epsg_code, site_out_dir / dir_cog_rel)

        tiles_rel = Path("tiles") / "velocity_m_yr" / pair_tag
        export_xyz_png_tiles(
            arr=clipped["speed_m_yr"],
            x=x_c,
            y=y_c,
            src_epsg=epsg_code,
            tiles_dir=site_out_dir / tiles_rel,
            vmin=float(c_vel.get("vmin", 0.0)),
            vmax=float(c_vel.get("vmax", 150.0)),
            cmap_name=str(c_vel.get("cmap", "plasma")),
        )

        export_velocity_vectors_geojson(
            v_east_m_yr=clipped["v_east_m_yr"],
            v_north_m_yr=clipped["v_north_m_yr"],
            speed_m_yr=clipped["speed_m_yr"],
            direction_deg=clipped["direction_deg"],
            x=x_c,
            y=y_c,
            epsg=epsg_code,
            out_geojson_path=site_out_dir / vec_rel,
        )
        speed_cogs.append(str(spd_cog_rel))
        speed_tiles.append(f"{tiles_rel}/{{z}}/{{x}}/{{y}}.png")
        pair_dates.append(pair_tag)

    catalog_entries = {
        "velocity_m_yr": {
            "id": "velocity_m_yr",
            "title": c_vel.get("title", "Glacier Surface Speed"),
            "units": "m/yr",
            "dates": pair_dates,
            "bounds": bounds_lonlat,
            "crs": f"EPSG:{epsg_code}",
            "color_range": [c_vel.get("vmin", 0.0), c_vel.get("vmax", 150.0)],
            "colormap": c_vel.get("cmap", "plasma"),
            "cog_files": speed_cogs,
            "tile_templates": speed_tiles,
            "vector_file": str(vec_rel),
            "track": track,
            "frame": frame,
            "direction": direction,
        }
    }
    update_catalog_json(site_out_dir / "catalog.json", site_name, site_cfg.get("title", site_name), catalog_entries)
    plot_human_friendly_guide(site_out_dir, show_inline=False)


def build_sme2_for_site(
    site_name: str,
    cfg: Dict[str, Any],
    records: Sequence[GranuleRecord],
    token: str,
    download: bool = False,
    max_dates: Optional[int] = None,
) -> None:
    """Build L3 SME2 volumetric soil moisture COGs, Zarr stack, tiles, and point time series."""
    require_verified_layer(SME2_SOIL_MOISTURE, product="SME2")
    require_verified_layer(SME2_QUALITY_FLAG, product="SME2")
    site_cfg = cfg["sites"][site_name]
    bbox = site_cfg["bbox"]
    looks = int(site_cfg.get("looks", 1))
    stream_cfg = cfg.get("streaming", {})
    raw_dir = Path(cfg.get("raw_dir", "/content/nisar_raw"))
    site_out_dir = Path(cfg.get("out_dir", "out")) / site_name
    color_cfg = cfg.get("color_ranges", {})

    group_key, group_records = select_primary_group(records)
    track, frame, direction = group_key
    if max_dates is not None and max_dates > 0:
        group_records = group_records[:max_dates]

    sm_cogs: List[str] = []
    sm_tiles: List[str] = []
    sm_dates: List[str] = []
    bounds_lonlat: List[float] = list(bbox)
    epsg_code: int = 6933
    c_sm = color_cfg.get("soil_moisture", {})

    for idx, rec in enumerate(group_records):
        logger.info("[%d/%d] Streaming SME2 window for %s", idx + 1, len(group_records), rec.file_name)

        def _read_sm(h5: Any) -> Any:
            x_full = h5[SME2_X][:]
            y_full = h5[SME2_Y][:]
            epsg = read_epsg_from_projection(h5[SME2_PROJECTION])
            sy, sx, _ = bbox_to_utm_slices(x_full, y_full, bbox, epsg)
            sm = np.asarray(h5[SME2_SOIL_MOISTURE][sy, sx], dtype=np.float32)
            bad = ~np.isfinite(sm) | (sm <= 0.0) | (sm > 1.0)
            if SME2_QUALITY_FLAG in h5:
                qflag = np.asarray(h5[SME2_QUALITY_FLAG][sy, sx])
                bad |= qflag != 0
            sm[bad] = np.nan
            return block_mean(sm, looks), coord_mean(x_full[sx], looks), coord_mean(y_full[sy], looks), epsg

        sm_arr, x_w, y_w, epsg_code = run_with_h5_retry(
            url_or_path=rec.url,
            reader_fn=_read_sm,
            token=token,
            download=download,
            raw_dir=raw_dir,
            max_tries=int(stream_cfg.get("max_retries", 6)),
            sleep_sec=float(stream_cfg.get("retry_sleep_sec", 10)),
            block_size_mb=int(stream_cfg.get("block_size_mb", 4)),
        )
        bounds_lonlat = utm_bounds_to_lonlat(x_w, y_w, epsg_code)
        cog_rel = Path("cog") / "soil_moisture" / f"{rec.date_str}.tif"
        write_float32_cog(sm_arr, x_w, y_w, epsg_code, site_out_dir / cog_rel)
        append_to_zarr_stack(sm_arr, x_w, y_w, epsg_code, rec.date_str, "soil_moisture", site_out_dir / "stack" / "soil_moisture.zarr")

        tiles_rel = Path("tiles") / "soil_moisture" / rec.date_str
        export_xyz_png_tiles(
            arr=sm_arr,
            x=x_w,
            y=y_w,
            src_epsg=epsg_code,
            tiles_dir=site_out_dir / tiles_rel,
            vmin=float(c_sm.get("vmin", 0.02)),
            vmax=float(c_sm.get("vmax", 0.50)),
            cmap_name=str(c_sm.get("cmap", "YlGnBu")),
        )
        sm_cogs.append(str(cog_rel))
        sm_tiles.append(f"{tiles_rel}/{{z}}/{{x}}/{{y}}.png")
        sm_dates.append(rec.date_str)

        sample_pts = site_cfg.get("sample_points", [])
        if sample_pts:
            update_points_timeseries_json(
                points_cfg=sample_pts,
                date_iso=rec.time[:10],
                layers={"soil_moisture": sm_arr},
                x=x_w,
                y=y_w,
                epsg=epsg_code,
                out_json_path=site_out_dir / "timeseries" / "points.json",
            )

    update_catalog_json(
        site_out_dir / "catalog.json",
        site_name,
        site_cfg.get("title", site_name),
        {
            "soil_moisture": {
                "id": "soil_moisture",
                "title": c_sm.get("title", "Volumetric Soil Moisture"),
                "units": "m3/m3",
                "dates": sm_dates,
                "bounds": bounds_lonlat,
                "crs": f"EPSG:{epsg_code}",
                "color_range": [c_sm.get("vmin", 0.02), c_sm.get("vmax", 0.50)],
                "colormap": c_sm.get("cmap", "YlGnBu"),
                "cog_files": sm_cogs,
                "tile_templates": sm_tiles,
                "zarr_stack": "stack/soil_moisture.zarr",
                "track": track,
                "frame": frame,
                "direction": direction,
            }
        },
    )
    plot_human_friendly_guide(site_out_dir, show_inline=False)


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI argument parser."""
    ap = argparse.ArgumentParser(
        prog="run_site.py",
        description="NASA-ISRO NISAR data-pull, preprocessing, and human-friendly plotting CLI.",
    )
    ap.add_argument("cmd", choices=["search", "inspect", "build", "plot"])
    ap.add_argument(
        "--site",
        choices=["california", "nile_delta", "himalaya"],
        default="california",
    )
    ap.add_argument(
        "--products",
        nargs="+",
        default=["GCOV"],
        choices=["GCOV", "GUNW", "GOFF", "SME2"],
    )
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--start", help="Override ISO start date (YYYY-MM-DD).")
    ap.add_argument("--end", help="Override ISO end date (YYYY-MM-DD).")
    ap.add_argument("--track", type=int, help="Filter by pathNumber / track.")
    ap.add_argument("--frame", type=int, help="Filter by frameNumber.")
    ap.add_argument("--direction", choices=["A", "D", "ASC", "DESC"])
    ap.add_argument("--idx", type=int, default=0, help="Granule index for inspect.")
    ap.add_argument("--max-dates", type=int, default=None, help="Cap number of dates to build.")
    ap.add_argument("--download", action="store_true", help="Download full granules with .part resume.")
    ap.add_argument("--config", type=Path, default=None, help="Path to custom config.yaml.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    site_cfg = cfg["sites"][args.site]
    if args.bbox:
        site_cfg["bbox"] = list(args.bbox)
    bbox = site_cfg["bbox"]
    start = args.start or site_cfg.get("start", "2025-07-30")
    end = args.end or site_cfg.get(
        "end", datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    site_out_dir = Path(cfg.get("out_dir", "out")) / args.site

    if args.cmd == "plot":
        if (site_out_dir / "catalog.json").exists():
            plot_human_friendly_guide(site_out_dir, show_inline=False)
        else:
            logger.info(
                "No existing catalog in %s; streaming latest GCOV granule to generate explorer plot.",
                site_out_dir,
            )
            token = get_edl_token()
            records = search_nisar(bbox=bbox, processing_level="GCOV", start=start, end=end)
            _, grp = select_primary_group(records)
            target = next((r for r in reversed(grp) if "_0005_NADV_" not in r.file_name), grp[-1])
            win = run_with_h5_retry(
                url_or_path=target.url,
                token=token,
                download=args.download,
                reader_fn=lambda h5: load_gcov_window(
                    h5,
                    bbox_lonlat=bbox,
                    looks=int(site_cfg.get("looks", 4)),
                    track=target.track,
                    frame=target.frame,
                    direction=target.direction,
                    time=target.time,
                    date_str=target.date_str,
                    file_name=target.file_name,
                ),
            )
            prod = compute_single_date_products(win)
            plot_pol_and_freq_explorer(
                win=win,
                prod=prod,
                site_out_dir=site_out_dir,
                site_title=site_cfg.get("title", args.site),
            )
        return 0

    for product in args.products:
        records = search_nisar(
            bbox=bbox,
            processing_level=product,
            start=start,
            end=end,
            track=args.track,
            frame=args.frame,
            direction=args.direction,
        )
        print_search_table(records, product=product)

        if not records:
            logger.warning(
                "No NISAR %s granules found for site='%s' (bbox=%s, %s to %s).",
                product,
                args.site,
                bbox,
                start,
                end,
            )
            continue

        if args.cmd == "search":
            groups = group_by_track_frame(records)
            for (trk, frm, dr), grp in groups.items():
                logger.info(
                    "Group (track=%d, frame=%d, dir=%s): %d acquisition(s)",
                    trk,
                    frm,
                    dr,
                    len(grp),
                )
            continue

        token = get_edl_token()
        stream_cfg = cfg.get("streaming", {})
        raw_dir = Path(cfg.get("raw_dir", "/content/nisar_raw"))

        if args.cmd == "inspect":
            target = records[args.idx]
            logger.info("Inspecting %s granule: %s", product, target.file_name)

            def _do_inspect(h5: Any) -> None:
                inspect_granule(h5, print_output=True)
                if product == "GCOV":
                    win = load_gcov_window(
                        h5,
                        bbox_lonlat=bbox,
                        looks=int(site_cfg.get("looks", 2)),
                        track=target.track,
                        frame=target.frame,
                        direction=target.direction,
                        time=target.time,
                        date_str=target.date_str,
                        file_name=target.file_name,
                    )
                    for k, v in win.pols.items():
                        logger.info(
                            "GCOV AOI window %s: shape=%s, median_linear=%.5f, has_dual_freq=%s",
                            k,
                            v.shape,
                            float(np.nanmedian(v)),
                            win.has_dual_freq,
                        )

            run_with_h5_retry(
                url_or_path=target.url,
                reader_fn=_do_inspect,
                token=token,
                download=args.download,
                raw_dir=raw_dir,
                max_tries=int(stream_cfg.get("max_retries", 6)),
                sleep_sec=float(stream_cfg.get("retry_sleep_sec", 10)),
                block_size_mb=int(stream_cfg.get("block_size_mb", 4)),
            )
            continue

        if args.cmd == "build":
            if product == "GCOV":
                build_gcov_for_site(
                    args.site, cfg, records, token, args.download, args.max_dates
                )
            elif product == "GUNW":
                build_gunw_for_site(
                    args.site, cfg, records, token, args.download, args.max_dates
                )
            elif product == "GOFF":
                build_goff_for_site(
                    args.site, cfg, records, token, args.download, args.max_dates
                )
            elif product == "SME2":
                build_sme2_for_site(
                    args.site, cfg, records, token, args.download, args.max_dates
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())