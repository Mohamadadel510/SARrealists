#!/usr/bin/env python3
"""
CLI entrypoint for NISAR Space Apps preprocessing pipeline:
  python -m nisar_pipeline.run_site search  --site california --products GCOV GUNW
  python -m nisar_pipeline.run_site inspect --site california --products GCOV --idx 0
  python -m nisar_pipeline.run_site build   --site california --products GCOV
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pyproj import Transformer
import yaml

from nisar_pipeline.gcov import (
    append_to_zarr_stack,
    compute_pairwise_change,
    compute_single_date_products,
    update_points_timeseries_json,
    write_float32_cog,
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
from nisar_pipeline.window import GCOVWindow, load_gcov_window

logger = logging.getLogger("nisar_pipeline")


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML pipeline configuration."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def utm_bounds_to_lonlat(
    x: np.ndarray, y: np.ndarray, epsg: int
) -> List[float]:
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


def update_catalog_json(
    catalog_path: Path,
    site_name: str,
    site_title: str,
    layer_entries: Dict[str, Dict[str, Any]],
) -> Path:
    """
    Create or update `out/<site>/catalog.json` with metadata for every generated layer:
    id, title, units, dates, bounds (lon/lat), crs, color range, colormap, file paths,
    track/frame/direction, polarization flags.
    """
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog: Dict[str, Any] = {
        "site": site_name,
        "title": site_title,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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

    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
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
    Process one GCOV date at a time for the primary (track, frame, direction) group:
    - Stream only the AOI window
    - Multilook in linear power, convert to dB, compute RVI
    - Write float32 COGs in native UTM (`out/<site>/cog/<layer>/<YYYYMMDD>.tif`)
    - Incrementally append to Zarr stack (`out/<site>/stack/<layer>.zarr`)
    - Extract point time series (`out/<site>/timeseries/points.json`)
    - Compute pre/post change between earliest and latest dates in the group
    - Write `out/<site>/catalog.json`
    """
    site_cfg = cfg["sites"][site_name]
    bbox = site_cfg["bbox"]
    looks = int(site_cfg.get("looks", 2))
    invalid_mask_vals = cfg.get("gcov_defaults", {}).get("invalid_mask_values", [0, 255])
    stream_cfg = cfg.get("streaming", {})
    raw_dir = Path(cfg.get("raw_dir", "/content/nisar_raw"))
    site_out_dir = Path(cfg.get("out_dir", "out")) / site_name
    color_cfg = cfg.get("color_ranges", {})

    group_key, group_records = select_primary_group(records)
    track, frame, direction = group_key
    if max_dates is not None and max_dates > 0:
        group_records = group_records[:max_dates]

    logger.info(
        "Building GCOV for site='%s' on group (track=%d, frame=%d, dir=%s) with %d date(s).",
        site_name,
        track,
        frame,
        direction,
        len(group_records),
    )

    first_window: Optional[GCOVWindow] = None
    last_window: Optional[GCOVWindow] = None
    bounds_lonlat: List[float] = list(bbox)
    epsg_code: int = int(site_cfg.get("expected_epsg") or 4326)

    layer_dates: Dict[str, List[str]] = {
        "hh_db": [],
        "hv_db": [],
        "rvi": [],
        "pol_ratio_db": [],
    }
    layer_cog_paths: Dict[str, List[str]] = {k: [] for k in layer_dates}
    pol_flags_by_date: Dict[str, str] = {}

    for idx, rec in enumerate(group_records):
        # Skip 0005_NADV commissioning granules that have Frequency A disabled
        if "_0005_NADV_" in rec.file_name:
            logger.warning(
                "[%d/%d] Skipping commissioning granule %s (Frequency A disabled: 0005_NADV).",
                idx + 1,
                len(group_records),
                rec.file_name,
            )
            continue

        logger.info(
            "[%d/%d] Streaming GCOV window for %s (%s)",
            idx + 1,
            len(group_records),
            rec.time,
            rec.file_name,
        )
        try:
            win: GCOVWindow = run_with_h5_retry(
                url_or_path=rec.url,
                token=token,
                download=download,
                raw_dir=raw_dir,
                max_tries=int(stream_cfg.get("max_retries", 6)),
                sleep_sec=float(stream_cfg.get("retry_sleep_sec", 10)),
                block_size_mb=int(stream_cfg.get("block_size_mb", 4)),
                reader_fn=lambda h5: load_gcov_window(
                    h5,
                    bbox_lonlat=bbox,
                    looks=looks,
                    invalid_mask_values=invalid_mask_vals,
                    track=rec.track,
                    frame=rec.frame,
                    direction=rec.direction,
                    time=rec.time,
                    date_str=rec.date_str,
                    file_name=rec.file_name,
                ),
            )
        except Exception as exc:
            logger.error(
                "[%d/%d] Skipping granule %s on %s due to error: %s",
                idx + 1,
                len(group_records),
                rec.file_name,
                rec.time,
                exc,
            )
            continue

        prod = compute_single_date_products(win)
        epsg_code = win.epsg
        bounds_lonlat = utm_bounds_to_lonlat(win.x, win.y, win.epsg)
        pol_flags_by_date[rec.date_str] = prod["polarization_flag"]

        for layer_name, arr in prod["layers"].items():
            if not np.any(np.isfinite(arr)):
                logger.info(
                    "Layer '%s' is all-NaN on %s (polarization=%s); writing NaN COG & Zarr slice.",
                    layer_name,
                    rec.date_str,
                    prod["polarization_flag"],
                )
            cog_rel = Path("cog") / layer_name / f"{rec.date_str}.tif"
            cog_abs = site_out_dir / cog_rel
            write_float32_cog(
                arr=arr,
                x=win.x,
                y=win.y,
                epsg=win.epsg,
                out_path=cog_abs,
                metadata={
                    "date": rec.date_str,
                    "time": rec.time,
                    "track": rec.track,
                    "frame": rec.frame,
                    "direction": rec.direction,
                    "polarization_flag": prod["polarization_flag"],
                },
            )
            zarr_abs = site_out_dir / "stack" / f"{layer_name}.zarr"
            append_to_zarr_stack(
                arr=arr,
                x=win.x,
                y=win.y,
                epsg=win.epsg,
                date_str=rec.date_str,
                layer_name=layer_name,
                zarr_path=zarr_abs,
                attrs={
                    "track": rec.track,
                    "frame": rec.frame,
                    "direction": rec.direction,
                },
            )
            layer_dates[layer_name].append(rec.date_str)
            layer_cog_paths[layer_name].append(str(cog_rel))

        # Update point time series JSON incrementally
        sample_pts = site_cfg.get("sample_points", [])
        if sample_pts:
            update_points_timeseries_json(
                points_cfg=sample_pts,
                date_iso=rec.time[:10],
                layers=prod["layers"],
                x=win.x,
                y=win.y,
                epsg=win.epsg,
                out_json_path=site_out_dir / "timeseries" / "points.json",
            )

        # Keep only earliest and latest windows in memory for pairwise change
        if first_window is None:
            first_window = win
        last_window = win

    catalog_entries: Dict[str, Dict[str, Any]] = {}
    for layer_name, dates in layer_dates.items():
        c_meta = color_cfg.get(layer_name, {})
        catalog_entries[layer_name] = {
            "id": layer_name,
            "title": c_meta.get("title", layer_name),
            "units": c_meta.get("units", ""),
            "dates": dates,
            "bounds": bounds_lonlat,
            "crs": f"EPSG:{epsg_code}",
            "color_range": [c_meta.get("vmin", -25.0), c_meta.get("vmax", 0.0)],
            "colormap": c_meta.get("cmap", "viridis"),
            "cog_files": layer_cog_paths[layer_name],
            "zarr_stack": f"stack/{layer_name}.zarr",
            "track": track,
            "frame": frame,
            "direction": direction,
            "polarization_flags": pol_flags_by_date,
        }

    # Pairwise change between earliest and latest dates in the same group
    if (
        first_window is not None
        and last_window is not None
        and first_window.date_str != last_window.date_str
    ):
        change_res = compute_pairwise_change(first_window, last_window)
        pair_tag = f"{first_window.date_str}_{last_window.date_str}"
        for ch_name, ch_arr in change_res["layers"].items():
            cog_rel = Path("cog") / ch_name / f"{pair_tag}.tif"
            write_float32_cog(
                arr=ch_arr,
                x=change_res["x"],
                y=change_res["y"],
                epsg=change_res["epsg"],
                out_path=site_out_dir / cog_rel,
                metadata={
                    "pre_date": first_window.date_str,
                    "post_date": last_window.date_str,
                    "track": track,
                    "frame": frame,
                    "direction": direction,
                    "cross_pol_gap": change_res["cross_pol_gap"],
                },
            )
            c_meta = color_cfg.get(ch_name, {})
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
                "track": track,
                "frame": frame,
                "direction": direction,
                "polarization_flags": {
                    pair_tag: "CROSS_POL_GAP" if change_res["cross_pol_gap"] else "DUAL_POL"
                },
            }

    catalog_path = update_catalog_json(
        catalog_path=site_out_dir / "catalog.json",
        site_name=site_name,
        site_title=site_cfg.get("title", site_name),
        layer_entries=catalog_entries,
    )
    logger.info("Updated site catalog: %s", catalog_path)


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI argument parser."""
    ap = argparse.ArgumentParser(
        prog="run_site.py",
        description="NASA-ISRO NISAR data-pull and preprocessing CLI.",
    )
    ap.add_argument(
        "cmd",
        choices=["search", "inspect", "build"],
        help="Command to execute.",
    )
    ap.add_argument(
        "--site",
        choices=["california", "nile_delta", "himalaya"],
        default="california",
        help="Case-study site key from config.yaml.",
    )
    ap.add_argument(
        "--products",
        nargs="+",
        default=["GCOV"],
        choices=["GCOV", "GUNW", "GOFF", "SME2"],
        help="NISAR product levels to search/inspect/build.",
    )
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="Override site bounding box (west south east north in EPSG:4326).",
    )
    ap.add_argument("--start", help="Override ISO start date (YYYY-MM-DD).")
    ap.add_argument("--end", help="Override ISO end date (YYYY-MM-DD).")
    ap.add_argument("--track", type=int, help="Filter by pathNumber / track.")
    ap.add_argument("--frame", type=int, help="Filter by frameNumber.")
    ap.add_argument(
        "--direction",
        choices=["A", "D", "ASC", "DESC"],
        help="Filter by flightDirection.",
    )
    ap.add_argument(
        "--idx",
        type=int,
        default=0,
        help="Granule index from search table to inspect (default: 0).",
    )
    ap.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Optional cap on number of dates to build (for quick testing).",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Use resumable full-granule download fallback instead of HTTPS streaming.",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config.yaml.",
    )
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
    start = args.start or site_cfg.get("start", "2025-10-01")
    end = args.end or site_cfg.get(
        "end", datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

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
            if args.idx < 0 or args.idx >= len(records):
                raise IndexError(
                    f"--idx {args.idx} out of range for {len(records)} found granules."
                )
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
                            "GCOV AOI window %s: shape=%s, median_linear=%.5f",
                            k,
                            v.shape,
                            float(np.nanmedian(v)),
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
                    site_name=args.site,
                    cfg=cfg,
                    records=records,
                    token=token,
                    download=args.download,
                    max_dates=args.max_dates,
                )
            else:
                raise NotImplementedError(
                    f"Build for product '{product}' requires gunw.py / goff.py / export_web.py "
                    "(to be added in follow-up steps) and layer verification via `inspect`."
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())