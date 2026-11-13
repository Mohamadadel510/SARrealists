"""
Human-understandable radar visualization and diagnostics module:
1. Plain-English legend descriptors with technical units ([dB], [mm/yr]) in brackets
2. `enrich_catalog_with_human_labels`: embeds plain-English legend stops into catalog.json
3. `plot_pol_and_freq_explorer`: 6-panel figure comparing HH Co-Pol, HV Cross-Pol,
   Polarimetric RGB Composite, Frequency A, Frequency B, and Dual-Frequency Difference (A - B)
4. `plot_human_friendly_guide`: 6-panel site summary dashboard from built COGs, GeoJSONs,
   and `points.json` with plain-English colorbar labels
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from nisar_pipeline.gcov import compute_affine_and_orient
from nisar_pipeline.window import GCOVWindow

logger = logging.getLogger(__name__)

# Plain-English interpretation guide with technical values in brackets
HUMAN_LAYER_SPECS: Dict[str, Dict[str, Any]] = {
    "hh_db": {
        "friendly_title": "Surface Roughness & Moisture (HH Co-Pol)",
        "what_it_shows": "Dark = Calm Water / Flat Sand | Bright = Wet Fields, Canals & Buildings",
        "vmin": -22.0,
        "vmax": 0.0,
        "cmap": "gray",
        "stops": [
            (-22.0, "Calm Water / Flat Sand [-22 dB]"),
            (-15.0, "Dry / Smooth Soil [-15 dB]"),
            (-8.0, "Moist Fields / Rough Ground [-8 dB]"),
            (0.0, "Buildings / Canal Banks [0 dB]"),
        ],
    },
    "hv_db": {
        "friendly_title": "Cross-Pol Backscatter: Crop Volume & Rotated Structures (HV)",
        "what_it_shows": "Dark = Water / Bare Sand | Mid Green = Irrigated Crops | Darkest Green = Orchards & Rotated Village Buildings",
        "vmin": -30.0,
        "vmax": -8.0,
        "cmap": "YlGn",
        "stops": [
            (-30.0, "Water / Bare Soil [-30 dB]"),
            (-22.0, "Young / Sparse Crops [-22 dB]"),
            (-15.0, "Dense Irrigated Field Crops [-15 dB]"),
            (-8.0, "Orchards / Village Corner Bounce [-8 dB]"),
        ],
    },
    "rvi": {
        "friendly_title": "Radar Vegetation Health Index (Dual-Pol RVI)",
        "what_it_shows": "Combines HH + HV to separate bare ground/cities from true crop biomass",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "YlGn",
        "stops": [
            (0.0, "Bare Ground / Water / Buildings [0.0]"),
            (0.35, "Light Vegetation [0.35]"),
            (0.70, "Healthy Crop Cover [0.70]"),
            (1.0, "Dense Canopy Biomass [1.0]"),
        ],
    },
    "pol_ratio_db": {
        "friendly_title": "Canopy-to-Surface Scattering Ratio (HV / HH)",
        "what_it_shows": "Low = Flat Surface Bounce Dominates | High = Canopy Volume Bounce Dominates",
        "vmin": -15.0,
        "vmax": -2.0,
        "cmap": "viridis",
        "stops": [
            (-15.0, "Bare Surface Bounce [-15 dB]"),
            (-8.5, "Mixed Soil & Plants [-8.5 dB]"),
            (-2.0, "Pure Canopy Bounce [-2 dB]"),
        ],
    },
    "freq_diff_db": {
        "friendly_title": "Dual-Frequency Split-Spectrum Difference (Freq A - Freq B)",
        "what_it_shows": "Compares High-Res Frequency A vs. Coarser 5 MHz Frequency B Sub-Band",
        "vmin": -4.0,
        "vmax": 4.0,
        "cmap": "coolwarm",
        "stops": [
            (-4.0, "Freq B Stronger Bounce [-4 dB]"),
            (-2.0, "Slightly Stronger in Freq B [-2 dB]"),
            (0.0, "Equal Bounce in Both Bands [0 dB]"),
            (2.0, "Slightly Stronger in Freq A [+2 dB]"),
            (4.0, "Freq A Stronger (Fine Edges) [+4 dB]"),
        ],
    },
    "rgb_composite": {
        "friendly_title": "Polarimetric SAR RGB Composite (Green=Crops, Magenta=Cities)",
        "what_it_shows": "Green = Irrigated Crops & Trees | Magenta/Pink = Cities & Villages | Dark = Water",
        "vmin": 0.0,
        "vmax": 255.0,
        "cmap": "rgb",
        "stops": [],
    },
    "hv_db_change": {
        "friendly_title": "Wildfire Burn & Canopy Change (Post vs. Pre HV)",
        "what_it_shows": "Red = Burned Trees / Harvested Crops | Blue = New Growth / Structure",
        "vmin": -6.0,
        "vmax": 6.0,
        "cmap": "RdBu",
        "stops": [
            (-6.0, "Severe Burn / Canopy Loss [-6 dB]"),
            (-3.0, "Moderate Vegetation Drop [-3 dB]"),
            (0.0, "No Change [0 dB]"),
            (3.0, "Vegetation Growth [+3 dB]"),
            (6.0, "Strong Regrowth / Wetting [+6 dB]"),
        ],
    },
    "hh_db_change": {
        "friendly_title": "Surface Moisture & Flood Change (Post vs. Pre HH)",
        "what_it_shows": "Red = Drying or Smooth Floodwater | Blue = Soil Wetting / Plowing",
        "vmin": -6.0,
        "vmax": 6.0,
        "cmap": "RdBu",
        "stops": [
            (-6.0, "Standing Floodwater / Drying [-6 dB]"),
            (0.0, "Unchanged Surface [0 dB]"),
            (6.0, "Increased Moisture / Roughness [+6 dB]"),
        ],
    },
    "rvi_change": {
        "friendly_title": "Biomass & Crop Cover Change (Post vs. Pre RVI)",
        "what_it_shows": "Red = Biomass Loss (Fire / Harvest) | Blue = Crop Growth",
        "vmin": -0.4,
        "vmax": 0.4,
        "cmap": "RdBu",
        "stops": [
            (-0.4, "Canopy Destroyed / Harvested [-0.4]"),
            (0.0, "Stable Vegetation [0.0]"),
            (0.4, "Rapid Crop Growth [+0.4]"),
        ],
    },
    "soil_moisture": {
        "friendly_title": "Root-Zone Soil Moisture (L3 SME2)",
        "what_it_shows": "Fraction of soil volume filled with water in top 5 cm",
        "vmin": 0.02,
        "vmax": 0.50,
        "cmap": "YlGnBu",
        "stops": [
            (0.02, "Dry Desert / Fallow (2% [0.02 m³/m³])"),
            (0.18, "Damp Soil (18% [0.18 m³/m³])"),
            (0.34, "Moist Irrigated Field (34% [0.34 m³/m³])"),
            (0.50, "Saturated / Flooded (50% [0.50 m³/m³])"),
        ],
    },
    "los_disp_mm": {
        "friendly_title": "Ground Sinking / Uplift Along Radar Beam",
        "what_it_shows": "Blue = Ground Sinking (Subsidence) | Red = Ground Rising (Uplift)",
        "vmin": -100.0,
        "vmax": 100.0,
        "cmap": "RdBu_r",
        "stops": [
            (-100.0, "Strong Sinking [-100 mm]"),
            (-50.0, "Moderate Sinking [-50 mm]"),
            (0.0, "Stable Ground [0 mm]"),
            (50.0, "Moderate Uplift [+50 mm]"),
            (100.0, "Strong Uplift [+100 mm]"),
        ],
    },
    "vert_disp_mm": {
        "friendly_title": "Vertical Land Subsidence",
        "what_it_shows": "Total vertical ground movement relative to stable reference area",
        "vmin": -120.0,
        "vmax": 120.0,
        "cmap": "RdBu_r",
        "stops": [
            (-120.0, "Severe Subsidence [-120 mm]"),
            (-60.0, "Moderate Sinking [-60 mm]"),
            (0.0, "Stable Ground [0 mm]"),
            (60.0, "Slight Uplift [+60 mm]"),
            (120.0, "Strong Uplift [+120 mm]"),
        ],
    },
    "disp_rate_mm_yr": {
        "friendly_title": "Annual Land Sinking Speed",
        "what_it_shows": "Long-term ground sinking or rising trend per year",
        "vmin": -50.0,
        "vmax": 50.0,
        "cmap": "RdBu_r",
        "stops": [
            (-50.0, "Fast Sinking [-50 mm/yr]"),
            (-25.0, "Moderate Sinking [-25 mm/yr]"),
            (0.0, "Stable Ground [0 mm/yr]"),
            (25.0, "Moderate Rising [+25 mm/yr]"),
            (50.0, "Fast Rising [+50 mm/yr]"),
        ],
    },
    "velocity_m_yr": {
        "friendly_title": "Glacier Ice Flow Speed",
        "what_it_shows": "How fast the glacier ice surface moves downhill per year",
        "vmin": 0.0,
        "vmax": 150.0,
        "cmap": "plasma",
        "stops": [
            (0.0, "Frozen Bedrock / Stationary [0 m/yr]"),
            (50.0, "Slow Ice Creep [50 m/yr]"),
            (100.0, "Active Glacier Flow [100 m/yr]"),
            (150.0, "Fast Glacier Tongue [150 m/yr]"),
        ],
    },
}


def enrich_catalog_with_human_labels(catalog_path: Path) -> Path:
    """
    Add `friendly_title`, `what_it_shows`, and plain-English `legend_stops`
    (with technical units in brackets) to every layer in `catalog.json`.
    """
    if not catalog_path.exists():
        return catalog_path
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    layers = catalog.get("layers", {})
    for lid, meta in layers.items():
        spec = HUMAN_LAYER_SPECS.get(lid)
        if spec:
            meta["friendly_title"] = spec["friendly_title"]
            meta["what_it_shows"] = spec["what_it_shows"]
            meta["legend_stops"] = [
                {"value": float(val), "label": str(lbl)} for val, lbl in spec["stops"]
            ]
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog_path


def _rgb_chw_to_rgba_hwc(rgb_chw: np.ndarray) -> np.ndarray:
    """Convert `(3, H, W)` uint8 array (`0` = nodata) into `(H, W, 4)` RGBA uint8 with transparent nodata."""
    rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
    valid = (rgb_hwc[:, :, 0] > 0) | (rgb_hwc[:, :, 1] > 0) | (rgb_hwc[:, :, 2] > 0)
    alpha = np.where(valid, 255, 0).astype(np.uint8)[..., np.newaxis]
    return np.concatenate([rgb_hwc, alpha], axis=-1)


def plot_pol_and_freq_explorer(
    win: GCOVWindow,
    prod: Dict[str, Any],
    site_out_dir: Path,
    site_title: str,
) -> Path:
    """
    Save a 6-panel figure (`out/<site>/plots/pol_freq_explorer_<YYYYMMDD>.png`) showing:
      Row 1 (Polarization Comparison):
        1. HH Co-Pol (Surface Roughness, Moisture & Buildings)
        2. HV Cross-Pol (Crop Volume & Rotated Structures)
        3. Combined Polarimetric RGB Composite (Transparent outside swath)
      Row 2 (Dual-Frequency Split-Spectrum Comparison):
        4. Frequency A High-Res Science Band (MHz & pixel spacing)
        5. Frequency B Simultaneous 5 MHz Sub-Band (MHz & pixel spacing)
        6. Dual-Frequency Difference (Freq A - Freq B)
    """
    plots_dir = site_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = plots_dir / f"pol_freq_explorer_{win.date_str or 'latest'}.png"

    hh_a_db, x_asc, y_desc, _ = compute_affine_and_orient(prod["layers"]["hh_db"], win.x, win.y)
    hv_a_db, _, _, _ = compute_affine_and_orient(prod["layers"]["hv_db"], win.x, win.y)
    freq_diff_db, _, _, _ = compute_affine_and_orient(prod["layers"]["freq_diff_db"], win.x, win.y)
    rgb_oriented, _, _, _ = compute_affine_and_orient(prod["rgb"], win.x, win.y)
    rgba_hwc = _rgb_chw_to_rgba_hwc(rgb_oriented)

    hh_b_raw = prod.get("hh_b_db")
    hh_b_db = (
        compute_affine_and_orient(hh_b_raw, win.x, win.y)[0]
        if hh_b_raw is not None
        else None
    )

    ext_km = [
        float(x_asc.min()) / 1e3,
        float(x_asc.max()) / 1e3,
        float(y_desc.min()) / 1e3,
        float(y_desc.max()) / 1e3,
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Co-Pol (HH/VV)
    im0 = axes[0, 0].imshow(hh_a_db, extent=ext_km, origin="upper", cmap="gray", vmin=-22.0, vmax=0.0)
    c0 = fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    c0.set_ticks([-22.0, -15.0, -8.0, 0.0])
    c0.ax.set_yticklabels(
        [
            "Calm Water / Sand [-22 dB]",
            "Smooth Soil [-15 dB]",
            "Moist Fields [-8 dB]",
            "Buildings / Banks [0 dB]",
        ],
        fontsize=8,
    )
    axes[0, 0].set_title(
        f"1. {prod['co_pol_name']} Co-Pol: Surface & Moisture Bounce\n"
        "(Wet soil, canals & buildings reflect strongly)",
        fontweight="bold",
        fontsize=10,
    )

    # Panel 2: Cross-Pol (HV/VH)
    if prod["has_dual_pol"]:
        im1 = axes[0, 1].imshow(hv_a_db, extent=ext_km, origin="upper", cmap="YlGn", vmin=-30.0, vmax=-8.0)
        c1 = fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
        c1.set_ticks([-30.0, -22.0, -15.0, -8.0])
        c1.ax.set_yticklabels(
            [
                "Water / Bare [-30 dB]",
                "Sparse Crops [-22 dB]",
                "Irrigated Crops [-15 dB]",
                "Orchards / Villages [-8 dB]",
            ],
            fontsize=8,
        )
        axes[0, 1].set_title(
            f"2. {prod['cross_pol_name']} Cross-Pol: Crop & Structure Volume Bounce\n"
            "(Crop leaves, orchards & rotated village blocks twist the wave)",
            fontweight="bold",
            fontsize=10,
        )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            f"Single-Pol Cycle ({prod['polarization_flag']})\nCross-Pol Not Acquired",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("2. Cross-Pol: Not Present in Cycle", fontweight="bold")

    # Panel 3: Polarimetric RGB Composite (with transparent alpha outside swath)
    axes[0, 2].imshow(rgba_hwc, extent=ext_km, origin="upper")
    axes[0, 2].set_title(
        "3. Polarimetric SAR RGB Composite (Urban-Filtered)\n"
        "Green = Irrigated Crops | Magenta = Cities/Villages | Dark = Water",
        fontweight="bold",
        fontsize=10,
    )

    # Panel 4: Frequency A
    im3 = axes[1, 0].imshow(hh_a_db, extent=ext_km, origin="upper", cmap="viridis", vmin=-22.0, vmax=0.0)
    c3 = fig.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    c3.set_ticks([-22.0, -11.0, 0.0])
    c3.ax.set_yticklabels(
        ["Smooth [-22 dB]", "Medium [-11 dB]", "Rough [0 dB]"], fontsize=8
    )
    fa_lbl = f"{win.freq_a_mhz} MHz" if win.freq_a_mhz else "Primary Band"
    axes[1, 0].set_title(
        f"4. Frequency A ({fa_lbl}, {win.pixel_spacing_a_m}m grid)\n"
        "High-Resolution Primary L-Band Science Channel",
        fontweight="bold",
        fontsize=10,
    )

    # Panel 5 & 6: Frequency B and Dual-Frequency Difference (Freq A - Freq B)
    if hh_b_db is not None and np.any(np.isfinite(hh_b_db)):
        im4 = axes[1, 1].imshow(hh_b_db, extent=ext_km, origin="upper", cmap="viridis", vmin=-22.0, vmax=0.0)
        c4 = fig.colorbar(im4, ax=axes[1, 1], shrink=0.8)
        c4.set_ticks([-22.0, -11.0, 0.0])
        c4.ax.set_yticklabels(
            ["Smooth [-22 dB]", "Medium [-11 dB]", "Rough [0 dB]"], fontsize=8
        )
        fb_lbl = f"{win.freq_b_mhz} MHz" if win.freq_b_mhz else "Secondary Band"
        axes[1, 1].set_title(
            f"5. Frequency B ({fb_lbl}, {win.pixel_spacing_b_m}m grid)\n"
            "Simultaneous 5 MHz Secondary L-Band Sub-Band",
            fontweight="bold",
            fontsize=10,
        )

        im5 = axes[1, 2].imshow(freq_diff_db, extent=ext_km, origin="upper", cmap="coolwarm", vmin=-4.0, vmax=4.0)
        c5 = fig.colorbar(im5, ax=axes[1, 2], shrink=0.8)
        c5.set_ticks([-4.0, -2.0, 0.0, 2.0, 4.0])
        c5.ax.set_yticklabels(
            [
                "Freq B Stronger [-4 dB]",
                "Freq B Slightly Higher [-2 dB]",
                "Equal Bounce [0 dB]",
                "Freq A Slightly Higher [+2 dB]",
                "Freq A Stronger [+4 dB]",
            ],
            fontsize=8,
        )
        axes[1, 2].set_title(
            "6. Dual-Frequency Difference (Freq A - Freq B)\n"
            "Highlights fine edges & small-scale Bragg roughness",
            fontweight="bold",
            fontsize=10,
        )
    else:
        for ax_idx, title_txt in [
            (1, "5. Frequency B (Single-Frequency Mode)"),
            (2, "6. Dual-Frequency Difference (Requires Freq A + B)"),
        ]:
            axes[1, ax_idx].text(
                0.5,
                0.5,
                "Granule operated in single-frequency mode\n(Frequency B not active)",
                ha="center",
                va="center",
                transform=axes[1, ax_idx].transAxes,
            )
            axes[1, ax_idx].set_title(title_txt, fontweight="bold", fontsize=10)

    for ax in axes.ravel():
        ax.set_xlabel(f"West <-> East Distance (km, EPSG:{win.epsg})")
        ax.set_ylabel("South <-> North Distance (km)")

    bands_str = "+".join(win.active_bands)
    fig.suptitle(
        f"NISAR Multi-Polarization & Dual-Frequency Explorer ({bands_str}) — {site_title} ({win.date_str})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Saved Polarization & Dual-Frequency Explorer plot: %s", out_png)
    return out_png


def plot_human_friendly_guide(site_out_dir: Path, show_inline: bool = False) -> Optional[Path]:
    """
    Generate a 6-panel human-readable site overview dashboard from built artifacts in
    `out/<site>/`, saving to `out/<site>/plots/human_friendly_guide.png` and
    `out/<site>/qa_preview.png`.
    """
    cat_path = site_out_dir / "catalog.json"
    if not cat_path.exists():
        logger.warning("No catalog.json found in %s; skipping plot.", site_out_dir)
        return None

    enrich_catalog_with_human_labels(cat_path)
    catalog = json.loads(cat_path.read_text(encoding="utf-8"))
    layers = catalog.get("layers", {})
    if not layers:
        return None

    plots_dir = site_out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    priority = [
        "rgb_composite",
        "hh_db",
        "hv_db",
        "rvi",
        "freq_diff_db",
        "hv_db_change",
        "soil_moisture",
        "velocity_m_yr",
        "disp_rate_mm_yr",
        "los_disp_mm",
    ]
    ordered = [k for k in priority if k in layers] + [
        k for k in layers if k not in priority
    ]
    selected = ordered[:4]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    ax_list = axes.ravel()

    for i in range(4):
        ax = ax_list[i]
        if i >= len(selected):
            ax.axis("off")
            continue

        lid = selected[i]
        meta = layers[lid]
        spec = HUMAN_LAYER_SPECS.get(lid, {})
        cog_files = meta.get("cog_files", [])
        if not cog_files:
            ax.axis("off")
            continue

        cog_path = site_out_dir / cog_files[-1]
        with rasterio.open(cog_path) as src:
            b = src.bounds
            ext_km = [b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3]
            if src.count == 3:
                rgba_arr = _rgb_chw_to_rgba_hwc(src.read([1, 2, 3]))
                ax.imshow(rgba_arr, extent=ext_km, origin="upper")
            else:
                arr = src.read(1)
                vmin = spec.get("vmin", meta.get("color_range", [-25.0, 0.0])[0])
                vmax = spec.get("vmax", meta.get("color_range", [-25.0, 0.0])[1])
                cmap = spec.get("cmap", meta.get("colormap", "viridis"))
                im = ax.imshow(arr, extent=ext_km, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
                if spec.get("stops"):
                    cbar.set_ticks([s[0] for s in spec["stops"]])
                    cbar.ax.set_yticklabels([s[1] for s in spec["stops"]], fontsize=8)

        date_lbl = meta.get("dates", [""])[-1]
        ax.set_title(
            f"{spec.get('friendly_title', meta.get('title', lid))} ({date_lbl})\n"
            f"{spec.get('what_it_shows', '')}",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_xlabel("West <-> East (km)")
        ax.set_ylabel("South <-> North (km)")

    # Panel 4: Vectors (Glacier Velocity Arrows or Subsidence Rate) or Dual-Freq / Change
    ax_vec = ax_list[4]
    vel_geojson = site_out_dir / "vectors" / "glacier_velocity.geojson"
    sub_geojson = site_out_dir / "vectors" / "subsidence_rates.geojson"

    if vel_geojson.exists():
        fc = json.loads(vel_geojson.read_text(encoding="utf-8"))
        feats = fc.get("features", [])
        if feats:
            lons = np.array([f["geometry"]["coordinates"][0] for f in feats])
            lats = np.array([f["geometry"]["coordinates"][1] for f in feats])
            ve = np.array([f["properties"]["ve_m_yr"] for f in feats])
            vn = np.array([f["properties"]["vn_m_yr"] for f in feats])
            spd = np.array([f["properties"]["speed"] for f in feats])
            q = ax_vec.quiver(lons, lats, ve, vn, spd, cmap="plasma", angles="xy")
            cb = fig.colorbar(q, ax=ax_vec, shrink=0.80)
            cb.set_ticks([0, 50, 100, 150])
            cb.ax.set_yticklabels(
                ["Stationary [0 m/yr]", "Slow [50 m/yr]", "Active [100 m/yr]", "Fast [150 m/yr]"],
                fontsize=8,
            )
            ax_vec.set_title(
                f"Glacier Ice Flow Arrows ({len(feats)} vectors)\n"
                "Direction & Speed of Moving Ice",
                fontweight="bold",
                fontsize=9.5,
            )
            ax_vec.set_xlabel("Longitude (°E)")
            ax_vec.set_ylabel("Latitude (°N)")
        else:
            ax_vec.axis("off")
    elif sub_geojson.exists():
        fc = json.loads(sub_geojson.read_text(encoding="utf-8"))
        feats = fc.get("features", [])
        if feats:
            lons = np.array([f["geometry"]["coordinates"][0] for f in feats])
            lats = np.array([f["geometry"]["coordinates"][1] for f in feats])
            rates = np.array(
                [
                    f["properties"]["rate_mm_yr"]
                    if f["properties"]["rate_mm_yr"] is not None
                    else np.nan
                    for f in feats
                ]
            )
            sc = ax_vec.scatter(lons, lats, c=rates, cmap="RdBu_r", vmin=-50, vmax=50, s=12)
            cb = fig.colorbar(sc, ax=ax_vec, shrink=0.80)
            cb.set_ticks([-50, -25, 0, 25, 50])
            cb.ax.set_yticklabels(
                [
                    "Fast Sinking [-50 mm/yr]",
                    "Sinking [-25 mm/yr]",
                    "Stable [0 mm/yr]",
                    "Rising [+25 mm/yr]",
                    "Fast Rising [+50 mm/yr]",
                ],
                fontsize=8,
            )
            ax_vec.set_title(
                f"Ground Sinking / Subsidence Map ({len(feats)} pts)\n"
                "Blue = Sinking Land | Red = Rising Land",
                fontweight="bold",
                fontsize=9.5,
            )
            ax_vec.set_xlabel("Longitude (°E)")
            ax_vec.set_ylabel("Latitude (°N)")
        else:
            ax_vec.axis("off")
    elif len(ordered) > 4:
        extra_lid = ordered[4]
        meta = layers[extra_lid]
        spec = HUMAN_LAYER_SPECS.get(extra_lid, {})
        cog_path = site_out_dir / meta["cog_files"][-1]
        with rasterio.open(cog_path) as src:
            arr = src.read(1)
            b = src.bounds
            ext_km = [b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3]
            vmin = spec.get("vmin", meta.get("color_range", [-6.0, 6.0])[0])
            vmax = spec.get("vmax", meta.get("color_range", [-6.0, 6.0])[1])
            cmap = spec.get("cmap", meta.get("colormap", "RdBu"))
            im = ax_vec.imshow(arr, extent=ext_km, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax_vec, shrink=0.80, pad=0.02)
            if spec.get("stops"):
                cbar.set_ticks([s[0] for s in spec["stops"]])
                cbar.ax.set_yticklabels([s[1] for s in spec["stops"]], fontsize=8)
            ax_vec.set_title(
                f"{spec.get('friendly_title', extra_lid)}\n{spec.get('what_it_shows', '')}",
                fontsize=9.5,
                fontweight="bold",
            )
            ax_vec.set_xlabel("West <-> East (km)")
            ax_vec.set_ylabel("South <-> North (km)")
    else:
        ax_vec.axis("off")

    # Panel 5: Point Time Series (points.json)
    ax_ts = ax_list[5]
    pts_path = site_out_dir / "timeseries" / "points.json"
    metric_labels = {
        "hv_db": "Canopy Bounce HV [dB]",
        "hh_db": "Surface Bounce HH [dB]",
        "disp_mm": "Ground Movement [mm]",
        "sm": "Soil Moisture [m³/m³]",
    }
    if pts_path.exists():
        pts_data = json.loads(pts_path.read_text(encoding="utf-8"))
        plotted = False
        for pt in pts_data:
            pid = pt["id"]
            dates = pt.get("dates", [])
            for metric, style in [
                ("hv_db", "-o"),
                ("hh_db", "--s"),
                ("disp_mm", "-^"),
                ("sm", "-d"),
            ]:
                vals = pt.get(metric, [])
                if vals and any(v is not None for v in vals):
                    y_vals = [float(v) if v is not None else np.nan for v in vals]
                    ax_ts.plot(
                        dates,
                        y_vals,
                        style,
                        label=f"{pid}: {metric_labels.get(metric, metric)}",
                    )
                    plotted = True
        if plotted:
            ax_ts.set_title(
                "Sample Locations Over Time\n(Tracked across satellite passes)",
                fontweight="bold",
                fontsize=9.5,
            )
            ax_ts.tick_params(axis="x", rotation=30)
            ax_ts.grid(True, alpha=0.3)
            ax_ts.legend(fontsize=7.5, loc="best")
        else:
            ax_ts.set_title("Sample Point Time Series (No overlapping points)")
    else:
        ax_ts.axis("off")

    fig.suptitle(
        f"User-Friendly Radar Summary — {catalog.get('title', site_out_dir.name.upper())}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    guide_png = plots_dir / "human_friendly_guide.png"
    qa_png = site_out_dir / "qa_preview.png"
    fig.savefig(guide_png, dpi=150)
    fig.savefig(qa_png, dpi=150)
    if show_inline:
        plt.show()
    plt.close(fig)
    logger.info("Saved human-friendly guide plot: %s", guide_png)
    return guide_png