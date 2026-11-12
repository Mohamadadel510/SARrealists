"""
Synthetic unit test suite for NISAR pipeline (runs offline with pytest, no credentials).

Verifies:
1. Multilook in linear power BEFORE converting to dB (and inequality with dB averaging)
2. Dual-pol RVI bounds (0..1), exact formula 4*HV/(HH+HV), and HH-only missing-pol fallback
3. Pairwise change sign (burn scar cross-pol drop -> negative dB and negative RVI change)
   and strict same-(track, frame, direction) enforcement
4. Chained cumulative displacement sum across 12-day pairs
5. Stable reference box subtraction in UTM coordinates
6. AOI-to-UTM window slicing, layover/shadow mask filtering, de-duplication, and retry wrapper
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure the repo root (/content/SARrealists) is on sys.path for bare `pytest` invocations
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import pytest
from pyproj import Transformer

from nisar_pipeline.gcov import (
    compute_pairwise_change,
    compute_rvi,
    compute_single_date_products,
    linear_to_db,
)
from nisar_pipeline.io_nisar import (
    normalize_and_dedupe_granules,
    run_with_h5_retry,
)
from nisar_pipeline.layers import GCOV_GRID_GROUP, GUNW_UNWRAPPED_PHASE, require_verified_layer
from nisar_pipeline.window import (
    GCOVWindow,
    block_mean,
    chain_pairwise_displacements,
    load_gcov_window,
    subtract_reference_window,
)


def _create_synthetic_gcov_h5(
    epsg: int = 32611,
    xmin: float = 348000.0,
    ymax: float = 3776000.0,
    dx: float = 20.0,
    dy: float = 20.0,
    ny: int = 60,
    nx: int = 60,
    include_hv: bool = True,
) -> h5py.File:
    """Create an in-memory HDF5 file matching the verified NISAR L2 GCOV schema."""
    bio = io.BytesIO()
    h5 = h5py.File(bio, "w")
    g = h5.create_group(GCOV_GRID_GROUP)

    x = xmin + np.arange(nx, dtype=np.float64) * dx
    y = ymax - np.arange(ny, dtype=np.float64) * dy
    g.create_dataset("xCoordinates", data=x)
    g.create_dataset("yCoordinates", data=y)

    proj = g.create_dataset("projection", data=np.int32(epsg))
    proj.attrs["epsg_code"] = np.int32(epsg)

    hh = np.full((ny, nx), 0.25, dtype=np.float32)
    # Inject nonpositive values at (0, 0) and (0, 1) to verify masking
    hh[0, 0] = -1.0
    hh[0, 1] = 0.0
    g.create_dataset("HHHH", data=hh)

    if include_hv:
        hv = np.full((ny, nx), 0.05, dtype=np.float32)
        g.create_dataset("HVHV", data=hv)

    # Mask: 1 = valid, 0 = outside swath, 255 = layover/shadow
    mask = np.ones((ny, nx), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    mask[4:6, 4:6] = 0
    g.create_dataset("mask", data=mask)

    return h5


def test_multilook_linear_before_db_order() -> None:
    """Verify multilooking in linear power before 10*log10 differs from averaging in dB."""
    raw_linear = np.array(
        [[0.01, 0.09], [0.25, 0.81]], dtype=np.float32
    )
    ml_linear = block_mean(raw_linear, looks=2)
    db_from_ml_linear = linear_to_db(ml_linear)

    # Incorrect method: convert to dB first, then average
    wrong_avg_db = block_mean(linear_to_db(raw_linear), looks=2)

    expected_mean_lin = np.mean(raw_linear)
    expected_db = 10.0 * np.log10(expected_mean_lin)

    assert db_from_ml_linear.dtype == np.float32
    assert np.isclose(db_from_ml_linear[0, 0], expected_db, atol=1e-5)
    # By Jensen's inequality on concave log10, mean(dB) < dB(mean(linear))
    assert wrong_avg_db[0, 0] < db_from_ml_linear[0, 0]


def test_rvi_bounds_and_hh_only_fallback() -> None:
    """Verify RVI = 4*HV/(HH+HV) values, bounds [0, 1], and HH-only cycle NaN fallback."""
    hh = np.array([[0.30, 0.40], [0.20, 0.0]], dtype=np.float32)
    hv = np.array([[0.0001, 0.10], [0.0666667, 0.05]], dtype=np.float32)

    rvi = compute_rvi(hh, hv)
    assert rvi.dtype == np.float32
    # At (0, 1): 4 * 0.10 / (0.40 + 0.10) = 0.80
    assert np.isclose(rvi[0, 1], 0.80, atol=1e-5)
    # At (1, 0): 4 * (1/15) / (3/15 + 1/15) = 1.0 (clipped to 1.0)
    assert np.isclose(rvi[1, 0], 1.0, atol=1e-4)
    # At (1, 1): HH == 0 -> masked to NaN
    assert np.isnan(rvi[1, 1])

    valid_vals = rvi[np.isfinite(rvi)]
    assert np.all((valid_vals >= 0.0) & (valid_vals <= 1.0))

    # HH-only fallback when HV is None
    rvi_missing = compute_rvi(hh, None)
    assert np.all(np.isnan(rvi_missing))


def test_pairwise_change_sign_and_track_guard() -> None:
    """
    Verify vegetation burn loss (drop in HV linear power) yields negative hv_db_change
    and negative rvi_change, and cross-track comparisons raise ValueError.
    """
    x = np.array([350000.0, 350040.0], dtype=np.float64)
    y = np.array([3775000.0, 3774960.0], dtype=np.float64)

    pre_win = GCOVWindow(
        pols={
            "HHHH": np.full((2, 2), 0.20, dtype=np.float32),
            "HVHV": np.full((2, 2), 0.05, dtype=np.float32),
        },
        x=x,
        y=y,
        epsg=32611,
        track=13,
        frame=70,
        direction="D",
        date_str="20260101",
    )
    # Post-fire: HV drops by half (-3.01 dB), HH stays constant
    post_win = GCOVWindow(
        pols={
            "HHHH": np.full((2, 2), 0.20, dtype=np.float32),
            "HVHV": np.full((2, 2), 0.025, dtype=np.float32),
        },
        x=x,
        y=y,
        epsg=32611,
        track=13,
        frame=70,
        direction="D",
        date_str="20260113",
    )

    change = compute_pairwise_change(pre_win, post_win)
    hv_diff = change["layers"]["hv_db_change"]
    rvi_diff = change["layers"]["rvi_change"]

    assert np.all(hv_diff < 0.0)
    assert np.allclose(hv_diff, 10.0 * np.log10(0.5), atol=1e-4)
    assert np.all(rvi_diff < 0.0)
    assert change["cross_pol_gap"] is False

    # Cross-track comparison must fail
    wrong_track_win = GCOVWindow(
        pols=post_win.pols,
        x=x,
        y=y,
        epsg=32611,
        track=114,
        frame=70,
        direction="D",
        date_str="20260113",
    )
    with pytest.raises(ValueError, match="different track/frame/direction"):
        compute_pairwise_change(pre_win, wrong_track_win)


def test_chained_cumulative_sum() -> None:
    """Verify nearest-neighbor 12-day pair displacements chain into cumulative series."""
    d1 = np.array([[2.0, -3.0], [np.nan, 1.5]], dtype=np.float32)
    d2 = np.array([[1.0, -2.0], [np.nan, 2.5]], dtype=np.float32)
    d3 = np.array([[-0.5, -1.0], [np.nan, 0.0]], dtype=np.float32)

    cum = chain_pairwise_displacements([d1, d2, d3], propagate_nan=True)
    assert cum.shape == (4, 2, 2)
    assert cum.dtype == np.float32

    # Pixel (0, 0): 0 -> 2.0 -> 3.0 -> 2.5
    assert np.allclose(cum[:, 0, 0], [0.0, 2.0, 3.0, 2.5])
    # Pixel (0, 1): 0 -> -3.0 -> -5.0 -> -6.0
    assert np.allclose(cum[:, 0, 1], [0.0, -3.0, -5.0, -6.0])
    # Pixel (1, 0): all NaN
    assert np.all(np.isnan(cum[:, 1, 0]))


def test_reference_subtraction() -> None:
    """Verify displacement array is referenced to the median of a stable lon/lat box."""
    epsg = 32636
    x = 300000.0 + np.arange(10, dtype=np.float64) * 100.0
    y = 3450000.0 - np.arange(10, dtype=np.float64) * 100.0

    # Convert pixel block [2..5, 2..5] back to lon/lat to define reference_box
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon_min, lat_min = tr.transform(x[2], y[5])
    lon_max, lat_max = tr.transform(x[5], y[2])
    ref_bbox = [lon_min, lat_min, lon_max, lat_max]

    raw_disp = np.full((10, 10), 12.5, dtype=np.float32)
    raw_disp[8, 8] = 2.5  # -10.0 relative to reference

    referenced, ref_val = subtract_reference_window(raw_disp, x, y, ref_bbox, epsg)
    assert np.isclose(ref_val, 12.5)
    assert np.isclose(referenced[3, 3], 0.0)
    assert np.isclose(referenced[8, 8], -10.0)


def test_window_selection_mask_and_retry() -> None:
    """Verify AOI window slicing, layover/shadow mask removal, and retry reopen logic."""
    h5 = _create_synthetic_gcov_h5(epsg=32611, ny=60, nx=60, include_hv=False)
    tr = Transformer.from_crs(32611, 4326, always_xy=True)
    # Pad by 5 m outside pixel centers (348000.0, 3776000.0) to avoid round-trip boundary drift
    w, s = tr.transform(348000.0 - 5.0, 3776000.0 - 20 * 20.0 - 5.0)
    e, n = tr.transform(348000.0 + 20 * 20.0 + 5.0, 3776000.0 + 5.0)

    attempts = {"count": 0}

    def _flaky_open() -> h5py.File:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise OSError("Simulated transient socket read timeout")
        return _create_synthetic_gcov_h5(epsg=32611, ny=60, nx=60, include_hv=False)

    win: GCOVWindow = run_with_h5_retry(
        url_or_path="https://example.earthdata.nasa.gov/synthetic.h5",
        reader_fn=lambda f: load_gcov_window(f, [w, s, e, n], looks=2),
        max_tries=3,
        sleep_sec=0.01,
        open_fn=_flaky_open,
    )
    h5.close()

    assert attempts["count"] == 2
    assert win.epsg == 32611
    assert win.has_dual_pol is False
    assert "HVHV" in win.missing_pols
    # Layover/shadow block [2:4, 2:4] (mask=255) -> multilooked pixel [1, 1] must be NaN
    assert np.isnan(win.pols["HHHH"][1, 1])
    # Valid block [4:6, 4:6] (mask=0) -> multilooked pixel [2, 2] must also be NaN
    assert np.isnan(win.pols["HHHH"][2, 2])
    # Single-date product falls back cleanly to HHHH-only
    prod = compute_single_date_products(win)
    assert prod["polarization_flag"] == "HHHH_ONLY_GAP"
    assert np.all(np.isnan(prod["layers"]["hv_db"]))
    assert np.any(np.isfinite(prod["layers"]["hh_db"]))


def test_deduplication_and_unverified_guard() -> None:
    """Verify deterministic CRID deduplication and UNVERIFIED layer guard."""
    raw = [
        {
            "time": "2026-02-01T13:45:10",
            "track": 13,
            "frame": 70,
            "direction": "D",
            "file_name": "NISAR_L2_PR_GCOV_004_013_D_070_20260201T134510_P03100_001.h5",
            "url": "https://datapool.asf.alaska.edu/NISAR_P03100_001.h5",
        },
        {
            "time": "2026-02-01T13:45:10",
            "track": 13,
            "frame": 70,
            "direction": "D",
            "file_name": "NISAR_L2_PR_GCOV_004_013_D_070_20260201T134510_R04000_002.h5",
            "url": "https://datapool.asf.alaska.edu/NISAR_R04000_002.h5",
        },
    ]
    deduped = normalize_and_dedupe_granules(raw)
    assert len(deduped) == 1
    assert deduped[0].version_tag == "R04000_002"

    with pytest.raises(RuntimeError, match="is UNVERIFIED"):
        require_verified_layer(GUNW_UNWRAPPED_PHASE, product="GUNW")
def test_gunw_los_vertical_and_robust_rate() -> None:
    """Verify GUNW LOS formula (-phase*lambda/(4*pi)), flat-site vertical check, and rate fit."""
    from nisar_pipeline.gunw import (
        fit_robust_linear_rate_mm_yr,
        los_to_vertical_displacement_mm,
        phase_to_los_displacement_mm,
    )

    # 1 radian of positive unwrapped phase -> -0.24 / (4 * pi) * 1000 = -19.098593 mm
    phase = np.array([[0.0, float(np.pi)], [-float(np.pi), np.nan]], dtype=np.float32)
    los_mm = phase_to_los_displacement_mm(phase, wavelength_m=0.24)
    assert np.isclose(los_mm[0, 0], 0.0)
    assert np.isclose(los_mm[0, 1], -60.0, atol=1e-4)  # -pi * 0.24 / (4*pi) * 1000 = -60 mm
    assert np.isclose(los_mm[1, 0], 60.0, atol=1e-4)
    assert np.isnan(los_mm[1, 1])

    # Vertical displacement must fail when flat_terrain=False
    with pytest.raises(ValueError, match="only allowed for flat terrain"):
        los_to_vertical_displacement_mm(los_mm, incidence_angle_deg=60.0, flat_terrain=False)

    # At 60 deg incidence angle (cos=0.5), vertical = LOS / 0.5 = 2 * LOS
    vert_mm = los_to_vertical_displacement_mm(los_mm, incidence_angle_deg=60.0, flat_terrain=True)
    assert np.isclose(vert_mm[0, 1], -120.0, atol=1e-3)

    # Synthetic cumulative displacement stack over 4 dates spaced by ~0.25 yr with -20 mm/yr rate
    dates = ["20260101", "20260402", "20260702", "20261001"]
    t_yrs = np.array([0.0, 91.0 / 365.25, 182.0 / 365.25, 273.0 / 365.25], dtype=np.float32)
    cum = np.zeros((4, 2, 2), dtype=np.float32)
    for i, ty in enumerate(t_yrs):
        cum[i, 0, 0] = -20.0 * ty
        cum[i, 0, 1] = 15.0 * ty
    cum[:2, 1, 0] = np.nan  # Only 2 valid dates (< min_dates=3) -> must be NaN

    rate = fit_robust_linear_rate_mm_yr(dates, cum, min_dates=3)
    assert np.isclose(rate[0, 0], -20.0, atol=0.2)
    assert np.isclose(rate[0, 1], 15.0, atol=0.2)
    assert np.isnan(rate[1, 0])


def test_goff_velocity_and_3857_reproject(tmp_path: Path) -> None:
    """Verify GOFF velocity vector calculation and linear-before-dB EPSG:3857 reprojection."""
    from nisar_pipeline.export_web import (
        export_velocity_vectors_geojson,
        reproject_array_to_3857,
    )
    from nisar_pipeline.goff import compute_velocity_vectors

    # Zero slant-range offset, +12 m along-track offset over 36.525 days (0.1 yr) -> 120 m/yr along heading
    sr_off = np.zeros((4, 4), dtype=np.float32)
    az_off = np.full((4, 4), 12.0, dtype=np.float32)
    vel = compute_velocity_vectors(
        sr_off, az_off, dt_days=36.525, incidence_angle_deg=38.0, heading_deg=0.0
    )
    assert np.allclose(vel["speed_m_yr"], 120.0, atol=1e-3)
    assert np.allclose(vel["direction_deg"], 0.0, atol=1e-3)

    x = 300000.0 + np.arange(4, dtype=np.float64) * 80.0
    y = 3420000.0 - np.arange(4, dtype=np.float64) * 80.0
    geojson_out = export_velocity_vectors_geojson(
        vel["v_east_m_yr"],
        vel["v_north_m_yr"],
        vel["speed_m_yr"],
        vel["direction_deg"],
        x,
        y,
        epsg=32644,
        out_geojson_path=tmp_path / "glacier_velocity.geojson",
        stride=2,
    )
    assert geojson_out.exists()

    # Reproject linear power to EPSG:3857 first, then convert to dB
    lin_power = np.full((8, 8), 0.10, dtype=np.float32)
    x8 = 300000.0 + np.arange(8, dtype=np.float64) * 80.0
    y8 = 3420000.0 - np.arange(8, dtype=np.float64) * 80.0
    warped_db, _, _ = reproject_array_to_3857(
        lin_power, x8, y8, src_epsg=32644, convert_linear_to_db_after=True
    )
    finite_db = warped_db[np.isfinite(warped_db)]
    assert finite_db.size > 0
    assert np.allclose(finite_db, -10.0, atol=1e-2)