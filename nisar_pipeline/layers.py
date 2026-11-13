"""
HDF5 dataset path constants for NISAR L2/L3 products (GCOV, GUNW, GOFF, SME2).

All paths below have been verified against live ASF DAAC NISAR granules via
`run_site.py inspect`. Any unverified path passed to `require_verified_layer()`
will fail immediately with a RuntimeError.
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Common Identification Metadata
# ---------------------------------------------------------------------------
IDENTIFICATION_GROUP: str = "/science/LSAR/identification"  # VERIFIED

# ---------------------------------------------------------------------------
# L2 GCOV (Geocoded Polarimetric Covariance) - VERIFIED (Frequency A & B)
# ---------------------------------------------------------------------------
GCOV_GRIDS_ROOT: str = "/science/LSAR/GCOV/grids"  # VERIFIED
GCOV_GRID_GROUP: str = "/science/LSAR/GCOV/grids/frequencyA"  # VERIFIED
GCOV_GRID_GROUP_B: str = "/science/LSAR/GCOV/grids/frequencyB"  # VERIFIED
GCOV_HHHH: str = f"{GCOV_GRID_GROUP}/HHHH"  # VERIFIED
GCOV_HVHV: str = f"{GCOV_GRID_GROUP}/HVHV"  # VERIFIED
GCOV_VVVV: str = f"{GCOV_GRID_GROUP}/VVVV"  # VERIFIED
GCOV_VHVH: str = f"{GCOV_GRID_GROUP}/VHVH"  # VERIFIED
GCOV_MASK: str = f"{GCOV_GRID_GROUP}/mask"  # VERIFIED
GCOV_X: str = f"{GCOV_GRID_GROUP}/xCoordinates"  # VERIFIED
GCOV_Y: str = f"{GCOV_GRID_GROUP}/yCoordinates"  # VERIFIED
GCOV_PROJECTION: str = f"{GCOV_GRID_GROUP}/projection"  # VERIFIED
GCOV_CENTER_FREQ_A: str = f"{GCOV_GRID_GROUP}/centerFrequency"  # VERIFIED
GCOV_CENTER_FREQ_B: str = f"{GCOV_GRID_GROUP_B}/centerFrequency"  # VERIFIED
GCOV_INCIDENCE_ANGLE: str = (
    "/science/LSAR/GCOV/metadata/radarGrid/incidenceAngle"  # VERIFIED
)
GCOV_RADAR_GRID_X: str = (
    "/science/LSAR/GCOV/metadata/radarGrid/xCoordinates"  # VERIFIED
)
GCOV_RADAR_GRID_Y: str = (
    "/science/LSAR/GCOV/metadata/radarGrid/yCoordinates"  # VERIFIED
)

GCOV_POLARIZATIONS: tuple[str, ...] = ("HHHH", "HVHV", "VVVV", "VHVH")  # VERIFIED

# ---------------------------------------------------------------------------
# L2 GUNW (Geocoded Unwrapped Interferogram) - VERIFIED via inspect
# ---------------------------------------------------------------------------
GUNW_GRID_GROUP: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HH"  # VERIFIED
)
GUNW_UNWRAPPED_PHASE: str = (
    f"{GUNW_GRID_GROUP}/unwrappedPhase"  # VERIFIED (units: radians)
)
GUNW_COHERENCE: str = (
    f"{GUNW_GRID_GROUP}/coherenceMagnitude"  # VERIFIED (units: 1)
)
GUNW_CONNECTED_COMPONENTS: str = (
    f"{GUNW_GRID_GROUP}/connectedComponents"  # VERIFIED (uint16, _FillValue=65535)
)
GUNW_IONOSPHERE_PHASE: str = (
    f"{GUNW_GRID_GROUP}/ionospherePhaseScreen"  # VERIFIED (units: radians)
)
GUNW_TROPOSPHERE_WET: str = (
    "/science/LSAR/GUNW/metadata/radarGrid/wetTroposphericPhaseScreen"  # VERIFIED
)
GUNW_TROPOSPHERE_HYDROSTATIC: str = (
    "/science/LSAR/GUNW/metadata/radarGrid/hydrostaticTroposphericPhaseScreen"  # VERIFIED
)
GUNW_MASK: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/mask"  # VERIFIED
)
GUNW_X: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/xCoordinates"  # VERIFIED
)
GUNW_Y: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/yCoordinates"  # VERIFIED
)
GUNW_PROJECTION: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/projection"  # VERIFIED
)
GUNW_INCIDENCE_ANGLE: str = (
    "/science/LSAR/GUNW/metadata/radarGrid/incidenceAngle"  # VERIFIED
)

# ---------------------------------------------------------------------------
# L2 GOFF (Geocoded Pixel Offsets) - VERIFIED via inspect
# ---------------------------------------------------------------------------
GOFF_GRID_GROUP: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1"  # VERIFIED
)
GOFF_SLANT_RANGE_OFFSET: str = (
    f"{GOFF_GRID_GROUP}/slantRangeOffset"  # VERIFIED (units: meters)
)
GOFF_ALONG_TRACK_OFFSET: str = (
    f"{GOFF_GRID_GROUP}/alongTrackOffset"  # VERIFIED (units: meters)
)
GOFF_CORRELATION_PEAK: str = (
    f"{GOFF_GRID_GROUP}/correlationSurfacePeak"  # VERIFIED (units: 1)
)
GOFF_SNR: str = (
    f"{GOFF_GRID_GROUP}/snr"  # VERIFIED (units: 1)
)
GOFF_MASK: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/mask"  # VERIFIED
)
GOFF_X: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/xCoordinates"  # VERIFIED
)
GOFF_Y: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/yCoordinates"  # VERIFIED
)
GOFF_PROJECTION: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/projection"  # VERIFIED
)
GOFF_INCIDENCE_ANGLE: str = (
    "/science/LSAR/GOFF/metadata/radarGrid/incidenceAngle"  # VERIFIED
)

# ---------------------------------------------------------------------------
# L3 SME2 (Soil Moisture) - VERIFIED via inspect
# ---------------------------------------------------------------------------
SME2_GRID_GROUP: str = "/science/LSAR/SME2/grids"  # VERIFIED
SME2_SOIL_MOISTURE: str = (
    f"{SME2_GRID_GROUP}/soilMoisture"  # VERIFIED (units: meter^3/meter^3, _FillValue=-9999.0)
)
SME2_QUALITY_FLAG: str = (
    f"{SME2_GRID_GROUP}/retrievalQualityFlag"  # VERIFIED (0=clear, non-zero=anomaly)
)
SME2_X: str = f"{SME2_GRID_GROUP}/xCoordinates"  # VERIFIED
SME2_Y: str = f"{SME2_GRID_GROUP}/yCoordinates"  # VERIFIED
SME2_PROJECTION: str = f"{SME2_GRID_GROUP}/projection"  # VERIFIED (EPSG:6933)

# ---------------------------------------------------------------------------
# Verification Registry & Guard
# ---------------------------------------------------------------------------
VERIFIED_LAYERS: Dict[str, bool] = {
    # GCOV
    GCOV_GRIDS_ROOT: True,
    GCOV_GRID_GROUP: True,
    GCOV_GRID_GROUP_B: True,
    GCOV_HHHH: True,
    GCOV_HVHV: True,
    GCOV_VVVV: True,
    GCOV_VHVH: True,
    GCOV_MASK: True,
    GCOV_X: True,
    GCOV_Y: True,
    GCOV_PROJECTION: True,
    GCOV_CENTER_FREQ_A: True,
    GCOV_CENTER_FREQ_B: True,
    GCOV_INCIDENCE_ANGLE: True,
    # GUNW
    GUNW_GRID_GROUP: True,
    GUNW_UNWRAPPED_PHASE: True,
    GUNW_COHERENCE: True,
    GUNW_CONNECTED_COMPONENTS: True,
    GUNW_IONOSPHERE_PHASE: True,
    GUNW_TROPOSPHERE_WET: True,
    GUNW_TROPOSPHERE_HYDROSTATIC: True,
    GUNW_MASK: True,
    GUNW_X: True,
    GUNW_Y: True,
    GUNW_PROJECTION: True,
    GUNW_INCIDENCE_ANGLE: True,
    # GOFF
    GOFF_GRID_GROUP: True,
    GOFF_SLANT_RANGE_OFFSET: True,
    GOFF_ALONG_TRACK_OFFSET: True,
    GOFF_CORRELATION_PEAK: True,
    GOFF_SNR: True,
    GOFF_MASK: True,
    GOFF_X: True,
    GOFF_Y: True,
    GOFF_PROJECTION: True,
    GOFF_INCIDENCE_ANGLE: True,
    # SME2
    SME2_GRID_GROUP: True,
    SME2_SOIL_MOISTURE: True,
    SME2_QUALITY_FLAG: True,
    SME2_X: True,
    SME2_Y: True,
    SME2_PROJECTION: True,
}


def require_verified_layer(path: str, product: str = "UNKNOWN") -> str:
    """
    Ensure an HDF5 path has been verified via `inspect` before reading it.

    Raises RuntimeError if the path is not marked VERIFIED in VERIFIED_LAYERS.
    """
    if not VERIFIED_LAYERS.get(path, False):
        raise RuntimeError(
            f"HDF5 path '{path}' for product '{product}' is UNVERIFIED. "
            f"Run `python -m nisar_pipeline.run_site inspect --site <site> --products {product}` "
            f"to inspect a live granule, update the constant in `nisar_pipeline/layers.py`, "
            f"and set `VERIFIED_LAYERS['{path}'] = True`."
        )
    return path