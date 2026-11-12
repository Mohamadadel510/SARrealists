"""
HDF5 dataset path constants for NISAR L2/L3 products (GCOV, GUNW, GOFF, SME2).

Rule:
- Verified paths are marked `# VERIFIED`.
- Unconfirmed paths are marked `# UNVERIFIED - confirm with inspect` and flagged
  as False in `VERIFIED_LAYERS`. Calling `require_verified_layer()` on an
  unverified path raises a RuntimeError with instructions to run `inspect`.
"""
from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Common Identification Metadata
# ---------------------------------------------------------------------------
IDENTIFICATION_GROUP: str = "/science/LSAR/identification"  # VERIFIED

# ---------------------------------------------------------------------------
# L2 GCOV (Geocoded Polarimetric Covariance) - VERIFIED
# ---------------------------------------------------------------------------
GCOV_GRID_GROUP: str = "/science/LSAR/GCOV/grids/frequencyA"  # VERIFIED
GCOV_HHHH: str = f"{GCOV_GRID_GROUP}/HHHH"  # VERIFIED
GCOV_HVHV: str = f"{GCOV_GRID_GROUP}/HVHV"  # VERIFIED
GCOV_VVVV: str = f"{GCOV_GRID_GROUP}/VVVV"  # VERIFIED
GCOV_VHVH: str = f"{GCOV_GRID_GROUP}/VHVH"  # VERIFIED
GCOV_MASK: str = f"{GCOV_GRID_GROUP}/mask"  # VERIFIED
GCOV_X: str = f"{GCOV_GRID_GROUP}/xCoordinates"  # VERIFIED
GCOV_Y: str = f"{GCOV_GRID_GROUP}/yCoordinates"  # VERIFIED
GCOV_PROJECTION: str = f"{GCOV_GRID_GROUP}/projection"  # VERIFIED
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
# L2 GUNW (Geocoded Unwrapped Interferogram) - UNVERIFIED
# Confirm exact paths using:
#   python -m nisar_pipeline.run_site inspect --site <site> --product GUNW
# ---------------------------------------------------------------------------
GUNW_GRID_GROUP: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/HHHH"  # UNVERIFIED - confirm with inspect
)
GUNW_UNWRAPPED_PHASE: str = (
    f"{GUNW_GRID_GROUP}/unwrappedPhase"  # UNVERIFIED - confirm with inspect
)
GUNW_COHERENCE: str = (
    f"{GUNW_GRID_GROUP}/coherenceMagnitude"  # UNVERIFIED - confirm with inspect
)
GUNW_CONNECTED_COMPONENTS: str = (
    f"{GUNW_GRID_GROUP}/connectedComponents"  # UNVERIFIED - confirm with inspect
)
GUNW_IONOSPHERE_PHASE: str = (
    f"{GUNW_GRID_GROUP}/ionospherePhaseScreen"  # UNVERIFIED - confirm with inspect
)
GUNW_TROPOSPHERE_WET: str = (
    "/science/LSAR/GUNW/metadata/radarGrid/troposphereWet"  # UNVERIFIED - confirm with inspect
)
GUNW_TROPOSPHERE_HYDROSTATIC: str = (
    "/science/LSAR/GUNW/metadata/radarGrid/troposphereHydrostatic"  # UNVERIFIED - confirm with inspect
)
GUNW_MASK: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/mask"  # UNVERIFIED - confirm with inspect
)
GUNW_X: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/xCoordinates"  # UNVERIFIED - confirm with inspect
)
GUNW_Y: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/yCoordinates"  # UNVERIFIED - confirm with inspect
)
GUNW_PROJECTION: str = (
    "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/projection"  # UNVERIFIED - confirm with inspect
)
GUNW_INCIDENCE_ANGLE: str = (
    "/science/LSAR/GUNW/metadata/radarGrid/incidenceAngle"  # UNVERIFIED - confirm with inspect
)

# ---------------------------------------------------------------------------
# L2 GOFF (Geocoded Pixel Offsets) - UNVERIFIED
# Confirm exact paths using:
#   python -m nisar_pipeline.run_site inspect --site himalaya --product GOFF
# ---------------------------------------------------------------------------
GOFF_GRID_GROUP: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HHHH/layer1"  # UNVERIFIED - confirm with inspect
)
GOFF_SLANT_RANGE_OFFSET: str = (
    f"{GOFF_GRID_GROUP}/slantRangeOffset"  # UNVERIFIED - confirm with inspect
)
GOFF_ALONG_TRACK_OFFSET: str = (
    f"{GOFF_GRID_GROUP}/alongTrackOffset"  # UNVERIFIED - confirm with inspect
)
GOFF_CORRELATION_PEAK: str = (
    f"{GOFF_GRID_GROUP}/correlationSurfacePeak"  # UNVERIFIED - confirm with inspect
)
GOFF_SNR: str = (
    f"{GOFF_GRID_GROUP}/snr"  # UNVERIFIED - confirm with inspect
)
GOFF_MASK: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/mask"  # UNVERIFIED - confirm with inspect
)
GOFF_X: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/xCoordinates"  # UNVERIFIED - confirm with inspect
)
GOFF_Y: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/yCoordinates"  # UNVERIFIED - confirm with inspect
)
GOFF_PROJECTION: str = (
    "/science/LSAR/GOFF/grids/frequencyA/pixelOffsets/projection"  # UNVERIFIED - confirm with inspect
)
GOFF_INCIDENCE_ANGLE: str = (
    "/science/LSAR/GOFF/metadata/radarGrid/incidenceAngle"  # UNVERIFIED - confirm with inspect
)

# ---------------------------------------------------------------------------
# L3 SME2 (Soil Moisture) - UNVERIFIED
# Confirm exact paths using:
#   python -m nisar_pipeline.run_site inspect --site nile_delta --product SME2
# ---------------------------------------------------------------------------
SME2_GRID_GROUP: str = (
    "/science/LSAR/SME2/grids"  # UNVERIFIED - confirm with inspect
)
SME2_SOIL_MOISTURE: str = (
    f"{SME2_GRID_GROUP}/soilMoisture"  # UNVERIFIED - confirm with inspect
)
SME2_QUALITY_FLAG: str = (
    f"{SME2_GRID_GROUP}/qualityFlag"  # UNVERIFIED - confirm with inspect
)
SME2_X: str = (
    f"{SME2_GRID_GROUP}/xCoordinates"  # UNVERIFIED - confirm with inspect
)
SME2_Y: str = (
    f"{SME2_GRID_GROUP}/yCoordinates"  # UNVERIFIED - confirm with inspect
)
SME2_PROJECTION: str = (
    f"{SME2_GRID_GROUP}/projection"  # UNVERIFIED - confirm with inspect
)

# ---------------------------------------------------------------------------
# Verification Registry & Guard
# ---------------------------------------------------------------------------
VERIFIED_LAYERS: Dict[str, bool] = {
    # GCOV
    GCOV_GRID_GROUP: True,
    GCOV_HHHH: True,
    GCOV_HVHV: True,
    GCOV_VVVV: True,
    GCOV_VHVH: True,
    GCOV_MASK: True,
    GCOV_X: True,
    GCOV_Y: True,
    GCOV_PROJECTION: True,
    GCOV_INCIDENCE_ANGLE: True,
    # GUNW (Flip to True after running `inspect --product GUNW`)
    GUNW_GRID_GROUP: False,
    GUNW_UNWRAPPED_PHASE: False,
    GUNW_COHERENCE: False,
    GUNW_CONNECTED_COMPONENTS: False,
    GUNW_IONOSPHERE_PHASE: False,
    GUNW_TROPOSPHERE_WET: False,
    GUNW_TROPOSPHERE_HYDROSTATIC: False,
    GUNW_MASK: False,
    GUNW_X: False,
    GUNW_Y: False,
    GUNW_PROJECTION: False,
    GUNW_INCIDENCE_ANGLE: False,
    # GOFF (Flip to True after running `inspect --product GOFF`)
    GOFF_GRID_GROUP: False,
    GOFF_SLANT_RANGE_OFFSET: False,
    GOFF_ALONG_TRACK_OFFSET: False,
    GOFF_CORRELATION_PEAK: False,
    GOFF_SNR: False,
    GOFF_MASK: False,
    GOFF_X: False,
    GOFF_Y: False,
    GOFF_PROJECTION: False,
    GOFF_INCIDENCE_ANGLE: False,
    # SME2 (Flip to True after running `inspect --product SME2`)
    SME2_GRID_GROUP: False,
    SME2_SOIL_MOISTURE: False,
    SME2_QUALITY_FLAG: False,
    SME2_X: False,
    SME2_Y: False,
    SME2_PROJECTION: False,
}


def require_verified_layer(path: str, product: str = "UNKNOWN") -> str:
    """
    Ensure an HDF5 path has been verified via `inspect` before reading it.

    Raises RuntimeError if the path is still marked UNVERIFIED in VERIFIED_LAYERS.
    """
    if not VERIFIED_LAYERS.get(path, False):
        raise RuntimeError(
            f"HDF5 path '{path}' for product '{product}' is UNVERIFIED. "
            f"Run `python -m nisar_pipeline.run_site inspect --site <site> --product {product}` "
            f"to inspect a live granule, update the constant in `nisar_pipeline/layers.py`, "
            f"and set `VERIFIED_LAYERS['{path}'] = True`."
        )
    return path