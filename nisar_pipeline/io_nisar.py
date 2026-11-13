"""
ASF DAAC search, deterministic de-duplication, AOI footprint overlap scoring,
Earthdata token handling, retry-wrapped fsspec/h5py streaming, resumable download
fallback, and HDF5 inspect.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import aiohttp
import fsspec
import h5py
import numpy as np
import requests

from nisar_pipeline.layers import IDENTIFICATION_GROUP

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(
    r"_(?P<stage>[PRX])(?P<crid>\d{5})(?:_[A-Z0-9])+?_(?P<counter>\d{2,3})(?:\.h5)?$"
    r"|_(?P<stage_short>[PRX])(?P<crid_short>\d{5})_(?P<counter_short>\d{2,3})(?:\.h5)?$",
    re.IGNORECASE,
)

_STAGE_RANK: Dict[str, int] = {"X": 1, "P": 2, "R": 3}


class UnsupportedGranuleError(RuntimeError):
    """Raised when a granule is structurally incompatible and should not be retried."""


@dataclass(frozen=True)
class GranuleRecord:
    """Normalized metadata record for a single NISAR granule."""

    time: str  # YYYY-MM-DDTHH:MM:SS
    date_str: str  # YYYYMMDD
    track: int
    frame: int
    direction: str
    file_name: str
    url: str
    version_tag: str
    version_sort_key: Tuple[int, int, int, int, str]
    secondary_time: Optional[str] = None
    aoi_overlap_pct: float = 100.0

    @property
    def group_key(self) -> Tuple[int, int, str]:
        """Return (pathNumber, frameNumber, flightDirection) tuple."""
        return (self.track, self.frame, self.direction)


def aoi_wkt(bbox: Sequence[float]) -> str:
    """Convert (west, south, east, north) bounding box to WKT POLYGON."""
    if len(bbox) != 4:
        raise ValueError(f"Expected 4-element bbox (W, S, E, N), got {bbox}")
    w, s, e, n = [float(v) for v in bbox]
    if w >= e or s >= n:
        raise ValueError(f"Invalid bbox (W < E and S < N required): {bbox}")
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _extract_geometry_bbox(geom: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    """Extract (west, south, east, north) from a GeoJSON Polygon/MultiPolygon dictionary."""
    if not geom or not isinstance(geom, dict):
        return None
    coords = geom.get("coordinates")
    if not coords:
        return None
    lons: List[float] = []
    lats: List[float] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[0], (int, float)) and isinstance(node[1], (int, float)):
                lons.append(float(node[0]))
                lats.append(float(node[1]))
            else:
                for child in node:
                    _walk(child)

    _walk(coords)
    if not lons or not lats:
        return None
    return (min(lons), min(lats), max(lons), max(lats))


def compute_bbox_overlap_pct(
    aoi_bbox: Sequence[float],
    footprint_bbox: Optional[Sequence[float]],
) -> float:
    """
    Compute percentage (0..100) of `aoi_bbox` area covered by `footprint_bbox`.
    """
    if not footprint_bbox or len(footprint_bbox) != 4:
        return 100.0
    aw, as_, ae, an = [float(v) for v in aoi_bbox]
    fw, fs, fe, fn = [float(v) for v in footprint_bbox]

    aoi_area = max(1e-9, (ae - aw) * (an - as_))
    iw = max(aw, fw)
    is_ = max(as_, fs)
    ie = min(ae, fe)
    in_ = min(an, fn)

    if ie <= iw or in_ <= is_:
        return 0.0
    inter_area = (ie - iw) * (in_ - is_)
    return round(min(100.0, 100.0 * inter_area / aoi_area), 1)


def get_edl_token(
    env_var: str = "EDL_TOKEN",
    token_file: Path | str = "/content/.edl_token",
) -> str:
    """
    Read Earthdata Login bearer token from environment variable, `/content/.edl_token`
    cache, or interactive prompt. Never prints or logs the token value.
    """
    raw = os.environ.get(env_var, "").strip()

    tpath = Path(token_file)
    if not raw and tpath.exists():
        try:
            raw = tpath.read_text(encoding="utf-8").strip()
        except Exception:
            raw = ""

    if not raw and os.isatty(0):
        from getpass import getpass

        raw = getpass("Earthdata bearer token: ").strip()

    tok = raw.strip('"').strip("'")
    if tok.lower().startswith("bearer "):
        tok = tok[7:].strip()

    if not tok:
        raise RuntimeError(
            f"Missing Earthdata bearer token in environment variable '{env_var}'. "
            "In Google Colab, load it from Secrets into os.environ['EDL_TOKEN']."
        )
    return tok


def parse_nisar_version(
    file_name: str,
) -> Tuple[str, Tuple[int, int, int, int, str], bool]:
    """
    Parse production version from a NISAR HDF5 filename.
    Operational ('R') > Provisional ('P') > CalVal ('X'), and Nominal ('_PR_') > Urgent ('_UR_').
    """
    stem = Path(file_name).name
    is_nominal_pr = 1 if "_PR_" in stem else 0
    match = _VERSION_RE.search(stem)
    if match:
        stage = (match.group("stage") or match.group("stage_short")).upper()
        crid = int(match.group("crid") or match.group("crid_short"))
        counter = int(match.group("counter") or match.group("counter_short"))
        stage_rank = _STAGE_RANK.get(stage, 0)
        tag = f"{stage}{crid:05d}_{counter:03d}"
        return tag, (stage_rank, crid, counter, is_nominal_pr, stem), True
    return "UNKNOWN", (0, 0, 0, is_nominal_pr, stem), False


def _match_url_for_result(props: Dict[str, Any], urls: Sequence[str]) -> Optional[str]:
    """Match an ASF search result property dictionary to its .h5 HTTPS URL."""
    candidates: List[str] = []
    for k in ("fileName", "fileID", "sceneName"):
        val = props.get(k)
        if val:
            s = str(val).strip()
            candidates.append(s)
            candidates.append(s.rsplit(".", 1)[0])
    prop_url = props.get("url")
    if isinstance(prop_url, str) and prop_url.endswith(".h5"):
        return prop_url
    for u in urls:
        if any(c and c in u for c in candidates):
            return u
    return None


def normalize_and_dedupe_granules(
    raw_items: Iterable[Dict[str, Any]],
    aoi_bbox: Optional[Sequence[float]] = None,
) -> List[GranuleRecord]:
    """
    Convert raw search dictionaries into GranuleRecords and deterministically
    de-duplicate same-timestamp granules within each (track, frame, direction) group.
    """
    records: List[GranuleRecord] = []
    for item in raw_items:
        url = str(item["url"])
        file_name = str(item.get("file_name") or Path(urlparse(url).path).name)
        time_str = str(item.get("time", ""))[:19]
        date_str = time_str[:10].replace("-", "")
        track = int(item.get("track") or -1)
        frame = int(item.get("frame") or -1)
        direction = str(item.get("direction") or "U")[0].upper()
        secondary_time = item.get("secondary_time")
        sec_str = str(secondary_time)[:19] if secondary_time else None

        overlap_pct = float(item.get("aoi_overlap_pct", 100.0))
        if aoi_bbox is not None and "footprint_bbox" in item:
            overlap_pct = compute_bbox_overlap_pct(aoi_bbox, item.get("footprint_bbox"))

        v_tag, v_key, parsed_ok = parse_nisar_version(file_name)
        if not parsed_ok:
            logger.debug("Could not parse CRID version from filename: %s", file_name)

        records.append(
            GranuleRecord(
                time=time_str,
                date_str=date_str,
                track=track,
                frame=frame,
                direction=direction,
                file_name=file_name,
                url=url,
                version_tag=v_tag,
                version_sort_key=v_key,
                secondary_time=sec_str,
                aoi_overlap_pct=overlap_pct,
            )
        )

    by_slot: Dict[Tuple[int, int, str, str], List[GranuleRecord]] = defaultdict(list)
    for rec in records:
        slot = (rec.track, rec.frame, rec.direction, rec.time)
        by_slot[slot].append(rec)

    deduped: List[GranuleRecord] = []
    for slot, group in by_slot.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        group_sorted = sorted(group, key=lambda r: r.version_sort_key)
        chosen = group_sorted[-1]
        dropped = [r.file_name for r in group_sorted[:-1]]
        if any(r.version_tag == "UNKNOWN" for r in group_sorted):
            logger.warning(
                "Duplicate granules at %s (track=%s, frame=%s, dir=%s) had unparseable "
                "version tags; deterministically selected '%s' and dropped %s.",
                slot[3],
                slot[0],
                slot[1],
                slot[2],
                chosen.file_name,
                dropped,
            )
        else:
            logger.info(
                "De-duplicated %d granules at %s (track=%s, frame=%s, dir=%s): "
                "selected '%s' (%s), dropped %s.",
                len(group),
                slot[3],
                slot[0],
                slot[1],
                slot[2],
                chosen.file_name,
                chosen.version_tag,
                dropped,
            )
        deduped.append(chosen)

    deduped.sort(key=lambda r: (r.time, r.track, r.frame, r.direction))
    return deduped


def search_nisar(
    bbox: Sequence[float],
    processing_level: str,
    start: str,
    end: str,
    track: Optional[int] = None,
    frame: Optional[int] = None,
    direction: Optional[str] = None,
    max_retries: int = 3,
    retry_sleep_sec: float = 5.0,
    search_fn: Optional[Callable[..., Any]] = None,
) -> List[GranuleRecord]:
    """
    Search ASF DAAC for NISAR granules, compute AOI footprint overlap %, and de-duplicate.
    """
    if search_fn is None:
        import asf_search as asf

        search_fn = asf.search

    wkt = aoi_wkt(bbox)
    last_err: Optional[Exception] = None
    res = None

    for attempt in range(1, max_retries + 1):
        try:
            res = search_fn(
                dataset="NISAR",
                processingLevel=processing_level.upper(),
                intersectsWith=wkt,
                start=start,
                end=end,
            )
            break
        except Exception as exc:
            last_err = exc
            logger.warning(
                "ASF search attempt %d/%d failed (%s). Retrying in %.1fs...",
                attempt,
                max_retries,
                exc,
                retry_sleep_sec,
            )
            if attempt < max_retries:
                time.sleep(retry_sleep_sec)

    if res is None:
        raise RuntimeError(f"ASF search failed after {max_retries} attempts: {last_err}")

    urls: List[str] = list(res.find_urls(extension=".h5", directAccess=False))
    raw_items: List[Dict[str, Any]] = []

    for r in res:
        p = getattr(r, "properties", {}) or {}
        url = _match_url_for_result(p, urls)
        if url is None:
            continue
        file_name = str(p.get("fileName") or Path(urlparse(url).path).name)
        geom = getattr(r, "geometry", None)
        fp_bbox = _extract_geometry_bbox(geom)
        raw_items.append(
            {
                "time": str(p.get("startTime", ""))[:19],
                "secondary_time": p.get("stopTime"),
                "track": p.get("pathNumber") or -1,
                "frame": p.get("frameNumber") or -1,
                "direction": str(p.get("flightDirection") or "U"),
                "file_name": file_name,
                "url": url,
                "footprint_bbox": fp_bbox,
            }
        )

    records = normalize_and_dedupe_granules(raw_items, aoi_bbox=bbox)

    if track is not None:
        records = [r for r in records if r.track == int(track)]
    if frame is not None:
        records = [r for r in records if r.frame == int(frame)]
    if direction is not None:
        dir_norm = direction.strip()[0].upper()
        records = [r for r in records if r.direction == dir_norm]

    return records


def group_by_track_frame(
    records: Sequence[GranuleRecord],
) -> Dict[Tuple[int, int, str], List[GranuleRecord]]:
    """
    Group GranuleRecords by (pathNumber, frameNumber, flightDirection).
    Comparisons and time-series stacks are strictly restricted to a single group.
    """
    groups: Dict[Tuple[int, int, str], List[GranuleRecord]] = defaultdict(list)
    for r in records:
        groups[r.group_key].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.time)
    return dict(groups)


def select_primary_group(
    records: Sequence[GranuleRecord],
) -> Tuple[Tuple[int, int, str], List[GranuleRecord]]:
    """
    Select the best (track, frame, direction) group by prioritizing:
      1. Groups with at least 2 dates (so pairwise change / stacks can be built)
      2. Highest median AOI footprint overlap % (binned to nearest 5% so a frame covering
         85% of the Nile Delta beats a coastal frame covering only 20% of the top edge)
      3. Number of acquisitions in the group
      4. Most recent acquisition timestamp
    """
    if not records:
        raise ValueError("Cannot select a group from an empty granule list.")
    groups = group_by_track_frame(records)

    def _group_score(k: Tuple[int, int, str]) -> Tuple[int, float, int, str]:
        grp = groups[k]
        has_pair = 1 if len(grp) >= 2 else 0
        med_overlap = float(np.median([r.aoi_overlap_pct for r in grp]))
        overlap_bin = round(med_overlap / 5.0) * 5.0
        return (has_pair, overlap_bin, len(grp), grp[-1].time)

    best_key = max(groups.keys(), key=_group_score)
    chosen_grp = groups[best_key]
    med_ov = float(np.median([r.aoi_overlap_pct for r in chosen_grp]))
    logger.info(
        "Selected primary group (track=%d, frame=%d, dir=%s): %d date(s), median AOI overlap=%.1f%%",
        best_key[0],
        best_key[1],
        best_key[2],
        len(chosen_grp),
        med_ov,
    )
    return best_key, chosen_grp


def print_search_table(records: Sequence[GranuleRecord], product: str = "GCOV") -> None:
    """Print formatted summary table of discovered NISAR granules (including AOI overlap %)."""
    print(f"\n{len(records)} NISAR {product} granules (after de-duplication)\n")
    header = (
        f"{'idx':>3}  {'acquired (UTC)':<19}  {'track':>5}  "
        f"{'frame':>5}  {'dir':>3}  {'overlap':>7}  {'version':<10}  file_name"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(records):
        print(
            f"{i:>3}  {r.time:<19}  {r.track:>5}  "
            f"{r.frame:>5}  {r.direction:>3}  {r.aoi_overlap_pct:>6.1f}%  {r.version_tag:<10}  {r.file_name}"
        )
    print()


def download_granule_resumable(
    url: str,
    token: str,
    raw_dir: Path | str = "/content/nisar_raw",
    max_retries: int = 30,
    chunk_mb: int = 8,
    sleep_sec: float = 5.0,
) -> Path:
    """
    Download a granule to local disk with `.part` file resumption, Range headers,
    and 30 retries for unreliable connections.
    """
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(urlparse(url).path).name
    local_path = out_dir / file_name
    part_path = local_path.with_name(local_path.name + ".part")

    if local_path.exists() and local_path.stat().st_size > 0:
        logger.info("Using cached local granule: %s", local_path)
        return local_path

    chunk_bytes = max(1, int(chunk_mb)) * 1024 * 1024
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            done = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Authorization": f"Bearer {token}"}
            if done > 0:
                headers["Range"] = f"bytes={done}-"
                logger.info(
                    "Resuming download of %s from byte %d (attempt %d/%d)",
                    file_name,
                    done,
                    attempt,
                    max_retries,
                )
            else:
                logger.info(
                    "Starting download of %s (attempt %d/%d)",
                    file_name,
                    attempt,
                    max_retries,
                )

            with requests.get(
                url, headers=headers, stream=True, timeout=(30, 300)
            ) as resp:
                if resp.status_code == 416:
                    part_path.rename(local_path)
                    return local_path

                resp.raise_for_status()
                mode = "ab" if (done > 0 and resp.status_code == 206) else "wb"
                with open(part_path, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            fh.write(chunk)

            part_path.rename(local_path)
            logger.info("Completed download: %s", local_path)
            return local_path

        except Exception as exc:
            last_err = exc
            logger.warning(
                "Download error on %s (attempt %d/%d): %s. Sleeping %.1fs...",
                file_name,
                attempt,
                max_retries,
                exc,
                sleep_sec,
            )
            if attempt < max_retries:
                time.sleep(sleep_sec)

    raise RuntimeError(
        f"Failed to download {file_name} after {max_retries} attempts: {last_err}"
    )


def open_nisar_h5(
    url_or_path: str | Path,
    token: Optional[str] = None,
    download: bool = False,
    raw_dir: Path | str = "/content/nisar_raw",
    block_size_mb: int = 4,
    sock_connect: int = 60,
    sock_read: int = 900,
    download_retries: int = 30,
) -> h5py.File:
    """
    Open a local or remote NISAR HDF5 file.
    By default, streams via fsspec HTTPS + aiohttp with blockcache (2-4 MB blocks).
    """
    path_str = str(url_or_path)
    if not path_str.startswith(("http://", "https://")):
        return h5py.File(path_str, "r")

    if not token:
        token = get_edl_token()

    if download:
        local_file = download_granule_resumable(
            url=path_str,
            token=token,
            raw_dir=raw_dir,
            max_retries=download_retries,
        )
        return h5py.File(local_file, "r")

    block_bytes = max(2, min(4, int(block_size_mb))) * 1024 * 1024
    fs = fsspec.filesystem(
        "https",
        headers={"Authorization": f"Bearer {token}"},
        client_kwargs={
            "timeout": aiohttp.ClientTimeout(
                total=None, sock_connect=sock_connect, sock_read=sock_read
            )
        },
    )
    remote_file = fs.open(
        path_str,
        mode="rb",
        block_size=block_bytes,
        cache_type="blockcache",
    )
    return h5py.File(remote_file, "r")


def run_with_h5_retry(
    url_or_path: str | Path,
    reader_fn: Callable[[h5py.File], Any],
    token: Optional[str] = None,
    download: bool = False,
    raw_dir: Path | str = "/content/nisar_raw",
    max_tries: int = 6,
    sleep_sec: float = 10.0,
    block_size_mb: int = 4,
    open_fn: Optional[Callable[[], h5py.File]] = None,
) -> Any:
    """
    Execute `reader_fn(h5)` inside a retry loop (default 6 tries, 10 s sleep)
    that closes and reopens the HDF5 file handle on transient network failures.
    Does not retry if `UnsupportedGranuleError` is raised.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        h5: Optional[h5py.File] = None
        try:
            if open_fn is not None:
                h5 = open_fn()
            else:
                h5 = open_nisar_h5(
                    url_or_path=url_or_path,
                    token=token,
                    download=download,
                    raw_dir=raw_dir,
                    block_size_mb=block_size_mb,
                )
            return reader_fn(h5)
        except UnsupportedGranuleError:
            raise
        except Exception as exc:
            last_err = exc
            logger.warning(
                "HDF5 read attempt %d/%d failed on %s: %s",
                attempt,
                max_tries,
                Path(str(url_or_path)).name,
                exc,
            )
            if attempt < max_tries:
                time.sleep(sleep_sec)
        finally:
            if h5 is not None:
                try:
                    h5.close()
                except Exception:
                    pass

    raise RuntimeError(
        f"HDF5 operation failed after {max_tries} attempts on {url_or_path}: {last_err}"
    ) from last_err


def _decode_attr(val: Any) -> Any:
    """Convert HDF5 attribute values (bytes, numpy scalars/arrays) to clean Python types."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return _decode_attr(val.item())
        if val.size <= 8:
            return [_decode_attr(x) for x in val.ravel().tolist()]
        return f"array(shape={val.shape}, dtype={val.dtype})"
    if isinstance(val, np.generic):
        return val.item()
    return val


def inspect_granule(h5: h5py.File, print_output: bool = True) -> List[Dict[str, Any]]:
    """
    List every dataset path, shape, dtype, and key attributes in a NISAR HDF5 granule.
    """
    ident_info: Dict[str, Any] = {}
    ident = h5.get(IDENTIFICATION_GROUP)
    if ident is not None and isinstance(ident, h5py.Group):
        for k, v in ident.items():
            try:
                raw_val = v[()]
                ident_info[k] = _decode_attr(raw_val)
            except Exception:
                continue

    datasets: List[Dict[str, Any]] = []
    interesting_attrs = (
        "units",
        "description",
        "long_name",
        "standard_name",
        "_FillValue",
        "epsg_code",
        "spatial_ref",
        "grid_mapping",
    )

    def _visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            attrs_summary: Dict[str, Any] = {}
            for ak in interesting_attrs:
                if ak in obj.attrs:
                    attrs_summary[ak] = _decode_attr(obj.attrs[ak])
            datasets.append(
                {
                    "path": f"/{name.lstrip('/')}",
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": attrs_summary,
                }
            )

    h5.visititems(_visitor)

    if print_output:
        print("\n=== NISAR Granule Identification ===")
        for k, v in ident_info.items():
            print(f"  {k}: {v}")
        print(f"\n=== HDF5 Datasets ({len(datasets)} total) ===")
        print(f"{'Path':<72}  {'Shape':<18}  {'Dtype':<10}  Attributes")
        print("-" * 120)
        for d in datasets:
            attr_str = ", ".join(f"{k}={v!r}" for k, v in d["attrs"].items())
            print(
                f"{d['path']:<72}  {str(d['shape']):<18}  {d['dtype']:<10}  {attr_str}"
            )
        print()

    return datasets