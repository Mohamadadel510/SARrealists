#!/usr/bin/env python3
"""
nisar_wildfire.py  --  NISAR GCOV burn-scar explorer (Dancing with the SARs)

Pulls NISAR L2 GCOV (calibrated polarimetric backscatter) for a California
area of interest, streams ONLY the AOI window (no multi-GB download), and plots:
  * single date : co-pol dB, cross-pol dB, RVI, histogram   (what the data looks like)
  * two dates   : pre/post cross-pol dB, dB difference, RVI difference (burn change)

Install:
    pip install asf_search h5py fsspec aiohttp requests pyproj numpy matplotlib
    pip install rasterio      # optional, for GeoTIFF export

Auth:
    Create an Earthdata account, then generate a token at
    https://urs.earthdata.nasa.gov  (Generate Token) and either
        export EDL_TOKEN="eyJ..."        (Linux/macOS)
        $env:EDL_TOKEN="eyJ..."          (Windows PowerShell)
    or paste it when prompted. Searching needs no token; reading data does.

Usage:
    python nisar_wildfire.py search  --aoi palisades --start 2025-10-01
    python nisar_wildfire.py inspect --aoi palisades --start 2025-10-01 --idx 0
    python nisar_wildfire.py plot    --aoi palisades --start 2025-10-01
    python nisar_wildfire.py plot    --bbox -118.66 34.03 -118.52 34.12 \
                                     --start 2026-01-01 --end 2026-03-01 --pre 0 --post 3

NOTE: NISAR public data only starts in late 2025, so a fire BEFORE that has no
pre-fire NISAR image. For a true pre/post change map, choose a fire that
happened after the archive begins and set --bbox / --start / --end around it.
The 'palisades' / 'eaton' presets are rough boxes for exploring burn scars vs.
unburned surroundings in a single image.
"""
import argparse
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
import aiohttp
import fsspec
import numpy as np

# ---------------------------------------------------------------- config ----
PRESETS = {  # (west, south, east, north), approximate -- adjust as needed
    "palisades": (-118.66, 34.03, -118.52, 34.12),
    "eaton": (-118.19, 34.14, -118.05, 34.21),
}
GRID = "/science/LSAR/GCOV/grids/frequencyA"
POLS = ("HHHH", "HVHV", "VVVV", "VHVH")
OUTDIR = Path("nisar_out")


# ---------------------------------------------------------------- search ----
def aoi_wkt(b):
    w, s, e, n = b
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def search_gcov(bbox, start, end):
    import asf_search as asf

    res = asf.search(
        dataset="NISAR",
        processingLevel="GCOV",
        intersectsWith=aoi_wkt(bbox),
        start=start,
        end=end,
    )
    urls = res.find_urls(extension=".h5", directAccess=False)
    items = []
    for r in res:
        p = r.properties
        keys = [str(p.get(k, "")) for k in ("fileID", "sceneName", "fileName")]
        keys = [k.rsplit(".", 1)[0] for k in keys if k]
        url = next((u for u in urls if any(k and k in u for k in keys)), None)
        if url is None:
            continue
        items.append(
            {
                "time": str(p.get("startTime", ""))[:19],
                "track": p.get("pathNumber"),
                "frame": p.get("frameNumber"),
                "dir": p.get("flightDirection"),
                "url": url,
            }
        )
    items.sort(key=lambda d: d["time"])
    return items


def print_table(items):
    print(f"\n{len(items)} GCOV granules found\n")
    print(f"{'idx':>3}  {'acquired (UTC)':<19}  {'track':>5}  {'frame':>5}  dir")
    for i, d in enumerate(items):
        print(f"{i:>3}  {d['time']:<19}  {str(d['track']):>5}  {str(d['frame']):>5}  {d['dir']}")
    print()


def pick_pair(items, pre=None, post=None):
    """Default: earliest + latest acquisition of the biggest same-track group."""
    if pre is not None and post is not None:
        return items[pre], items[post]
    groups = defaultdict(list)
    for d in items:
        groups[(d["track"], d["frame"], d["dir"])].append(d)
    best = max(groups.values(), key=len)
    if len(best) < 2:
        return best[0], None
    return best[0], best[-1]


# ------------------------------------------------------------------ I/O -----
def get_token():
    tok = os.environ.get("EDL_TOKEN") or getpass("Earthdata bearer token: ")
    tok = tok.strip().strip('"').strip("'")
    if tok.lower().startswith("bearer "):
        tok = tok[7:]
    return tok


def open_h5(url, token, download=False):
    import h5py

    if download:
        import requests

        OUTDIR.mkdir(exist_ok=True)
        local = OUTDIR / Path(url).name
        part = local.with_name(local.name + ".part")
        if not local.exists():
            headers = {"Authorization": f"Bearer {token}"}
            done = part.stat().st_size if part.exists() else 0
            if done:
                headers["Range"] = f"bytes={done}-"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                if r.status_code == 416:      # part file is already complete
                    part.rename(local)
                else:
                    r.raise_for_status()
                    mode = "ab" if r.status_code == 206 else "wb"
                    with open(part, mode) as fh:
                        for chunk in r.iter_content(8 * 1024 * 1024):
                            fh.write(chunk)
                    part.rename(local)
        return h5py.File(local, "r")


    fs = fsspec.filesystem(
            "https",
            headers={"Authorization": f"Bearer {token}"},
            client_kwargs={
                "timeout": aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=900)
            },
        )
    f = fs.open(url, mode="rb", block_size=4 * 1024 * 1024, cache_type="blockcache")
    return h5py.File(f, "r")

def inspect_file(h5):
    print("\n--- identification ---")
    ident = h5.get("/science/LSAR/identification")
    if ident is not None:
        for k, v in ident.items():
            try:
                val = v[()]
                val = val.decode() if isinstance(val, bytes) else val
                print(f"  {k}: {val}")
            except Exception:
                pass
    print("\n--- datasets under /science/LSAR/GCOV/grids ---")

    def visit(name, obj):
        if name.startswith("science/LSAR/GCOV/grids") and hasattr(obj, "shape"):
            print(f"  /{name}  shape={obj.shape} dtype={obj.dtype}")

    h5.visititems(visit)


def block_mean(a, n):
    if n <= 1:
        return a
    h, w = (a.shape[0] // n) * n, (a.shape[1] // n) * n
    b = a[:h, :w].reshape(h // n, n, w // n, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3))


def coord_mean(c, n):
    if n <= 1:
        return c
    k = (len(c) // n) * n
    return c[:k].reshape(-1, n).mean(1)


def read_epsg(g):
    ds = g["projection"]
    try:
        return int(np.array(ds[()]).item())
    except Exception:
        return int(ds.attrs["epsg_code"])


def load_window(h5, bbox, looks=2):
    from pyproj import Transformer

    g = h5[GRID]
    x, y = g["xCoordinates"][:], g["yCoordinates"][:]
    epsg = read_epsg(g)
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    w, s, e, n = bbox
    xs, ys = zip(*[tr.transform(lo, la) for lo, la in [(w, s), (e, s), (e, n), (w, n)]])
    ix = np.where((x >= min(xs)) & (x <= max(xs)))[0]
    iy = np.where((y >= min(ys)) & (y <= max(ys)))[0]
    if len(ix) == 0 or len(iy) == 0:
        raise RuntimeError("AOI is outside this granule's footprint.")
    sl = (slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1))

    pols = {}
    for name in POLS:
        if name in g:
            a = g[name][sl].astype("float32")
            a[~np.isfinite(a) | (a <= 0)] = np.nan
            pols[name] = block_mean(a, looks)  # multilook in linear power
    if not pols:
        raise RuntimeError(f"No co/cross-pol layers found. Available: {list(g.keys())}")
    return {
        "pols": pols,
        "x": coord_mean(x[sl[1]], looks),
        "y": coord_mean(y[sl[0]], looks),
        "epsg": epsg,
    }


# ------------------------------------------------------------- products -----
def to_db(a):
    return 10 * np.log10(a)


def co_cross(pols):
    co = "HHHH" if "HHHH" in pols else "VVVV"
    cr = "HVHV" if "HVHV" in pols else "VHVH"
    if co not in pols or cr not in pols:
        raise RuntimeError(f"Need a co- and cross-pol pair; have {list(pols)}")
    return co, cr


def rvi(pols):
    """Dual-pol Radar Vegetation Index: 4*cross / (co + cross). 0=bare, ->1=dense veg."""
    co, cr = co_cross(pols)
    return 4 * pols[cr] / (pols[co] + pols[cr])


def clim(a, lo=2, hi=98):
    return np.nanpercentile(a, lo), np.nanpercentile(a, hi)


# ---------------------------------------------------------------- plots -----
def extent_km(d):
    return [d["x"].min() / 1e3, d["x"].max() / 1e3, d["y"].min() / 1e3, d["y"].max() / 1e3]


def plot_single(d, title, out):
    import matplotlib.pyplot as plt

    co, cr = co_cross(d["pols"])
    co_db, cr_db, r = to_db(d["pols"][co]), to_db(d["pols"][cr]), rvi(d["pols"])
    ext = extent_km(d)
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for a, img, name, cmap in [
        (ax[0, 0], co_db, f"{co} (dB)", "gray"),
        (ax[0, 1], cr_db, f"{cr} (dB)  <- burn scars drop here", "gray"),
        (ax[1, 0], r, "RVI (0 bare ... 1 dense vegetation)", "YlGn"),
    ]:
        lo, hi = clim(img)
        im = a.imshow(img, extent=ext, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(name)
        a.set_xlabel(f"Easting (km, EPSG:{d['epsg']})")
        a.set_ylabel("Northing (km)")
        fig.colorbar(im, ax=a, shrink=0.8)
    ax[1, 1].hist(cr_db[np.isfinite(cr_db)], bins=100, color="tab:green", alpha=0.7, label=cr)
    ax[1, 1].hist(co_db[np.isfinite(co_db)], bins=100, color="tab:blue", alpha=0.5, label=co)
    ax[1, 1].set_title("Backscatter histogram (dB)")
    ax[1, 1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_pair(pre, post, t_pre, t_post, out):
    import matplotlib.pyplot as plt

    # crop to common size (same track/frame should already share a grid)
    h = min(pre["pols"][k].shape[0] for k in pre["pols"]) if pre else 0
    h = min(h, *(post["pols"][k].shape[0] for k in post["pols"]))
    w = min(pre["pols"][k].shape[1] for k in pre["pols"])
    w = min(w, *(post["pols"][k].shape[1] for k in post["pols"]))
    for d in (pre, post):
        d["pols"] = {k: v[:h, :w] for k, v in d["pols"].items()}
        d["x"], d["y"] = d["x"][:w], d["y"][:h]
    if not (np.allclose(pre["x"], post["x"]) and np.allclose(pre["y"], post["y"])):
        print("WARNING: pre/post grids are not identical -- check track/frame match.")

    _, cr = co_cross(pre["pols"])
    _, cr2 = co_cross(post["pols"])
    db_pre, db_post = to_db(pre["pols"][cr]), to_db(post["pols"][cr2])
    d_db = db_post - db_pre
    d_rvi = rvi(post["pols"]) - rvi(pre["pols"])
    ext = extent_km(pre)

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for a, img, name in [(ax[0, 0], db_pre, f"{cr} pre  {t_pre}"), (ax[0, 1], db_post, f"{cr2} post {t_post}")]:
        lo, hi = clim(np.concatenate([db_pre[np.isfinite(db_pre)], db_post[np.isfinite(db_post)]]))
        im = a.imshow(img, extent=ext, cmap="gray", vmin=lo, vmax=hi)
        a.set_title(name + " (dB)")
        fig.colorbar(im, ax=a, shrink=0.8)
    lim = min(10, np.nanpercentile(np.abs(d_db), 98))
    im = ax[1, 0].imshow(d_db, extent=ext, cmap="RdBu", vmin=-lim, vmax=lim)
    ax[1, 0].set_title("Cross-pol change (dB)   red = decrease (vegetation loss)")
    fig.colorbar(im, ax=ax[1, 0], shrink=0.8)
    lim2 = min(0.6, np.nanpercentile(np.abs(d_rvi), 98))
    im = ax[1, 1].imshow(d_rvi, extent=ext, cmap="RdBu", vmin=-lim2, vmax=lim2)
    ax[1, 1].set_title("RVI change   red = decrease")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.8)
    for a in ax.ravel():
        a.set_xlabel(f"Easting (km, EPSG:{pre['epsg']})")
        a.set_ylabel("Northing (km)")
    fig.suptitle("NISAR GCOV burn / vegetation change")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    np.savez_compressed(OUTDIR / "change_arrays.npz", d_db=d_db, d_rvi=d_rvi, x=pre["x"], y=pre["y"], epsg=pre["epsg"])
    try:
        import rasterio
        from rasterio.transform import from_origin

        dx, dy = abs(pre["x"][1] - pre["x"][0]), abs(pre["y"][1] - pre["y"][0])
        tfm = from_origin(pre["x"].min() - dx / 2, pre["y"].max() + dy / 2, dx, dy)
        with rasterio.open(
            OUTDIR / "cross_pol_change_db.tif", "w", driver="GTiff", height=d_db.shape[0],
            width=d_db.shape[1], count=1, dtype="float32", crs=f"EPSG:{pre['epsg']}", transform=tfm, nodata=np.nan,
        ) as dst:
            dst.write(d_db.astype("float32"), 1)
        print(f"saved {OUTDIR / 'cross_pol_change_db.tif'}  (load into MapLibre/QGIS or tile it for the web app)")
    except ImportError:
        print("rasterio not installed -> skipped GeoTIFF export (npz saved).")


# ----------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["search", "inspect", "plot"])
    ap.add_argument("--aoi", choices=PRESETS, default="palisades")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--idx", type=int, default=0, help="granule index for 'inspect' / single-date plot")
    ap.add_argument("--pre", type=int, help="index of pre-event granule")
    ap.add_argument("--post", type=int, help="index of post-event granule")
    ap.add_argument("--looks", type=int, default=2, help="boxcar multilook factor (speckle reduction)")
    ap.add_argument("--download", action="store_true", help="download files instead of streaming")
    a = ap.parse_args()

    bbox = tuple(a.bbox) if a.bbox else PRESETS[a.aoi]
    items = search_gcov(bbox, a.start, a.end)
    if not items:
        sys.exit("No NISAR GCOV granules for that AOI/date range. Widen the dates or check coverage in Vertex.")
    print_table(items)
    if a.cmd == "search":
        return

    OUTDIR.mkdir(exist_ok=True)
    token = get_token()

    if a.cmd == "inspect":
        h5 = open_h5(items[a.idx]["url"], token, a.download)
        inspect_file(h5)
        d = load_window(h5, bbox, a.looks)
        for k, v in d["pols"].items():
            print(f"{k}: window shape={v.shape}  median={np.nanmedian(to_db(v)):.1f} dB")
        return

    pre_i, post_i = pick_pair(items, a.pre, a.post)
    if post_i is None:
        print(f"Only one date on a consistent track -> single-image view ({pre_i['time']})")
        d = load_window(open_h5(pre_i["url"], token, a.download), bbox, a.looks)
        plot_single(d, f"NISAR GCOV  {pre_i['time']}", OUTDIR / "single_date.png")
    else:
        print(f"Pair: pre={pre_i['time']}  post={post_i['time']}")
        d1 = load_window(open_h5(pre_i["url"], token, a.download), bbox, a.looks)
        d2 = load_window(open_h5(post_i["url"], token, a.download), bbox, a.looks)
        plot_single(d2, f"NISAR GCOV  {post_i['time']}", OUTDIR / "single_date.png")
        plot_pair(d1, d2, pre_i["time"][:10], post_i["time"][:10], OUTDIR / "change_map.png")


if __name__ == "__main__":
    main()
