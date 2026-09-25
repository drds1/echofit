"""
download_ngc5548_storm_data.py
================================

Downloads the real NGC 5548 AGN STORM continuum light curves from VizieR
(catalog J/ApJ/821/56, Fausnaugh, Denney, Barth, et al. 2016, ApJ 821, 56,
"Space Telescope and Optical Reverberation Mapping Project III: Optical
Continuum Emission and Broadband Time Delays in NGC 5548" -- D. Starkey is
a co-author) and reshapes them into the ``t y yerr`` per-band text files
``scripts/fit_lightcurves.py`` reads, the same format
``scripts/make_example_data.py`` writes for synthetic data.

13 bands total: 9 ground-based optical filters (Johnson/Cousins BVRI, SDSS
ugriz, table3.dat) plus 4 HST/COS UV continuum windows (1157.5, 1367,
1478.5, 1746 Angstrom, table4.dat) -- real, irregular cadence, real gaps,
real systematics from 16 different observatories, spanning ~1160-9160
Angstrom. This is exactly the shape of dataset ``lag_mode="physical"`` is
built for, with a well-documented literature result (lags follow the
lambda^(4/3) disk-reprocessing scaling, but imply a disk ~3x larger than
standard thin-disk theory predicts) to sanity-check a fit against.

Wavelengths used are the paper's own Table 5 pivot wavelengths (not a
generic/external reference), read directly from the paper text, since
they're specific to this campaign's actual filter responses and the
atmospheric cutoffs applied.

Usage
-----
    python scripts/download_ngc5548_storm_data.py
    python scripts/download_ngc5548_storm_data.py --outdir ngc5548_data
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

VIZIER_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/821/56"

# The paper's own Table 5 pivot wavelengths (Angstrom), computed from each
# filter's actual response curve with atmospheric cutoffs at 3000 A and 1
# micron imposed -- not a generic external reference.
OPTICAL_WAVELENGTHS = {
    "u": 3472.0, "B": 4369.0, "g": 4776.0, "V": 5404.0, "r": 6176.0,
    "R": 6440.0, "i": 7648.0, "I": 8561.0, "z": 9157.0,
}
# table4.dat's own 4 discrete HST/COS continuum windows (Angstrom).
UV_BAND_NAMES = {1157.5: "uv1158", 1367.0: "uv1367", 1478.5: "uv1479", 1746.0: "uv1746"}


def _download(name: str, raw_dir: Path) -> Path:
    path = raw_dir / name
    if not path.exists():
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(f"{VIZIER_BASE}/{name}", path)
    return path


def _parse_table3(path: Path) -> dict:
    """Optical continuum light curves: filter, HJD-2400000, flux, e_flux,
    telescope, differential counts, e_differential counts (ReadMe's
    byte-by-byte description of table3.dat) -- only the first four are
    needed here."""
    bands = {name: {"t": [], "y": [], "yerr": []} for name in OPTICAL_WAVELENGTHS}
    for line in path.read_text().splitlines():
        filt, hjd, flux, e_flux = line.split()[:4]
        bands[filt]["t"].append(float(hjd))
        bands[filt]["y"].append(float(flux))
        bands[filt]["yerr"].append(float(e_flux))
    return bands


def _parse_table4(path: Path) -> dict:
    """HST/COS UV continuum light curves: wavelength, HJD-2400000, flux,
    e_flux (ReadMe's byte-by-byte description of table4.dat)."""
    bands = {name: {"t": [], "y": [], "yerr": []} for name in UV_BAND_NAMES.values()}
    for line in path.read_text().splitlines():
        lam, hjd, flux, e_flux = line.split()
        name = UV_BAND_NAMES[float(lam)]
        bands[name]["t"].append(float(hjd))
        bands[name]["y"].append(float(flux))
        bands[name]["yerr"].append(float(e_flux))
    return bands


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("ngc5548_storm_data"))
    args = parser.parse_args()
    raw_dir = args.outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    table3_path = _download("table3.dat", raw_dir)
    table4_path = _download("table4.dat", raw_dir)
    _download("ReadMe", raw_dir)

    bands = {}
    for name, d in _parse_table3(table3_path).items():
        bands[name] = (OPTICAL_WAVELENGTHS[name], d)
    for name, d in _parse_table4(table4_path).items():
        wavelength = next(lam for lam, n in UV_BAND_NAMES.items() if n == name)
        bands[name] = (wavelength, d)

    # A shared time origin across every band (not per band) -- relative
    # lags between bands are the whole point, so they all need to share
    # one zero point.
    t0 = min(min(d["t"]) for _, d in bands.values())

    print(f"\n{'band':<8}{'wavelength (A)':>16}{'n points':>12}")
    for name, (wavelength, d) in sorted(bands.items(), key=lambda kv: kv[1][0]):
        t = np.asarray(d["t"]) - t0
        y = np.asarray(d["y"])
        yerr = np.asarray(d["yerr"])
        order = np.argsort(t)
        path = args.outdir / f"{name}_band.txt"
        np.savetxt(
            path, np.column_stack([t[order], y[order], yerr[order]]),
            header=f"t(days) y yerr -- band {name!r}, wavelength {wavelength:.1f} A "
                   f"(flux in 1e-15 erg/cm^2/s/A, Fausnaugh et al. 2016)",
            fmt="%.6f",
        )
        print(f"{name:<8}{wavelength:>16.1f}{len(t):>12}")

    band_args = " ".join(
        f"--band {name} {wavelength:.1f} {args.outdir}/{name}_band.txt"
        for name, (wavelength, _) in sorted(bands.items(), key=lambda kv: kv[1][0])
    )
    print(f"\nWrote {len(bands)} band files to {args.outdir}/")
    print(
        "\nNGC 5548's black hole mass varies noticeably across the literature "
        "(the campaign itself caught NGC 5548 in an unusual 'BLR holiday' state) "
        "-- commonly cited values range from ~5e7 Msun (classic single-epoch/RM "
        "catalog estimates) up to ~2.6e8 Msun (a more recent multi-season RM "
        "result). Check the current literature and pick one deliberately rather "
        "than trusting a default; example below uses 5e7."
    )
    print(f"\nExample fit (all 13 bands, a real multi-week run):")
    print(f"  python scripts/fit_lightcurves.py --m-bh 5e7 {band_args} \\")
    print(f"      --title ngc5548_storm --num-warmup 1000 --num-samples 2000 --dense-mass")
    print(f"\nFor a faster first look, try a handful of bands spanning the wavelength "
          f"range instead, e.g. --band uv1158 ... --band u ... --band g ... --band z ...")


if __name__ == "__main__":
    main()
