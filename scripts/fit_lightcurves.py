"""
fit_lightcurves.py
===================

Command-line entry point for fitting your own (real or otherwise
already-saved) light curves, without writing any Python. Wraps the same
``EchoFit`` API used throughout the README/notebook -- see there for the
underlying model.

Each light curve file is a plain text file, whitespace- or comma-separated,
three columns ``t y yerr`` (one observation per line; ``#``-prefixed lines
are ignored as comments/headers). Time ``t`` should be in days, on a
consistent zero-point across every band (and the driver, if given).

Usage
-----
Managed run (checkpointed, resumable, writes to outputs/<title>/run_.../ --
see README's "Fitting your own light curves"; forces --num-chains 1):

    python scripts/fit_lightcurves.py \\
        --title ngc_5548 --m-bh 1e8 \\
        --band g 4770 data/g_band.txt \\
        --band i 7625 data/i_band.txt \\
        --num-warmup 1000 --num-samples 2000 \\
        --checkpoint-every 200 --report-every 1000

Resume an interrupted managed run (reuses its original settings; only
--title and --output-dir matter here, everything else is ignored):

    python scripts/fit_lightcurves.py --title ngc_5548 --resume

Quick in-memory run with multiple chains for R-hat/ESS diagnostics (no
--title -- nothing is checkpointed, but a one-off report.html is still
written to --output-dir):

    python scripts/fit_lightcurves.py \\
        --m-bh 1e8 --band g 4770 data/g_band.txt --band i 7625 data/i_band.txt \\
        --num-warmup 1000 --num-samples 1000 --num-chains 4 --chain-method vectorized \\
        --output-dir diagnostic_run

Free-lag band (e.g. an emission line with no assumed physical lag law)
anchored by a driver light curve (see CLAUDE.md decision #7 on why the
driver is required for this to be identifiable):

    python scripts/fit_lightcurves.py \\
        --title ngc_5548_lines --m-bh 1e8 \\
        --band continuum 5100 data/continuum.txt \\
        --free-lag-band halpha 6563 data/halpha.txt \\
        --driver data/continuum.txt \\
        --num-warmup 1000 --num-samples 2000

See scripts/run_example_fit.sh for a fully worked, runnable example
(including generating example data first).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from echofit.echofit import EchoFit
from echofit import reporting


def _load_lightcurve(path: str):
    t, y, yerr = np.loadtxt(path, comments="#", unpack=True)
    return np.atleast_1d(t), np.atleast_1d(y), np.atleast_1d(yerr)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--band", nargs=3, action="append", default=[], metavar=("NAME", "WAVELENGTH", "PATH"),
        help="A lag_mode=\"physical\" band (repeatable): name, wavelength in Angstrom, "
             "path to its t/y/yerr text file.",
    )
    parser.add_argument(
        "--free-lag-band", nargs=3, action="append", default=[], metavar=("NAME", "WAVELENGTH", "PATH"),
        help="A lag_mode=\"free\" band (repeatable) -- an independently inferred lag, "
             "not tied to the others through M_BH/log_mdot. Needs --driver to be "
             "identifiable (see CLAUDE.md decision #7).",
    )
    parser.add_argument(
        "--driver", default=None, metavar="PATH",
        help="Optional direct (zero-lag) observation of the driving light curve -- "
             "required to anchor any --free-lag-band.",
    )
    parser.add_argument("--m-bh", type=float, default=None, help="Fixed black hole mass, solar masses.")

    parser.add_argument("--title", default=None, help="Run name -- enables checkpointed/resumable output management.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest run under --title instead of starting a new fit.")
    parser.add_argument("--output-dir", default=None, help="Output root directory (default: $ECHOFIT_OUTPUT_DIR or ./outputs).")

    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1, help="Ignored (forced to 1) when --title is given -- see CLAUDE.md decision #6.")
    parser.add_argument("--chain-method", default="parallel", choices=("parallel", "vectorized", "sequential"))
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument(
        "--dense-mass", action="store_true",
        help=(
            "Use a full covariance-based NUTS mass matrix instead of the default "
            "diagonal one -- see CLAUDE.md decision #17. Worth trying for most "
            "real runs (found to cut leapfrog steps per sample ~7.5x at no cost "
            "to recovery accuracy), but needs a longer --num-warmup to adapt "
            "properly; check the report's divergence count and raise --num-warmup "
            "if it's above a few percent."
        ),
    )
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Only used with --title.")
    parser.add_argument("--report-every", type=int, default=None, help="Only used with --title -- refresh report.html every this many new samples.")
    parser.add_argument("--no-progress-bar", action="store_true")

    parser.add_argument("--n-freq", type=int, default=None, help="Driver Fourier frequencies (default: build_grid()'s own default).")
    parser.add_argument("--n-tau", type=int, default=None, help="Lag grid points (default: build_grid()'s own default).")
    parser.add_argument("--tau-max", type=float, default=None, help="Maximum lag, days (default: half the observed time baseline).")

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.resume:
        if not args.title:
            raise SystemExit("--resume requires --title.")
        ef = EchoFit.resume(args.title, output_dir=args.output_dir)
        ef.fit(progress_bar=not args.no_progress_bar)
        print(f"Resumed and finished '{args.title}' -> {ef.run_dir}")
        return

    if not args.band and not args.free_lag_band:
        raise SystemExit("Add at least one --band or --free-lag-band.")

    ef = EchoFit(M_BH=args.m_bh, title=args.title, output_dir=args.output_dir)
    for name, wavelength, path in args.band:
        t, y, yerr = _load_lightcurve(path)
        ef.add_lightcurve(name, wavelength=float(wavelength), t=t, y=y, yerr=yerr)
    for name, wavelength, path in args.free_lag_band:
        t, y, yerr = _load_lightcurve(path)
        ef.add_lightcurve(name, wavelength=float(wavelength), t=t, y=y, yerr=yerr, lag_mode="free")
    if args.driver:
        t, y, yerr = _load_lightcurve(args.driver)
        ef.add_driver_lightcurve(t=t, y=y, yerr=yerr)

    build_grid_kwargs = {}
    if args.n_freq is not None:
        build_grid_kwargs["n_freq"] = args.n_freq
    if args.n_tau is not None:
        build_grid_kwargs["n_tau"] = args.n_tau
    if args.tau_max is not None:
        build_grid_kwargs["tau_max"] = args.tau_max
    ef.build_grid(**build_grid_kwargs)

    ef.fit(
        rng_seed=args.rng_seed,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        chain_method=args.chain_method,
        max_tree_depth=args.max_tree_depth,
        dense_mass=args.dense_mass,
        checkpoint_every=args.checkpoint_every,
        report_every=args.report_every,
        progress_bar=not args.no_progress_bar,
    )

    if args.title:
        print(f"Done -> {ef.run_dir}")
    else:
        # No --title: nothing was written to disk automatically (the
        # in-memory fit path, CLAUDE.md decision #6) -- write a one-off
        # report here instead so a plain multi-chain diagnostic run still
        # produces something to look at.
        out_dir = Path(args.output_dir or "fit_output")
        report_path = reporting.generate_report(ef, out_dir)
        print(f"Done -> {report_path}")


if __name__ == "__main__":
    main()
