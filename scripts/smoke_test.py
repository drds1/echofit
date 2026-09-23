"""
smoke_test.py
=============

Quick visual sanity check for the whole pipeline: generate a synthetic
multi-band dataset, run a short NUTS fit (enough to see whether the model
is doing something sensible, not enough to fully converge), and save plots
of the raw data, the posterior-predictive fit, and MCMC trace diagnostics.

This is a "does it look right" check to run by eye after touching
forward_model.py / model.py / echofit.py -- for a numeric pass/fail check
see tests/test_recovery.py.

Usage
-----
    python scripts/smoke_test.py
    python scripts/smoke_test.py --num-warmup 500 --num-samples 500 --outdir /tmp/smoke
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from echofit.synthetic import generate_synthetic_dataset
from echofit.echofit import EchoFit
from echofit import reporting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--noise-level", type=float, default=0.08,
        help=(
            "Fractional noise in the synthetic data. Deliberately higher than "
            "tests/test_recovery.py's 0.02 (which is tuned for a tight numeric "
            "check) so the posterior-predictive credible bands are actually "
            "visible by eye here -- at 0.02 the 95%% CI is ~1%% of the data's "
            "y-range, i.e. a couple of pixels."
        ),
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("smoke_test_output"),
        help="Directory to write PNGs + report.html into.",
    )
    parser.add_argument(
        "--no-gaps", action="store_true",
        help="Disable the default observing gaps (2 weeks from day 50, 3 weeks from day 150).",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Same well-resolved-lag setup as tests/test_recovery.py: M_BH large
    # enough that mean lags (days) are well above the sampling cadence, so
    # the echo shape is actually visible by eye rather than buried in noise.
    # Two observing gaps make the campaign more realistic (and harder for
    # NUTS/the frequency-grid estimate) than fully uniform random sampling.
    gaps = None if args.no_gaps else [(50, 14), (150, 21)]
    data = generate_synthetic_dataset(
        M_BH=1.0e9,
        log_mdot_true=0.0,
        inclination_true=35.0,
        sigma_drw_true=0.3,
        tau_drw_true=30.0,
        bands={"g": 4770.0, "i": 7625.0},
        t_span=200.0,
        n_obs_per_band=50,
        n_freq=15,
        n_tau=100,
        tau_max=50.0,
        noise_level=args.noise_level,
        gaps=gaps,
        seed=args.seed,
    )
    truth = data["truth"]

    ef = EchoFit(M_BH=truth["M_BH"])
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=15, n_tau=100)

    print(f"Fitting ({args.num_warmup} warmup + {args.num_samples} samples, 1 chain)...")
    t0 = time.time()
    ef.fit(
        rng_seed=0, num_warmup=args.num_warmup, num_samples=args.num_samples,
        num_chains=1, progress_bar=False,
    )
    dt = time.time() - t0

    diverging = np.asarray(ef.extra_fields.get("diverging", []))
    n_div = int(diverging.sum()) if diverging.size else 0
    log_mdot_mean = float(ef.samples["log_mdot"].mean())

    print(f"Fit finished in {dt:.1f}s")
    print(f"Divergent transitions: {n_div}/{len(diverging)}")
    print(f"log_mdot posterior mean: {log_mdot_mean:.3f} (truth: {truth['log_mdot']:.3f})")

    print("Rendering plots...")
    report_path = reporting.generate_report(ef, args.outdir, fit_seconds=dt, truth=truth)

    print(f"\nSaved plots to {args.outdir}/")
    print(f"Open {report_path} to view everything in one page.")


if __name__ == "__main__":
    main()
