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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--outdir", type=Path, default=Path("smoke_test_output"),
        help="Directory to write PNGs + report.html into.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Same well-resolved-lag setup as tests/test_recovery.py: M_BH large
    # enough that mean lags (days) are well above the sampling cadence, so
    # the echo shape is actually visible by eye rather than buried in noise.
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
        noise_level=0.02,
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

    diverging = np.asarray(ef.mcmc.get_extra_fields().get("diverging", []))
    n_div = int(diverging.sum()) if diverging.size else 0
    log_mdot_mean = float(ef.samples["log_mdot"].mean())

    print(f"Fit finished in {dt:.1f}s")
    print(f"Divergent transitions: {n_div}/{len(diverging)}")
    print(f"log_mdot posterior mean: {log_mdot_mean:.3f} (truth: {truth['log_mdot']:.3f})")

    print("Rendering plots...")
    fig_raw, _ = ef.plot_raw_lightcurves()
    fig_fits, _ = ef.plot_lightcurve_fits()
    fig_diag, _ = ef.plot_mcmc_diagnostics()

    paths = {
        "raw": args.outdir / "raw_lightcurves.png",
        "fits": args.outdir / "lightcurve_fits.png",
        "diagnostics": args.outdir / "mcmc_diagnostics.png",
    }
    fig_raw.savefig(paths["raw"], dpi=150, bbox_inches="tight")
    fig_fits.savefig(paths["fits"], dpi=150, bbox_inches="tight")
    fig_diag.savefig(paths["diagnostics"], dpi=150, bbox_inches="tight")

    report_path = args.outdir / "report.html"
    report_path.write_text(_report_html(dt, n_div, len(diverging), log_mdot_mean, truth, paths))

    print(f"\nSaved plots to {args.outdir}/")
    print(f"Open {report_path} to view everything in one page.")


def _report_html(dt, n_div, n_total, log_mdot_mean, truth, paths) -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td>{Path(v).name}</td></tr>" for k, v in paths.items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>echofit smoke test</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
img {{ max-width: 100%; display: block; margin: 1rem 0; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
td, th {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
code {{ background: #f2f2f2; padding: 1px 4px; }}
</style></head>
<body>
<h1>echofit smoke test</h1>
<p>Fit wall time: <b>{dt:.1f}s</b> &nbsp;|&nbsp; Divergent transitions: <b>{n_div}/{n_total}</b>
&nbsp;|&nbsp; log_mdot posterior mean: <b>{log_mdot_mean:.3f}</b> (truth: {truth['log_mdot']:.3f})</p>

<h2>Raw light curves</h2>
<img src="{paths['raw'].name}">

<h2>Posterior-predictive fit + response function</h2>
<p>Shaded bands are 68%/95% credible intervals; black points are the data. The
right-hand panels are the inferred response function &psi;(&tau;) per band.</p>
<img src="{paths['fits'].name}">

<h2>MCMC trace diagnostics</h2>
<p>Traces should look like noisy horizontal bands (well-mixed), not
slow drifts or a chain stuck at one value.</p>
<img src="{paths['diagnostics'].name}">
</body></html>
"""


if __name__ == "__main__":
    main()
