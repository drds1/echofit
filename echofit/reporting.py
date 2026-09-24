"""
reporting.py
============

Shared visual-report generation for a fitted ``EchoFit``: the same plots
and ``report.html`` used by ``scripts/smoke_test.py`` and by
``EchoFit.fit(title=...)``'s automatic per-run report. Kept separate from
``echofit.py`` so both call sites (and any future ones) share one
implementation instead of duplicating the HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Sample sites that aren't single scalars per posterior draw (per-frequency
# driver coefficients, per-observation predictions) -- excluded from the
# scalar summary table.
_NON_SCALAR_PREFIXES = ("y_pred_", "S_raw", "C_raw")
_NON_SCALAR_NAMES = ("S", "C")


def _scalar_param_names(samples: dict):
    names = []
    for k, v in samples.items():
        if k in _NON_SCALAR_NAMES or k.startswith(_NON_SCALAR_PREFIXES):
            continue
        if np.asarray(v).ndim == 1:
            names.append(k)
    return sorted(names)


def generate_report(
    ef,
    out_dir,
    fit_seconds: Optional[float] = None,
    truth: Optional[dict] = None,
    title: Optional[str] = None,
) -> Path:
    """Render the standard plot set for a fitted ``ef`` and write
    ``report.html`` (plus the PNGs it references) into ``out_dir``.

    Parameters
    ----------
    ef : EchoFit
        Already-fitted (``ef.samples`` populated).
    out_dir : path-like
        Directory to write into (created if missing).
    fit_seconds : float, optional
        Wall time of the fit, shown in the report header if given.
    truth : dict, optional
        Ground-truth parameter values (as in ``synthetic.generate_synthetic_dataset``'s
        ``data["truth"]``) to compare the posterior against -- only
        meaningful for synthetic data. Omit for real-data runs.
    title : str, optional
        Run title, shown in the report header.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_raw, _ = ef.plot_raw_lightcurves()
    fig_fits, _ = ef.plot_lightcurve_fits()
    fig_power, _ = ef.plot_power_spectrum()
    fig_diag, _ = ef.plot_mcmc_diagnostics()

    paths = {
        "raw": out_dir / "raw_lightcurves.png",
        "fits": out_dir / "lightcurve_fits.png",
        "power": out_dir / "power_spectrum.png",
        "diagnostics": out_dir / "mcmc_diagnostics.png",
    }
    import matplotlib.pyplot as plt

    fig_raw.savefig(paths["raw"], dpi=150, bbox_inches="tight")
    fig_fits.savefig(paths["fits"], dpi=150, bbox_inches="tight")
    fig_power.savefig(paths["power"], dpi=150, bbox_inches="tight")
    fig_diag.savefig(paths["diagnostics"], dpi=150, bbox_inches="tight")
    figs_to_close = [fig_raw, fig_fits, fig_power, fig_diag]

    # log_mdot/inclination only exist if at least one band used
    # lag_mode="physical" (CLAUDE.md decision #7) -- skip the corner plot
    # rather than error for a free-lag-only fit.
    if {"log_mdot", "inclination"} <= set(ef.samples):
        true_values = {"log_mdot": _truth_value_for("log_mdot", truth), "inclination": _truth_value_for("inclination", truth)}
        true_values = {k: v for k, v in true_values.items() if v is not None} or None
        fig_corner, _ = ef.plot_corner(true_values=true_values)
        paths["corner"] = out_dir / "corner.png"
        fig_corner.savefig(paths["corner"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_corner)

    band_param_names = [f"{p}_{name}" for name in ef.bands for p in ("S", "C")]
    true_values = {name: _truth_value_for(name, truth) for name in band_param_names}
    true_values = {k: v for k, v in true_values.items() if v is not None} or None
    fig_bands, _ = ef.plot_corner_bands(true_values=true_values)
    paths["corner_bands"] = out_dir / "corner_bands.png"
    fig_bands.savefig(paths["corner_bands"], dpi=150, bbox_inches="tight")
    figs_to_close.append(fig_bands)

    if any(d["lag_mode"] == "free" for d in ef.bands.values()):
        free_lag_names = [f"tau_{name}" for name, d in ef.bands.items() if d["lag_mode"] == "free"]
        true_values = {name: _truth_value_for(name, truth) for name in free_lag_names}
        true_values = {k: v for k, v in true_values.items() if v is not None} or None
        fig_free_lag, _ = ef.plot_corner_free_lag(true_values=true_values)
        paths["corner_free_lag"] = out_dir / "corner_free_lag.png"
        fig_free_lag.savefig(paths["corner_free_lag"], dpi=150, bbox_inches="tight")
        figs_to_close.append(fig_free_lag)

    fig_fourier = ef.plot_fourier_correlation()[0]
    paths["fourier_correlation"] = out_dir / "fourier_correlation.png"
    fig_fourier.savefig(paths["fourier_correlation"], dpi=150, bbox_inches="tight")
    figs_to_close.append(fig_fourier)

    for fig in figs_to_close:
        plt.close(fig)

    diverging = np.asarray(ef.extra_fields.get("diverging", []))
    n_div = int(diverging.sum()) if diverging.size else 0
    n_total = len(diverging)

    report_path = out_dir / "report.html"
    report_path.write_text(
        _report_html(ef, paths, fit_seconds, n_div, n_total, truth, title)
    )
    return report_path


def _truth_value_for(name: str, truth: Optional[dict]):
    """Look up ``name``'s ground-truth value from a
    ``synthetic.generate_*_dataset``-style ``truth`` dict, or None if there
    isn't one (real data, or a name truth has nothing to say about).
    Shared by the summary table and the corner plots' true-value crosshairs
    so the two don't drift apart."""
    if truth is None:
        return None
    if name in truth:
        return truth[name]
    if name in ("S_driver", "C_driver"):
        return truth.get("driver", {}).get(name)
    if name.startswith("tau_"):
        return truth.get("bands", {}).get(name[len("tau_"):], {}).get("tau")
    if name.startswith(("S_", "C_")):
        prefix, band = name.split("_", 1)
        return truth.get("bands", {}).get(band, {}).get(f"{prefix}_band")
    return None


def _summary_table_html(ef, truth: Optional[dict]) -> str:
    rows = []
    for name in _scalar_param_names(ef.samples):
        s = np.asarray(ef.samples[name])
        truth_val = _truth_value_for(name, truth)
        truth_cell = f"{truth_val:.4g}" if truth_val is not None else "&mdash;"
        rows.append(
            f"<tr><td>{name}</td><td>{s.mean():.4g}</td><td>{s.std():.4g}</td>"
            f"<td>{truth_cell}</td></tr>"
        )
    return (
        "<table><tr><th>parameter</th><th>posterior mean</th>"
        "<th>posterior std</th><th>truth</th></tr>" + "".join(rows) + "</table>"
    )


def _report_html(ef, paths, fit_seconds, n_div, n_total, truth, title) -> str:
    header_bits = []
    if title:
        header_bits.append(f"run: <b>{title}</b>")
    if fit_seconds is not None:
        header_bits.append(f"fit wall time: <b>{fit_seconds:.1f}s</b>")
    header_bits.append(f"divergent transitions: <b>{n_div}/{n_total}</b>")
    header_line = " &nbsp;|&nbsp; ".join(header_bits)

    corner_sections = []
    if "corner" in paths:
        corner_sections.append(f"""
<h2>Posterior corner plot: disk parameters</h2>
<p>log_mdot / inclination, coloured per chain. Chains that land in visibly
different places here are the same thing a Gelman-Rubin R-hat check would
flag, made visible -- see CLAUDE.md's note on why a single chain isn't
sufficient evidence of convergence.</p>
<img src="{paths['corner'].name}">
""")
    if "corner_bands" in paths:
        corner_sections.append(f"""
<h2>Posterior corner plot: band offset/stretch parameters</h2>
<p>S_band / C_band for every band -- the linear scale and offset absorbing
each band's own flux calibration, not physically meaningful on their own
but worth checking for the same reason as the disk corner plot above:
chains disagreeing here means the fit hasn't converged.</p>
<img src="{paths['corner_bands'].name}">
""")
    if "corner_free_lag" in paths:
        corner_sections.append(f"""
<h2>Posterior corner plot: free-lag (top-hat) centroids</h2>
<p>tau_band for every lag_mode="free" band -- the independently inferred
lag each such band's top-hat response is centred on (see
CLAUDE.md decision #7 on why these need a driver light curve to be
identifiable at all).</p>
<img src="{paths['corner_free_lag'].name}">
""")
    if "fourier_correlation" in paths:
        corner_sections.append(f"""
<h2>Driver Fourier coefficient correlation</h2>
<p>Posterior correlation matrix of the driver's S/C Fourier coefficients
(pooled across chains). Mostly-diagonal (near zero off-diagonal) is what
the non-centred DRW prior parameterisation assumes; strong off-diagonal
structure would be worth a closer look.</p>
<img src="{paths['fourier_correlation'].name}">
""")
    corner_section = "".join(corner_sections)

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>echofit report{f' -- {title}' if title else ''}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
img {{ max-width: 100%; display: block; margin: 1rem 0; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
td, th {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
code {{ background: #f2f2f2; padding: 1px 4px; }}
</style></head>
<body>
<h1>echofit report</h1>
<p>{header_line}</p>

<h2>Posterior summary</h2>
{_summary_table_html(ef, truth)}

<h2>Raw light curves</h2>
<img src="{paths['raw'].name}">

<h2>Posterior-predictive fit + response function</h2>
<p>Shaded bands are 68%/95% credible intervals; black points are the data. The
top panel is the inferred driving light curve (time-aligned with the bands
below it), extended a bit before/after the data -- the credible band should
widen roughly like t^(1/2) outside the data before saturating, since the
driver is a DRW. The right-hand panels are the inferred response function
&psi;(&tau;) per band.</p>
<img src="{paths['fits'].name}">

<h2>Driver power spectrum</h2>
<p>Posterior P(&omega;) = (S<sup>2</sup>+C<sup>2</sup>)/(2&Delta;&omega;) per
frequency (black squares) vs. the fitted DRW Lorentzian from posterior
sigma_drw/tau_drw draws (blue dashed) and a plain &omega;<sup>-2</sup>
random-walk reference (red dotted). These should roughly track each other --
if the posterior power spectrum diverges from the Lorentzian shape a lot,
that's worth a closer look.</p>
<img src="{paths['power'].name}">

<h2>MCMC trace diagnostics</h2>
<p>Traces should look like noisy horizontal bands (well-mixed), not
slow drifts or a chain stuck at one value.</p>
<img src="{paths['diagnostics'].name}">
{corner_section}</body></html>
"""
