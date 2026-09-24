"""
make_fit_animation.py
======================

Renders a GIF of the fitted driving light curve and per-band echo curves
"taking shape" sample by sample, early in a NUTS chain, as the driver's
Fourier coefficients and disk parameters are still finding their way --
purely a demo/visualisation aid (see the animation embedded in README.md),
not something a real fit needs. Reuses the same closed-form echo evaluation
as ``EchoFit.plot_lightcurve_fits`` (``forward_model.driver_at``/
``transfer_coeffs``/``compute_echo``), just evaluated per-sample in chain
order instead of as a posterior-predictive summary over a random subset.

A short warmup on purpose: the point is to see the chain still finding the
answer, not a converged posterior from frame one.

Usage
-----
    python scripts/make_fit_animation.py
    python scripts/make_fit_animation.py --num-frames 200 --fps 20 --outdir docs/images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from echofit import model as _model
from echofit.echofit import EchoFit
from echofit.forward_model import transfer_coeffs, compute_echo, driver_at
from echofit.synthetic import generate_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-warmup", type=int, default=15, help="Deliberately short -- see module docstring.")
    parser.add_argument("--num-frames", type=int, default=150, help="Post-warmup samples to animate, one per frame.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_dataset(
        bands={"g": 4770.0, "i": 7625.0}, M_BH=1.0e8, n_obs_per_band=40,
        n_freq=20, n_tau=150, tau_max=40.0, noise_level=0.05, seed=0,
    )
    ef = EchoFit(M_BH=1.0e8)
    for name, d in data["bands"].items():
        ef.add_lightcurve(name, wavelength=d["wavelength"], t=d["t"], y=d["y"], yerr=d["yerr"])
    ef.build_grid(n_freq=20, n_tau=150)
    ef.fit(rng_seed=args.seed, num_warmup=args.num_warmup, num_samples=args.num_frames, progress_bar=False)

    all_t = np.concatenate([d["t"] for d in ef.bands.values()])
    t_fine = jnp.linspace(all_t.min() - 5.0, all_t.max() + 5.0, 200)

    S, C = jnp.asarray(ef.samples["S"]), jnp.asarray(ef.samples["C"])
    log_mdot, inclination = jnp.asarray(ef.samples["log_mdot"]), jnp.asarray(ef.samples["inclination"])

    def echo_draw(S_s, C_s, log_mdot_s, incl_s, wavelength, S_band_s, C_band_s):
        # Via the model module's attribute (not a direct forward_model import)
        # so a swapped model.response_function is honoured here too -- see
        # CLAUDE.md decision #5.
        psi = _model.response_function(ef.tau_grid, log_mdot=log_mdot_s, wavelength=wavelength, inclination=incl_s, M_BH=ef.M_BH)
        A, B = transfer_coeffs(ef.tau_grid, psi, ef.freqs)
        return S_band_s * compute_echo(S_s, C_s, ef.freqs, A, B, t_fine) + C_band_s

    y_by_band = {}
    for name, d in ef.bands.items():
        S_band, C_band = jnp.asarray(ef.samples[f"S_{name}"]), jnp.asarray(ef.samples[f"C_{name}"])
        y_by_band[name] = np.asarray(jax.vmap(
            echo_draw, in_axes=(0, 0, 0, 0, None, 0, 0)
        )(S, C, log_mdot, inclination, d["wavelength"], S_band, C_band))
    driver_samples = np.asarray(jax.vmap(lambda S_s, C_s: driver_at(S_s, C_s, ef.freqs, t_fine))(S, C))

    n_bands = len(ef.bands)
    fig, axes = plt.subplots(n_bands + 1, 1, figsize=(7, 1.8 * (n_bands + 1)), sharex=True)
    y_pad = 0.15 * (driver_samples.max() - driver_samples.min())
    axes[0].set_ylim(driver_samples.min() - y_pad, driver_samples.max() + y_pad)
    axes[0].set_ylabel("driver X(t)")
    (driver_line,) = axes[0].plot([], [], color="0.2", lw=1.5)

    band_lines = {}
    for ax, (name, d) in zip(axes[1:], ef.bands.items()):
        ax.errorbar(d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=3, color="k", alpha=0.5, zorder=1)
        y = y_by_band[name]
        pad = 0.15 * (y.max() - y.min())
        ax.set_ylim(min(y.min(), d["y"].min()) - pad, max(y.max(), d["y"].max()) + pad)
        ax.set_ylabel(f"{name} ({d['wavelength']:.0f} Å)")
        (band_lines[name],) = ax.plot([], [], color="C0", lw=1.5, zorder=2)
    axes[-1].set_xlabel("time (days)")
    axes[-1].set_xlim(float(t_fine.min()), float(t_fine.max()))
    title = axes[0].set_title("", pad=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def update(i):
        driver_line.set_data(t_fine, driver_samples[i])
        for name, line in band_lines.items():
            line.set_data(t_fine, y_by_band[name][i])
        title.set_text(f"MCMC sample {i + 1}/{args.num_frames} -- model taking shape")
        return [driver_line, title, *band_lines.values()]

    ani = FuncAnimation(fig, update, frames=args.num_frames, blit=True)
    out_path = args.outdir / "fit_animation.gif"
    ani.save(out_path, writer="pillow", fps=args.fps, dpi=80)
    plt.close(fig)

    # Re-palette to shrink the file -- a full-colour GIF from matplotlib is
    # far larger than this mostly-white, few-colour plot needs.
    from PIL import Image, ImageSequence

    src = Image.open(out_path)
    frames = [f.convert("RGB").quantize(colors=64, method=Image.MEDIANCUT) for f in ImageSequence.Iterator(src)]
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
