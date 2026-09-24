"""
make_fit_animation.py
======================

Renders a GIF of the fit "taking shape" sample by sample, early in a NUTS
chain, as the driver's Fourier coefficients and disk parameters are still
finding their way -- purely a demo/visualisation aid (see the animation
embedded in README.md), not something a real fit needs. Reuses the same
closed-form echo evaluation as ``EchoFit.plot_lightcurve_fits``
(``forward_model.driver_at``/``transfer_coeffs``/``compute_echo``), just
evaluated per-sample in chain order instead of as a posterior-predictive
summary over a random subset.

Two columns, one row per band plus a shared top row:

* top-left: the inferred driving light curve X(t).
* top-right: an illustrative top-down view of the accretion disk, coloured
  by ``forward_model.disk_temperature_profile`` (Shakura-Sunyaev viscous
  temperature, hotter/bluer-in-real-life shown brighter here), squashed
  vertically by cos(inclination) each frame to suggest the tilt, with a
  simple eye-on-a-sphere icon showing the observer's implied vantage point
  (higher above the disk at low inclination, level with it as inclination
  approaches edge-on). This is a genuinely simplified 2-D projection, not a
  3-D/raytraced render, and it always uses the viscous thin-disk profile
  for the picture regardless of which response function the fit itself
  used, since only the thin-disk family has a literal disk geometry to
  show.
* per band: its echo light curve (left) and response function psi(tau)
  (right), mirroring plot_lightcurve_fits's own layout.

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
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

from echofit import model as _model
from echofit.echofit import EchoFit
from echofit.forward_model import (
    transfer_coeffs, compute_echo, driver_at,
    disk_temperature_profile, _schwarzschild_radius_light_days, lag_scaling,
)
from echofit.synthetic import generate_synthetic_dataset

# Purely for the illustrative disk-temperature panel -- the fit itself may
# use bands at other wavelengths; this just sets the picture's colour scale.
_DISK_REFERENCE_WAVELENGTH = 5000.0
_DISK_N_R, _DISK_N_PHI = 40, 80


def _draw_observer(ax, x, y, size):
    """A little eye-bearing sphere marking the observer's vantage point."""
    sphere = mpatches.Circle((x, y), size, facecolor="#2b6cb0", edgecolor="0.2", lw=0.6, zorder=5)
    eye_white = mpatches.Ellipse((x, y), size * 1.5, size * 0.8, facecolor="white", edgecolor="0.2", lw=0.6, zorder=6)
    pupil = mpatches.Circle((x, y), size * 0.22, facecolor="k", zorder=7)
    for p in (sphere, eye_white, pupil):
        ax.add_patch(p)
    return [sphere, eye_white, pupil]


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
    tau_grid_np = np.asarray(ef.tau_grid)

    S, C = jnp.asarray(ef.samples["S"]), jnp.asarray(ef.samples["C"])
    log_mdot, inclination = jnp.asarray(ef.samples["log_mdot"]), jnp.asarray(ef.samples["inclination"])

    def echo_and_psi(S_s, C_s, log_mdot_s, incl_s, wavelength, S_band_s, C_band_s):
        # Via the model module's attribute (not a direct forward_model import)
        # so a swapped model.response_function is honoured here too -- see
        # CLAUDE.md decision #5.
        psi = _model.response_function(ef.tau_grid, log_mdot=log_mdot_s, wavelength=wavelength, inclination=incl_s, M_BH=ef.M_BH)
        A, B = transfer_coeffs(ef.tau_grid, psi, ef.freqs)
        echo = S_band_s * compute_echo(S_s, C_s, ef.freqs, A, B, t_fine) + C_band_s
        return echo, psi

    y_by_band, psi_by_band = {}, {}
    for name, d in ef.bands.items():
        S_band, C_band = jnp.asarray(ef.samples[f"S_{name}"]), jnp.asarray(ef.samples[f"C_{name}"])
        echo_all, psi_all = jax.vmap(
            echo_and_psi, in_axes=(0, 0, 0, 0, None, 0, 0)
        )(S, C, log_mdot, inclination, d["wavelength"], S_band, C_band)
        y_by_band[name] = np.asarray(echo_all)
        psi_by_band[name] = np.asarray(psi_all)
    driver_samples = np.asarray(jax.vmap(lambda S_s, C_s: driver_at(S_s, C_s, ef.freqs, t_fine))(S, C))

    # -- disk-panel geometry (fixed across frames; only colour + squash vary) --
    r_in = 3.0 * float(_schwarzschild_radius_light_days(ef.M_BH))
    tau_ref_per_sample = np.asarray(lag_scaling(log_mdot, _DISK_REFERENCE_WAVELENGTH, ef.M_BH))
    r_out = 6.0 * float(tau_ref_per_sample.max())
    r_grid = np.linspace(r_in, r_out, _DISK_N_R)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, _DISK_N_PHI)
    R_grid, PHI_grid = np.meshgrid(r_grid, phi_grid, indexing="ij")
    X_faceon, Y_faceon = R_grid * np.cos(PHI_grid), R_grid * np.sin(PHI_grid)
    T_per_r = np.asarray(jax.vmap(
        lambda lm: disk_temperature_profile(r_grid, lm, _DISK_REFERENCE_WAVELENGTH, ef.M_BH)
    )(log_mdot))  # (n_frames, n_r)
    obs_x, obs_y_max, obs_size = 1.3 * r_out, 0.9 * r_out, 0.09 * r_out

    n_bands = len(ef.bands)
    fig, axes = plt.subplots(n_bands + 1, 2, figsize=(11, 1.8 * (n_bands + 1)))

    ax_drv, ax_disk = axes[0, 0], axes[0, 1]
    y_pad = 0.15 * (driver_samples.max() - driver_samples.min())
    ax_drv.set_ylim(driver_samples.min() - y_pad, driver_samples.max() + y_pad)
    ax_drv.set_ylabel("driver X(t)")
    ax_drv.set_title("Driving light curve", fontsize=9)
    (driver_line,) = ax_drv.plot([], [], color="0.2", lw=1.5)

    ax_disk.set_xlim(-1.1 * r_out, 1.5 * r_out)
    ax_disk.set_ylim(-1.1 * r_out, 1.1 * r_out)
    ax_disk.set_aspect("equal")
    ax_disk.set_xticks([])
    ax_disk.set_yticks([])
    ax_disk.set_title("Accretion disk (illustrative)", fontsize=9)
    disk_artists = []

    band_lines, psi_lines = {}, {}
    for row, (name, d) in enumerate(ef.bands.items(), start=1):
        ax_lc, ax_psi = axes[row, 0], axes[row, 1]
        ax_lc.errorbar(d["t"], d["y"], yerr=d["yerr"], fmt="o", ms=3, color="k", alpha=0.5, zorder=1)
        y = y_by_band[name]
        pad = 0.15 * (y.max() - y.min())
        ax_lc.set_ylim(min(y.min(), d["y"].min()) - pad, max(y.max(), d["y"].max()) + pad)
        ax_lc.set_ylabel(f"{name} ({d['wavelength']:.0f} Å)")
        (band_lines[name],) = ax_lc.plot([], [], color="C0", lw=1.5, zorder=2)
        ax_lc.sharex(ax_drv)

        psi = psi_by_band[name]
        ax_psi.set_ylim(0.0, 1.05 * psi.max())
        ax_psi.set_xlim(float(tau_grid_np.min()), float(tau_grid_np.max()))
        ax_psi.set_ylabel(f"{name}\nψ(τ)")
        (psi_lines[name],) = ax_psi.plot([], [], color="C3", lw=1.5)
        if row > 1:
            ax_psi.sharex(axes[1, 1])

    axes[-1, 0].set_xlabel("time (days)")
    axes[-1, 1].set_xlabel("lag τ (days)")
    ax_drv.set_xlim(float(t_fine.min()), float(t_fine.max()))
    title = fig.suptitle("")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    def update(i):
        driver_line.set_data(t_fine, driver_samples[i])
        for name in ef.bands:
            band_lines[name].set_data(t_fine, y_by_band[name][i])
            psi_lines[name].set_data(tau_grid_np, psi_by_band[name][i])
        title.set_text(f"MCMC sample {i + 1}/{args.num_frames} -- model taking shape")

        for artist in disk_artists:
            artist.remove()
        disk_artists.clear()
        incl_rad = np.deg2rad(float(inclination[i]))
        Y_proj = Y_faceon * np.cos(incl_rad)
        T_grid = np.broadcast_to(T_per_r[i][:, None], X_faceon.shape)
        mesh = ax_disk.pcolormesh(X_faceon, Y_proj, T_grid[:-1, :-1], cmap="inferno", shading="flat")
        disk_artists.append(mesh)
        obs_y = obs_y_max * np.cos(incl_rad)
        (sightline,) = ax_disk.plot([0.0, obs_x], [0.0, obs_y], "--", color="0.5", lw=0.8, zorder=4)
        disk_artists.append(sightline)
        disk_artists.extend(_draw_observer(ax_disk, obs_x, obs_y, obs_size))

        return [driver_line, title, *band_lines.values(), *psi_lines.values(), *disk_artists]

    ani = FuncAnimation(fig, update, frames=args.num_frames, blit=False)
    out_path = args.outdir / "fit_animation.gif"
    ani.save(out_path, writer="pillow", fps=args.fps, dpi=80)
    plt.close(fig)

    # Re-palette to shrink the file -- a full-colour GIF from matplotlib is
    # far larger than this plot needs.
    from PIL import Image, ImageSequence

    src = Image.open(out_path)
    frames = [f.convert("RGB").quantize(colors=96, method=Image.MEDIANCUT) for f in ImageSequence.Iterator(src)]
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True,
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
