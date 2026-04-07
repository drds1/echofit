import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from .forward_model import lag_scaling, compute_echo
from .forward_model import build_response_function


def plot_lightcurve_fits(samples, data):

    grid_t = np.array(data["grid_t"])
    tau_grid = np.array(data["tau_grid"])
    M_BH = data["M_BH"]
    driver = np.array(data["driver"])

    bands = data["bands"]

    n_bands = len(bands)
    n_rows = n_bands + 1  # +1 for driver

    n_draws = min(100, len(samples["log_mdot"]))
    n_spaghetti = min(8, n_draws)
    rng_idx = np.random.choice(n_draws, n_spaghetti, replace=False)

    # -----------------------------
    # BUILD FIGURE GRID
    # -----------------------------
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, 3 * n_rows),
        gridspec_kw={"width_ratios": [2, 1]},
        sharex="col",
    )

    # ============================================================
    # ROW 0: DRIVING LIGHT CURVE
    # ============================================================
    ax_l = axes[0, 0]
    ax_r = axes[0, 1]

    ax_l.plot(grid_t, driver, color="tab:blue", linewidth=2)
    ax_l.set_title("Driving Light Curve")
    ax_l.set_ylabel("Flux")
    ax_l.grid(alpha=0.2)

    ax_r.axis("off")  # no response function for driver

    # ============================================================
    # ECHO BANDS
    # ============================================================
    for b, band in enumerate(bands):

        lc_preds = []
        psi_preds = []

        # -------------------------
        # build posterior samples
        # -------------------------
        for i in range(n_draws):

            log_mdot = samples["log_mdot"][i]
            inc = samples["inclination"][i]

            echo = compute_echo(
                data["driver"],
                tau_grid,
                log_mdot,
                band["wavelength"],
                inc,
                M_BH,
            )

            S = samples[f"S_{b}"][i]
            C = samples[f"C_{b}"][i]

            lc_preds.append(S * echo + C)

            psi = build_response_function(
                tau_grid,
                log_mdot,
                band["wavelength"],
                inc,
                M_BH,
            )

            psi_preds.append(psi)

        lc_preds = jnp.stack(lc_preds)
        psi_preds = jnp.stack(psi_preds)

        # -------------------------
        # stats helper
        # -------------------------
        def stats(x):
            return (
                jnp.mean(x, axis=0),
                jnp.percentile(x, 16, axis=0),
                jnp.percentile(x, 50, axis=0),
                jnp.percentile(x, 84, axis=0),
            )

        lc_mean, lc_p16, lc_p50, lc_p84 = stats(lc_preds)
        psi_mean, psi_p16, psi_p50, psi_p84 = stats(psi_preds)

        row = b + 1  # shift because row 0 is driver

        ax_l = axes[row, 0]
        ax_r = axes[row, 1]

        # =========================
        # LEFT: LIGHT CURVE
        # =========================
        ax_l.fill_between(grid_t, lc_p16, lc_p84, alpha=0.25)
        ax_l.plot(grid_t, lc_p50, linewidth=2)
        ax_l.plot(grid_t, lc_mean, linestyle="--", linewidth=1.5)

        for i in rng_idx:
            ax_l.plot(grid_t, lc_preds[i], alpha=0.15, linewidth=1)

        ax_l.errorbar(
            band["t"],
            band["y"],
            yerr=band["yerr"],
            fmt="o",
            color="black",
            markersize=3,
        )

        ax_l.set_ylabel(f"Band {b}")
        ax_l.grid(alpha=0.2)

        if b == 0:
            ax_l.set_title("Echo Light Curves + Response Functions")

        # =========================
        # RIGHT: RESPONSE FUNCTION
        # =========================
        ax_r.fill_between(tau_grid, psi_p16, psi_p84, alpha=0.25)
        ax_r.plot(tau_grid, psi_p50, linewidth=2)
        ax_r.plot(tau_grid, psi_mean, linestyle="--", linewidth=1.5)

        for i in rng_idx:
            ax_r.plot(tau_grid, psi_preds[i], alpha=0.15, linewidth=1)

        ax_r.set_ylabel("ψ(τ)")
        ax_r.grid(alpha=0.2)

    # shared labels
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Lag τ")

    plt.tight_layout()
    plt.show()


def plot_mcmc_diagnostics(mcmc):
    samples = mcmc.get_samples()
    
    param_names = list(samples.keys())

    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2.5 * n_params), sharex=True)

    if n_params == 1:
        axes = [axes]

    for i, name in enumerate(param_names):
        ax = axes[i]
        vals = np.array(samples[name])

        ax.plot(vals, alpha=0.7)
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("MCMC step")

    plt.suptitle("Trace plots (MCMC chains)")
    plt.tight_layout()
    plt.show()
