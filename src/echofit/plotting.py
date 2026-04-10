import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from .forward_model import compute_echo
from .forward_model import build_response_function
import corner


def plot_triangle(samples, fixed_params=None):
    """
    Triangle (corner) plot of posterior samples.
    Automatically handles fixed parameters.
    """

    fixed_params = fixed_params or {}
    resolver = ParamResolver(samples, fixed_params)

    # -----------------------------------
    # Collect parameters to plot
    # -----------------------------------
    param_names = []

    for name in samples.keys():
        if name == "loglike":
            continue
        param_names.append(name)

    # also include fixed params (optional)
    for name in fixed_params:
        if name not in param_names:
            param_names.append(name)

    # -----------------------------------
    # Build sample matrix
    # -----------------------------------
    chains = []
    labels = []

    for name in param_names:
        try:
            vals = resolver(name)
            if np.std(vals) == 0:
                continue
            chains.append(vals)
            labels.append(name)
        except KeyError:
            continue

    if len(chains) == 0:
        raise ValueError("No parameters available for triangle plot.")

    X = np.vstack(chains).T  # shape (n_samples, n_params)

    # -----------------------------------
    # Make corner plot
    # -----------------------------------
    fig = corner.corner(
        X,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        quantiles=[0.16, 0.5, 0.84],
    )

    plt.suptitle("Posterior Distributions (Triangle Plot)")
    plt.show()


class ParamResolver:
    def __init__(self, samples, fixed_params):
        self.samples = samples
        self.fixed = fixed_params or {}

        # find any array-like entry safely
        lengths = []

        for v in samples.values():
            try:
                v_arr = np.asarray(v)
                if v_arr.ndim > 0:
                    lengths.append(len(v_arr))
            except Exception:
                pass

        if len(lengths) == 0:
            raise ValueError("No valid sampled parameters found in MCMC output.")

        self.n = max(lengths)

    def __call__(self, name, i=None):
        # sampled parameter
        if name in self.samples:
            arr = np.array(self.samples[name])
            if i is None:
                return arr
            return arr[i]

        # fixed parameter
        if name in self.fixed:
            val = self.fixed[name]
            if i is None:
                return np.full(self.n, val)
            return val

        raise KeyError(f"Parameter '{name}' not found in samples or fixed_params")


def build_convolution_matrix(grid_t, tau_grid, psi):
    """
    Build convolution matrix H such that:
    (H @ X) ≈ convolution of X with psi
    """
    N = len(grid_t)
    H = np.zeros((N, N))

    dt = grid_t[1] - grid_t[0]

    for i in range(N):
        for j in range(N):
            lag = grid_t[i] - grid_t[j]
            if lag < 0:
                continue

            k = int(lag / dt)
            if k < len(psi):
                H[i, j] = psi[k]

    return H


def reconstruct_driver_posterior(data, resolver):
    """
    Proper GP posterior reconstruction of latent driver.
    Returns mean and std.
    """

    grid_t = np.array(data["grid_t"])
    tau_grid = np.array(data["tau_grid"])
    bands = data["bands"]
    M_BH = data["M_BH"]

    # --- posterior hyperparameters ---
    log_tau = np.mean(resolver("log_tau_drw"))
    log_sigma = np.mean(resolver("log_sigma"))

    tau = np.exp(log_tau)
    sigma = np.exp(log_sigma)

    # --- DRW covariance ---
    dt = np.abs(grid_t[:, None] - grid_t[None, :])
    Kxx = sigma**2 * np.exp(-dt / (tau + 1e-8))

    H_blocks = []
    y_blocks = []

    for b, band in enumerate(bands):

        # mean parameters for reconstruction
        log_mdot = np.mean(resolver("log_mdot"))
        inc = np.mean(resolver("inclination"))
        S = np.mean(resolver(f"S_{b}"))
        C = np.mean(resolver(f"C_{b}"))

        # --- response function ---
        psi = build_response_function(
            tau_grid,
            log_mdot,
            band["wavelength"],
            inc,
            M_BH,
        )

        # --- convolution operator ---
        H_conv = build_convolution_matrix(grid_t, tau_grid, psi)

        # --- sampling operator ---
        t_obs = np.array(band["t"])
        W = np.zeros((len(t_obs), len(grid_t)))

        for i, to in enumerate(t_obs):
            idx = np.argmin(np.abs(grid_t - to))
            W[i, idx] = 1.0

        # --- full operator ---
        H_band = S * (W @ H_conv)

        H_blocks.append(H_band)
        y_blocks.append(np.array(band["y"]) - C)

    H = np.vstack(H_blocks)
    y_all = np.concatenate(y_blocks)

    # --- noise ---
    noise = np.concatenate([b["yerr"] for b in bands])
    R = np.diag(noise**2 + 1e-6)

    # --- GP conditioning ---
    Ky = H @ Kxx @ H.T + R
    Kxy = Kxx @ H.T

    alpha = np.linalg.solve(Ky, y_all)
    mu = Kxy @ alpha

    # --- uncertainty ---
    cov = Kxx - Kxy @ np.linalg.solve(Ky, Kxy.T)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))

    return mu, std


def plot_lightcurve_fits(samples, data, fixed_params=None):

    fixed_params = fixed_params or {}
    resolver = ParamResolver(samples, fixed_params)

    niterations = len(samples["log_mdot"])

    grid_t = np.array(data["grid_t"])
    tau_grid = np.array(data["tau_grid"])
    M_BH = data["M_BH"]

    # parameter chains
    log_mdot_chain = resolver("log_mdot")
    inc_chain = resolver("inclination")
    log_tau_drw_chain = resolver("log_tau_drw")
    log_sigma_chain = resolver("log_sigma")

    driver, driver_std = reconstruct_driver_posterior(data, resolver)

    bands = data["bands"]

    n_bands = len(bands)
    n_rows = n_bands + 1  # +1 for driver

    n_draws = min(100, niterations)
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

    ax_l.plot(grid_t, driver, color="tab:blue", linewidth=2, label="mean")

    ax_l.fill_between(
        grid_t,
        driver - driver_std,
        driver + driver_std,
        alpha=0.3,
        label="±1σ",
    )

    ax_l.legend()
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
        S_chain = resolver(f"S_{b}")
        C_chain = resolver(f"C_{b}")

        for i in range(n_draws):

            log_mdot = log_mdot_chain[i]
            inc = inc_chain[i]

            echo = compute_echo(
                driver,
                tau_grid,
                log_mdot,
                band["wavelength"],
                inc,
                M_BH,
            )

            S = S_chain[i]
            C = C_chain[i]

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


def plot_diagnostics_extended(mcmc, data, fixed_params=None):
    samples = mcmc.get_samples()

    # ----------------------------------------
    # 1. BADNESS OF FIT TRACE
    # ----------------------------------------
    loglike = np.array(samples["loglike"])
    cost = -loglike

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(cost, alpha=0.8)
    ax.set_title("Negative Log-Likelihood (Cost)")
    ax.set_xlabel("MCMC step")
    ax.set_ylabel("-log L")
    ax.grid(alpha=0.3)

    # Optional: rolling mean to show stabilization
    window = max(10, len(cost) // 50)
    smooth = np.convolve(cost, np.ones(window) / window, mode="valid")
    ax.plot(
        np.arange(len(smooth)) + window // 2,
        smooth,
        linewidth=2,
        label="smoothed",
    )
    ax.legend()

    # ----------------------------------------
    # 2. POWER SPECTRUM OF DRIVER
    # ----------------------------------------
    resolver = ParamResolver(samples, fixed_params)
    driver, _ = reconstruct_driver_posterior(data, resolver)
    grid_t = np.array(data["grid_t"])

    dt = np.median(np.diff(grid_t))

    # FFT
    fft = np.fft.rfft(driver)
    freq = np.fft.rfftfreq(len(driver), dt)
    power = np.abs(fft) ** 2

    ax = axes[1]

    ax.plot(freq[1:], power[1:])  # skip zero freq
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Driver Power Spectrum")

    # ----------------------------------------
    # DRW timescale marker
    # ----------------------------------------
    tau_drw = np.exp(np.mean(samples["log_tau_drw"]))
    f_drw = 1.0 / tau_drw

    ax.axvline(f_drw, linestyle="--", linewidth=2, label="1/tau_drw")

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
