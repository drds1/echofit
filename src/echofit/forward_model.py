import jax
import jax.numpy as jnp


@jax.jit
def forward_model(
    a,
    b,
    X_sin,
    X_cos,
    xray_obs_time,
    xray_obs_flux,
    xray_sigma,
    time_dict,
    flux_dict,
    sigma_dict,
    t_model,
    cache_dt,
    M_BH,
    acc_rate,
    incl,
    frequencies,
    wavelengths,
):
    """
    Fully deterministic model:
    Fourier + echo + likelihood terms (no NumPyro here)
    """

    # =========================================================
    # 1. FOURIER DRIVER
    # =========================================================
    xray_model = jnp.sum(X_sin * a, axis=1) + jnp.sum(X_cos * b, axis=1)

    # =========================================================
    # 2. OPTIONAL X-RAY LIKELIHOOD
    # =========================================================
    def xray_ll():
        xray_interp = jnp.interp(xray_obs_time, t_model, xray_model)
        return -0.5 * jnp.sum(((xray_obs_flux - xray_interp) / xray_sigma) ** 2)

    # =========================================================
    # 3. ECHO MODEL
    # =========================================================
    tau0 = (M_BH / 1e8) ** (1 / 3) * (acc_rate) ** (1 / 3)
    width = 0.3 + 0.7 * jnp.sin(incl)

    def band_loglike(wavelength, t_obs, flux_obs, sigma_obs):
        tau = tau0 * (wavelength / 5000.0) ** (4 / 3)

        kernel = jnp.exp(-0.5 * ((cache_dt - tau) / width) ** 2)
        kernel = kernel / jnp.sum(kernel, axis=1, keepdims=True)

        y = kernel @ xray_model

        y_interp = jnp.interp(t_obs, t_model, y)

        return -0.5 * jnp.sum(((flux_obs - y_interp) / sigma_obs) ** 2)

    # =========================================================
    # 4. TOTAL LOG LIKELIHOOD
    # =========================================================
    logp = 0.0

    if "xray" in time_dict:
        logp += xray_ll()

    for band in flux_dict.keys():
        if band == "xray":
            continue

        logp += band_loglike(
            wavelengths[band],
            time_dict[band],
            flux_dict[band],
            sigma_dict[band],
        )

    return logp