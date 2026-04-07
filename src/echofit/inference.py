import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax.scipy.linalg import cho_factor, cho_solve

from .forward_model import build_response_function


# ----------------------------
# DRW covariance
# ----------------------------
def drw_covariance(t, sigma, tau):
    dt = jnp.abs(t[:, None] - t[None, :])
    return sigma**2 * jnp.exp(-dt / (tau + 1e-8))


# ----------------------------
# NUMPYRO MODEL
# ----------------------------
def model(data):
    bands = data["bands"]
    tau_grid = data["tau_grid"]
    M_BH = data["M_BH"]

    # -------------------------
    # DRW hyperparameters
    # -------------------------
    log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 1.0))
    log_tau_drw = numpyro.sample("log_tau_drw", dist.Normal(2.0, 1.0))

    sigma = jnp.exp(log_sigma)
    tau_drw = jnp.exp(log_tau_drw)

    # -------------------------
    # Disk parameters
    # -------------------------
    log_mdot = numpyro.sample("log_mdot", dist.Normal(0.0, 1.0))
    inclination = numpyro.sample("inclination", dist.Uniform(0.0, jnp.pi / 2))

    # -------------------------
    # Per-band parameters
    # -------------------------
    S_list, C_list = [], []

    for b in range(len(bands)):
        S_list.append(numpyro.sample(f"S_{b}", dist.Normal(0.0, 1.0)))
        C_list.append(numpyro.sample(f"C_{b}", dist.Normal(0.0, 1.0)))

    # -------------------------
    # Build observation vectors
    # -------------------------
    all_t = jnp.concatenate([b["t"] for b in bands])
    y = jnp.concatenate([b["y"] for b in bands])
    noise = jnp.concatenate([b["yerr"] for b in bands])

    N = len(all_t)

    # -------------------------
    # Base DRW covariance
    # -------------------------
    K_drw = drw_covariance(all_t, sigma, tau_drw)

    # -------------------------
    # Build response functions (NORMALISED)
    # -------------------------
    psi_list = []

    for b, band in enumerate(bands):
        psi = build_response_function(
            tau_grid,
            log_mdot,
            band["wavelength"],
            inclination,
            M_BH,
        )

        # 🔥 CRITICAL FIX: normalise response function
        psi = psi / (jnp.sum(psi) + 1e-8)

        psi_list.append(psi)

    # -------------------------
    # Construct covariance
    # -------------------------
    K = jnp.zeros((N, N))

    idx_i = 0

    for b1, band1 in enumerate(bands):
        n1 = len(band1["t"])
        idx_j = 0

        for b2, band2 in enumerate(bands):
            n2 = len(band2["t"])

            # interaction strength between bands
            kernel_scale = jnp.sqrt(jnp.sum(psi_list[b1]) * jnp.sum(psi_list[b2]))

            block = (
                kernel_scale
                * K_drw[
                    idx_i : idx_i + n1,
                    idx_j : idx_j + n2,
                ]
            )

            K = K.at[
                idx_i : idx_i + n1,
                idx_j : idx_j + n2,
            ].set(block)

            idx_j += n2

        idx_i += n1

    # -------------------------
    # Mean model (ONLY offsets)
    # -------------------------
    mean = jnp.concatenate(
        [C_list[i] * jnp.ones(len(b["t"])) for i, b in enumerate(bands)]
    )

    # -------------------------
    # Stabilisation
    # -------------------------
    K = K + jnp.diag(noise**2 + 1e-6)

    # -------------------------
    # GP likelihood
    # -------------------------
    L, lower = cho_factor(K, lower=True)
    alpha = cho_solve((L, lower), y - mean)

    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    loglike = -0.5 * (jnp.dot((y - mean), alpha) + logdet + N * jnp.log(2 * jnp.pi))

    numpyro.factor("gp_loglike", loglike)


# ----------------------------
# MCMC runner
# ----------------------------
def run_mcmc(model_fn, data, rng_key, num_warmup=500, num_samples=1000):
    kernel = NUTS(model_fn)

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=True,
    )

    mcmc.run(rng_key, data=data)
    return mcmc
