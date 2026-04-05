# src/echofit/inference.py

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist

from .echo_cache import EchoCache
from .fourier_cache import build_fourier_matrices
from .model import evaluate_echo_model_matrix
from .config import frequencies


# =========================================================
# HELPER: build random walk precision matrix
# =========================================================
def build_rw_precision(K, sigma):

    # first-order difference operator
    D = jnp.eye(K) - jnp.eye(K, k=-1)
    D = D[1:]  # remove first row

    # precision = (D^T D) / sigma^2
    Q = (D.T @ D) / (sigma ** 2)

    # small regularisation
    Q += 1e-6 * jnp.eye(K)

    return Q


# =========================================================
# MODEL
# =========================================================
def model(time_dict, flux_dict, sigma_dict, wavelengths):

    # ---------------------------------------------------------
    # STATIC PRECOMPUTE (ONCE)
    # ---------------------------------------------------------
    t_min = min([t.min() for t in time_dict.values()])
    t_max = max([t.max() for t in time_dict.values()])
    t_model = jnp.linspace(t_min, t_max, 2000)

    # Fourier design
    X_sin, X_cos = build_fourier_matrices(t_model, frequencies)
    X = jnp.concatenate([X_sin, X_cos], axis=1)  # (T, 2K)

    # Echo cache
    cache = EchoCache(t_model, wavelengths)

    # Pre-stack data
    y_list = []
    sigma_list = []
    interp_indices = []

    for band in flux_dict:
        if band == "xray":
            continue

        y_list.append(flux_dict[band])
        sigma_list.append(sigma_dict[band])

        interp_indices.append(time_dict[band])

    y_data = jnp.concatenate(y_list)
    sigma_data = jnp.concatenate(sigma_list)

    # ---------------------------------------------------------
    # NUMPYRO MODEL
    # ---------------------------------------------------------
    def _model():

        # -------------------------------
        # 1. DISK PARAMETERS
        # -------------------------------
        M_BH = numpyro.sample("M_BH", dist.LogUniform(5, 10))
        acc_rate = numpyro.sample("acc_rate", dist.LogUniform(0.01, 1.0))
        incl = numpyro.sample("incl", dist.Uniform(0, 90))

        params = (M_BH, acc_rate, incl)

        # -------------------------------
        # 2. RANDOM WALK PRIOR STRENGTH
        # -------------------------------
        sigma_rw = numpyro.sample("sigma_rw", dist.LogNormal(0.0, 1.0))

        K = X.shape[1]

        Q = build_rw_precision(K, sigma_rw)  # prior precision

        # -----------------------------------------------------
        # 3. BUILD DESIGN MATRIX A(θ)
        # -----------------------------------------------------

        # 🔥 VECTORISED ECHO PROPAGATION
        model_dict = evaluate_echo_model_matrix(
            cache,
            X,
            params
        )
        
        # 🔥 BUILD A MATRIX WITHOUT LOOPING OVER K
        A_blocks = []
        
        for band in flux_dict:
            if band == "xray":
                continue
            
            Y = model_dict[band]  # (T, K)
        
            interp = jnp.vstack([
                jnp.interp(time_dict[band], t_model, Y[:, i])
                for i in range(Y.shape[1])
            ]).T  # (N_band, K)
        
            A_blocks.append(interp)
        
        A = jnp.concatenate(A_blocks, axis=0)

        # -----------------------------------------------------
        # 4. LINEAR SOLVE (RIDGE WITH PRIOR)
        # -----------------------------------------------------

        Sigma_inv = jnp.diag(1.0 / (sigma_data ** 2))

        At_Sinv = A.T @ Sigma_inv

        precision = At_Sinv @ A + Q

        rhs = At_Sinv @ y_data

        beta_hat = jnp.linalg.solve(precision, rhs)

        # -----------------------------------------------------
        # 5. LOG LIKELIHOOD (CONDITIONAL)
        # -----------------------------------------------------

        y_model = A @ beta_hat

        resid = (y_data - y_model) / sigma_data

        loglike = -0.5 * jnp.sum(resid ** 2)

        numpyro.factor("loglike", loglike)

    return _model


# =========================================================
# INFERENCE
# =========================================================
def run_inference(
    time_dict,
    flux_dict,
    sigma_dict,
    wavelengths,
    num_warmup=500,
    num_samples=1000
):

    rng_key = jax.random.PRNGKey(0)

    kernel = NUTS(model(time_dict, flux_dict, sigma_dict, wavelengths))

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

    mcmc.run(rng_key)

    return mcmc


def get_samples(mcmc):
    return mcmc.get_samples()