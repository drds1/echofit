import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from .model import evaluate_echo_model_matrix
import jax

def model(
    X,
    y_data,
    sigma_data,
    cache,
    interp_idx,
    bands,
    band_sizes,
    t_model,
):

    # ---------------------------------------
    # PARAMETERS
    # ---------------------------------------
    M_BH = numpyro.sample("M_BH", dist.LogNormal(1.0, 0.5))
    acc_rate = numpyro.sample("acc_rate", dist.LogNormal(-1.0, 0.5))
    incl = numpyro.sample("incl", dist.Uniform(0, 90))

    sigma_rw = numpyro.sample("sigma_rw", dist.LogNormal(0.0, 0.5))

    n_bands = len(bands)

    C = numpyro.sample("C", dist.Normal(0.0, 1.0).expand([n_bands]))
    logS = numpyro.sample("logS", dist.Normal(0.0, 0.5).expand([n_bands]))
    S = jnp.exp(logS)

    params = (M_BH, acc_rate, incl)

    # ---------------------------------------
    # DESIGN MATRIX
    # ---------------------------------------
    model_dict = evaluate_echo_model_matrix(cache, X, params)

    A_blocks = []
    for b in bands:
        A_b = model_dict[b][interp_idx[b], :]
        A_blocks.append(A_b)

    A = jnp.concatenate(A_blocks, axis=0)

    # ---------------------------------------
    # LATENT β (THIS IS THE KEY FIX)
    # ---------------------------------------

    K = A.shape[1]

    D = jnp.eye(K) - jnp.eye(K, k=-1)
    D = D[1:]

    Q = (D.T @ D) / (sigma_rw ** 2) + 1e-6 * jnp.eye(K)

    beta = numpyro.sample(
        "beta",
        dist.MultivariateNormal(
            loc=jnp.zeros(K),
            covariance_matrix=jnp.linalg.inv(Q)
        )
    )

    # ---------------------------------------
    # FORWARD MODEL
    # ---------------------------------------
    y_base = A @ beta

    y_out = []
    offset = 0

    for i, b in enumerate(bands):
        n = band_sizes[b]
        y_b = y_base[offset:offset + n]

        y_b = C[i] + S[i] * y_b

        y_out.append(y_b)
        offset += n

    y_pred = jnp.concatenate(y_out)

    # ---------------------------------------
    # LIKELIHOOD
    # ---------------------------------------
    numpyro.sample(
        "obs",
        dist.Normal(y_pred, sigma_data),
        obs=y_data
    )



def run_inference(
    X,
    y_data,
    sigma_data,
    cache,
    interp_idx,
    bands,
    band_sizes,
    t_model,
    num_warmup,
    num_samples,
):
    

    kernel = NUTS(model)

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )

    rng_key = jax.random.PRNGKey(0)


    mcmc.run(
        rng_key,
        X,
        y_data,
        sigma_data,
        cache,
        interp_idx,
        bands,
        band_sizes,
        t_model,
    )

    return mcmc