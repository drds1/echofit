import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from jaxopt import ScipyMinimize

from .forward import forward_model


# =========================================================
# NUMPYRO MODEL
# =========================================================
def model(ctx):

    def _model():

        # -------------------------------
        # 1. Disk parameters
        # -------------------------------
        M_BH = numpyro.sample("M_BH", dist.LogUniform(5, 10))
        acc_rate = numpyro.sample("acc_rate", dist.LogUniform(0.01, 1.0))
        incl = numpyro.sample("incl", dist.Uniform(0, 90))

        params = (M_BH, acc_rate, incl)

        # -------------------------------
        # 2. Random walk prior strength
        # -------------------------------
        sigma_rw = numpyro.sample("sigma_rw", dist.LogNormal(0.0, 1.0))

        # -------------------------------
        # 3. Per-band calibration
        # -------------------------------
        n_bands = len(ctx.bands)

        C = numpyro.sample("C", dist.Normal(0.0, 10.0).expand([n_bands]))

        S = numpyro.sample("S", dist.LogNormal(0.0, 1.0).expand([n_bands]))

        # -------------------------------
        # 4. Forward model (NOW INCLUDES C/S)
        # -------------------------------
        y_model = forward_model(
            ctx.cache,
            ctx.X,
            ctx.t_model,
            ctx.interp_idx,
            ctx,
            params,
            sigma_rw,
            C,
            S,
        )

        # -------------------------------
        # 5. Likelihood
        # -------------------------------
        resid = (ctx.y_data - y_model) / ctx.sigma_data
        loglike = -0.5 * jnp.sum(resid**2)

        numpyro.factor("loglike", loglike)

    return _model


# =========================================================
# NUTS INFERENCE
# =========================================================
def run_inference(ctx, num_warmup=200, num_samples=1000):

    rng_key = jax.random.PRNGKey(0)

    kernel = NUTS(model(ctx))

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )

    mcmc.run(rng_key)

    return mcmc


def get_samples(mcmc):
    return mcmc.get_samples()


# =========================================================
# MAP (OPTIONAL)
# =========================================================
def make_logprob_fn(ctx):

    def logprob(params):

        M_BH, acc_rate, incl, sigma_rw = params

        # NOTE: MAP does NOT include C/S unless you explicitly optimise them
        # so we marginalise them out implicitly by fixing to 1 and 0

        C = jnp.zeros(len(ctx.bands))
        S = jnp.ones(len(ctx.bands))

        y_model = forward_model(
            ctx.cache,
            ctx.X,
            ctx.t_model,
            ctx.interp_idx,
            ctx,
            (M_BH, acc_rate, incl),
            sigma_rw,
            C,
            S,
        )

        # data concatenation
        y_list = []
        sigma_list = []

        for band in ctx.bands:
            y_list.append(ctx.flux_dict[band])
            sigma_list.append(ctx.sigma_dict[band])

        y_data = jnp.concatenate(y_list)
        sigma_data = jnp.concatenate(sigma_list)

        resid = (y_data - y_model) / sigma_data

        return -0.5 * jnp.sum(resid**2)

    return logprob


def run_map(ctx):

    logprob_fn = make_logprob_fn(ctx)

    init_params = jnp.array(
        [
            6.0,
            0.1,
            45.0,
            1.0,
        ]
    )

    solver = ScipyMinimize(
        method="L-BFGS-B",
        fun=lambda p: -logprob_fn(p),
    )

    return solver.run(init_params)
