import jax.numpy as jnp


def compute_posterior_beta(A, y, Sigma_diag, Q):
    """
    Stable posterior in PRECISION form

    Q = prior precision
    Sigma_diag = observation noise variances
    """

    # -----------------------------------------------------
    # Observation precision
    # -----------------------------------------------------
    Sigma_inv = jnp.diag(1.0 / Sigma_diag)

    # -----------------------------------------------------
    # Posterior precision
    # -----------------------------------------------------
    A_t_Sinv = A.T @ Sigma_inv

    Lambda = A_t_Sinv @ A + Q

    # -----------------------------------------------------
    # RHS
    # -----------------------------------------------------
    rhs = A_t_Sinv @ y

    # -----------------------------------------------------
    # Solve posterior mean
    # -----------------------------------------------------
    beta_mean = jnp.linalg.solve(Lambda, rhs)

    # -----------------------------------------------------
    # Posterior covariance (optional)
    # -----------------------------------------------------
    beta_cov = jnp.linalg.inv(Lambda)

    return beta_mean, beta_cov


def log_marginal_likelihood(ctx, A, Qinv):

    y = ctx.y_data
    Sigma = jnp.diag(ctx.sigma_data ** 2)

    beta_mean, beta_cov = compute_posterior_beta(A, y, Sigma, Qinv)

    # 🔥 STORE IN CONTEXT
    ctx.beta_mean = beta_mean
    ctx.beta_cov = beta_cov

    # marginal likelihood (unchanged conceptually)
    Sigma_inv = jnp.diag(1.0 / (ctx.sigma_data ** 2))
    Lambda = A.T @ Sigma_inv @ A + Q

    sign, logdet = jnp.linalg.slogdet(Lambda)

    rhs = A.T @ Sigma_inv @ y
    beta = jnp.linalg.solve(Lambda, rhs)

    loglike = -0.5 * (
        y.T @ Sigma_inv @ y
        - rhs.T @ jnp.linalg.solve(Lambda, rhs)
        + logdet
        + len(y) * jnp.log(2 * jnp.pi)
    )

    return loglike