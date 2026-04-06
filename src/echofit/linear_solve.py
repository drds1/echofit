import jax.numpy as jnp


def solve_beta(A, y, sigma, Q_prior):
    """
    Shared stable Bayesian linear solve
    """

    Sigma_inv = jnp.diag(1.0 / (sigma ** 2))

    At_S = A.T @ Sigma_inv
    lhs = At_S @ A + Q_prior
    rhs = At_S @ y

    beta = jnp.linalg.solve(lhs, rhs)

    return beta