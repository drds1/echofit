import jax.numpy as jnp
from jax import jit


def forward_core(
    A,
    y_data,
    sigma_data,
    sigma_rw,
    K,   # <-- stays here
):
    """
    Core linear solve + likelihood
    """

    # -------------------------
    # RANDOM WALK PRIOR
    # -------------------------
    D = jnp.eye(K) - jnp.eye(K, k=-1)
    D = D[1:]

    Q = (D.T @ D) / (sigma_rw ** 2)
    Q = Q + 1e-6 * jnp.eye(K)

    # -------------------------
    # WEIGHTED LEAST SQUARES
    # -------------------------
    Sigma_inv = jnp.diag(1.0 / (sigma_data ** 2))

    At_Sinv = A.T @ Sigma_inv
    precision = At_Sinv @ A + Q
    rhs = At_Sinv @ y_data

    beta_hat = jnp.linalg.solve(precision, rhs)

    y_model = A @ beta_hat

    return y_model


# =========================================================
# ✅ JIT WRAPPER (IMPORTANT FIX HERE)
# =========================================================

forward_core_jit = jit(
    forward_core,
    static_argnames=("K",)   # 🔥 KEY FIX
)