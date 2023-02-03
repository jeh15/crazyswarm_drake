from typing import Any

import jax
import jax.numpy as jnp
from jax import (
    jit,
    jacfwd,
    jacrev,
)


@jit
def _inequality_constraints(
    y: jax.Array,
    x: jax.Array,
) -> jnp.ndarray:
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    inequality_constraints = m[1:] - m[:-1]
    return inequality_constraints


@jit
def _objective_function(
    y: jax.Array,
    y_data: jax.Array,
    w: jax.Array,
) -> jnp.ndarray:
    minimize_error = w * (y_data - y) ** 2
    objective_function = jnp.sum(minimize_error, axis=0)

    return objective_function


# How do you type hint a return is a callable?
def jit_functions() -> Any:
    # Isolate Functions with Lambda Expressions
    inequality_func = lambda y, x: _inequality_constraints(
        y=y,
        x=x,
    )
    objective_func = lambda y, yd, w: _objective_function(
        y=y,
        y_data=yd,
        w=w,
    )

    # Compute A and b matrices for inequality constraints:
    jit_A = jax.jit(jacfwd(inequality_func))
    jit_b = inequality_func

    # Compute H and f matrices for objective function:
    jit_H = jax.jit(jacfwd(jacrev(objective_func)))
    jit_f = jax.jit(jacfwd(objective_func))

    # Return function handles
    return jit_A, jit_b, jit_H, jit_f
