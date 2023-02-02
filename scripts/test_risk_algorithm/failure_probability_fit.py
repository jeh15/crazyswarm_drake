from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import (
    jit,
    jacfwd,
    jacrev,
)


@partial(jit, static_argnames=['spline_resolution'])
def _objective_function(
    y: jax.Array,
    x: jax.Array,
    y_data: jax.Array,
    x_data: jax.Array,
    spline_resolution: int,
) -> jnp.ndarray:
    # Format the data:
    x_data = jnp.reshape(x_data, (spline_resolution, -1))
    y_data = jnp.reshape(y_data, (spline_resolution, -1))

    def linear_interpolation(x, y, x_data, y_data):
        output = []
        for i in range(0, x.shape[0]-1):
            output.append(y_data[i, :] - (y[i] + (x_data[i, :] - x[i]) * (y[i+1] - y[i]) / (x[i+1] - x[i])))
        return jnp.array(output)

    minimize_error = linear_interpolation(x, y, x_data, y_data).flatten()
    objective_function = jnp.sum(minimize_error ** 2, axis=0)

    return objective_function


# How do you type hint a return is a callable?
def jit_functions(num_spline: int) -> Any:
    objective_func = lambda y, x, yd, xd: _objective_function(
        y=y,
        x=x,
        y_data=yd,
        x_data=xd,
        spline_resolution=num_spline,
    )

    # Compute H and f matrcies for objective function:
    jit_H = jax.jit(jacfwd(jacrev(objective_func)))
    jit_f = jax.jit(jacfwd(objective_func))

    return jit_H, jit_f
