import numpy as np
import jax.numpy as jnp

a = np.random.rand(1, 10)
b = np.zeros((1, 10))
b[:, 5] = 1.0

c = np.vstack([a, b])

# Binning Operation
sums, edges = jnp.histogram(
    a,
    bins=5,
    weights=b,
)
counts, _ = jnp.histogram(
    a,
    bins=5,
)
y = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts!=0)
x = (edges[:-1] + edges[1:]) / 2
binned_data = jnp.vstack([x, y])

print(f"x values: {a}")
print(f"y values: {b}")
print(f"{binned_data}")
