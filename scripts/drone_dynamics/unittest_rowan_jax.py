from absl.testing import absltest
from functools import partial

import numpy as np
import numpy.typing
import jax
import jax.numpy as jnp

import rowan
from rowan.functions import _promote_vec as rowan_promote_vec


@jax.jit
def multiply(qi: jax.typing.ArrayLike, qj: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Multiply two quaternions together where
        qi is on the left and qj is on the right.
    """

    qi = jnp.asarray(qi)
    _qi = jnp.expand_dims(qi[..., 0], axis=-1)
    qj = jnp.asarray(qj)
    _qj = jnp.expand_dims(qj[..., 0], axis=-1)

    a = (
        qi[..., 0] * qj[..., 0] - jnp.sum(
            qi[..., 1:] * qj[..., 1:], axis=-1
        )
    )

    b = (
        _qi * qj[..., 1:]
        + _qj * qi[..., 1:]
        + jnp.cross(qi[..., 1:], qj[..., 1:])
    )

    q = jnp.concatenate(
        (jnp.expand_dims(a, axis=-1), b),
        axis=-1,
    )

    return q


@jax.jit
def exp(q: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Compute the exponential of quaternion q.
    """

    q = jnp.asarray(q)

    # Create output array:
    q_exp = jnp.zeros_like(q)
    norms = jnp.linalg.norm(
        q[..., 1:],
        axis=-1,
    )
    _norms = jnp.expand_dims(norms, axis=-1)
    e = jnp.exp(q[..., 0])
    _e = jnp.expand_dims(e, axis=-1)

    """
        Break calculation into subcomponents:
            exp(q) = exp(a) * (cos(||q||) + (q / ||q||) * sin(||q||))
            a = exp(a) * cos(||q||)
            b = exp(a) * (q / ||q||) * sin(||q||)
    """
    a = e * jnp.cos(norms)
    b = (
        _e
        * jnp.where(_norms == 0, 0, q[..., 1:] / _norms)
        * jnp.sin(_norms)
    )
    q_exp = q_exp.at[..., 0].set(a)
    q_exp = q_exp.at[..., 1:].set(b)

    return q_exp


@partial(jax.jit, static_argnames=['dt'])
def integrate(q: jax.typing.ArrayLike, v: jax.typing.ArrayLike, dt: float) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/calculus/__init__.py
    q = jnp.asarray(q)
    v = jnp.asarray(v)
    dt = jnp.asarray(dt)
    return multiply(exp(_jax_promote_vector(0.5 * v * dt)), q)


@jax.jit
def conjugate(q: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Compute the conjugate of quaternion q.
    """
    q = jnp.asarray(q)
    q = (
        q.at[..., 1:]
        .set(-1 * q[..., 1:])
    )
    return q


@jax.jit
def rotate(q: jax.typing.ArrayLike, v: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Rotate vectors v by quaternions q.
    """
    q = jnp.asarray(q)
    v = jnp.asarray(v)
    _v = _jax_promote_vector(v)
    return multiply(q, multiply(_v, conjugate(q)))[..., 1:]


@jax.jit
def jax_normalize(q: jax.typing.ArrayLike) -> jax.Array:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Normalize quaternions q such that the first element is the identity element.
    """
    q = jnp.asarray(q)
    norms = jnp.expand_dims(
        jnp.linalg.norm(q, axis=-1),
        axis=-1,
    )
    return q / norms


def numpy_normalize(q: numpy.typing.ArrayLike) -> np.ndarray:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Normalize quaternions q such that the first element is the identity element.
    """
    q = np.asarray(q)
    norms = np.linalg.norm(q, axis=-1)[..., np.newaxis]
    return q / norms


@jax.jit
def _jax_promote_vector(q: numpy.typing.ArrayLike) -> np.ndarray:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Converts 3 element vectors to 4 element vectors
        for compatability with quaternion math operations
    """
    return jnp.concatenate((jnp.zeros(q.shape[:-1] + (1,)), q), axis=-1)


def _numpy_promote_vector(q: numpy.typing.ArrayLike) -> np.ndarray:
    # Implementation from: https://github.com/glotzerlab/rowan/blob/master/rowan/functions.py
    """
        Converts 3 element vectors to 4 element vectors
        for compatability with quaternion math operations
    """
    return np.concatenate((np.zeros(q.shape[:-1] + (1,)), q), axis=-1)


class TestQuaternionMath(absltest.TestCase):
    def test_exp(self):
        # Random Matrix of Quaternions:
        v = np.random.rand(10, 4)

        # Exponential Quaternion Function:
        result_1 = exp(v)
        result_2 = exp(v[0, :])

        # Rowan Result:
        true_value = rowan.exp(v)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result_1,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            true_value[0, :], result_2,
            atol=1e-6,
        )

    def test_multiply(self):
        qi = np.random.rand(10, 4)
        qj = np.random.rand(10, 4)
        result = multiply(qi, qj)

        # Rowan Result:
        true_value = rowan.multiply(qi, qj)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_promote_vector(self):
        v = np.random.rand(10, 3)
        result = _jax_promote_vector(v)

        # Rowan Result:
        true_value = rowan_promote_vec(v)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_integrate(self):
        q = np.random.rand(10, 4)
        q = q / np.linalg.norm(q, axis=-1)[..., np.newaxis]
        v = np.random.rand(10, 3)
        dt = 0.1
        result = integrate(q, v, dt)

        # Rowan Result:
        true_value = rowan.calculus.integrate(q, v, dt)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_normalize(self):
        q = np.random.rand(10, 4)
        result_1 = jax_normalize(q)
        result_2 = numpy_normalize(q)

        # Rowan Result:
        true_value = rowan.normalize(q)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result_1,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            true_value, result_2,
            atol=1e-6,
        )

    def test_conjugate(self):
        q = np.random.rand(10, 4)
        result = conjugate(q)

        # Rowan Result:
        true_value = rowan.conjugate(q)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )

    def test_rotation(self):
        q = np.random.rand(10, 4)
        v = np.random.rand(10, 3)
        result = rotate(q, v)

        # Rowan Result:
        true_value = rowan. rotate(q, v)

        # Assertion Test:
        np.testing.assert_allclose(
            true_value, result,
            atol=1e-6,
        )


if __name__ == "__main__":
    absltest.main()
