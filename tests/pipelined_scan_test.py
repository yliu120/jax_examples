from functools import partial

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from jax_examples.magics.transforms.pipelined_scan import pipelined_scan


class PipelinedScanTest(parameterized.TestCase):
    @parameterized.named_parameters(
        [
            ("no_checkpoint", lambda x: x),
            ("checkpoint", jax.checkpoint),
        ]
    )
    def test_pipelined_scan_with_no_scan_args(self, checkpoint_fn):
        to_scan = checkpoint_fn(lambda c, _: (jnp.sin(c), None))
        carry_inp = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)

        def loss_fn(x, scan_impl=None):
            return jnp.sum(scan_impl(to_scan, x, length=3)[0])

        value_and_grad_fn = jax.value_and_grad(
            partial(loss_fn, scan_impl=pipelined_scan)
        )
        value, grad = jax.jit(value_and_grad_fn)(carry_inp)
        value_and_grad_fn_2 = jax.value_and_grad(
            partial(loss_fn, scan_impl=jax.lax.scan)
        )
        expected_value, expected_grad = jax.jit(value_and_grad_fn_2)(carry_inp)

        np.testing.assert_allclose(value, expected_value)
        np.testing.assert_allclose(grad, expected_grad)

    @parameterized.named_parameters(
        [
            ("no_checkpoint", lambda x: x),
            ("checkpoint", jax.checkpoint),
        ]
    )
    def test_pipelined_scan_with_scan_args(self, checkpoint_fn):
        to_scan = checkpoint_fn(lambda c, x: (jnp.sin(c + x), x**2))
        carry_inp = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
        scan_inp = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)

        def loss_fn(init, xs, scan_impl=None):
            return jnp.sum(scan_impl(to_scan, init, xs)[0])

        value_and_grad_fn = jax.value_and_grad(
            partial(loss_fn, scan_impl=pipelined_scan)
        )
        value, grad = jax.jit(value_and_grad_fn)(carry_inp, scan_inp)
        value_and_grad_fn_2 = jax.value_and_grad(
            partial(loss_fn, scan_impl=jax.lax.scan)
        )
        expected_value, expected_grad = jax.jit(value_and_grad_fn_2)(
            carry_inp, scan_inp
        )

        np.testing.assert_allclose(value, expected_value)
        np.testing.assert_allclose(grad, expected_grad)

    @parameterized.named_parameters(
        [
            ("no_checkpoint", lambda x: x),
            ("checkpoint", jax.checkpoint),
        ]
    )
    def test_multiple_scans(self, checkpoint_fn):
        to_scan = checkpoint_fn(lambda c, x: (jnp.sin(c + x), x**2))
        carry_inp = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
        scan_inp = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)

        def loss_fn(init, xs, scan_impl=None):
            c, ys = scan_impl(to_scan, init, xs)
            return jnp.sum(scan_impl(to_scan, c, ys)[0])

        value_and_grad_fn = jax.value_and_grad(
            partial(loss_fn, scan_impl=pipelined_scan)
        )
        value, grad = jax.jit(value_and_grad_fn)(carry_inp, scan_inp)
        value_and_grad_fn_2 = jax.value_and_grad(
            partial(loss_fn, scan_impl=jax.lax.scan)
        )
        expected_value, expected_grad = jax.jit(value_and_grad_fn_2)(
            carry_inp, scan_inp
        )

        np.testing.assert_allclose(value, expected_value)
        np.testing.assert_allclose(grad, expected_grad)


if __name__ == "__main__":
    absltest.main()
