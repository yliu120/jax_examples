from collections import namedtuple
from functools import partial
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp

from jax_examples.magics.ops import allocate_buffer, annotate_memory_space
from jax_examples.magics.transforms.utils import replace_pytree_leaves

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")
BodyFn = Callable[[Carry, X], tuple[Carry, Y]]


# The carry pytree has the following fields:
# - The original passed by user.
# - The index of the current layer. (Additional field for pipelining)
# - The host storage for residuals. (Additional field for pipelining)
# - The VJP pytree that we passes to the next iteration. (Additional field for pipelining)
WrappedCarry = namedtuple(
    "WrappedCarry", ["carry", "layer_idx", "residuals_on_host", "vjp_fn"]
)


def fwd_body(c: WrappedCarry, x: jax.tree_util.PyTreeDef, fun: BodyFn = None):
    assert fun is not None
    out, layer_idx, residuals_on_host, prev_vjp_fn = c
    residuals_on_host = jax.tree.map(
        lambda x, y: annotate_memory_space(
            jax.lax.dynamic_update_slice_in_dim(
                x, jnp.expand_dims(y, axis=0), layer_idx - 1, axis=0
            ),
            memory_space="host",
        ),
        residuals_on_host,
        prev_vjp_fn,
    )
    (out, y), vjp_fn = jax.vjp(fun, out, x)
    new_vjp_fn = replace_pytree_leaves(prev_vjp_fn, vjp_fn)
    return WrappedCarry(
        carry=out,
        layer_idx=layer_idx + 1,
        residuals_on_host=residuals_on_host,
        vjp_fn=new_vjp_fn,
    ), y


def bwd_body(c: WrappedCarry, x_cotangent: jax.tree_util.PyTreeDef):
    # The vjp_fn here is constructed by the previous layer from host.
    out_grad, layer_idx, residuals_on_host, vjp_fn = c
    out_grad, y_cotangent = vjp_fn((out_grad, x_cotangent))
    next_vjp_fn = jax.tree.map(
        lambda x: jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, layer_idx - 1, 1)),
        residuals_on_host,
    )
    return WrappedCarry(
        carry=out_grad,
        layer_idx=layer_idx - 1,
        residuals_on_host=residuals_on_host,
        vjp_fn=next_vjp_fn,
    ), y_cotangent


@partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def _pipelined_scan(
    fun: BodyFn, init: Carry, xs: X | None = None, length: int | None = None
):
    return jax.lax.scan(fun, init, xs, length=length)


def _pipelined_scan_fwd(
    fun: BodyFn, init: Carry, xs: X | None = None, length: int | None = None
):
    first_x = jax.tree.map(lambda x: x[0], xs)
    rest_x = jax.tree.map(lambda x: x[1:], xs)
    # Peels the first layer out.
    (out, first_y), first_vjp_fn = jax.vjp(fun, init, first_x)
    residuals_on_host = jax.tree.map(
        lambda x: annotate_memory_space(
            allocate_buffer((length - 1, *x.shape), x.dtype), memory_space="host"
        ),
        first_vjp_fn,
    )
    layer_idx = 0
    final_fwd_carry, rest_ys = jax.lax.scan(
        partial(fwd_body, fun=fun),
        WrappedCarry(
            carry=out,
            layer_idx=layer_idx + 1,
            residuals_on_host=residuals_on_host,
            vjp_fn=first_vjp_fn,
        ),
        rest_x,
        length=length - 1 if length is not None else None,
    )
    ys = (
        jax.tree.map(
            lambda x, y: jnp.concatenate([jnp.expand_dims(x, axis=0), y], axis=0),
            first_y,
            rest_ys,
        )
        if first_y is not None
        else None
    )
    c, final_layer_idx, residuals_on_host, last_vjp_fn = final_fwd_carry
    return (c, ys), (final_layer_idx, residuals_on_host, last_vjp_fn)


def _pipelined_scan_bwd(fun: BodyFn, length: int, res, g):
    del fun
    final_layer_idx, residuals_on_host, last_vjp_fn = res
    c_cotangent, ys_cotangent = g

    first_y_cot = jax.tree.map(lambda x: x[0], ys_cotangent)
    rest_y_cot = jax.tree.map(lambda x: x[1:], ys_cotangent)
    final_bwd_carry, rest_ys_grad = jax.lax.scan(
        bwd_body,
        WrappedCarry(
            carry=c_cotangent,
            layer_idx=final_layer_idx - 1,
            residuals_on_host=residuals_on_host,
            vjp_fn=last_vjp_fn,
        ),
        rest_y_cot,
        length=length - 1 if length is not None else None,
        reverse=True,
    )
    c_cotangent, _, _, first_vjp_fn = final_bwd_carry

    # First layer backward.
    c_cotangent, first_y_grad = first_vjp_fn((c_cotangent, first_y_cot))

    ys_grad = jax.tree.map(
        lambda x, y: jnp.concatenate([x[None, :], y[::-1]], axis=0),
        first_y_grad,
        rest_ys_grad,
    )
    # return tuple(filter(lambda x: x is not None, (c_cotangent, ys_grad)))
    return c_cotangent, ys_grad


_pipelined_scan.defvjp(_pipelined_scan_fwd, _pipelined_scan_bwd)


def pipelined_scan(
    fun: BodyFn,
    init: Carry,
    xs: X | None = None,
    length: int | None = None,
    **kwargs,
) -> tuple[Carry, Y]:
    """Pipelines activation offloading for scan.

    This is a drop-in replacement for jax.lax.scan when no unrolling is desired.
    Currently, this is only used for pipelining things manually in scan.

    Args:
      **kwargs: Keyword arguments passed to jax.lax.scan.
    """
    reverse = kwargs.get("reverse", False)
    if reverse:
        raise NotImplementedError("Reverse pipelined scan is not implemented.")
    unroll = kwargs.get("unroll", None)
    if unroll is not None:
        raise NotImplementedError("Unrolled pipelined scan is not implemented.")
    if length is None:
        assert xs is not None
        flattened, _ = jax.tree.flatten(xs)
        length = flattened[0].shape[0]
    return _pipelined_scan(fun, init, xs, length)
