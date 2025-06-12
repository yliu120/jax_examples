from collections import namedtuple
from functools import partial
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp

from xlm2.jax.ops import allocate_buffer, annotate_memory_space
from xlm2.jax.transforms.utils import replace_pytree_leaves

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")
BodyFn = Callable[[Carry, X], tuple[Carry, Y]]


# The carry pytree has the following fields:
# - The original passed by user.
# - The index for slicing the residuals on host in current layer.
#   (Important: Please keep the current way of slicing because it is important performance-wise)
# - The host storage for residuals. (Additional field for pipelining)
# - The VJP pytree that we passes to the next iteration. (Additional field for pipelining)
WrappedCarry = namedtuple(
    "WrappedCarry", ["carry", "slice_idx", "residuals_on_host", "vjp_fn"]
)


def fwd_body(c: WrappedCarry, x: jax.tree_util.PyTreeDef, fun: BodyFn = None):
    assert fun is not None
    out, slice_idx, residuals_on_host, prev_vjp_fn = c

    def _memcpy_to_host(x, y):
        return annotate_memory_space(
            jax.lax.dynamic_update_slice_in_dim(
                x, y, slice_idx, axis=0, allow_negative_indices=False
            ),
            memory_space="host",
        )

    residuals_on_host = jax.tree.map(
        lambda x, y: _memcpy_to_host(x, jnp.expand_dims(y, axis=0)),
        residuals_on_host,
        prev_vjp_fn,
    )
    (out, y), vjp_fn = jax.vjp(fun, out, x)
    new_vjp_fn = replace_pytree_leaves(prev_vjp_fn, vjp_fn)
    return WrappedCarry(
        carry=out,
        slice_idx=slice_idx + 1,
        residuals_on_host=residuals_on_host,
        vjp_fn=new_vjp_fn,
    ), y


def bwd_body(c: WrappedCarry, x_cotangents: jax.tree_util.PyTreeDef):
    # The vjp_fn here is constructed by the previous layer from host.
    out_grad, slice_idx, residuals_on_host, vjp_fn = c
    x_cotangent = x_cotangents

    def _memcpy_from_host(x):
        return annotate_memory_space(
            jax.lax.dynamic_slice_in_dim(
                annotate_memory_space(x, memory_space="host"),
                slice_idx,
                1,
                allow_negative_indices=False,
            ),
            memory_space="device",
        )

    out_grad, y_cotangent = vjp_fn((out_grad, x_cotangent))
    next_vjp_fn = jax.tree.map(
        lambda x: jnp.squeeze(_memcpy_from_host(x), axis=0),
        residuals_on_host,
    )
    return WrappedCarry(
        carry=out_grad,
        slice_idx=slice_idx - 1,
        residuals_on_host=residuals_on_host,
        vjp_fn=next_vjp_fn,
    ), y_cotangent


@partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def _pipelined_scan(
    fun: BodyFn, init: Carry, xs: X | None = None, length: int | None = None
):
    assert length > 1, (
        "When length is 1, we fallback to non-pipelined scan so this should not happen."
    )
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
    final_fwd_carry, rest_ys = jax.lax.scan(
        partial(fwd_body, fun=fun),
        WrappedCarry(
            carry=out,
            slice_idx=0,
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
    c, _, residuals_on_host, last_vjp_fn = final_fwd_carry
    return (c, ys), (residuals_on_host, last_vjp_fn)


def _pipelined_scan_bwd(fun: BodyFn, length: int, res, g):
    del fun
    residuals_on_host, last_vjp_fn = res
    c_cotangent, ys_cotangent = g

    first_y_cot = jax.tree.map(lambda x: x[0], ys_cotangent)
    rest_y_cot = jax.tree.map(lambda x: x[1:], ys_cotangent)
    final_bwd_carry, rest_ys_grad = jax.lax.scan(
        bwd_body,
        WrappedCarry(
            carry=c_cotangent,
            slice_idx=length - 2,
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
        lambda x, y: jnp.concatenate([jnp.expand_dims(x, axis=0), y], axis=0),
        first_y_grad,
        rest_ys_grad,
    )
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
    if length is None:
        assert xs is not None
        flattened, _ = jax.tree.flatten(xs)
        length = flattened[0].shape[0]
    unroll = kwargs.get("unroll", 1)
    needs_unroll = (isinstance(unroll, int) and unroll > 1) or (
        isinstance(unroll, bool) and unroll
    )
    if length <= 1 or needs_unroll:
        # Fallback to non-pipelined scan if length is 1.
        return jax.lax.scan(fun, init, xs, length=length, unroll=unroll)
    return _pipelined_scan(fun, init, xs, length)
