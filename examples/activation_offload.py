import dataclasses
from typing import Callable

from enum import Enum

import jax
import jax.nn.initializers
import jax.numpy as jnp

from jax._src.lib.mlir.dialects import hlo
from jax.extend import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir


# Annotates a tensor to a specific memory space.
# This version is modified from internal code for illustration.
annotate_memory_space_p = core.Primitive("annotate_memory_space")
annotate_memory_space_p.def_impl(lambda x, memory_space=None: x)
annotate_memory_space_p.def_abstract_eval(lambda x, memory_space=None: x)


def _annotate_memory_space_lowering(ctx, x, memory_space: str = None):
    if not memory_space:
        return x

    backend_config = ir.DictAttr.get(
        {
            "memory_space": ir.StringAttr.get(memory_space),
        }
    )
    out = hlo.CustomCallOp(
        result=[x.type],
        inputs=[x],
        call_target_name=ir.StringAttr.get("AnnotateMemorySpace"),
        backend_config=backend_config,
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 4),
    )
    return out.results


mlir.register_lowering(annotate_memory_space_p, _annotate_memory_space_lowering)


def annotate_memory_space(x, memory_space: str = None):
    return annotate_memory_space_p.bind(x, memory_space=memory_space)


def _replace_residuals_in_vjp_fn(
    vjp_fn: jax.tree_util.Partial, new_vjp_fn: jax.tree_util.Partial
):
    """Constructs a vjp_fn with the residuals."""
    _, treedef = jax.tree.flatten(vjp_fn)
    return jax.tree.unflatten(treedef, jax.tree.leaves(new_vjp_fn))


# Defines some simple abstractions for holding parameters in two different phases:
#  1. During the initialization phase, we register parameters here with metadata.
#  2. During the run phase, parameters are instantiated with values.
@dataclasses.dataclass
class ParameterMetadata:
    shape: tuple[int]
    dtype: jnp.dtype
    initializer: jax.nn.initializers.Initializer


AbstractOrInstantiatedParam = ParameterMetadata | jax.Array
Parameters = dict[str, AbstractOrInstantiatedParam]


class Phase(Enum):
    """Phases of the layer context."""

    INIT = 1
    RUN = 2


class ParameterHub:
    # A registry of parameters for all layers.
    params: Parameters
    initialized_params: dict[str, jax.Array]
    context_state: Phase

    def __init__(self, params: Parameters = None):
        self.params = params or {}
        self.context_state = Phase.INIT

    def register_param(
        self,
        name: str,
        shape: tuple[int],
        dtype: jnp.dtype,
        initializer: jax.nn.initializers.Initializer,
    ):
        if name in self.params:
            raise ValueError(f"Parameter {name} already registered.")
        self.params[name] = ParameterMetadata(shape, dtype, initializer)

    def initialize(self, key: jax.Array) -> Parameters:
        """Initializes the parameters."""
        return jax.tree.map(lambda p: p.initializer(key, p.shape, p.dtype), self.params)

    def get_param(self, name: str) -> AbstractOrInstantiatedParam:
        return self.params[name]

    def set_phase(self, phase: Phase):
        self.context_state = phase

    def is_init(self) -> bool:
        return self.context_state == Phase.INIT


class SineLayer:
    """Uses a sine layer to discern residuals."""

    def __init__(self, ctx: ParameterHub):
        self.ctx = ctx

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.ctx.is_init():
            return x
        with jax.named_scope("sine"):
            return jnp.sin(x)


class LinearLayer:
    def __init__(self, ctx: ParameterHub, name: str, n_features: int):
        self.ctx = ctx
        self.name = name
        self.n_features = n_features

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.ctx.is_init():
            self.ctx.register_param(
                f"{self.name}/w",
                (x.shape[-1], self.n_features),
                jnp.float32,
                jax.nn.initializers.normal(stddev=0.01),
            )
            self.ctx.register_param(
                f"{self.name}/b",
                (self.n_features,),
                jnp.float32,
                jax.nn.initializers.zeros,
            )
            return x

        w = self.ctx.get_param(f"{self.name}/w")
        b = self.ctx.get_param(f"{self.name}/b")
        with jax.named_scope(f"{self.name}/project"):
            x = x @ w
        with jax.named_scope(f"{self.name}/add_bias"):
            # Implicit broadcasting.
            x = x + b
        with jax.named_scope(f"{self.name}/relu"):
            x = jax.nn.relu(x)
        return x


# We can simply use free function to define combinations of layers.
def single_layer(
    ctx: ParameterHub, x: jax.Array, n_features: int = 1024
) -> tuple[ParameterHub, jax.Array]:
    x = LinearLayer(ctx, "linear_1", n_features)(x)
    x = SineLayer(ctx)(x)
    return ctx, x


def _stacked_and_pipelined(
    f: Callable[[ParameterHub, jax.Array], tuple[ParameterHub, jax.Array]],
    loss_fn: Callable,
    x: jax.Array,
    num_layers: int = 10,
):
    """Transforms f to a stacked and pipelined function in the run phase."""

    def _per_layer_fwd_fn(params: Parameters, x: jax.Array):
        ctx = ParameterHub(params=params)
        ctx.set_phase(Phase.RUN)
        _, x = f(ctx, x)
        return x

    def _fwd_bwd_fn(ctx: ParameterHub, x: jax.Array):
        # Forward pass: pipeline backward for fwd pass.
        def _fwd_body(carry, params):
            activation, layer_idx, residuals_on_host, prev_vjp_fn = carry
            # Offloads residuals produced by previous layer to host.
            residuals_on_host = jax.tree.map(
                lambda x, y: annotate_memory_space(
                    jax.lax.dynamic_update_slice_in_dim(
                        x, y[None, :], layer_idx, axis=0
                    ),
                    memory_space="host",
                ),
                residuals_on_host,
                prev_vjp_fn,
            )
            activation, vjp_fn = jax.vjp(_per_layer_fwd_fn, params, activation)
            new_vjp_fn = _replace_residuals_in_vjp_fn(prev_vjp_fn, vjp_fn)
            return (activation, layer_idx + 1, residuals_on_host, new_vjp_fn), None

        # Pipeline backward for fwd pass.
        # Peels the first layer out.
        first_layer_params = jax.tree.map(lambda x: x[0], ctx.params)
        rest_layer_params = jax.tree.map(lambda x: x[1:], ctx.params)
        x, first_layer_vjp_fn = jax.vjp(_per_layer_fwd_fn, first_layer_params, x)

        # Allocate host buffer for residuals.
        residuals_on_host = jax.tree.map(
            lambda x: annotate_memory_space(
                jnp.zeros((num_layers - 1, *x.shape), x.dtype), memory_space="host"
            ),
            first_layer_vjp_fn,
        )
        layer_idx = 0
        final_fwd_carry, _ = jax.lax.scan(
            _fwd_body,
            (x, layer_idx, residuals_on_host, first_layer_vjp_fn),
            rest_layer_params,
            length=num_layers - 1,
        )
        final_activation, last_layer_idx, residuals_on_host, last_vjp_fn = (
            final_fwd_carry
        )

        # Backward pass: pipeline forward for bwd pass.
        def _bwd_body(carry, _):
            activation_grad, layer_idx, residuals_on_host, vjp_fn = carry
            layer_params_grad, activation_grad = vjp_fn(activation_grad)
            next_vjp_fn = jax.tree.map(
                lambda x: jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, layer_idx, 1)),
                residuals_on_host,
            )
            return (
                activation_grad,
                layer_idx - 1,
                residuals_on_host,
                next_vjp_fn,
            ), layer_params_grad

        init_cotangent = jax.grad(loss_fn)(final_activation)
        final_bwd_carry, rest_params_grad = jax.lax.scan(
            _bwd_body,
            (init_cotangent, last_layer_idx, residuals_on_host, last_vjp_fn),
            length=num_layers - 1,
            reverse=True,
        )
        activation_grad, _, _, first_vjp_fn = final_bwd_carry

        # Peels the first layer out.
        first_layer_params_grad, _ = first_vjp_fn(activation_grad)
        params_grad = jax.tree.map(
            lambda x, y: jnp.concatenate([x[None, :], y[::-1]], axis=0),
            first_layer_params_grad,
            rest_params_grad,
        )
        return loss_fn(final_activation), params_grad

    return _fwd_bwd_fn


def stacked_and_pipelined(
    f: Callable[[ParameterHub, jax.Array], tuple[ParameterHub, jax.Array]],
    loss_fn: Callable,
    x: jax.Array,
    num_layers: int = 10,
):
    """Stacks, pipelines, and autodiffs the per-layer forward function f.

    Roughly equivalent to:
    f -> jax.value_and_grad(jax.lax.scan(f, x, length=num_layers)), with
    pipelined activation offloading.
    """
    # Transforms layer context.
    body_ctx = ParameterHub()
    body_ctx.set_phase(Phase.INIT)
    body_ctx, _ = f(body_ctx, x)

    def _stack_param(p):
        return ParameterMetadata(
            shape=(num_layers, *p.shape),
            dtype=p.dtype,
            initializer=p.initializer,
        )

    stacked_params = jax.tree.map(_stack_param, body_ctx.params)

    # Constructs the final context.
    ctx = ParameterHub(params=stacked_params)
    return ctx, _stacked_and_pipelined(f, loss_fn, x, num_layers)


def main():
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (128, 1024), dtype=jnp.float32)

    # Define a simple L2 loss function.
    def loss_fn(x: jax.Array) -> jax.Array:
        return jnp.mean(x**2)

    # Transform the model function.
    ctx, _fwd_bwd_fn = stacked_and_pipelined(single_layer, loss_fn, data)

    # Phase 1: Initialize the parameters.
    def init_fn(key: jax.Array) -> Parameters:
        return ctx.initialize(key)

    # Phase 2: Run the model.
    def fwd_bwd_fn(params: Parameters, x: jax.Array):
        ctx = ParameterHub(params=params)
        ctx.set_phase(Phase.RUN)
        return _fwd_bwd_fn(ctx, x)

    params_shape = jax.eval_shape(init_fn, key)
    jitted_fwd_bwd_fn = jax.jit(fwd_bwd_fn)
    # Prints lowered jit function.
    print(jitted_fwd_bwd_fn.lower(params_shape, data).as_text())


if __name__ == "__main__":
    main()
