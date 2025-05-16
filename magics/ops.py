import functools
from typing import Sequence

import jax
import jax.numpy as jnp
from jax._src.custom_partitioning_sharding_rule import (
    sdy_sharding_rule_to_mlir,
    str_to_sdy_sharding_rule,
)
from jax._src.lib.mlir.dialects import hlo
from jax.core import ShapedArray
from jax.experimental.shard_map import shard_map
from jax.extend import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax.sharding import Mesh, PartitionSpec

# Annotates a tensor to a specific memory space.
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
    rule = str_to_sdy_sharding_rule("... -> ...")
    out.attributes["sdy.sharding_rule"] = sdy_sharding_rule_to_mlir(
        rule,
        [x.type for x in out.operands],
        [
            out.results[0].type,
        ],
    )
    return out.results


mlir.register_lowering(annotate_memory_space_p, _annotate_memory_space_lowering)


def annotate_memory_space(x, memory_space: str = None):
    return annotate_memory_space_p.bind(x, memory_space=memory_space)


# Annotates a tensor to a specific memory space.
allocate_buffer_p = core.Primitive("allocate_buffer")
allocate_buffer_p.def_impl(lambda shape=None, dtype=None: jnp.zeros(shape, dtype))
allocate_buffer_p.def_abstract_eval(
    lambda shape=None, dtype=None: ShapedArray(shape, dtype)
)


def _allocate_buffer_lowering(ctx, shape=None, dtype=None):
    result_types = [mlir.aval_to_ir_type(s) for s in ctx.avals_out]
    out = hlo.CustomCallOp(
        result_types,
        inputs=[],
        call_target_name=ir.StringAttr.get("AllocateBuffer"),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 4),
    )
    return out.results


mlir.register_lowering(allocate_buffer_p, _allocate_buffer_lowering)


def allocate_buffer(shape=None, dtype=None):
    return allocate_buffer_p.bind(shape=shape, dtype=dtype)
