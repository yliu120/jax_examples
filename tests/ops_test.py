import functools

from jax_examples.magics.cuda import initialize_cuda
initialize_cuda()

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from jax_examples.magics.ops import allocate_buffer, annotate_memory_space


class MemoryAnnotateOpTest(absltest.TestCase):
    def test_annotate_memory_space(self):
        def f(x):
            # Obtain a buffer with global size (16,) in persistent temp
            # memory space.
            const = jnp.zeros(16)
            const = annotate_memory_space(const, memory_space="persistent_temp")
            return x + const

        mesh = jax.make_mesh((8,), ("x"))
        in_sharding = NamedSharding(mesh, P("x"))

        inp = jnp.arange(16)
        aot_compiled = (
            jax.jit(f, in_shardings=(in_sharding,), out_shardings=in_sharding)
            .lower(inp)
            .compile()
        )

        self.assertIn("f32[2]{0:S(2)} broadcast", aot_compiled.as_text())
        aot_compiled(inp)  # Don't crash.
        jax.effects_barrier()

    def test_allocate_buffer(self):
        f = lambda: (
            allocate_buffer(shape=(16,), dtype=jnp.float32),
            allocate_buffer(shape=(16,), dtype=jnp.float32),
        )
        compiled = jax.jit(f).lower().compile()
        self.assertEqual(compiled.as_text().count("AllocateBuffer"), 2)

        a, b = compiled()
        self.assertEqual(a.shape, (16,))
        self.assertEqual(b.shape, (16,))

    def test_allocate_buffer_with_multiple_devices(self):
        mesh = jax.make_mesh((8,), ("x"))
        out_sharding = NamedSharding(mesh, P("x"))
        f = lambda: (
            allocate_buffer(shape=(16,), dtype=jnp.float32),
            allocate_buffer(shape=(16,), dtype=jnp.float32),
        )
        compiled = (
            jax.jit(f, out_shardings=(out_sharding, out_sharding)).lower().compile()
        )
        self.assertEqual(compiled.as_text().count("AllocateBuffer"), 2)

        a, b = compiled()
        self.assertEqual(a.shape, (16,))
        self.assertEqual(b.shape, (16,))


if __name__ == "__main__":
    absltest.main()
