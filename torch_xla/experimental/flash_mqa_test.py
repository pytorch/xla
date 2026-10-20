"""Demo of existing custom kernels."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
from jax._src import test_util as jtu
import jax.numpy as jnp
# from jax._src.pallas.google.tpu_ops import flash_mqa
from torch_xla.experimental.pallas_kernels import flash_mqa



RUN_BENCHMARK = False


@jtu.with_config(jax_legacy_prng_key='allow')
class FlashMQATest(jtu.JaxTestCase):

  @parameterized.product(
      causal=(True,),
      block_q=(128,),
      block_k_major=(128,),
      block_k=(128,),
  )
  @jtu.skip_on_flag("jax_skip_slow_tests", True)
  def test_flash_attention(
      self, causal, block_q, block_k_major, block_k
  ):
    if block_k_major < block_k:
      self.skipTest("Invalid block_k.")
    if causal and block_q > block_k:
      # TODO(sharadmv, apaszke): enable this
      self.skipTest("Not yet working")
    q_seq_len = kv_seq_len = 1024
    n_heads = 2
    batch_size = 4
    head_dim = 256
    dtype = jnp.bfloat16
    kv_shape = (batch_size, kv_seq_len, head_dim)
    q_shape = (batch_size, n_heads, q_seq_len, head_dim)
    q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)
    q = random.normal(q_key, q_shape, dtype=dtype)
    k = random.normal(k_key, kv_shape, dtype=dtype)
    v = random.normal(v_key, kv_shape, dtype=dtype)
    out = flash_mqa.flash_mqa(
        q, k, v, causal=causal, block_k=block_k, block_k_major=block_k_major,
        block_q=block_q
    )
    out_ref = flash_mqa.mqa_reference(q, k, v, causal=causal)
    self.assertAllClose(out, out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
