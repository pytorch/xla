import torch
from torch_xla2 import tensor  # pylint: disable=unused-import
import torch_xla2
import torch_xla2.export

from .. import test_base
from . import llama_model
from . import model_exportable
from torch.utils import _pytree as pytree


class LlamaTest(test_base.TestCase):

  def test_can_run(self):
    with torch_xla2.default_env():
      sample_args = (
          torch.randint(0, 32000, (1, 2048), device='jax:0'),
          torch.arange(0, 2048, device='jax:0'),
      )

      model_args = llama_model.ModelArgs(
          block_size=2048,
          vocab_size=32000,
          n_layer=2,
          n_head=4,
          dim=256,
      )
      m = llama_model.Transformer(model_args)
      m.to(torch.bfloat16)
      m.setup_caches(1, 2048)
      m = m.to('jax')

      print(m(*sample_args))


  def test_can_run_exportable(self):
      model_args = model_exportable.ModelArgs(
          vocab_size=32000,
          n_layers=2,
          n_heads=4,
          dim=256,
      )
      m = model_exportable.Transformer(model_args)
      context_length = 2048
      input_shape_prefill = (1, context_length)
      input_shape_decode = (1, 1)

      def make_cache(args, batch_size):
        n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        n_local_heads = args.n_heads
        n_local_kv_heads = n_kv_heads
        n_rep = n_local_heads // n_local_kv_heads
        head_dim = args.dim // args.n_heads
        res = []
        for i in range(args.n_layers):
          if batch_size is None:
            size = (
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,
            )
          else:
            size = (
                batch_size,
                args.max_seq_len,
                n_local_kv_heads,
                head_dim,
            )
          res.append(
              (torch.zeros(
                  size,
                  dtype=torch.bfloat16 if args.bf16_enable else torch.float),
              torch.zeros(
                  size,
                  dtype=torch.bfloat16 if args.bf16_enable else torch.float)))
        return res

      prefill_caches = make_cache(model_args, 1)

      sample_input_prefill = (
          torch.randint(0, 1000, input_shape_prefill,
                        dtype=torch.int32),  # len seq length
          torch.arange(0, context_length, dtype=torch.int32),  # input indexes
          torch.arange(0, context_length, dtype=torch.int32),  # context indexes
          prefill_caches,
          True,  # prefil
      )
      with torch.no_grad():
        m_prefill = torch.export.export(m, sample_input_prefill)

      weights, mj_prefill = torch_xla2.export.exported_program_to_jax(m_prefill)
      sample_inputs = pytree.tree_map_only(torch.Tensor, tensor.t2j,
                                          sample_input_prefill)
      print('Prefill', mj_prefill(weights, sample_inputs))

      sample_input_decode = (
          torch.randint(0, 1000, input_shape_decode,
                        dtype=torch.int32),  # len = 1
          torch.tensor([0], dtype=torch.int32),
          torch.roll(torch.arange(context_length, dtype=torch.int32), 1, 0),
          prefill_caches,
          False  # prefill
      )
      with torch.no_grad():
        m_decode = torch.export.export(m, sample_input_decode)
      weights, mj_decode = torch_xla2.export.exported_program_to_jax(m_decode)
      sample_inputs = pytree.tree_map_only(torch.Tensor, tensor.t2j,
                                          sample_input_decode)
      print('Decode', mj_decode(weights, sample_inputs))


if __name__ == "__main__":
  test_base.main()
