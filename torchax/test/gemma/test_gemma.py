import torch
import unittest
import torchax
from torch.utils import _pytree as pytree
from . import config
from . import model as gemma


class GemmaTest(unittest.TestCase):

  def setup(self):
    torch.manual_seed(0)

  def test_gemma(self):
    mconfig = config.GemmaConfig(
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=256,
        intermediate_size=16384,
        dtype=torch.float32)
    model = gemma.GemmaForCausalLM(mconfig)
    batch_size = 1
    max_seq_len = 1000
    min_prompt_len = 1000
    device = 'cpu'
    pad_id = -1
    temperature = 0.8
    top_k = 100
    top_p = 1.0

    # prepare inputs
    token_ids_tensor = torch.randint(
        0, max_seq_len, (batch_size, max_seq_len), dtype=torch.int64)

    # build KV caches
    kv_caches = []
    for _ in range(model.config.num_hidden_layers):
      size = (batch_size, max_seq_len, model.config.num_key_value_heads,
              model.config.head_dim)
      dtype = model.config.get_dtype()
      k_cache = torch.zeros(size=size, dtype=dtype, device=device)
      v_cache = torch.zeros(size=size, dtype=dtype, device=device)
      kv_caches.append((k_cache, v_cache))

    token_ids_tensor = token_ids_tensor.to(device)
    prompt_mask_tensor = torch.ones_like(token_ids_tensor)
    input_positions_tensor = torch.arange(
        0, min_prompt_len, dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                             -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = torch.FloatTensor([temperature] *
                                            batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

    inputs = (
        token_ids_tensor,
        input_positions_tensor,
        None,  # kv_write_indexes
        kv_caches,
        mask_tensor,
        output_positions_tensor,
        temperatures_tensor,
        top_ps_tensor,
        top_ks_tensor,
    )

    weights, jax_func = torchax.extract_jax(model)
    env = torchax.default_env()
    inputs_jax = env.t2j_copy(inputs)

    import jax
    print(jax.jit(jax_func)(weights, inputs_jax))


if __name__ == '__main__':
  unittest.main()
