import torch_xla


def _next_power_of_2_bit_manipulation(x):
  """
    Finds the smallest power of 2 >= x using bit manipulation.
    Assumes x is an integer.

    Args:
      x: The input number (should be an integer).

    Returns:
      The smallest integer power of 2 that is >= x.
      Returns 1 if x <= 0.
    """
  if x <= 0:
    return 1
  if x == 1:
    return 1
  return 1 << (x - 1).bit_length()


# ragged_paged_attention
# key: (q_head_num, kv_head_num, token_num, max_model_len)
# value: (num_kv_pages_per_block, num_queries_per_block)


def _simplify_key_ragged_paged_attention(q_head_num, kv_head_num, token_num,
                                         max_model_len):
  token_num = _next_power_of_2_bit_manipulation(token_num)
  max_model_len = _next_power_of_2_bit_manipulation(max_model_len)
  return q_head_num, kv_head_num, token_num, max_model_len


# TODO: add more tuned block sizes in the table
# q_head_num, kv_head_num, token_num, max_model_len
_ragged_attention_table = {
    (32, 8, 4096, 2048): (128, 64),
    (4, 1, 4096, 2048): (128, 128),
    (32, 8, 2048, 2048): (128, 32),
    (4, 1, 2048, 2048): (128, 64),
    (32, 8, 1024, 2048): (64, 32),
    (1, 1, 1024, 2048): (64, 32),
    (32, 8, 4096, 4096): (128, 64),
    (4, 1, 4096, 4096): (128, 128),
    (32, 8, 2048, 4096): (128, 32),
    (4, 1, 2048, 4096): (128, 64),
    (32, 8, 1024, 4096): (64, 32),
    (1, 1, 1024, 4096): (64, 32),
    (32, 8, 4096, 64): (32, 32),
    (4, 1, 4096, 64): (32, 32),
    (32, 8, 2048, 64): (32, 32),
    (4, 1, 2048, 64): (32, 32),
    (32, 8, 1024, 64): (32, 32),
    (1, 1, 1024, 64): (32, 32),
    (32, 8, 4096, 128): (32, 32),
    (4, 1, 4096, 128): (32, 32),
    (32, 8, 2048, 128): (32, 32),
    (4, 1, 2048, 128): (32, 32),
    (32, 8, 1024, 128): (32, 32),
    (1, 1, 1024, 128): (32, 32),
    (10, 2, 4096, 2048): (128, 32),  # Qwen/Qwen2.5-32B
    (10, 2, 2048, 2048): (128, 32),  # Qwen/Qwen2.5-32B
    (10, 2, 1024, 2048): (128, 32),  # Qwen/Qwen2.5-32B
    (5, 1, 4098, 2048): (128, 64),  # Qwen/Qwen2.5-32B
    (5, 1, 2048, 2048): (128, 32),  # Qwen/Qwen2.5-32B
    (5, 1, 1024, 2048): (128, 32),  # Qwen/Qwen2.5-32B
}


def get_ragged_attention_tuned_block_size(q_head_num, kv_head_num, token_num,
                                          max_model_len):
  tpu_version = torch_xla.tpu.version()
  if tpu_version < 4:
    raise NotImplementedError("TPU version must be 4 or higher.")
  if tpu_version == 4:
    # This default block size is not tuned, only make sure there's no
    # OOM in vmem
    num_kv_pages_per_block = 16
    num_queries_per_block = 128
    return num_kv_pages_per_block, num_queries_per_block

  key = _simplify_key_ragged_paged_attention(q_head_num, kv_head_num, token_num,
                                             max_model_len)
  block_sizes = _ragged_attention_table.get(key, (128, 32))
  return block_sizes
