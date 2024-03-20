import functools
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils import _pytree as pytree


@functools.lru_cache
def has_tf_package() -> bool:
  try:
    import tensorflow
    return tensorflow is not None
  except ImportError:
    return False


def wrap_func_as_nn_module(f):

  class M(torch.nn.Module):

    def __init__(self):
      super().__init__()

    def forward(self, *args):
      return f(*args)

  return M().eval()


def load_save_model_and_inference(path: str, args: Tuple[Any, ...]) -> Dict:
  assert has_tf_package()
  import tensorflow as tf
  loaded_m = tf.saved_model.load(path)
  tf_input = pytree.tree_map_only(torch.Tensor,
                                  lambda x: tf.constant(x.numpy()), args)
  tf_output = loaded_m.f(*tf_input)
  return tf_output


def compare_exported_program_and_saved_model_result(ep,
                                                    saved_model_path,
                                                    args,
                                                    atol=1e-7):
  tf_output = load_save_model_and_inference(saved_model_path, args)
  with torch.no_grad():
    torch_output = ep.module()(*args)
  if not isinstance(torch_output, tuple):
    torch_output = (torch_output,)
  assert len(torch_output) == len(tf_output)
  for idx in range(len(torch_output)):
    torch_output_np = torch_output[idx].numpy()
    tf_output_np = tf_output[idx].numpy()
    assert torch_output_np.dtype == tf_output_np.dtype, f"torch dtype: {torch_output[idx].dtype}, tf dtype: {tf_output[idx].dtype}"
    assert np.allclose(torch_output_np, tf_output_np, atol=atol)
  return tuple(map(lambda x: x.numpy(),
                   torch_output)), tuple(map(lambda x: x.numpy(), tf_output))
