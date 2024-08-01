import os

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch_xla
from torch.utils import _pytree as pytree
from torch.export import Dim, export
from torch_xla.stablehlo import exported_program_to_stablehlo
from torch_xla.tf_saved_model_integration import \
    save_torch_module_as_tf_saved_model
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

os.environ["EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM"] = "1"


class WrapModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self._model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/tinyroberta-squad2")

  def forward(self, input, mask):
    res = self._model.forward(input, mask)
    # return tuple(
    #     x for x in (res.loss, res.start_logits, res.end_logits,
    #                 res.hidden_states) if x is not None)
    return res.start_logits, res.end_logits


def _get_fake_pipeline_model_inputs():
  tokens_len = 10
  input_ids = torch.randint(
      low=0, high=2000, size=(3, tokens_len), dtype=torch.int64)
  attention_mask = torch.ones((3, tokens_len), dtype=torch.int64)
  return (input_ids, attention_mask)


model = WrapModel()
args = _get_fake_pipeline_model_inputs()
dynamic_shapes = ({0: Dim("bs")}, {0: Dim("bs")})
# dynamic_shapes = None
ep = export(model, args=args, dynamic_shapes=dynamic_shapes)

tmp_dir = "/tmp/tiny_roberta/tiny_roberta_export"
save_torch_module_as_tf_saved_model(
    model, args, tmp_dir, dynamic_shapes=dynamic_shapes)

tokens_len = 10
args = (torch.randint(
    low=0, high=2000, size=(2, tokens_len),
    dtype=torch.int64), torch.ones((2, tokens_len), dtype=torch.int64))
loaded_m = tf.saved_model.load(tmp_dir)
tf_input = pytree.tree_map_only(torch.Tensor, lambda x: tf.constant(x.numpy()),
                                args)

tf_output = loaded_m.f(*tf_input)
with torch.no_grad():
  torch_output = model(*args)
  print(np.max(torch_output[0].numpy() - tf_output[0].numpy()))
  print(np.max(torch_output[1].numpy() - tf_output[1].numpy()))
