import os

import numpy as np
import tensorflow as tf
import torch
import torch_xla
from torch.export import Dim, export
from torch.utils import _pytree as pytree
from torch_xla.stablehlo import exported_program_to_stablehlo
from torch_xla.tf_saved_model_integration import save_torch_module_as_tf_saved_model
from transformers import Wav2Vec2ForCTC

os.environ["EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM"] = "1"


class ModelWrapper(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self._model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

  def forward(self, input):
    r = self._model(input)
    return r.logits


model = ModelWrapper().eval()
args = (torch.rand(3, 800),)
dynamic_shapes = ({0: Dim("bs")},)
ep = export(model, args=args, dynamic_shapes=dynamic_shapes)

tmp_dir = "/tmp/wav2vec2-export/tmp"
save_torch_module_as_tf_saved_model(
    model, args, tmp_dir, dynamic_shapes=dynamic_shapes)

# Verify numeric accuracy with an input with a different BS.
args = (torch.rand(2, 800),)
loaded_m = tf.saved_model.load(tmp_dir)
tf_input = pytree.tree_map_only(torch.Tensor, lambda x: tf.constant(x.numpy()),
                                args)
tf_output = loaded_m.f(*tf_input)
with torch.no_grad():
  torch_output = model(*args)
  print(np.max(torch_output.numpy() - tf_output[0].numpy()))
