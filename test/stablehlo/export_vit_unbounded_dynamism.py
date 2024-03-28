import os
from typing import Callable, List, Tuple, Type, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch_xla
from torch.export import Dim, export
from torch.utils import _pytree as pytree
from torch_xla.stablehlo import exported_program_to_stablehlo
from torch_xla.tf_saved_model_integration import \
    save_torch_module_as_tf_saved_model
from transformers import ViTForImageClassification

os.environ['EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM'] = '1'


class ViTForImageClassificationModelWrapper(nn.Module):

  def __init__(self, model_name):
    super().__init__()
    self.m = ViTForImageClassification.from_pretrained(model_name)

  def forward(self, img):
    return self.m(pixel_values=img).logits


model = ViTForImageClassificationModelWrapper(
    'google/vit-base-patch16-224').eval()
args = (torch.rand(10, 3, 224, 224),)
dynamic_shapes = ({0: Dim("dim")},)

# Export to saved_model
tmp_dir = "/tmp/vit-export/vit-1"
save_torch_module_as_tf_saved_model(
    model, args, tmp_dir, dynamic_shapes=dynamic_shapes)

# Verify numeric accuracy with an input with a different BS.
args = (torch.rand(2, 3, 224, 224),)
loaded_m = tf.saved_model.load(tmp_dir)
tf_input = pytree.tree_map_only(torch.Tensor, lambda x: tf.constant(x.numpy()),
                                args)
tf_output = loaded_m.f(*tf_input)
with torch.no_grad():
  torch_output = model(*args)
  print(np.max(torch_output.numpy() - tf_output[0].numpy()))
