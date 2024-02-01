import os
from typing import Callable, List, Tuple, Type, Union

import requests
import torch
import torch.nn as nn
import torch_xla
from torch.export import Dim, export
from torch_xla.core import xla_model as xm
from torch_xla.stablehlo import exported_program_to_stablehlo
from torch_xla.tf_saved_model_integration import \
    save_torch_module_as_tf_saved_model
from transformers import ViTForImageClassification
from utils import (compare_exported_program_and_saved_model_result,
                   has_tf_package, load_save_model_and_inference,
                   wrap_func_as_nn_module)

class ViTForImageClassificationModelWrapper(nn.Module):

  def __init__(self, model_name):
    super().__init__()
    self.m = ViTForImageClassification.from_pretrained(model_name)

  def forward(self, img):
    return self.m(pixel_values=img).logits

model = ViTForImageClassificationModelWrapper('google/vit-base-patch16-224').eval()
args = (torch.rand(2, 3, 224, 224),)
dynamic_shapes = ({0: Dim("dim")},)
ep = export(model, args=args, dynamic_shapes=dynamic_shapes)
torch_output = ep.module()(*args)

import pdb;pdb.set_trace()

tf_output = load_save_model_and_inference('/tmp/vit-export/tmp1', args)




