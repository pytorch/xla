import os
import torch
import torch_xla
from torch_xla.core import xla_model as xm
import torch.nn as nn
from torch.export import export, Dim
from typing import Tuple, Type, Callable, Union, List
import requests
from transformers import ViTForImageClassification
from torch_xla.stablehlo import exported_program_to_stablehlo

os.environ['EXPERIMENTAL_XLA_UNBOUNDED_DYNAMISM'] = '1'

device = xm.xla_device()
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
args = (torch.rand(10, 3, 224, 224),)
dynamic_shapes = ({0: Dim("dim")},)
ep = export(model, args=args, dynamic_shapes=dynamic_shapes)

# ep.graph_module.graph.print_tabular()
shlo_module = exported_program_to_stablehlo(ep)
shlo_text = shlo_module.get_stablehlo_text()
print(shlo_text)
