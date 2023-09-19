import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch._export import capture_pre_autograd_graph
import torchvision
from torch_xla.experimental import quantize_utils

# Step 1: export resnet18
input_args = (torch.randn(1, 3, 224, 224),)
m = torchvision.models.resnet18().eval()
m = capture_pre_autograd_graph(m, input_args)

# Step 2: Insert observers or fake quantize modules
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
m = prepare_pt2e(m, quantizer)

# Step 3: Quantize the model
m = convert_pt2e(m)

# Trace with torch/xla and export stablehlo
stablehlo_txt = quantize_utils.pt2e_reference_model_to_stablehlo(m, input_args)
print(stablehlo_txt)
