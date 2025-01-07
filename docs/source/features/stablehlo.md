# Torch Export to StableHLO

This document describes how to use torch export + torch xla to export to
[StableHLO](https://github.com/openxla/stablehlo) format.

``` python
from torch.export import export
import torchvision
import torch
import torch_xla2 as tx
import torch_xla2.export

resnet18 = torchvision.models.resnet18()
# Sample input is a tuple
sample_input = (torch.randn(4, 3, 224, 224), )
output = resnet18(*sample_input)
exported = export(resnet18, sample_input)

weights, stablehlo = tx.export.exported_program_to_stablehlo(exported)
print(stablehlo.mlir_module())
# Can store weights and/or stablehlo object however you like
```

The stablehlo object is of type `jax.export.Exported`. 
Feel free to explore: https://openxla.org/stablehlo/tutorials/jax-export 
for more details on how to use the MLIR code generated from it.


## Preserving High-Level PyTorch Operations in StableHLO by generating `stablehlo.composite`

High level PyTorch ops (e.g. `F.scaled_dot_product_attention`) will be
decomposed into low level ops during PyTorch -\> StableHLO lowering.
Capturing the high level op in downstream ML compilers can be crucial
for genearting a performant, efficient specialized kernels. While
pattern matching a bunch of low level ops in the ML compiler can be
challenging and error-prone, we offer a more robust method to outline
the high-level PyTorch op in StableHLO program - by generating
[stablehlo.composite](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#composite)
for the high level PyTorch ops.

The following example shows a pratical use case - capturing
`scaled_product_attention`

``` python
import torch
import torch.nn.functional as F
from torch_xla import stablehlo
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(128, 128, bias=False)
        self.k_proj = torch.nn.Linear(128, 128, bias=False)
        self.v_proj = torch.nn.Linear(128, 128, bias=False)
        # Initialize the StableHLOCompositeBuilder with the name of the composite op and its attributes
        # Note: To capture the value of non-tensor inputs, please pass them as attributes to the builder
        self.b = StableHLOCompositeBuilder("test.sdpa", {"scale": 0.25, "other_attr": "val"})

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = self.b.mark_inputs(q, k, v)
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=0.25)
        attn_out = self.b.mark_outputs(attn_out)
        attn_out = attn_out + x
        return attn_out

input_args = (torch.randn((10, 8, 128)), )
# torch.export to Exported Program
exported = torch.export.export(M(), input_args)
# Exported Program to StableHLO
stablehlo_gm = stablehlo.exported_program_to_stablehlo(exported)
stablehlo = stablehlo_gm.get_stablehlo_text()
print(stablehlo)
```

The main StableHLO graph is shown below:

``` none
module @IrToHlo.56 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<10x8x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<10x8x128xf32> {
    ...
    %10 = stablehlo.composite "test.sdpa" %3, %6, %9 {composite_attributes = {other_attr = "val", scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl} : (tensor<10x8x128xf32>, tensor<10x8x128xf32>, tensor<10x8x128xf32>) -> tensor<10x8x128xf32>
    %11 = stablehlo.add %10, %arg0 : tensor<10x8x128xf32>
    return %11 : tensor<10x8x128xf32>
  }

  func.func private @test.sdpa.impl(%arg0: tensor<10x8x128xf32>, %arg1: tensor<10x8x128xf32>, %arg2: tensor<10x8x128xf32>) -> tensor<10x8x128xf32> {
    // Actual implementation of the composite
    ...
    return %11 : tensor<10x8x128xf32>
  }
```

The sdpa operation is encapsulated as a stablehlo composite call within
the main graph. The name and attributes specified in the torch.nn.Module
are propagated.

``` none
%10 = stablehlo.composite "test.sdpa" %3, %6, %9 {composite_attributes = {other_attr = "val", scale = 2.500000e-01 : f32}, decomposition = @test.sdpa.impl}
```

The reference PyTorch decomposition of the sdpa operation is captured in
a StableHLO function:

``` none
func.func private @test.sdpa.impl(%arg0: tensor<10x8x128xf32>, %arg1: tensor<10x8x128xf32>, %arg2: tensor<10x8x128xf32>) -> tensor<10x8x128xf32> {
    // Actual implementation of the composite
    ...
    return %11 : tensor<10x8x128xf32>
  }
```
