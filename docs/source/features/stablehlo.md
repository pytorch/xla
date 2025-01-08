# Torch Export to StableHLO

This document describes how to use torch export + torch xla to export to
[StableHLO](https://github.com/openxla/stablehlo) format.

There are 2 ways to accomplish this:

1. First do torch.export to create a ExportedProgram, which contains the program
   in torch.fx graph. Then use `exported_program_to_stablehlo` to convert it into an object that contains 
   stablehlo MLIR code.
2. First convert pytorch model to a jax function, then use jax utilities to convert it
   to stablehlo

## Using `torch.export`

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

## Using `extract_jax`

```python
from torch.export import export
import torchvision
import torch
import torch_xla2 as tx
import torch_xla2.export
import jax
import jax.numpy as jnp

resnet18 = torchvision.models.resnet18()
# Sample input is a tuple
sample_input = (torch.randn(4, 3, 224, 224), )
output = resnet18(*sample_input)

weights, jfunc = tx.extract_jax(resnet18)

# Below are APIs from jax

stablehlo = jax.export.export(jax.jit(jfunc))(weights, (jax.ShapedDtypeStruct((4, 3, 224, 224), jnp.float32.dtype)))

print(stablehlo.mlir_module())
# Can store weights and/or stablehlo object however you like
```

The second to last line we used `jax.ShapedDtypeStruct` to specify the input shape.
You can also pass a numpy array here.


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

For using `composite` we need to use the jax-centric export now. (i.e. no torch.export)
We are working in adding support for torch.export now.

``` python
import torch
import torch.nn.functional as F
import torch_xla2 as tx
import torch_xla2.interop
import torch_xla2.export

import jax
import jax.numpy as jnp



# We will use jax.lax.composite to accomplish this.
wrap_composite = tx.interop.torch_view(jax.lax.composite)


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(128, 128, bias=False)
        self.k_proj = torch.nn.Linear(128, 128, bias=False)
        self.v_proj = torch.nn.Linear(128, 128, bias=False)

        self._composite_sdpa = wrap_composite(F.scaled_dot_product_attention, name="test.sdpa")

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self._composite_sdpa(q, k, v, scale=0.25)
        return attn_out

weights, jfunc = tx.extract_jax(M())
stablehlo = jax.export.export(jax.jit(jfunc))(
    weights, jax.ShapeDtypeStruct((4, 8, 128), jnp.float32.dtype))
print(stablehlo.mlir_module())
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
%12 = stablehlo.composite "test.sdpa" %3, %7, %11 {composite_attributes = {scale = 2.500000e-01 : f64}, decomposition = @test.sdpa} : (tensor<4x8x128xf32>, tensor<4x8x128xf32>, tensor<4x8x128xf32>) -> tensor<4x8x128xf32> loc(#loc95)
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

As we see, to emit a stablehlo function into composite, first we make a python function
representing the region of code that we want to call, (
in this case `F.scaled_dot_product_attention` is already such function). 
Then we wrap the function with `wrap_composite`.

NOTE: currently a model with `wrap_composite` call will not work with `torch.export`.
We are actively working to make it work.
