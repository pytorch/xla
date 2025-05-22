# Torch Export to StableHLO

This document describes how to use torch export + torch xla to export to
[StableHLO](https://github.com/openxla/stablehlo) format.

There are 2 ways to accomplish this:

1. First do torch.export to create a ExportedProgram, which contains the program
   in torch.fx graph. Then use `exported_program_to_stablehlo` to convert it
   into an object that contains stablehlo MLIR code.
1. First convert pytorch model to a jax function, then use jax utilities to
   convert it to stablehlo

## Using `torch.export`

```python
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

The stablehlo object is of type `jax.export.Exported`. Feel free to explore:
https://openxla.org/stablehlo/tutorials/jax-export for more details on how to
use the MLIR code generated from it.

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

The second to last line we used `jax.ShapedDtypeStruct` to specify the input
shape. You can also pass a numpy array here.

## Preserving High-Level PyTorch Operations in StableHLO by generating `stablehlo.composite`

High level PyTorch ops (e.g. `F.scaled_dot_product_attention`) will be
decomposed into low level ops during PyTorch -> StableHLO lowering. Capturing
the high level op in downstream ML compilers can be crucial for genearting a
performant, efficient specialized kernels. While pattern matching a bunch of low
level ops in the ML compiler can be challenging and error-prone, we offer a more
robust method to outline the high-level PyTorch op in StableHLO program - by
generating
[stablehlo.composite](https://github.com/openxla/stablehlo/blob/main/docs/spec.md#composite)
for the high level PyTorch ops.

The following example shows a pratical use case - capturing
`scaled_product_attention`

For using `composite` we need to use the jax-centric export now. (i.e. no
torch.export) We are working in adding support for torch.export now.

```python
import unittest
import torch
import torch.nn.functional as F
from torch.library import Library, impl, impl_abstract
import torch_xla2
import torch_xla2.export
from torch_xla2.ops import jaten
from torch_xla2.ops import jlibrary


# Create a `mylib` library which has a basic SDPA op.
m = Library("mylib", "DEF")
m.define("scaled_dot_product_attention(Tensor q, Tensor k, Tensor v) -> Tensor")

@impl(m, "scaled_dot_product_attention", "CompositeExplicitAutograd")
def _mylib_scaled_dot_product_attention(q, k, v):
  """Basic scaled dot product attention without all the flags/features."""
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)
  y = F.scaled_dot_product_attention(
      q,
      k,
      v,
      dropout_p=0,
      is_causal=False,
      scale=None,
  )
  return y.transpose(1, 2)

@impl_abstract("mylib::scaled_dot_product_attention")
def _mylib_scaled_dot_product_attention_meta(q, k, v):
  return torch.empty_like(q)

# Register library op as a composite for export using the `@impl` method
# for a torch decomposition.
jlibrary.register_torch_composite(
  "mylib.scaled_dot_product_attention",
  _mylib_scaled_dot_product_attention,
  torch.ops.mylib.scaled_dot_product_attention,
  torch.ops.mylib.scaled_dot_product_attention.default
)

# Also register ATen softmax as a composite for export in the `mylib` library
# using the JAX ATen decomposition from `jaten`.
jlibrary.register_jax_composite(
  "mylib.softmax",
  jaten._aten_softmax,
  torch.ops.aten._softmax,
  static_argnums=1  # Required by JAX jit
)

class LibraryTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    torch_xla2.default_env().config.use_torch_native_for_cpu_tensor = False

  def test_basic_sdpa_library(self):

    class CustomOpExample(torch.nn.Module):
      def forward(self, q,k,v):
        x = torch.ops.mylib.scaled_dot_product_attention(q, k, v)
        x = x + 1
        return x

    # Export and check for composite operations
    model = CustomOpExample()
    arg = torch.rand(32, 8, 128, 64)
    args = (arg, arg, arg, )

    exported = torch.export.export(model, args=args)
    stablehlo = torch_xla2.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    ## TODO Update this machinery from producing function calls to producing
    ## stablehlo.composite ops.
    self.assertIn("call @mylib.scaled_dot_product_attention", module_str)
    self.assertIn("call @mylib.softmax", module_str)


if __name__ == '__main__':
  unittest.main()
```

As we see, to emit a stablehlo function into composite, first we make a python
function representing the region of code that we want to call, then, we register
it so that pytorch and jlibrary understands it's a custom region. Then, th
emitted Stablehlo will have `mylib.scaled_dot_product_attention` and
`mylib.softmax` outlined stablehlo functions.
