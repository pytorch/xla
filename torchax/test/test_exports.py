import unittest
import torch
import torch.nn.functional as F
import jax
import jax.export
import torchax
import torchax.export
from torchax import tensor
from torchax.ops import mappings


class Interpolate(torch.nn.Module):

  def forward(self, masks: torch.Tensor) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        size=(500, 500),
        mode="bilinear",
        align_corners=False,
    )
    return masks


class TensorConstant(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a):
    return a / torch.tensor(3)


class ExportTest(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(0)
    torchax.enable_accuracy_mode()

  def test_interpolate(self):

    # Check Accuracy
    arg = (torch.randn(3, 3, 200, 200),)
    model = Interpolate()
    ans = model(*arg)

    env = torchax.default_env()

    with torch.no_grad():
      exported = torch.export.export(model, arg)
    weights, func = torchax.export.exported_program_to_jax(exported)
    argj = env.t2j_copy(arg[0])
    ans2 = jax.jit(func)(weights, (argj,))[0]
    ans2 = env.j2t_copy(ans2)
    self.assertTrue(torch.allclose(ans, ans2, atol=1e-3))

    # Convert to StableHLO
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())
    self.assertIn("func.func public @main", module_str)
    self.assertIn("func.func private @clip(%arg0: tensor<500xf32>", module_str)
    self.assertIn("stablehlo.minimum", module_str)

  def test_constant(self):

    # Check Accuracy
    arg = (torch.randn(10, 10),)
    model = TensorConstant()
    ans = model(*arg)

    with torch.no_grad():
      exported = torch.export.export(model, arg)
    env = torchax.default_env()
    weights, func = torchax.export.exported_program_to_jax(exported)
    argj = env.t2j_copy(arg[0])
    ans2 = jax.jit(func)(weights, (argj,))[0]
    ans2 = env.j2t_copy(ans2)
    self.assertTrue(torch.allclose(ans, ans2, atol=1e-5))

    # Convert to StableHLO
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())
    self.assertIn("func.func public @main", module_str)
    self.assertIn("stablehlo.divide", module_str)

  def test_interpolate_dynamic(self):
    # Export with dynamic dimension constraints on both min and max
    arg = (torch.randn(3, 3, 200, 200),)
    model = Interpolate()
    ans = model(*arg)
    dynamic_shapes = ({0: torch.export.Dim("b", min=3, max=10)},)

    with torch.no_grad():
      exported = torch.export.export(model, arg, dynamic_shapes=dynamic_shapes)
    weights, stablehlo = torchax.export.exported_program_to_stablehlo(exported)
    module_str = str(stablehlo.mlir_module())

    # Look for dynamic shape artifacts
    self.assertIn("func.func public @main(%arg0: tensor<?x3x200x200xf32>",
                  module_str)
    self.assertIn("stablehlo.dynamic_broadcast_in_dim", module_str)
    self.assertIn("stablehlo.dynamic_gather", module_str)

  def test_export_dtypes(self):
    DTYPE_TO_MLIR_STR = {
        # NO_MAPPING        : jnp.float0 (signless scalar int)
        torch.bool:
            "i1",
        # NO_MAPPING        : "i4"
        torch.int8:
            "i8",
        torch.int16:
            "i16",
        torch.int32:
            "i32",
        torch.int64:
            "i64",
        torch.long:
            "i64",
        # NO_MAPPING        : "ui4"
        torch.uint8:
            "ui8",
        # NOTE(qihqi): torch export for uint16 seems broken at torch 2.4
        # torch.uint16        : "ui16",
        torch.uint32:
            "ui32",
        torch.uint64:
            "ui64",
        # NO_MAPPING        : "f8E4M3B11FNUZ"
        torch.float8_e4m3fn:
            "f8E4M3FN",
        # NO_MAPPING        : f8E4M3FNUZ
        torch.float8_e5m2:
            "f8E5M2",
        # NO_MAPPING        : f8E5M2FNUZ
        torch.bfloat16:
            "bf16",
        torch.half:
            "f16",
        torch.float16:
            "f16",
        torch.float32:
            "f32",
        torch.float64:
            "f64",
        torch.double:
            "f64",
        torch.complex64:
            "complex<f32>",
        torch.complex128:
            "complex<f64>",
        None:
            None,
    }

    model = TensorConstant()
    for torch_dtype in DTYPE_TO_MLIR_STR.keys():
      if torch_dtype == None:
        ## TODO: Figure out what the None mapping should be, seems like:
        ##   torch.tensor(dtype=None) maps to f32
        ##   jnp.tensor(dtype=None) maps to f64
        continue
      arg = (torch.randn(10).to(torch_dtype),)
      with torch.no_grad():
        exported = torch.export.export(model, arg)
      weights, stablehlo = torchax.export.exported_program_to_stablehlo(
          exported)
      module_str = str(stablehlo.mlir_module())
      self.assertIn(DTYPE_TO_MLIR_STR[torch_dtype], module_str)


if __name__ == '__main__':
  unittest.main()
