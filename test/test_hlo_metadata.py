import sys

# Normal imports section starts here.
import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest
import json
from custom_debug_lowering import CustomOpNameLowering


class TestHloMetaData(unittest.TestCase):

  def setUp(self):
    torch.manual_seed(42)
    self.pre_test_tensor_type = torch.get_default_dtype()
    self.pre_test_ir_debug = torch_xla._XLAC._get_ir_debug()
    torch.set_default_tensor_type(torch.FloatTensor)
    torch_xla._XLAC._set_ir_debug(True)
    super(TestHloMetaData, self).setUp()

  def tearDown(self):
    super(TestHloMetaData, self).tearDown()
    torch_xla._XLAC._set_ir_debug(self.pre_test_ir_debug)

  def test_metadata(self):
    layer1 = torch.nn.Linear(4, 4)
    nl1 = torch.nn.ReLU()
    layer2 = torch.nn.Linear(4, 2)
    nl2 = torch.nn.Tanh()
    model = torch.nn.Sequential(layer1, nl1, layer2, nl2)

    with CustomOpNameLowering():
      model = model.to(device=xm.xla_device())
      inp = torch.rand(4, 4, device=xm.xla_device())
      out = model(inp)

    ctx = torch_xla._XLAC.lowering.LoweringContext()
    ctx.build([out])
    hlo_text = ctx.hlo_json()

    # Strings to match in the lowering
    bingo = {
        "torch/_ops.py": False,
        #"torch/nn/modules/linear.py": False,
        #"torch/nn/modules/activation.py": False,
        #"torch/nn/functional.py": False,
        "Sequential[model]/Linear[0]": False,
        "Sequential[model]/ReLU[1]": False,
        "Sequential[model]/Linear[2]": False,
        "Sequential[model]/Tanh[3]": False,
        "aten__addmm": False,
        "aten__relu": False,
        "aten__tanh": False,
        "aten__permute": False
    }

    non_zero_metadata = False

    local_json = json.loads(hlo_text)
    assert "computations" in local_json
    for c in local_json["computations"]:
      if "instructions" in c:
        i = c["instructions"]
        for op in i:
          if 'metadata' in op:
            meta = op["metadata"]
            print(meta)
            if len(meta) > 0:
              non_zero_metadata = True
            for km, vm in meta.items():
              for k in bingo.keys():
                if isinstance(vm, str) and k in vm:
                  bingo[k] = True

    assert non_zero_metadata, "No metadata was lowered - an issue with turning on IR DEBUG?"

    for k, v in bingo.items():
      assert v, f"Keyword {k} was not found as expected in HLO metadata for simple test"

    print("All required metadata symbols matched")


if __name__ == '__main__':
  test = unittest.main(exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(met.metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
