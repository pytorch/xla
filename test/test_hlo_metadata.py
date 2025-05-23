import sys

# Normal imports section starts here.
import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import unittest
import json
import inspect
import copy
from custom_debug_lowering import CustomOpNameLowering, StackLayerSignature


class HloStackExtractor:

  def __init__(self, hlo_json):
    assert 'stackFrameIndex' in hlo_json
    assert 'fileLocations' in hlo_json['stackFrameIndex']
    assert 'stackFrames' in hlo_json['stackFrameIndex']
    assert 'fileNames' in hlo_json['stackFrameIndex']
    assert 'functionNames' in hlo_json['stackFrameIndex']

    self.file_locations = hlo_json['stackFrameIndex']['fileLocations']
    self.stack_frames = hlo_json['stackFrameIndex']['stackFrames']
    self.file_names = hlo_json['stackFrameIndex']['fileNames']
    self.function_names = hlo_json['stackFrameIndex']['functionNames']

  def extract(self, stack_frame_id):
    stack_sigs = []

    stack_frame = self.stack_frames[stack_frame_id - 1]

    while True:
      file_location_id = stack_frame['fileLocationId']
      file_location = self.file_locations[file_location_id - 1]
      file_name_id = file_location['fileNameId']
      function_name_id = file_location['functionNameId']
      line = file_location['line']
      file_name = self.file_names[file_name_id - 1]
      function_name = self.function_names[function_name_id - 1]

      sig = StackLayerSignature(file_name, function_name, line)
      stack_sigs.append(sig)

      stack_frame_id = 0
      if 'parentFrameId' in stack_frame:
        stack_frame_id = stack_frame['parentFrameId']

      if stack_frame_id == 0:
        break
      else:
        stack_frame = self.stack_frames[stack_frame_id - 1]

    return stack_sigs


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

    with CustomOpNameLowering() as c:
      model = model.to(device='xla')
      inp = torch.rand(4, 4, device='xla')
      #inp = torch.rand(4, 4)
      #inp = inp.to(device='xla')
      out = model(inp)

      # Get outer frames
      stack_sigs = c.stack_sigs

    ctx = torch_xla._XLAC.lowering.LoweringContext()
    ctx.build([out])
    hlo_text = ctx.hlo_json()

    # Strings to match in the lowering
    bingo = {
        "torch/nn/modules/linear.py": False,
        "torch/nn/modules/activation.py": False,
        "torch/nn/functional.py": False,
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

    #with open("./hlo.json", "w") as f:
    #  f.write(json.dumps(local_json, indent=2))

    hloEx = HloStackExtractor(local_json)

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

            # Decode stack frame id and check it matches one of the
            # the passed in stacks
            stack_frame_match = False
            if 'stackFrameId' in meta:
              hlo_stack_sig = hloEx.extract(meta['stackFrameId'])

              for t_sig in stack_sigs:
                if len(hlo_stack_sig) == len(t_sig) and hlo_stack_sig == t_sig:
                  stack_frame_match = True
                  break
                elif len(hlo_stack_sig) > len(t_sig):
                  hlo_stack_sig_copy = copy.copy(hlo_stack_sig)
                  discards = []
                  while len(hlo_stack_sig_copy) > len(t_sig):
                    discards.append(hlo_stack_sig_copy.pop(0))
                  # Print an error message on a partial match
                  if hlo_stack_sig_copy == t_sig:
                    print(f"** PARTIAL MATCH: Discarded {discards}")

              assert stack_frame_match, f"Stack\n{hlo_stack_sig} does not match any of\n{stack_sigs}"

    assert non_zero_metadata, "No metadata was lowered - an issue with turning on IR DEBUG?"

    for k, v in bingo.items():
      assert v, f"Keyword {k} was not found as expected in HLO metadata for simple test"

    print("All required metadata symbols matched")


if __name__ == '__main__':
  test = unittest.main(exit=False)
  if xu.getenv_as('METRICS_DEBUG', bool, defval=False):
    print(met.metrics_report())
  sys.exit(0 if test.result.wasSuccessful() else 1)
