import os
import re
import sys
from pathlib import Path

import unittest

import test_xla_sharding_base

import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.debug.metrics as met
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm


class TestSPMDLoweringContext(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def _get_computation_hlo_txt(self, ctx):
    hlo = ctx.hlo()
    comp = xb.computation_from_module_proto("my_custom_comp", hlo)
    return xb.get_computation_hlo(comp)

  def test_basic(self):
    save_file = os.getenv('XLA_SAVE_TENSORS_FILE')
    save_format = os.getenv('XLA_SAVE_TENSORS_FMT')
    assert save_file, "This test should be run with XLA_SAVE_TENSORS_FILE"
    save_file += '.0'  # Identify a single device
    assert save_format == 'hlo', "This test should be run with XLA_SAVE_TENSORS_FMT=hlo"

    model_axis = max(1, self.n_devices // 2)
    data_axis = self.n_devices // model_axis
    mesh_shape = (data_axis, model_axis)
    spmd_mesh = self._get_mesh(mesh_shape, axis_names=('x', 'y'))

    device = xm.xla_device()
    a = torch.zeros(2048, device=device, requires_grad=True)
    xs.mark_sharding(a, spmd_mesh, ('x',))
    b = torch.randn([32, 2048], device=device, requires_grad=True)
    xs.mark_sharding(b, spmd_mesh, (None, 'y'))

    def fn(x, y):
      x = x + 1
      return x, y * 2

    result = fn(a, b)
    ctx = torch_xla._XLAC.lowering.LoweringContext("MyCustomName")
    ctx.build(list(result))
    torch_xla.sync()

    # Sanity HLO check.
    hlo_text = ctx.hlo_text()
    self.assertIn('MyCustomName', hlo_text)
    self.assertIn('opcode: "parameter"', hlo_text)
    self.assertIn('opcode: "add"', hlo_text)
    self.assertIn('sharding', hlo_text)

    # Ensure that the corresponding input parameters contain the expected sharding.
    hlo_comp_txt = self._get_computation_hlo_txt(ctx)
    a_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(a)
    self.assertRegex(
        hlo_comp_txt,
        rf'%custom-call.*.*f32[2048]{{0}}.*sharding={re.escape(a_sharding_spec)}'
    )
    b_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(b)
    self.assertRegex(
        hlo_comp_txt,
        rf'%custom-call.*f32[32,2048]{{0}}.*sharding={re.escape(b_sharding_spec)}'
    )

    # Ensure that the results retain the same sharding specs.
    result_a, result_b = result
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(result_a), a_sharding_spec)
    self.assertEqual(
        torch_xla._XLAC._get_xla_sharding_spec(result_b), b_sharding_spec)

    hlo_content = Path(save_file).read_text()
    assert len(re.findall('END_GRAPH',
                          hlo_content)) == 1, "There is a single graph"

    # Extract the content between OUTPUT_SHARDING_BEGIN and OUTPUT_SHARDING_END
    pattern = r'#OUTPUT_SHARDING_BEGIN\n(.*?)\n#OUTPUT_SHARDING_END'
    match = re.search(pattern, hlo_content, re.DOTALL)
    assert match is not None, "#OUTPUT_SHARDING not found in the file"
    assert len(match.groups()
              ) == 1, f"Expected 1 group, but found {len(match.groups())}"
    expected_output = match.group(1).strip().split('\n')

    # Assert that the output sharding match our expectation.
    assert len(expected_output
              ) == 4, f"Expected 4 lines, but found {len(expected_output)}"
    assert expected_output[0] == f"f32[2048] {a_sharding_spec}"
    assert expected_output[1] == f"f32[32,2048] {b_sharding_spec}"
    assert expected_output[2] == f"f32[2048] {a_sharding_spec}"
    assert expected_output[3] == f"f32[32,2048] {b_sharding_spec}"

  def test_device_parameter_id_tensor_mapping(self):
    met.clear_all()

    model_axis = max(1, self.n_devices // 2)
    data_axis = self.n_devices // model_axis
    mesh_shape = (data_axis, model_axis)
    spmd_mesh = self._get_mesh(mesh_shape, axis_names=('x', 'y'))

    device = xm.xla_device()
    a = torch.randn([32, 2048]).to(device)
    xs.mark_sharding(a, spmd_mesh, ('x', 'y'))
    b = torch.ones(2048).to(device)
    xs.mark_sharding(b, spmd_mesh, ('x',))

    def fn(a, b):
      return a + b

    result = fn(a, b)
    ctx = torch_xla._XLAC.lowering.LoweringContext("MyCustomName")
    ctx.build([result])
    torch_xla.sync()

    mapping = ctx.device_parameter_id_tensor_mapping()
    num_params = len(mapping)
    self.assertEqual(num_params, 2)
    self.assertNotEqual(ctx.tensor_parameter_id(a), -1)
    self.assertNotEqual(ctx.tensor_parameter_id(b), -1)
    self.assertEqual(met.counter_value("VirtualDeviceUsage"), num_params)

    # Ensure that the parameter mapping does not require transferring data
    # from the device to the host when sharded.
    self.assertFalse(met.metric_data("TransferFromDeviceTime"))
    self.assertFalse(met.counter_value("ReplicateShardedData"))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
