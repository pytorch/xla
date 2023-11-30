import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import unittest
from typing import NamedTuple


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def extract_execution_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Execution Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


def extract_compilation_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Compilation Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


class GraphInfo(NamedTuple):
  hash: str
  num_input: int
  num_output: int


def extract_graph_infos(lines):
  infos = []
  for i in range(len(lines)):
    if 'Graph Info' in lines[i].decode():
      hash = lines[i + 1].decode().split('Graph Hash: ')[1].strip()
      num_input = lines[i +
                        2].decode().split('Number of Graph Inputs:')[1].strip()
      num_output = lines[i + 3].decode().split(
          'Number of Graph Outputs:')[1].strip()
      infos.append(GraphInfo(hash, int(num_input), int(num_output)))

  return infos


class PtXLADebugTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    if not check_env_flag('PT_XLA_DEBUG'):
      assert False, "This test should be run with PT_XLA_DEBUG"
    cls.debug_file_name = os.getenv('PT_XLA_DEBUG_FILE')
    if not cls.debug_file_name:
      assert False, "This test should be run with PT_XLA_DEBUG_FILE"
    open(cls.debug_file_name, 'w').close()

  def test_user_mark_step(self):
    device = xm.xla_device()
    t1 = torch.randn(2, 2, device=device)
    xm.mark_step()
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      executation_causes = extract_execution_cause(lines)
      compilation_causes = extract_compilation_cause(lines)
      graph_infos = extract_graph_infos(lines)

    self.assertEqual(len(executation_causes), 1)
    self.assertIn('user mark_step', executation_causes[0])

    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('user mark_step', compilation_causes[0])

    self.assertEqual(len(graph_infos), 2)
    # one graph info from compilation, one from execution, hash should match
    self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[1].num_input, 1)
    self.assertEqual(graph_infos[1].num_output, 1)
    open(self.debug_file_name, 'w').close()

  def test_step_trace(self):
    device = xm.xla_device()
    with xp.StepTrace('train_pt_xla_debug'):
      t1 = torch.randn(3, 3, device=device)
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
      compilation_causes = extract_compilation_cause(lines)
      graph_infos = extract_graph_infos(lines)

    self.assertEqual(len(causes), 1)
    self.assertIn('mark_step when exiting a profiler StepTrace region',
                  causes[0])

    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('mark_step when exiting a profiler StepTrace region',
                  compilation_causes[0])

    self.assertEqual(len(graph_infos), 2)
    # one graph info from compilation, one from execution, hash should match
    self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[1].num_input, 1)
    self.assertEqual(graph_infos[1].num_output, 1)
    open(self.debug_file_name, 'w').close()

  def test_dynamo(self):
    device = xm.xla_device()
    t1 = torch.randn(4, 4, device=device)

    def toy_program(t1):
      return t1 * 100

    compiled = torch.compile(toy_program, backend="openxla")
    res = compiled(t1)
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      executation_causes = extract_execution_cause(lines)
      compilation_causes = extract_compilation_cause(lines)
      graph_infos = extract_graph_infos(lines)

    self.assertEqual(len(executation_causes), 2)
    self.assertIn('mark_step when dynamo processing input graphs',
                  executation_causes[0])
    self.assertIn('dynamo is executing a compiled program',
                  executation_causes[1])

    self.assertEqual(len(compilation_causes), 2)
    self.assertIn('mark_step when dynamo processing input graphs',
                  compilation_causes[0])
    self.assertIn('dynamo is compiling a FX graph to HLO',
                  compilation_causes[1])

    # one graph info from compilation, one from execution, hash should match
    self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[1].num_input, 1)
    self.assertEqual(graph_infos[1].num_output, 1)

    # one graph info from dynamo compilation, one from dynamo execution, hash should match
    self.assertEqual(graph_infos[2].hash, graph_infos[3].hash)
    # this graph has two input(t1, 100) and one output
    self.assertEqual(graph_infos[3].num_input, 2)
    self.assertEqual(graph_infos[3].num_output, 1)
    open(self.debug_file_name, 'w').close()

  def test_parallel_loader(self):
    device = xm.xla_device()

    train_dataset_len = 100
    batch_size = 10
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(batch_size, 3, 128,
                          128), torch.zeros(batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // 10)

    train_device_loader = pl.MpDeviceLoader(
        train_loader,
        device,
        loader_prefetch_size=8,
        device_prefetch_size=4,
        host_to_device_transfer_threads=1)

    for step, (data, target) in enumerate(train_device_loader):
      dummy = data + 100

    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      executation_causes = extract_execution_cause(lines)
      compilation_causes = extract_compilation_cause(lines)
      graph_infos = extract_graph_infos(lines)

    self.assertEqual(len(executation_causes), batch_size)
    for cause in executation_causes:
      self.assertIn('mark_step in parallel loader at step end', cause)

    # We should only compile once.
    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('mark_step in parallel loader at step end',
                  compilation_causes[0])

    self.assertEqual(len(graph_infos), batch_size + 1)
    # one graph info from compilation, batch size from execution, hash should match
    for i in range(batch_size + 1):
      self.assertEqual(graph_infos[0].hash, graph_infos[i].hash)
      # this graph has two input(data, 100) and one output(dummy)
      self.assertEqual(graph_infos[i].num_input, 2)
      self.assertEqual(graph_infos[i].num_output, 1)
    open(self.debug_file_name, 'w').close()

  def test_print(self):
    device = xm.xla_device()
    t1 = torch.randn(5, 5, device=device)
    print(t1)
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      executation_causes = extract_execution_cause(lines)
      compilation_causes = extract_compilation_cause(lines)
      graph_infos = extract_graph_infos(lines)

    self.assertEqual(len(executation_causes), 1)
    self.assertIn('user code trying to access tensor value',
                  executation_causes[0])

    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('user code trying to access tensor value',
                  compilation_causes[0])

    # one graph info from compilation, one from execution, hash should match
    self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[1].num_input, 1)
    self.assertEqual(graph_infos[1].num_output, 1)
    open(self.debug_file_name, 'w').close()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
