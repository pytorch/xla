import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import unittest
from extract_debug_helper import (check_env_flag, extract_execution_cause,
                                  extract_compilation_cause,
                                  extract_graph_infos, extract_python_frames,
                                  extract_post_compilation_analysis)


class PtXLADebugTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    pt_xla_debug_enabled = xu.getenv_as('PT_XLA_DEBUG', bool, False)
    cls.debug_level = xu.getenv_as('PT_XLA_DEBUG_LEVEL', int, -1)
    cls.debug_level = 100 if (cls.debug_level == -1 and
                              pt_xla_debug_enabled) else cls.debug_level
    if not check_env_flag('PT_XLA_DEBUG') and cls.debug_level == -1:
      assert False, "This test should be run with PT_XLA_DEBUG"
    cls.debug_file_name = os.getenv('PT_XLA_DEBUG_FILE')
    if not cls.debug_file_name:
      assert False, "This test should be run with PT_XLA_DEBUG_FILE"
    open(cls.debug_file_name, 'w').close()

  def test_eager_mark_step(self):
    torch_xla.experimental.eager_mode(True)
    device = xm.xla_device()
    t1 = torch.randn(5, 9, device=device)
    xm.mark_step()
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
    # We expect PT_XLA_BUDEG not to output anything under the eager mode
    self.assertEqual(len(lines), 0)
    torch_xla.experimental.eager_mode(False)
    open(self.debug_file_name, 'w').close()

  def test_user_mark_step(self):
    device = xm.xla_device()
    t1 = torch.randn(2, 2, device=device)
    xm.mark_step()
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      executation_causes = extract_execution_cause(lines)
      compilation_causes = extract_compilation_cause(lines)
      graph_infos = extract_graph_infos(lines)
      post_compilation_infos = extract_post_compilation_analysis(lines)

    self.assertEqual(len(post_compilation_infos), 1)
    self.assertIn('GB', post_compilation_infos[0].input_size)
    self.assertIn('GB', post_compilation_infos[0].output_size)
    self.assertIn('GB', post_compilation_infos[0].aliased_size)
    self.assertIn('GB', post_compilation_infos[0].intermediate_size)
    self.assertIn('GB', post_compilation_infos[0].program_size)

    if self.debug_level > 1:
      self.assertEqual(len(executation_causes), 1)
      self.assertIn('user mark_step', executation_causes[0])
    else:
      self.assertEqual(len(executation_causes), 0)

    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('user mark_step', compilation_causes[0])

    if self.debug_level > 1:
      self.assertEqual(len(graph_infos), 2)
      # one graph info from compilation, one from execution, hash should match
      self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    else:
      self.assertEqual(len(graph_infos), 1)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[0].num_input, 1)
    self.assertEqual(graph_infos[0].num_output, 1)
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

    if self.debug_level > 1:
      self.assertEqual(len(causes), 1)
      self.assertIn('mark_step when exiting a profiler StepTrace region',
                    causes[0])
    else:
      self.assertEqual(len(causes), 0)

    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('mark_step when exiting a profiler StepTrace region',
                  compilation_causes[0])

    if self.debug_level > 1:
      self.assertEqual(len(graph_infos), 2)
      # one graph info from compilation, one from execution, hash should match
      self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    else:
      self.assertEqual(len(graph_infos), 1)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[0].num_input, 1)
    self.assertEqual(graph_infos[0].num_output, 1)
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

    if self.debug_level > 1:
      self.assertEqual(len(executation_causes), 2)
      self.assertIn('mark_step when dynamo processing input graphs',
                    executation_causes[0])
      self.assertIn('dynamo is executing a compiled program',
                    executation_causes[1])
    else:
      self.assertEqual(len(executation_causes), 0)

    self.assertEqual(len(compilation_causes), 2)
    self.assertIn('mark_step when dynamo processing input graphs',
                  compilation_causes[0])
    self.assertIn('dynamo is compiling a FX graph to HLO',
                  compilation_causes[1])

    if self.debug_level > 1:
      # one graph info from compilation, one from execution, hash should match
      self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[0].num_input, 1)
    self.assertEqual(graph_infos[0].num_output, 1)

    if self.debug_level > 1:
      # one graph info from dynamo compilation, one from dynamo execution, hash should match
      self.assertEqual(graph_infos[2].hash, graph_infos[3].hash)
      # this graph has two input(t1, 100) and one output
      self.assertEqual(graph_infos[3].num_input, 2)
      self.assertEqual(graph_infos[3].num_output, 1)
    else:
      # this graph has two input(t1, 100) and one output
      self.assertEqual(graph_infos[1].num_input, 2)
      self.assertEqual(graph_infos[1].num_output, 1)

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

    if self.debug_level > 1:
      self.assertEqual(len(executation_causes), batch_size)
      for cause in executation_causes:
        self.assertIn('mark_step in parallel loader at step end', cause)
    else:
      self.assertEqual(len(executation_causes), 0)

    # We should only compile once.
    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('mark_step in parallel loader at step end',
                  compilation_causes[0])

    if self.debug_level > 1:
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

    if self.debug_level > 1:
      self.assertEqual(len(executation_causes), 1)
      self.assertIn('user code trying to access tensor value',
                    executation_causes[0])
      # one graph info from compilation, one from execution, hash should match
      self.assertEqual(graph_infos[0].hash, graph_infos[1].hash)
    else:
      self.assertEqual(len(executation_causes), 0)

    self.assertEqual(len(compilation_causes), 1)
    self.assertIn('user code trying to access tensor value',
                  compilation_causes[0])

    # this graph has one input(random seed) and one output(t1)
    self.assertEqual(graph_infos[0].num_input, 1)
    self.assertEqual(graph_infos[0].num_output, 1)
    open(self.debug_file_name, 'w').close()

  def test_frame(self):
    device = xm.xla_device()
    t1 = torch.randn(6, 6, device=device)
    xm.mark_step()
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      frames = extract_python_frames(lines)

    # one for compilation, one for post-compilation analysis, one for execution
    if self.debug_level > 1:
      self.assertEqual(len(frames), 3)
    else:
      self.assertEqual(len(frames), 2)
    max_frame = os.getenv('PT_XLA_DEBUG_MAX_FRAME', 8)
    # Additonal lines are
    # 1. Python Frame Triggered Execution:
    # 2. ....
    # 3. empty line
    self.assertEqual(len(frames[0].split('\n')), max_frame + 3)
    # second frame will be empty from the post-compilation-analysis
    if self.debug_level > 1:
      self.assertEqual(len(frames[2].split('\n')), max_frame + 3)
    # Check mark_step is the first frame
    self.assertIn('mark_step', frames[0].split('\n')[1])

    open(self.debug_file_name, 'w').close()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
