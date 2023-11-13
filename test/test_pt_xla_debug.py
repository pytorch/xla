import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
import unittest


def check_env_flag(name, default=''):
  return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def extract_execution_cause(lines):
  causes = []
  for i in range(len(lines)):
    if 'Execution Cause' in lines[i].decode():
      causes.append(lines[i + 1].decode())
  return causes


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
      causes = extract_execution_cause(lines)
    self.assertEqual(len(causes), 1)
    self.assertIn('user mark_step', causes[0])
    open(self.debug_file_name, 'w').close()

  def test_step_trace(self):
    device = xm.xla_device()
    with xp.StepTrace('train_pt_xla_debug'):
      t1 = torch.randn(2, 2, device=device)
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
    self.assertEqual(len(causes), 1)
    self.assertIn('mark_step when exiting a profiler StepTrace region',
                  causes[0])
    open(self.debug_file_name, 'w').close()

  def test_dynamo(self):
    device = xm.xla_device()
    t1 = torch.randn(2, 2, device=device)

    def toy_program(t1):
      return t1 + t1

    compiled = torch.compile(toy_program, backend="openxla")
    res = compiled(t1)
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
    self.assertEqual(len(causes), 4)
    self.assertIn('mark_step when dynamo processing input graphs', causes[0])
    self.assertIn('mark_step when dynamo processing input graphs', causes[1])
    self.assertIn('dynamo is compiling a FX graph to HLO', causes[2])
    self.assertIn('dynamo is executing a compiled program', causes[3])
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
      pass

    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
    self.assertEqual(len(causes), batch_size + 2)
    for cause in causes:
      self.assertIn('mark_step in parallel loader at step end', cause)
    open(self.debug_file_name, 'w').close()

  def test_print(self):
    device = xm.xla_device()
    t1 = torch.randn(2, 2, device=device)
    print(t1)
    with open(self.debug_file_name, 'rb') as f:
      lines = f.readlines()
      causes = extract_execution_cause(lines)
    self.assertEqual(len(causes), 1)
    self.assertIn('user code trying to access tensor value', causes[0])
    open(self.debug_file_name, 'w').close()


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
