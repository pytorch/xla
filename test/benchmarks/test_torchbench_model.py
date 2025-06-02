import unittest

from benchmarks.torchbench_model import TorchBenchModel


class MockExperiment:

  def __init__(self, accelerator, test):
    self.accelerator = accelerator
    self.test = "train"


class TorchBenchModelTest(unittest.TestCase):

  def test_do_not_use_amp_on_cpu_and_warns(self):
    experiment = MockExperiment("cpu", "train")
    model = TorchBenchModel("torchbench or other", "super_deep_model",
                            experiment)
    with self.assertLogs('benchmarks.torchbench_model', level='WARNING') as cm:
      use_amp = model.use_amp()
      self.assertEqual(len(cm.output), 1)
      self.assertIn("AMP is not used", cm.output[0])
      self.assertFalse(use_amp)

  def test_use_amp_on_cuda(self):
    experiment = MockExperiment("cuda", "train")
    model = TorchBenchModel("torchbench or other", "super_deep_model",
                            experiment)
    self.assertTrue(model.use_amp())


if __name__ == '__main__':
  unittest.main()
