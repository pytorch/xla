import unittest

from benchmark_experiment import BenchmarkExperiment


class BenchmarkExperimentTest(unittest.TestCase):

  def test_to_dict(self):
    be = BenchmarkExperiment("cpu", "PJRT", "some xla_flags", "openxla", None,
                             False, "train", "123", False)
    actual = be.to_dict()
    self.assertEqual(9, len(actual))
    self.assertEqual("cpu", actual["accelerator"])
    self.assertTrue("accelerator_model" in actual)
    self.assertEqual("PJRT", actual["xla"])
    self.assertEqual("some xla_flags", actual["xla_flags"])
    self.assertEqual("openxla", actual["dynamo"])
    self.assertEqual(None, actual["torch_xla2"])
    self.assertEqual(False, actual["keep_model_data_on_cuda"])
    self.assertEqual("train", actual["test"])
    self.assertEqual("123", actual["batch_size"])
    self.assertEqual(False, actual["enable_functionalization"])


if __name__ == '__main__':
  unittest.main()
