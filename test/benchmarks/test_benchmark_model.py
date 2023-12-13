import unittest

from benchmark_model import BenchmarkModel


class BenchmarkModelTest(unittest.TestCase):

  def test_to_dict(self):
    bm = BenchmarkModel("torchbench or other", "super_deep_model",
                        "placeholder")
    actual = bm.to_dict()
    self.assertEqual(2, len(actual))
    self.assertEqual("torchbench or other", actual["suite_name"])
    self.assertEqual("super_deep_model", actual["model_name"])


if __name__ == '__main__':
  unittest.main()
