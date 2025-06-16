import unittest
import torch
from torch_xla.core import xla_builder


class PythonBackendTest(unittest.TestCase):

  def test_flatten_func(self):

    def func_that_takes_dict_and_list(nested_dict, nested_list):
      return nested_dict['a']['b'] + nested_list[2][3]

    dict1 = {'a': {'b': 5}}
    list1 = [0, 0, [0, 0, 0, 5]]

    expected = func_that_takes_dict_and_list(dict1, list1)

    flattened = xla_builder.FlattenedInputFunc(func_that_takes_dict_and_list)

    actual = flattened.postprocess(
        flattened.flat_call(flattened.preprocess((dict1, list1))))

    self.assertEqual(expected, actual)

    dict2 = {'a': {'b': 5, 'c': 7}}
    list2 = [0, 0, [0, 0, 0, 5, 6]]

    expected2 = func_that_takes_dict_and_list(dict1, list1)
    actual2 = flattened.postprocess(
        flattened.flat_call(flattened.preprocess((dict2, list2))))
    self.assertEqual(expected2, actual2)

  def test_xla_callable(self):

    def add(a, b):
      return xla_builder.Op.sin(a) + b

    xla_func = xla_builder.XlaCallable(add)

    a = torch.randn(2, 2, device='xla')
    b = torch.randn(2, 2, device='xla')
    res = xla_func(a, b)

    expected = torch.sin(a.cpu()) + b.cpu()

    self.assertTrue(torch.allclose(expected, res.cpu()))


if __name__ == '__main__':
  unittest.main()
