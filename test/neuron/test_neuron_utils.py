import os
import pytest
import unittest
from torch_xla._internal.neuron_utils import *


class NeuronUtilsTest(unittest.TestCase):

  def test_get_visible_cores_list(self):
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1"
    assert (get_visible_cores_list() == [1])
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1,2,3"
    assert (get_visible_cores_list() == [1, 2, 3])
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3"
    assert (get_visible_cores_list() == [1, 2, 3])
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8"
    assert (get_visible_cores_list() == [1, 2, 3, 5, 6, 7, 8])
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1,3,5-8"
    assert (get_visible_cores_list() == [1, 3, 5, 6, 7, 8])
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8,3-5"
    with pytest.raises(ValueError):
      get_visible_cores_list()
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8-5"
    with pytest.raises(ValueError):
      get_visible_cores_list()
    os.environ["NEURON_RT_VISIBLE_CORES"] = "a-b,5-8-5"
    with pytest.raises(Exception):
      get_visible_cores_list()
    os.environ["NEURON_RT_VISIBLE_CORES"] = "a"
    with pytest.raises(Exception):
      get_visible_cores_list()

  def test_remap_visible_cores(self):
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1"
    remap_visible_cores(0, 1)
    assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "1")
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1,2,3"
    remap_visible_cores(1, 3)
    assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "2")
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3"
    remap_visible_cores(2, 3)
    assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "3")
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1-3,5-8"
    remap_visible_cores(5, 7)
    assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "7")
    os.environ["NEURON_RT_VISIBLE_CORES"] = "1,3,5-8"
    remap_visible_cores(5, 6)
    assert (os.environ['NEURON_RT_VISIBLE_CORES'] == "8")
    with pytest.raises(ValueError):
      remap_visible_cores(5, 9)
    with pytest.raises(ValueError):
      remap_visible_cores(6, 6)


if __name__ == "__main__":
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
