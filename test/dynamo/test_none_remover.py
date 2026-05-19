import unittest
import sys

import torch_xla
from torch_xla._dynamo.dynamo_bridge import NoneRemover


class TestNoneRemover(unittest.TestCase):

  def test_remove_nones_list(self):
    test_list = ["foo", None, "bar"]
    none_remover = NoneRemover()
    test_list = none_remover.remove_nones(test_list)
    self.assertEqual(test_list, ["foo", "bar"])
    self.assertEqual(none_remover.none_poslist, [1])

  def test_remove_nones_tuple(self):
    test_tuple = ("foo", None, "bar", None, "baz")
    none_remover = NoneRemover()
    test_tuple = none_remover.remove_nones(test_tuple)
    self.assertEqual(test_tuple, ("foo", "bar", "baz"))
    self.assertEqual(none_remover.none_poslist, [1, 3])

  def test_add_nones_list(self):
    test_list = ["foo", "bar"]
    none_remover = NoneRemover()
    none_remover.none_poslist = [1]
    none_remover.add_nones(test_list)
    self.assertEqual(test_list, ["foo", None, "bar"])

  def test_add_nones_tuple(self):
    input_tuple = ("foo", "bar")
    none_remover = NoneRemover()
    none_remover.none_poslist = [1]
    recovered_tuple = none_remover.add_nones(input_tuple)
    self.assertEqual(recovered_tuple, ("foo", None, "bar"))

  def test_add_nones_to_list_from_tuple_removal(self):
    # Simulate removing from tuple and adding to list
    original_tuple = ("foo", None, "bar", None, "baz")
    none_remover = NoneRemover()
    processed_tuple = none_remover.remove_nones(original_tuple)
    self.assertEqual(processed_tuple, ("foo", "bar", "baz"))

    # Now add nones back to a list representation
    recovered_list = list(processed_tuple)
    none_remover.add_nones(recovered_list)
    self.assertEqual(recovered_list, ["foo", None, "bar", None, "baz"])

  def test_add_nones_to_tuple_from_list_removal(self):
    # Simulate removing from list and adding to tuple
    original_list = ["foo", None, "bar", None, "baz"]
    none_remover = NoneRemover()
    processed_list_ref = none_remover.remove_nones(original_list)
    self.assertEqual(original_list, ["foo", "bar", "baz"])

    # Now add nones back to a tuple representation
    recovered_list = tuple(original_list)
    recovered_tuple = none_remover.add_nones(recovered_list)
    self.assertEqual(recovered_tuple, ("foo", None, "bar", None, "baz"))

  def test_empty_list(self):
    test_list = []
    none_remover = NoneRemover()
    processed_list = none_remover.remove_nones(test_list)
    self.assertEqual(processed_list, [])
    self.assertEqual(none_remover.none_poslist, [])
    none_remover.add_nones(processed_list)
    self.assertEqual(processed_list, [])

  def test_empty_tuple(self):
    test_tuple = ()
    none_remover = NoneRemover()
    processed_tuple = none_remover.remove_nones(test_tuple)
    self.assertEqual(processed_tuple, ())
    self.assertEqual(none_remover.none_poslist, [])
    recovered_tuple = none_remover.add_nones(processed_tuple)
    self.assertEqual(recovered_tuple, ())

  def test_list_no_nones(self):
    test_list = ["foo", "bar", "baz"]
    none_remover = NoneRemover()
    processed_list = none_remover.remove_nones(test_list)
    self.assertEqual(processed_list, ["foo", "bar", "baz"])
    self.assertEqual(none_remover.none_poslist, [])
    none_remover.add_nones(processed_list)
    self.assertEqual(processed_list, ["foo", "bar", "baz"])

  def test_tuple_no_nones(self):
    test_tuple = ("foo", "bar", "baz")
    none_remover = NoneRemover()
    processed_tuple = none_remover.remove_nones(test_tuple)
    self.assertEqual(processed_tuple, ("foo", "bar", "baz"))
    self.assertEqual(none_remover.none_poslist, [])
    recovered_tuple = none_remover.add_nones(processed_tuple)
    self.assertEqual(recovered_tuple, ("foo", "bar", "baz"))

  def test_list_all_nones(self):
    test_list = [None, None, None]
    none_remover = NoneRemover()
    processed_list = none_remover.remove_nones(test_list)
    self.assertEqual(processed_list, [])
    self.assertEqual(none_remover.none_poslist, [0, 1, 2])
    none_remover.add_nones(processed_list)
    self.assertEqual(processed_list, [None, None, None])

  def test_tuple_all_nones(self):
    test_tuple = (None, None, None)
    none_remover = NoneRemover()
    processed_tuple = none_remover.remove_nones(test_tuple)
    self.assertEqual(processed_tuple, ())
    self.assertEqual(none_remover.none_poslist, [0, 1, 2])
    recovered_tuple = none_remover.add_nones(processed_tuple)
    self.assertEqual(recovered_tuple, (None, None, None))


if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
