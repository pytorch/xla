import unittest
from torchax.util import partition, merge


# Helper predicate functions for testing partition
def is_even(n):
  return isinstance(n, int) and n % 2 == 0


def is_positive(n):
  return isinstance(n, (int, float)) and n > 0


def is_string(s):
  return isinstance(s, str)


class TestListUtils(unittest.TestCase):

  # --- Tests for partition ---

  def test_partition_empty_list(self):
    """Test partition with an empty list."""
    self.assertEqual(partition([], is_even), ([], []))

  def test_partition_even_odd(self):
    """Test partitioning numbers into even and odd."""
    nums = [1, 2, 3, 4, 5, 6]
    expected_truthy = [None, 2, None, 4, None, 6]
    expected_falsy = [1, None, 3, None, 5, None]
    self.assertEqual(
        partition(nums, is_even), (expected_truthy, expected_falsy))

  def test_partition_all_true(self):
    """Test partition when the predicate is always true."""
    evens = [2, 4, 6, 8]
    expected_truthy = [2, 4, 6, 8]
    expected_falsy = [None, None, None, None]
    self.assertEqual(
        partition(evens, is_even), (expected_truthy, expected_falsy))

  def test_partition_all_false(self):
    """Test partition when the predicate is always false."""
    odds = [1, 3, 5, 7]
    expected_truthy = [None, None, None, None]
    expected_falsy = [1, 3, 5, 7]
    self.assertEqual(
        partition(odds, is_even), (expected_truthy, expected_falsy))

  def test_partition_mixed_types(self):
    """Test partition with a list of mixed types."""
    mixed = [1, "hello", 2.5, "world", 3, None]
    # Using is_string as the predicate
    expected_truthy = [None, "hello", None, "world", None, None]
    expected_falsy = [1, None, 2.5, None, 3,
                      None]  # Note: None itself is not a string
    self.assertEqual(
        partition(mixed, is_string), (expected_truthy, expected_falsy))

  def test_partition_with_lambda(self):
    """Test partition using a lambda function as the predicate."""
    nums = [-2, -1, 0, 1, 2]
    expected_truthy = [None, None, None, 1, 2]
    expected_falsy = [-2, -1, 0, None, None]
    self.assertEqual(
        partition(nums, lambda x: isinstance(x, int) and x > 0),
        (expected_truthy, expected_falsy))

  # --- Tests for merge ---

  def test_merge_empty_lists(self):
    """Test merge with empty lists."""
    self.assertEqual(merge([], []), [])

  def test_merge_basic(self):
    """Test basic merging with None values in the first list."""
    list1 = [1, None, 3, None, 5]
    list2 = [None, 2, None, 4, None]
    expected = [1, 2, 3, 4, 5]
    self.assertEqual(merge(list1, list2), expected)

  def test_merge_no_none_in_list1(self):
    """Test merge when list1 has no None values."""
    list1 = ['a', 'b', 'c']
    list2 = [1, 2, 3]
    expected = ['a', 'b', 'c']  # Should be identical to list1
    self.assertEqual(merge(list1, list2), expected)

  def test_merge_all_none_in_list1(self):
    """Test merge when list1 contains only None."""
    list1 = [None, None, None]
    list2 = ['x', 'y', 'z']
    expected = ['x', 'y', 'z']  # Should be identical to list2
    self.assertEqual(merge(list1, list2), expected)

  def test_merge_mixed_types(self):
    """Test merge with mixed data types."""
    list1 = [1, None, "hello", None]
    list2 = [None, 2.5, None, True]
    expected = [1, 2.5, "hello", True]
    self.assertEqual(merge(list1, list2), expected)

  def test_merge_unequal_lengths(self):
    """Test that merge raises AssertionError for lists of unequal length."""
    list1 = [1, 2, 3]
    list2 = [4, 5]
    # Use assertRaises as a context manager
    with self.assertRaises(AssertionError) as cm:
      merge(list1, list2)

    list3 = [6, 7]
    list4 = [8, 9, 10]
    with self.assertRaises(AssertionError):
      merge(list3, list4)  # No need to check message again if already checked


if __name__ == '__main__':
  unittest.main()  # For running from command line
