from typing import Any, Callable


def partition(original: list[Any],
              func: Callable[[Any], bool]) -> tuple[list[Any], list[Any]]:
  """Partitions elements into two parallel lists based on a predicate function.

  Iterates through the 'original' list, applying 'func' to each element 'a'.
  - If `func(a)` returns True, 'a' is appended to the first list ('truthy')
    and `None` is appended to the second list ('falsy').
  - If `func(a)` returns False, `None` is appended to the first list ('truthy')
    and 'a' is appended to the second list ('falsy').

  The result is two lists of the same length as the 'original' list, acting
  as parallel representations of the partitioned elements, using `None` as
  placeholders.

  This is useful when we want to mark a group of elements as static (via passing
  static_argnums) or donated (via donate_argnums) when combining with jax.jit
  and friends.

  Args:
      original: The list of elements to partition.
      func: A callable (function or lambda) that accepts an element from
            'original' and returns a boolean value (True or False).

  Returns:
      A tuple containing two lists (`truthy`, `falsy`), both of the same
      length as `original`:
      - The first list contains elements `x` where `func(x)` was True, and
        `None` otherwise.
      - The second list contains elements `x` where `func(x)` was False, and
        `None` otherwise.

  Example:
      >>> def is_even(n): return n % 2 == 0
      >>> nums = [1, 2, 3, 4, 5, 6]
      >>> truthy_list, falsy_list = partition(nums, is_even)
      >>> truthy_list
      [None, 2, None, 4, None, 6]
      >>> falsy_list
      [1, None, 3, None, 5, None]
  """
  truthy = []
  falsy = []
  for a in original:
    t, f = (a, None) if func(a) else (None, a)
    truthy.append(t)
    falsy.append(f)
  return truthy, falsy


def merge(list1: list[Any], list2: list[Any]) -> list[Any]:
  """Merges two lists element-wise, prioritizing non-None elements from list1.

  Creates a new list where each element is taken from the corresponding position
  in 'list1', unless that element is None, in which case the element from the
  corresponding position in 'list2' is used. Assumes both lists have the
  same length.

  Invariant: merge(*partion(input_list, predicate)) == input_list for any predicate

  Args:
      list1: The primary list. Its elements are preferred unless they are None.
      list2: The secondary list. Its elements are used as fallbacks when the
              corresponding element in list1 is None.

  Returns:
      A new list representing the merged result.

  Raises:
      AssertionError: If 'list1' and 'list2' do not have the same length.

  Example:
      >>> l1 = [1, None, 3, None]
      >>> l2 = [None, 2, None, 4]
      >>> merge(l1, l2)
      [1, 2, 3, 4]
      >>> l3 = [None, 'b', None]
      >>> l4 = ['a', None, 'c']
      >>> merge(l3, l4)
      ['a', 'b', 'c']
  """
  assert len(list1) == len(list2)
  res = []
  for a, b in zip(list1, list2):
    res.append(b if a is None else a)
  return res
