import torch
import torch_xla

from torch_xla.experimental.scan import scan


def scan_example_cumsum():
  """
  This example uses the `scan` function to compute the cumulative sum of a tensor.
  """

  # 1) Define a combine function that takes in the accumulated sum and the next element,
  #    and returns the new accumulated sum. We return two values, one is the "carry" that
  #    will be passed to the next iteration of this function call, and the other is the
  #    "output" that will be stacked into the final result.
  def cumsum(accumulated, element):
    accumulated += element
    return accumulated, accumulated

  # 2) Define an initial carry and the input tensor.
  init_sum = torch.tensor([0.0], device='xla')
  xs = torch.tensor([1.0, 2.0, 3.0], device='xla')
  torch_xla.sync()

  # 3) Call `scan` with our combine function, initial carry, and input tensor.
  final, result = scan(cumsum, init_sum, xs)
  torch_xla.sync()

  print("Final sum:", final)
  print("History of sums", result)


def scan_example_pytree():
  """
  This example uses the `scan` function to compute a running mean.

  It demonstrates using PyTrees as inputs and outputs, in particular, dictionaries.
  """
  # 1) Define an initial carry as a dictionary with two leaves:
  #    - 'sum' to accumulate the sum of all seen values
  #    - 'count' to count how many values have been seen
  carry = {
      'sum': torch.tensor([0.0], device='xla'),
      'count': torch.tensor([0.0], device='xla')
  }

  # 2) Define our input PyTree, which in this case is just a dictionary with one leaf:
  #    - 'values' is a 1D tensor representing data points we want to scan over.
  xs = {'values': torch.arange(1, 6, dtype=torch.float32, device='xla')}

  # Here, xs['values'] has shape [5]. The `scan` function will automatically slice
  # out one element (shape []) each iteration.

  # 3) Define our function (akin to a "step" function in jax.lax.scan). It:
  #    - takes in the current carry and the current slice of xs,
  #    - updates the sum/count in the carry,
  #    - computes a new output (the running mean),
  #    - returns the updated carry and that output.
  def fn(carry_dict, x_dict):
    new_sum = carry_dict['sum'] + x_dict['values']
    new_count = carry_dict['count'] + 1.0
    new_carry = {'sum': new_sum, 'count': new_count}
    running_mean = new_sum / new_count
    return new_carry, running_mean

  # 4) Call `scan` with our step function, initial carry, and input dictionary.
  final_carry, means_over_time = scan(fn, carry, xs)

  # 5) `final_carry` contains the final sum/count, while `means_over_time` is
  #    a 1D tensor with the running mean at each step.
  print("Final carry:", final_carry)
  print("Means over time:", means_over_time)


if __name__ == "__main__":
  for example in [
      scan_example_cumsum,
      scan_example_pytree,
  ]:
    print(f"\nRunning example: {example.__name__}", flush=True)
    example()
    print(flush=True)
