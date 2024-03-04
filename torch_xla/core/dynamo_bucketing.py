import torch
import math
import warnings

import torch_xla.core.xla_model as xm

# Given the input size, return the next power of two for bucketing.
def _next_power_of_two(size: int) -> int:
    if size == 0:
        return 1

    base:int = int(math.log2(size))
    base_two:int = int(math.pow(2, base))
    if base_two == size:
        return base_two

    return int(math.pow(2, base + 1))

# This assumes dynamo has args in the form of:
# (tensor_size_1, tensor_size_2, actual_tensor_1, actual_tensor_2)
# This returns a list of [(tensor_size1, tensor_size_2), (actual_tensor_1, actual_tensor_2)]
def _parse_real_sizes(xla_args: tuple) -> tuple:
    tensor_sizes:list[int] = []
    tensors: list[torch.Tensor] = []

    for arg in xla_args:
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
        elif isinstance(arg, int):
            tensor_sizes.append(arg)
        else:
            warnings.warn("Found an argument to XLA Dynamo Bridge that isn't tensor or arg.")

    return (tensor_sizes, tensors)

def maybe_pad_tensor(tensor: torch.Tensor, padded_size:int) -> torch.Tensor:
    assert len(tensor.size()) == 1, "Only supports single dimension padding"
    original_tensor_size:int = tensor.size()[0]
    bucketed_tensor_size:int = _next_power_of_two(original_tensor_size)

    padding_size:int = bucketed_tensor_size - original_tensor_size
    padding: tuple = (0, padding_size)
    padded_tensor = torch.nn.functional.pad(tensor, padding, "constant")
    return padded_tensor.to(tensor.device)

# Given the input xla_args, if we're passed in real sizes, assume
# that the function is being compiled with dynamic=True. In those cases
# automatically pad the XLA arg sizes.
def maybe_pad_xla_args(xla_args: tuple) -> tuple:
    tensor_sizes, tensors = _parse_real_sizes(xla_args)
    if len(tensor_sizes) == 0:
        return (), tensors

    assert(len(tensor_sizes) == len(tensors))
    padded_tensors = []
    padded_sizes = []
    for i in range(0, len(tensor_sizes)):
        original_tensor_size:int = tensor_sizes[i]
        bucketed_tensor_size:int = _next_power_of_two(original_tensor_size)
        original_tensor: torch.Tensor = tensors[i]
        assert len(original_tensor.size()) == 1, "Only supports single dimension padding"

        padded_tensor = maybe_pad_tensor(original_tensor, bucketed_tensor_size).to(xm.xla_device())

        padded_sizes.append(bucketed_tensor_size)
        padded_tensors.append(padded_tensor)

    return tuple(padded_sizes), tuple(padded_tensors)