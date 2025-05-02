"""torch_xla.backends is intended to be moved to torch.backends.xla"""

# See https://github.com/pytorch/pytorch/blob/main/torch/backends/mps/__init__.py
# for an example of how a backend is implemented in PyTorch.

# Literal is availabe from Python 3.8,
# matching the Python versions for PyTorch and PyTorchXLA.
from typing import Final, Literal, TypeAlias

import torch_xla

__all__ = ['set_mat_mul_precision', 'get_mat_mul_precision']

# Valid values for get_mat_mul_precision/set_mat_mul_precision
# Note: it is idiomatic to PyTorch to use strings rather than enums.
# See https://github.com/pytorch/pytorch/blob/v2.7.0/torch/backends/cpu/__init__.py#L9

_DEFAULT: Final = 'default'
_HIGH: Final = 'high'
_HIGHEST: Final = 'highest'

# Use of variables with Final typehint instead of literals is valid.
_PrecisionType: TypeAlias = Literal[_DEFAULT, _HIGH, _HIGHEST] # pyright: ignore[reportInvalidTypeForm]

# Some of this description adapted from Jax documentation.
def set_mat_mul_precision(precision: _PrecisionType) -> None:
    r"""Control the default matmul and conv precision for 32bit inputs.

    Some platforms, like TPU, offer configurable precision levels for
    matrix multiplication and convolution computations,
    trading off accuracy for speed.

    This option can be used to control the default precision level for
    computations involved in matrix multiplication and convolution on
    32bit inputs. The levels describe the precision at
    which scalar products are computed.

    On a TPU:
    * `default` is the fastest and least precise, essentially
    downcasting an FP32 to BF16 before multiplying.

    * `high` breaks the FP32 inputs into upper and lower bits and
    computes the product of the pair of upper bits, and the two pairs of
    upper and lower bits. The pair of lower bits is ignored. Since the BF16
    format has 9 bits of precision (7 bits of mantissa plus one implicit
    leading one bit plus one sign bit), this delivers approximately
    18 bits of precision, more than the TensorFloat32 (10 bits)
    but less than a full FP32 (23 bits).
    It requires three passes over the data.

    * `highest` is the most precise, and the slowest.
    It further breaks up each input into a triple of upper, middle,
    and lower bits and effectively calculates
    23 bits of precision. This is the most precise option, but also
    requires six passes.

    Args:
        precision (str): The precision to set for matrix multiplication. Must be
            one of 'default', 'high', or 'highest'.
    """
    if precision not in [_DEFAULT, _HIGH, _HIGHEST]:
        raise ValueError(f"Invalid precision: {precision}. "
                         "Must be one of 'float32', 'bfloat16', or 'float16'.")

    torch_xla._XLAC._xla_set_mat_mul_precision(precision)

def get_mat_mul_precision() -> _PrecisionType:
    r"""Get the current mat mul precision for 32bit inputs.

    Returns:
        str: The current precision setting for matrix multiplication,
            one of 'default', 'high', or 'highest'.
    """
    precision = torch_xla._XLAC._xla_get_mat_mul_precision()
    assert precision in [_DEFAULT, _HIGH, _HIGHEST], (
        f"Invalid precision: {precision}. "
        "Must be one of 'float32', 'bfloat16', or 'float16'.")
    return precision
