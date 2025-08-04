import torch
import torch.utils._pytree as torch_pytree
import jax
from enum import Enum
from typing import Union, List, Tuple, Optional, Any, cast
from abc import ABC, abstractmethod

# Reference to original PyTorch native functions
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml


class ViewInfoType(Enum):
  INVALID = 0
  NARROW = 1
  NO_OP = 2
  PERMUTE = 3
  RESHAPE = 4
  RESIZE = 5
  SELECT = 6
  AS_STRIDED = 7
  DIAGONAL = 8


class ViewInfo(ABC):
  """Abstract base class for all view operations.

  This class defines the interface for applying and updating view transformations
  on JAX arrays. Each subclass represents a specific type of view, such as
  a slice, reshape, or permutation.
  """

  def __init__(
      self,
      view_info_type: ViewInfoType = ViewInfoType.INVALID,
  ):
    """Initializes a ViewInfo object.

    Args:
        view_info_type: The type of view operation.
    """
    self.view_info_type = view_info_type

  @abstractmethod
  def update_tensor(self, new_value: jax.Array,
                    jax_array: jax.Array) -> jax.Array:
    """Applies this view transformation to a JAX array and updates its value.

    Args:
        new_value: The new values to set in the view.
        jax_array: The parent array to update.

    Returns:
        The updated array.
    """
    pass

  @abstractmethod
  def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
    """Applies this view transformation to a JAX array.

    Args:
        jax_array: The array to transform.

    Returns:
        The transformed array.
    """
    pass

  @abstractmethod
  def calculate_output_shape(self, source: jax.Array) -> List[int]:
    """Calculates the resulting shape after applying this view.

    Args:
        source: The original JAX array before transformation.

    Returns:
        The resulting shape after transformation.
    """
    pass


class NarrowInfo(ViewInfo):
  """Represents a slicing operation on a tensor.

  This class handles operations like `tensor[1:3, :, 2:5:2]`.
  """

  def __init__(self, slices: Union[slice, Tuple[slice]]) -> None:
    """Initializes a NarrowInfo object.

    Args:
        slices: The slice(s) to apply to the tensor.
                For example, `jax_array.at[slices]` will return the
                transformed tensor.
    """
    super().__init__(ViewInfoType.NARROW)
    self.slices = slices

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, NarrowInfo):
      return False
    return self.slices == other.slices

  def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
    try:
      return jax_array[self.slices]
    except IndexError as e:
      raise IndexError("Invalid slice operation") from e

  def update_tensor(self, new_value: jax.Array,
                    jax_array: jax.Array) -> jax.Array:
    return jax_array.at[self.slices].set(new_value)

  def calculate_output_shape(self, source: jax.Array) -> List[int]:
    return source[self.slices].shape


class SelectInfo(ViewInfo):
  """Represents a selection operation on a tensor.

  This is typically used for indexing operations that select specific elements.
  """

  def __init__(self,
               dim: int = 0,
               start: int = 0,
               end: int = 0,
               stride: int = 0) -> None:
    super().__init__(ViewInfoType.SELECT)
    self.dim: int = dim
    self.start: int = start
    self.end: int = end
    self.stride: int = stride

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, SelectInfo):
      return False
    return (self.dim == other.dim and self.start == other.start and
            self.end == other.end and self.stride == other.stride)

  def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
    raise NotImplementedError("SelectInfo.apply not implemented")

  def update_tensor(self, new_value: jax.Array,
                    jax_array: jax.Array) -> jax.Array:
    raise NotImplementedError("SelectInfo.update not implemented")

  def calculate_output_shape(self, source: jax.Array) -> List[int]:
    raise NotImplementedError(
        "SelectInfo.calculate_output_shape not implemented")


class AsStridedInfo(ViewInfo):
  """Represents an `as_strided` operation on a tensor."""

  def __init__(self, stride: List[int], offset: int = 0) -> None:
    super().__init__(ViewInfoType.AS_STRIDED)
    self.stride: List[int] = stride
    self.offset: int = offset

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, AsStridedInfo):
      return False
    return self.offset == other.offset and self.stride == other.stride

  def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
    raise NotImplementedError("AsStridedInfo.apply not implemented")

  def update_tensor(self, new_value: jax.Array,
                    jax_array: jax.Array) -> jax.Array:
    raise NotImplementedError("AsStridedInfo.update not implemented")

  def calculate_output_shape(self, source: jax.Array) -> List[int]:
    raise NotImplementedError(
        "AsStridedInfo.calculate_output_shape not implemented")


class DiagonalInfo(ViewInfo):
  """Represents a diagonal operation on a tensor.

  This class is used to extract diagonal elements from a tensor.
  """

  def __init__(self, offset: int = 0, dim1: int = 0, dim2: int = 1) -> None:
    """Initializes a DiagonalInfo object.

    Args:
        offset: The offset from the main diagonal.
        dim1: The first dimension for diagonal extraction.
        dim2: The second dimension for diagonal extraction.
    """
    super().__init__(ViewInfoType.DIAGONAL)
    self.offset: int = offset
    self.dim1: int = dim1
    self.dim2: int = dim2

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, DiagonalInfo):
      return False
    return (self.offset == other.offset and self.dim1 == other.dim1 and
            self.dim2 == other.dim2)

  def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
    raise NotImplementedError("DiagonalInfo.apply not implemented")

  def update_tensor(self, new_value: jax.Array,
                    jax_array: jax.Array) -> jax.Array:
    raise NotImplementedError("DiagonalInfo.update not implemented")

  def calculate_output_shape(self, source: jax.Array) -> List[int]:
    raise NotImplementedError(
        "DiagonalInfo.calculate_output_shape not implemented")


class View(torch.Tensor):
  """A `torch.Tensor` subclass that represents a view of another tensor.

  A `View` holds a reference to a parent `torchax.Tensor` or another `View`,
  along with a `ViewInfo` object that describes the transformation to be
  applied. This allows for lazy evaluation of view operations and efficient
  in-place updates.
  """

  @staticmethod
  def __new__(cls, parent: Union["torchax.Tensor", "View"], view_info: ViewInfo,
              env: Any) -> "View":
    """Creates a new `View` object.

    Args:
        parent: The parent tensor or view.
        view_info: Information about the view transformation.
        env: The `torchax` environment.
    """
    shape = view_info.calculate_output_shape(parent.jax())
    return torch.Tensor._make_wrapper_subclass(
        cls,
        shape,
        device="meta",
        dtype=parent.dtype,
        requires_grad=False,
    )

  def __init__(self, parent: Union["torchax.Tensor", "View"],
               view_info: ViewInfo, env: Any) -> None:
    super().__init__()
    self.parent = parent
    self.view_info = view_info
    self._env = env

  def get_transformation_chain(self) -> List[ViewInfo]:
    """Returns the chain of view transformations from the source tensor to this view."""
    if isinstance(self.parent, View):
      transformations = self.parent.get_transformation_chain()
      transformations.append(self.view_info)
      return transformations
    else:
      return [self.view_info]

  __torch_function__ = torch._C._disabled_torch_function_impl

  def source_jax(self) -> jax.Array:
    """Returns the underlying `jax.Array` of the source tensor."""
    if isinstance(self.parent, View):
      return self.parent.source_jax()
    else:
      return self.parent.jax()

  def replace_source_jax(self, new_value: jax.Array) -> None:
    """Updates the source tensor with a new `jax.Array`."""
    if isinstance(self.parent, View):
      self.parent.replace_source_jax(new_value)
    else:
      assert new_value.shape == self.parent._elem.shape
      self.parent._elem = new_value

  def torch(self) -> "torchax.Tensor":
    """Returns a `torchax.Tensor` representing this view after all transformations."""
    from torchax.tensor import Tensor

    return Tensor(self.jax(), self._env)

  def update(
      self,
      new_values: Union[jax.Array, "View", "torchax.Tensor"],
      view_infos: Optional[List[ViewInfo]] = None,
  ) -> None:
    """Updates this view with new values, propagating changes back to the source.

    If `view_infos` is not provided, it will use the transformation chain
    from the source tensor.
    """
    if view_infos is None:
      view_infos = self.get_transformation_chain()

    # Get the source JAX array
    source_array = self.source_jax()

    # Get the new value
    from torchax.tensor import Tensor

    if isinstance(new_values, View) or isinstance(new_values, Tensor):
      new_values = new_values.jax()

    # Apply all view transformations to the source array
    # And store intermediate values
    intermediate_values = [source_array]
    for view_info in view_infos[:-1]:
      intermediate_values.append(
          view_info.transform_tensor(intermediate_values[-1]))

    # TODO: Investigate efficiency of this algorithm
    # Update the source array with the new value by
    # applying inverse transformations in reverse order
    for view_info, parent_array in zip(
        reversed(view_infos), reversed(intermediate_values)):
      # Apply the inverse transformation to propagate changes back
      new_values = view_info.update_tensor(new_values, parent_array)

    # Update the source tensor with the new values
    self.replace_source_jax(new_values)

  @classmethod
  def __torch_dispatch__(
      cls,
      func: Any,
      types: Tuple[Any, ...],
      args: Tuple[Any, ...] = (),
      kwargs: Optional[dict] = None,
  ) -> Any:
    raise AssertionError(
        'torchax Tensors can only do math within the torchax environment.'
        'Please wrap your code with `with torchax.default_env()` or '
        'call torchax.enable_globally() before.')

  def create_sub_view(self, view_info: ViewInfo) -> "View":
    """Creates a new view that is a child of this view."""
    return View(self, view_info, self._env)

  def __str__(self) -> str:
    return f"View({self.torch()})"

  def jax(self) -> jax.Array:
    """Returns a copy of the source tensor after all transformations have been applied."""
    result = self.source_jax()
    for view_info in self.get_transformation_chain():
      result = view_info.transform_tensor(result)
    return result

  def __setitem__(self, indexes, val):
    view_infos = self.get_transformation_chain() + [NarrowInfo(indexes)]
    self.update(view_infos=view_infos, new_values=val)

  def dim(self):
    return self.ndim

  @property
  def device(self):
    return torch.device("jax:0")

  @property
  def jax_device(self):
    return self.jax().device

  @property
  def ndim(self):
    return len(self.shape)

  __repr__ = __str__
