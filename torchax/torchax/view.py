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
    """
    Abstract base class for all view operations.
    Defines the interface for applying and updating view transformations.
    """

    def __init__(
        self,
        view_info_type: ViewInfoType = ViewInfoType.INVALID,
    ):
        """
        Initialize a ViewInfo object.

        Args:
            view_info_type: The type of view operation
        """
        self.view_info_type = view_info_type

    @abstractmethod
    def update_tensor(self, new_value: jax.Array, jax_array: jax.Array) -> jax.Array:
        """
        Apply this view transformation to a JAX array and update its value. 

        Args:
            new_value: The new values to set in the view
            parent: The parent array to update

        Returns:
            Updated parent array
        """
        pass

    @abstractmethod
    def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
        """
        Apply this view transformation to a JAX array.

        Args:
            jax_array: The array to transform

        Returns:
            Transformed array
        """
        pass

    @abstractmethod
    def calculate_output_shape(self, source_shape: List[int]) -> List[int]:
        """
        Calculate the resulting shape after applying this view.

        Args:
            source_shape: Original shape before transformation

        Returns:
            Resulting shape after transformation
        """
        pass


class NarrowInfo(ViewInfo):
    """
    Represents a slicing operation on a tensor.
    Handles operations like tensor[1:3, :, 2:5:2].
    """

    def __init__(self, slices: Union[slice, Tuple[slice]]) -> None:
        """
        Args:
            slices: The slice(s) to apply to the tensor.
                E.g. jax_array.at[slices] will return the transformed tensor.
        """
        super().__init__(ViewInfoType.NARROW)
        self.slices = slices

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NarrowInfo):
            return False
        return self.slices == other.slices

    def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
        return jax_array[self.slices]

    def update_tensor(self, new_value: jax.Array, jax_array: jax.Array) -> jax.Array:
        return jax_array.at[self.slices].set(new_value)

    def calculate_output_shape(self, source_shape: List[int]) -> List[int]:
        result_shape = []
        for dim_idx, dim_size in enumerate(source_shape):
            # Create range for the original dimension
            dim_range = range(dim_size)

            # Apply appropriate slice to this dimension
            if isinstance(self.slices, slice) and dim_idx == 0:
                applied_slice = dim_range[self.slices]
            elif isinstance(self.slices, tuple) and dim_idx < len(self.slices):
                if self.slices[dim_idx] is Ellipsis:  # This checks for ...
                    applied_slice = dim_range[:]
                elif self.slices[dim_idx] is None: 
                    applied_slice = dim_range[:]
                else:
                    applied_slice = dim_range[self.slices[dim_idx]]
            else:
                applied_slice = dim_range

            # Determine dimension size in result
            if isinstance(applied_slice, int):
                result_shape.append(1)
            else:
                result_shape.append(len(applied_slice))

        return result_shape


class SelectInfo(ViewInfo):
    """
    Represents a selection operation on a tensor.
    Typically used for indexing operations that select specific elements.
    """

    def __init__(
        self, dim: int = 0, start: int = 0, end: int = 0, stride: int = 0
    ) -> None:
        super().__init__(ViewInfoType.SELECT)
        self.dim: int = dim
        self.start: int = start
        self.end: int = end
        self.stride: int = stride

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelectInfo):
            return False
        return (
            self.dim == other.dim
            and self.start == other.start
            and self.end == other.end
            and self.stride == other.stride
        )

    def transform_tensor(self, jax_array: jax.Array) -> jax.Array:
        raise NotImplementedError("SelectInfo.apply not implemented")

    def update_tensor(self, new_value: jax.Array, jax_array: jax.Array) -> jax.Array:
        raise NotImplementedError("SelectInfo.update not implemented")

    def calculate_output_shape(self, source_shape: List[int]) -> List[int]:
        raise NotImplementedError("SelectInfo.calculate_output_shape not implemented")


class AsStridedInfo(ViewInfo):
    """
    Information for as_strided operations.
    """

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

    def update_tensor(self, new_value: jax.Array, jax_array: jax.Array) -> jax.Array:
        raise NotImplementedError("AsStridedInfo.update not implemented")

    def calculate_output_shape(self, source_shape: List[int]) -> List[int]:
        raise NotImplementedError("AsStridedInfo.calculate_output_shape not implemented")


class DiagonalInfo(ViewInfo):
    """
    Information for diagonal operations.
    Extracts diagonal elements from a tensor.
    """

    def __init__(self, offset: int = 0, dim1: int = 0, dim2: int = 1) -> None:
        """
        Args:
            offset: Offset from the main diagonal
            dim1: First dimension for diagonal extraction
            dim2: Second dimension for diagonal extraction
        """
        super().__init__(ViewInfoType.DIAGONAL)
        self.offset: int = offset
        self.dim1: int = dim1
        self.dim2: int = dim2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiagonalInfo):
            return False
        return (
            self.offset == other.offset
            and self.dim1 == other.dim1
            and self.dim2 == other.dim2
        )

    def apply(self, jax_array: jax.Array) -> jax.Array:
        raise NotImplementedError("DiagonalInfo.apply not implemented")

    def update(self, new_value: jax.Array, parent: jax.Array) -> jax.Array:
        raise NotImplementedError("DiagonalInfo.update not implemented")

    def calculate_output_shape(self, source_shape: List[int]) -> List[int]:
        raise NotImplementedError("DiagonalInfo.calculate_output_shape not implemented")


class View(torch.Tensor):
    """
    A View is a reference to another Tensor or another View,
    with a transformation applied to it.
    """

    @staticmethod
    def __new__(
        cls, parent: Union["torchax.Tensor", "View"], view_info: ViewInfo, env: Any
    ) -> "View":
        """
        Args:
            parent: Parent tensor or view
            view_info: Information about the view transformation
            env: Environment for tensor operations
        """
        shape = view_info.calculate_output_shape(parent.shape)
        print("initizing view with dtype", parent.dtype)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device="meta",
            dtype=parent.dtype,
            requires_grad=False,
        )

    def __init__(
        self, parent: Union["torchax.Tensor", "View"], view_info: ViewInfo, env: Any
    ) -> None:
        super().__init__()
        self.parent = parent
        self.view_info = view_info
        self._env = env

    def get_transformation_chain(self) -> List[ViewInfo]:
        """
        Get all view transformations from the source tensor to this view.
        """
        if isinstance(self.parent, View):
            return self.parent.get_transformation_chain() + [self.view_info]
        else:
            return [self.view_info]

    __torch_function__ = torch._C._disabled_torch_function_impl

    def source_jax(self) -> jax.Array:
        """
        Returns the source tensor.
        """
        if isinstance(self.parent, View):
            return self.parent.source_jax()
        else:
            return self.parent._elem
    
    def replace_source_jax(self, new_value: jax.Array) -> None:
        """
        Update the source tensor with new values.
        """
        if isinstance(self.parent, View):
            self.parent.replace_source_jax(new_value)
        else:
            assert new_value.shape == self.parent._elem.shape
            self.parent._elem = new_value
        
    def torch(self) -> "torchax.Tensor":
        """
        Returns a Torchax tensor representing this view after all transformations
        """
        from torchax.tensor import Tensor

        return Tensor(self.jax(), self._env)

    def update(
        self,
        new_values: Union[jax.Array, "View", "torchax.Tensor"],
    ) -> None:
        """
        Update this view with new values, propagating changes back to source.
        """
        view_infos = self.get_transformation_chain()

        # Get the source JAX array
        source_array = self.source_jax()

        # Get the new value
        from torchax.tensor import Tensor
        if isinstance(new_values, View) or isinstance(new_values, Tensor):
            new_values = new_values.jax()
        assert isinstance(new_values, jax.Array), "Update value must be a JAX array"

        # Apply all view transformations to the source array
        # And store intermediate values
        intermediate_values = [source_array]
        for view_info in view_infos[:-1]:
            intermediate_values.append(view_info.transform_tensor(intermediate_values[-1]))

        # Update the source array with the new value by
        # applying inverse transformations in reverse order
        for view_info, parent_array in zip(
            reversed(view_infos), reversed(intermediate_values)
        ):
            # Apply the inverse transformation to propagate changes back
            new_values = view_info.update_tensor(new_values, parent_array)

        # Update the source tensor with the new values
        self.replace_source_jax(new_values)

    def __getitem__(
        self, indices: Union[int, slice, Tuple[Union[int, slice], ...]]
    ) -> "View":
        """
        Get a slice of this view.

        Args:
            indices: Indices for slicing

        Returns:
            New view representing the slice
        """
        return self.create_sub_view(NarrowInfo(indices))

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Tuple[Any, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        from torchax.tensor import Tensor

        env = None
        for arg in torch_pytree.arg_tree_leaves(*args, **(kwargs or {})):
            if isinstance(arg, Tensor) or isinstance(arg, View):
                env = arg._env
                break

        with env:
            res = func(*args, **(kwargs or {}))
            return res

    def create_sub_view(self, view_info: ViewInfo) -> "View":
        """
        Create a new view that is a child of this view.
        """
        return View(self, view_info, self._env)

    def __str__(self) -> str:
        return f"View({self.torch()})"
    
    def jax(self) -> jax.Array:
        """
        Returns a copy of the source tensor after transformations.
        """
        result = self.source_jax()
        for view_info in self.get_transformation_chain():
            result = view_info.transform_tensor(result)
        return result

    # Use the same string representation for all string conversion methods
    __repr__ = __str__
    __format__ = __str__
