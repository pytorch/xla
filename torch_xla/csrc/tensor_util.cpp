#include "tensor_util.h"

#include <algorithm>
#include <functional>
#include <list>
#include <numeric>

#include "helpers.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

namespace torch_xla {
namespace {

// Creates a minor-to-major layout from given dimensions.
xla::Shape MakeTorchTensorLayout(const std::vector<xla::int64>& dimensions,
                                 xla::PrimitiveType type) {
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
}

// Copies n bytes from source to dest, with different stride values for source
// and destination.
template <typename S, typename D>
void StridedCopy(D* dest, xla::int64 dest_stride, const S* source,
                 xla::int64 source_stride, xla::int64 n) {
  for (; n > 0; --n, dest += dest_stride, source += source_stride) {
    *dest = static_cast<D>(*source);
  }
}

// Computes the offset of the value at a given index, assuming a contiguous/flat
// tensor data representation.
template <typename S>
xla::int64 GetFlatTensorOffset(const S& strides,
                               const std::vector<xla::int64>& indices) {
  xla::int64 base = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    base += indices[i] * strides[i];
  }
  return base;
}

std::vector<xla::int64> GetXlaStrides(const xla::Shape& shape) {
  std::vector<xla::int64> strides(shape.rank());
  xla::int64 stride = 1;
  for (auto dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }
  return strides;
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, xla::int64 n) {
  StridedCopy(dest, 1, source, 1, n);
}

template <>
void CopyData<float, float>(float* dest, const float* source, xla::int64 n) {
  std::copy(source, source + n, dest);
}

template <>
void CopyData<xla::int64, int64_t>(xla::int64* dest, const int64_t* source,
                                   xla::int64 n) {
  std::copy(source, source + n, dest);
}

std::vector<xla::int64> GetIterationDimensions(const xla::Shape& shape) {
  // Return the most minor dimension order, to iterate the literal memory in a
  // cache friendly way.
  // Another strategy could be to return the higher value dimension first, to
  // reduce the number of outer loops in TensorToBuffer(), but that leads to
  // StridedCopy() calls in which both source and destination are jumping off
  // memory locations.
  return std::vector<xla::int64>(shape.layout().minor_to_major().begin(),
                                 shape.layout().minor_to_major().end());
}

template <typename AtenNative, typename XlaNative>
void TensorToBuffer(const at::Tensor& tensor, const xla::Shape& shape,
                    void* dest_buffer, size_t dest_buffer_size) {
  const at::Tensor& contiguous_tensor = tensor.contiguous();
  auto contiguous_ptr = contiguous_tensor.data<AtenNative>();
  const auto& tensor_sizes = contiguous_tensor.sizes();
  XLA_CHECK_EQ(tensor_sizes.size(), shape.rank());
  xla::int64 total_elements =
      std::accumulate(tensor_sizes.begin(), tensor_sizes.end(), 1,
                      std::multiplies<xla::int64>());
  XLA_CHECK_EQ(dest_buffer_size, total_elements * sizeof(XlaNative));
  XlaNative* literal_data = reinterpret_cast<XlaNative*>(dest_buffer);
  if (total_elements == 1 ||
      xla::LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    // The Torch tensor is array layout, and so is the shape. We can issue a
    // fast copy of the elements.
    CopyData<XlaNative, AtenNative>(literal_data, contiguous_ptr,
                                    total_elements);
  } else {
    const auto& tensor_strides = contiguous_tensor.strides();
    const auto& xla_tensor_strides = GetXlaStrides(shape);
    std::vector<xla::int64> indices(tensor_sizes.size());
    std::vector<xla::int64> iter_dims = GetIterationDimensions(shape);
    xla::int64 n = 0;
    while (n < tensor_sizes.size()) {
      StridedCopy(
          literal_data + GetFlatTensorOffset(xla_tensor_strides, indices),
          xla_tensor_strides[iter_dims.front()],
          contiguous_ptr + GetFlatTensorOffset(tensor_strides, indices),
          tensor_strides[iter_dims.front()],
          shape.dimensions(iter_dims.front()));
      // Compute the next index. Skip the lower iteration dimension, as we loop
      // over it using the StridedCopy() call above.
      for (n = 1; n < iter_dims.size(); ++n) {
        xla::int64 dim = iter_dims[n];
        indices[dim] += 1;
        if (indices[dim] < shape.dimensions(dim)) {
          break;
        }
        indices[dim] = 0;
      }
    }
  }
}

void PopulateTensorBuffer(const at::Tensor& tensor, const xla::Shape& shape,
                          void* dest_buffer, size_t dest_buffer_size) {
  switch (tensor.type().scalarType()) {
    case at::ScalarType::Float:
      if (shape.element_type() == xla::PrimitiveType::BF16) {
        TensorToBuffer<float, tensorflow::bfloat16>(tensor, shape, dest_buffer,
                                                    dest_buffer_size);
      } else {
        TensorToBuffer<float, float>(tensor, shape, dest_buffer,
                                     dest_buffer_size);
      }
      break;
    case at::ScalarType::Long:
      TensorToBuffer<int64_t, xla::int64>(tensor, shape, dest_buffer,
                                          dest_buffer_size);
      break;
    default:
      XLA_ERROR() << "Tensor type not supported: " << tensor.type();
  }
}

std::shared_ptr<xla::ComputationClient::Data> TensorToXlaData(
    const at::Tensor& tensor, const xla::Shape& shape, const Device& device) {
  auto populate_fn =
      [&](const xla::ComputationClient::TensorSource& source_tensor,
          void* dest_buffer, size_t dest_buffer_size) {
        PopulateTensorBuffer(tensor, source_tensor.shape, dest_buffer,
                             dest_buffer_size);
      };

  std::vector<xla::ComputationClient::TensorSource> source_tensors;
  source_tensors.emplace_back(shape, device.ToString(), std::move(populate_fn));

  auto handles =
      xla::ComputationClient::Get()->TransferToServer(source_tensors);
  XLA_CHECK_EQ(handles.size(), 1);
  return std::move(handles.front());
}

}  // namespace

at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal) {
  const xla::Literal* literal_ptr = &literal;
  xla::Literal f32_literal;
  if (literal_ptr->shape().element_type() == xla::PrimitiveType::BF16) {
    // If ever PyTorch will support BF16, remove this cast to F32.
    f32_literal = xla::LiteralUtil::ConvertBF16ToF32(*literal_ptr);
    literal_ptr = &f32_literal;
  }
  std::vector<int64_t> dimensions;
  for (const auto result_dimension : literal_ptr->shape().dimensions()) {
    dimensions.push_back(result_dimension);
  }
  xla::Shape torch_shape = MakeTorchTensorLayout(
      XlaHelpers::I64List(dimensions), literal_ptr->shape().element_type());
  xla::Literal literal_with_torch_layout;
  if (!xla::ShapeUtil::Equal(literal_ptr->shape(), torch_shape)) {
    literal_with_torch_layout = literal_ptr->Relayout(torch_shape);
    literal_ptr = &literal_with_torch_layout;
  }
  switch (literal_ptr->shape().element_type()) {
    case xla::PrimitiveType::F32: {
      const auto result_slice = literal_ptr->data<float>();
      at::Tensor result_tensor =
          at::empty(dimensions, at::TensorOptions(at::kFloat));
      std::copy(result_slice.begin(), result_slice.end(),
                result_tensor.data<float>());
      return result_tensor;
    }
    case xla::PrimitiveType::S64: {
      const auto result_slice = literal_ptr->data<xla::int64>();
      at::Tensor result_tensor =
          at::empty(dimensions, at::TensorOptions(at::kLong));
      std::copy(result_slice.begin(), result_slice.end(),
                result_tensor.data<int64_t>());
      return result_tensor;
    }
    default:
      XLA_ERROR() << "Unsupported literal type: " << literal_ptr->shape();
  }
}

xla::Shape MakeArrayShapeFromDimensions(const at::IntList& tensor_dimensions,
                                        xla::PrimitiveType type,
                                        DeviceType device_type) {
  const auto dimensions = XlaHelpers::I64List(tensor_dimensions);
  if (dimensions.size() == 4 && device_type == DeviceType::TPU) {
    // Use a TPU-compatible layout for 4D tensors -- batch and feature in minor
    // dimensions (HWCN).
    return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, {0, 1, 3, 2});
  }
  return MakeTorchTensorLayout(dimensions, type);
}

std::shared_ptr<xla::ComputationClient::Data> TensorToXlaData(
    const at::Tensor& tensor, const Device& device) {
  return TensorToXlaData(
      tensor,
      MakeArrayShapeFromDimensions(
          tensor.sizes(),
          XlaHelpers::MakeXlaPrimitiveType(tensor.type().scalarType()),
          device.hw_type),
      device);
}

std::vector<std::shared_ptr<xla::ComputationClient::Data>> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  XLA_CHECK_EQ(tensors.size(), devices.size());
  std::vector<xla::ComputationClient::TensorSource> source_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    Device device(devices[i]);
    xla::Shape shape = MakeArrayShapeFromDimensions(
        tensors[i].sizes(),
        XlaHelpers::MakeXlaPrimitiveType(tensors[i].type().scalarType()),
        device.hw_type);
    auto populate_fn =
        [&, i](const xla::ComputationClient::TensorSource& source_tensor,
               void* dest_buffer, size_t dest_buffer_size) {
          PopulateTensorBuffer(tensors[i], source_tensor.shape, dest_buffer,
                               dest_buffer_size);
        };
    source_tensors.emplace_back(std::move(shape), devices[i],
                                std::move(populate_fn));
  }
  return xla::ComputationClient::Get()->TransferToServer(source_tensors);
}

xla::Literal GetTensorLiteral(const at::Tensor& tensor,
                              const xla::Shape* shape) {
  xla::Shape computed_shape;
  if (shape == nullptr) {
    auto dimensions = XlaHelpers::I64List(tensor.sizes());
    computed_shape = MakeTorchTensorLayout(
        dimensions,
        XlaHelpers::MakeXlaPrimitiveType(tensor.type().scalarType()));
    shape = &computed_shape;
  }
  xla::Literal literal(*shape);
  PopulateTensorBuffer(tensor, *shape, literal.untyped_data(),
                       literal.size_bytes());
  return literal;
}

std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape) {
  std::vector<xla::Shape> component_shapes;
  if (shape.IsTuple()) {
    for (const xla::Shape& component_shape : shape.tuple_shapes()) {
      XLA_CHECK(!component_shape.IsTuple());
      component_shapes.push_back(component_shape);
    }
  } else {
    component_shapes.push_back(shape);
  }
  return component_shapes;
}

xla::Shape MakeShapeWithDeviceLayout(const xla::Shape& shape,
                                     DeviceType device_type) {
  std::vector<xla::Shape> shape_components = GetComponentShapes(shape);
  std::vector<xla::Shape> shape_components_with_layout;
  XLA_CHECK(!shape_components.empty());
  for (const auto& shape_component : shape_components) {
    std::vector<int64_t> shape_component_dimensions(
        shape_component.dimensions().begin(),
        shape_component.dimensions().end());
    shape_components_with_layout.push_back(MakeArrayShapeFromDimensions(
        shape_component_dimensions, shape_component.element_type(),
        device_type));
  }
  return shape_components_with_layout.size() > 1
             ? xla::ShapeUtil::MakeTupleShape(shape_components_with_layout)
             : shape_components_with_layout.front();
}

}  // namespace torch_xla
