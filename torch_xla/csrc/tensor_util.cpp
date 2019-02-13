#include "tensor_util.h"

#include <algorithm>
#include <functional>
#include <list>
#include <numeric>

#include "helpers.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"

namespace torch_xla {
namespace {

bool ShouldUseBF16() {
  int use_fp16 = xla::sys_util::GetEnvInt("XLA_USE_BF16", 0);
  if (use_fp16 != 0) {
    TF_LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_fp16 != 0;
}

xla::PrimitiveType XlaTypeFromTensorType(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return xla::PrimitiveType::F32;
    case at::ScalarType::Byte:
      return xla::PrimitiveType::U8;
    case at::ScalarType::Char:
      return xla::PrimitiveType::S8;
    case at::ScalarType::Short:
      return xla::PrimitiveType::S16;
    case at::ScalarType::Int:
      return xla::PrimitiveType::S32;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    default:
      XLA_ERROR() << "Type not supported: " << scalar_type;
  }
}

// Creates a minor-to-major layout from given dimensions.
xla::Shape MakeTorchTensorLayout(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::PrimitiveType type) {
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
}

// Copies n bytes from source to dest, with different stride values for source
// and destination.
template <typename S, typename D>
void StridedCopy(D* dest, xla::int64 dest_stride, const S* source,
                 xla::int64 source_stride, xla::int64 n) {
  const S* source_top = source + n * source_stride;
  for (; source < source_top; dest += dest_stride, source += source_stride) {
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

// The tensorflow::bfloat16 does not have implicit cast operations, so using
// std::copy() for it, is not going to work.
struct CopyDirect {};
struct CopyCasted {};

template <typename T>
struct NeedCast {
  static constexpr bool value = false;
};
template <>
struct NeedCast<tensorflow::bfloat16> {
  static constexpr bool value = true;
};

template <bool CAST>
struct CopyType {
  using type = CopyDirect;
};
template <>
struct CopyType<true> {
  using type = CopyCasted;
};

template <typename D, typename S>
void CopyData(D* dest, const S* source, xla::int64 n, const CopyDirect&) {
  std::copy(source, source + n, dest);
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, xla::int64 n, const CopyCasted&) {
  // Use strided copy with step 1 since it has the static_cast<> required to
  // convert from/to bfloat16.
  StridedCopy(dest, 1, source, 1, n);
}

std::vector<xla::int64> GetIterationDimensions(const xla::Shape& shape) {
  // We want to favor the most minor dimension as core iteration dimension, as
  // this walks one of the two tensors buffers in a cache friendly fashion.
  // Though, if the most minor dimension is too small, we will end up doing more
  // StridedCopy() iterations in CopyTensors().
  // So we select the most minor dimension, unless one of the other dimensions
  // is more than kMinorDimScale times the most minor one.
  static const xla::int64 kMinorDimScale = 8;
  std::vector<xla::int64> iter_dims =
      xla::util::ToVector<xla::int64>(shape.layout().minor_to_major());
  size_t index = 0;
  xla::int64 scaled_dim_size =
      kMinorDimScale * shape.dimensions(iter_dims[index]);
  for (size_t i = 1; i < iter_dims.size(); ++i) {
    xla::int64 dim = iter_dims[i];
    if (shape.dimensions(dim) > scaled_dim_size) {
      index = i;
      scaled_dim_size = shape.dimensions(dim);
    }
  }
  std::swap(iter_dims[0], iter_dims[index]);
  return iter_dims;
}

template <typename SType, typename DType>
void CopyTensors(const void* src_buffer, const xla::Shape& src_shape,
                 void* dest_buffer, size_t dest_buffer_size,
                 const xla::Shape& dest_shape) {
  XLA_CHECK(xla::ShapeUtil::SameDimensions(src_shape, dest_shape))
      << src_shape << " vs. " << dest_shape;

  xla::int64 total_elements = xla::ShapeUtil::ElementsIn(src_shape);
  XLA_CHECK_EQ(dest_buffer_size, total_elements * sizeof(DType));

  const SType* src_data = reinterpret_cast<const SType*>(src_buffer);
  DType* dest_data = reinterpret_cast<DType*>(dest_buffer);
  if (src_shape.layout().minor_to_major() ==
      dest_shape.layout().minor_to_major()) {
    CopyData<DType, SType>(dest_data, src_data, total_elements,
                           typename CopyType < NeedCast<SType>::value ||
                               NeedCast<DType>::value > ::type());
  } else {
    std::vector<xla::int64> src_strides = ComputeShapeStrides(src_shape);
    std::vector<xla::int64> dest_strides = ComputeShapeStrides(dest_shape);
    std::vector<xla::int64> indices(src_strides.size());
    std::vector<xla::int64> iter_dims = GetIterationDimensions(dest_shape);
    xla::int64 inner_src_stride = src_strides[iter_dims.front()];
    xla::int64 inner_dest_stride = dest_strides[iter_dims.front()];
    xla::int64 n = 0;
    while (n < iter_dims.size()) {
      StridedCopy(dest_data + GetFlatTensorOffset(dest_strides, indices),
                  inner_dest_stride,
                  src_data + GetFlatTensorOffset(src_strides, indices),
                  inner_src_stride, dest_shape.dimensions(iter_dims.front()));
      // Compute the next index. Skip the lower iteration dimension, as we loop
      // over it using the StridedCopy() call above.
      for (n = 1; n < iter_dims.size(); ++n) {
        xla::int64 dim = iter_dims[n];
        indices[dim] += 1;
        if (indices[dim] < dest_shape.dimensions(dim)) {
          break;
        }
        indices[dim] = 0;
      }
    }
  }
}

template <typename SType, typename DType>
void TensorToBuffer(const at::Tensor& tensor, const xla::Shape& dest_shape,
                    void* dest_buffer, size_t dest_buffer_size) {
  at::Tensor contiguous_tensor = tensor.contiguous();
  xla::Shape src_shape = MakeTorchTensorLayout(
      XlaHelpers::I64List(contiguous_tensor.sizes()),
      XlaTypeFromTensorType(contiguous_tensor.type().scalarType()));
  CopyTensors<SType, DType>(contiguous_tensor.data<SType>(), src_shape,
                            dest_buffer, dest_buffer_size, dest_shape);
}

template <typename SType>
void TensorToBufferSType(const at::Tensor& tensor, const xla::Shape& dest_shape,
                         void* dest_buffer, size_t dest_buffer_size) {
  switch (dest_shape.element_type()) {
    case xla::PrimitiveType::BF16:
      TensorToBuffer<SType, tensorflow::bfloat16>(
          tensor, dest_shape, dest_buffer, dest_buffer_size);
      break;
    case xla::PrimitiveType::F32:
      TensorToBuffer<SType, float>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size);
      break;
    case xla::PrimitiveType::U8:
      TensorToBuffer<SType, xla::uint8>(tensor, dest_shape, dest_buffer,
                                        dest_buffer_size);
      break;
    case xla::PrimitiveType::S8:
      TensorToBuffer<SType, xla::int8>(tensor, dest_shape, dest_buffer,
                                       dest_buffer_size);
      break;
    case xla::PrimitiveType::S16:
      TensorToBuffer<SType, xla::int16>(tensor, dest_shape, dest_buffer,
                                        dest_buffer_size);
      break;
    case xla::PrimitiveType::S32:
      TensorToBuffer<SType, xla::int32>(tensor, dest_shape, dest_buffer,
                                        dest_buffer_size);
      break;
    case xla::PrimitiveType::S64:
      TensorToBuffer<SType, xla::int64>(tensor, dest_shape, dest_buffer,
                                        dest_buffer_size);
      break;
    default:
      XLA_ERROR() << "Destination shape type not supported: " << dest_shape;
  }
}

void PopulateTensorBuffer(const at::Tensor& tensor,
                          const xla::Shape& dest_shape, void* dest_buffer,
                          size_t dest_buffer_size) {
  switch (tensor.type().scalarType()) {
    case at::ScalarType::Float:
      TensorToBufferSType<float>(tensor, dest_shape, dest_buffer,
                                 dest_buffer_size);
      break;
    case at::ScalarType::Byte:
      TensorToBufferSType<uint8_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size);
      break;
    case at::ScalarType::Char:
      TensorToBufferSType<int8_t>(tensor, dest_shape, dest_buffer,
                                  dest_buffer_size);
      break;
    case at::ScalarType::Short:
      TensorToBufferSType<int16_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size);
      break;
    case at::ScalarType::Int:
      TensorToBufferSType<int32_t>(tensor, dest_shape, dest_buffer,
                                   dest_buffer_size);
      break;
    case at::ScalarType::Long:
      TensorToBufferSType<int64_t>(tensor, dest_shape, dest_buffer,
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

template <typename SType, typename DType>
at::Tensor XlaLiteralToTensor(const xla::Literal& literal, at::ScalarType atype,
                              xla::PrimitiveType xtype) {
  std::vector<int64_t> dimensions =
      xla::util::ToVector<int64_t>(literal.shape().dimensions());
  xla::Shape torch_shape =
      MakeTorchTensorLayout(literal.shape().dimensions(), xtype);
  xla::int64 total_elements = xla::ShapeUtil::ElementsIn(torch_shape);

  const auto literal_data = literal.data<SType>();
  at::Tensor tensor = at::empty(dimensions, at::TensorOptions(atype));
  CopyTensors<SType, DType>(literal_data.data(), literal.shape(),
                            tensor.data<DType>(),
                            total_elements * sizeof(DType), torch_shape);
  return tensor;
}

}  // namespace

namespace detail {

bool UseBF16() {
  static bool use_fp16 = ShouldUseBF16();
  return use_fp16;
}

}  // namespace detail

std::vector<xla::int64> ComputeShapeStrides(const xla::Shape& shape) {
  std::vector<xla::int64> strides(shape.rank());
  xla::int64 stride = 1;
  for (auto dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }
  return strides;
}

at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal,
                                    at::ScalarType dest_element_type) {
  switch (literal.shape().element_type()) {
    case xla::PrimitiveType::BF16: {
      return XlaLiteralToTensor<tensorflow::bfloat16, float>(
          literal, at::ScalarType::Float, xla::PrimitiveType::F32);
    }
    case xla::PrimitiveType::F32: {
      return XlaLiteralToTensor<float, float>(literal, at::ScalarType::Float,
                                              xla::PrimitiveType::F32);
    }
    case xla::PrimitiveType::U8: {
      return XlaLiteralToTensor<xla::uint8, uint8_t>(
          literal, at::ScalarType::Byte, xla::PrimitiveType::U8);
    }
    case xla::PrimitiveType::S8: {
      return XlaLiteralToTensor<xla::int8, int8_t>(
          literal, at::ScalarType::Char, xla::PrimitiveType::S8);
    }
    case xla::PrimitiveType::S16: {
      return XlaLiteralToTensor<xla::int16, int16_t>(
          literal, at::ScalarType::Short, xla::PrimitiveType::S16);
    }
    case xla::PrimitiveType::S32: {
      switch (dest_element_type) {
        case at::ScalarType::Int: {
          return XlaLiteralToTensor<xla::int32, int32_t>(
              literal, at::ScalarType::Int, xla::PrimitiveType::S32);
        }
        case at::ScalarType::Byte: {
          return XlaLiteralToTensor<xla::int32, uint8_t>(
              literal, at::ScalarType::Byte, xla::PrimitiveType::S32);
        }
        default:
          XLA_ERROR() << "Unexpected destination type: " << dest_element_type;
      }
    }
    case xla::PrimitiveType::S64: {
      return XlaLiteralToTensor<xla::int64, int64_t>(
          literal, at::ScalarType::Long, xla::PrimitiveType::S64);
    }
    default:
      XLA_ERROR() << "Unsupported literal type: " << literal.shape();
  }
}

xla::Shape MakeArrayShapeFromDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::PrimitiveType type, DeviceType device_type) {
  if (dimensions.size() == 4 && device_type == DeviceType::TPU) {
    // Use a TPU-compatible layout for 4D tensors -- batch and feature in minor
    // dimensions (HWCN).
    return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, {0, 1, 3, 2});
  }
  return MakeTorchTensorLayout(dimensions, type);
}

xla::Shape MakeArrayShapeFromDimensions(const at::IntList& dimensions,
                                        xla::PrimitiveType type,
                                        DeviceType device_type) {
  return MakeArrayShapeFromDimensions(XlaHelpers::I64List(dimensions), type,
                                      device_type);
}

std::shared_ptr<xla::ComputationClient::Data> TensorToXlaData(
    const at::Tensor& tensor, const Device& device) {
  return TensorToXlaData(
      tensor, CreateComputationShapeFromTensor(tensor, &device), device);
}

std::vector<std::shared_ptr<xla::ComputationClient::Data>> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  XLA_CHECK_EQ(tensors.size(), devices.size());
  std::vector<xla::ComputationClient::TensorSource> source_tensors;
  for (size_t i = 0; i < tensors.size(); ++i) {
    Device device(devices[i]);
    xla::Shape shape = CreateComputationShapeFromTensor(tensors[i], &device);
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
        dimensions, XlaTypeFromTensorType(tensor.type().scalarType()));
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
      XLA_CHECK(!component_shape.IsTuple()) << shape;
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
    shape_components_with_layout.push_back(MakeArrayShapeFromDimensions(
        xla::util::ToVector<int64_t>(shape_component.dimensions()),
        shape_component.element_type(), device_type));
  }
  return shape.IsTuple()
             ? xla::ShapeUtil::MakeTupleShape(shape_components_with_layout)
             : shape_components_with_layout.front();
}

xla::Shape CreateComputationShapeFromTensor(const at::Tensor& tensor,
                                            const Device* device) {
  if (device == nullptr) {
    device = GetDefaultDevice();
  }
  return MakeArrayShapeFromDimensions(
      tensor.sizes(), MakeXlaPrimitiveType(tensor.type().scalarType(), device),
      device->hw_type);
}

at::ScalarType TensorTypeFromXlaType(xla::PrimitiveType xla_type) {
  switch (xla_type) {
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F32:
      return at::ScalarType::Float;
    case xla::PrimitiveType::U8:
      return at::ScalarType::Byte;
    case xla::PrimitiveType::S8:
      return at::ScalarType::Char;
    case xla::PrimitiveType::S16:
      return at::ScalarType::Short;
    case xla::PrimitiveType::S32:
      return at::ScalarType::Int;
    case xla::PrimitiveType::S64:
      return at::ScalarType::Long;
    default:
      XLA_ERROR() << "XLA type not supported: " << xla_type;
  }
}

xla::PrimitiveType MakeXlaPrimitiveType(at::ScalarType scalar_type,
                                        const Device* device) {
  if (device == nullptr) {
    device = GetDefaultDevice();
  }
  switch (scalar_type) {
    case at::ScalarType::Float:
      // When PyTorch will support native BF16 type, the global configuration
      // can be replaced (or augmented) with the proper mapping.
      return detail::UseBF16() ? xla::PrimitiveType::BF16
                               : xla::PrimitiveType::F32;
    case at::ScalarType::Byte:
      return device->hw_type != DeviceType::TPU ? xla::PrimitiveType::U8
                                                : xla::PrimitiveType::S64;
    case at::ScalarType::Char:
      return device->hw_type != DeviceType::TPU ? xla::PrimitiveType::S8
                                                : xla::PrimitiveType::S64;
    case at::ScalarType::Short:
      return device->hw_type != DeviceType::TPU ? xla::PrimitiveType::S16
                                                : xla::PrimitiveType::S64;
    case at::ScalarType::Int:
      return device->hw_type != DeviceType::TPU ? xla::PrimitiveType::S32
                                                : xla::PrimitiveType::S64;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    default:
      XLA_ERROR() << "Type not supported: " << scalar_type;
  }
}

}  // namespace torch_xla
