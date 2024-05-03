#include "torch_xla/csrc/dl_convertor.h"

#include "absl/types/span.h"
#include <ATen/DLConvertor.h>

#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/unwrap_data.h"
#include "torch_xla/csrc/tensor_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/status.h"

namespace torch_xla {
namespace {

std::shared_ptr<runtime::ComputationClient::Data> get_data_handle(const at::Tensor& input) {
  XLATensorPtr xtensor = bridge::GetXlaTensor(input);
  XLA_CHECK(xtensor) << "The input has to be an XLA tensor.";
  if (xtensor->CurrentDataHandle() != nullptr) {
    return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(xtensor->CurrentDataHandle());
  } else if (xtensor->CurrentIrValue().node != nullptr) {
    DeviceData* device_data =
        DeviceData::Cast(xtensor->CurrentIrValue().node.get());
    if (device_data != nullptr) {
      return UnwrapXlaData(device_data->data());
    }
    TF_VLOG(4) << "The xla tensor has IR value but does not have device data.";
  }
  return nullptr;
}

struct TorchXLADLMTensor {
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;
};

void TorchXLADLMTensorDeleter(DLManagedTensor* t) {
  if (t) {
    delete static_cast<TorchXLADLMTensor*>(t->manager_ctx);
  }
}

DLDeviceType DLDeviceTypeForDevice(const xla::PjRtDevice& device) {
  if (device.client()->platform_id() == xla::CpuId()) {
    return DLDeviceType::kDLCPU;
  } else if (device.client()->platform_id() == xla::CudaId()) {
    return DLDeviceType::kDLCUDA;
  } else if (device.client()->platform_id() == xla::RocmId()) {
    return DLDeviceType::kDLROCM;
  }
  XLA_ERROR() << "Device " << device.DebugString() << " cannot be used as a DLPack device.";
}

DLDevice DLDeviceForDevice(const xla::PjRtDevice& device) {
  DLDevice dlDevice;
  dlDevice.device_type = DLDeviceTypeForDevice(device);
  dlDevice.device_id = device.local_hardware_id();
  return dlDevice;
}

DLDataType PrimitiveTypeToDLDataType(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::S8:
      return DLDataType{kDLInt, 8, 1};
    case xla::PrimitiveType::S16:
      return DLDataType{kDLInt, 16, 1};
    case xla::PrimitiveType::S32:
      return DLDataType{kDLInt, 32, 1};
    case xla::PrimitiveType::S64:
      return DLDataType{kDLInt, 64, 1};
    case xla::PrimitiveType::U8:
      return DLDataType{kDLUInt, 8, 1};
    case xla::PrimitiveType::U16:
      return DLDataType{kDLUInt, 16, 1};
    case xla::PrimitiveType::U32:
      return DLDataType{kDLUInt, 32, 1};
    case xla::PrimitiveType::U64:
      return DLDataType{kDLUInt, 64, 1};
    case xla::PrimitiveType::F16:
      return DLDataType{kDLFloat, 16, 1};
    case xla::PrimitiveType::F32:
      return DLDataType{kDLFloat, 32, 1};
    case xla::PrimitiveType::F64:
      return DLDataType{kDLFloat, 64, 1};
    case xla::PrimitiveType::BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case xla::PrimitiveType::PRED:
      return DLDataType{kDLBool, 8, 1};
    case xla::PrimitiveType::C64:
      return DLDataType{kDLComplex, 64, 1};
    case xla::PrimitiveType::C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      XLA_ERROR() << "XLA type " << xla::PrimitiveType_Name(type) << " has no DLPack equivalent";
  }
}

std::vector<int64_t> StridesForShape(xla::PrimitiveType element_type,
                                     absl::Span<const int64_t> dimensions,
                                     const xla::Layout& layout) {
  XLA_CHECK_EQ(dimensions.size(), layout.minor_to_major().size());
  std::vector<int64_t> strides;
  strides.resize(dimensions.size());
  int64_t stride = 1;
  for (int i : layout.minor_to_major()) {
    strides[i] = stride;
    stride *= dimensions[i];
  }
  return strides;
}

// Convert an XLA tensor to dlPack tensor.
DLManagedTensor* toDLPack(const at::Tensor& input) {
  std::shared_ptr<runtime::ComputationClient::Data> handle = get_data_handle(input);
  XLA_CHECK(handle) << "Could not extract a valid data handle from the input tensor";

  // std::shared_ptr<PjRtData> pjrt_data = std::dynamic_pointer_cast<PjRtData>(data);
  // xla::PjRtBuffer* pjrt_buffer = pjrt_data->buffer.get();
  xla::PjRtBuffer* pjrt_buffer = runtime::GetComputationClient()->GetPjRtBuffer(handle).get();

  if (pjrt_buffer->IsTuple()) {
    XLA_ERROR() << "Unimplemented. BufferToDLPackManagedTensor is not implemented for tuple buffers.";
  }
  if (pjrt_buffer->has_dynamic_dimensions()) {
    XLA_ERROR() << "Unimplemented. DynamicShape is not implemented in DLPack.";
  }

  auto torchXlaDLMTensor = std::make_unique<TorchXLADLMTensor>();
  DLTensor& dt = torchXlaDLMTensor->tensor.dl_tensor;
  {
    auto external_ref = pjrt_buffer->AcquireExternalReference();
    XLA_CHECK_OK(external_ref.status());
    torchXlaDLMTensor->external_reference = std::move(external_ref.value());
    xla::PjRtFuture<> future = pjrt_buffer->GetReadyFuture();
    absl::Status status = future.Await();
    XLA_CHECK_OK(status);
  }
  // pack->buffer_reference = nb::borrow<nb::object>(py_buffer); // xw32: should we do it?

  dt.data = torchXlaDLMTensor->external_reference->OpaqueDeviceMemoryDataPointer();
  torchXlaDLMTensor->tensor.manager_ctx = torchXlaDLMTensor.get();
  torchXlaDLMTensor->tensor.deleter = TorchXLADLMTensorDeleter;
  dt.device = DLDeviceForDevice(*pjrt_buffer->device());
  dt.device.device_id = pjrt_buffer->device()->local_hardware_id();
  dt.ndim = pjrt_buffer->dimensions().size();
  dt.dtype = PrimitiveTypeToDLDataType(pjrt_buffer->element_type());

  torchXlaDLMTensor->shape = std::vector<int64_t>(pjrt_buffer->dimensions().begin(), pjrt_buffer->dimensions().end());
  xla::Layout xla_layout = xla::GetXlaLayoutUnsafe(pjrt_buffer->layout());
  torchXlaDLMTensor->strides = StridesForShape(pjrt_buffer->element_type(), pjrt_buffer->dimensions(), xla_layout);
  dt.shape = reinterpret_cast<std::int64_t*>(torchXlaDLMTensor->shape.data());
  dt.strides = reinterpret_cast<std::int64_t*>(torchXlaDLMTensor->strides.data());
  dt.byte_offset = 0;

  return &(torchXlaDLMTensor.release()->tensor);
}

absl::StatusOr<xla::PjRtDevice*> DeviceForDLDevice(const DLDevice& context) {
  switch (context.device_type) {
    case DLDeviceType::kDLCPU:
      // if (cpu_client == nullptr) {
      //   return InvalidArgument(
      //       "DLPack tensor is on CPU, but no CPU backend was provided.");
      // }
      XLA_CHECK_EQ(runtime::GetComputationClient()->GetPlatformID(), xla::CpuId());
      return runtime::GetComputationClient()->LookupAddressableDevice(context.device_id);
    case DLDeviceType::kDLCUDA:
      // if (gpu_client == nullptr) { // xw32 TODO: check if client_ is GPU client
      //   return InvalidArgument(
      //       "DLPack tensor is on GPU, but no GPU backend was provided.");
      // }
      XLA_CHECK_EQ(runtime::GetComputationClient()->GetPlatformID(), xla::CudaId());
      return runtime::GetComputationClient()->LookupAddressableDevice(context.device_id);
    // case DLDeviceType::kDLROCM:
    //   // if (gpu_client == nullptr) {
    //   //   return InvalidArgument(
    //   //       "DLPack tensor is on GPU, but no GPU backend was provided.");
    //   // }
    //   XLA_CHECK_EQ(pjrt_client->platform_id(), xla::RocmId());
    //   xla::PjRtDevice* device = pjrt_client->addressable_devices()[context.device_id];
    //   return device;
    default:
      return tsl::errors::InvalidArgument("Unknown/unsupported DLPack device type %d",
                             context.device_type);
  }
}

absl::StatusOr<xla::PrimitiveType> DLDataTypeToPrimitiveType(DLDataType type) {
  if (type.lanes != 1) {
    return tsl::errors::Unimplemented("DLPack types with lanes != 1 not implemented, got %d",
                         type.lanes);
  }
  switch (type.code) {
    case kDLBool:
      switch (type.bits) {
        case 8:
          return xla::PrimitiveType::PRED;
        default:
          return tsl::errors::Unimplemented(
              "Only 8-bit DLPack booleans are supported, got %d bits",
              type.bits);
      }
    case kDLInt:
      switch (type.bits) {
        case 8:
          return xla::PrimitiveType::S8;
        case 16:
          return xla::PrimitiveType::S16;
        case 32:
          return xla::PrimitiveType::S32;
        case 64:
          return xla::PrimitiveType::S64;
        default:
          return tsl::errors::Unimplemented(
              "Invalid or unsupported DLPack integer width: %d bits",
              type.bits);
      }
    case kDLUInt:
      switch (type.bits) {
        case 8:
          return xla::PrimitiveType::U8;
        case 16:
          return xla::PrimitiveType::U16;
        case 32:
          return xla::PrimitiveType::U32;
        case 64:
          return xla::PrimitiveType::U64;
        default:
          return tsl::errors::Unimplemented(
              "Invalid or unsupported DLPack unsigned integer width: %d bits",
              type.bits);
      }
    case kDLFloat:
      switch (type.bits) {
        case 16:
          return xla::PrimitiveType::F16;
        case 32:
          return xla::PrimitiveType::F32;
        case 64:
          return xla::PrimitiveType::F64;
        default:
          return tsl::errors::Unimplemented(
              "Invalid or unsupported DLPack float width: %d bits", type.bits);
      }
    case kDLBfloat:
      switch (type.bits) {
        case 16:
          return xla::PrimitiveType::BF16;
        default:
          return tsl::errors::Unimplemented(
              "Invalid or unsupported DLPack Bfloat width: %d bits", type.bits);
      }
    case kDLComplex:
      switch (type.bits) {
        case 64:
          return xla::PrimitiveType::C64;
        case 128:
          return xla::PrimitiveType::C128;
        default:
          return tsl::errors::Unimplemented(
              "Invalid or unsupported DLPack complex width: %d bits",
              type.bits);
      }
    default:
      return tsl::errors::Unimplemented("Unknown or invalid DLPack type code %d", type.code);
  }
}

absl::StatusOr<std::vector<int64_t>> StridesToLayout(
    absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  XLA_CHECK_EQ(dims.size(), strides.size());
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });
  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return tsl::errors::Unimplemented(
          "Only DLPack tensors with trivial (compact) striding are supported; "
          "i.e., tensors whose striding represents a transposition of the "
          "underlying buffer but not broadcasting. Dimensions were: [%s], "
          "strides were [%s].",
          absl::StrJoin(dims, ","), absl::StrJoin(strides, ","));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}

at::Tensor fromDLPack(DLManagedTensor* dlmt) {
  if (dlmt->dl_tensor.ndim < 0) {
    XLA_ERROR() << "Number of dimensions in DLManagedTensor must be nonnegative, got " << dlmt->dl_tensor.ndim;
  }
  xla::PjRtDevice* device = DeviceForDLDevice(dlmt->dl_tensor.device).value(); // client_ is a xla::PjRtClient. So this fromDLPack should be inside pjrt_computation_client class.
  absl::Span<int64_t const> dimensions(
      const_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  xla::PrimitiveType element_type = DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype).value();

  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        const_cast<int64_t*>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    minor_to_major = StridesToLayout(dimensions, strides).value();
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                                    minor_to_major);

  // Raise an error if the resulting PjRtBuffer would have a non-default layout.
  // TODO(skyewm): we do this because JAX doesn't currently have good support
  // for non-default layouts, and will return wrong results if a non-default
  // layout is passed to a computation expecting default layouts. Remove this
  // special case when non-default layouts are better supported by JAX.
  absl::StatusOr<xla::Layout> default_layout_from_client =
      device->client()->GetDefaultLayout(element_type, dimensions);
  xla::Layout default_layout;
  if (default_layout_from_client.ok()) {
    default_layout = *default_layout_from_client;
  } else if (absl::IsUnimplemented(default_layout_from_client.status())) {
    // TODO(skyewm): consider remove the fallback path when GetDefaultLayout is
    // unimplemented.
    xla::Shape host_shape = xla::ShapeUtil::MakeShape(element_type, dimensions);
    default_layout = xla::LayoutUtil::GetWithDefaultLayout(host_shape).layout();
  } else {
    XLA_ERROR() << "default_layout_from_client.status() is not ok.";
  }
  // if (shape.layout() != default_layout) {
  //   XLA_ERROR() << "from_dlpack got array with non-default layout with minor-to-major dimensions (" << absl::StrJoin(shape.layout().minor_to_major(), ",") << "), expected (" << absl::StrJoin(default_layout.minor_to_major(), ",") << ")";
  // }

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt]() { dlmt->deleter(dlmt); };
  }
  xla::StatusOr<std::unique_ptr<xla::PjRtBuffer>> pjrt_buffer = device->client()->CreateViewOfDeviceBuffer(
                          static_cast<char*>(dlmt->dl_tensor.data) +
                              dlmt->dl_tensor.byte_offset,
                          shape, device, on_delete_callback);

  runtime::ComputationClient::DataPtr data = runtime::GetComputationClient()->CreateData(runtime::GetComputationClient()->PjRtDeviceToString(device), shape, std::move(pjrt_buffer.value()));
  
  // xw32 note: XlaDataToTensors does a fromDeviceToHost transfer.XlaDataToTensors
  at::ScalarType tensor_type = at::toScalarType(dlmt->dl_tensor.dtype);
  return XlaDataToTensors({data}, {tensor_type})[0];

}

} // xw32: why do we need the extra namespace?
}
