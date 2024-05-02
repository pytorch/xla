#include "torch_xla/csrc/dl_convertor.h"

#include "absl/types/span.h"

#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/unwrap_data.h"
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

} // xw32: why do we need the extra namespace?
}
