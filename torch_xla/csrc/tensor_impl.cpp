#include "torch_xla/csrc/tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

thread_local c10::Device g_current_device(at::DeviceType::XLA, 0);

struct XLAGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::XLA; }

  c10::Device exchangeDevice(c10::Device device) const override {
    std::swap(g_current_device, device);
    return device;
  }

  c10::Device getDevice() const override { return g_current_device; }

  void setDevice(c10::Device device) const override {
    g_current_device = device;
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    g_current_device = device;
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, g_current_device);
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return xla::ComputationClient::Get()->GetNumDevices();
  }
};

C10_REGISTER_GUARD_IMPL(XLA, XLAGuardImpl);

}  // namespace

XLATensorImpl::XLATensorImpl(XLATensor tensor)
    : c10::TensorImpl(c10::XLATensorId(), GetTypeMeta(tensor),
                      bridge::XlaDeviceToAtenDevice(tensor.GetDevice())),
      tensor_(std::move(tensor)) {}

void XLATensorImpl::set_tensor(XLATensor xla_tensor) {
  tensor_ = std::move(xla_tensor);
  generation_ = 0;
}

c10::intrusive_ptr<c10::TensorImpl> XLATensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<XLATensorImpl>(tensor_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

void XLATensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  XLATensorImpl* xla_impl = dynamic_cast<XLATensorImpl*>(impl.get());
  copy_tensor_metadata(
      /*src_impl=*/xla_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());

  tensor_ = XLATensor::clone(xla_impl->tensor_);
  generation_ = 0;
}

at::IntArrayRef XLATensorImpl::sizes() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::sizes();
}

int64_t XLATensorImpl::dim() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::dim();
}

int64_t XLATensorImpl::numel() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::numel();
}

bool XLATensorImpl::is_contiguous(at::MemoryFormat memory_format) const {
  // Only check that the storage is already contiguous.
  XLA_CHECK(is_contiguous_) << "Non-contiguous storage for XLA tensor";
  return true;
}

int64_t XLATensorImpl::size(int64_t d) const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return c10::TensorImpl::size(d);
}

void XLATensorImpl::SetupSizeProperties() {
  size_t generation = tensor_.generation();
  if (generation != generation_) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto shape = tensor_.shape();
    sizes_.clear();
    numel_ = 1;
    for (auto dim : shape.get().dimensions()) {
      sizes_.push_back(dim);
      numel_ *= dim;
    }
    strides_.clear();
    xla::Shape torch_shape = MakeTorchTensorLayout(shape.get().dimensions(),
                                                   shape.get().element_type());
    for (auto stride : ComputeShapeStrides(torch_shape)) {
      strides_.push_back(stride);
    }
    generation_ = generation;
  }
}

caffe2::TypeMeta XLATensorImpl::GetTypeMeta(const XLATensor& tensor) {
  return c10::scalarTypeToTypeMeta(tensor.dtype());
}

c10::Device XLATensorImpl::GetCurrentAtenDevice() { return g_current_device; }

c10::Device XLATensorImpl::SetCurrentAtenDevice(c10::Device device) {
  std::swap(g_current_device, device);
  TF_VLOG(2) << "New Aten device : " << g_current_device;
  return device;
}

void XLATensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

const at::Storage& XLATensorImpl::storage() const {
  XLA_ERROR() << "XLA tensors do not have storage";
}

bool XLATensorImpl::has_storage() const { return false; }

}  // namespace torch_xla
