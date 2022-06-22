#include "torch_xla/csrc/tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct XLAGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::XLA; }

  c10::Device exchangeDevice(c10::Device device) const override {
    return bridge::SetCurrentDevice(device);
  }

  c10::Device getDevice() const override {
    return bridge::GetCurrentAtenDevice();
  }

  void setDevice(c10::Device device) const override {
    bridge::SetCurrentDevice(device);
  }

  void uncheckedSetDevice(c10::Device device) const noexcept override {
    bridge::SetCurrentDevice(device);
  }

  c10::Stream getStream(c10::Device device) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, bridge::GetCurrentAtenDevice());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return xla::ComputationClient::Get()->GetNumDevices();
  }
};

C10_REGISTER_GUARD_IMPL(XLA, XLAGuardImpl);

}  // namespace

XLATensorImpl::XLATensorImpl(XLATensor tensor)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::XLA,
                                          c10::DispatchKey::AutogradXLA},
                      GetTypeMeta(tensor),
                      bridge::XlaDeviceToAtenDevice(tensor.GetDevice())),
      tensor_(std::move(tensor)) {
  is_non_overlapping_and_dense_ = false;
  set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
}

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

c10::intrusive_ptr<c10::TensorImpl> XLATensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<XLATensorImpl>(tensor_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
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
  xla_impl->tensor_.ShallowCopyTo(&tensor_);
  generation_ = 0;
}

at::IntArrayRef XLATensorImpl::sizes_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return sizes_default();
}

c10::SymIntArrayRef XLATensorImpl::sym_sizes_custom() const {
  auto sizes = sizes_custom();
  return c10::SymIntArrayRef(reinterpret_cast<const c10::SymInt*>(sizes.data()),
                             sizes.size());
}

c10::SymIntArrayRef XLATensorImpl::sym_sizes() const {
  // it isn't strictly necessary to delegate to `sym_sizes_custom`
  // however, it's consistent with pytorch core
  return sym_sizes_custom();
}

at::IntArrayRef XLATensorImpl::strides_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return strides_default();
}

int64_t XLATensorImpl::dim_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return dim_default();
}

int64_t XLATensorImpl::numel_custom() const {
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return numel_default();
}

bool XLATensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  // Only check that the storage is already contiguous.
  XLA_CHECK(is_contiguous_) << "Non-contiguous storage for XLA tensor";
  return true;
}

void XLATensorImpl::SetupSizeProperties() {
  size_t generation = tensor_.generation();
  if (generation != generation_) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto shape = tensor_.shape();
    c10::SmallVector<int64_t, 5> updated_sizes;
    numel_ = 1;
    for (auto dim : shape.get().dimensions()) {
      updated_sizes.push_back(dim);
      numel_ *= dim;
    }
    sizes_and_strides_.set_sizes(updated_sizes);
    auto updated_strides = torch::lazy::ComputeArrayStrides(
        torch::lazy::ToVector<int64_t>(shape.get().dimensions()));
    for (int i = 0; i < updated_strides.size(); i++) {
      sizes_and_strides_.stride_at_unchecked(i) = updated_strides[i];
    }
    generation_ = generation;
  }
}

caffe2::TypeMeta XLATensorImpl::GetTypeMeta(const XLATensor& tensor) {
  return c10::scalarTypeToTypeMeta(tensor.dtype());
}

void XLATensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

const at::Storage& XLATensorImpl::storage() const {
  XLA_ERROR() << "XLA tensors do not have storage";
}

bool XLATensorImpl::has_storage() const { return false; }

}  // namespace torch_xla
