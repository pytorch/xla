#include "torch_xla/csrc/tensor_impl.h"

#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir_builder.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
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
    auto* client = runtime::GetComputationClientIfInitialized();

    if (client == nullptr) {
      TF_VLOG(5) << "XLA client uninitialized. Returning 0 devices.";
      return 0;
    }

    return client->GetNumLocalDevices();
  }
};

C10_REGISTER_GUARD_IMPL(XLA, XLAGuardImpl);

}  // namespace

XLATensorImpl::XLATensorImpl(XLATensor&& tensor)
    : c10::TensorImpl(c10::DispatchKeySet{c10::DispatchKey::XLA,
                                          c10::DispatchKey::AutogradXLA},
                      GetTypeMeta(tensor),
                      bridge::XlaDeviceToAtenDevice(tensor.GetDevice())),
      tensor_(c10::make_intrusive<XLATensor>(std::move(tensor))) {
  // Update the Autocast key based off the backend device.
  // Upstream TensorImpl cannot differentiate between XLA:TPU and XLA:GPU
  // so we must manually update Autocast to AutocastCUDA on XLA:GPU.
  torch::lazy::BackendDevice current_device = bridge::GetCurrentDevice();
  auto dev_type = static_cast<XlaDeviceType>(current_device.type());
  if (dev_type == XlaDeviceType::CUDA) {
    auto autocast_cuda_ks = c10::DispatchKeySet(c10::DispatchKey::AutocastCUDA);
    auto autocast_xla_ks = c10::DispatchKeySet(c10::DispatchKey::AutocastXLA);
    key_set_ = (key_set_ - autocast_xla_ks) | autocast_cuda_ks;
  }
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  set_sizes_and_strides(sym_sizes_, c10::fromIntArrayRefSlow(
                                        sizes_and_strides_.strides_arrayref()));
  set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);
}

XLATensorImpl::XLATensorImpl(XLATensor& tensor)
    : XLATensorImpl(XLATensor(tensor)) {}

XLATensorImpl::XLATensorImpl(XLATensorPtr tensor)
    : XLATensorImpl(XLATensor(*tensor)) {}

void XLATensorImpl::set_tensor(XLATensorPtr xla_tensor) {
  tensor_ = xla_tensor;
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
  xla_impl->tensor_->ShallowCopyTo(tensor_);
  generation_ = 0;
}

at::IntArrayRef XLATensorImpl::sizes_custom() const {
  XLA_CHECK(!has_symbolic_sizes_strides_)
      << "Cannot call sizes_custom() on an XLA tensor with symbolic "
         "sizes/strides";
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return sizes_default();
}

c10::SymIntArrayRef XLATensorImpl::sym_sizes_custom() const {
  // N.B. SetupSizeProperties also updates sym_sizes_
  const_cast<XLATensorImpl*>(this)->SetupSizeProperties();
  return c10::SymIntArrayRef(sym_sizes_.data(), sym_sizes_.size());
}

c10::SymInt XLATensorImpl::sym_numel_custom() const {
  auto sym_sizes = sym_sizes_custom();
  c10::SymInt prod{1};
  for (auto s : sym_sizes) {
    prod *= s;
  }
  return prod;
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
  // Storage is always contiguous, but the tensor metadata is_contiguous_ might
  // be false due to the update in the functionalization layer..
  return true;
}

void XLATensorImpl::SetupSizeProperties() {
  size_t generation = tensor_->generation();
  if (generation != generation_) {
    // Fill up the basic dimension data members which the base class
    // implementation uses in its APIs.
    auto shape = tensor_->shape();
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
    SetupSymSizeProperties();
    generation_ = generation;
  }
}

void XLATensorImpl::SetupSymSizeProperties() {
  auto shape = tensor_->shape();
  auto rank = shape.get().dimensions_size();
  std::vector<c10::SymInt> sym_sizes;
  sym_sizes.reserve(rank);

  XLAIrBuilder a = XLAIrBuilder();
  for (auto i : c10::irange(rank)) {
    if (shape.get().is_dynamic_dimension(i)) {
      auto dim_node = a.MakeSizeNode(tensor_->GetIrValue(), i);
      auto symint_node =
          c10::make_intrusive<XLASymNodeImpl>(dim_node, PyType::INT);
      sym_sizes.push_back(c10::SymInt(
          static_cast<c10::intrusive_ptr<c10::SymNodeImpl>>(symint_node)));
    } else {
      sym_sizes.push_back(c10::SymInt(shape.get().dimensions(i)));
    }
  }
  sym_sizes_ = sym_sizes;
}

caffe2::TypeMeta XLATensorImpl::GetTypeMeta(const XLATensor& tensor) {
  return c10::scalarTypeToTypeMeta(tensor.dtype());
}

void XLATensorImpl::AtenInitialize() {
  // ATEN specific initialization calls placed below.
}

const at::Storage& XLATensorImpl::storage() const { return tensor_->Storage(); }

bool XLATensorImpl::has_storage() const { return tensor_->Storage(); }

}  // namespace torch_xla
