#include "tensor_impl.h"

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

#include "tensor_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace {

struct XLAGuardImpl : public c10::impl::DeviceGuardImplInterface {
  at::DeviceType type() const override { return at::DeviceType::XLA; }

  c10::Device exchangeDevice(c10::Device) const override {
    return c10::Device(at::DeviceType::XLA, 0);
  }

  c10::Device getDevice() const override {
    return c10::Device(at::DeviceType::XLA, 0);
  }

  void setDevice(c10::Device) const override {}

  void uncheckedSetDevice(c10::Device d) const noexcept override {}

  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(at::DeviceType::XLA, 0));
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(at::DeviceType::XLA, 0));
  }

  c10::DeviceIndex deviceCount() const override { return 1; }
};

}  // namespace

C10_REGISTER_GUARD_IMPL(XLA, XLAGuardImpl);

// TODO: Replace UndefinedTensorId with proper type.
XLATensorImpl::XLATensorImpl(XLATensor tensor)
    : c10::TensorImpl(GetStorage(tensor), c10::XLATensorId(),
                      /*is_variable=*/false),
      tensor_(std::move(tensor)) {
  SetupSizeProperties();
}

XLATensorImpl::XLATensorImpl(XLATensor tensor, bool is_variable)
    : c10::TensorImpl(GetStorage(tensor), c10::XLATensorId(), is_variable),
      tensor_(std::move(tensor)) {
  SetupSizeProperties();
}

c10::intrusive_ptr<c10::TensorImpl> XLATensorImpl::shallow_copy_and_detach()
    const {
  auto impl = c10::make_intrusive<XLATensorImpl>(tensor_, is_variable());
  impl->is_wrapped_number_ = is_wrapped_number_;
  impl->reserved_ = reserved_;
  return impl;
}

void XLATensorImpl::SetupSizeProperties() {
  // Fill up the basic dimension data members which the base class
  // implementation uses in its APIs.
  auto shape = tensor_.shape();
  for (auto dim : shape.get().dimensions()) {
    sizes_.push_back(dim);
    numel_ *= dim;
  }
  for (auto stride : ComputeShapeStrides(shape.get())) {
    strides_.push_back(stride);
  }
}

XLATensorImpl::XLATensorImpl()
    : c10::TensorImpl(c10::UndefinedTensorId(), caffe2::TypeMeta(), nullptr,
                      /*is variable=*/false) {}

caffe2::TypeMeta XLATensorImpl::GetTypeMeta(const XLATensor& tensor) {
  auto shape = tensor.shape();
  switch (shape.get().element_type()) {
    case xla::PrimitiveType::F32:
      return caffe2::TypeMeta::Make<float>();
    case xla::PrimitiveType::U8:
      return caffe2::TypeMeta::Make<uint8_t>();
    case xla::PrimitiveType::S8:
      return caffe2::TypeMeta::Make<int8_t>();
    case xla::PrimitiveType::S16:
      return caffe2::TypeMeta::Make<int16_t>();
    case xla::PrimitiveType::S32:
      return caffe2::TypeMeta::Make<int32_t>();
    case xla::PrimitiveType::S64:
      return caffe2::TypeMeta::Make<int64_t>();
    default:
      XLA_ERROR() << "Type not supported: " << shape;
  }
}

c10::Storage XLATensorImpl::GetStorage(const XLATensor& tensor) {
  Device device = tensor.GetDevice();
  return c10::Storage(at::Device(c10::DeviceType::XLA, device.ordinal),
                      GetTypeMeta(tensor));
}

XLAUndefinedTensorImpl::XLAUndefinedTensorImpl() {}

at::IntList XLAUndefinedTensorImpl::sizes() const {
  AT_ERROR("sizes() called on undefined Tensor");
}

int64_t XLAUndefinedTensorImpl::size(int64_t d) const {
  AT_ERROR("size(dim) called on an undefined Tensor");
}

int64_t XLAUndefinedTensorImpl::stride(int64_t d) const {
  AT_ERROR("stride(dim) called on an undefined Tensor");
}

int64_t XLAUndefinedTensorImpl::dim() const {
  AT_ERROR("dim() called on undefined Tensor");
}

const at::Storage& XLAUndefinedTensorImpl::storage() const {
  AT_ERROR("storage() called on undefined Tensor");
}

int64_t XLAUndefinedTensorImpl::storage_offset() const {
  AT_ERROR("storage_offset() called on an undefined Tensor");
}

at::IntList XLAUndefinedTensorImpl::strides() const {
  AT_ERROR("strides() called on undefined Tensor");
}
XLAUndefinedTensorImpl XLAUndefinedTensorImpl::_singleton;

}  // namespace torch_xla
