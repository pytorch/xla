#include "torch_xla/csrc/aten_xla_type.h"

#include <ATen/Context.h>

#include <mutex>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/einsum.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

struct XlaOptions {
  XlaOptions(const at::TensorOptions& options,
             c10::optional<Device> device_opt = c10::nullopt)
      : device(std::move(device_opt)) {
    if (options.has_device()) {
      device = bridge::AtenDeviceToXlaDevice(options.device());
    }
    if (options.has_dtype()) {
      scalar_type = c10::typeMetaToScalarType(options.dtype());
    }
  }

  Device get_device() const { return device ? *device : *GetDefaultDevice(); }

  at::ScalarType get_scalar_type(
      at::ScalarType defval = at::ScalarType::Float) const {
    return scalar_type ? *scalar_type : defval;
  }

  c10::optional<Device> device;
  c10::optional<at::ScalarType> scalar_type;
};

// Returns true if dilation is non-trivial (not 1) in at least one dimension.
bool IsNonTrivialDilation(at::IntArrayRef dilation) {
  return std::any_of(
      dilation.begin(), dilation.end(),
      [](const int64_t dim_dilation) { return dim_dilation != 1; });
}

// Returns true if padding is non-trivial (not 0) in at least one dimension.
bool IsNonTrivialPadding(at::IntArrayRef padding) {
  return std::any_of(
      padding.begin(), padding.end(),
      [](const int64_t dim_padding) { return dim_padding != 0; });
}

void AtenInitialize() {
  auto& context = at::globalContext();
  context.registerType(
      at::Backend::XLA,
      new AtenXlaType(c10::XLATensorId(), /*is_variable=*/false,
                      /*is_undefined=*/false));
  XLATensorImpl::AtenInitialize();
}

}  // namespace

AtenXlaType::AtenXlaType(at::TensorTypeId type_id, bool is_variable,
                         bool is_undefined)
    : AtenXlaTypeBase(type_id, is_variable, is_undefined) {}

int64_t AtenXlaType::numel(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return xla::ShapeUtil::ElementsIn(self_tensor.shape());
}

at::Tensor AtenXlaType::__and__(const at::Tensor& self,
                                at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::__and__(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::__and__(const at::Tensor& self,
                                const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::__and__(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::__iand__(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__iand__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__iand__(at::Tensor& self,
                                  const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__iand__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::__ilshift__(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__ilshift__(at::Tensor& self,
                                     const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ilshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::__ior__(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ior__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__ior__(at::Tensor& self,
                                 const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ior__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::__irshift__(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__irshift__(at::Tensor& self,
                                     const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__irshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::__ixor__(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ixor__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__ixor__(at::Tensor& self,
                                  const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ixor__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::__lshift__(const at::Tensor& self,
                                   at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::__lshift__(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::__lshift__(const at::Tensor& self,
                                   const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::__lshift__(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::__or__(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::__or__(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::__or__(const at::Tensor& self,
                               const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::__or__(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::__rshift__(const at::Tensor& self,
                                   at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::__rshift__(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::__rshift__(const at::Tensor& self,
                                   const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::__rshift__(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::__xor__(const at::Tensor& self,
                                at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::__xor__(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::__xor__(const at::Tensor& self,
                                const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::__xor__(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::_adaptive_avg_pool2d(
    const at::Tensor& self, at::IntArrayRef output_size) const {
  if (self.dim() != 4 ||
      !IsSupportedAdaptiveAvgPool2d(XlaHelpers::I64List(self.sizes()),
                                    XlaHelpers::I64List(output_size))) {
    return AtenXlaTypeBase::_adaptive_avg_pool2d(self, output_size);
  }
  return bridge::AtenFromXlaTensor(XLATensor::_adaptive_avg_pool2d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(output_size)));
}

at::Tensor AtenXlaType::_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) const {
  if (grad_output.dim() != 4) {
    return AtenXlaTypeBase::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  std::vector<xla::int64> output_size{grad_output.size(2), grad_output.size(3)};
  if (!IsSupportedAdaptiveAvgPool2d(XlaHelpers::I64List(self.sizes()),
                                    output_size)) {
    return AtenXlaTypeBase::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return bridge::AtenFromXlaTensor(XLATensor::_adaptive_avg_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::_cast_Byte(const at::Tensor& self,
                                   bool /* non_blocking */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cast(bridge::GetXlaTensor(self), at::ScalarType::Byte));
}

at::Tensor AtenXlaType::_cast_Char(const at::Tensor& self,
                                   bool /* non_blocking */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cast(bridge::GetXlaTensor(self), at::ScalarType::Char));
}

at::Tensor AtenXlaType::_cast_Float(const at::Tensor& self,
                                    bool /* non_blocking */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cast(bridge::GetXlaTensor(self), at::ScalarType::Float));
}

at::Tensor AtenXlaType::_cast_Int(const at::Tensor& self,
                                  bool /* non_blocking */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cast(bridge::GetXlaTensor(self), at::ScalarType::Int));
}

at::Tensor AtenXlaType::_cast_Long(const at::Tensor& self,
                                   bool /* non_blocking */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cast(bridge::GetXlaTensor(self), at::ScalarType::Long));
}

at::Tensor AtenXlaType::_cast_Short(const at::Tensor& self,
                                    bool /* non_blocking */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cast(bridge::GetXlaTensor(self), at::ScalarType::Short));
}

at::Tensor AtenXlaType::_dim_arange(const at::Tensor& like, int64_t dim) const {
  return arange(like.size(dim), like.options().dtype(at::kLong));
}

at::Tensor AtenXlaType::_log_softmax(const at::Tensor& self, int64_t dim,
                                     bool /* half_to_float */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::log_softmax(bridge::GetXlaTensor(self), dim));
}

at::Tensor AtenXlaType::_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    const at::Tensor& /* self*/) const {
  return bridge::AtenFromXlaTensor(XLATensor::log_softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

at::Tensor AtenXlaType::_softmax(const at::Tensor& self, int64_t dim,
                                 bool /* half_to_float */) const {
  return softmax(self, dim);
}

at::Tensor AtenXlaType::_softmax_backward_data(const at::Tensor& grad_output,
                                               const at::Tensor& output,
                                               int64_t dim,
                                               const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

at::Tensor AtenXlaType::_trilinear(
    const at::Tensor& i1, const at::Tensor& i2, const at::Tensor& i3,
    at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3,
    at::IntArrayRef sumdim, int64_t unroll_dim) const {
  return at::native::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim,
                                unroll_dim);
}

at::Tensor AtenXlaType::_unsafe_view(const at::Tensor& self,
                                     at::IntArrayRef size) const {
  return view(self, size);
}

at::Tensor AtenXlaType::abs(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::abs(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::abs_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::abs_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::acos(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::acos(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::acos_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::acos_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Tensor& other,
                            at::Scalar alpha) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::add(
      self_tensor, bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()),
      alpha));
}

at::Tensor AtenXlaType::add(const at::Tensor& self, at::Scalar other,
                            at::Scalar alpha) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::add(bridge::GetXlaTensor(self), other, alpha));
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, const at::Tensor& other,
                              at::Scalar alpha) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::add_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()),
                  alpha);
  return self;
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, at::Scalar other,
                              at::Scalar alpha) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::add_(self_tensor, other, alpha);
  return self;
}

at::Tensor AtenXlaType::addcdiv(const at::Tensor& self,
                                const at::Tensor& tensor1,
                                const at::Tensor& tensor2,
                                at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::addcdiv(
      bridge::GetXlaTensor(self), value, bridge::GetXlaTensor(tensor1),
      bridge::GetXlaTensor(tensor2)));
}

at::Tensor& AtenXlaType::addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2,
                                  at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::addcdiv_(self_tensor, value, bridge::GetXlaTensor(tensor1),
                      bridge::GetXlaTensor(tensor2));
  return self;
}

at::Tensor AtenXlaType::addcmul(const at::Tensor& self,
                                const at::Tensor& tensor1,
                                const at::Tensor& tensor2,
                                at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::addcmul(
      bridge::GetXlaTensor(self), value, bridge::GetXlaTensor(tensor1),
      bridge::GetXlaTensor(tensor2)));
}

at::Tensor& AtenXlaType::addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2,
                                  at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::addcmul_(self_tensor, value, bridge::GetXlaTensor(tensor1),
                      bridge::GetXlaTensor(tensor2));
  return self;
}

at::Tensor AtenXlaType::addmm(const at::Tensor& self, const at::Tensor& mat1,
                              const at::Tensor& mat2, at::Scalar beta,
                              at::Scalar alpha) const {
  if (beta.to<double>() != 1 || alpha.to<double>() != 1) {
    return AtenXlaTypeBase::addmm(self, mat1, mat2, beta, alpha);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::addmm(bridge::GetXlaTensor(mat1),
                       /*weight=*/bridge::GetXlaTensor(mat2),
                       /*bias=*/bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::alias(const at::Tensor& self) const { return self; }

at::Tensor AtenXlaType::all(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::all(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions*/ false));
}

at::Tensor AtenXlaType::all(const at::Tensor& self, int64_t dim,
                            bool keepdim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::all(bridge::GetXlaTensor(self), {dim}, keepdim));
}

at::Tensor AtenXlaType::any(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::any(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions*/ false));
}

at::Tensor AtenXlaType::any(const at::Tensor& self, int64_t dim,
                            bool keepdim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::any(bridge::GetXlaTensor(self), {dim}, keepdim));
}

at::Tensor AtenXlaType::arange(at::Scalar end,
                               const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(XLATensor::arange(
      0, end, 1, xla_options.get_device(), xla_options.get_scalar_type()));
}

at::Tensor AtenXlaType::arange(at::Scalar start, at::Scalar end,
                               const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(XLATensor::arange(
      start, end, 1, xla_options.get_device(), xla_options.get_scalar_type()));
}

at::Tensor AtenXlaType::arange(at::Scalar start, at::Scalar end,
                               at::Scalar step,
                               const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(
      XLATensor::arange(start, end, step, xla_options.get_device(),
                        xla_options.get_scalar_type()));
}

at::Tensor AtenXlaType::argmax(const at::Tensor& self,
                               c10::optional<int64_t> dim, bool keepdim) const {
  return dim ? bridge::AtenFromXlaTensor(
                   XLATensor::argmax(bridge::GetXlaTensor(self), *dim, keepdim))
             : bridge::AtenFromXlaTensor(
                   XLATensor::argmax(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::argmin(const at::Tensor& self,
                               c10::optional<int64_t> dim, bool keepdim) const {
  return dim ? bridge::AtenFromXlaTensor(
                   XLATensor::argmin(bridge::GetXlaTensor(self), *dim, keepdim))
             : bridge::AtenFromXlaTensor(
                   XLATensor::argmin(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::argsort(const at::Tensor& self, int64_t dim,
                                bool descending) const {
  return std::get<1>(sort(self, dim, descending));
}

at::Tensor AtenXlaType::as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) const {
  if (!ir::ops::AsStrided::StrideIsSupported(XlaHelpers::I64List(size),
                                             XlaHelpers::I64List(stride))) {
    return AtenXlaTypeBase::as_strided(self, size, stride, storage_offset);
  }
  return bridge::AtenFromXlaTensor(XLATensor::as_strided(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(size),
      XlaHelpers::I64Optional(storage_offset)));
}

at::Tensor& AtenXlaType::as_strided_(
    at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) const {
  if (!ir::ops::AsStrided::StrideIsSupported(XlaHelpers::I64List(size),
                                             XlaHelpers::I64List(stride))) {
    return AtenXlaTypeBase::as_strided_(self, size, stride, storage_offset);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::as_strided_(self_tensor, XlaHelpers::I64List(size),
                         XlaHelpers::I64Optional(storage_offset));
  return self;
}

at::Tensor AtenXlaType::asin(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::asin(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::asin_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::asin_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::atan(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::atan(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::atan2(const at::Tensor& self,
                              const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::atan2(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::atan2_(at::Tensor& self,
                                const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::atan2_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::atan_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::atan_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::avg_pool1d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad) const {
  // Lowering when ceil_mode is set not supported yet.
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool1d(self, kernel_size, stride, padding,
                                       ceil_mode, count_include_pad);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/1,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), count_include_pad));
}

at::Tensor AtenXlaType::avg_pool2d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad) const {
  // Lowering when ceil_mode is set not supported yet.
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool2d(self, kernel_size, stride, padding,
                                       ceil_mode, count_include_pad);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), count_include_pad));
}

at::Tensor AtenXlaType::avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
  // Lowering when ceil_mode is set not supported yet.
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool2d_backward(grad_output, self, kernel_size,
                                                stride, padding, ceil_mode,
                                                count_include_pad);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding),
      count_include_pad));
}

at::Tensor AtenXlaType::avg_pool3d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad) const {
  // Lowering when ceil_mode is set not supported yet.
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool3d(self, kernel_size, stride, padding,
                                       ceil_mode, count_include_pad);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), count_include_pad));
}

at::Tensor AtenXlaType::avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) const {
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool3d_backward(grad_output, self, kernel_size,
                                                stride, padding, ceil_mode,
                                                count_include_pad);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding),
      count_include_pad));
}

at::Tensor AtenXlaType::bartlett_window(
    int64_t window_length, const at::TensorOptions& options) const {
  return at::native::bartlett_window(window_length, options);
}

at::Tensor AtenXlaType::bartlett_window(
    int64_t window_length, bool periodic,
    const at::TensorOptions& options) const {
  return at::native::bartlett_window(window_length, periodic, options);
}

at::Tensor AtenXlaType::batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    bool training, double momentum, double eps, bool cudnn_enabled) const {
  if (cudnn_enabled || !training) {
    return AtenXlaTypeBase::batch_norm(input, weight, bias, running_mean,
                                       running_var, training, momentum, eps,
                                       cudnn_enabled);
  }
  XLATensor input_tensor = bridge::GetXlaTensorUnwrap(input);
  const Device& device = input_tensor.GetDevice();
  return bridge::AtenFromXlaTensor(XLATensor::batch_norm(
      bridge::GetXlaTensor(input), bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(bias, device), momentum, eps));
}

at::Tensor AtenXlaType::bernoulli(const at::Tensor& self, double p,
                                  at::Generator* generator) const {
  if (generator != nullptr) {
    return AtenXlaTypeBase::bernoulli(self, p, generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::bernoulli(bridge::GetXlaTensor(self), p));
}

at::Tensor AtenXlaType::bernoulli(const at::Tensor& self,
                                  at::Generator* generator) const {
  if (generator != nullptr) {
    return AtenXlaTypeBase::bernoulli(self, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::bernoulli(self_tensor));
}

at::Tensor& AtenXlaType::bernoulli_(at::Tensor& self, double p,
                                    at::Generator* generator) const {
  if (generator != nullptr) {
    return AtenXlaTypeBase::bernoulli_(self, p, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor& AtenXlaType::bernoulli_(at::Tensor& self, const at::Tensor& p,
                                    at::Generator* generator) const {
  if (generator != nullptr) {
    return AtenXlaTypeBase::bernoulli_(self, p, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bernoulli_(self_tensor, bridge::GetXlaTensor(p));
  return self;
}

at::Tensor AtenXlaType::bilinear(const at::Tensor& input1,
                                 const at::Tensor& input2,
                                 const at::Tensor& weight,
                                 const at::Tensor& bias) const {
  return at::native::bilinear(input1, input2, weight, bias);
}

at::Tensor AtenXlaType::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    const at::Tensor& pos_weight, int64_t reduction) const {
  return at::native::binary_cross_entropy_with_logits(self, target, weight,
                                                      pos_weight, reduction);
}

at::Tensor AtenXlaType::binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight,
    const at::Tensor& pos_weight, int64_t reduction) const {
  return at::native::binary_cross_entropy_with_logits_backward(
      grad_output, self, target, weight, pos_weight, reduction);
}

at::Tensor AtenXlaType::blackman_window(
    int64_t window_length, const at::TensorOptions& options) const {
  return at::native::blackman_window(window_length, options);
}

at::Tensor AtenXlaType::blackman_window(
    int64_t window_length, bool periodic,
    const at::TensorOptions& options) const {
  return at::native::blackman_window(window_length, periodic, options);
}

at::Tensor AtenXlaType::bmm(const at::Tensor& self,
                            const at::Tensor& mat2) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::bmm(bridge::GetXlaTensor(self), bridge::GetXlaTensor(mat2)));
}

std::vector<at::Tensor> AtenXlaType::broadcast_tensors(
    at::TensorList tensors) const {
  return bridge::AtenFromXlaTensors(
      XLATensor::broadcast_tensors(bridge::GetXlaTensors(tensors)));
}

at::Tensor AtenXlaType::cat(at::TensorList tensors, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cat(bridge::GetXlaTensors(tensors), dim));
}

at::Tensor AtenXlaType::ceil(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::ceil(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::ceil_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ceil_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::celu(const at::Tensor& self, at::Scalar alpha) const {
  return at::native::celu(self, alpha);
}

at::Tensor& AtenXlaType::celu_(at::Tensor& self, at::Scalar alpha) const {
  return at::native::celu_(self, alpha);
}

at::Tensor AtenXlaType::chain_matmul(at::TensorList matrices) const {
  return at::native::chain_matmul(matrices);
}

at::Tensor AtenXlaType::cholesky(const at::Tensor& self, bool upper) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cholesky(bridge::GetXlaTensor(self), upper));
}

at::Tensor AtenXlaType::clamp(const at::Tensor& self,
                              c10::optional<at::Scalar> min,
                              c10::optional<at::Scalar> max) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, max));
}

at::Tensor& AtenXlaType::clamp_(at::Tensor& self, c10::optional<at::Scalar> min,
                                c10::optional<at::Scalar> max) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, min, max);
  return self;
}

at::Tensor AtenXlaType::clamp_max(const at::Tensor& self,
                                  at::Scalar max) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), c10::nullopt, max));
}

at::Tensor& AtenXlaType::clamp_max_(at::Tensor& self, at::Scalar max) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, c10::nullopt, max);
  return self;
}

at::Tensor AtenXlaType::clamp_min(const at::Tensor& self,
                                  at::Scalar min) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, c10::nullopt));
}

at::Tensor& AtenXlaType::clamp_min_(at::Tensor& self, at::Scalar min) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, min, c10::nullopt);
  return self;
}

at::Tensor AtenXlaType::clone(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::clone(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::constant_pad_nd(const at::Tensor& self,
                                        at::IntArrayRef pad,
                                        at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::constant_pad_nd(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(pad), value));
}

at::Tensor AtenXlaType::contiguous(const at::Tensor& self,
                                   at::MemoryFormat memory_format) const {
  return self;
}

at::Tensor AtenXlaType::conv2d(const at::Tensor& input,
                               const at::Tensor& weight, const at::Tensor& bias,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, int64_t groups) const {
  // Dilated or grouped convolutions aren't lowered to XLA yet.
  if (IsNonTrivialDilation(dilation) || groups != 1) {
    return AtenXlaTypeBase::conv2d(input, weight, bias, stride, padding,
                                   dilation, groups);
  }
  if (bias.defined()) {
    return bridge::AtenFromXlaTensor(XLATensor::conv2d(
        bridge::GetXlaTensor(input), bridge::GetXlaTensor(weight),
        bridge::GetXlaTensor(bias), XlaHelpers::I64List(stride),
        XlaHelpers::I64List(padding)));
  } else {
    return bridge::AtenFromXlaTensor(XLATensor::conv2d(
        bridge::GetXlaTensor(input), bridge::GetXlaTensor(weight),
        XlaHelpers::I64List(stride), XlaHelpers::I64List(padding)));
  }
}

at::Tensor AtenXlaType::conv_transpose2d(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef output_padding, int64_t groups,
    at::IntArrayRef dilation) const {
  // Dilated or grouped transposed convolutions aren't lowered to XLA yet.
  if (IsNonTrivialPadding(output_padding) || IsNonTrivialDilation(dilation) ||
      groups != 1) {
    return AtenXlaTypeBase::conv_transpose2d(
        input, weight, bias, stride, padding, output_padding, groups, dilation);
  }
  if (bias.defined()) {
    return bridge::AtenFromXlaTensor(XLATensor::conv_transpose2d(
        /*input=*/bridge::GetXlaTensor(input),
        /*weight=*/bridge::GetXlaTensor(weight),
        /*bias=*/bridge::GetXlaTensor(bias),
        /*stride=*/XlaHelpers::I64List(stride),
        /*padding=*/XlaHelpers::I64List(padding)));
  } else {
    return bridge::AtenFromXlaTensor(XLATensor::conv_transpose2d(
        /*input=*/bridge::GetXlaTensor(input),
        /*weight=*/bridge::GetXlaTensor(weight),
        /*stride=*/XlaHelpers::I64List(stride),
        /*padding=*/XlaHelpers::I64List(padding)));
  }
}

at::Tensor& AtenXlaType::copy_(at::Tensor& self, const at::Tensor& src,
                               bool non_blocking) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  c10::optional<XLATensor> src_tensor = bridge::TryGetXlaTensor(src);
  if (src_tensor) {
    XLATensor::copy_(self_tensor, *src_tensor);
  } else {
    self_tensor.SetTensor(CopyTensor(ToTensor(src), self.scalar_type()));
  }
  return self;
}

at::Tensor AtenXlaType::cos(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::cos(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::cos_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::cos_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::cosh(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::cosh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::cosh_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::cosh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::cosine_embedding_loss(const at::Tensor& input1,
                                              const at::Tensor& input2,
                                              const at::Tensor& target,
                                              double margin,
                                              int64_t reduction) const {
  return at::native::cosine_embedding_loss(input1, input2, target, margin,
                                           reduction);
}

at::Tensor AtenXlaType::cosine_similarity(const at::Tensor& x1,
                                          const at::Tensor& x2, int64_t dim,
                                          double eps) const {
  return at::native::cosine_similarity(x1, x2, dim, eps);
}

at::Tensor AtenXlaType::cross(const at::Tensor& self, const at::Tensor& other,
                              c10::optional<int64_t> dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cross(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other),
                       XlaHelpers::I64Optional(dim)));
}

at::Tensor AtenXlaType::cumprod(const at::Tensor& self, int64_t dim,
                                at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cumprod(bridge::GetXlaTensor(self), dim, dtype));
}

at::Tensor AtenXlaType::cumprod(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cumprod(bridge::GetXlaTensor(self), dim, c10::nullopt));
}

at::Tensor AtenXlaType::cumsum(const at::Tensor& self, int64_t dim,
                               at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cumsum(bridge::GetXlaTensor(self), dim, dtype));
}

at::Tensor AtenXlaType::cumsum(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::cumsum(bridge::GetXlaTensor(self), dim, c10::nullopt));
}

at::Tensor AtenXlaType::diag(const at::Tensor& self, int64_t diagonal) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::diag(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor AtenXlaType::diagflat(const at::Tensor& self, int64_t offset) const {
  return at::native::diagflat(self, offset);
}

at::Tensor AtenXlaType::diagonal(const at::Tensor& self, int64_t offset,
                                 int64_t dim1, int64_t dim2) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::diagonal(bridge::GetXlaTensor(self), offset, dim1, dim2));
}

at::Tensor AtenXlaType::div(const at::Tensor& self,
                            const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::div(
      self_tensor,
      bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice())));
}

at::Tensor AtenXlaType::div(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::div(bridge::GetXlaTensor(self), other));
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::div_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()));
  return self;
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::div_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::dot(const at::Tensor& self,
                            const at::Tensor& tensor) const {
  XLA_CHECK_EQ(self.dim(), 1)
      << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  XLA_CHECK_EQ(tensor.dim(), 1)
      << "dot: Expected 1-D argument tensor, but got " << tensor.dim() << "-D";
  return matmul(self, tensor);
}

at::Tensor AtenXlaType::dropout(const at::Tensor& input, double p,
                                bool /* train */) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::dropout(bridge::GetXlaTensor(input), p));
}

at::Tensor& AtenXlaType::dropout_(at::Tensor& self, double p,
                                  bool /* train */) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::dropout_(self_tensor, p);
  return self;
}

at::Tensor AtenXlaType::einsum(std::string equation,
                               at::TensorList tensors) const {
  if (tensors.size() != 2 || !ir::ops::Einsum::SupportsEquation(equation)) {
    return at::native::einsum(equation, tensors);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::einsum(equation, bridge::GetXlaTensors(tensors)));
}

at::Tensor AtenXlaType::elu(const at::Tensor& self, at::Scalar alpha,
                            at::Scalar scale, at::Scalar input_scale) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::elu(bridge::GetXlaTensor(self), alpha, scale, input_scale));
}

at::Tensor& AtenXlaType::elu_(at::Tensor& self, at::Scalar alpha,
                              at::Scalar scale, at::Scalar input_scale) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::elu_(self_tensor, alpha, scale, input_scale);
  return self;
}

at::Tensor AtenXlaType::elu_backward(const at::Tensor& grad_output,
                                     at::Scalar alpha, at::Scalar scale,
                                     at::Scalar input_scale,
                                     const at::Tensor& output) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::elu_backward(bridge::GetXlaTensor(grad_output), alpha, scale,
                              input_scale, bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::embedding(const at::Tensor& weight,
                                  const at::Tensor& indices,
                                  int64_t padding_idx, bool scale_grad_by_freq,
                                  bool sparse) const {
  // TODO: for now route to native, which dispatches supported XLA operations.
  // We need to make use of the TPU embedding core here eventually.
  return at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq,
                               sparse);
}

at::Tensor AtenXlaType::embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) const {
  return bridge::AtenFromXlaTensor(XLATensor::embedding_dense_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

at::Tensor AtenXlaType::empty(at::IntArrayRef size,
                              const at::TensorOptions& options) const {
  // PT empty*() are optimizations to avoid initializing the data when it is
  // known it will be completely rewritten. But since for us doing a zero*()
  // does not actually end up doing any memory initialization, we use that and
  // avoid going to CPU for it. A common PT pattern is indeed doing empty() plus
  // s_copy_().
  return full(size, 0, options);
}

at::Tensor AtenXlaType::empty_like(const at::Tensor& self) const {
  return full_like(self, 0);
}

at::Tensor AtenXlaType::empty_like(const at::Tensor& self,
                                   const at::TensorOptions& options) const {
  return full_like(self, 0, options);
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::eq(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::eq(const at::Tensor& self,
                           const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::eq(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::eq_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::eq_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::erf(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::erf(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::erf_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::erf_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::erfc(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::erfc(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::erfc_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::erfc_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::erfinv(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::erfinv(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::erfinv_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::erfinv_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::exp(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::exp(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::exp_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::exp_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::expand(const at::Tensor& self, at::IntArrayRef size,
                               bool implicit) const {
  return bridge::AtenFromXlaTensor(XLATensor::expand(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(size)));
}

at::Tensor AtenXlaType::expand_as(const at::Tensor& self,
                                  const at::Tensor& other) const {
  XLATensor other_tensor = bridge::GetXlaTensor(other);
  return bridge::AtenFromXlaTensor(XLATensor::expand(
      bridge::GetXlaTensor(self), other_tensor.shape().get().dimensions()));
}

at::Tensor AtenXlaType::expm1(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::expm1(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::expm1_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::expm1_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::eye(int64_t n, const at::TensorOptions& options) const {
  return eye(n, n, options);
}

at::Tensor AtenXlaType::eye(int64_t n, int64_t m,
                            const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(XLATensor::eye(
      n, m, xla_options.get_device(), xla_options.get_scalar_type()));
}

at::Tensor& AtenXlaType::fill_(at::Tensor& self, at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fill_(self_tensor, value);
  return self;
}

at::Tensor& AtenXlaType::fill_(at::Tensor& self,
                               const at::Tensor& value) const {
  XLA_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return fill_(self, value.item());
}

at::Tensor AtenXlaType::flatten(const at::Tensor& self, int64_t start_dim,
                                int64_t end_dim) const {
  return at::native::flatten(self, start_dim, end_dim);
}

at::Tensor AtenXlaType::flip(const at::Tensor& self,
                             at::IntArrayRef dims) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::flip(bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor AtenXlaType::floor(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::floor(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::floor_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::floor_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::fmod(const at::Tensor& self,
                             const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::fmod(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::fmod(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::fmod(bridge::GetXlaTensor(self), other));
}

at::Tensor& AtenXlaType::fmod_(at::Tensor& self,
                               const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fmod_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::fmod_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fmod_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::frac(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::frac(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::frac_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::frac_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::frobenius_norm(const at::Tensor& self) const {
  return at::native::frobenius_norm(self);
}

at::Tensor AtenXlaType::frobenius_norm(const at::Tensor& self,
                                       at::IntArrayRef dim,
                                       bool keepdim) const {
  return at::native::frobenius_norm(self, dim, keepdim);
}

at::Tensor AtenXlaType::full(at::IntArrayRef size, at::Scalar fill_value,
                             const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(
      XLATensor::full(XlaHelpers::I64List(size), fill_value,
                      xla_options.get_device(), xla_options.get_scalar_type()));
}

at::Tensor AtenXlaType::full_like(const at::Tensor& self,
                                  at::Scalar fill_value) const {
  XLATensor self_tensor = bridge::GetXlaTensorUnwrap(self);
  return bridge::AtenFromXlaTensor(XLATensor::full_like(
      self_tensor, fill_value, self_tensor.GetDevice(), c10::nullopt));
}

at::Tensor AtenXlaType::full_like(const at::Tensor& self, at::Scalar fill_value,
                                  const at::TensorOptions& options) const {
  XLATensor self_tensor = bridge::GetXlaTensorUnwrap(self);
  XlaOptions xla_options(options, self_tensor.GetDevice());
  return bridge::AtenFromXlaTensor(
      XLATensor::full_like(self_tensor, fill_value, xla_options.get_device(),
                           xla_options.scalar_type));
}

at::Tensor AtenXlaType::gather(const at::Tensor& self, int64_t dim,
                               const at::Tensor& index,
                               bool /* sparse_grad */) const {
  return bridge::AtenFromXlaTensor(XLATensor::gather(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::ge(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self,
                           const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::ge(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ge_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ge_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::group_norm(const at::Tensor& input, int64_t num_groups,
                                   const at::Tensor& weight,
                                   const at::Tensor& bias, double eps,
                                   bool cudnn_enabled) const {
  return at::native::group_norm(input, num_groups, weight, bias, eps,
                                cudnn_enabled);
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::gt(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::gt(const at::Tensor& self,
                           const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::gt(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::gt_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::gt_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::hamming_window(int64_t window_length,
                                       const at::TensorOptions& options) const {
  return at::native::hamming_window(window_length, options);
}

at::Tensor AtenXlaType::hamming_window(int64_t window_length, bool periodic,
                                       const at::TensorOptions& options) const {
  return at::native::hamming_window(window_length, periodic, options);
}

at::Tensor AtenXlaType::hamming_window(int64_t window_length, bool periodic,
                                       double alpha,
                                       const at::TensorOptions& options) const {
  return at::native::hamming_window(window_length, periodic, alpha, options);
}

at::Tensor AtenXlaType::hamming_window(int64_t window_length, bool periodic,
                                       double alpha, double beta,
                                       const at::TensorOptions& options) const {
  return at::native::hamming_window(window_length, periodic, alpha, beta,
                                    options);
}

at::Tensor AtenXlaType::hann_window(int64_t window_length,
                                    const at::TensorOptions& options) const {
  return at::native::hann_window(window_length, options);
}

at::Tensor AtenXlaType::hann_window(int64_t window_length, bool periodic,
                                    const at::TensorOptions& options) const {
  return at::native::hann_window(window_length, periodic, options);
}

at::Tensor AtenXlaType::hardshrink(const at::Tensor& self,
                                   at::Scalar lambda) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::hardshrink(bridge::GetXlaTensor(self), lambda));
}

at::Tensor AtenXlaType::hardshrink_backward(const at::Tensor& grad_out,
                                            const at::Tensor& self,
                                            at::Scalar lambda) const {
  return bridge::AtenFromXlaTensor(XLATensor::hardshrink_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(self), lambda));
}

at::Tensor AtenXlaType::hardtanh(const at::Tensor& self, at::Scalar min_val,
                                 at::Scalar max_val) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min_val, max_val));
}

at::Tensor& AtenXlaType::hardtanh_(at::Tensor& self, at::Scalar min_val,
                                   at::Scalar max_val) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, min_val, max_val);
  return self;
}

at::Tensor AtenXlaType::hardtanh_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          at::Scalar min_val,
                                          at::Scalar max_val) const {
  return bridge::AtenFromXlaTensor(XLATensor::hardtanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), min_val,
      max_val));
}

at::Tensor AtenXlaType::hinge_embedding_loss(const at::Tensor& self,
                                             const at::Tensor& target,
                                             double margin,
                                             int64_t reduction) const {
  return at::native::hinge_embedding_loss(self, target, margin, reduction);
}

at::Tensor AtenXlaType::index(const at::Tensor& self,
                              at::TensorList indices) const {
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  return bridge::AtenFromXlaTensor(
      XLATensor::index(bridge::GetXlaTensor(canonical_index_info.base),
                       bridge::GetXlaTensors(canonical_index_info.indices),
                       canonical_index_info.start_dim));
}

at::Tensor AtenXlaType::index_add(const at::Tensor& self, int64_t dim,
                                  const at::Tensor& index,
                                  const at::Tensor& source) const {
  return bridge::AtenFromXlaTensor(XLATensor::index_add(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
      bridge::GetXlaTensor(source)));
}

at::Tensor& AtenXlaType::index_add_(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& source) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_add_(self_tensor, dim, bridge::GetXlaTensor(index),
                        bridge::GetXlaTensor(source));
  return self;
}

at::Tensor AtenXlaType::index_copy(const at::Tensor& self, int64_t dim,
                                   const at::Tensor& index,
                                   const at::Tensor& source) const {
  return bridge::AtenFromXlaTensor(XLATensor::index_copy(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
      bridge::GetXlaTensor(source)));
}

at::Tensor& AtenXlaType::index_copy_(at::Tensor& self, int64_t dim,
                                     const at::Tensor& index,
                                     const at::Tensor& source) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_copy_(self_tensor, dim, bridge::GetXlaTensor(index),
                         bridge::GetXlaTensor(source));
  return self;
}

at::Tensor AtenXlaType::index_fill(const at::Tensor& self, int64_t dim,
                                   const at::Tensor& index,
                                   at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::index_fill(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index), value));
}

at::Tensor AtenXlaType::index_fill(const at::Tensor& self, int64_t dim,
                                   const at::Tensor& index,
                                   const at::Tensor& value) const {
  return bridge::AtenFromXlaTensor(XLATensor::index_fill(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
      bridge::GetXlaTensor(value)));
}

at::Tensor& AtenXlaType::index_fill_(at::Tensor& self, int64_t dim,
                                     const at::Tensor& index,
                                     at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index), value);
  return self;
}

at::Tensor& AtenXlaType::index_fill_(at::Tensor& self, int64_t dim,
                                     const at::Tensor& index,
                                     const at::Tensor& value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index),
                         bridge::GetXlaTensor(value));
  return self;
}

at::Tensor AtenXlaType::index_put(const at::Tensor& self,
                                  at::TensorList indices,
                                  const at::Tensor& values,
                                  bool accumulate) const {
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  return bridge::AtenFromXlaTensor(XLATensor::index_put(
      bridge::GetXlaTensor(canonical_index_info.base),
      bridge::GetXlaTensors(canonical_index_info.indices),
      canonical_index_info.start_dim, bridge::GetXlaTensor(values), accumulate,
      canonical_index_info.result_permutation));
}

at::Tensor& AtenXlaType::index_put_(at::Tensor& self, at::TensorList indices,
                                    const at::Tensor& values,
                                    bool accumulate) const {
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_put_(
      self_tensor, bridge::GetXlaTensor(canonical_index_info.base),
      bridge::GetXlaTensors(canonical_index_info.indices),
      canonical_index_info.start_dim, bridge::GetXlaTensor(values), accumulate,
      canonical_index_info.result_permutation);
  return self;
}

at::Tensor AtenXlaType::index_select(const at::Tensor& self, int64_t dim,
                                     const at::Tensor& index) const {
  return bridge::AtenFromXlaTensor(XLATensor::index_select(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor AtenXlaType::instance_norm(const at::Tensor& input,
                                      const at::Tensor& weight,
                                      const at::Tensor& bias,
                                      const at::Tensor& running_mean,
                                      const at::Tensor& running_var,
                                      bool use_input_stats, double momentum,
                                      double eps, bool cudnn_enabled) const {
  if (cudnn_enabled || !use_input_stats) {
    return AtenXlaTypeBase::instance_norm(input, weight, bias, running_mean,
                                          running_var, use_input_stats,
                                          momentum, eps, cudnn_enabled);
  }
  return at::native::instance_norm(input, weight, bias, running_mean,
                                   running_var, use_input_stats, momentum, eps,
                                   cudnn_enabled);
}

at::Tensor AtenXlaType::kl_div(const at::Tensor& self, const at::Tensor& target,
                               int64_t reduction) const {
  return at::native::kl_div(self, target, reduction);
}

at::Tensor AtenXlaType::kl_div_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        int64_t reduction) const {
  return bridge::AtenFromXlaTensor(XLATensor::kl_div_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::kthvalue(const at::Tensor& self,
                                                         int64_t k, int64_t dim,
                                                         bool keepdim) const {
  auto results =
      XLATensor::kthvalue(bridge::GetXlaTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::layer_norm(const at::Tensor& input,
                                   at::IntArrayRef normalized_shape,
                                   const at::Tensor& weight,
                                   const at::Tensor& bias, double eps,
                                   bool cudnn_enable) const {
  return at::native::layer_norm(input, normalized_shape, weight, bias, eps,
                                cudnn_enable);
}

at::Tensor AtenXlaType::le(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::le(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::le(const at::Tensor& self,
                           const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::le(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::le_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::le_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::leaky_relu(const at::Tensor& self,
                                   at::Scalar negative_slope) const {
  return bridge::AtenFromXlaTensor(XLATensor::leaky_relu(
      bridge::GetXlaTensor(self), negative_slope.to<double>()));
}

at::Tensor& AtenXlaType::leaky_relu_(at::Tensor& self,
                                     at::Scalar negative_slope) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::leaky_relu_(self_tensor, negative_slope.to<double>());
  return self;
}

at::Tensor AtenXlaType::leaky_relu_backward(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            at::Scalar negative_slope) const {
  return bridge::AtenFromXlaTensor(XLATensor::leaky_relu_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      negative_slope.to<double>()));
}

at::Tensor AtenXlaType::linear(const at::Tensor& input,
                               const at::Tensor& weight,
                               const at::Tensor& bias) const {
  return at::native::linear(input, weight, bias);
}

at::Tensor AtenXlaType::log(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::log(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::log10(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::log_base(
      bridge::GetXlaTensor(self), ir::OpKind(at::aten::log10), 10.0));
}

at::Tensor& AtenXlaType::log10_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log_base_(self_tensor, ir::OpKind(at::aten::log10), 10.0);
  return self;
}

at::Tensor AtenXlaType::log1p(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::log1p(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::log1p_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log1p_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::log2(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::log_base(
      bridge::GetXlaTensor(self), ir::OpKind(at::aten::log2), 2.0));
}

at::Tensor& AtenXlaType::log2_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log_base_(self_tensor, ir::OpKind(at::aten::log2), 2.0);
  return self;
}

at::Tensor& AtenXlaType::log_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::log_sigmoid(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::log_sigmoid(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::log_sigmoid_backward(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             const at::Tensor& buffer) const {
  return bridge::AtenFromXlaTensor(XLATensor::log_sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::log_sigmoid_forward(
    const at::Tensor& self) const {
  auto result_tuple =
      XLATensor::log_sigmoid_forward(bridge::GetXlaTensor(self));
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromXlaTensor(std::get<1>(result_tuple)));
}

at::Tensor AtenXlaType::log_softmax(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::log_softmax(bridge::GetXlaTensor(self), dim));
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::lt(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::lt(const at::Tensor& self,
                           const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::lt(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::lt_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::lt_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::margin_ranking_loss(const at::Tensor& input1,
                                            const at::Tensor& input2,
                                            const at::Tensor& target,
                                            double margin,
                                            int64_t reduction) const {
  return at::native::margin_ranking_loss(input1, input2, target, margin,
                                         reduction);
}

at::Tensor AtenXlaType::masked_fill(const at::Tensor& self,
                                    const at::Tensor& mask,
                                    at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::masked_fill(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(mask), value));
}

at::Tensor AtenXlaType::masked_fill(const at::Tensor& self,
                                    const at::Tensor& mask,
                                    const at::Tensor& value) const {
  XLA_CHECK_EQ(value.dim(), 0) << "masked_fill only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill(self, mask, value.item());
}

at::Tensor& AtenXlaType::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                      at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::masked_fill_(self_tensor, bridge::GetXlaTensor(mask), value);
  return self;
}

at::Tensor& AtenXlaType::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                      const at::Tensor& value) const {
  XLA_CHECK_EQ(value.dim(), 0) << "masked_fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill_(self, mask, value.item());
}

at::Tensor AtenXlaType::matmul(const at::Tensor& self,
                               const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::matmul(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::max(const at::Tensor& self,
                            const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::max(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::max(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::max(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max(const at::Tensor& self,
                                                    int64_t dim,
                                                    bool keepdim) const {
  auto outputs = XLATensor::max(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor AtenXlaType::max_pool1d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) const {
  // Lowering when dilation is non-trivial or ceil_mode is set not supported.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool1d(self, kernel_size, stride, padding,
                                       dilation, ceil_mode);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/1,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::max_pool2d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) const {
  // Lowering when dilation is non-trivial or ceil_mode is set not supported.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool2d(self, kernel_size, stride, padding,
                                       dilation, ceil_mode);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) const {
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  // TODO(asuhan): Here we return a placeholder tensor for the indices we hope
  // to never evaluate, which works for the backward of max_pool2d. However, the
  // user could request the indices to be returned, in which case we'd throw. We
  // need to either provide a lowering or improve our infrastructure to be able
  // to route to ATen the evaluation of outputs we hope to be unused.
  XLATensor result = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding));
  xla::Shape indices_shape = result.shape();
  indices_shape.set_element_type(xla::PrimitiveType::S64);
  XLATensor indices_not_supported = XLATensor::not_supported(
      "aten::max_pool2d_with_indices.indices", indices_shape,
      bridge::GetXlaTensor(self).GetDevice());
  return std::make_tuple(bridge::AtenFromXlaTensor(result),
                         bridge::AtenFromXlaTensor(indices_not_supported));
}

at::Tensor AtenXlaType::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) const {
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool2d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
        indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::max_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) const {
  // Lowering when dilation is non-trivial or ceil_mode is set not supported.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool3d(self, kernel_size, stride, padding,
                                       dilation, ceil_mode);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) const {
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool3d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
        indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) const {
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool3d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  // TODO(asuhan): Here we return a placeholder tensor for the indices we hope
  // to never evaluate, which works for the backward of max_pool3d. However, the
  // user could request the indices to be returned, in which case we'd throw. We
  // need to either provide a lowering or improve our infrastructure to be able
  // to route to ATen the evaluation of outputs we hope to be unused.
  XLATensor result = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding));
  xla::Shape indices_shape = result.shape();
  indices_shape.set_element_type(xla::PrimitiveType::S64);
  XLATensor indices_not_supported = XLATensor::not_supported(
      "aten::max_pool3d_with_indices.indices", indices_shape,
      bridge::GetXlaTensor(self).GetDevice());
  return std::make_tuple(bridge::AtenFromXlaTensor(result),
                         bridge::AtenFromXlaTensor(indices_not_supported));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self,
                             at::ScalarType dtype) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions*/ false, dtype));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions*/ false, c10::nullopt));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self, at::IntArrayRef dim,
                             bool keepdim, at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions*/ keepdim, dtype));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self, at::IntArrayRef dim,
                             bool keepdim) const {
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions*/ keepdim, c10::nullopt));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self, at::IntArrayRef dim,
                             at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions*/ false, dtype));
}

std::vector<at::Tensor> AtenXlaType::meshgrid(at::TensorList tensors) const {
  return at::native::meshgrid(tensors);
}

at::Tensor AtenXlaType::min(const at::Tensor& self,
                            const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::min(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::min(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::min(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::min(const at::Tensor& self,
                                                    int64_t dim,
                                                    bool keepdim) const {
  auto outputs = XLATensor::min(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor AtenXlaType::mm(const at::Tensor& self,
                           const at::Tensor& mat2) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::mm(/*input=*/bridge::GetXlaTensor(self),
                    /*weight=*/bridge::GetXlaTensor(mat2)));
}

at::Tensor AtenXlaType::mul(const at::Tensor& self,
                            const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::mul(
      self_tensor,
      bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice())));
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::mul(bridge::GetXlaTensor(self), other));
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::mul_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()));
  return self;
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::mul_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::narrow(const at::Tensor& self, int64_t dim,
                               int64_t start, int64_t length) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::narrow(bridge::GetXlaTensor(self), dim, start, length));
}

at::Tensor AtenXlaType::narrow_copy(const at::Tensor& self, int64_t dim,
                                    int64_t start, int64_t length) const {
  return at::native::narrow_copy_dense(self, dim, start, length);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::native_batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    bool training, double momentum, double eps) const {
  if (!training) {
    return AtenXlaTypeBase::native_batch_norm(input, weight, bias, running_mean,
                                              running_var, training, momentum,
                                              eps);
  }
  XLATensor input_tensor = bridge::GetXlaTensorUnwrap(input);
  const Device& device = input_tensor.GetDevice();
  auto outputs = XLATensor::native_batch_norm(
      bridge::GetXlaTensor(input), bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(bias, device), momentum, eps);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& running_mean,
    const at::Tensor& running_var, const at::Tensor& save_mean,
    const at::Tensor& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) const {
  if (input.dim() != 4 || !train) {
    return AtenXlaTypeBase::native_batch_norm_backward(
        grad_out, input, weight, running_mean, running_var, save_mean,
        save_invstd, train, eps, output_mask);
  }
  at::Tensor undefined;
  XLATensor grad_out_tensor = bridge::GetXlaTensorUnwrap(grad_out);
  const Device& device = grad_out_tensor.GetDevice();
  auto gradients = XLATensor::native_batch_norm_backward(
      /*grad_out=*/bridge::GetXlaTensor(grad_out),
      /*input=*/bridge::GetXlaTensor(input),
      /*weight=*/bridge::GetOrCreateXlaTensor(weight, device),
      /*save_mean=*/bridge::GetXlaTensor(save_mean),
      /*save_invstd=*/bridge::GetXlaTensor(save_invstd),
      /*eps=*/eps);
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromXlaTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromXlaTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromXlaTensor(std::get<2>(gradients))
                     : undefined);
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::ne(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::ne(const at::Tensor& self,
                           const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::ne(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ne_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ne_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::neg(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::neg(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::neg_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::neg_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::nll_loss(const at::Tensor& self,
                                 const at::Tensor& target,
                                 const at::Tensor& weight, int64_t reduction,
                                 int64_t ignore_index) const {
  if (reduction != Reduction::Mean || ignore_index >= 0 || weight.defined()) {
    return AtenXlaTypeBase::nll_loss(self, target, weight, reduction,
                                     ignore_index);
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target)));
}

at::Tensor AtenXlaType::nll_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) const {
  if (reduction != Reduction::Mean || ignore_index >= 0 || weight.defined()) {
    return AtenXlaTypeBase::nll_loss_backward(grad_output, self, target, weight,
                                              reduction, ignore_index,
                                              total_weight);
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss_backward(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target, const at::Tensor& weight,
    int64_t reduction, int64_t ignore_index) const {
  if (weight.defined()) {
    return AtenXlaTypeBase::nll_loss_forward(self, target, weight, reduction,
                                             ignore_index);
  }
  at::Tensor total_weight =
      at::ones({}, at::TensorOptions(at::ScalarType::Float));
  return std::make_tuple(
      nll_loss(self, target, weight, reduction, ignore_index), total_weight);
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             c10::optional<at::Scalar> p,
                             at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, dtype, {}, /*keepdim=*/false));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self, at::Scalar p) const {
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, c10::nullopt, {}, /*keepdim=*/false));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             c10::optional<at::Scalar> p, at::IntArrayRef dim,
                             bool keepdim, at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::norm(bridge::GetXlaTensor(self), p, dtype, dim, keepdim));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             c10::optional<at::Scalar> p, at::IntArrayRef dim,
                             bool keepdim) const {
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, c10::nullopt, dim, keepdim));
}

at::Tensor AtenXlaType::nuclear_norm(const at::Tensor& self,
                                     bool keepdim) const {
  return at::native::nuclear_norm(self, keepdim);
}

at::Tensor AtenXlaType::ones(at::IntArrayRef size,
                             const at::TensorOptions& options) const {
  return full(size, 1, options);
}

at::Tensor AtenXlaType::ones_like(const at::Tensor& self) const {
  return full_like(self, 1);
}

at::Tensor AtenXlaType::ones_like(const at::Tensor& self,
                                  const at::TensorOptions& options) const {
  return full_like(self, 1, options);
}

at::Tensor AtenXlaType::pairwise_distance(const at::Tensor& x1,
                                          const at::Tensor& x2, double p,
                                          double eps, bool keepdim) const {
  return at::native::pairwise_distance(x1, x2, p, eps, keepdim);
}

at::Tensor AtenXlaType::permute(const at::Tensor& self,
                                at::IntArrayRef dims) const {
  return bridge::AtenFromXlaTensor(XLATensor::permute(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor AtenXlaType::pixel_shuffle(const at::Tensor& self,
                                      int64_t upscale_factor) const {
  return at::native::pixel_shuffle(self, upscale_factor);
}

at::Tensor AtenXlaType::pinverse(const at::Tensor& self, double rcond) const {
  return at::native::pinverse(self, rcond);
}

at::Tensor AtenXlaType::pow(const at::Tensor& self, at::Scalar exponent) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::pow(bridge::GetXlaTensor(self), exponent));
}

at::Tensor AtenXlaType::pow(const at::Tensor& self,
                            const at::Tensor& exponent) const {
  return bridge::AtenFromXlaTensor(XLATensor::pow(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(exponent)));
}

at::Tensor AtenXlaType::pow(at::Scalar self, const at::Tensor& exponent) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::pow(self, bridge::GetXlaTensor(exponent)));
}

at::Tensor& AtenXlaType::pow_(at::Tensor& self, at::Scalar exponent) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::pow_(self_tensor, exponent);
  return self;
}

at::Tensor& AtenXlaType::pow_(at::Tensor& self,
                              const at::Tensor& exponent) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::pow_(self_tensor, bridge::GetXlaTensor(exponent));
  return self;
}

at::Tensor AtenXlaType::prod(const at::Tensor& self,
                             at::ScalarType dtype) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::prod(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenXlaType::prod(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::prod(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, c10::nullopt));
}

at::Tensor AtenXlaType::prod(const at::Tensor& self, int64_t dim, bool keepdim,
                             at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::prod(bridge::GetXlaTensor(self), {dim}, keepdim, dtype));
}

at::Tensor AtenXlaType::prod(const at::Tensor& self, int64_t dim,
                             bool keepdim) const {
  return bridge::AtenFromXlaTensor(XLATensor::prod(
      bridge::GetXlaTensor(self), {dim}, keepdim, c10::nullopt));
}

at::Tensor AtenXlaType::prod(const at::Tensor& self, int64_t dim,
                             at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::prod(bridge::GetXlaTensor(self), {dim},
                      /*keep_reduced_dimensions=*/false, dtype));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::qr(
    const at::Tensor& self) const {
  // Currently, ATen doesn't have a full_matrices option. Once this
  // is added, this API will have to be changed.
  auto results = XLATensor::qr(bridge::GetXlaTensor(self), false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::randperm(int64_t n,
                                 const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(XLATensor::randperm(
      n, xla_options.get_device(), xla_options.get_scalar_type(at::kLong)));
}

at::Tensor AtenXlaType::randperm(int64_t n, at::Generator* generator,
                                 const at::TensorOptions& options) const {
  if (generator != nullptr) {
    return AtenXlaTypeBase::randperm(n, generator, options);
  }
  return randperm(n, options);
}

at::Tensor AtenXlaType::reciprocal(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::reciprocal(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::reciprocal_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::reciprocal_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::relu(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::relu(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::relu_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::relu_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::remainder(const at::Tensor& self,
                                  const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::remainder(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::remainder(const at::Tensor& self,
                                  at::Scalar other) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::remainder(bridge::GetXlaTensor(self), other));
}

at::Tensor& AtenXlaType::remainder_(at::Tensor& self,
                                    const at::Tensor& other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::remainder_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::remainder_(at::Tensor& self, at::Scalar other) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::remainder_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::repeat(const at::Tensor& self,
                               at::IntArrayRef repeats) const {
  return bridge::AtenFromXlaTensor(XLATensor::repeat(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(repeats)));
}

at::Tensor AtenXlaType::reshape(const at::Tensor& self,
                                at::IntArrayRef shape) const {
  return bridge::AtenFromXlaTensor(XLATensor::reshape(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(shape)));
}

at::Tensor AtenXlaType::rsqrt(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::rsqrt(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::rsqrt_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::rsqrt_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::rsub(const at::Tensor& self, const at::Tensor& other,
                             at::Scalar alpha) const {
  return bridge::AtenFromXlaTensor(XLATensor::rsub(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other), alpha));
}

at::Tensor AtenXlaType::rsub(const at::Tensor& self, at::Scalar other,
                             at::Scalar alpha) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::rsub(bridge::GetXlaTensor(self), other, alpha));
}

at::Tensor AtenXlaType::scatter(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index,
                                const at::Tensor& src) const {
  return bridge::AtenFromXlaTensor(XLATensor::scatter(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
      bridge::GetXlaTensor(src)));
}

at::Tensor AtenXlaType::scatter(const at::Tensor& self, int64_t dim,
                                const at::Tensor& index,
                                at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::scatter(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index), value));
}

at::Tensor& AtenXlaType::scatter_(at::Tensor& self, int64_t dim,
                                  const at::Tensor& index,
                                  const at::Tensor& src) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::scatter_(self_tensor, dim, bridge::GetXlaTensor(index),
                      bridge::GetXlaTensor(src));
  return self;
}

at::Tensor& AtenXlaType::scatter_(at::Tensor& self, int64_t dim,
                                  const at::Tensor& index,
                                  at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::scatter_(self_tensor, dim, bridge::GetXlaTensor(index), value);
  return self;
}

at::Tensor AtenXlaType::select(const at::Tensor& self, int64_t dim,
                               int64_t index) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::select(bridge::GetXlaTensor(self), dim, index));
}

at::Tensor AtenXlaType::selu(const at::Tensor& self) const {
  return at::native::selu(self);
}

at::Tensor& AtenXlaType::selu_(at::Tensor& self) const {
  return at::native::selu_(self);
}

at::Tensor AtenXlaType::sigmoid(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::sigmoid(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sigmoid_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sigmoid_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::sigmoid_backward(const at::Tensor& grad_output,
                                         const at::Tensor& output) const {
  return bridge::AtenFromXlaTensor(XLATensor::sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::sign(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::sign(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sign_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sign_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::sin(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::sin(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sin_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sin_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::sinh(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::sinh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sinh_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sinh_(self_tensor);
  return self;
}

int64_t AtenXlaType::size(const at::Tensor& self, int64_t dim) const {
  return bridge::GetXlaTensor(self).size(dim);
}

at::Tensor AtenXlaType::slice(const at::Tensor& self, int64_t dim,
                              int64_t start, int64_t end, int64_t step) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::slice(bridge::GetXlaTensor(self), dim, start, end, step));
}

at::Tensor AtenXlaType::smooth_l1_loss(const at::Tensor& self,
                                       const at::Tensor& target,
                                       int64_t reduction) const {
  return bridge::AtenFromXlaTensor(XLATensor::smooth_l1_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target), reduction));
}

at::Tensor AtenXlaType::smooth_l1_loss_backward(const at::Tensor& grad_output,
                                                const at::Tensor& self,
                                                const at::Tensor& target,
                                                int64_t reduction) const {
  return bridge::AtenFromXlaTensor(XLATensor::smooth_l1_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction));
}

at::Tensor AtenXlaType::softmax(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::softmax(bridge::GetXlaTensor(self), dim));
}

at::Tensor AtenXlaType::softplus(const at::Tensor& self, at::Scalar beta,
                                 at::Scalar threshold) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::softplus(bridge::GetXlaTensor(self), beta, threshold));
}

at::Tensor AtenXlaType::softplus_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          at::Scalar beta, at::Scalar threshold,
                                          const at::Tensor& output) const {
  return bridge::AtenFromXlaTensor(XLATensor::softplus_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), beta,
      threshold, bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::softshrink(const at::Tensor& self,
                                   at::Scalar lambda) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::softshrink(bridge::GetXlaTensor(self), lambda));
}

at::Tensor AtenXlaType::softshrink_backward(const at::Tensor& grad_out,
                                            const at::Tensor& self,
                                            at::Scalar lambda) const {
  return bridge::AtenFromXlaTensor(XLATensor::softshrink_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(self), lambda));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::sort(const at::Tensor& self,
                                                     int64_t dim,
                                                     bool descending) const {
  auto results = XLATensor::topk(bridge::GetXlaTensor(self), self.size(dim),
                                 dim, descending, true);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::vector<at::Tensor> AtenXlaType::split(const at::Tensor& self,
                                           int64_t split_size,
                                           int64_t dim) const {
  auto xla_tensors =
      XLATensor::split(bridge::GetXlaTensor(self), split_size, dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

std::vector<at::Tensor> AtenXlaType::split_with_sizes(
    const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) const {
  auto xla_tensors = XLATensor::split_with_sizes(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(split_sizes), dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

at::Tensor AtenXlaType::sqrt(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::sqrt(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sqrt_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sqrt_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::squeeze(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::squeeze(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::squeeze(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::squeeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::squeeze_(self_tensor);
  return self;
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self, int64_t dim) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::squeeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenXlaType::stack(at::TensorList tensors, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::stack(bridge::GetXlaTensors(tensors), dim));
}

at::Tensor AtenXlaType::sub(const at::Tensor& self, const at::Tensor& other,
                            at::Scalar alpha) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::sub(
      self_tensor, bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()),
      alpha));
}

at::Tensor AtenXlaType::sub(const at::Tensor& self, at::Scalar other,
                            at::Scalar alpha) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::sub(bridge::GetXlaTensor(self), other, alpha));
}

at::Tensor& AtenXlaType::sub_(at::Tensor& self, const at::Tensor& other,
                              at::Scalar alpha) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sub_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()),
                  alpha);
  return self;
}

at::Tensor& AtenXlaType::sub_(at::Tensor& self, at::Scalar other,
                              at::Scalar alpha) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sub_(self_tensor, other, alpha);
  return self;
}

at::Tensor AtenXlaType::sum(const at::Tensor& self,
                            at::ScalarType dtype) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenXlaType::sum(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, c10::nullopt));
}

at::Tensor AtenXlaType::sum(const at::Tensor& self, at::IntArrayRef dim,
                            bool keepdim, at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::sum(bridge::GetXlaTensor(self),
                     xla::util::ToVector<xla::int64>(dim), keepdim, dtype));
}

at::Tensor AtenXlaType::sum(const at::Tensor& self, at::IntArrayRef dim,
                            bool keepdim) const {
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim), keepdim,
      c10::nullopt));
}

at::Tensor AtenXlaType::sum(const at::Tensor& self, at::IntArrayRef dim,
                            at::ScalarType dtype) const {
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenXlaType::sum_to_size(const at::Tensor& self,
                                    at::IntArrayRef size) const {
  return at::native::sum_to_size(self, size);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::svd(
    const at::Tensor& self, bool some, bool compute_uv) const {
  auto results = XLATensor::svd(bridge::GetXlaTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)),
                         bridge::AtenFromXlaTensor(std::get<2>(results)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::symeig(const at::Tensor& self,
                                                       bool eigenvectors,
                                                       bool upper) const {
  auto results =
      XLATensor::symeig(bridge::GetXlaTensor(self), eigenvectors, upper);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::t(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::transpose(bridge::GetXlaTensor(self), 0, 1));
}

at::Tensor& AtenXlaType::t_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor AtenXlaType::tan(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::tan(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::tan_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tan_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::tanh(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(XLATensor::tanh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::tanh_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tanh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::tanh_backward(const at::Tensor& grad_output,
                                      const at::Tensor& output) const {
  return bridge::AtenFromXlaTensor(XLATensor::tanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::tensordot(const at::Tensor& self,
                                  const at::Tensor& other,
                                  at::IntArrayRef dims_self,
                                  at::IntArrayRef dims_other) const {
  return at::native::tensordot(self, other, dims_self, dims_other);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::thnn_conv2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& weight, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor& finput,
    const at::Tensor& fgrad_input, std::array<bool, 3> output_mask) const {
  at::Tensor undefined;
  auto gradients = XLATensor::conv2d_backward(
      /*out_backprop=*/bridge::GetXlaTensor(grad_output),
      /*input=*/bridge::GetXlaTensor(self),
      /*weight=*/bridge::GetXlaTensor(weight),
      /*stride=*/XlaHelpers::I64List(stride),
      /*padding=*/XlaHelpers::I64List(padding));
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromXlaTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromXlaTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromXlaTensor(std::get<2>(gradients))
                     : undefined);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::thnn_conv2d_forward(
    const at::Tensor& self, const at::Tensor& weight,
    at::IntArrayRef kernel_size, const at::Tensor& bias, at::IntArrayRef stride,
    at::IntArrayRef padding) const {
  at::Tensor undefined = at::empty({});
  // TODO(asuhan): double check it's ok to return undefined for finput and
  // fgrad_input.
  return std::make_tuple(
      conv2d(/*input=*/self, /*weight=*/weight, /*bias=*/bias,
             /*stride=*/stride, /*padding=*/padding, /*dilation=*/{1, 1},
             /*groups=*/1),
      undefined, undefined);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::thnn_conv_transpose2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& weight, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef output_padding, at::IntArrayRef dilation,
    const at::Tensor& columns, const at::Tensor& ones,
    std::array<bool, 3> output_mask) const {
  // Dilated or grouped transposed convolutions aren't lowered to XLA yet.
  if (IsNonTrivialPadding(output_padding) || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::thnn_conv_transpose2d_backward(
        grad_output, self, weight, kernel_size, stride, padding, output_padding,
        dilation, columns, ones, output_mask);
  }
  at::Tensor undefined;
  auto gradients = XLATensor::conv_transpose2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(weight), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding));
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromXlaTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromXlaTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromXlaTensor(std::get<2>(gradients))
                     : undefined);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::thnn_conv_transpose2d_forward(
    const at::Tensor& self, const at::Tensor& weight,
    at::IntArrayRef kernel_size, const at::Tensor& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef output_padding,
    at::IntArrayRef dilation) const {
  // When there's dilation or output padding, PyTorch requires the result of
  // im2col to be returned as well, which is why we can't let this fall through.
  if (IsNonTrivialPadding(output_padding) || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::thnn_conv_transpose2d_forward(
        self, weight, kernel_size, bias, stride, padding, output_padding,
        dilation);
  }
  at::Tensor undefined = at::empty({});
  // TODO(asuhan): double check it's ok to return undefined for finput and
  // fgrad_input.
  return std::make_tuple(
      conv_transpose2d(
          /*input=*/self, /*weight=*/weight, /*bias=*/bias, /*stride=*/stride,
          /*padding=*/padding,
          /*output_padding=*/output_padding, /*groups=*/1,
          /*dilation=*/dilation),
      undefined, undefined);
}

at::Tensor AtenXlaType::threshold(const at::Tensor& self, at::Scalar threshold,
                                  at::Scalar value) const {
  return bridge::AtenFromXlaTensor(XLATensor::threshold(
      bridge::GetXlaTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor& AtenXlaType::threshold_(at::Tensor& self, at::Scalar threshold,
                                    at::Scalar value) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::threshold_(self_tensor, threshold.to<double>(),
                        value.to<double>());
  return self;
}

at::Tensor AtenXlaType::threshold_backward(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           at::Scalar threshold) const {
  return bridge::AtenFromXlaTensor(XLATensor::threshold_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      threshold.to<double>()));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::topk(const at::Tensor& self,
                                                     int64_t k, int64_t dim,
                                                     bool largest,
                                                     bool sorted) const {
  auto results =
      XLATensor::topk(bridge::GetXlaTensor(self), k, dim, largest, sorted);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::trace(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::trace(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::one_hot(const at::Tensor& self,
                                int64_t num_classes) const {
  return at::native::one_hot(self, num_classes);
}

at::Tensor AtenXlaType::transpose(const at::Tensor& self, int64_t dim0,
                                  int64_t dim1) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::transpose(bridge::GetXlaTensor(self), dim0, dim1));
}

at::Tensor& AtenXlaType::transpose_(at::Tensor& self, int64_t dim0,
                                    int64_t dim1) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::triangular_solve(
    const at::Tensor& b, const at::Tensor& A, bool upper, bool transpose,
    bool unitriangular) const {
  // Currently, ATen doesn't have a left_side option. Once this
  // is added, this API will have to be changed.
  auto results = XLATensor::triangular_solve(
      bridge::GetXlaTensor(b), bridge::GetXlaTensor(A), /*left_side=*/true,
      upper, transpose, unitriangular);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::tril(const at::Tensor& self, int64_t diagonal) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::tril(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor& AtenXlaType::tril_(at::Tensor& self, int64_t diagonal) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tril_(self_tensor, diagonal);
  return self;
}

at::Tensor AtenXlaType::triplet_margin_loss(const at::Tensor& anchor,
                                            const at::Tensor& positive,
                                            const at::Tensor& negative,
                                            double margin, double p, double eps,
                                            bool swap,
                                            int64_t reduction) const {
  return at::native::triplet_margin_loss(anchor, positive, negative, margin, p,
                                         eps, swap, reduction);
}

at::Tensor AtenXlaType::triu(const at::Tensor& self, int64_t diagonal) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::triu(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor& AtenXlaType::triu_(at::Tensor& self, int64_t diagonal) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::triu_(self_tensor, diagonal);
  return self;
}

at::Tensor AtenXlaType::trunc(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::trunc(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::trunc_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::trunc_(self_tensor);
  return self;
}

std::vector<at::Tensor> AtenXlaType::unbind(const at::Tensor& self,
                                            int64_t dim) const {
  return bridge::AtenFromXlaTensors(
      XLATensor::unbind(bridge::GetXlaTensor(self), dim));
}

at::Tensor AtenXlaType::unsqueeze(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::unsqueeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor& AtenXlaType::unsqueeze_(at::Tensor& self, int64_t dim) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenXlaType::view(const at::Tensor& self,
                             at::IntArrayRef size) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::view(bridge::GetXlaTensor(self), XlaHelpers::I64List(size)));
}

at::Tensor AtenXlaType::view_as(const at::Tensor& self,
                                const at::Tensor& other) const {
  return view(self, other.sizes());
}

at::Tensor AtenXlaType::where(const at::Tensor& condition,
                              const at::Tensor& self,
                              const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(XLATensor::where(
      bridge::GetXlaTensor(condition), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::zero_(at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::zero_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::zeros(at::IntArrayRef size,
                              const at::TensorOptions& options) const {
  return full(size, 0, options);
}

at::Tensor AtenXlaType::zeros_like(const at::Tensor& self) const {
  return full_like(self, 0);
}

at::Tensor AtenXlaType::zeros_like(const at::Tensor& self,
                                   const at::TensorOptions& options) const {
  return full_like(self, 0, options);
}

void AtenXlaType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

}  // namespace torch_xla
