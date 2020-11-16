#include "torch_xla/csrc/aten_xla_type.h"

#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>

#include <mutex>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/aten_autograd_ops.h"
#include "torch_xla/csrc/aten_tensor_ops.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/aten_xla_type_default.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/version.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't exist in AtenXlaType,
//   call at::native::func instead.
//   E.g. don't call tensor.is_floating_point() or
//   at::is_floating_point(tensor), use at::native::is_floating_point(tensor).

namespace torch_xla {
namespace {

Device GetXlaDeviceOrCurrent(const c10::optional<c10::Device>& device) {
  auto xla_device_opt = bridge::GetXlaDevice(device);
  return xla_device_opt ? *xla_device_opt : GetCurrentDevice();
}

at::ScalarType GetScalarTypeOrFloat(c10::optional<at::ScalarType> scalar_type) {
  return scalar_type ? *scalar_type : at::ScalarType::Float;
}

bool IsOperationOnType(const c10::optional<at::ScalarType>& opt_dtype,
                       at::ScalarType tensor_type, at::ScalarType type) {
  if (opt_dtype && *opt_dtype == type) {
    return true;
  }
  return tensor_type == type;
}

void CheckSubOperandTypes(at::ScalarType type1, at::ScalarType type2) {
  XLA_CHECK(type1 != at::kBool || type2 != at::kBool)
      << "Subtraction, the `-` operator, with two bool tensors is not "
         "supported. Use the `^` or `logical_xor()` operator instead.";
  XLA_CHECK(type1 != at::kBool && type2 != at::kBool)
      << "Subtraction, the `-` operator, with a bool tensor is not "
         "supported. If you are trying to invert a mask, use the `~` or "
         "`logical_not()` operator instead.";
}

c10::optional<at::ScalarType> PromoteIntegralType(
    at::ScalarType src_dtype, const c10::optional<at::ScalarType>& opt_dtype) {
  return opt_dtype.has_value()
             ? opt_dtype.value()
             : at::isIntegralType(src_dtype, /*includeBool=*/true) ? at::kLong
                                                                   : opt_dtype;
}

std::pair<XLATensor, XLATensor> GetBinaryOperands(const at::Tensor& self,
                                                  const at::Tensor& other) {
  XLATensor self_tensor;
  XLATensor other_tensor;
  auto self_xtensor = bridge::TryGetXlaTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::GetXlaTensor(other);
    self_tensor = bridge::GetOrCreateXlaTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = *self_xtensor;
    other_tensor = bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice());
  }
  return std::pair<XLATensor, XLATensor>(self_tensor, other_tensor);
}

// The input is in format of {N, C, H, W} and the output will be {H, W}.
std::vector<xla::int64> GetOutputSizeWithScale(
    absl::Span<const xla::int64> input_size,
    const c10::optional<at::ArrayRef<double>>& scale_factors,
    const c10::optional<at::IntArrayRef>& output_size) {
  if (!output_size) {
    XLA_CHECK(scale_factors);
    XLA_CHECK_EQ(scale_factors->size(), 2);
    // Calculate the output size from input_shape and scale_factors
    XLA_CHECK_EQ(input_size.size(), 4);
    xla::int64 output_h = input_size[2] * (*scale_factors)[0];
    xla::int64 output_w = input_size[3] * (*scale_factors)[1];
    return {output_h, output_w};
  }
  XLA_CHECK(!scale_factors);
  return xla::util::ToVector<xla::int64>(*output_size);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<XLATensor, XLATensor> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  XLATensor result = bin_op(operands.first, operands.second, dtype);
  return bridge::AtenFromXlaTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor result = bin_op(self_tensor, other, dtype);
  return bridge::AtenFromXlaTensor(result);
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Tensor& other) {
  at::ScalarType resultType = at::result_type(self, other);
  XLA_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Scalar& other) {
  at::ScalarType resultType = at::result_type(self, other);
  XLA_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

void AtenInitialize() {
  TF_VLOG(1) << "PyTorch GIT revision: " << TORCH_GITREV;
  TF_VLOG(1) << "XLA GIT revision: " << XLA_GITREV;

  XLATensorImpl::AtenInitialize();
}

}  // namespace

at::Tensor& AtenXlaType::__ilshift__(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__ilshift__(at::Tensor& self,
                                     const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ilshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::__irshift__(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__irshift__(at::Tensor& self,
                                     const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__irshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::__lshift__(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::__lshift__(xself, other, dtype);
                    });
}

at::Tensor AtenXlaType::__lshift__(const at::Tensor& self,
                                   const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::__rshift__(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::__rshift__(xself, other, dtype);
                    });
}

at::Tensor AtenXlaType::__rshift__(const at::Tensor& self,
                                   const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::_adaptive_avg_pool2d(const at::Tensor& self,
                                             at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool2d(XlaHelpers::I64List(self.sizes()),
                                    output_size_list)) {
    return AtenXlaTypeDefault::_adaptive_avg_pool2d(self, output_size);
  }
  return bridge::AtenFromXlaTensor(XLATensor::_adaptive_avg_pool2d(
      bridge::GetXlaTensor(self), output_size_list));
}

at::Tensor AtenXlaType::_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<xla::int64> output_size{grad_output.size(rank - 2),
                                      grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool2d(XlaHelpers::I64List(self.sizes()),
                                    output_size)) {
    return AtenXlaTypeDefault::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return bridge::AtenFromXlaTensor(XLATensor::_adaptive_avg_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::_copy_from(const at::Tensor& self,
                                   const at::Tensor& dst, bool non_blocking) {
  XLA_FN_COUNTER("xla::");
  auto dst_tensor = bridge::TryGetXlaTensor(dst);
  auto self_tensor = bridge::TryGetXlaTensor(self);
  if (!self_tensor) {
    static bool sync_update =
        xla::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    XLA_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    XLATensor::copy_(*dst_tensor, *self_tensor);
    bridge::ReplaceXlaTensor(dst, *dst_tensor);
  }
  return dst;
}

at::Tensor& AtenXlaType::_index_put_impl_(at::Tensor& self,
                                          at::TensorList indices,
                                          const at::Tensor& values,
                                          bool accumulate, bool /* unsafe */) {
  XLA_FN_COUNTER("xla::");
  return index_put_(self, indices, values, accumulate);
}

at::Tensor AtenXlaType::_log_softmax(const at::Tensor& self, int64_t dim,
                                     bool /* half_to_float */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::log_softmax(bridge::GetXlaTensor(self), dim, c10::nullopt));
}

at::Tensor AtenXlaType::_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    const at::Tensor& /* self */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::_pack_padded_sequence(
    const at::Tensor& input, const at::Tensor& lengths, bool batch_first) {
  XLA_FN_COUNTER("xla::");
  std::vector<at::Tensor> xla_tensors = {lengths};
  auto cpu_tensors = bridge::XlaCreateTensorList(xla_tensors);
  return at::native::_pack_padded_sequence(input, cpu_tensors[0], batch_first);
}

at::Tensor AtenXlaType::_s_where(const at::Tensor& condition,
                                 const at::Tensor& self,
                                 const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::where(
      bridge::GetXlaTensor(condition), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::_softmax(const at::Tensor& self, int64_t dim,
                                 bool /* half_to_float */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::softmax(bridge::GetXlaTensor(self), dim, c10::nullopt));
}

at::Tensor AtenXlaType::_softmax_backward_data(const at::Tensor& grad_output,
                                               const at::Tensor& output,
                                               int64_t dim,
                                               const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

at::Tensor AtenXlaType::_trilinear(const at::Tensor& i1, const at::Tensor& i2,
                                   const at::Tensor& i3,
                                   at::IntArrayRef expand1,
                                   at::IntArrayRef expand2,
                                   at::IntArrayRef expand3,
                                   at::IntArrayRef sumdim, int64_t unroll_dim) {
  XLA_FN_COUNTER("xla::");
  return at::native::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim,
                                unroll_dim);
}

at::Tensor AtenXlaType::_unsafe_view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  XLA_FN_COUNTER("xla::");
  return view(self, size);
}

at::Tensor AtenXlaType::abs(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::abs(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::abs_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::abs_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::acos(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::acos(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::acos_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::acos_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::acosh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::acosh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::acosh_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::acosh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Tensor& other,
                            at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenXlaType::add(const at::Tensor& self, at::Scalar other,
                            at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::add(xself, other, alpha, dtype);
                    });
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, const at::Tensor& other,
                              at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::add_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()),
                  alpha);
  return self;
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, at::Scalar other,
                              at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::add_(self_tensor, other, alpha);
  return self;
}

at::Tensor AtenXlaType::addcdiv(const at::Tensor& self,
                                const at::Tensor& tensor1,
                                const at::Tensor& tensor2, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::addcdiv(
      bridge::GetXlaTensor(self), value, bridge::GetXlaTensor(tensor1),
      bridge::GetXlaTensor(tensor2)));
}

at::Tensor& AtenXlaType::addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::addcdiv_(self_tensor, value, bridge::GetXlaTensor(tensor1),
                      bridge::GetXlaTensor(tensor2));
  return self;
}

at::Tensor AtenXlaType::addcmul(const at::Tensor& self,
                                const at::Tensor& tensor1,
                                const at::Tensor& tensor2, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::addcmul(
      bridge::GetXlaTensor(self), value, bridge::GetXlaTensor(tensor1),
      bridge::GetXlaTensor(tensor2)));
}

at::Tensor& AtenXlaType::addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::addcmul_(self_tensor, value, bridge::GetXlaTensor(tensor1),
                      bridge::GetXlaTensor(tensor2));
  return self;
}

at::Tensor AtenXlaType::addmm(const at::Tensor& self, const at::Tensor& mat1,
                              const at::Tensor& mat2, at::Scalar beta,
                              at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (beta.to<double>() != 1 || alpha.to<double>() != 1 ||
      !at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat1) ||
      !at::native::is_floating_point(mat2)) {
    return AtenXlaTypeDefault::addmm(self, mat1, mat2, beta, alpha);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::addmm(bridge::GetXlaTensor(mat1),
                       /*weight=*/bridge::GetXlaTensor(mat2),
                       /*bias=*/bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::alias(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return self;
}

at::Tensor AtenXlaType::all(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::all(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor AtenXlaType::all(const at::Tensor& self, int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::all(bridge::GetXlaTensor(self), {dim}, keepdim));
}

at::Tensor AtenXlaType::any(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::any(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor AtenXlaType::any(const at::Tensor& self, int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::any(bridge::GetXlaTensor(self), {dim}, keepdim));
}

at::Tensor& AtenXlaType::arange_out(at::Tensor& out, at::Scalar start,
                                    at::Scalar end, at::Scalar step) {
  XLA_FN_COUNTER("xla::");
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::arange_out(out_tensor, start, end, step, out.scalar_type());
  return out;
}

at::Tensor AtenXlaType::argmax(const at::Tensor& self,
                               c10::optional<int64_t> dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return dim ? bridge::AtenFromXlaTensor(
                   XLATensor::argmax(bridge::GetXlaTensor(self), *dim, keepdim))
             : bridge::AtenFromXlaTensor(
                   XLATensor::argmax(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::argmin(const at::Tensor& self,
                               c10::optional<int64_t> dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return dim ? bridge::AtenFromXlaTensor(
                   XLATensor::argmin(bridge::GetXlaTensor(self), *dim, keepdim))
             : bridge::AtenFromXlaTensor(
                   XLATensor::argmin(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                   at::IntArrayRef stride,
                                   c10::optional<int64_t> storage_offset) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided(self, size, stride, storage_offset);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::as_strided(self_tensor, std::move(xsize), std::move(xstride),
                            XlaHelpers::I64Optional(storage_offset)));
}

at::Tensor& AtenXlaType::as_strided_(at::Tensor& self, at::IntArrayRef size,
                                     at::IntArrayRef stride,
                                     c10::optional<int64_t> storage_offset) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided_(self, size, stride, storage_offset);
  }
  XLATensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                         XlaHelpers::I64Optional(storage_offset));
  return self;
}

at::Tensor AtenXlaType::asin(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::asin(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::asin_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::asin_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::asinh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::asinh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::asinh_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::asinh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::atan(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::atan(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::atanh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::atanh(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::atan2(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  // xla::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return AtenXlaTypeDefault::atan2(self, other);
  }
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::atan2(xself, xother, dtype);
                    });
}

at::Tensor& AtenXlaType::atan2_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  // xla::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return AtenXlaTypeDefault::atan2_(self, other);
  }
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::atan2_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::atan_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::atan_(self_tensor);
  return self;
}

at::Tensor& AtenXlaType::atanh_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::atanh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::avg_pool2d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad,
                                   c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenXlaTypeDefault::avg_pool2d(self, kernel_size, stride, padding,
                                          ceil_mode, count_include_pad,
                                          divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor AtenXlaType::avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenXlaTypeDefault::avg_pool2d_backward(
        grad_output, self, kernel_size, stride, padding, ceil_mode,
        count_include_pad, divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor AtenXlaType::avg_pool3d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding, bool ceil_mode,
                                   bool count_include_pad,
                                   c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenXlaTypeDefault::avg_pool3d(self, kernel_size, stride, padding,
                                          ceil_mode, count_include_pad,
                                          divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor AtenXlaType::avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return AtenXlaTypeDefault::avg_pool3d_backward(
        grad_output, self, kernel_size, stride, padding, ceil_mode,
        count_include_pad, divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor AtenXlaType::baddbmm(const at::Tensor& self,
                                const at::Tensor& batch1,
                                const at::Tensor& batch2, at::Scalar beta,
                                at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) ||
      !at::native::is_floating_point(batch2)) {
    return AtenXlaTypeDefault::baddbmm(self, batch1, batch2, beta, alpha);
  }
  return bridge::AtenFromXlaTensor(XLATensor::baddbmm(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(batch1),
      bridge::GetXlaTensor(batch2), beta, alpha));
}

at::Tensor& AtenXlaType::baddbmm_(at::Tensor& self, const at::Tensor& batch1,
                                  const at::Tensor& batch2, at::Scalar beta,
                                  at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) ||
      !at::native::is_floating_point(batch2)) {
    return AtenXlaTypeDefault::baddbmm_(self, batch1, batch2, beta, alpha);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::baddbmm_(self_tensor, bridge::GetXlaTensor(batch1),
                      bridge::GetXlaTensor(batch2), beta, alpha);
  return self;
}

at::Tensor AtenXlaType::bernoulli(const at::Tensor& self,
                                  c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::bernoulli(self, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::bernoulli(self_tensor));
}

at::Tensor& AtenXlaType::bernoulli_(at::Tensor& self, double p,
                                    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::bernoulli_(self, p, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor& AtenXlaType::bernoulli_(at::Tensor& self, const at::Tensor& p,
                                    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::bernoulli_(self, p, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bernoulli_(self_tensor, bridge::GetXlaTensor(p));
  return self;
}

at::Tensor AtenXlaType::binary_cross_entropy(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromXlaTensor(XLATensor::binary_cross_entropy(
      self_tensor, bridge::GetXlaTensor(target), weight_tensor, reduction));
}

at::Tensor AtenXlaType::binary_cross_entropy_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromXlaTensor(XLATensor::binary_cross_entropy_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction));
}

at::Tensor AtenXlaType::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight, int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return at::native::binary_cross_entropy_with_logits(
      self, target, IsDefined(weight) ? *weight : at::Tensor(),
      IsDefined(pos_weight) ? *pos_weight : at::Tensor(), reduction);
}

at::Tensor& AtenXlaType::bitwise_and_out(at::Tensor& out,
                                         const at::Tensor& self,
                                         at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::bitwise_and_out(out_tensor, bridge::GetXlaTensor(self), other);
  return out;
}

at::Tensor& AtenXlaType::bitwise_and_out(at::Tensor& out,
                                         const at::Tensor& self,
                                         const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::bitwise_and_out(out_tensor, bridge::GetXlaTensor(self),
                             bridge::GetXlaTensor(other));
  return out;
}

at::Tensor& AtenXlaType::bitwise_not_out(at::Tensor& out,
                                         const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bitwise_not_out(out_tensor, self_tensor);
  return out;
}

at::Tensor& AtenXlaType::bitwise_or_out(at::Tensor& out, const at::Tensor& self,
                                        at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::bitwise_or_out(out_tensor, bridge::GetXlaTensor(self), other);
  return out;
}

at::Tensor& AtenXlaType::bitwise_or_out(at::Tensor& out, const at::Tensor& self,
                                        const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::bitwise_or_out(out_tensor, bridge::GetXlaTensor(self),
                            bridge::GetXlaTensor(other));
  return out;
}

at::Tensor& AtenXlaType::bitwise_xor_out(at::Tensor& out,
                                         const at::Tensor& self,
                                         at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::bitwise_xor_out(out_tensor, bridge::GetXlaTensor(self), other);
  return out;
}

at::Tensor& AtenXlaType::bitwise_xor_out(at::Tensor& out,
                                         const at::Tensor& self,
                                         const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::bitwise_xor_out(out_tensor, bridge::GetXlaTensor(self),
                             bridge::GetXlaTensor(other));
  return out;
}

at::Tensor AtenXlaType::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return AtenXlaTypeDefault::bmm(self, mat2);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::bmm(bridge::GetXlaTensor(self), bridge::GetXlaTensor(mat2)));
}

at::Tensor AtenXlaType::cat(at::TensorList tensors, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::cat(bridge::GetXlaTensors(tensors), dim));
}

at::Tensor AtenXlaType::ceil(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::ceil(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::ceil_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ceil_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::cholesky(const at::Tensor& self, bool upper) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::cholesky(bridge::GetXlaTensor(self), upper));
}

at::Tensor AtenXlaType::clamp(const at::Tensor& self,
                              c10::optional<at::Scalar> min,
                              c10::optional<at::Scalar> max) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, max));
}

at::Tensor& AtenXlaType::clamp_(at::Tensor& self, c10::optional<at::Scalar> min,
                                c10::optional<at::Scalar> max) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, min, max);
  return self;
}

at::Tensor AtenXlaType::clamp_max(const at::Tensor& self, at::Scalar max) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), c10::nullopt, max));
}

at::Tensor& AtenXlaType::clamp_max_(at::Tensor& self, at::Scalar max) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, c10::nullopt, max);
  return self;
}

at::Tensor AtenXlaType::clamp_min(const at::Tensor& self, at::Scalar min) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, c10::nullopt));
}

at::Tensor& AtenXlaType::clamp_min_(at::Tensor& self, at::Scalar min) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, min, c10::nullopt);
  return self;
}

at::Tensor AtenXlaType::clone(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> /* memory_format */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clone(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::constant_pad_nd(const at::Tensor& self,
                                        at::IntArrayRef pad, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::constant_pad_nd(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(pad), value));
}

// This functions covers the whole convolution lowering.
at::Tensor AtenXlaType::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  XLA_FN_COUNTER("xla::");
  if (IsDefined(bias)) {
    return bridge::AtenFromXlaTensor(XLATensor::convolution_overrideable(
        bridge::GetXlaTensor(input), bridge::GetXlaTensor(weight),
        bridge::GetXlaTensor(*bias), XlaHelpers::I64List(stride),
        XlaHelpers::I64List(padding), XlaHelpers::I64List(dilation), transposed,
        XlaHelpers::I64List(output_padding), groups));
  } else {
    return bridge::AtenFromXlaTensor(XLATensor::convolution_overrideable(
        bridge::GetXlaTensor(input), bridge::GetXlaTensor(weight),
        XlaHelpers::I64List(stride), XlaHelpers::I64List(padding),
        XlaHelpers::I64List(dilation), transposed,
        XlaHelpers::I64List(output_padding), groups));
  }
}

// This functions covers the whole convolution backward lowering.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  XLA_FN_COUNTER("xla::");
  auto gradients = XLATensor::convolution_backward_overrideable(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(input),
      bridge::GetXlaTensor(weight), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), XlaHelpers::I64List(dilation), transposed,
      XlaHelpers::I64List(output_padding), groups);
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromXlaTensor(std::get<0>(gradients))
                     : at::Tensor(),
      output_mask[1] ? bridge::AtenFromXlaTensor(std::get<1>(gradients))
                     : at::Tensor(),
      output_mask[2] ? bridge::AtenFromXlaTensor(std::get<2>(gradients))
                     : at::Tensor());
}

at::Tensor AtenXlaType::cos(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::cos(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::cos_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::cos_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::cosh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::cosh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::cosh_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::cosh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::cross(const at::Tensor& self, const at::Tensor& other,
                              c10::optional<int64_t> dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::cross(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other),
                       XlaHelpers::I64Optional(dim)));
}

at::Tensor AtenXlaType::cumprod(const at::Tensor& self, int64_t dim,
                                c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  c10::optional<at::ScalarType> promoted_dtype =
      PromoteIntegralType(self_tensor.dtype(), dtype);
  if (IsOperationOnType(promoted_dtype, self_tensor.dtype(),
                        at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return AtenXlaTypeDefault::cumprod(self, dim, dtype);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::cumprod(self_tensor, dim, promoted_dtype));
}

at::Tensor AtenXlaType::cumsum(const at::Tensor& self, int64_t dim,
                               c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  if (IsOperationOnType(dtype, self_tensor.dtype(), at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return AtenXlaTypeDefault::cumsum(self, dim, dtype);
  }
  return bridge::AtenFromXlaTensor(XLATensor::cumsum(self_tensor, dim, dtype));
}

at::Tensor AtenXlaType::diag(const at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::diag(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor AtenXlaType::diagonal(const at::Tensor& self, int64_t offset,
                                 int64_t dim1, int64_t dim2) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::diagonal(bridge::GetXlaTensor(self), offset, dim1, dim2));
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  at::ScalarType dtype = at::result_type(self, other);
  auto operands = GetBinaryOperands(self, other);
  return bridge::AtenFromXlaTensor(
      XLATensor::div(operands.first, operands.second, dtype));
}

at::Tensor AtenXlaType::div(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::div(bridge::GetXlaTensor(self), other));
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::div_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()));
  return self;
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::div_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::dot(const at::Tensor& self, const at::Tensor& tensor) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(self.dim(), 1)
      << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  XLA_CHECK_EQ(tensor.dim(), 1)
      << "dot: Expected 1-D argument tensor, but got " << tensor.dim() << "-D";
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(tensor)) {
    return AtenXlaTypeDefault::dot(self, tensor);
  }
  return bridge::AtenFromXlaTensor(XLATensor::matmul(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(tensor)));
}

at::Tensor AtenXlaType::elu(const at::Tensor& self, at::Scalar alpha,
                            at::Scalar scale, at::Scalar input_scale) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::elu(bridge::GetXlaTensor(self), alpha, scale, input_scale));
}

at::Tensor& AtenXlaType::elu_(at::Tensor& self, at::Scalar alpha,
                              at::Scalar scale, at::Scalar input_scale) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::elu_(self_tensor, alpha, scale, input_scale);
  return self;
}

at::Tensor AtenXlaType::elu_backward(const at::Tensor& grad_output,
                                     at::Scalar alpha, at::Scalar scale,
                                     at::Scalar input_scale,
                                     const at::Tensor& output) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::elu_backward(bridge::GetXlaTensor(grad_output), alpha, scale,
                              input_scale, bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::embedding(const at::Tensor& weight,
                                  const at::Tensor& indices,
                                  int64_t padding_idx, bool scale_grad_by_freq,
                                  bool sparse) {
  XLA_FN_COUNTER("xla::");
  // TODO: for now route to native, which dispatches supported XLA operations.
  // We need to make use of the TPU embedding core here eventually.
  return at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq,
                               sparse);
}

at::Tensor AtenXlaType::embedding_dense_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& indices,
                                                 int64_t num_weights,
                                                 int64_t padding_idx,
                                                 bool scale_grad_by_freq) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::embedding_dense_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

at::Tensor AtenXlaType::empty(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> /* memory_format */) {
  XLA_FN_COUNTER("xla::");
  // PT empty*() are optimizations to avoid initializing the data when it is
  // known it will be completely rewritten. But since for us doing a zero*()
  // does not actually end up doing any memory initialization, we use that and
  // avoid going to CPU for it. A common PT pattern is indeed doing empty() plus
  // s_copy_().
  return bridge::AtenFromXlaTensor(XLATensor::full(
      XlaHelpers::I64List(size), 0, GetXlaDeviceOrCurrent(device),
      GetScalarTypeOrFloat(dtype)));
}

at::Tensor AtenXlaType::empty_strided(at::IntArrayRef size,
                                      at::IntArrayRef stride,
                                      c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout,
                                      c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory) {
  XLA_FN_COUNTER("xla::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::eq(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::eq(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::eq_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::eq_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::erf(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::erf(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::erf_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::erf_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::erfc(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::erfc(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::erfc_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::erfc_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::erfinv(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::erfinv(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::erfinv_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::erfinv_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::exp(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::exp(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::exp_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::exp_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::expand(const at::Tensor& self, at::IntArrayRef size,
                               bool implicit) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::expand(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(size)));
}

at::Tensor AtenXlaType::expm1(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::expm1(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::expm1_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::expm1_(self_tensor);
  return self;
}

at::Tensor& AtenXlaType::exponential_(at::Tensor& self, double lambd,
                                      c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::exponential_(self, lambd, generator);
  }
  XLA_CHECK_GE(lambd, 0.0);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::exponential_(self_tensor, lambd);
  return self;
}

at::Tensor& AtenXlaType::eye_out(at::Tensor& out, int64_t n) {
  XLA_FN_COUNTER("xla::");
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::eye_out(out_tensor, n, n);
  return out;
}

at::Tensor& AtenXlaType::eye_out(at::Tensor& out, int64_t n, int64_t m) {
  XLA_FN_COUNTER("xla::");
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::eye_out(out_tensor, n, m);
  return out;
}

at::Tensor& AtenXlaType::fill_(at::Tensor& self, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fill_(self_tensor, value);
  return self;
}

at::Tensor& AtenXlaType::fill_(at::Tensor& self, const at::Tensor& value) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return fill_(self, value.item());
}

at::Tensor AtenXlaType::flip(const at::Tensor& self, at::IntArrayRef dims) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::flip(bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor AtenXlaType::floor(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::floor(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::floor_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::floor_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::fmod(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::fmod(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::fmod(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::fmod(xself, other, dtype);
                    });
}

at::Tensor& AtenXlaType::fmod_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fmod_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::fmod_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fmod_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::frac(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::frac(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::frac_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::frac_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::gather(const at::Tensor& self, int64_t dim,
                               const at::Tensor& index,
                               bool /* sparse_grad */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::gather(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ge(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ge(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ge_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ge_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::gelu(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::gelu(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::gelu_backward(const at::Tensor& grad,
                                      const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::gelu_backward(
      bridge::GetXlaTensor(grad), bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::ger(const at::Tensor& self, const at::Tensor& vec2) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ger(bridge::GetXlaTensor(self), bridge::GetXlaTensor(vec2)));
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::gt(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::gt(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::gt_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::gt_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::hardshrink(const at::Tensor& self, at::Scalar lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::hardshrink(bridge::GetXlaTensor(self), lambda));
}

at::Tensor AtenXlaType::hardsigmoid(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::hardsigmoid(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::hardsigmoid_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::hardsigmoid_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::hardsigmoid_backward(const at::Tensor& grad_output,
                                             const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardsigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::hardshrink_backward(const at::Tensor& grad_out,
                                            const at::Tensor& self,
                                            at::Scalar lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardshrink_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(self), lambda));
}

at::Tensor AtenXlaType::hardtanh(const at::Tensor& self, at::Scalar min_val,
                                 at::Scalar max_val) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min_val, max_val));
}

at::Tensor& AtenXlaType::hardtanh_(at::Tensor& self, at::Scalar min_val,
                                   at::Scalar max_val) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::clamp_(self_tensor, min_val, max_val);
  return self;
}

at::Tensor AtenXlaType::hardtanh_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          at::Scalar min_val,
                                          at::Scalar max_val) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardtanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), min_val,
      max_val));
}

at::Tensor AtenXlaType::index(const at::Tensor& self, at::TensorList indices) {
  XLA_FN_COUNTER("xla::");
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  return bridge::AtenFromXlaTensor(
      XLATensor::index(bridge::GetXlaTensor(canonical_index_info.base),
                       bridge::GetXlaTensors(canonical_index_info.indices),
                       canonical_index_info.start_dim));
}

at::Tensor& AtenXlaType::index_add_(at::Tensor& self, int64_t dim,
                                    const at::Tensor& index,
                                    const at::Tensor& source) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_add_(self_tensor, dim, bridge::GetXlaTensor(index),
                        bridge::GetXlaTensor(source));
  return self;
}

at::Tensor& AtenXlaType::index_copy_(at::Tensor& self, int64_t dim,
                                     const at::Tensor& index,
                                     const at::Tensor& source) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_copy_(self_tensor, dim, bridge::GetXlaTensor(index),
                         bridge::GetXlaTensor(source));
  return self;
}

at::Tensor& AtenXlaType::index_fill_(at::Tensor& self, int64_t dim,
                                     const at::Tensor& index,
                                     at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index), value);
  return self;
}

at::Tensor& AtenXlaType::index_fill_(at::Tensor& self, int64_t dim,
                                     const at::Tensor& index,
                                     const at::Tensor& value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index),
                         bridge::GetXlaTensor(value));
  return self;
}

at::Tensor& AtenXlaType::index_put_(at::Tensor& self, at::TensorList indices,
                                    const at::Tensor& values, bool accumulate) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(self.scalar_type() == values.scalar_type());
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
                                     const at::Tensor& index) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::index_select(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor AtenXlaType::inverse(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::inverse(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::kl_div(const at::Tensor& self, const at::Tensor& target,
                               int64_t reduction, bool log_target) {
  XLA_FN_COUNTER("xla::");
  return at::native::kl_div(self, target, reduction, log_target);
}

at::Tensor AtenXlaType::kl_div_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        int64_t reduction, bool log_target) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::kl_div_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction, log_target));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::kthvalue(const at::Tensor& self,
                                                         int64_t k, int64_t dim,
                                                         bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::kthvalue(bridge::GetXlaTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::l1_loss(const at::Tensor& self,
                                const at::Tensor& target, int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::l1_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target), reduction));
}

at::Tensor AtenXlaType::l1_loss_backward(const at::Tensor& grad_output,
                                         const at::Tensor& self,
                                         const at::Tensor& target,
                                         int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::l1_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction));
}

at::Tensor AtenXlaType::le(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::le(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::le(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::le(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::le_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::le_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::leaky_relu(const at::Tensor& self,
                                   at::Scalar negative_slope) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::leaky_relu(
      bridge::GetXlaTensor(self), negative_slope.to<double>()));
}

at::Tensor& AtenXlaType::leaky_relu_(at::Tensor& self,
                                     at::Scalar negative_slope) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::leaky_relu_(self_tensor, negative_slope.to<double>());
  return self;
}

at::Tensor AtenXlaType::leaky_relu_backward(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            at::Scalar negative_slope,
                                            bool self_is_result) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(!self_is_result || negative_slope.to<double>() > 0.0);
  return bridge::AtenFromXlaTensor(XLATensor::leaky_relu_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      negative_slope.to<double>()));
}

at::Tensor AtenXlaType::log(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::log10(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_base(
      bridge::GetXlaTensor(self), ir::OpKind(at::aten::log10), 10.0));
}

at::Tensor& AtenXlaType::log10_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log_base_(self_tensor, ir::OpKind(at::aten::log10), 10.0);
  return self;
}

at::Tensor AtenXlaType::log1p(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::log1p(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::log1p_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log1p_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::log2(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_base(
      bridge::GetXlaTensor(self), ir::OpKind(at::aten::log2), 2.0));
}

at::Tensor& AtenXlaType::log2_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log_base_(self_tensor, ir::OpKind(at::aten::log2), 2.0);
  return self;
}

at::Tensor& AtenXlaType::log_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::log_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::log_sigmoid_backward(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             const at::Tensor& buffer) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::log_sigmoid_forward(
    const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  auto result_tuple =
      XLATensor::log_sigmoid_forward(bridge::GetXlaTensor(self));
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromXlaTensor(std::get<1>(result_tuple)));
}

at::Tensor AtenXlaType::logsumexp(const at::Tensor& self, at::IntArrayRef dim,
                                  bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::logsumexp(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions=*/keepdim));
}

at::Tensor AtenXlaType::logdet(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::logdet(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::lt(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::lt(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::lt_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::lt_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                      at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::masked_fill_(self_tensor, bridge::GetXlaTensor(mask), value);
  return self;
}

at::Tensor& AtenXlaType::masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                      const at::Tensor& value) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(value.dim(), 0) << "masked_fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill_(self, mask, value.item());
}

at::Tensor& AtenXlaType::masked_scatter_(at::Tensor& self,
                                         const at::Tensor& mask,
                                         const at::Tensor& source) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::masked_scatter_(self_tensor, bridge::GetXlaTensor(mask),
                             bridge::GetXlaTensor(source));
  return self;
}

at::Tensor AtenXlaType::masked_select(const at::Tensor& self,
                                      const at::Tensor& mask) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  // Initially make XLA handled masked_select() handling experimental, and
  // opt-in.
  if (!DebugUtil::ExperimentEnabled("masked_select")) {
    return AtenXlaTypeDefault::masked_select(self, mask);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::masked_select(self_tensor, bridge::GetXlaTensor(mask)));
}

at::Tensor AtenXlaType::max(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max(const at::Tensor& self,
                                                    int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto outputs = XLATensor::max(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor AtenXlaType::maximum(const at::Tensor& self,
                                const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::max(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> AtenXlaType::max_out(
    at::Tensor& max, at::Tensor& max_values, const at::Tensor& self,
    int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensor max_tensor = bridge::GetXlaTensor(max);
  XLATensor max_values_tensor = bridge::GetXlaTensor(max_values);
  XLATensor::max_out(max_tensor, max_values_tensor, bridge::GetXlaTensor(self),
                     dim, keepdim);
  return std::forward_as_tuple(max, max_values);
}

at::Tensor AtenXlaType::max_pool2d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  return aten_autograd_ops::MaxPool2dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeDefault::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  auto outputs = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor AtenXlaType::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeDefault::max_pool2d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
        indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
}

at::Tensor AtenXlaType::max_pool3d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  return aten_autograd_ops::MaxPool3dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeDefault::max_pool3d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
        indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeDefault::max_pool3d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  auto outputs = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor AtenXlaType::max_unpool2d(const at::Tensor& self,
                                     const at::Tensor& indices,
                                     at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max_unpool(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor AtenXlaType::max_unpool2d_backward(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& indices,
                                              at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max_unpool_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor AtenXlaType::max_unpool3d(const at::Tensor& self,
                                     const at::Tensor& indices,
                                     at::IntArrayRef output_size,
                                     at::IntArrayRef stride,
                                     at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max_unpool(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor AtenXlaType::max_unpool3d_backward(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& indices,
                                              at::IntArrayRef output_size,
                                              at::IntArrayRef stride,
                                              at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max_unpool_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self,
                             c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenXlaType::mean(const at::Tensor& self, at::IntArrayRef dim,
                             bool keepdim,
                             c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions=*/keepdim, dtype));
}

at::Tensor AtenXlaType::min(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::min(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::min(const at::Tensor& self,
                                                    int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto outputs = XLATensor::min(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor AtenXlaType::minimum(const at::Tensor& self,
                                const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::min(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> AtenXlaType::min_out(
    at::Tensor& min, at::Tensor& min_indices, const at::Tensor& self,
    int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensor min_tensor = bridge::GetXlaTensor(min);
  XLATensor min_indices_tensor = bridge::GetXlaTensor(min_indices);
  XLATensor::min_out(min_tensor, min_indices_tensor, bridge::GetXlaTensor(self),
                     dim, keepdim);
  return std::forward_as_tuple(min, min_indices);
}

at::Tensor AtenXlaType::mm(const at::Tensor& self, const at::Tensor& mat2) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return AtenXlaTypeDefault::mm(self, mat2);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::mm(/*input=*/bridge::GetXlaTensor(self),
                    /*weight=*/bridge::GetXlaTensor(mat2)));
}

at::Tensor AtenXlaType::mse_loss(const at::Tensor& self,
                                 const at::Tensor& target, int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mse_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target), reduction));
}

at::Tensor AtenXlaType::mse_loss_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Tensor& target,
                                          int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mse_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction));
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::mul(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::mul(xself, other, dtype);
                    });
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::mul_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()));
  return self;
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::mul_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::mv(const at::Tensor& self, const at::Tensor& vec) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(vec)) {
    return AtenXlaTypeDefault::mv(self, vec);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::mv(bridge::GetXlaTensor(self), bridge::GetXlaTensor(vec)));
}

at::Tensor& AtenXlaType::mv_out(at::Tensor& out, const at::Tensor& self,
                                const at::Tensor& vec) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(vec)) {
    return AtenXlaTypeDefault::mv_out(out, self, vec);
  }
  XLATensor out_tensor = bridge::GetXlaTensor(out);
  XLATensor::mv_out(out_tensor, bridge::GetXlaTensor(self),
                    bridge::GetXlaTensor(vec));
  return out;
}

at::Tensor AtenXlaType::narrow_copy(const at::Tensor& self, int64_t dim,
                                    int64_t start, int64_t length) {
  XLA_FN_COUNTER("xla::");
  return at::native::narrow_copy_dense(self, dim, start, length);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  XLA_FN_COUNTER("xla::");
  XLATensor input_tensor = bridge::GetXlaTensor(input);
  const Device& device = input_tensor.GetDevice();
  XLATensor running_mean_tensor =
      bridge::GetOrCreateXlaTensor(running_mean, device);
  XLATensor running_var_tensor =
      bridge::GetOrCreateXlaTensor(running_var, device);
  auto outputs = XLATensor::native_batch_norm(
      bridge::GetXlaTensor(input), bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  XLA_FN_COUNTER("xla::");
  XLATensor grad_out_tensor = bridge::GetXlaTensor(grad_out);
  const Device& device = grad_out_tensor.GetDevice();
  auto gradients = XLATensor::native_batch_norm_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(input),
      bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(save_mean, device),
      bridge::GetOrCreateXlaTensor(save_invstd, device), train, eps);
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromXlaTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromXlaTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromXlaTensor(std::get<2>(gradients))
                     : undefined);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t N,
    int64_t C, int64_t HxW, int64_t group, std::array<bool, 3> output_mask) {
  XLA_FN_COUNTER("xla::");
  return aten_tensor_ops::native_group_norm_backward(
      grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean,
    const at::Tensor& rstd, const c10::optional<at::Tensor>& weight, int64_t M,
    int64_t N, std::array<bool, 3> output_mask) {
  XLA_FN_COUNTER("xla::");
  return aten_tensor_ops::native_layer_norm_backward(
      grad_out, input, mean, rstd, weight, M, N, output_mask);
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ne(bridge::GetXlaTensor(self), other));
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ne(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ne_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::ne_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::neg(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(self.scalar_type() != at::kBool)
      << "Negation, the `-` operator, on a bool tensor is not supported. If "
         "you are trying to invert a mask, use the `~` or `logical_not()` "
         "operator instead.";
  return bridge::AtenFromXlaTensor(XLATensor::neg(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::neg_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::neg_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::nll_loss2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor.GetDevice());
  XLATensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateXlaTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss2d_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor total_weight =
      XLATensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromXlaTensor(XLATensor::nll_loss2d(
          self_tensor, bridge::GetXlaTensor(target),
          bridge::GetOrCreateXlaTensor(weight, self_tensor.GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromXlaTensor(total_weight));
}

at::Tensor AtenXlaType::nll_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor.GetDevice());
  XLATensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateXlaTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor total_weight =
      XLATensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromXlaTensor(XLATensor::nll_loss(
          self_tensor, bridge::GetXlaTensor(target),
          bridge::GetOrCreateXlaTensor(weight, self_tensor.GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromXlaTensor(total_weight));
}

at::Tensor AtenXlaType::nonzero(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  // Initially make XLA handled nonzero() handling experimental, and opt-in.
  if (!DebugUtil::ExperimentEnabled("nonzero")) {
    return AtenXlaTypeDefault::nonzero(self);
  }
  return bridge::AtenFromXlaTensor(XLATensor::nonzero(self_tensor));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             c10::optional<at::Scalar> p,
                             at::ScalarType dtype) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return AtenXlaTypeDefault::norm(self, p, dtype);
  }
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, dtype, {}, /*keepdim=*/false));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self, at::Scalar p) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.toDouble() == 0) {
    return AtenXlaTypeDefault::norm(self, p);
  }
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, c10::nullopt, {}, /*keepdim=*/false));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             c10::optional<at::Scalar> p, at::IntArrayRef dim,
                             bool keepdim, at::ScalarType dtype) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return AtenXlaTypeDefault::norm(self, p, dim, keepdim, dtype);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::norm(bridge::GetXlaTensor(self), p, dtype, dim, keepdim));
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             c10::optional<at::Scalar> p, at::IntArrayRef dim,
                             bool keepdim) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return AtenXlaTypeDefault::norm(self, p, dim, keepdim);
  }
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, c10::nullopt, dim, keepdim));
}

at::Tensor AtenXlaType::normal(const at::Tensor& mean, double std,
                               c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::normal(bridge::GetXlaTensor(mean), std));
}

at::Tensor AtenXlaType::normal(double mean, const at::Tensor& std,
                               c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::normal(mean, bridge::GetXlaTensor(std)));
}

at::Tensor AtenXlaType::normal(const at::Tensor& mean, const at::Tensor& std,
                               c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::normal(mean, std, generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::normal(bridge::GetXlaTensor(mean), bridge::GetXlaTensor(std)));
}

at::Tensor& AtenXlaType::normal_(at::Tensor& self, double mean, double std,
                                 c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::normal_(self, mean, std, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::normal_(self_tensor, mean, std);
  return self;
}

at::Tensor AtenXlaType::permute(const at::Tensor& self, at::IntArrayRef dims) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::permute(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor AtenXlaType::pow(const at::Tensor& self, at::Scalar exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenXlaTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::pow(bridge::GetXlaTensor(self), exponent));
}

at::Tensor AtenXlaType::pow(const at::Tensor& self,
                            const at::Tensor& exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenXlaTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromXlaTensor(XLATensor::pow(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(exponent)));
}

at::Tensor AtenXlaType::pow(at::Scalar self, const at::Tensor& exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!self.isFloatingPoint()) {
    return AtenXlaTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::pow(self, bridge::GetXlaTensor(exponent)));
}

at::Tensor& AtenXlaType::pow_(at::Tensor& self, at::Scalar exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenXlaTypeDefault::pow_(self, exponent);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::pow_(self_tensor, exponent);
  return self;
}

at::Tensor& AtenXlaType::pow_(at::Tensor& self, const at::Tensor& exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenXlaTypeDefault::pow_(self, exponent);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::pow_(self_tensor, bridge::GetXlaTensor(exponent));
  return self;
}

at::Tensor AtenXlaType::prod(const at::Tensor& self,
                             c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::prod(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false,
      PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor AtenXlaType::prod(const at::Tensor& self, int64_t dim, bool keepdim,
                             c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::prod(bridge::GetXlaTensor(self), {dim}, keepdim,
                      PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor& AtenXlaType::put_(at::Tensor& self, const at::Tensor& index,
                              const at::Tensor& source, bool accumulate) {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::put_(self_tensor, bridge::GetXlaTensor(index),
                  bridge::GetXlaTensor(source), accumulate);
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::qr(const at::Tensor& self,
                                                   bool some) {
  XLA_FN_COUNTER("xla::");
  auto results = XLATensor::qr(bridge::GetXlaTensor(self), some);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::reciprocal(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::reciprocal(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::reciprocal_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::reciprocal_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::reflection_pad2d(const at::Tensor& self,
                                         at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::reflection_pad2d(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(padding)));
}

at::Tensor AtenXlaType::reflection_pad2d_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::reflection_pad2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      xla::util::ToVector<xla::int64>(padding)));
}

at::Tensor AtenXlaType::relu(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::relu(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::relu_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::relu_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::remainder(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::remainder(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor AtenXlaType::remainder(const at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::remainder(bridge::GetXlaTensor(self), other));
}

at::Tensor& AtenXlaType::remainder_(at::Tensor& self, const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::remainder_(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& AtenXlaType::remainder_(at::Tensor& self, at::Scalar other) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::remainder_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::repeat(const at::Tensor& self,
                               at::IntArrayRef repeats) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::repeat(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(repeats)));
}

at::Tensor AtenXlaType::replication_pad1d(const at::Tensor& self,
                                          at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad1d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::replication_pad1d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad1d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::replication_pad2d(const at::Tensor& self,
                                          at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad2d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::replication_pad2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

at::Tensor& AtenXlaType::resize_(
    at::Tensor& self, at::IntArrayRef size,
    c10::optional<at::MemoryFormat> /* memory_format */) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::resize_(self_tensor, XlaHelpers::I64List(size));
  return self;
}

at::Tensor AtenXlaType::round(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::round(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::round_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::round_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::rrelu_with_noise(
    const at::Tensor& self, const at::Tensor& noise, at::Scalar lower,
    at::Scalar upper, bool training, c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    XLA_CHECK_EQ(training, false);
    return AtenXlaTypeDefault::rrelu_with_noise(self, noise, lower, upper,
                                                training, generator);
  }
  XLATensor noise_tensor = bridge::GetXlaTensor(noise);
  return bridge::AtenFromXlaTensor(XLATensor::rrelu_with_noise(
      bridge::GetXlaTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor AtenXlaType::rrelu_with_noise_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, at::Scalar lower, at::Scalar upper, bool training,
    bool self_is_result) {
  XLA_FN_COUNTER("xla::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  XLA_CHECK(!self_is_result || negative_slope > 0.0);
  XLATensor noise_tensor = bridge::GetXlaTensor(noise);
  return bridge::AtenFromXlaTensor(XLATensor::rrelu_with_noise_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor AtenXlaType::rsqrt(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::rsqrt(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::rsqrt_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::rsqrt_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::rsub(const at::Tensor& self, const at::Tensor& other,
                             at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::rsub(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenXlaType::rsub(const at::Tensor& self, at::Scalar other,
                             at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return bridge::AtenFromXlaTensor(
      XLATensor::rsub(bridge::GetXlaTensor(self), other, alpha));
}

at::Tensor& AtenXlaType::scatter_(at::Tensor& self, int64_t dim,
                                  const at::Tensor& index,
                                  const at::Tensor& src) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::scatter_(self_tensor, dim, bridge::GetXlaTensor(index),
                      bridge::GetXlaTensor(src));
  return self;
}

at::Tensor& AtenXlaType::scatter_(at::Tensor& self, int64_t dim,
                                  const at::Tensor& index, at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::scatter_(self_tensor, dim, bridge::GetXlaTensor(index), value);
  return self;
}

at::Tensor& AtenXlaType::scatter_add_(at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      const at::Tensor& src) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::scatter_add_(self_tensor, dim, bridge::GetXlaTensor(index),
                          bridge::GetXlaTensor(src));
  return self;
}

at::Tensor AtenXlaType::select(const at::Tensor& self, int64_t dim,
                               int64_t index) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::select(bridge::GetXlaTensor(self), dim, index));
}

at::Tensor AtenXlaType::sigmoid(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::sigmoid(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sigmoid_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sigmoid_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::sigmoid_backward(const at::Tensor& grad_output,
                                         const at::Tensor& output) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::sign(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sign(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sign_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sign_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::sin(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sin(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sin_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sin_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::sinh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sinh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sinh_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sinh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::slice(const at::Tensor& self, int64_t dim,
                              int64_t start, int64_t end, int64_t step) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::slice(bridge::GetXlaTensor(self), dim, start, end, step));
}

at::Tensor AtenXlaType::smooth_l1_loss(const at::Tensor& self,
                                       const at::Tensor& target,
                                       int64_t reduction, double beta) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::smooth_l1_loss(bridge::GetXlaTensor(self),
                                bridge::GetXlaTensor(target), reduction, beta));
}

at::Tensor AtenXlaType::smooth_l1_loss_backward(const at::Tensor& grad_output,
                                                const at::Tensor& self,
                                                const at::Tensor& target,
                                                int64_t reduction,
                                                double beta) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::smooth_l1_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction, beta));
}

at::Tensor AtenXlaType::softplus(const at::Tensor& self, at::Scalar beta,
                                 at::Scalar threshold) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::softplus(bridge::GetXlaTensor(self), beta, threshold));
}

at::Tensor AtenXlaType::softplus_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          at::Scalar beta, at::Scalar threshold,
                                          const at::Tensor& output) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::softplus_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), beta,
      threshold, bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::softshrink(const at::Tensor& self, at::Scalar lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::softshrink(bridge::GetXlaTensor(self), lambda));
}

at::Tensor AtenXlaType::softshrink_backward(const at::Tensor& grad_out,
                                            const at::Tensor& self,
                                            at::Scalar lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::softshrink_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(self), lambda));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::sort(const at::Tensor& self,
                                                     int64_t dim,
                                                     bool descending) {
  XLA_FN_COUNTER("xla::");
  auto results = XLATensor::topk(bridge::GetXlaTensor(self), self.size(dim),
                                 dim, descending, true);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::vector<at::Tensor> AtenXlaType::split(const at::Tensor& self,
                                           int64_t split_size, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  auto xla_tensors =
      XLATensor::split(bridge::GetXlaTensor(self), split_size, dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

std::vector<at::Tensor> AtenXlaType::split_with_sizes(
    const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  auto xla_tensors = XLATensor::split_with_sizes(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(split_sizes), dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

at::Tensor AtenXlaType::sqrt(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sqrt(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::sqrt_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sqrt_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::squeeze(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::squeeze(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::squeeze(const at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::squeeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::squeeze_(self_tensor);
  return self;
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::squeeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenXlaType::stack(at::TensorList tensors, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::stack(bridge::GetXlaTensors(tensors), dim));
}

at::Tensor AtenXlaType::std(const at::Tensor& self, bool unbiased) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::std(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, unbiased));
}

at::Tensor AtenXlaType::std(const at::Tensor& self, at::IntArrayRef dim,
                            bool unbiased, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::std(
      bridge::GetXlaTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions=*/keepdim, unbiased));
}

at::Tensor AtenXlaType::sub(const at::Tensor& self, const at::Tensor& other,
                            at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const XLATensor& xother,
                        at::ScalarType dtype) {
                      return XLATensor::sub(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenXlaType::sub(const at::Tensor& self, at::Scalar other,
                            at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return DoBinaryOp(self, other,
                    [&](const XLATensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::sub(xself, other, alpha, dtype);
                    });
}

at::Tensor& AtenXlaType::sub_(at::Tensor& self, const at::Tensor& other,
                              at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  at::native::alpha_check(at::result_type(self, other), alpha);
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sub_(self_tensor,
                  bridge::GetOrCreateXlaTensor(other, self_tensor.GetDevice()),
                  alpha);
  return self;
}

at::Tensor& AtenXlaType::sub_(at::Tensor& self, at::Scalar other,
                              at::Scalar alpha) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::sub_(self_tensor, other, alpha);
  return self;
}

at::Tensor AtenXlaType::sum(const at::Tensor& self,
                            c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor AtenXlaType::sum(const at::Tensor& self, at::IntArrayRef dim,
                            bool keepdim, c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::sum(bridge::GetXlaTensor(self),
                     xla::util::ToVector<xla::int64>(dim), keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::svd(
    const at::Tensor& self, bool some, bool compute_uv) {
  XLA_FN_COUNTER("xla::");
  auto results = XLATensor::svd(bridge::GetXlaTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)),
                         bridge::AtenFromXlaTensor(std::get<2>(results)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::symeig(const at::Tensor& self,
                                                       bool eigenvectors,
                                                       bool upper) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::symeig(bridge::GetXlaTensor(self), eigenvectors, upper);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::t(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::transpose(bridge::GetXlaTensor(self), 0, 1));
}

at::Tensor& AtenXlaType::t_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor AtenXlaType::take(const at::Tensor& self, const at::Tensor& index) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::take(bridge::GetXlaTensor(self), bridge::GetXlaTensor(index)));
}

at::Tensor AtenXlaType::tan(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::tan(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::tan_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tan_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::tanh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::tanh(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::tanh_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tanh_(self_tensor);
  return self;
}

at::Tensor AtenXlaType::tanh_backward(const at::Tensor& grad_output,
                                      const at::Tensor& output) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::tanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor AtenXlaType::threshold(const at::Tensor& self, at::Scalar threshold,
                                  at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::threshold(
      bridge::GetXlaTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor& AtenXlaType::threshold_(at::Tensor& self, at::Scalar threshold,
                                    at::Scalar value) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::threshold_(self_tensor, threshold.to<double>(),
                        value.to<double>());
  return self;
}

at::Tensor AtenXlaType::threshold_backward(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           at::Scalar threshold) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::threshold_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      threshold.to<double>()));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::topk(const at::Tensor& self,
                                                     int64_t k, int64_t dim,
                                                     bool largest,
                                                     bool sorted) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::topk(bridge::GetXlaTensor(self), k, dim, largest, sorted);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::trace(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::trace(bridge::GetXlaTensor(self)));
}

at::Tensor AtenXlaType::transpose(const at::Tensor& self, int64_t dim0,
                                  int64_t dim1) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::transpose(bridge::GetXlaTensor(self), dim0, dim1));
}

at::Tensor& AtenXlaType::transpose_(at::Tensor& self, int64_t dim0,
                                    int64_t dim1) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::triangular_solve(
    const at::Tensor& b, const at::Tensor& A, bool upper, bool transpose,
    bool unitriangular) {
  XLA_FN_COUNTER("xla::");
  // Currently, ATen doesn't have a left_side option. Once this
  // is added, this API will have to be changed.
  auto results = XLATensor::triangular_solve(
      bridge::GetXlaTensor(b), bridge::GetXlaTensor(A), /*left_side=*/true,
      upper, transpose, unitriangular);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor AtenXlaType::tril(const at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::tril(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor& AtenXlaType::tril_(at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tril_(self_tensor, diagonal);
  return self;
}

at::Tensor AtenXlaType::triu(const at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::triu(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor& AtenXlaType::triu_(at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::triu_(self_tensor, diagonal);
  return self;
}

at::Tensor AtenXlaType::trunc(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::trunc(bridge::GetXlaTensor(self)));
}

at::Tensor& AtenXlaType::trunc_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::trunc_(self_tensor);
  return self;
}

std::vector<at::Tensor> AtenXlaType::unbind(const at::Tensor& self,
                                            int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensors(
      XLATensor::unbind(bridge::GetXlaTensor(self), dim));
}

at::Tensor& AtenXlaType::uniform_(at::Tensor& self, double from, double to,
                                  c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return AtenXlaTypeDefault::uniform_(self, from, to, generator);
  }
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::uniform_(self_tensor, from, to);
  return self;
}

at::Tensor AtenXlaType::unsqueeze(const at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::unsqueeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor& AtenXlaType::unsqueeze_(at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenXlaType::upsample_bilinear2d(const at::Tensor& self,
                                            at::IntArrayRef output_size,
                                            bool align_corners,
                                            c10::optional<double> scales_h,
                                            c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return AtenXlaTypeDefault::upsample_bilinear2d(
        self, output_size, align_corners, scales_h, scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_bilinear2d(
      self_tensor, xla::util::ToVector<xla::int64>(output_size),
      align_corners));
}

at::Tensor AtenXlaType::upsample_bilinear2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensor grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return AtenXlaTypeDefault::upsample_bilinear2d_backward(
        grad_output, output_size, input_size, align_corners, scales_h,
        scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_bilinear2d_backward(
      grad_output_tensor, xla::util::ToVector<xla::int64>(output_size),
      xla::util::ToVector<xla::int64>(input_size), align_corners));
}

at::Tensor AtenXlaType::upsample_nearest2d(
    const at::Tensor& input, c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  XLA_FN_COUNTER("xla::");
  XLATensor input_tensor = bridge::GetXlaTensor(input);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (input_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return AtenXlaTypeDefault::upsample_nearest2d(input, output_size,
                                                  scale_factors);
  }
  absl::Span<const xla::int64> input_dims =
      input_tensor.shape().get().dimensions();
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d(
      input_tensor,
      GetOutputSizeWithScale(input_dims, scale_factors, output_size)));
}

at::Tensor AtenXlaType::upsample_nearest2d_backward(
    const at::Tensor& grad_output, c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  XLA_FN_COUNTER("xla::");
  XLATensor grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return AtenXlaTypeDefault::upsample_nearest2d_backward(
        grad_output, output_size, input_size, scale_factors);
  }
  std::vector<xla::int64> input_dim =
      xla::util::ToVector<xla::int64>(input_size);
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d_backward(
      grad_output_tensor,
      GetOutputSizeWithScale(input_dim, scale_factors, output_size),
      input_dim));
}

at::Tensor AtenXlaType::upsample_nearest2d(const at::Tensor& self,
                                           at::IntArrayRef output_size,
                                           c10::optional<double> scales_h,
                                           c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return AtenXlaTypeDefault::upsample_nearest2d(self, output_size, scales_h,
                                                  scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d(
      self_tensor, xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor AtenXlaType::upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensor grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return AtenXlaTypeDefault::upsample_nearest2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d_backward(
      grad_output_tensor, xla::util::ToVector<xla::int64>(output_size),
      xla::util::ToVector<xla::int64>(input_size)));
}

at::Tensor AtenXlaType::var(const at::Tensor& self, bool unbiased) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      XLATensor::var(bridge::GetXlaTensor(self),
                     xla::util::Iota<xla::int64>(
                         bridge::GetXlaTensor(self).shape().get().rank()),
                     unbiased,
                     /*keep_reduced_dimensions=*/false));
}

at::Tensor AtenXlaType::var(const at::Tensor& self, at::IntArrayRef dim,
                            bool unbiased, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      XLATensor::var(self_tensor, XlaHelpers::I64List(dim), unbiased, keepdim));
}

at::Tensor AtenXlaType::view(const at::Tensor& self, at::IntArrayRef size) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::view(bridge::GetXlaTensor(self), XlaHelpers::I64List(size)));
}

at::Tensor& AtenXlaType::zero_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XLATensor::zero_(self_tensor);
  return self;
}

void AtenXlaType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

}  // namespace torch_xla
