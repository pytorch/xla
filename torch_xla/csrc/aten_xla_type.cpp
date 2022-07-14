#include <ATen/ExpandUtils.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/TypeProperties.h>

#include <mutex>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/aten_autograd_ops.h"
#include "torch_xla/csrc/aten_cpu_fallback.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/generated/XLANativeFunctions.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't have a kernel registered
// according to xla_native_functions.yaml,
//   you can call a boxed CPU fallback kernel instead.
//   E.g. don't call tensor.op() or at::op(tensor).
//   use at::native::call_fallback_fn<&xla_cpu_fallback,
//         ATEN_OP2(op_name, overload_name)>::call(args...)
//   ATEN_OP accepts an operator name without an overload, and
//   ATEN_OP2 accepts an operator name along with its overload name.
//   The description of these acros can be found in
//   https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/templates/Operators.h
//   (You can find some examples below)

namespace torch_xla {
namespace {

torch::lazy::BackendDevice GetXlaDeviceOrCurrent(
    const c10::optional<c10::Device>& device) {
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

bool IsTypeWithLargerRangeThanLong(torch::ScalarType dtype) {
  return dtype == at::ScalarType::BFloat16 || dtype == at::ScalarType::Float ||
         dtype == at::ScalarType::Double;
}

// Return the upper limit for a given type. For floating point typesreturn
// 2^mantissa to ensure that every value is representable.
int64_t GetIntegerUpperLimitForType(torch::ScalarType dtype) {
  xla::PrimitiveType xla_type = TensorTypeToRawXlaType(dtype);
  switch (xla_type) {
    case xla::PrimitiveType::F16:
      return static_cast<int64_t>(1) << std::numeric_limits<xla::half>::digits;
    case xla::PrimitiveType::BF16:
      return static_cast<int64_t>(1)
             << std::numeric_limits<xla::bfloat16>::digits;
    case xla::PrimitiveType::F32:
      return static_cast<int64_t>(1) << std::numeric_limits<float>::digits;
    case xla::PrimitiveType::F64:
      return static_cast<int64_t>(1) << std::numeric_limits<double>::digits;
    default:
      return XlaHelpers::MinMaxValues(xla_type).max.toLong();
  }
}

void CheckRangeValues(torch::ScalarType dtype, int64_t from, int64_t to) {
  XlaHelpers::MinMax min_max;
  // Bound the min_max by int64_t since types of "from" and "to" are int64.
  if (IsTypeWithLargerRangeThanLong(dtype)) {
    min_max = XlaHelpers::MinMaxValues(xla::PrimitiveType::S64);
  } else {
    min_max = XlaHelpers::MinMaxValues(TensorTypeToRawXlaType(dtype));
  }
  XLA_CHECK_GE(from, min_max.min.toLong());
  XLA_CHECK_LE(from, min_max.max.toLong());
  XLA_CHECK_GE(to, min_max.min.toLong());
  XLA_CHECK_LE(to, min_max.max.toLong());
}

std::pair<XLATensorPtr, XLATensorPtr> GetBinaryOperands(
    const at::Tensor& self, const at::Tensor& other) {
  XLATensorPtr self_tensor;
  XLATensorPtr other_tensor;
  auto self_xtensor = bridge::TryGetXlaTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::GetXlaTensor(other);
    self_tensor = bridge::GetOrCreateXlaTensor(self, other_tensor->GetDevice());
  } else {
    self_tensor = self_xtensor;
    other_tensor =
        bridge::GetOrCreateXlaTensor(other, self_tensor->GetDevice());
  }
  return std::pair<XLATensorPtr, XLATensorPtr>(self_tensor, other_tensor);
}

// The input is in format of {N, C, H, W} and the output will be {H, W}.
std::vector<int64_t> GetOutputSizeWithScale(
    absl::Span<const int64_t> input_size,
    const c10::optional<at::ArrayRef<double>>& scale_factors,
    const at::OptionalIntArrayRef& output_size) {
  if (!output_size) {
    XLA_CHECK(scale_factors);
    XLA_CHECK_EQ(scale_factors->size(), 2);
    // Calculate the output size from input_shape and scale_factors
    XLA_CHECK_EQ(input_size.size(), 4);
    int64_t output_h = input_size[2] * (*scale_factors)[0];
    int64_t output_w = input_size[3] * (*scale_factors)[1];
    return {output_h, output_w};
  }
  XLA_CHECK(!scale_factors);
  return torch::lazy::ToVector<int64_t>(*output_size);
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

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<XLATensorPtr, XLATensorPtr> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  XLATensorPtr result = bin_op(operands.first, operands.second, dtype);
  return bridge::AtenFromXlaTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr result = bin_op(self_tensor, other, dtype);
  return bridge::AtenFromXlaTensor(result);
}

template <typename B>
at::Tensor DoBinaryOpWithoutPromo(const at::Tensor& self,
                                  const at::Tensor& other, const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<XLATensorPtr, XLATensorPtr> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  XLATensorPtr result = bin_op(operands.first, operands.second);
  return bridge::AtenFromXlaTensor(result);
}

template <typename B>
at::Tensor DoBinaryOpWithoutPromo(const at::Tensor& self,
                                  const at::Scalar& other, const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr result = bin_op(self_tensor, other);
  return bridge::AtenFromXlaTensor(result);
}

template <typename B>
void DoBinaryOpOut(const at::Tensor& self, const at::Tensor& other,
                   at::Tensor& out, const B& bin_op_out) {
  at::ScalarType dtype = at::result_type(self, other);
  XLA_CHECK(at::canCast(/*from=*/dtype, /*to=*/out.scalar_type()));
  std::pair<XLATensorPtr, XLATensorPtr> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  bin_op_out(operands.first, operands.second, out_tensor);
}

}  // namespace

at::Tensor& XLANativeFunctions::__ilshift__(at::Tensor& self,
                                            const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& XLANativeFunctions::__ilshift__(at::Tensor& self,
                                            const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__ilshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& XLANativeFunctions::__irshift__(at::Tensor& self,
                                            const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& XLANativeFunctions::__irshift__(at::Tensor& self,
                                            const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::__irshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor XLANativeFunctions::__lshift__(const at::Tensor& self,
                                          const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::__lshift__(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::__lshift__(const at::Tensor& self,
                                          const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::__rshift__(const at::Tensor& self,
                                          const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::__rshift__(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::__rshift__(const at::Tensor& self,
                                          const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool3d(
    const at::Tensor& self, at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(_adaptive_avg_pool3d)>::call(self,
                                                                output_size);
  }
  return bridge::AtenFromXlaTensor(XLATensor::adaptive_avg_pool3d(
      bridge::GetXlaTensor(self), output_size_list));
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 3),
                                   grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()), output_size,
                               /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(_adaptive_avg_pool3d_backward)>::call(grad_output, self);
  }
  return bridge::AtenFromXlaTensor(XLATensor::adaptive_avg_pool3d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool2d(
    const at::Tensor& self, at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(_adaptive_avg_pool2d)>::call(self,
                                                                output_size);
  }
  return bridge::AtenFromXlaTensor(XLATensor::_adaptive_avg_pool2d(
      bridge::GetXlaTensor(self), output_size_list));
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()), output_size,
                               /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(_adaptive_avg_pool2d_backward)>::call(grad_output, self);
  }
  return bridge::AtenFromXlaTensor(XLATensor::_adaptive_avg_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::adaptive_max_pool2d(
    const at::Tensor& self, at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(adaptive_max_pool2d)>::call(self,
                                                               output_size);
  }
  std::tuple<XLATensorPtr, XLATensorPtr> res = XLATensor::adaptive_max_pool2d(
      bridge::GetXlaTensor(self), output_size_list);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(res)),
                         bridge::AtenFromXlaTensor(std::get<1>(res)));
}

at::Tensor XLANativeFunctions::adaptive_max_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& indices) {
  XLA_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()), output_size,
                               /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(adaptive_max_pool2d_backward)>::call(grad_output, self,
                                                     indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::adaptive_max_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

void XLANativeFunctions::_amp_foreach_non_finite_check_and_unscale_(
    at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr found_inf_tensor = bridge::GetXlaTensor(found_inf);
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(found_inf_tensor->GetDevice().type());
  XLA_CHECK(hw_type == XlaDeviceType::GPU || hw_type == XlaDeviceType::CPU)
      << "AMP should be used with XLA:GPU";
  XLATensor::_amp_foreach_non_finite_check_and_unscale_(
      bridge::GetXlaTensors(self), found_inf_tensor,
      bridge::GetXlaTensor(inv_scale));
}

at::Tensor& XLANativeFunctions::_amp_update_scale_(at::Tensor& current_scale,
                                                   at::Tensor& growth_tracker,
                                                   const at::Tensor& found_inf,
                                                   double scale_growth_factor,
                                                   double scale_backoff_factor,
                                                   int64_t growth_interval) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr growth_tracker_tensor = bridge::GetXlaTensor(growth_tracker);
  XLATensorPtr current_scale_tensor = bridge::GetXlaTensor(current_scale);
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(growth_tracker_tensor->GetDevice().type());
  XLA_CHECK(hw_type == XlaDeviceType::GPU || hw_type == XlaDeviceType::CPU)
      << "AMP should be used with XLA:GPU";
  XLATensor::_amp_update_scale_(growth_tracker_tensor, current_scale_tensor,
                                bridge::GetXlaTensor(found_inf),
                                scale_growth_factor, scale_backoff_factor,
                                growth_interval);
  return current_scale;
}

at::Tensor XLANativeFunctions::_copy_from(const at::Tensor& self,
                                          const at::Tensor& dst,
                                          bool non_blocking) {
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
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    XLATensor::copy_(dst_tensor, self_tensor);
    bridge::ReplaceXlaTensor(dst, dst_tensor);
  }
  return dst;
}

at::Tensor XLANativeFunctions::_copy_from_and_resize(const at::Tensor& self,
                                                     const at::Tensor& dst) {
  XLA_FN_COUNTER("xla::");
  auto dst_tensor = bridge::TryGetXlaTensor(dst);
  auto self_tensor = bridge::TryGetXlaTensor(self);
  if (!self_tensor) {
    XLA_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is an XLA tensor
    XLATensorImpl* dest_impl =
        dynamic_cast<XLATensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor()->UpdateFromTensorOut(self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

std::vector<at::Tensor> XLANativeFunctions::_to_cpu(at::TensorList tensors) {
  XLA_FN_COUNTER("xla::");
  return bridge::XlaCreateTensorList(tensors);
}

at::Tensor& XLANativeFunctions::_index_put_impl_(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool /* unsafe */) {
  XLA_FN_COUNTER("xla::");
  return torch_xla::XLANativeFunctions::index_put_(self, indices, values,
                                                   accumulate);
}

at::Tensor XLANativeFunctions::_log_softmax(const at::Tensor& self, int64_t dim,
                                            bool /* half_to_float */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::log_softmax(bridge::GetXlaTensor(self), dim, c10::nullopt));
}

at::Tensor XLANativeFunctions::_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType /* input_dtype */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::_pack_padded_sequence(
    const at::Tensor& input, const at::Tensor& lengths, bool batch_first) {
  XLA_FN_COUNTER("xla::");
  std::vector<at::Tensor> xla_tensors = {lengths};
  auto cpu_tensors = bridge::XlaCreateTensorList(xla_tensors);
  return at::native::_pack_padded_sequence(input, cpu_tensors[0], batch_first);
}

at::Tensor XLANativeFunctions::_softmax(const at::Tensor& self, int64_t dim,
                                        bool /* half_to_float */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::softmax(bridge::GetXlaTensor(self), dim, c10::nullopt));
}

at::Tensor XLANativeFunctions::_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType input_dtype) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

at::Tensor XLANativeFunctions::_trilinear(
    const at::Tensor& i1, const at::Tensor& i2, const at::Tensor& i3,
    at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3,
    at::IntArrayRef sumdim, int64_t unroll_dim) {
  XLA_FN_COUNTER("xla::");
  return at::native::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim,
                                unroll_dim);
}

at::Tensor XLANativeFunctions::_unsafe_view(const at::Tensor& self,
                                            at::IntArrayRef size) {
  XLA_FN_COUNTER("xla::");
  return view(self, size);
}

at::Tensor XLANativeFunctions::add(const at::Tensor& self,
                                   const at::Tensor& other,
                                   const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::add(const at::Tensor& self,
                                   const at::Scalar& other,
                                   const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::add(xself, other, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::addcdiv(const at::Tensor& self,
                                       const at::Tensor& tensor1,
                                       const at::Tensor& tensor2,
                                       const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::addcdiv(
      bridge::GetXlaTensor(self), value, bridge::GetXlaTensor(tensor1),
      bridge::GetXlaTensor(tensor2)));
}

at::Tensor& XLANativeFunctions::addcdiv_(at::Tensor& self,
                                         const at::Tensor& tensor1,
                                         const at::Tensor& tensor2,
                                         const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::addcdiv_(self_tensor, value, bridge::GetXlaTensor(tensor1),
                      bridge::GetXlaTensor(tensor2));
  return self;
}

at::Tensor XLANativeFunctions::addcmul(const at::Tensor& self,
                                       const at::Tensor& tensor1,
                                       const at::Tensor& tensor2,
                                       const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::addcmul(
      bridge::GetXlaTensor(self), value, bridge::GetXlaTensor(tensor1),
      bridge::GetXlaTensor(tensor2)));
}

at::Tensor XLANativeFunctions::addmm(const at::Tensor& self,
                                     const at::Tensor& mat1,
                                     const at::Tensor& mat2,
                                     const at::Scalar& beta,
                                     const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (beta.to<double>() != 1 || alpha.to<double>() != 1 ||
      !at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat1) ||
      !at::native::is_floating_point(mat2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(addmm)>::call(self, mat1, mat2,
                                                              beta, alpha);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::addmm(bridge::GetXlaTensor(mat1),
                       /*weight=*/bridge::GetXlaTensor(mat2),
                       /*bias=*/bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::alias(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return self;
}

at::Tensor XLANativeFunctions::all(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::all(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor XLANativeFunctions::all(const at::Tensor& self, int64_t dim,
                                   bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::all(bridge::GetXlaTensor(self), {dim}, keepdim));
}

at::Tensor XLANativeFunctions::amax(const at::Tensor& self, at::IntArrayRef dim,
                                    bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto xdim = XlaHelpers::I64List(dim);
  return bridge::AtenFromXlaTensor(
      XLATensor::amax(bridge::GetXlaTensor(self), std::move(xdim), keepdim));
}

at::Tensor XLANativeFunctions::amin(const at::Tensor& self, at::IntArrayRef dim,
                                    bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto xdim = XlaHelpers::I64List(dim);
  return bridge::AtenFromXlaTensor(
      XLATensor::amin(bridge::GetXlaTensor(self), std::move(xdim), keepdim));
}

at::Tensor XLANativeFunctions::any(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::any(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor XLANativeFunctions::any(const at::Tensor& self, int64_t dim,
                                   bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::any(bridge::GetXlaTensor(self), {dim}, keepdim));
}

at::Tensor& XLANativeFunctions::arange_out(const at::Scalar& start,
                                           const at::Scalar& end,
                                           const at::Scalar& step,
                                           at::Tensor& out) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  XLATensor::arange_out(out_tensor, start, end, step, out.scalar_type());
  return out;
}

at::Tensor XLANativeFunctions::argmax(const at::Tensor& self,
                                      c10::optional<int64_t> dim,
                                      bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return dim ? bridge::AtenFromXlaTensor(
                   XLATensor::argmax(bridge::GetXlaTensor(self), *dim, keepdim))
             : bridge::AtenFromXlaTensor(
                   XLATensor::argmax(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::argmin(const at::Tensor& self,
                                      c10::optional<int64_t> dim,
                                      bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return dim ? bridge::AtenFromXlaTensor(
                   XLATensor::argmin(bridge::GetXlaTensor(self), *dim, keepdim))
             : bridge::AtenFromXlaTensor(
                   XLATensor::argmin(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!AsStrided::StrideIsSupported(self_tensor->shape(), xsize, xstride,
                                    storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(as_strided)>::call(self, size, stride,
                                                      storage_offset);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::as_strided(self_tensor, std::move(xsize), std::move(xstride),
                            XlaHelpers::I64Optional(storage_offset)));
}

const at::Tensor& XLANativeFunctions::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!AsStrided::StrideIsSupported(self_tensor->shape(), xsize, xstride,
                                    storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(as_strided_)>::call(self, size, stride,
                                                       storage_offset);
  }
  XLATensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                         XlaHelpers::I64Optional(storage_offset));
  return self;
}

at::Tensor XLANativeFunctions::atan2(const at::Tensor& self,
                                     const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  // xla::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(atan2)>::call(self, other);
  }
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::atan2(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::avg_pool2d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(avg_pool2d)>::call(self, kernel_size, stride,
                                                      padding, ceil_mode,
                                                      count_include_pad,
                                                      divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor XLANativeFunctions::avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::
        call_fallback_fn<&xla_cpu_fallback, ATEN_OP(avg_pool2d_backward)>::call(
            grad_output, self, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor XLANativeFunctions::avg_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(avg_pool3d)>::call(self, kernel_size, stride,
                                                      padding, ceil_mode,
                                                      count_include_pad,
                                                      divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor XLANativeFunctions::avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  XLA_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::
        call_fallback_fn<&xla_cpu_fallback, ATEN_OP(avg_pool3d_backward)>::call(
            grad_output, self, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor XLANativeFunctions::baddbmm(const at::Tensor& self,
                                       const at::Tensor& batch1,
                                       const at::Tensor& batch2,
                                       const at::Scalar& beta,
                                       const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) ||
      !at::native::is_floating_point(batch2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(baddbmm)>::call(self, batch1,
                                                                batch2, beta,
                                                                alpha);
  }
  return bridge::AtenFromXlaTensor(XLATensor::baddbmm(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(batch1),
      bridge::GetXlaTensor(batch2), beta, alpha));
}

at::Tensor XLANativeFunctions::bernoulli(
    const at::Tensor& self, c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(bernoulli)>::call(self,
                                                                  generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::bernoulli(self_tensor));
}

at::Tensor& XLANativeFunctions::bernoulli_(
    at::Tensor& self, double p, c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(bernoulli_, float)>::call(self, p,
                                                              generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor& XLANativeFunctions::bernoulli_(
    at::Tensor& self, const at::Tensor& p,
    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(bernoulli_, Tensor)>::call(self, p,
                                                               generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::bernoulli_(self_tensor, bridge::GetXlaTensor(p));
  return self;
}

at::Tensor XLANativeFunctions::binary_cross_entropy(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice());
  return bridge::AtenFromXlaTensor(XLATensor::binary_cross_entropy(
      self_tensor, bridge::GetXlaTensor(target), weight_tensor, reduction));
}

at::Tensor XLANativeFunctions::binary_cross_entropy_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice());
  return bridge::AtenFromXlaTensor(XLATensor::binary_cross_entropy_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction));
}

at::Tensor XLANativeFunctions::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight, int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return at::native::binary_cross_entropy_with_logits(
      self, target, IsDefined(weight) ? *weight : at::Tensor(),
      IsDefined(pos_weight) ? *pos_weight : at::Tensor(), reduction);
}

at::Tensor XLANativeFunctions::logical_not(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::logical_not(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::logical_xor(const at::Tensor& self,
                                           const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::logical_xor(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::logical_and(const at::Tensor& self,
                                           const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::logical_and(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::logical_or(const at::Tensor& self,
                                          const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::logical_or(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::bitwise_and(const at::Tensor& self,
                                           const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  return bridge::AtenFromXlaTensor(
      XLATensor::bitwise_and(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::bitwise_and(const at::Tensor& self,
                                           const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const XLATensorPtr& other) {
        return XLATensor::bitwise_and(xself, other);
      });
}

at::Tensor XLANativeFunctions::bitwise_not(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::bitwise_not(self_tensor));
}

at::Tensor XLANativeFunctions::bitwise_or(const at::Tensor& self,
                                          const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const at::Scalar& xother) {
        return XLATensor::bitwise_or(xself, xother);
      });
}

at::Tensor XLANativeFunctions::bitwise_or(const at::Tensor& self,
                                          const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const XLATensorPtr& xother) {
        return XLATensor::bitwise_or(xself, xother);
      });
}

at::Tensor XLANativeFunctions::bitwise_xor(const at::Tensor& self,
                                           const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const at::Scalar& xother) {
        return XLATensor::bitwise_xor(xself, xother);
      });
}

at::Tensor XLANativeFunctions::bitwise_xor(const at::Tensor& self,
                                           const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const XLATensorPtr& xother) {
        return XLATensor::bitwise_xor(xself, xother);
      });
}

at::Tensor XLANativeFunctions::bmm(const at::Tensor& self,
                                   const at::Tensor& mat2) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(bmm)>::call(
        self, mat2);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::bmm(bridge::GetXlaTensor(self), bridge::GetXlaTensor(mat2)));
}

at::Tensor XLANativeFunctions::cat(at::TensorList tensors, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::cat(
      bridge::GetXlaTensors(tensors), dim, at::native::result_type(tensors)));
}

at::Tensor XLANativeFunctions::ceil(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::ceil(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::celu(const at::Tensor& self,
                                    const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::celu(bridge::GetXlaTensor(self), alpha));
}

at::Tensor& XLANativeFunctions::celu_(at::Tensor& self,
                                      const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::celu_(self_tensor, alpha);
  return self;
}

at::Tensor XLANativeFunctions::cholesky(const at::Tensor& self, bool upper) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::cholesky(bridge::GetXlaTensor(self), upper));
}

at::Tensor XLANativeFunctions::clamp(const at::Tensor& self,
                                     const c10::optional<at::Scalar>& min,
                                     const c10::optional<at::Scalar>& max) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, max));
}

at::Tensor XLANativeFunctions::clamp(const at::Tensor& self,
                                     const c10::optional<at::Tensor>& min,
                                     const c10::optional<at::Tensor>& max) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, max));
}

at::Tensor XLANativeFunctions::clamp_max(const at::Tensor& self,
                                         const at::Scalar& max) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), c10::nullopt, max));
}

at::Tensor XLANativeFunctions::clamp_max(const at::Tensor& self,
                                         const at::Tensor& max) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), c10::nullopt, max));
}

at::Tensor XLANativeFunctions::clamp_min(const at::Tensor& self,
                                         const at::Scalar& min) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, c10::nullopt));
}

at::Tensor XLANativeFunctions::clamp_min(const at::Tensor& self,
                                         const at::Tensor& min) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min, c10::nullopt));
}

at::Tensor XLANativeFunctions::clone(
    const at::Tensor& self,
    c10::optional<at::MemoryFormat> /* memory_format */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clone(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::constant_pad_nd(const at::Tensor& self,
                                               at::IntArrayRef pad,
                                               const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::constant_pad_nd(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(pad), value));
}

// This functions covers the whole convolution lowering.
at::Tensor XLANativeFunctions::convolution_overrideable(
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
XLANativeFunctions::convolution_backward_overrideable(
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

at::Tensor XLANativeFunctions::cross(const at::Tensor& self,
                                     const at::Tensor& other,
                                     c10::optional<int64_t> dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::cross(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other),
                       XlaHelpers::I64Optional(dim)));
}

at::Tensor XLANativeFunctions::cumprod(const at::Tensor& self, int64_t dim,
                                       c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  c10::optional<at::ScalarType> promoted_dtype =
      PromoteIntegralType(self_tensor->dtype(), dtype);
  if (IsOperationOnType(promoted_dtype, self_tensor->dtype(),
                        at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(cumprod)>::call(self, dim,
                                                                dtype);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::cumprod(self_tensor, dim, promoted_dtype));
}

at::Tensor XLANativeFunctions::cumsum(const at::Tensor& self, int64_t dim,
                                      c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  if (IsOperationOnType(dtype, self_tensor->dtype(), at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(cumsum)>::call(self, dim,
                                                               dtype);
  }
  return bridge::AtenFromXlaTensor(XLATensor::cumsum(self_tensor, dim, dtype));
}

at::Tensor XLANativeFunctions::diag(const at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::diag(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor XLANativeFunctions::diagonal(const at::Tensor& self, int64_t offset,
                                        int64_t dim1, int64_t dim2) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::diagonal(bridge::GetXlaTensor(self), offset, dim1, dim2));
}

at::Tensor XLANativeFunctions::div(const at::Tensor& self,
                                   const at::Tensor& other) {
  return torch_xla::XLANativeFunctions::div(self, other,
                                            /*rounding_mode=*/c10::nullopt);
}

at::Tensor XLANativeFunctions::div(
    const at::Tensor& self, const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  XLA_FN_COUNTER("xla::");
  at::ScalarType dtype = at::result_type(self, other);
  auto operands = GetBinaryOperands(self, UnwrapNumber(other, dtype));
  return bridge::AtenFromXlaTensor(
      XLATensor::div(operands.first, operands.second, rounding_mode, dtype));
}

at::Tensor XLANativeFunctions::div(const at::Tensor& self,
                                   const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::div(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::dot(const at::Tensor& self,
                                   const at::Tensor& tensor) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(self.dim(), 1)
      << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  XLA_CHECK_EQ(tensor.dim(), 1)
      << "dot: Expected 1-D argument tensor, but got " << tensor.dim() << "-D";
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(tensor)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(dot)>::call(
        self, tensor);
  }
  return bridge::AtenFromXlaTensor(XLATensor::matmul(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(tensor)));
}

at::Tensor XLANativeFunctions::elu(const at::Tensor& self,
                                   const at::Scalar& alpha,
                                   const at::Scalar& scale,
                                   const at::Scalar& input_scale) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::elu(bridge::GetXlaTensor(self), alpha, scale, input_scale));
}

at::Tensor& XLANativeFunctions::elu_(at::Tensor& self, const at::Scalar& alpha,
                                     const at::Scalar& scale,
                                     const at::Scalar& input_scale) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::elu_(self_tensor, alpha, scale, input_scale);
  return self;
}

at::Tensor XLANativeFunctions::elu_backward(const at::Tensor& grad_output,
                                            const at::Scalar& alpha,
                                            const at::Scalar& scale,
                                            const at::Scalar& input_scale,
                                            bool self,
                                            const at::Tensor& self_or_result) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(!self || alpha.to<double>() >= 0.0)
      << "In-place elu backward calculation is triggered with a negative slope "
         "which is not supported.";
  return bridge::AtenFromXlaTensor(XLATensor::elu_backward(
      bridge::GetXlaTensor(grad_output), alpha, scale, input_scale,
      bridge::GetXlaTensor(self_or_result)));
}

at::Tensor XLANativeFunctions::embedding(const at::Tensor& weight,
                                         const at::Tensor& indices,
                                         int64_t padding_idx,
                                         bool scale_grad_by_freq, bool sparse) {
  XLA_FN_COUNTER("xla::");
  // TODO: for now route to native, which dispatches supported XLA operations.
  // We need to make use of the TPU embedding core here eventually.
  return at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq,
                               sparse);
}

at::Tensor XLANativeFunctions::embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::embedding_dense_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

at::Tensor XLANativeFunctions::empty(
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

at::Tensor XLANativeFunctions::empty_symint(
    c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout, c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<at::MemoryFormat> memory_format) {
  // TODO: support SymIntNodes as well
  return empty(c10::asIntArrayRefSlow(size), dtype, layout, device, pin_memory,
               memory_format);
}

at::Tensor XLANativeFunctions::empty_strided(
    at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  XLA_FN_COUNTER("xla::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return torch_xla::XLANativeFunctions::as_strided(t, size, stride,
                                                   /*storage_offset=*/0);
}

at::Tensor XLANativeFunctions::eq(const at::Tensor& self,
                                  const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::eq(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::eq(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::eq(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::expand(const at::Tensor& self,
                                      at::IntArrayRef size, bool implicit) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::expand(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(size)));
}

at::Tensor XLANativeFunctions::expand_symint(const at::Tensor& self,
                                             c10::SymIntArrayRef size,
                                             bool implicit) {
  XLA_FN_COUNTER("xla::");
  SymIntElements size_elements = SymIntElements(size);
  // Replace -1 concrete int dim with the true shape value
  std::vector<c10::SymInt> _sizes = torch::lazy::ToVector<c10::SymInt>(size);
  int64_t num_new_dimensions = _sizes.size() - self.dim();
  std::vector<int64_t> padded_self(num_new_dimensions, 0);
  padded_self.insert(padded_self.end(), self.sizes().begin(),
                     self.sizes().end());
  for (const auto idx : c10::irange(_sizes.size())) {
    if (!_sizes[idx].is_symbolic() && _sizes[idx].expect_int() == -1) {
      size_elements.upper_bounds[idx] = padded_self[idx];
    }
  }
  return bridge::AtenFromXlaTensor(XLATensor::expand(
      bridge::GetXlaTensor(self), size_elements.size_nodes,
      size_elements.upper_bounds, size_elements.dynamic_dims));
}

at::Tensor XLANativeFunctions::expm1(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::expm1(bridge::GetXlaTensor(self)));
}

at::Tensor& XLANativeFunctions::exponential_(
    at::Tensor& self, double lambd, c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(exponential_)>::call(self,
                                                                     lambd,
                                                                     generator);
  }
  XLA_CHECK_GE(lambd, 0.0);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::exponential_(self_tensor, lambd);
  return self;
}

at::Tensor& XLANativeFunctions::eye_out(int64_t n, at::Tensor& out) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  XLATensor::eye_out(out_tensor, n, n);
  return out;
}

at::Tensor& XLANativeFunctions::eye_out(int64_t n, int64_t m, at::Tensor& out) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  XLATensor::eye_out(out_tensor, n, m);
  return out;
}

at::Tensor& XLANativeFunctions::fill_(at::Tensor& self,
                                      const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::fill_(self_tensor, value);
  return self;
}

at::Tensor& XLANativeFunctions::fill_(at::Tensor& self,
                                      const at::Tensor& value) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return torch_xla::XLANativeFunctions::fill_(self, value.item());
}

at::Tensor XLANativeFunctions::flip(const at::Tensor& self,
                                    at::IntArrayRef dims) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::flip(bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor XLANativeFunctions::fmod(const at::Tensor& self,
                                    const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::fmod(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::fmod(const at::Tensor& self,
                                    const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::fmod(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::frac(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::frac(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::gather(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      bool /* sparse_grad */) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::gather(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor XLANativeFunctions::ge(const at::Tensor& self,
                                  const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ge(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::ge(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ge(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::gelu(const at::Tensor& self,
                                    c10::string_view approximate) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::gelu(bridge::GetXlaTensor(self), approximate));
}

at::Tensor XLANativeFunctions::gelu_backward(const at::Tensor& grad,
                                             const at::Tensor& self,
                                             c10::string_view approximate) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::gelu_backward(
      bridge::GetXlaTensor(grad), bridge::GetXlaTensor(self), approximate));
}

at::Tensor XLANativeFunctions::ger(const at::Tensor& self,
                                   const at::Tensor& vec2) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ger(bridge::GetXlaTensor(self), bridge::GetXlaTensor(vec2)));
}

at::Tensor XLANativeFunctions::gt(const at::Tensor& self,
                                  const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::gt(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::gt(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::gt(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::hardshrink(const at::Tensor& self,
                                          const at::Scalar& lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::hardshrink(bridge::GetXlaTensor(self), lambda));
}

at::Tensor XLANativeFunctions::hardsigmoid(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::hardsigmoid(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::hardsigmoid_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardsigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::hardswish(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::hardswish(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::hardswish_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardswish_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::hardshrink_backward(const at::Tensor& grad_out,
                                                   const at::Tensor& self,
                                                   const at::Scalar& lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardshrink_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(self), lambda));
}

at::Tensor XLANativeFunctions::hardtanh(const at::Tensor& self,
                                        const at::Scalar& min_val,
                                        const at::Scalar& max_val) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::clamp(bridge::GetXlaTensor(self), min_val, max_val));
}

at::Tensor XLANativeFunctions::tanh(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::tanh(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::hardtanh_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 const at::Scalar& min_val,
                                                 const at::Scalar& max_val) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::hardtanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), min_val,
      max_val));
}

at::Tensor XLANativeFunctions::index(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  XLA_FN_COUNTER("xla::");
  bool indices_on_cpu_or_xla =
      std::all_of(indices.begin(), indices.end(),
                  [=](const c10::optional<at::Tensor>& opt) {
                    return opt.has_value() && opt->defined()
                               ? (opt->is_cpu() || bridge::IsXlaTensor(*opt))
                               : true;
                  });
  XLA_CHECK(bridge::IsXlaTensor(self) && indices_on_cpu_or_xla)
      << "indices should be either on cpu or on the same"
      << " device as the indexed tensor (XLA)."
      << " When using XLA, the indexed tensor must be an XLA tensor.";
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  c10::optional<torch::lazy::BackendDevice> device =
      bridge::GetXlaDevice(canonical_index_info.base);
  if (!device.has_value()) {
    device = bridge::GetXlaDevice(canonical_index_info.indices);
  }
  XLA_CHECK(device.has_value());
  return bridge::AtenFromXlaTensor(XLATensor::index(
      bridge::GetOrCreateXlaTensor(canonical_index_info.base, *device),
      bridge::GetOrCreateXlaTensors(canonical_index_info.indices, *device),
      canonical_index_info.start_dim));
}

at::Tensor XLANativeFunctions::index_add(const at::Tensor& self, int64_t dim,
                                         const at::Tensor& index,
                                         const at::Tensor& source,
                                         const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::index_add(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
      bridge::GetXlaTensor(source), alpha));
}

at::Tensor XLANativeFunctions::index_copy(const at::Tensor& self, int64_t dim,
                                          const at::Tensor& index,
                                          const at::Tensor& source) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      XLATensor::index_copy(self_tensor, dim, bridge::GetXlaTensor(index),
                            bridge::GetXlaTensor(source)));
}

at::Tensor& XLANativeFunctions::index_fill_(at::Tensor& self, int64_t dim,
                                            const at::Tensor& index,
                                            const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index), value);
  return self;
}

at::Tensor& XLANativeFunctions::index_fill_(at::Tensor& self, int64_t dim,
                                            const at::Tensor& index,
                                            const at::Tensor& value) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index),
                         bridge::GetXlaTensor(value));
  return self;
}

at::Tensor& XLANativeFunctions::index_put_(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(self.scalar_type() == values.scalar_type());
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  c10::optional<torch::lazy::BackendDevice> device =
      bridge::GetXlaDevice(canonical_index_info.base);
  if (!device.has_value()) {
    device = bridge::GetXlaDevice(canonical_index_info.indices);
  }
  XLA_CHECK(device.has_value());
  XLATensorPtr self_tensor = bridge::GetOrCreateXlaTensor(self, *device);
  XLATensor::index_put_(
      self_tensor,
      bridge::GetOrCreateXlaTensor(canonical_index_info.base, *device),
      bridge::GetOrCreateXlaTensors(canonical_index_info.indices, *device),
      canonical_index_info.start_dim,
      bridge::GetOrCreateXlaTensor(values, *device), accumulate,
      canonical_index_info.result_permutation);
  return self;
}

at::Tensor XLANativeFunctions::index_select(const at::Tensor& self, int64_t dim,
                                            const at::Tensor& index) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::index_select(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor XLANativeFunctions::isnan(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::isnan(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::kl_div(const at::Tensor& self,
                                      const at::Tensor& target,
                                      int64_t reduction, bool log_target) {
  XLA_FN_COUNTER("xla::");
  return at::native::kl_div(self, target, reduction, log_target);
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::kthvalue(
    const at::Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::kthvalue(bridge::GetXlaTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::le(const at::Tensor& self,
                                  const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::le(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::le(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::le(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::leaky_relu(const at::Tensor& self,
                                          const at::Scalar& negative_slope) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::leaky_relu(
      bridge::GetXlaTensor(self), negative_slope.to<double>()));
}

at::Tensor XLANativeFunctions::leaky_relu_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Scalar& negative_slope, bool self_is_result) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(!self_is_result || negative_slope.to<double>() >= 0.0);
  return bridge::AtenFromXlaTensor(XLATensor::leaky_relu_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      negative_slope.to<double>()));
}

at::Tensor XLANativeFunctions::lerp(const at::Tensor& self,
                                    const at::Tensor& end,
                                    const at::Tensor& weight) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(self.dtype(), end.dtype())
      << "expected dtype " << self.dtype() << " for `end` but got dtype "
      << end.dtype();
  XLA_CHECK_EQ(self.dtype(), weight.dtype())
      << "expected dtype " << self.dtype() << " for `weight` but got dtype "
      << weight.dtype();
  return bridge::AtenFromXlaTensor(
      XLATensor::lerp(bridge::GetXlaTensor(self), bridge::GetXlaTensor(end),
                      bridge::GetXlaTensor(weight)));
}

at::Tensor XLANativeFunctions::lerp(const at::Tensor& self,
                                    const at::Tensor& end,
                                    const at::Scalar& weight) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(self.dtype(), end.dtype())
      << "expected dtype " << self.dtype() << " for `end` but got dtype "
      << end.dtype();
  return bridge::AtenFromXlaTensor(XLATensor::lerp(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(end), weight));
}

at::Tensor XLANativeFunctions::linspace(const at::Scalar& start,
                                        const at::Scalar& end, int64_t steps,
                                        c10::optional<at::ScalarType> dtype,
                                        c10::optional<at::Layout> layout,
                                        c10::optional<at::Device> device,
                                        c10::optional<bool> pin_memory) {
  XLA_FN_COUNTER("xla::");
  // Fall back to CPU if layout or pin_memory are not default
  if (layout.value_or(at::Layout::Strided) != at::Layout::Strided ||
      pin_memory.value_or(false)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(linspace)>::call(start, end,
                                                                 steps, dtype,
                                                                 layout, device,
                                                                 pin_memory);
  }

  return bridge::AtenFromXlaTensor(
      XLATensor::linspace(start, end, steps, GetScalarTypeOrFloat(dtype),
                          GetXlaDeviceOrCurrent(device)));
}

at::Tensor XLANativeFunctions::log(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::log10(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_base(
      bridge::GetXlaTensor(self), torch::lazy::OpKind(at::aten::log10), 10.0));
}

at::Tensor XLANativeFunctions::log1p(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::log1p(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::log2(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_base(
      bridge::GetXlaTensor(self), torch::lazy::OpKind(at::aten::log2), 2.0));
}

at::Tensor XLANativeFunctions::log_sigmoid_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& buffer) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::log_sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::log_sigmoid_forward(
    const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  auto result_tuple =
      XLATensor::log_sigmoid_forward(bridge::GetXlaTensor(self));
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromXlaTensor(std::get<1>(result_tuple)));
}

at::Tensor XLANativeFunctions::logsumexp(const at::Tensor& self,
                                         at::IntArrayRef dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::logsumexp(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(dim),
      /*keep_reduced_dimensions=*/keepdim));
}

at::Tensor XLANativeFunctions::xlogy(const at::Tensor& self,
                                     const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::xlogy(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::lt(const at::Tensor& self,
                                  const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::lt(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::lt(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::lt(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor& XLANativeFunctions::masked_fill_(at::Tensor& self,
                                             const at::Tensor& mask,
                                             const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::masked_fill_(self_tensor, bridge::GetXlaTensor(mask), value);
  return self;
}

at::Tensor& XLANativeFunctions::masked_fill_(at::Tensor& self,
                                             const at::Tensor& mask,
                                             const at::Tensor& value) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK_EQ(value.dim(), 0) << "masked_fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill_(self, mask, value.item());
}

at::Tensor& XLANativeFunctions::masked_scatter_(at::Tensor& self,
                                                const at::Tensor& mask,
                                                const at::Tensor& source) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::masked_scatter_(self_tensor, bridge::GetXlaTensor(mask),
                             bridge::GetXlaTensor(source));
  return self;
}

at::Tensor XLANativeFunctions::masked_select(const at::Tensor& self,
                                             const at::Tensor& mask) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  // Initially make XLA handled masked_select() handling experimental, and
  // opt-in.
  if (!DebugUtil::ExperimentEnabled("masked_select")) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(masked_select)>::call(self,
                                                                      mask);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::masked_select(self_tensor, bridge::GetXlaTensor(mask)));
}

at::Tensor XLANativeFunctions::max(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::max(
    const at::Tensor& self, int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto outputs = XLATensor::max(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

std::tuple<at::Tensor&, at::Tensor&> XLANativeFunctions::max_out(
    const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& max,
    at::Tensor& max_values) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr max_tensor = bridge::GetXlaTensor(max);
  XLATensorPtr max_values_tensor = bridge::GetXlaTensor(max_values);
  XLATensor::max_out(max_tensor, max_values_tensor, bridge::GetXlaTensor(self),
                     dim, keepdim);
  return std::forward_as_tuple(max, max_values);
}

at::Tensor XLANativeFunctions::max_pool2d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  return aten_autograd_ops::MaxPool2dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(max_pool2d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
  }
  auto outputs = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output, self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
}

at::Tensor XLANativeFunctions::max_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  return aten_autograd_ops::MaxPool3dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor XLANativeFunctions::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output, self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  XLA_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(max_pool3d_with_indices)>::call(self,
                                                                   kernel_size,
                                                                   stride,
                                                                   padding,
                                                                   dilation,
                                                                   ceil_mode);
  }
  auto outputs = XLATensor::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::max_unpool2d(const at::Tensor& self,
                                            const at::Tensor& indices,
                                            at::IntArrayRef output_size) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max_unpool(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(indices),
      torch::lazy::ToVector<int64_t>(output_size)));
}

at::Tensor XLANativeFunctions::max_unpool3d(const at::Tensor& self,
                                            const at::Tensor& indices,
                                            at::IntArrayRef output_size,
                                            at::IntArrayRef stride,
                                            at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::max_unpool(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(indices),
      torch::lazy::ToVector<int64_t>(output_size)));
}

at::Tensor XLANativeFunctions::mean(const at::Tensor& self,
                                    c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor XLANativeFunctions::mean(const at::Tensor& self, at::IntArrayRef dim,
                                    bool keepdim,
                                    c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mean(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(dim),
      /*keep_reduced_dimensions=*/keepdim, dtype));
}

at::Tensor XLANativeFunctions::min(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::min(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::min(
    const at::Tensor& self, int64_t dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  auto outputs = XLATensor::min(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::mish(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mish(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor&, at::Tensor&> XLANativeFunctions::min_out(
    const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& min,
    at::Tensor& min_indices) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr min_tensor = bridge::GetXlaTensor(min);
  XLATensorPtr min_indices_tensor = bridge::GetXlaTensor(min_indices);
  XLATensor::min_out(min_tensor, min_indices_tensor, bridge::GetXlaTensor(self),
                     dim, keepdim);
  return std::forward_as_tuple(min, min_indices);
}

at::Tensor XLANativeFunctions::mm(const at::Tensor& self,
                                  const at::Tensor& mat2) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(mm)>::call(
        self, mat2);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::mm(/*input=*/bridge::GetXlaTensor(self),
                    /*weight=*/bridge::GetXlaTensor(mat2)));
}

at::Tensor XLANativeFunctions::mse_loss(const at::Tensor& self,
                                        const at::Tensor& target,
                                        int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mse_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target), reduction));
}

at::Tensor XLANativeFunctions::mse_loss_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 const at::Tensor& target,
                                                 int64_t reduction) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::mse_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction));
}

at::Tensor XLANativeFunctions::mul(const at::Tensor& self,
                                   const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::mul(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::mul(const at::Tensor& self,
                                   const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::mul(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::mv(const at::Tensor& self,
                                  const at::Tensor& vec) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(vec)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(mv)>::call(
        self, vec);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::mv(bridge::GetXlaTensor(self), bridge::GetXlaTensor(vec)));
}

at::Tensor& XLANativeFunctions::mv_out(const at::Tensor& self,
                                       const at::Tensor& vec, at::Tensor& out) {
  XLA_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(vec)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(mv_out)>::call(self, vec, out);
  }
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  XLATensor::mv_out(out_tensor, bridge::GetXlaTensor(self),
                    bridge::GetXlaTensor(vec));
  return out;
}

at::Tensor XLANativeFunctions::nan_to_num(const at::Tensor& self,
                                          c10::optional<double> nan,
                                          c10::optional<double> posinf,
                                          c10::optional<double> neginf) {
  XLA_FN_COUNTER("xla::");
  // nan_to_num doesn't apply to integer types.
  if (!at::native::is_floating_point(self)) {
    return torch::lazy::CopyTensor(self);
  }
  XLATensorPtr input_tensor = bridge::GetXlaTensor(self);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  auto element_type = MakeXlaPrimitiveType(self.scalar_type(), &device);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(element_type);
  at::Scalar nan_replacement = nan.has_value() ? *nan : 0.0;
  at::Scalar posinf_replacement = posinf.has_value() ? *posinf : min_max.max;
  at::Scalar neginf_replacement = neginf.has_value() ? *neginf : min_max.min;
  for (const auto& replacement :
       {nan_replacement, posinf_replacement, neginf_replacement}) {
    XLA_CHECK(min_max.min.toDouble() <= replacement.toDouble() &&
              replacement.toDouble() <= min_max.max.toDouble())
        << "Type " << self.scalar_type() << " replacement value "
        << replacement.toDouble() << " must be in the range ["
        << min_max.min.toDouble() << ", " << min_max.max.toDouble() << "].";
  }
  return bridge::AtenFromXlaTensor(XLATensor::nan_to_num(
      input_tensor, nan_replacement, posinf_replacement, neginf_replacement));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr input_tensor = bridge::GetXlaTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  XLATensorPtr running_mean_tensor =
      bridge::GetOrCreateXlaTensor(running_mean, device);
  XLATensorPtr running_var_tensor =
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
XLANativeFunctions::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr grad_out_tensor = bridge::GetXlaTensor(grad_out);
  const torch::lazy::BackendDevice& device = grad_out_tensor->GetDevice();
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

at::Tensor XLANativeFunctions::ne(const at::Tensor& self,
                                  const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ne(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::ne(const at::Tensor& self,
                                  const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::ne(bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::neg(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLA_CHECK(self.scalar_type() != at::kBool)
      << "Negation, the `-` operator, on a bool tensor is not supported. If "
         "you are trying to invert a mask, use the `~` or `logical_not()` "
         "operator instead.";
  return bridge::AtenFromXlaTensor(XLATensor::neg(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::nll_loss2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice());
  XLATensorPtr total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateXlaTensor(total_weight, self_tensor->GetDevice());
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss2d_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr total_weight =
      XLATensor::full({}, 1, self_tensor->GetDevice(), self_tensor->dtype());
  return std::make_tuple(
      bridge::AtenFromXlaTensor(XLATensor::nll_loss2d(
          self_tensor, bridge::GetXlaTensor(target),
          bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromXlaTensor(total_weight));
}

at::Tensor XLANativeFunctions::nll_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice());
  XLATensorPtr total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateXlaTensor(total_weight, self_tensor->GetDevice());
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr total_weight =
      XLATensor::full({}, 1, self_tensor->GetDevice(), self_tensor->dtype());
  return std::make_tuple(
      bridge::AtenFromXlaTensor(XLATensor::nll_loss(
          self_tensor, bridge::GetXlaTensor(target),
          bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromXlaTensor(total_weight));
}

at::Tensor XLANativeFunctions::nonzero(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  /*
   * REMOVE THIS SECTION TO ENABLE CREATING DYNAMIC SHAPES FOR POC
   * TODO: REMOVE THIS SECTION AFTER POC SUCCEEDS:
   * https://github.com/pytorch/xla/pull/3558
   *
   * // Initially make XLA handled nonzero() handling experimental, and opt-in.
   * if (!DebugUtil::ExperimentEnabled("nonzero")) {
   *   return at::native::call_fallback_fn<&xla_cpu_fallback,
   *                                       ATEN_OP(nonzero)>::call(self);
   * }
   */
  return bridge::AtenFromXlaTensor(XLATensor::nonzero(self_tensor));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const c10::optional<at::Scalar>& p,
                                    at::ScalarType dtype) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(norm, ScalarOpt_dtype)>::call(self, p,
                                                                  dtype);
  }
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, dtype, {}, /*keepdim=*/false));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const at::Scalar& p) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.toDouble() == 0) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(norm, Scalar)>::call(self, p);
  }
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, c10::nullopt, {}, /*keepdim=*/false));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const c10::optional<at::Scalar>& p,
                                    at::IntArrayRef dim, bool keepdim,
                                    at::ScalarType dtype) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(norm, ScalarOpt_dim_dtype)>::call(self, p,
                                                                      dim,
                                                                      keepdim,
                                                                      dtype);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::norm(bridge::GetXlaTensor(self), p, dtype, dim, keepdim));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const c10::optional<at::Scalar>& p,
                                    at::IntArrayRef dim, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(norm, ScalarOpt_dim)>::call(self, p, dim,
                                                                keepdim);
  }
  return bridge::AtenFromXlaTensor(XLATensor::norm(
      bridge::GetXlaTensor(self), p, c10::nullopt, dim, keepdim));
}

at::Tensor XLANativeFunctions::normal(const at::Tensor& mean, double std,
                                      c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(normal, Tensor_float)>::call(mean, std,
                                                                 generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::normal(bridge::GetXlaTensor(mean), std));
}

at::Tensor XLANativeFunctions::normal(double mean, const at::Tensor& std,
                                      c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(normal, float_Tensor)>::call(mean, std,
                                                                 generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::normal(mean, bridge::GetXlaTensor(std)));
}

at::Tensor XLANativeFunctions::normal(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(normal, Tensor_Tensor)>::call(mean, std,
                                                                  generator);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::normal(bridge::GetXlaTensor(mean), bridge::GetXlaTensor(std)));
}

at::Tensor& XLANativeFunctions::normal_(
    at::Tensor& self, double mean, double std,
    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(normal_)>::call(self, mean, std,
                                                                generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::normal_(self_tensor, mean, std);
  return self;
}

at::Tensor XLANativeFunctions::permute(const at::Tensor& self,
                                       at::IntArrayRef dims) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::permute(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor XLANativeFunctions::pow(const at::Tensor& self,
                                   const at::Scalar& exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(pow, Tensor_Scalar)>::call(self, exponent);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::pow(bridge::GetXlaTensor(self), exponent));
}

at::Tensor XLANativeFunctions::pow(const at::Tensor& self,
                                   const at::Tensor& exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(pow, Tensor_Tensor)>::call(self, exponent);
  }
  return bridge::AtenFromXlaTensor(XLATensor::pow(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(exponent)));
}

at::Tensor XLANativeFunctions::pow(const at::Scalar& self,
                                   const at::Tensor& exponent) {
  XLA_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!self.isFloatingPoint()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(pow, Scalar)>::call(self,
                                                                     exponent);
  }
  return bridge::AtenFromXlaTensor(
      XLATensor::pow(self, bridge::GetXlaTensor(exponent)));
}

at::Tensor XLANativeFunctions::prelu(const at::Tensor& self,
                                     const at::Tensor& weight) {
  XLA_FN_COUNTER("xla::");

  // If multiple weights, check channel size == number of weights.
  int64_t weight_num = weight.numel();
  if (weight.numel() > 1) {
    int64_t input_dim = self.dim();
    XLA_CHECK_GT(input_dim, 0) << "Input tensor dimension cannot be 0";

    int64_t channel_size = input_dim > 1 ? self.size(1) : 1;
    XLA_CHECK_EQ(channel_size, weight_num)
        << "Mismatch of parameter numbers and input channel size. Found "
           "parameter numbers = "
        << weight_num << " and channel size = " << channel_size;
  }

  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor = bridge::GetXlaTensor(weight);

  return bridge::AtenFromXlaTensor(
      XLATensor::prelu(self_tensor, weight_tensor));
}

at::Tensor XLANativeFunctions::prod(const at::Tensor& self,
                                    c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::prod(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false,
      PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor XLANativeFunctions::prod(const at::Tensor& self, int64_t dim,
                                    bool keepdim,
                                    c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::prod(bridge::GetXlaTensor(self), {dim}, keepdim,
                      PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor& XLANativeFunctions::put_(at::Tensor& self, const at::Tensor& index,
                                     const at::Tensor& source,
                                     bool accumulate) {
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::put_(self_tensor, bridge::GetXlaTensor(index),
                  bridge::GetXlaTensor(source), accumulate);
  return self;
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::qr(
    const at::Tensor& self, bool some) {
  XLA_FN_COUNTER("xla::");
  auto results = XLATensor::qr(bridge::GetXlaTensor(self), some);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

// The value generated should be within (from, to].
at::Tensor& XLANativeFunctions::random_(
    at::Tensor& self, int64_t from, c10::optional<int64_t> to,
    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(random_, from)>::call(self, from, to,
                                                          generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  at::ScalarType dtype = self_tensor->dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  int64_t to_val = (to) ? *to : GetIntegerUpperLimitForType(dtype) + inc;
  XLA_CHECK_LE(from, to_val);
  CheckRangeValues(self_tensor->dtype(), from, to_val - 1);
  XLATensor::random_(self_tensor, from, to_val);
  return self;
}

// The value generated should be in (0, to].
at::Tensor& XLANativeFunctions::random_(
    at::Tensor& self, int64_t to, c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(random_, to)>::call(self, to,
                                                                     generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLA_CHECK_GT(to, 0);
  CheckRangeValues(self_tensor->dtype(), 0, to - 1);
  XLATensor::random_(self_tensor, 0, to);
  return self;
}

// The value generated should be in (self_type_min, self_type_max).
at::Tensor& XLANativeFunctions::random_(
    at::Tensor& self, c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(random_)>::call(self,
                                                                generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  at::ScalarType dtype = self_tensor->dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  XLATensor::random_(self_tensor, 0, GetIntegerUpperLimitForType(dtype) + inc);
  return self;
}

at::Tensor XLANativeFunctions::reflection_pad2d(const at::Tensor& self,
                                                at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::reflection_pad2d(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::reflection_pad2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::reflection_pad2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::relu(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::relu(bridge::GetXlaTensor(self)));
}

at::Tensor& XLANativeFunctions::relu_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::relu_(self_tensor);
  return self;
}

at::Tensor XLANativeFunctions::remainder(const at::Tensor& self,
                                         const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::remainder(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::remainder(const at::Tensor& self,
                                         const at::Scalar& other) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::remainder(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::repeat(const at::Tensor& self,
                                      at::IntArrayRef repeats) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::repeat(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(repeats)));
}

at::Tensor XLANativeFunctions::replication_pad1d(const at::Tensor& self,
                                                 at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad1d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad1d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad1d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad2d(const at::Tensor& self,
                                                 at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad2d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::replication_pad2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

const at::Tensor& XLANativeFunctions::resize_(
    const at::Tensor& self, at::IntArrayRef size,
    c10::optional<at::MemoryFormat> /* memory_format */) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::resize_(self_tensor, XlaHelpers::I64List(size));
  return self;
}

at::Tensor XLANativeFunctions::roll(const at::Tensor& self,
                                    at::IntArrayRef shifts,
                                    at::IntArrayRef dims) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::roll(bridge::GetXlaTensor(self),
                                                   XlaHelpers::I64List(shifts),
                                                   XlaHelpers::I64List(dims)));
}

at::Tensor XLANativeFunctions::round(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::round(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::rrelu_with_noise(
    const at::Tensor& self, const at::Tensor& noise, const at::Scalar& lower,
    const at::Scalar& upper, bool training,
    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    XLA_CHECK_EQ(training, false);
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(rrelu_with_noise)>::call(self, noise, lower,
                                                            upper, training,
                                                            generator);
  }
  XLATensorPtr noise_tensor = bridge::GetXlaTensor(noise);
  return bridge::AtenFromXlaTensor(XLATensor::rrelu_with_noise(
      bridge::GetXlaTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor XLANativeFunctions::rrelu_with_noise_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, const at::Scalar& lower, const at::Scalar& upper,
    bool training, bool self_is_result) {
  XLA_FN_COUNTER("xla::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  XLA_CHECK(!self_is_result || negative_slope > 0.0);
  XLATensorPtr noise_tensor = bridge::GetXlaTensor(noise);
  return bridge::AtenFromXlaTensor(XLATensor::rrelu_with_noise_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor XLANativeFunctions::rsqrt(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::rsqrt(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::rsub(const at::Tensor& self,
                                    const at::Tensor& other,
                                    const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::rsub(xself, xother, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::rsub(const at::Tensor& self,
                                    const at::Scalar& other,
                                    const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return bridge::AtenFromXlaTensor(
      XLATensor::rsub(bridge::GetXlaTensor(self), other, alpha));
}

at::Tensor scatter_reduce_helper(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, const at::Tensor& src,
                                 c10::optional<c10::string_view> reduce) {
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  if (!reduce.has_value()) {
    return bridge::AtenFromXlaTensor(
        XLATensor::scatter(self_tensor, dim, bridge::GetXlaTensor(index),
                           bridge::GetXlaTensor(src)));
  } else if (*reduce == "add") {
    return bridge::AtenFromXlaTensor(
        XLATensor::scatter_add(self_tensor, dim, bridge::GetXlaTensor(index),
                               bridge::GetXlaTensor(src)));
  } else {
    // TODO: implement scatter_mul
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(scatter, reduce)>::call(self, dim, index,
                                                            src, *reduce);
  }
}

at::Tensor scatter_reduce_helper(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index,
                                 const at::Scalar& value,
                                 c10::optional<c10::string_view> reduce) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  if (!reduce.has_value()) {
    return bridge::AtenFromXlaTensor(XLATensor::scatter(
        self_tensor, dim, bridge::GetXlaTensor(index), value));
  } else if (*reduce == "add") {
    return bridge::AtenFromXlaTensor(XLATensor::scatter_add(
        self_tensor, dim, bridge::GetXlaTensor(index), value));
  } else {
    // TODO: implement scatter_mul
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(scatter, value_reduce)>::call(self, dim,
                                                                  index, value,
                                                                  *reduce);
  }
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Tensor& src) {
  XLA_FN_COUNTER("xla::");
  return scatter_reduce_helper(self, dim, index, src, c10::nullopt);
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  return scatter_reduce_helper(self, dim, index, value, c10::nullopt);
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Tensor& src,
                                       c10::string_view reduce) {
  XLA_FN_COUNTER("xla::");
  return scatter_reduce_helper(self, dim, index, src, reduce);
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Scalar& value,
                                       c10::string_view reduce) {
  XLA_FN_COUNTER("xla::");
  return scatter_reduce_helper(self, dim, index, value, reduce);
}

at::Tensor XLANativeFunctions::scatter_add(const at::Tensor& self, int64_t dim,
                                           const at::Tensor& index,
                                           const at::Tensor& src) {
  XLA_FN_COUNTER("xla::");
  return scatter_reduce_helper(self, dim, index, src, "add");
}

at::Tensor XLANativeFunctions::select(const at::Tensor& self, int64_t dim,
                                      int64_t index) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::select(bridge::GetXlaTensor(self), dim, index));
}

at::Tensor XLANativeFunctions::selu(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::selu(bridge::GetXlaTensor(self)));
}

at::Tensor& XLANativeFunctions::selu_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::selu_(self_tensor);
  return self;
}

at::Tensor XLANativeFunctions::silu(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::silu(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::silu_backward(const at::Tensor& grad_output,
                                             const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      XLATensor::silu_backward(grad_output_tensor, self_tensor));
}

at::Tensor XLANativeFunctions::sigmoid(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::sigmoid(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::sigmoid_backward(const at::Tensor& grad_output,
                                                const at::Tensor& output) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor XLANativeFunctions::slice(const at::Tensor& self, int64_t dim,
                                     c10::optional<int64_t> start,
                                     c10::optional<int64_t> end, int64_t step) {
  XLA_FN_COUNTER("xla::");
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  return bridge::AtenFromXlaTensor(XLATensor::slice(
      bridge::GetXlaTensor(self), dim, start_val, end_val, step));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::slogdet(
    const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto outputs = XLATensor::slogdet(self_tensor);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::smooth_l1_loss(const at::Tensor& self,
                                              const at::Tensor& target,
                                              int64_t reduction, double beta) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::smooth_l1_loss(bridge::GetXlaTensor(self),
                                bridge::GetXlaTensor(target), reduction, beta));
}

at::Tensor XLANativeFunctions::smooth_l1_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, int64_t reduction, double beta) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::smooth_l1_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction, beta));
}

at::Tensor XLANativeFunctions::softplus(const at::Tensor& self,
                                        const at::Scalar& beta,
                                        const at::Scalar& threshold) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::softplus(bridge::GetXlaTensor(self), beta, threshold));
}

at::Tensor XLANativeFunctions::softplus_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 const at::Scalar& beta,
                                                 const at::Scalar& threshold) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::softplus_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), beta,
      threshold));
}

at::Tensor XLANativeFunctions::softshrink(const at::Tensor& self,
                                          const at::Scalar& lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::softshrink(bridge::GetXlaTensor(self), lambda));
}

at::Tensor XLANativeFunctions::softshrink_backward(const at::Tensor& grad_out,
                                                   const at::Tensor& self,
                                                   const at::Scalar& lambda) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::softshrink_backward(
      bridge::GetXlaTensor(grad_out), bridge::GetXlaTensor(self), lambda));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::sort(
    const at::Tensor& self, int64_t dim, bool descending) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::topk(bridge::GetXlaTensor(self), self.size(dim), dim,
                      descending, /*sorted=*/true, /*stable=*/false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::sort(
    const at::Tensor& self, c10::optional<bool> stable, int64_t dim,
    bool descending) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::topk(bridge::GetXlaTensor(self), self.size(dim), dim,
                      descending, /*sorted=*/false,
                      /*stable=*/stable.has_value() ? stable.value() : false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::vector<at::Tensor> XLANativeFunctions::split(const at::Tensor& self,
                                                  int64_t split_size,
                                                  int64_t dim) {
  XLA_FN_COUNTER("xla::");
  auto xla_tensors =
      XLATensor::split(bridge::GetXlaTensor(self), split_size, dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

std::vector<at::Tensor> XLANativeFunctions::split_with_sizes(
    const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  auto xla_tensors = XLATensor::split_with_sizes(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(split_sizes), dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

at::Tensor XLANativeFunctions::sqrt(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::sqrt(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::squeeze(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::squeeze(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::squeeze(const at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::squeeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor& XLANativeFunctions::squeeze_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::squeeze_(self_tensor);
  return self;
}

at::Tensor& XLANativeFunctions::squeeze_(at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::squeeze_(self_tensor, dim);
  return self;
}

at::Tensor XLANativeFunctions::stack(at::TensorList tensors, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::stack(bridge::GetXlaTensors(tensors), dim));
}

at::Tensor XLANativeFunctions::std(const at::Tensor& self, bool unbiased) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::std(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false, /*correction=*/unbiased ? 1 : 0));
}

at::Tensor XLANativeFunctions::std(const at::Tensor& self, at::IntArrayRef dim,
                                   bool unbiased, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::std(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(dim), keepdim,
      /*correction=*/unbiased ? 1 : 0));
}

at::Tensor XLANativeFunctions::std(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim,
                                   c10::optional<int64_t> correction,
                                   bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::std(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, correction ? *correction : 1));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::std_mean(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto results = XLATensor::std_mean(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      correction ? *correction : 1, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::sub(const at::Tensor& self,
                                   const at::Tensor& other,
                                   const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return XLATensor::sub(xself, xother, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::sub(const at::Tensor& self,
                                   const at::Scalar& other,
                                   const at::Scalar& alpha) {
  XLA_FN_COUNTER("xla::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return XLATensor::sub(xself, other, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::sum(const at::Tensor& self,
                                   c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor XLANativeFunctions::sum(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim, bool keepdim,
                                   c10::optional<at::ScalarType> dtype) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::sum(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> XLANativeFunctions::svd(
    const at::Tensor& self, bool some, bool compute_uv) {
  XLA_FN_COUNTER("xla::");
  auto results = XLATensor::svd(bridge::GetXlaTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)),
                         bridge::AtenFromXlaTensor(std::get<2>(results)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::symeig(
    const at::Tensor& self, bool eigenvectors, bool upper) {
  XLA_FN_COUNTER("xla::");
  auto results =
      XLATensor::symeig(bridge::GetXlaTensor(self), eigenvectors, upper);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::t(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::transpose(bridge::GetXlaTensor(self), 0, 1));
}

at::Tensor& XLANativeFunctions::t_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor XLANativeFunctions::take(const at::Tensor& self,
                                    const at::Tensor& index) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::take(bridge::GetXlaTensor(self), bridge::GetXlaTensor(index)));
}

at::Tensor XLANativeFunctions::tanh_backward(const at::Tensor& grad_output,
                                             const at::Tensor& output) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::tanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor XLANativeFunctions::threshold(const at::Tensor& self,
                                         const at::Scalar& threshold,
                                         const at::Scalar& value) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::threshold(
      bridge::GetXlaTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor XLANativeFunctions::threshold_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  const at::Scalar& threshold) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::threshold_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      threshold.to<double>()));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::topk(
    const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
  XLA_FN_COUNTER("xla::");
  auto results = XLATensor::topk(bridge::GetXlaTensor(self), k, dim, largest,
                                 sorted, /*stable=*/false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::trace(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::trace(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::transpose(const at::Tensor& self, int64_t dim0,
                                         int64_t dim1) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::transpose(bridge::GetXlaTensor(self), dim0, dim1));
}

at::Tensor& XLANativeFunctions::transpose_(at::Tensor& self, int64_t dim0,
                                           int64_t dim1) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::triangular_solve(
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

at::Tensor XLANativeFunctions::tril(const at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::tril(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor& XLANativeFunctions::tril_(at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::tril_(self_tensor, diagonal);
  return self;
}

at::Tensor XLANativeFunctions::triu(const at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::triu(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor& XLANativeFunctions::triu_(at::Tensor& self, int64_t diagonal) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::triu_(self_tensor, diagonal);
  return self;
}

at::Tensor XLANativeFunctions::trunc(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::trunc(bridge::GetXlaTensor(self)));
}

std::vector<at::Tensor> XLANativeFunctions::unbind(const at::Tensor& self,
                                                   int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensors(
      XLATensor::unbind(bridge::GetXlaTensor(self), dim));
}

at::Tensor& XLANativeFunctions::uniform_(
    at::Tensor& self, double from, double to,
    c10::optional<at::Generator> generator) {
  XLA_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(uniform_)>::call(self, from, to,
                                                                 generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::uniform_(self_tensor, from, to);
  return self;
}

at::Tensor XLANativeFunctions::unsqueeze(const at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::unsqueeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor& XLANativeFunctions::unsqueeze_(at::Tensor& self, int64_t dim) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor XLANativeFunctions::upsample_bilinear2d(
    const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(self_tensor->GetDevice().type());
  if (hw_type != XlaDeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(upsample_bilinear2d)>::call(self,
                                                               output_size,
                                                               align_corners,
                                                               scales_h,
                                                               scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_bilinear2d(
      self_tensor, torch::lazy::ToVector<int64_t>(output_size), align_corners));
}

at::Tensor XLANativeFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(grad_output_tensor->GetDevice().type());
  if (hw_type != XlaDeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(upsample_bilinear2d_backward)>::call(grad_output, output_size,
                                                     input_size, align_corners,
                                                     scales_h, scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_bilinear2d_backward(
      grad_output_tensor, torch::lazy::ToVector<int64_t>(output_size),
      torch::lazy::ToVector<int64_t>(input_size), align_corners));
}

at::Tensor XLANativeFunctions::upsample_nearest2d(
    const at::Tensor& input, at::OptionalIntArrayRef output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr input_tensor = bridge::GetXlaTensor(input);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(input_tensor->GetDevice().type());
  if (hw_type != XlaDeviceType::TPU) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(upsample_nearest2d,
                                                 vec)>::call(input, output_size,
                                                             scale_factors);
  }
  absl::Span<const int64_t> input_dims =
      input_tensor->shape().get().dimensions();
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d(
      input_tensor,
      GetOutputSizeWithScale(input_dims, scale_factors, output_size)));
}

at::Tensor XLANativeFunctions::upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::OptionalIntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(grad_output_tensor->GetDevice().type());
  if (hw_type != XlaDeviceType::TPU) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(upsample_nearest2d_backward,
                                                 vec)>::call(grad_output,
                                                             output_size,
                                                             input_size,
                                                             scale_factors);
  }
  std::vector<int64_t> input_dim = torch::lazy::ToVector<int64_t>(input_size);
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d_backward(
      grad_output_tensor,
      GetOutputSizeWithScale(input_dim, scale_factors, output_size),
      input_dim));
}

at::Tensor XLANativeFunctions::upsample_nearest2d(
    const at::Tensor& self, at::IntArrayRef output_size,
    c10::optional<double> scales_h, c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(self_tensor->GetDevice().type());
  if (hw_type != XlaDeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(upsample_nearest2d)>::call(self, output_size,
                                                              scales_h,
                                                              scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d(
      self_tensor, torch::lazy::ToVector<int64_t>(output_size)));
}

at::Tensor XLANativeFunctions::upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(grad_output_tensor->GetDevice().type());
  if (hw_type != XlaDeviceType::TPU || (scales_h && *scales_h != 1.0) ||
      (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(upsample_nearest2d_backward)>::call(grad_output, output_size,
                                                    input_size, scales_h,
                                                    scales_w);
  }
  return bridge::AtenFromXlaTensor(XLATensor::upsample_nearest2d_backward(
      grad_output_tensor, torch::lazy::ToVector<int64_t>(output_size),
      torch::lazy::ToVector<int64_t>(input_size)));
}

at::Tensor XLANativeFunctions::var(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim,
                                   c10::optional<int64_t> correction,
                                   bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::var(
      self_tensor,
      dim ? XlaHelpers::I64List(*dim)
          : torch::lazy::Iota<int64_t>(
                bridge::GetXlaTensor(self)->shape().get().rank()),
      correction ? *correction : 1, keepdim));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::var_mean(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction, bool keepdim) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto results = XLATensor::var_mean(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      correction ? *correction : 1, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::view(const at::Tensor& self,
                                    at::IntArrayRef size) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(
      XLATensor::view(bridge::GetXlaTensor(self), XlaHelpers::I64List(size)));
}

at::Tensor XLANativeFunctions::where(const at::Tensor& condition,
                                     const at::Tensor& self,
                                     const at::Tensor& other) {
  XLA_FN_COUNTER("xla::");
  c10::MaybeOwned<at::Tensor> b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) =
      expand_outplace(condition, self, other, "where");
  return bridge::AtenFromXlaTensor(XLATensor::where(
      bridge::GetXlaTensor(*b_condition), bridge::GetXlaTensor(*b_self),
      bridge::GetXlaTensor(*b_other)));
}

at::Tensor& XLANativeFunctions::zero_(at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensor::zero_(self_tensor);
  return self;
}

at::Scalar XLANativeFunctions::_local_scalar_dense(const at::Tensor& self) {
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    // sync tensors in order to save computation when step is marked later.
    XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
    XLATensor::SyncLiveTensorsGraph(&self_tensor->GetDevice(), /*devices=*/{},
                                    /*wait=*/true);
    XLA_COUNTER("EarlySyncLiveTensorsCount", 1);
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(_local_scalar_dense)>::call(self);
}

// re-use the composite kernel from core, that way we don't need to provide a
// backwards formula for native_layer_norm
std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_layer_norm(const at::Tensor& input,
                                      at::IntArrayRef normalized_shape,
                                      const c10::optional<at::Tensor>& weight,
                                      const c10::optional<at::Tensor>& bias,
                                      double eps) {
  return at::native::math_native_layer_norm(input, normalized_shape, weight,
                                            bias, eps);
}

// re-use the composite kernel from core, that way we don't need to provide a
// backwards formula for native_group_norm
std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_group_norm(const at::Tensor& input,
                                      const c10::optional<at::Tensor>& weight,
                                      const c10::optional<at::Tensor>& bias,
                                      int64_t N, int64_t C, int64_t HxW,
                                      int64_t group, double eps) {
  return at::native::math_group_norm(input, weight, bias, N, C, HxW, group,
                                     eps);
}

// All of the below ops correspond to CompositeExplicitAutograd kernels from
// core that call into view operators internally. These are all composite ops
// that LTC can technically re-use / get for free, but we need to
// "functionalize" them to remove the view ops before we can use them.
at::Tensor XLANativeFunctions::block_diag(at::TensorList tensors) {
  return at::native::block_diag(tensors);
}
at::Tensor XLANativeFunctions::new_empty_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
    c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  return at::native::new_empty_strided(self, size, stride, dtype, layout,
                                       device, pin_memory);
}

at::Tensor XLANativeFunctions::narrow_copy(const at::Tensor& self, int64_t dim,
                                           int64_t start, int64_t length) {
  return at::native::narrow_copy_dense(self, dim, start, length);
}
at::Tensor XLANativeFunctions::pixel_shuffle(const at::Tensor& self,
                                             int64_t upscale_factor) {
  return at::native::math_pixel_shuffle(self, upscale_factor);
}
at::Tensor XLANativeFunctions::pixel_unshuffle(const at::Tensor& self,
                                               int64_t downscale_factor) {
  return at::native::math_pixel_unshuffle(self, downscale_factor);
}
at::Tensor XLANativeFunctions::select_backward(const at::Tensor& grad_output,
                                               at::IntArrayRef input_sizes,
                                               int64_t dim, int64_t index) {
  return at::native::select_backward(grad_output, input_sizes, dim, index);
}
::std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::linalg_inv_ex(
    const at::Tensor& self, bool check_errors) {
  return at::native::linalg_inv_ex(self, check_errors);
}
at::Tensor XLANativeFunctions::linalg_pinv(
    const at::Tensor& self, const c10::optional<at::Tensor>& atol,
    const c10::optional<at::Tensor>& rtol, bool hermitian) {
  return at::native::linalg_pinv(self, atol, rtol, hermitian);
}

at::Tensor XLANativeFunctions::diagonal_backward(const at::Tensor& grad_output,
                                                 at::IntArrayRef input_sizes,
                                                 int64_t offset, int64_t dim1,
                                                 int64_t dim2) {
  return at::native::diagonal_backward(grad_output, input_sizes, offset, dim1,
                                       dim2);
}

at::Tensor XLANativeFunctions::slice_backward(const at::Tensor& grad_output,
                                              at::IntArrayRef input_sizes,
                                              int64_t dim, int64_t start,
                                              int64_t end, int64_t step) {
  return at::native::slice_backward(grad_output, input_sizes, dim, start, end,
                                    step);
}

}  // namespace torch_xla
