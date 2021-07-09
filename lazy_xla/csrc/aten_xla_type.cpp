#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>

#include <mutex>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/debug_util.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/index_ops.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_xla/csrc/XLANativeFunctions.h"
#include "lazy_xla/csrc/aten_autograd_ops.h"
#include "lazy_xla/csrc/aten_autograd_ops_nnc.h"
#include "lazy_xla/csrc/aten_cpu_fallback.h"
#include "lazy_xla/csrc/compiler/nnc_computation_client.h"
#include "lazy_xla/csrc/compiler/pooling.h"
#include "lazy_xla/csrc/compiler/tensor_util.h"
#include "lazy_xla/csrc/version.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

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

namespace torch_lazy_tensors {
namespace {

Device GetLtcDeviceOrCurrent(const c10::optional<c10::Device>& device) {
  auto xla_device_opt = bridge::GetLtcDevice(device);
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
  LTC_CHECK(type1 != at::kBool || type2 != at::kBool)
      << "Subtraction, the `-` operator, with two bool tensors is not "
         "supported. Use the `^` or `logical_xor()` operator instead.";
  LTC_CHECK(type1 != at::kBool && type2 != at::kBool)
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
  lazy_tensors::PrimitiveType ltc_type = TensorTypeToLtcType(dtype);
  switch (ltc_type) {
    case lazy_tensors::PrimitiveType::F16:
      return static_cast<int64_t>(1) << std::numeric_limits<xla::half>::digits;
    case lazy_tensors::PrimitiveType::BF16:
      return static_cast<int64_t>(1)
             << std::numeric_limits<xla::bfloat16>::digits;
    case lazy_tensors::PrimitiveType::F32:
      return static_cast<int64_t>(1) << std::numeric_limits<float>::digits;
    case lazy_tensors::PrimitiveType::F64:
      return static_cast<int64_t>(1) << std::numeric_limits<double>::digits;
    default:
      return Helpers::MinMaxValues(ltc_type).max.toLong();
  }
}

void CheckRangeValues(torch::ScalarType dtype, int64_t from, int64_t to) {
  Helpers::MinMax min_max;
  // Bound the min_max by int64 since types of "from" and "to" are int64.
  if (IsTypeWithLargerRangeThanLong(dtype)) {
    min_max = Helpers::MinMaxValues(lazy_tensors::PrimitiveType::S64);
  } else {
    min_max = Helpers::MinMaxValues(TensorTypeToLtcType(dtype));
  }
  LTC_CHECK_GE(from, min_max.min.toLong());
  LTC_CHECK_LE(from, min_max.max.toLong());
  LTC_CHECK_GE(to, min_max.min.toLong());
  LTC_CHECK_LE(to, min_max.max.toLong());
}

std::pair<LazyTensor, LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  LazyTensor self_tensor;
  LazyTensor other_tensor;
  auto self_xtensor = bridge::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::GetLtcTensor(other);
    self_tensor = bridge::GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = *self_xtensor;
    other_tensor = bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<LazyTensor, LazyTensor>(self_tensor, other_tensor);
}

// The input is in format of {N, C, H, W} and the output will be {H, W}.
std::vector<xla::int64> GetOutputSizeWithScale(
    absl::Span<const xla::int64> input_size,
    const c10::optional<at::ArrayRef<double>>& scale_factors,
    const c10::optional<at::IntArrayRef>& output_size) {
  if (!output_size) {
    LTC_CHECK(scale_factors);
    LTC_CHECK_EQ(scale_factors->size(), 2);
    // Calculate the output size from input_shape and scale_factors
    LTC_CHECK_EQ(input_size.size(), 4);
    xla::int64 output_h = input_size[2] * (*scale_factors)[0];
    xla::int64 output_w = input_size[3] * (*scale_factors)[1];
    return {output_h, output_w};
  }
  LTC_CHECK(!scale_factors);
  return xla::util::ToVector<xla::int64>(*output_size);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<LazyTensor, LazyTensor> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  LazyTensor result = bin_op(operands.first, operands.second, dtype);
  return bridge::AtenFromLtcTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor result = bin_op(self_tensor, other, dtype);
  return bridge::AtenFromLtcTensor(result);
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Tensor& other) {
  at::ScalarType resultType = at::result_type(self, other);
  LTC_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Scalar& other) {
  at::ScalarType resultType = at::result_type(self, other);
  LTC_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

void MarkAsInteropView(at::Tensor& t) {
  dynamic_cast<LTCTensorImpl*>(t.unsafeGetTensorImpl())->MarkAsInteropView();
}

bool ForceNNC() {
  static bool force_nnc =
      lazy_tensors::sys_util::GetEnvBool("FORCE_NNC", false);
  return force_nnc;
}

bool UseNNC(const at::Tensor& self) {
  static int threshold =
      lazy_tensors::sys_util::GetEnvInt("NNC_NUMEL_THRESHOLD", 500000);
  return ForceNNC() || (self.numel() > threshold && GetPythonFrameTop());
}

enum ExecutionKind { NNC, Interop };

ExecutionKind InPlaceMustUseNNC(const at::Tensor& self) {
  const LazyTensor self_tensor = bridge::GetLtcTensor(self);
  const bool must_use_interop = bridge::IsInteropView(self);
  const bool must_use_nnc = self_tensor.GetViewAliasId();
  LTC_CHECK(!must_use_nnc || !must_use_interop);
  return must_use_nnc ? ExecutionKind::NNC : ExecutionKind::Interop;
}

ExecutionKind InPlaceUseNNC(const at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    return ExecutionKind::NNC;
  }
  const bool must_use_interop = bridge::IsInteropView(self);
  if (must_use_interop) {
    return ExecutionKind::Interop;
  }
  return UseNNC(self) ? ExecutionKind::NNC : ExecutionKind::Interop;
}

bool UseNNCViews(const LazyTensor& self_tensor) {
  static bool force_nnc_views =
      lazy_tensors::sys_util::GetEnvBool("FORCE_NNC_VIEWS", false);
  const auto device_data =
      ir::ops::DeviceData::Cast(self_tensor.GetIrValue().node.get());
  return !device_data || force_nnc_views;
}
}  // namespace

at::Tensor& __ilshift__(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& __ilshift__(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor& __irshift__(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& __irshift__(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor __lshift__(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, other, dtype);
                    });
}

at::Tensor __lshift__(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor __rshift__(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, other, dtype);
                    });
}

at::Tensor __rshift__(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor _adaptive_avg_pool3d(const at::Tensor& self,
                                at::IntArrayRef output_size) {
  LTC_FN_COUNTER("xla::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()),
                                  output_size_list, /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(_adaptive_avg_pool3d)>::call(self,
                                                                output_size);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::adaptive_avg_pool3d(
      bridge::GetLtcTensor(self), output_size_list));
}

at::Tensor _adaptive_avg_pool3d_backward(const at::Tensor& grad_output,
                                         const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<xla::int64> output_size{grad_output.size(rank - 3),
                                      grad_output.size(rank - 2),
                                      grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(_adaptive_avg_pool3d_backward)>::call(grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::adaptive_avg_pool3d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self)));
}

at::Tensor _adaptive_avg_pool2d(const at::Tensor& self,
                                at::IntArrayRef output_size) {
  LTC_FN_COUNTER("xla::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()),
                                  output_size_list, /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(_adaptive_avg_pool2d)>::call(self,
                                                                output_size);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::_adaptive_avg_pool2d(
      bridge::GetLtcTensor(self), output_size_list));
}

at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor& grad_output,
                                         const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<xla::int64> output_size{grad_output.size(rank - 2),
                                      grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(_adaptive_avg_pool2d_backward)>::call(grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::_adaptive_avg_pool2d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self)));
}

void _amp_foreach_non_finite_check_and_unscale_(at::TensorList self,
                                                at::Tensor& found_inf,
                                                const at::Tensor& inv_scale) {
  LTC_FN_COUNTER("xla::");
  LazyTensor found_inf_tensor = bridge::GetLtcTensor(found_inf);
  DeviceType hw_type = found_inf_tensor.GetDevice().hw_type;
  LTC_CHECK(hw_type == DeviceType::GPU || hw_type == DeviceType::CPU)
      << "AMP should be used with XLA:GPU";
  LazyTensor::_amp_foreach_non_finite_check_and_unscale_(
      bridge::GetLtcTensors(self), found_inf_tensor,
      bridge::GetLtcTensor(inv_scale));
}

at::Tensor& _amp_update_scale_(at::Tensor& current_scale,
                               at::Tensor& growth_tracker,
                               const at::Tensor& found_inf,
                               double scale_growth_factor,
                               double scale_backoff_factor,
                               int64_t growth_interval) {
  LTC_FN_COUNTER("xla::");
  LazyTensor growth_tracker_tensor = bridge::GetLtcTensor(growth_tracker);
  LazyTensor current_scale_tensor = bridge::GetLtcTensor(current_scale);
  DeviceType hw_type = growth_tracker_tensor.GetDevice().hw_type;
  LTC_CHECK(hw_type == DeviceType::GPU || hw_type == DeviceType::CPU)
      << "AMP should be used with XLA:GPU";
  LazyTensor::_amp_update_scale_(growth_tracker_tensor, current_scale_tensor,
                                 bridge::GetLtcTensor(found_inf),
                                 scale_growth_factor, scale_backoff_factor,
                                 growth_interval);
  return current_scale;
}

at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst,
                      bool non_blocking) {
  LTC_FN_COUNTER("xla::");
  auto dst_tensor = bridge::TryGetLtcTensor(dst);
  auto self_tensor = bridge::TryGetLtcTensor(self);
  if (!self_tensor) {
    static bool sync_update =
        lazy_tensors::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    LTC_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    LazyTensor::copy_(*dst_tensor, *self_tensor);
    bridge::ReplaceLtcTensor(dst, *dst_tensor);
  }
  return dst;
}

at::Tensor& _index_put_impl_(
    at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool /* unsafe */) {
  LTC_FN_COUNTER("xla::");
  return torch_lazy_tensors::index_put_(self, indices, values, accumulate);
}

at::Tensor _copy_from_and_resize(const at::Tensor& self,
                                 const at::Tensor& dst) {
  XLA_FN_COUNTER("xla::");
  auto dst_tensor = bridge::TryGetLtcTensor(dst);
  auto self_tensor = bridge::TryGetLtcTensor(self);
  if (!self_tensor) {
    XLA_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensorOut(self);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    // at this point we know dst is an XLA tensor
    LTCTensorImpl* dest_impl =
        dynamic_cast<LTCTensorImpl*>(dst.unsafeGetTensorImpl());
    dest_impl->tensor().UpdateFromTensorOut(*self_tensor);
    dest_impl->force_refresh_sizes();
  }
  return dst;
}

std::vector<at::Tensor> _to_cpu(at::TensorList tensors) {
  XLA_FN_COUNTER("xla::");
  return bridge::LtcCreateTensorList(tensors);
}

at::Tensor _log_softmax(const at::Tensor& self, int64_t dim,
                        bool /* half_to_float */) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::log_softmax(bridge::GetLtcTensor(self), dim, c10::nullopt));
}

at::Tensor _log_softmax_backward_data(const at::Tensor& grad_output,
                                      const at::Tensor& output, int64_t dim,
                                      const at::Tensor& /* self */) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_softmax_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output), dim));
}

std::tuple<at::Tensor, at::Tensor> _pack_padded_sequence(
    const at::Tensor& input, const at::Tensor& lengths, bool batch_first) {
  LTC_FN_COUNTER("xla::");
  std::vector<at::Tensor> xla_tensors = {lengths};
  auto cpu_tensors = bridge::LtcCreateTensorList(xla_tensors);
  return at::native::_pack_padded_sequence(input, cpu_tensors[0], batch_first);
}

at::Tensor _s_where(const at::Tensor& condition, const at::Tensor& self,
                    const at::Tensor& other) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::where(
        bridge::GetLtcTensor(condition), bridge::GetLtcTensor(self),
        bridge::GetLtcTensor(other)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(_s_where)>::call(condition, self,
                                                               other);
}

at::Tensor _softmax(const at::Tensor& self, int64_t dim,
                    bool /* half_to_float */) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softmax(bridge::GetLtcTensor(self), dim, c10::nullopt));
}

at::Tensor _softmax_backward_data(const at::Tensor& grad_output,
                                  const at::Tensor& output, int64_t dim,
                                  const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::softmax_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output), dim));
}

at::Tensor _trilinear(const at::Tensor& i1, const at::Tensor& i2,
                      const at::Tensor& i3, at::IntArrayRef expand1,
                      at::IntArrayRef expand2, at::IntArrayRef expand3,
                      at::IntArrayRef sumdim, int64_t unroll_dim) {
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP(_trilinear)>::call(i1, i2, i3, expand1,
                                                    expand2, expand3, sumdim,
                                                    unroll_dim);
}

at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return view(self, size);
  }
  auto result =
      at::native::call_fallback_fn<&xla_cpu_fallback,
                                   ATEN_OP(_unsafe_view)>::call(self, size);
  MarkAsInteropView(result);
  return result;
}

at::Tensor abs(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::abs(bridge::GetLtcTensor(self)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(abs)>::call(
      self);
}

at::Tensor acos(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::acos(bridge::GetLtcTensor(self)));
}

at::Tensor acosh(const at::Tensor& self) {
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(acosh)>::call(
      self);
}

at::Tensor add(const at::Tensor& self, const at::Tensor& other,
               const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    at::native::alpha_check(at::result_type(self, other), alpha);
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const LazyTensor& xother,
                          at::ScalarType dtype) {
                        return LazyTensor::add(xself, xother, alpha, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(add, Tensor)>::call(self, other,
                                                                   alpha);
}

at::Tensor add(const at::Tensor& self, const at::Scalar& other,
               const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const at::Scalar& other,
                          at::ScalarType dtype) {
                        return LazyTensor::add(xself, other, alpha, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(add, Scalar)>::call(self, other,
                                                                   alpha);
}

at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& tensor1,
                   const at::Tensor& tensor2, const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcdiv(
      bridge::GetLtcTensor(self), value, bridge::GetLtcTensor(tensor1),
      bridge::GetLtcTensor(tensor2)));
}

at::Tensor& addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                     const at::Tensor& tensor2, const at::Scalar& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::addcdiv_(self_tensor, value, bridge::GetLtcTensor(tensor1),
                         bridge::GetLtcTensor(tensor2));
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(addcdiv_)>::call(self, tensor1,
                                                               tensor2, value);
}

at::Tensor addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                   const at::Tensor& tensor2, const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcmul(
      bridge::GetLtcTensor(self), value, bridge::GetLtcTensor(tensor1),
      bridge::GetLtcTensor(tensor2)));
}

at::Tensor addmm(const at::Tensor& self, const at::Tensor& mat1,
                 const at::Tensor& mat2, const at::Scalar& beta,
                 const at::Scalar& alpha) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (beta.to<double>() != 1 || alpha.to<double>() != 1 ||
      !at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat1) ||
      !at::native::is_floating_point(mat2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(addmm)>::call(self, mat1, mat2,
                                                              beta, alpha);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::addmm(bridge::GetLtcTensor(mat1),
                        /*weight=*/bridge::GetLtcTensor(mat2),
                        /*bias=*/bridge::GetLtcTensor(self)));
}

at::Tensor alias(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return self;
}

at::Tensor all(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::all(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor all(const at::Tensor& self, int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::all(bridge::GetLtcTensor(self), {dim}, keepdim));
}

at::Tensor any(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::any(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor any(const at::Tensor& self, int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::any(bridge::GetLtcTensor(self), {dim}, keepdim));
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end,
                       const at::Scalar& step, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::arange_out(out_tensor, start, end, step, out.scalar_type());
  return out;
}

at::Tensor argmax(const at::Tensor& self, c10::optional<int64_t> dim,
                  bool keepdim) {
  LTC_FN_COUNTER("xla::");
  return dim ? bridge::AtenFromLtcTensor(LazyTensor::argmax(
                   bridge::GetLtcTensor(self), *dim, keepdim))
             : bridge::AtenFromLtcTensor(
                   LazyTensor::argmax(bridge::GetLtcTensor(self)));
}

at::Tensor argmin(const at::Tensor& self, c10::optional<int64_t> dim,
                  bool keepdim) {
  LTC_FN_COUNTER("xla::");
  return dim ? bridge::AtenFromLtcTensor(LazyTensor::argmin(
                   bridge::GetLtcTensor(self), *dim, keepdim))
             : bridge::AtenFromLtcTensor(
                   LazyTensor::argmin(bridge::GetLtcTensor(self)));
}

at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size,
                      at::IntArrayRef stride,
                      c10::optional<int64_t> storage_offset) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  const auto device_data =
      ir::ops::DeviceData::Cast(self_tensor.GetIrValue().node.get());
  if (!UseNNCViews(self_tensor) &&
      (device_data || self_tensor.CurrentTensorData())) {
    auto result =
        at::native::call_fallback_fn<&xla_cpu_fallback,
                                     ATEN_OP(as_strided)>::call(self, size,
                                                                stride,
                                                                storage_offset);
    return result;
  }
  LTC_FN_COUNTER("xla::");
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(as_strided)>::call(self, size, stride,
                                                      storage_offset);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::as_strided(self_tensor, std::move(xsize), std::move(xstride),
                             Helpers::I64Optional(storage_offset)));
}

const at::Tensor& as_strided_(const at::Tensor& self, at::IntArrayRef size,
                              at::IntArrayRef stride,
                              c10::optional<int64_t> storage_offset) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC &&
      ir::ops::AsStrided::StrideIsSupported(self_tensor.shape(), xsize, xstride,
                                            storage_offset.value_or(0))) {
    LTC_FN_COUNTER("xla::");
    LazyTensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                            Helpers::I64Optional(storage_offset));
    return self;
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::as_strided_", 1);
  LTC_VLOG(3) << "XLA as_strided_ :"
              << " self=" << self.toString();
  auto xlatens = bridge::LtcCreateTensorList({self});
  at::as_strided_(xlatens[0], size, stride, storage_offset);
  bridge::LtcUpdateTensors({self}, xlatens, {0});
  return self;
}

at::Tensor asin(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::asin(bridge::GetLtcTensor(self)));
}

at::Tensor asinh(const at::Tensor& self) {
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(asinh)>::call(
      self);
}

at::Tensor atan(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::atan(bridge::GetLtcTensor(self)));
}

at::Tensor atanh(const at::Tensor& self) {
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(atanh)>::call(
      self);
}

at::Tensor atan2(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  // xla::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(atan2)>::call(self, other);
  }
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::atan2(xself, xother, dtype);
                    });
}

at::Tensor avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding,
                      bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(avg_pool2d)>::call(self, kernel_size, stride,
                                                      padding, ceil_mode,
                                                      count_include_pad,
                                                      divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd(
      bridge::GetLtcTensor(self), /*spatial_dim_count=*/2,
      Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor avg_pool2d_backward(const at::Tensor& grad_output,
                               const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::
        call_fallback_fn<&xla_cpu_fallback, ATEN_OP(avg_pool2d_backward)>::call(
            grad_output, self, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size),
      Helpers::I64List(stride), Helpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor avg_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding,
                      bool ceil_mode, bool count_include_pad,
                      c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(avg_pool3d)>::call(self, kernel_size, stride,
                                                      padding, ceil_mode,
                                                      count_include_pad,
                                                      divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd(
      bridge::GetLtcTensor(self), /*spatial_dim_count=*/3,
      Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode, count_include_pad));
}

at::Tensor avg_pool3d_backward(const at::Tensor& grad_output,
                               const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               c10::optional<int64_t> divisor_override) {
  LTC_FN_COUNTER("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::
        call_fallback_fn<&xla_cpu_fallback, ATEN_OP(avg_pool3d_backward)>::call(
            grad_output, self, kernel_size, stride, padding, ceil_mode,
            count_include_pad, divisor_override);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::avg_pool_nd_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size),
      Helpers::I64List(stride), Helpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor baddbmm(const at::Tensor& self, const at::Tensor& batch1,
                   const at::Tensor& batch2, const at::Scalar& beta,
                   const at::Scalar& alpha) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(batch1) ||
      !at::native::is_floating_point(batch2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(baddbmm)>::call(self, batch1,
                                                                batch2, beta,
                                                                alpha);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::baddbmm(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(batch1),
      bridge::GetLtcTensor(batch2), beta, alpha));
}

at::Tensor bernoulli(const at::Tensor& self,
                     c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(bernoulli)>::call(self,
                                                                  generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::bernoulli(self_tensor));
}

at::Tensor& bernoulli_(at::Tensor& self, double p,
                       c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(bernoulli_, float)>::call(self, p,
                                                              generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::bernoulli_(self_tensor, p);
  return self;
}

at::Tensor& bernoulli_(at::Tensor& self, const at::Tensor& p,
                       c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(bernoulli_, Tensor)>::call(self, p,
                                                               generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::bernoulli_(self_tensor, bridge::GetLtcTensor(p));
  return self;
}

at::Tensor binary_cross_entropy(const at::Tensor& self,
                                const at::Tensor& target,
                                const c10::optional<at::Tensor>& weight,
                                int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor weight_tensor =
      bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromLtcTensor(LazyTensor::binary_cross_entropy(
      self_tensor, bridge::GetLtcTensor(target), weight_tensor, reduction));
}

at::Tensor binary_cross_entropy_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const c10::optional<at::Tensor>& weight,
    int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor weight_tensor =
      bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  return bridge::AtenFromLtcTensor(LazyTensor::binary_cross_entropy_backward(
      bridge::GetLtcTensor(grad_output), self_tensor,
      bridge::GetLtcTensor(target), weight_tensor, reduction));
}

at::Tensor binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight, int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  return at::native::binary_cross_entropy_with_logits(
      self, target, IsDefined(weight) ? *weight : at::Tensor(),
      IsDefined(pos_weight) ? *pos_weight : at::Tensor(), reduction);
}

at::Tensor& bitwise_not_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::bitwise_not_out(out_tensor, self_tensor);
  return out;
}

at::Tensor& bitwise_or_out(const at::Tensor& self, const at::Scalar& other,
                           at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::GetLtcTensor(self), other);
  return out;
}

at::Tensor& bitwise_or_out(const at::Tensor& self, const at::Tensor& other,
                           at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::GetLtcTensor(self),
                             bridge::GetLtcTensor(other));
  return out;
}

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Scalar& other,
                            at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::GetLtcTensor(self), other);
  return out;
}

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Tensor& other,
                            at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::GetLtcTensor(self),
                              bridge::GetLtcTensor(other));
  return out;
}

at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(bmm)>::call(
        self, mat2);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::bmm(bridge::GetLtcTensor(self), bridge::GetLtcTensor(mat2)));
}

at::Tensor cat(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cat(bridge::GetLtcTensors(tensors), dim));
}

at::Tensor ceil(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ceil(bridge::GetLtcTensor(self)));
}

at::Tensor cholesky(const at::Tensor& self, bool upper) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cholesky(bridge::GetLtcTensor(self), upper));
}

at::Tensor clamp(const at::Tensor& self, const c10::optional<at::Scalar>& min,
                 const c10::optional<at::Scalar>& max) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min, max));
}

at::Tensor clamp(const at::Tensor& self, const c10::optional<at::Tensor>& min,
                 const c10::optional<at::Tensor>& max) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min, max));
}

at::Tensor clamp_max(const at::Tensor& self, const at::Scalar& max) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), c10::nullopt, max));
}

at::Tensor& clamp_max_out(const at::Tensor& self, const at::Tensor& max,
                          at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::clamp_out(out_tensor, bridge::GetLtcTensor(self), c10::nullopt,
                        max);
  return out;
}

at::Tensor clamp_min(const at::Tensor& self, const at::Scalar& min) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min, c10::nullopt));
}

at::Tensor& clamp_min_out(const at::Tensor& self, const at::Tensor& min,
                          at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::clamp_out(out_tensor, bridge::GetLtcTensor(self), min,
                        c10::nullopt);
  return out;
}

at::Tensor clone(const at::Tensor& self,
                 c10::optional<at::MemoryFormat> memory_format) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (ForceNNC()) {
    return bridge::AtenFromLtcTensor(LazyTensor::clone(self_tensor));
  }
  if (self_tensor.CurrentTensorData()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(clone)>::call(self,
                                                              memory_format);
  }
  const at::DeviceType device_type =
      xla::compiler::NNCComputationClient::HardwareDeviceType();
  return bridge::CreateLtcTensor(
      bridge::AtenFromLtcTensor(LazyTensor::clone(self_tensor)).to(device_type),
      bridge::GetLtcDevice(self));
}

at::Tensor constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad,
                           const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::constant_pad_nd(
      bridge::GetLtcTensor(self), Helpers::I64List(pad), value));
}

// This functions covers the whole convolution lowering.
at::Tensor convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  LTC_FN_COUNTER("xla::");
  if (IsDefined(bias)) {
    return bridge::AtenFromLtcTensor(LazyTensor::convolution_overrideable(
        bridge::GetLtcTensor(input), bridge::GetLtcTensor(weight),
        bridge::GetLtcTensor(*bias), Helpers::I64List(stride),
        Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups));
  } else {
    return bridge::AtenFromLtcTensor(LazyTensor::convolution_overrideable(
        bridge::GetLtcTensor(input), bridge::GetLtcTensor(weight),
        Helpers::I64List(stride), Helpers::I64List(padding),
        Helpers::I64List(dilation), transposed,
        Helpers::I64List(output_padding), groups));
  }
}

// This functions covers the whole convolution backward lowering.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("xla::");
  auto gradients = LazyTensor::convolution_backward_overrideable(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(input),
      bridge::GetLtcTensor(weight), Helpers::I64List(stride),
      Helpers::I64List(padding), Helpers::I64List(dilation), transposed,
      Helpers::I64List(output_padding), groups);
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients))
                     : at::Tensor(),
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients))
                     : at::Tensor(),
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients))
                     : at::Tensor());
}

at::Tensor cos(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::cos(bridge::GetLtcTensor(self)));
}

at::Tensor cosh(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cosh(bridge::GetLtcTensor(self)));
}

at::Tensor cross(const at::Tensor& self, const at::Tensor& other,
                 c10::optional<int64_t> dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cross(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other),
                        Helpers::I64Optional(dim)));
}

at::Tensor cumprod(const at::Tensor& self, int64_t dim,
                   c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  c10::optional<at::ScalarType> promoted_dtype =
      PromoteIntegralType(self_tensor.dtype(), dtype);
  if (IsOperationOnType(promoted_dtype, self_tensor.dtype(),
                        at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(cumprod)>::call(self, dim,
                                                                dtype);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::cumprod(self_tensor, dim, promoted_dtype));
}

at::Tensor cumsum(const at::Tensor& self, int64_t dim,
                  c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (IsOperationOnType(dtype, self_tensor.dtype(), at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(cumsum)>::call(self, dim,
                                                               dtype);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::cumsum(self_tensor, dim, dtype));
}

at::Tensor diag(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diag(bridge::GetLtcTensor(self), diagonal));
}

at::Tensor diagonal(const at::Tensor& self, int64_t offset, int64_t dim1,
                    int64_t dim2) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diagonal(bridge::GetLtcTensor(self), offset, dim1, dim2));
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other) {
  return torch_lazy_tensors::div(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor div(const at::Tensor& self, const at::Tensor& other,
               c10::optional<c10::string_view> rounding_mode) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    at::ScalarType dtype = at::result_type(self, other);
    auto operands = GetBinaryOperands(self, other);
    return bridge::AtenFromLtcTensor(
        LazyTensor::div(operands.first, operands.second, rounding_mode, dtype));
  }
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP2(div, Tensor_mode)>::call(self, other,
                                                           rounding_mode);
}

at::Tensor div(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::div(bridge::GetLtcTensor(self), other));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(div, Scalar)>::call(self, other);
}

at::Tensor dot(const at::Tensor& self, const at::Tensor& tensor) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK_EQ(self.dim(), 1)
      << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  LTC_CHECK_EQ(tensor.dim(), 1)
      << "dot: Expected 1-D argument tensor, but got " << tensor.dim() << "-D";
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(tensor)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(dot)>::call(
        self, tensor);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::matmul(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(tensor)));
}

at::Tensor elu(const at::Tensor& self, const at::Scalar& alpha,
               const at::Scalar& scale, const at::Scalar& input_scale) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::elu(bridge::GetLtcTensor(self), alpha, scale, input_scale));
}

at::Tensor& elu_(at::Tensor& self, const at::Scalar& alpha,
                 const at::Scalar& scale, const at::Scalar& input_scale) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::elu_(self_tensor, alpha, scale, input_scale);
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(elu_)>::call(
      self, alpha, scale, input_scale);
}

at::Tensor elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha,
                        const at::Scalar& scale, const at::Scalar& input_scale,
                        bool self, const at::Tensor& self_or_result) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK(!self || alpha.to<double>() >= 0.0)
      << "In-place elu backward calculation is triggered with a negative slope "
         "which is not supported.";
  return bridge::AtenFromLtcTensor(LazyTensor::elu_backward(
      bridge::GetLtcTensor(grad_output), alpha, scale, input_scale,
      bridge::GetLtcTensor(self_or_result)));
}

at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices,
                     int64_t padding_idx, bool scale_grad_by_freq,
                     bool sparse) {
  LTC_FN_COUNTER("xla::");
  // TODO: for now route to native, which dispatches supported XLA operations.
  // We need to make use of the TPU embedding core here eventually.
  return at::native::embedding(weight, indices, padding_idx, scale_grad_by_freq,
                               sparse);
}

at::Tensor embedding_dense_backward(const at::Tensor& grad_output,
                                    const at::Tensor& indices,
                                    int64_t num_weights, int64_t padding_idx,
                                    bool scale_grad_by_freq) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::embedding_dense_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

at::Tensor empty(at::IntArrayRef size, c10::optional<at::ScalarType> dtype,
                 c10::optional<at::Layout> layout,
                 c10::optional<at::Device> device,
                 c10::optional<bool> pin_memory,
                 c10::optional<at::MemoryFormat> memory_format) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    // PT empty*() are optimizations to avoid initializing the data when it is
    // known it will be completely rewritten. But since for us doing a zero*()
    // does not actually end up doing any memory initialization, we use that and
    // avoid going to CPU for it. A common PT pattern is indeed doing empty()
    // plus s_copy_().
    return bridge::AtenFromLtcTensor(LazyTensor::full(
        Helpers::I64List(size), 0, GetLtcDeviceOrCurrent(device),
        GetScalarTypeOrFloat(dtype)));
  }
  const auto device_type =
      xla::compiler::NNCComputationClient::HardwareDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(device));
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         c10::optional<at::ScalarType> dtype,
                         c10::optional<at::Layout> layout,
                         c10::optional<at::Device> device,
                         c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("xla::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return torch_lazy_tensors::as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor eq(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::eq(bridge::GetLtcTensor(self), other));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(eq, Scalar)>::call(self, other);
}

at::Tensor eq(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::eq(
        bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(eq, Tensor)>::call(self, other);
}

at::Tensor erf(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::erf(bridge::GetLtcTensor(self)));
}

at::Tensor erfc(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::erfc(bridge::GetLtcTensor(self)));
}

at::Tensor erfinv(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::erfinv(bridge::GetLtcTensor(self)));
}

at::Tensor exp(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::exp(bridge::GetLtcTensor(self)));
}

at::Tensor expand(const at::Tensor& self, at::IntArrayRef size, bool implicit) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::expand(bridge::GetLtcTensor(self),
                           lazy_tensors::util::ToVector<xla::int64>(size)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(expand)>::call(
      self, size, implicit);
}

at::Tensor expm1(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::expm1(bridge::GetLtcTensor(self)));
}

at::Tensor& exponential_(at::Tensor& self, double lambd,
                         c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(exponential_)>::call(self,
                                                                     lambd,
                                                                     generator);
  }
  LTC_CHECK_GE(lambd, 0.0);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::exponential_(self_tensor, lambd);
  return self;
}

at::Tensor& eye_out(int64_t n, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::eye_out(out_tensor, n, n);
  return out;
}

at::Tensor& eye_out(int64_t n, int64_t m, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::eye_out(out_tensor, n, m);
  return out;
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::fill_(self_tensor, value);
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(fill_, Scalar)>::call(self,
                                                                     value);
}

at::Tensor& fill_(at::Tensor& self, const at::Tensor& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LTC_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                                 << "value tensor, but got tensor "
                                 << "with " << value.dim() << " dimension(s).";
    return torch_lazy_tensors::fill_(self, value.item());
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(fill_, Tensor)>::call(self,
                                                                     value);
}

at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::flip(bridge::GetLtcTensor(self), Helpers::I64List(dims)));
}

at::Tensor floor(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::floor(bridge::GetLtcTensor(self)));
}

at::Tensor fmod(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, xother, dtype);
                    });
}

at::Tensor fmod(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, other, dtype);
                    });
}

at::Tensor frac(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::frac(bridge::GetLtcTensor(self)));
}

at::Tensor gather(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                  bool /* sparse_grad */) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::gather(
      bridge::GetLtcTensor(self), dim, bridge::GetLtcTensor(index)));
}

at::Tensor ge(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ge(bridge::GetLtcTensor(self), other));
}

at::Tensor ge(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ge(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor gelu(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::gelu(bridge::GetLtcTensor(self)));
}

at::Tensor gelu_backward(const at::Tensor& grad, const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::gelu_backward(
      bridge::GetLtcTensor(grad), bridge::GetLtcTensor(self)));
}

at::Tensor ger(const at::Tensor& self, const at::Tensor& vec2) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ger(bridge::GetLtcTensor(self), bridge::GetLtcTensor(vec2)));
}

at::Tensor gt(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::gt(bridge::GetLtcTensor(self), other));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(gt, Scalar)>::call(self, other);
}

at::Tensor gt(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::gt(
        bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(gt, Tensor)>::call(self, other);
}

at::Tensor hardshrink(const at::Tensor& self, const at::Scalar& lambda) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardshrink(bridge::GetLtcTensor(self), lambda));
}

at::Tensor hardsigmoid(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::hardsigmoid(bridge::GetLtcTensor(self)));
}

at::Tensor hardsigmoid_backward(const at::Tensor& grad_output,
                                const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::hardsigmoid_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self)));
}

at::Tensor hardshrink_backward(const at::Tensor& grad_out,
                               const at::Tensor& self,
                               const at::Scalar& lambda) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::hardshrink_backward(
      bridge::GetLtcTensor(grad_out), bridge::GetLtcTensor(self), lambda));
}

at::Tensor hardtanh(const at::Tensor& self, const at::Scalar& min_val,
                    const at::Scalar& max_val) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min_val, max_val));
}

at::Tensor hardtanh_backward(const at::Tensor& grad_output,
                             const at::Tensor& self, const at::Scalar& min_val,
                             const at::Scalar& max_val) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::hardtanh_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self), min_val,
      max_val));
}

at::Tensor index(const at::Tensor& self,
                 const c10::List<c10::optional<at::Tensor>>& indices) {
  LTC_FN_COUNTER("xla::");
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  return bridge::AtenFromLtcTensor(
      LazyTensor::index(bridge::GetLtcTensor(canonical_index_info.base),
                        bridge::GetLtcTensors(canonical_index_info.indices),
                        canonical_index_info.start_dim));
}

at::Tensor& index_add_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                       const at::Tensor& source) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::index_add_(self_tensor, dim, bridge::GetLtcTensor(index),
                         bridge::GetLtcTensor(source));
  return self;
}

at::Tensor& index_copy_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Tensor& source) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::index_copy_(self_tensor, dim, bridge::GetLtcTensor(index),
                          bridge::GetLtcTensor(source));
  return self;
}

at::Tensor& index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::index_fill_(self_tensor, dim, bridge::GetLtcTensor(index), value);
  return self;
}

at::Tensor& index_fill_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                        const at::Tensor& value) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::index_fill_(self_tensor, dim, bridge::GetLtcTensor(index),
                          bridge::GetLtcTensor(value));
  return self;
}

at::Tensor& index_put_(at::Tensor& self,
                       const c10::List<c10::optional<at::Tensor>>& indices,
                       const at::Tensor& values, bool accumulate) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK(self.scalar_type() == values.scalar_type());
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::index_put_(
      self_tensor, bridge::GetLtcTensor(canonical_index_info.base),
      bridge::GetLtcTensors(canonical_index_info.indices),
      canonical_index_info.start_dim, bridge::GetLtcTensor(values), accumulate,
      canonical_index_info.result_permutation);
  return self;
}

at::Tensor index_select(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::index_select(
      bridge::GetLtcTensor(self), dim, bridge::GetLtcTensor(index)));
}

at::Tensor inverse(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::inverse(bridge::GetLtcTensor(self)));
}

at::Tensor isnan(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::isnan(bridge::GetLtcTensor(self)));
}

at::Tensor kl_div(const at::Tensor& self, const at::Tensor& target,
                  int64_t reduction, bool log_target) {
  LTC_FN_COUNTER("xla::");
  return at::native::kl_div(self, target, reduction, log_target);
}

at::Tensor kl_div_backward(const at::Tensor& grad_output,
                           const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction, bool log_target) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::kl_div_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(target), reduction, log_target));
}

std::tuple<at::Tensor, at::Tensor> kthvalue(const at::Tensor& self, int64_t k,
                                            int64_t dim, bool keepdim) {
  LTC_FN_COUNTER("xla::");
  auto results =
      LazyTensor::kthvalue(bridge::GetLtcTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor l1_loss(const at::Tensor& self, const at::Tensor& target,
                   int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::l1_loss(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(target), reduction));
}

at::Tensor l1_loss_backward(const at::Tensor& grad_output,
                            const at::Tensor& self, const at::Tensor& target,
                            int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::l1_loss_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(target), reduction));
}

at::Tensor le(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::le(bridge::GetLtcTensor(self), other));
}

at::Tensor le(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::le(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor leaky_relu(const at::Tensor& self,
                      const at::Scalar& negative_slope) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::leaky_relu(
      bridge::GetLtcTensor(self), negative_slope.to<double>()));
}

at::Tensor leaky_relu_backward(const at::Tensor& grad_output,
                               const at::Tensor& self,
                               const at::Scalar& negative_slope,
                               bool self_is_result) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    LTC_CHECK(!self_is_result || negative_slope.to<double>() > 0.0);
    return bridge::AtenFromLtcTensor(LazyTensor::leaky_relu_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
        negative_slope.to<double>()));
  }
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP(leaky_relu_backward)>::call(grad_output, self,
                                                             negative_slope,
                                                             self_is_result);
}

at::Tensor lerp(const at::Tensor& self, const at::Tensor& end,
                const at::Tensor& weight) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK_EQ(self.dtype(), end.dtype())
      << "expected dtype " << self.dtype() << " for `end` but got dtype "
      << end.dtype();
  LTC_CHECK_EQ(self.dtype(), weight.dtype())
      << "expected dtype " << self.dtype() << " for `weight` but got dtype "
      << weight.dtype();
  return bridge::AtenFromLtcTensor(
      LazyTensor::lerp(bridge::GetLtcTensor(self), bridge::GetLtcTensor(end),
                       bridge::GetLtcTensor(weight)));
}

at::Tensor lerp(const at::Tensor& self, const at::Tensor& end,
                const at::Scalar& weight) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK_EQ(self.dtype(), end.dtype())
      << "expected dtype " << self.dtype() << " for `end` but got dtype "
      << end.dtype();
  return bridge::AtenFromLtcTensor(LazyTensor::lerp(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(end), weight));
}

at::Tensor log(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log(bridge::GetLtcTensor(self)));
}

at::Tensor log10(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(
      bridge::GetLtcTensor(self), ir::OpKind(at::aten::log10), 10.0));
}

at::Tensor log1p(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::log1p(bridge::GetLtcTensor(self)));
}

at::Tensor& log1p_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::log1p_(self_tensor);
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(log1p_)>::call(
      self);
}

at::Tensor log2(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(
      bridge::GetLtcTensor(self), ir::OpKind(at::aten::log2), 2.0));
}

at::Tensor log_sigmoid_backward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                const at::Tensor& buffer) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_sigmoid_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  auto result_tuple =
      LazyTensor::log_sigmoid_forward(bridge::GetLtcTensor(self));
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromLtcTensor(std::get<1>(result_tuple)));
}

at::Tensor logsumexp(const at::Tensor& self, at::IntArrayRef dim,
                     bool keepdim) {
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(logsumexp)>::call(self, dim,
                                                                keepdim);
}

at::Tensor logdet(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::logdet(bridge::GetLtcTensor(self)));
}

at::Tensor lt(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::lt(bridge::GetLtcTensor(self), other));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(lt, Scalar)>::call(self, other);
}

at::Tensor lt(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::lt(
        bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(lt, Tensor)>::call(self, other);
}

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask,
                         const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::masked_fill_(self_tensor, bridge::GetLtcTensor(mask), value);
  return self;
}

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask,
                         const at::Tensor& value) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK_EQ(value.dim(), 0) << "masked_fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return masked_fill_(self, mask, value.item());
}

at::Tensor& masked_scatter_(at::Tensor& self, const at::Tensor& mask,
                            const at::Tensor& source) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::masked_scatter_(self_tensor, bridge::GetLtcTensor(mask),
                              bridge::GetLtcTensor(source));
  return self;
}

at::Tensor masked_select(const at::Tensor& self, const at::Tensor& mask) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  // Initially make XLA handled masked_select() handling experimental, and
  // opt-in.
  if (!DebugUtil::ExperimentEnabled("masked_select")) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(masked_select)>::call(self,
                                                                      mask);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::masked_select(self_tensor, bridge::GetLtcTensor(mask)));
}

at::Tensor max(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::max(bridge::GetLtcTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self, int64_t dim,
                                       bool keepdim) {
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(max, dim)>::call(self, dim,
                                                                keepdim);
}

at::Tensor maximum(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::max(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> max_out(const at::Tensor& self,
                                             int64_t dim, bool keepdim,
                                             at::Tensor& max,
                                             at::Tensor& max_values) {
  LTC_FN_COUNTER("xla::");
  LazyTensor max_tensor = bridge::GetLtcTensor(max);
  LazyTensor max_values_tensor = bridge::GetLtcTensor(max_values);
  LazyTensor::max_out(max_tensor, max_values_tensor, bridge::GetLtcTensor(self),
                      dim, keepdim);
  return std::forward_as_tuple(max, max_values);
}

at::Tensor max_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("xla::");
  return aten_autograd_ops::MaxPool2dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("xla::");
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
  auto outputs = LazyTensor::max_pool_nd(
      bridge::GetLtcTensor(self), /*spatial_dim_count=*/2,
      Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  LTC_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output, self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/2, Helpers::I64List(kernel_size),
      Helpers::I64List(stride), Helpers::I64List(padding), ceil_mode));
}

at::Tensor max_pool3d(const at::Tensor& self, at::IntArrayRef kernel_size,
                      at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("xla::");
  return aten_autograd_ops::MaxPool3dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  LTC_FN_COUNTER("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output, self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::max_pool_nd_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      /*spatial_dim_count=*/3, Helpers::I64List(kernel_size),
      Helpers::I64List(stride), Helpers::I64List(padding), ceil_mode));
}

std::tuple<at::Tensor, at::Tensor> max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("xla::");
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
  auto outputs = LazyTensor::max_pool_nd(
      bridge::GetLtcTensor(self), /*spatial_dim_count=*/3,
      Helpers::I64List(kernel_size), Helpers::I64List(stride),
      Helpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)));
}

at::Tensor max_unpool2d(const at::Tensor& self, const at::Tensor& indices,
                        at::IntArrayRef output_size) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor max_unpool2d_backward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 const at::Tensor& indices,
                                 at::IntArrayRef output_size) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor max_unpool3d(const at::Tensor& self, const at::Tensor& indices,
                        at::IntArrayRef output_size, at::IntArrayRef stride,
                        at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor max_unpool3d_backward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 const at::Tensor& indices,
                                 at::IntArrayRef output_size,
                                 at::IntArrayRef stride,
                                 at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::max_unpool_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(indices),
      xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor mean(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::mean(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor mean(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::mean(
      bridge::GetLtcTensor(self), xla::util::ToVector<xla::int64>(dim),
      /*keep_reduced_dimensions=*/keepdim, dtype));
}

at::Tensor min(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::min(bridge::GetLtcTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, int64_t dim,
                                       bool keepdim) {
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(min, dim)>::call(self, dim,
                                                                keepdim);
}

at::Tensor minimum(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::min(xself, xother, dtype);
                    });
}

std::tuple<at::Tensor&, at::Tensor&> min_out(const at::Tensor& self,
                                             int64_t dim, bool keepdim,
                                             at::Tensor& min,
                                             at::Tensor& min_indices) {
  LTC_FN_COUNTER("xla::");
  LazyTensor min_tensor = bridge::GetLtcTensor(min);
  LazyTensor min_indices_tensor = bridge::GetLtcTensor(min_indices);
  LazyTensor::min_out(min_tensor, min_indices_tensor,
                      bridge::GetLtcTensor(self), dim, keepdim);
  return std::forward_as_tuple(min, min_indices);
}

at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(mm)>::call(
        self, mat2);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::mm(/*input=*/bridge::GetLtcTensor(self),
                     /*weight=*/bridge::GetLtcTensor(mat2)));
}

at::Tensor mse_loss(const at::Tensor& self, const at::Tensor& target,
                    int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::mse_loss(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(target), reduction));
}

at::Tensor mse_loss_backward(const at::Tensor& grad_output,
                             const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::mse_loss_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(target), reduction));
}

at::Tensor mul(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const LazyTensor& xother,
                          at::ScalarType dtype) {
                        return LazyTensor::mul(xself, xother, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(mul, Tensor)>::call(self, other);
}

at::Tensor mul(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const at::Scalar& other,
                          at::ScalarType dtype) {
                        return LazyTensor::mul(xself, other, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(mul, Scalar)>::call(self, other);
}

at::Tensor mv(const at::Tensor& self, const at::Tensor& vec) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(vec)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(mv)>::call(
        self, vec);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::mv(bridge::GetLtcTensor(self), bridge::GetLtcTensor(vec)));
}

at::Tensor& mv_out(const at::Tensor& self, const at::Tensor& vec,
                   at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(vec)) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(mv_out)>::call(self, vec, out);
  }
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::mv_out(out_tensor, bridge::GetLtcTensor(self),
                     bridge::GetLtcTensor(vec));
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  LTC_FN_COUNTER("xla::");
  LazyTensor input_tensor = bridge::GetLtcTensor(input);
  const Device& device = input_tensor.GetDevice();
  LazyTensor running_mean_tensor =
      bridge::GetOrCreateLtcTensor(running_mean, device);
  LazyTensor running_var_tensor =
      bridge::GetOrCreateLtcTensor(running_var, device);
  auto outputs = LazyTensor::native_batch_norm(
      bridge::GetLtcTensor(input), bridge::GetOrCreateLtcTensor(weight, device),
      bridge::GetOrCreateLtcTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("xla::");
  LazyTensor grad_out_tensor = bridge::GetLtcTensor(grad_out);
  const Device& device = grad_out_tensor.GetDevice();
  auto gradients = LazyTensor::native_batch_norm_backward(
      bridge::GetLtcTensor(grad_out), bridge::GetLtcTensor(input),
      bridge::GetOrCreateLtcTensor(weight, device),
      bridge::GetOrCreateLtcTensor(save_mean, device),
      bridge::GetOrCreateLtcTensor(save_invstd, device), train, eps);
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients))
                     : undefined);
}

at::Tensor ne(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::ne(bridge::GetLtcTensor(self), other));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(ne, Scalar)>::call(self, other);
}

at::Tensor ne(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ne(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor neg(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    LTC_CHECK(self.scalar_type() != at::kBool)
        << "Negation, the `-` operator, on a bool tensor is not supported. If "
           "you are trying to invert a mask, use the `~` or `logical_not()` "
           "operator instead.";
    return bridge::AtenFromLtcTensor(
        LazyTensor::neg(bridge::GetLtcTensor(self)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(neg)>::call(
      self);
}

at::Tensor nll_loss2d_backward(const at::Tensor& grad_output,
                               const at::Tensor& self, const at::Tensor& target,
                               const c10::optional<at::Tensor>& weight,
                               int64_t reduction, int64_t ignore_index,
                               const at::Tensor& total_weight) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor weight_tensor =
      bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  LazyTensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateLtcTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(LazyTensor::nll_loss2d_backward(
      bridge::GetLtcTensor(grad_output), self_tensor,
      bridge::GetLtcTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor total_weight =
      LazyTensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromLtcTensor(LazyTensor::nll_loss2d(
          self_tensor, bridge::GetLtcTensor(target),
          bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromLtcTensor(total_weight));
}

at::Tensor nll_loss_backward(const at::Tensor& grad_output,
                             const at::Tensor& self, const at::Tensor& target,
                             const c10::optional<at::Tensor>& weight,
                             int64_t reduction, int64_t ignore_index,
                             const at::Tensor& total_weight) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor weight_tensor =
      bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice());
  LazyTensor total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateLtcTensor(total_weight, self_tensor.GetDevice());
  }
  return bridge::AtenFromLtcTensor(LazyTensor::nll_loss_backward(
      bridge::GetLtcTensor(grad_output), self_tensor,
      bridge::GetLtcTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target,
    const c10::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor total_weight =
      LazyTensor::full({}, 1, self_tensor.GetDevice(), self_tensor.dtype());
  return std::make_tuple(
      bridge::AtenFromLtcTensor(LazyTensor::nll_loss(
          self_tensor, bridge::GetLtcTensor(target),
          bridge::GetOrCreateLtcTensor(weight, self_tensor.GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromLtcTensor(total_weight));
}

at::Tensor nonzero(const at::Tensor& self) {
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(nonzero)>::call(self);
}

at::Tensor norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                at::ScalarType dtype) {
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP2(norm, ScalarOpt_dtype)>::call(self, p, dtype);
}

at::Tensor norm(const at::Tensor& self, const at::Scalar& p) {
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(norm, Scalar)>::call(self, p);
}

at::Tensor norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP2(norm, ScalarOpt_dim_dtype)>::call(self, p,
                                                                    dim,
                                                                    keepdim,
                                                                    dtype);
}

at::Tensor norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                at::IntArrayRef dim, bool keepdim) {
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP2(norm, ScalarOpt_dim)>::call(self, p, dim,
                                                              keepdim);
}

at::Tensor normal(const at::Tensor& mean, double std,
                  c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(normal, Tensor_float)>::call(mean, std,
                                                                 generator);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::normal(bridge::GetLtcTensor(mean), std));
}

at::Tensor normal(double mean, const at::Tensor& std,
                  c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(normal, float_Tensor)>::call(mean, std,
                                                                 generator);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::normal(mean, bridge::GetLtcTensor(std)));
}

at::Tensor normal(const at::Tensor& mean, const at::Tensor& std,
                  c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(normal, Tensor_Tensor)>::call(mean, std,
                                                                  generator);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::normal(
      bridge::GetLtcTensor(mean), bridge::GetLtcTensor(std)));
}

at::Tensor& normal_(at::Tensor& self, double mean, double std,
                    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(normal_)>::call(self, mean, std,
                                                                generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::normal_(self_tensor, mean, std);
  return self;
}

at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::permute(self_tensor, Helpers::I64List(dims)));
  }
  auto result =
      at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(permute)>::call(
          self, dims);
  MarkAsInteropView(result);
  return result;
}

at::Tensor pow(const at::Tensor& self, const at::Scalar& exponent) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    // xla::Pow() doesn't support integer types.
    if (!at::native::is_floating_point(self)) {
      return at::native::call_fallback_fn<
          &xla_cpu_fallback, ATEN_OP2(pow, Tensor_Scalar)>::call(self,
                                                                 exponent);
    }
    return bridge::AtenFromLtcTensor(
        LazyTensor::pow(bridge::GetLtcTensor(self), exponent));
  }
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP2(pow, Tensor_Scalar)>::call(self, exponent);
}

at::Tensor pow(const at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(pow, Tensor_Tensor)>::call(self, exponent);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::pow(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(exponent)));
}

at::Tensor pow(const at::Scalar& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!self.isFloatingPoint()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(pow, Scalar)>::call(self,
                                                                     exponent);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::pow(self, bridge::GetLtcTensor(exponent)));
}

at::Tensor prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::prod(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false,
      PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor prod(const at::Tensor& self, int64_t dim, bool keepdim,
                c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::prod(bridge::GetLtcTensor(self), {dim}, keepdim,
                       PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor& put_(at::Tensor& self, const at::Tensor& index,
                 const at::Tensor& source, bool accumulate) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::put_(self_tensor, bridge::GetLtcTensor(index),
                   bridge::GetLtcTensor(source), accumulate);
  return self;
}

std::tuple<at::Tensor, at::Tensor> qr(const at::Tensor& self, bool some) {
  LTC_FN_COUNTER("xla::");
  auto results = LazyTensor::qr(bridge::GetLtcTensor(self), some);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

// The value generated should be within (from, to].
at::Tensor& random_(at::Tensor& self, int64_t from, c10::optional<int64_t> to,
                    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(random_, from)>::call(self, from, to,
                                                          generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  at::ScalarType dtype = self_tensor.dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  int64_t to_val = (to) ? *to : GetIntegerUpperLimitForType(dtype) + inc;
  LTC_CHECK_LE(from, to_val);
  CheckRangeValues(self_tensor.dtype(), from, to_val - 1);
  LazyTensor::random_(self_tensor, from, to_val);
  return self;
}

// The value generated should be in (0, to].
at::Tensor& random_(at::Tensor& self, int64_t to,
                    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(random_, to)>::call(self, to,
                                                                     generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LTC_CHECK_GT(to, 0);
  CheckRangeValues(self_tensor.dtype(), 0, to - 1);
  LazyTensor::random_(self_tensor, 0, to);
  return self;
}

// The value generated should be in (self_type_min, self_type_max).
at::Tensor& random_(at::Tensor& self, c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(random_)>::call(self,
                                                                generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  at::ScalarType dtype = self_tensor.dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  LazyTensor::random_(self_tensor, 0, GetIntegerUpperLimitForType(dtype) + inc);
  return self;
}

at::Tensor reciprocal(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::reciprocal(bridge::GetLtcTensor(self)));
}

at::Tensor reflection_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::reflection_pad2d(
      bridge::GetLtcTensor(self), xla::util::ToVector<xla::int64>(padding)));
}

at::Tensor reflection_pad2d_backward(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::reflection_pad2d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      xla::util::ToVector<xla::int64>(padding)));
}

at::Tensor relu(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::relu(bridge::GetLtcTensor(self)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(relu)>::call(
      self);
}

at::Tensor& relu_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::relu_(self_tensor);
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(relu_)>::call(
      self);
}

at::Tensor remainder(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::remainder(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor remainder(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::remainder(bridge::GetLtcTensor(self), other));
}

at::Tensor repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(repeat)>::call(
      self, repeats);
}

at::Tensor replication_pad1d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad1d(
      bridge::GetLtcTensor(self), Helpers::I64List(padding)));
}

at::Tensor replication_pad1d_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad1d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      Helpers::I64List(padding)));
}

at::Tensor replication_pad2d(const at::Tensor& self, at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad2d(
      bridge::GetLtcTensor(self), Helpers::I64List(padding)));
}

at::Tensor replication_pad2d_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      at::IntArrayRef padding) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::replication_pad2d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      Helpers::I64List(padding)));
}

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size,
                          c10::optional<at::MemoryFormat> memory_format) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::resize_(self_tensor, Helpers::I64List(size));
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(resize_)>::call(self, size,
                                                              memory_format);
}

at::Tensor round(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::round(bridge::GetLtcTensor(self)));
}

at::Tensor rrelu_with_noise(const at::Tensor& self, const at::Tensor& noise,
                            const at::Scalar& lower, const at::Scalar& upper,
                            bool training,
                            c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    LTC_CHECK_EQ(training, false);
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(rrelu_with_noise)>::call(self, noise, lower,
                                                            upper, training,
                                                            generator);
  }
  LazyTensor noise_tensor = bridge::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise(
      bridge::GetLtcTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor rrelu_with_noise_backward(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& noise,
                                     const at::Scalar& lower,
                                     const at::Scalar& upper, bool training,
                                     bool self_is_result) {
  LTC_FN_COUNTER("xla::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  LTC_CHECK(!self_is_result || negative_slope > 0.0);
  LazyTensor noise_tensor = bridge::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor rsqrt(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::rsqrt(bridge::GetLtcTensor(self)));
}

at::Tensor rsub(const at::Tensor& self, const at::Tensor& other,
                const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const LazyTensor& xother,
                          at::ScalarType dtype) {
                        return LazyTensor::rsub(xself, xother, alpha, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(rsub, Tensor)>::call(self, other,
                                                                    alpha);
}

at::Tensor rsub(const at::Tensor& self, const at::Scalar& other,
                const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
    return bridge::AtenFromLtcTensor(
        LazyTensor::rsub(bridge::GetLtcTensor(self), other, alpha));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(rsub, Scalar)>::call(self, other,
                                                                    alpha);
}

at::Tensor& scatter_reduce_out_helper(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      const at::Tensor& src,
                                      c10::optional<c10::string_view> reduce,
                                      at::Tensor& out) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  if (!reduce.has_value()) {
    LazyTensor::scatter_out(out_tensor, self_tensor, dim,
                            bridge::GetLtcTensor(index),
                            bridge::GetLtcTensor(src));
    return out;
  } else if (*reduce == "add") {
    LazyTensor::scatter_add_out(out_tensor, self_tensor, dim,
                                bridge::GetLtcTensor(index),
                                bridge::GetLtcTensor(src));
  } else {
    // TODO: implement scatter_mul
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(scatter, reduce_out)>::call(self, dim,
                                                                index, src,
                                                                *reduce, out);
  }
  return out;
}

at::Tensor& scatter_reduce_out_helper(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      const at::Scalar& value,
                                      c10::optional<c10::string_view> reduce,
                                      at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  if (!reduce.has_value()) {
    LazyTensor::scatter_out(out_tensor, self_tensor, dim,
                            bridge::GetLtcTensor(index), value);
    return out;
  } else if (*reduce == "add") {
    LazyTensor::scatter_add_out(out_tensor, self_tensor, dim,
                                bridge::GetLtcTensor(index), value);
  } else {
    // TODO: implement scatter_mul
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP2(scatter, value_reduce_out)>::call(self, dim,
                                                                      index,
                                                                      value,
                                                                      *reduce,
                                                                      out);
  }
  return out;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Tensor& src,
                        at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  return scatter_reduce_out_helper(self, dim, index, src, c10::nullopt, out);
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Scalar& value,
                        at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  return scatter_reduce_out_helper(self, dim, index, value, c10::nullopt, out);
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Tensor& src,
                        c10::string_view reduce, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  return scatter_reduce_out_helper(self, dim, index, src, reduce, out);
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim,
                        const at::Tensor& index, const at::Scalar& value,
                        c10::string_view reduce, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  return scatter_reduce_out_helper(self, dim, index, value, reduce, out);
}

at::Tensor& scatter_add_out(const at::Tensor& self, int64_t dim,
                            const at::Tensor& index, const at::Tensor& src,
                            at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  return scatter_reduce_out_helper(self, dim, index, src, "add", out);
}

at::Tensor& scatter_add_(at::Tensor& self, int64_t dim, const at::Tensor& index,
                         const at::Tensor& src) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::scatter_add_(self_tensor, dim, bridge::GetLtcTensor(index),
                           bridge::GetLtcTensor(src));
  return self;
}

at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::select(bridge::GetLtcTensor(self), dim, index));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(select, int)>::call(self, dim,
                                                                   index);
}

at::Tensor& silu_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::silu_out(self_tensor, out_tensor);
  return out;
}

at::Tensor sigmoid(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::sigmoid(bridge::GetLtcTensor(self)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(sigmoid)>::call(self);
}

at::Tensor sigmoid_backward(const at::Tensor& grad_output,
                            const at::Tensor& output) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::sigmoid_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output)));
  }
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP(sigmoid_backward)>::call(grad_output, output);
}

at::Tensor sign(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::sign(bridge::GetLtcTensor(self)));
}

at::Tensor sin(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::sin(bridge::GetLtcTensor(self)));
}

at::Tensor sinh(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::sinh(bridge::GetLtcTensor(self)));
}

at::Tensor slice(const at::Tensor& self, int64_t dim,
                 c10::optional<int64_t> start, c10::optional<int64_t> end,
                 int64_t step) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    int64_t start_val = start.has_value() ? start.value() : 0;
    int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::slice(
        bridge::GetLtcTensor(self), dim, start_val, end_val, step));
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::slice", 1);
  LTC_VLOG(3) << "XLA slice :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  auto x_result = at::slice(xlatens[0], dim, start, end, step);
  auto result = bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(self));
  MarkAsInteropView(result);
  return result;
}

at::Tensor smooth_l1_loss(const at::Tensor& self, const at::Tensor& target,
                          int64_t reduction, double beta) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::smooth_l1_loss(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(target), reduction,
      beta));
}

at::Tensor smooth_l1_loss_backward(const at::Tensor& grad_output,
                                   const at::Tensor& self,
                                   const at::Tensor& target, int64_t reduction,
                                   double beta) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::smooth_l1_loss_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(target), reduction, beta));
}

at::Tensor softplus(const at::Tensor& self, const at::Scalar& beta,
                    const at::Scalar& threshold) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softplus(bridge::GetLtcTensor(self), beta, threshold));
}

at::Tensor softplus_backward(const at::Tensor& grad_output,
                             const at::Tensor& self, const at::Scalar& beta,
                             const at::Scalar& threshold,
                             const at::Tensor& output) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::softplus_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self), beta,
      threshold, bridge::GetLtcTensor(output)));
}

at::Tensor softshrink(const at::Tensor& self, const at::Scalar& lambda) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softshrink(bridge::GetLtcTensor(self), lambda));
}

at::Tensor softshrink_backward(const at::Tensor& grad_out,
                               const at::Tensor& self,
                               const at::Scalar& lambda) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::softshrink_backward(
      bridge::GetLtcTensor(grad_out), bridge::GetLtcTensor(self), lambda));
}

std::tuple<at::Tensor, at::Tensor> sort(const at::Tensor& self, int64_t dim,
                                        bool descending) {
  LTC_FN_COUNTER("xla::");
  auto results = LazyTensor::topk(bridge::GetLtcTensor(self), self.size(dim),
                                  dim, descending, true);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

std::vector<at::Tensor> split(const at::Tensor& self, int64_t split_size,
                              int64_t dim) {
  LTC_FN_COUNTER("xla::");
  auto xla_tensors =
      LazyTensor::split(bridge::GetLtcTensor(self), split_size, dim);
  return bridge::AtenFromLtcTensors(xla_tensors);
}

std::vector<at::Tensor> split_with_sizes(const at::Tensor& self,
                                         at::IntArrayRef split_sizes,
                                         int64_t dim) {
  LTC_FN_COUNTER("xla::");
  auto xla_tensors = LazyTensor::split_with_sizes(
      bridge::GetLtcTensor(self), Helpers::I64List(split_sizes), dim);
  return bridge::AtenFromLtcTensors(xla_tensors);
}

at::Tensor sqrt(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::sqrt(bridge::GetLtcTensor(self)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(sqrt)>::call(
      self);
}

at::Tensor squeeze(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::squeeze(bridge::GetLtcTensor(self)));
}

at::Tensor squeeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::squeeze(bridge::GetLtcTensor(self), dim));
}

at::Tensor& squeeze_(at::Tensor& self) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  LTC_VLOG(3) << "XLA squeeze_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_();
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens,
                                 xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor& squeeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  LTC_VLOG(3) << "XLA squeeze_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_(dim);
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens,
                                 xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor stack(at::TensorList tensors, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::stack(bridge::GetLtcTensors(tensors), dim));
}

at::Tensor std(const at::Tensor& self, bool unbiased) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, /*correction=*/unbiased ? 1 : 0));
}

at::Tensor std(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
               bool keepdim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      bridge::GetLtcTensor(self), xla::util::ToVector<xla::int64>(dim), keepdim,
      /*correction=*/unbiased ? 1 : 0));
}

at::Tensor std(const at::Tensor& self, c10::optional<at::IntArrayRef> dim,
               c10::optional<int64_t> correction, bool keepdim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::std(
      self_tensor,
      dim ? xla::util::ToVector<xla::int64>(*dim)
          : xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      keepdim, correction ? *correction : 1));
}

std::tuple<at::Tensor, at::Tensor> std_mean(const at::Tensor& self,
                                            c10::optional<at::IntArrayRef> dim,
                                            c10::optional<int64_t> correction,
                                            bool keepdim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto results = LazyTensor::std_mean(
      self_tensor,
      dim ? xla::util::ToVector<xla::int64>(*dim)
          : xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      correction ? *correction : 1, keepdim);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor sub(const at::Tensor& self, const at::Tensor& other,
               const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
    at::native::alpha_check(at::result_type(self, other), alpha);
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const LazyTensor& xother,
                          at::ScalarType dtype) {
                        return LazyTensor::sub(xself, xother, alpha, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(sub, Tensor)>::call(self, other,
                                                                   alpha);
}

at::Tensor sub(const at::Tensor& self, const at::Scalar& other,
               const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const at::Scalar& other,
                          at::ScalarType dtype) {
                        return LazyTensor::sub(xself, other, alpha, dtype);
                      });
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP2(sub, Scalar)>::call(self, other,
                                                                   alpha);
}

at::Tensor sum(const at::Tensor& self, c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::sum(
      self_tensor,
      xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor sum(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
               c10::optional<at::ScalarType> dtype) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::sum(bridge::GetLtcTensor(self),
                      xla::util::ToVector<xla::int64>(dim), keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd(const at::Tensor& self,
                                                   bool some, bool compute_uv) {
  LTC_FN_COUNTER("xla::");
  auto results = LazyTensor::svd(bridge::GetLtcTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)),
                         bridge::AtenFromLtcTensor(std::get<2>(results)));
}

std::tuple<at::Tensor, at::Tensor> symeig(const at::Tensor& self,
                                          bool eigenvectors, bool upper) {
  LTC_FN_COUNTER("xla::");
  auto results =
      LazyTensor::symeig(bridge::GetLtcTensor(self), eigenvectors, upper);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor t(const at::Tensor& self) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::transpose(bridge::GetLtcTensor(self), 0, 1));
  }
  auto result =
      at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(t)>::call(self);
  MarkAsInteropView(result);
  return result;
}

at::Tensor& t_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::transpose_(self_tensor, 0, 1);
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(t_)>::call(
      self);
}

at::Tensor take(const at::Tensor& self, const at::Tensor& index) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::take(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(index)));
}

at::Tensor tan(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::tan(bridge::GetLtcTensor(self)));
}

at::Tensor tanh(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::tanh(bridge::GetLtcTensor(self)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(tanh)>::call(
      self);
}

at::Tensor tanh_backward(const at::Tensor& grad_output,
                         const at::Tensor& output) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::tanh_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output)));
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback,
                                      ATEN_OP(tanh_backward)>::call(grad_output,
                                                                    output);
}

at::Tensor threshold(const at::Tensor& self, const at::Scalar& threshold,
                     const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::threshold(
      bridge::GetLtcTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor threshold_backward(const at::Tensor& grad_output,
                              const at::Tensor& self,
                              const at::Scalar& threshold) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::threshold_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
        threshold.to<double>()));
  }
  return at::native::call_fallback_fn<
      &xla_cpu_fallback, ATEN_OP(threshold_backward)>::call(grad_output, self,
                                                            threshold);
}

std::tuple<at::Tensor, at::Tensor> topk(const at::Tensor& self, int64_t k,
                                        int64_t dim, bool largest,
                                        bool sorted) {
  LTC_FN_COUNTER("xla::");
  auto results =
      LazyTensor::topk(bridge::GetLtcTensor(self), k, dim, largest, sorted);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor trace(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::trace(bridge::GetLtcTensor(self)));
}

at::Tensor transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::transpose(bridge::GetLtcTensor(self), dim0, dim1));
  }
  auto result =
      at::native::call_fallback_fn<&xla_cpu_fallback,
                                   ATEN_OP2(transpose, int)>::call(self, dim0,
                                                                   dim1);
  MarkAsInteropView(result);
  return result;
}

at::Tensor& transpose_(at::Tensor& self, int64_t dim0, int64_t dim1) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

std::tuple<at::Tensor, at::Tensor> triangular_solve(const at::Tensor& b,
                                                    const at::Tensor& A,
                                                    bool upper, bool transpose,
                                                    bool unitriangular) {
  LTC_FN_COUNTER("xla::");
  // Currently, ATen doesn't have a left_side option. Once this
  // is added, this API will have to be changed.
  auto results = LazyTensor::triangular_solve(
      bridge::GetLtcTensor(b), bridge::GetLtcTensor(A), /*left_side=*/true,
      upper, transpose, unitriangular);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor tril(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::tril(bridge::GetLtcTensor(self), diagonal));
}

at::Tensor& tril_(at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::tril_(self_tensor, diagonal);
  return self;
}

at::Tensor triu(const at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::triu(bridge::GetLtcTensor(self), diagonal));
}

at::Tensor& triu_(at::Tensor& self, int64_t diagonal) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::triu_(self_tensor, diagonal);
  return self;
}

at::Tensor trunc(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::trunc(bridge::GetLtcTensor(self)));
}

std::vector<at::Tensor> unbind(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensors(
      LazyTensor::unbind(bridge::GetLtcTensor(self), dim));
}

at::Tensor& uniform_(at::Tensor& self, double from, double to,
                     c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP(uniform_)>::call(self, from, to,
                                                                 generator);
  }
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::uniform_(self_tensor, from, to);
  return self;
}

at::Tensor unsqueeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::unsqueeze(bridge::GetLtcTensor(self), dim));
}

at::Tensor& unsqueeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor upsample_bilinear2d(const at::Tensor& self,
                               at::IntArrayRef output_size, bool align_corners,
                               c10::optional<double> scales_h,
                               c10::optional<double> scales_w) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(upsample_bilinear2d)>::call(self,
                                                               output_size,
                                                               align_corners,
                                                               scales_h,
                                                               scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_bilinear2d(
      self_tensor, xla::util::ToVector<xla::int64>(output_size),
      align_corners));
}

at::Tensor upsample_bilinear2d_backward(const at::Tensor& grad_output,
                                        at::IntArrayRef output_size,
                                        at::IntArrayRef input_size,
                                        bool align_corners,
                                        c10::optional<double> scales_h,
                                        c10::optional<double> scales_w) {
  LTC_FN_COUNTER("xla::");
  LazyTensor grad_output_tensor = bridge::GetLtcTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(upsample_bilinear2d_backward)>::call(grad_output, output_size,
                                                     input_size, align_corners,
                                                     scales_h, scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_bilinear2d_backward(
      grad_output_tensor, xla::util::ToVector<xla::int64>(output_size),
      xla::util::ToVector<xla::int64>(input_size), align_corners));
}

at::Tensor upsample_nearest2d(
    const at::Tensor& input, c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_COUNTER("xla::");
  LazyTensor input_tensor = bridge::GetLtcTensor(input);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (input_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(upsample_nearest2d,
                                                 vec)>::call(input, output_size,
                                                             scale_factors);
  }
  absl::Span<const xla::int64> input_dims =
      input_tensor.shape().get().dimensions();
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d(
      input_tensor,
      GetOutputSizeWithScale(input_dims, scale_factors, output_size)));
}

at::Tensor upsample_nearest2d_backward(
    const at::Tensor& grad_output, c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  LTC_FN_COUNTER("xla::");
  LazyTensor grad_output_tensor = bridge::GetLtcTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by our
  // XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU) {
    return at::native::call_fallback_fn<&xla_cpu_fallback,
                                        ATEN_OP2(upsample_nearest2d_backward,
                                                 vec)>::call(grad_output,
                                                             output_size,
                                                             input_size,
                                                             scale_factors);
  }
  std::vector<xla::int64> input_dim =
      xla::util::ToVector<xla::int64>(input_size);
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d_backward(
      grad_output_tensor,
      GetOutputSizeWithScale(input_dim, scale_factors, output_size),
      input_dim));
}

at::Tensor upsample_nearest2d(const at::Tensor& self,
                              at::IntArrayRef output_size,
                              c10::optional<double> scales_h,
                              c10::optional<double> scales_w) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (self_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback, ATEN_OP(upsample_nearest2d)>::call(self, output_size,
                                                              scales_h,
                                                              scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d(
      self_tensor, xla::util::ToVector<xla::int64>(output_size)));
}

at::Tensor upsample_nearest2d_backward(const at::Tensor& grad_output,
                                       at::IntArrayRef output_size,
                                       at::IntArrayRef input_size,
                                       c10::optional<double> scales_h,
                                       c10::optional<double> scales_w) {
  LTC_FN_COUNTER("xla::");
  LazyTensor grad_output_tensor = bridge::GetLtcTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  if (grad_output_tensor.GetDevice().hw_type != DeviceType::TPU ||
      (scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    return at::native::call_fallback_fn<
        &xla_cpu_fallback,
        ATEN_OP(upsample_nearest2d_backward)>::call(grad_output, output_size,
                                                    input_size, scales_h,
                                                    scales_w);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::upsample_nearest2d_backward(
      grad_output_tensor, xla::util::ToVector<xla::int64>(output_size),
      xla::util::ToVector<xla::int64>(input_size)));
}

at::Tensor var(const at::Tensor& self, bool unbiased) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::var(bridge::GetLtcTensor(self),
                      xla::util::Iota<xla::int64>(
                          bridge::GetLtcTensor(self).shape().get().rank()),
                      /*correction=*/unbiased ? 1 : 0,
                      /*keep_reduced_dimensions=*/false));
}

at::Tensor var(const at::Tensor& self, at::IntArrayRef dim, bool unbiased,
               bool keepdim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::var(self_tensor, Helpers::I64List(dim),
                      /*correction=*/unbiased ? 1 : 0, keepdim));
}

at::Tensor var(const at::Tensor& self, c10::optional<at::IntArrayRef> dim,
               c10::optional<int64_t> correction, bool keepdim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::var(
      self_tensor,
      dim ? Helpers::I64List(*dim)
          : xla::util::Iota<xla::int64>(
                bridge::GetLtcTensor(self).shape().get().rank()),
      correction ? *correction : 1, keepdim));
}

std::tuple<at::Tensor, at::Tensor> var_mean(const at::Tensor& self,
                                            c10::optional<at::IntArrayRef> dim,
                                            c10::optional<int64_t> correction,
                                            bool keepdim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto results = LazyTensor::var_mean(
      self_tensor,
      dim ? xla::util::ToVector<xla::int64>(*dim)
          : xla::util::Iota<xla::int64>(self_tensor.shape().get().rank()),
      correction ? *correction : 1, keepdim);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(results)),
                         bridge::AtenFromLtcTensor(std::get<1>(results)));
}

at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::view(self_tensor, Helpers::I64List(size)));
  }
  auto result =
      at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(view)>::call(
          self, size);
  MarkAsInteropView(result);
  return result;
}

at::Tensor& zero_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::zero_(self_tensor);
    return self;
  }
  return at::native::call_fallback_fn<&xla_cpu_fallback, ATEN_OP(zero_)>::call(
      self);
}

}  // namespace torch_lazy_tensors
