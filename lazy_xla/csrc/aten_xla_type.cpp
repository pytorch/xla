#include "lazy_xla/csrc/aten_xla_type.h"

#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>

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
#include "lazy_xla/csrc/aten_autograd_ops.h"
#include "lazy_xla/csrc/aten_autograd_ops_nnc.h"
#include "lazy_xla/csrc/aten_xla_type_default.h"
#include "lazy_xla/csrc/compiler/nnc_computation_client.h"
#include "lazy_xla/csrc/compiler/pooling.h"
#include "lazy_xla/csrc/version.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't exist in AtenXlaType,
//   call at::native::func instead.
//   E.g. don't call tensor.is_floating_point() or
//   at::is_floating_point(tensor), use at::native::is_floating_point(tensor).

namespace torch_lazy_tensors {
namespace {

Device GetLtcDeviceOrCurrent(const c10::optional<c10::Device>& device) {
  auto xla_device_opt = bridge::GetLtcDevice(device);
  return xla_device_opt ? *xla_device_opt : GetCurrentDevice();
}

at::ScalarType GetScalarTypeOrFloat(c10::optional<at::ScalarType> scalar_type) {
  return scalar_type ? *scalar_type : at::ScalarType::Float;
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

void AtenInitialize() {
  LTC_VLOG(1) << "PyTorch GIT revision: " << lazy_xla::TORCH_GITREV;
  LTC_VLOG(1) << "XLA GIT revision: " << lazy_xla::XLA_GITREV;

  LTCTensorImpl::AtenInitialize();
}

at::Tensor subtensor(const at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  return tensor.narrow(dim, n * g, n).contiguous();
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

at::Tensor& AtenXlaType::__ilshift__(at::Tensor& self,
                                     const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__ilshift__(at::Tensor& self,
                                     const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__ilshift__(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor& AtenXlaType::__irshift__(at::Tensor& self,
                                     const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::__irshift__(at::Tensor& self,
                                     const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::__irshift__(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor AtenXlaType::__lshift__(const at::Tensor& self,
                                   const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, other, dtype);
                    });
}

at::Tensor AtenXlaType::__lshift__(const at::Tensor& self,
                                   const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::__rshift__(const at::Tensor& self,
                                   const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, other, dtype);
                    });
}

at::Tensor AtenXlaType::__rshift__(const at::Tensor& self,
                                   const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::_adaptive_avg_pool3d(const at::Tensor& self,
                                             at::IntArrayRef output_size) {
  LTC_FN_COUNTER("xla::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()),
                                  output_size_list, /*pool_dim=*/3)) {
    return AtenXlaTypeDefault::_adaptive_avg_pool3d(self, output_size);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::adaptive_avg_pool3d(
      bridge::GetLtcTensor(self), output_size_list));
}

at::Tensor AtenXlaType::_adaptive_avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<xla::int64> output_size{grad_output.size(rank - 3),
                                      grad_output.size(rank - 2),
                                      grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/3)) {
    return AtenXlaTypeDefault::_adaptive_avg_pool3d_backward(grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::adaptive_avg_pool3d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::_adaptive_avg_pool2d(const at::Tensor& self,
                                             at::IntArrayRef output_size) {
  LTC_FN_COUNTER("xla::");
  auto output_size_list = Helpers::I64List(output_size);
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()),
                                  output_size_list, /*pool_dim=*/2)) {
    return AtenXlaTypeDefault::_adaptive_avg_pool2d(self, output_size);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::_adaptive_avg_pool2d(
      bridge::GetLtcTensor(self), output_size_list));
}

at::Tensor AtenXlaType::_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  int64_t rank = grad_output.dim();
  std::vector<xla::int64> output_size{grad_output.size(rank - 2),
                                      grad_output.size(rank - 1)};
  if (!IsSupportedAdaptiveAvgPool(Helpers::I64List(self.sizes()), output_size,
                                  /*pool_dim=*/2)) {
    return AtenXlaTypeDefault::_adaptive_avg_pool2d_backward(grad_output, self);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::_adaptive_avg_pool2d_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self)));
}

void AtenXlaType::_amp_foreach_non_finite_check_and_unscale_(
    at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale) {
  LTC_FN_COUNTER("xla::");
  LazyTensor found_inf_tensor = bridge::GetLtcTensor(found_inf);
  LazyTensor::_amp_foreach_non_finite_check_and_unscale_(
      bridge::GetLtcTensors(self), found_inf_tensor,
      bridge::GetLtcTensor(inv_scale));
}

at::Tensor AtenXlaType::_amp_update_scale(at::Tensor& growth_tracker,
                                          const at::Tensor& current_scale,
                                          const at::Tensor& found_inf,
                                          double scale_growth_factor,
                                          double scale_backoff_factor,
                                          int64_t growth_interval) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::_amp_update_scale(
      bridge::GetLtcTensor(growth_tracker), bridge::GetLtcTensor(current_scale),
      bridge::GetLtcTensor(found_inf), scale_growth_factor,
      scale_backoff_factor, growth_interval));
}

at::Tensor AtenXlaType::_copy_from(const at::Tensor& self,
                                   const at::Tensor& dst, bool non_blocking) {
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
    if (!dst_tensor->CurrentIrValue()) {
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      LTC_CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/true));
      }
    } else {
      LazyTensor::copy_(*dst_tensor, *self_tensor);
      bridge::ReplaceLtcTensor(dst, *dst_tensor);
    }
  }
  return dst;
}

at::Tensor AtenXlaType::_s_where(const at::Tensor& condition,
                                 const at::Tensor& self,
                                 const at::Tensor& other) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::where(
        bridge::GetLtcTensor(condition), bridge::GetLtcTensor(self),
        bridge::GetLtcTensor(other)));
  }
  return AtenXlaTypeDefault::_s_where(condition, self, other);
}

at::Tensor AtenXlaType::_trilinear(const at::Tensor& i1, const at::Tensor& i2,
                                   const at::Tensor& i3,
                                   at::IntArrayRef expand1,
                                   at::IntArrayRef expand2,
                                   at::IntArrayRef expand3,
                                   at::IntArrayRef sumdim, int64_t unroll_dim) {
  return AtenXlaTypeDefault::_trilinear(i1, i2, i3, expand1, expand2, expand3,
                                        sumdim, unroll_dim);
}

at::Tensor AtenXlaType::_unsafe_view(const at::Tensor& self,
                                     at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return view(self, size);
  }
  auto result = AtenXlaTypeDefault::_unsafe_view(self, size);
  MarkAsInteropView(result);
  return result;
}

at::Tensor AtenXlaType::abs(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::abs(bridge::GetLtcTensor(self)));
  }
  return AtenXlaTypeDefault::abs(self);
}

at::Tensor& AtenXlaType::abs_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::abs_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::abs_(self);
}

at::Tensor AtenXlaType::acos(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::acos(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::acos_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::acos_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::acos_(self);
}

at::Tensor AtenXlaType::acosh(const at::Tensor& self) {
  return AtenXlaTypeDefault::acosh(self);
}

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Tensor& other,
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
  return AtenXlaTypeDefault::add(self, other, alpha);
}

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Scalar& other,
                            const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const at::Scalar& other,
                          at::ScalarType dtype) {
                        return LazyTensor::add(xself, other, alpha, dtype);
                      });
  }
  return AtenXlaTypeDefault::add(self, other, alpha);
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, const at::Tensor& other,
                              const at::Scalar& alpha) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    at::native::alpha_check(at::result_type(self, other), alpha);
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::add_(
        self_tensor,
        bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()), alpha);
    return self;
  }
  return AtenXlaTypeDefault::add_(self, other, alpha);
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, const at::Scalar& other,
                              const at::Scalar& alpha) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::add_(self_tensor, other, alpha);
    return self;
  }
  return AtenXlaTypeDefault::add_(self, other, alpha);
}

at::Tensor AtenXlaType::addcdiv(const at::Tensor& self,
                                const at::Tensor& tensor1,
                                const at::Tensor& tensor2,
                                const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcdiv(
      bridge::GetLtcTensor(self), value, bridge::GetLtcTensor(tensor1),
      bridge::GetLtcTensor(tensor2)));
}

at::Tensor& AtenXlaType::addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2,
                                  const at::Scalar& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::addcdiv_(self_tensor, value, bridge::GetLtcTensor(tensor1),
                         bridge::GetLtcTensor(tensor2));
    return self;
  }
  return AtenXlaTypeDefault::addcdiv_(self, tensor1, tensor2, value);
}

at::Tensor AtenXlaType::addcmul(const at::Tensor& self,
                                const at::Tensor& tensor1,
                                const at::Tensor& tensor2,
                                const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::addcmul(
      bridge::GetLtcTensor(self), value, bridge::GetLtcTensor(tensor1),
      bridge::GetLtcTensor(tensor2)));
}

at::Tensor& AtenXlaType::addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                                  const at::Tensor& tensor2,
                                  const at::Scalar& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::addcmul_(self_tensor, value, bridge::GetLtcTensor(tensor1),
                         bridge::GetLtcTensor(tensor2));
    return self;
  }
  return AtenXlaTypeDefault::addcmul_(self, tensor1, tensor2, value);
}

at::Tensor AtenXlaType::alias(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return self;
}

at::Tensor AtenXlaType::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                   at::IntArrayRef stride,
                                   c10::optional<int64_t> storage_offset) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  const auto device_data =
      ir::ops::DeviceData::Cast(self_tensor.GetIrValue().node.get());
  if (!UseNNCViews(self_tensor) &&
      (device_data || self_tensor.CurrentTensorData())) {
    auto result =
        AtenXlaTypeDefault::as_strided(self, size, stride, storage_offset);
    return result;
  }
  LTC_FN_COUNTER("xla::");
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided(self, size, stride, storage_offset);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::as_strided(self_tensor, std::move(xsize), std::move(xstride),
                             Helpers::I64Optional(storage_offset)));
}

at::Tensor& AtenXlaType::as_strided_(at::Tensor& self, at::IntArrayRef size,
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

at::Tensor AtenXlaType::asin(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::asin(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::asin_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::asin_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::asin_(self);
}

at::Tensor AtenXlaType::asinh(const at::Tensor& self) {
  return AtenXlaTypeDefault::asinh(self);
}

at::Tensor AtenXlaType::atan(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::atan(bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::atanh(const at::Tensor& self) {
  return AtenXlaTypeDefault::atanh(self);
}

at::Tensor AtenXlaType::atan2(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  // xla::Atan2 doesn't support integer types.
  if (!self.is_floating_point() || !other.is_floating_point()) {
    return AtenXlaTypeDefault::atan2(self, other);
  }
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::atan2(xself, xother, dtype);
                    });
}

at::Tensor& AtenXlaType::atan2_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  // xla::Atan2 doesn't support integer types.
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC &&
      self.is_floating_point() && other.is_floating_point()) {
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::atan2_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::atan2_(self, other);
}

at::Tensor& AtenXlaType::atan_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::atan_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::atan_(self);
}

at::Tensor& AtenXlaType::bitwise_and_out(const at::Tensor& self,
                                         const at::Scalar& other,
                                         at::Tensor& out) {
  if (UseNNC(out)) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(out, self, other);
    LazyTensor out_tensor = bridge::GetLtcTensor(out);
    LazyTensor::bitwise_and_out(out_tensor, bridge::GetLtcTensor(self), other);
    return out;
  }
  return AtenXlaTypeDefault::bitwise_and_out(self, other, out);
}

at::Tensor& AtenXlaType::bitwise_and_out(const at::Tensor& self,
                                         const at::Tensor& other,
                                         at::Tensor& out) {
  if (UseNNC(out)) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(out, self, other);
    LazyTensor out_tensor = bridge::GetLtcTensor(out);
    LazyTensor::bitwise_and_out(out_tensor, bridge::GetLtcTensor(self),
                                bridge::GetLtcTensor(other));
    return out;
  }
  return AtenXlaTypeDefault::bitwise_and_out(self, other, out);
}

at::Tensor& AtenXlaType::bitwise_or_out(const at::Tensor& self,
                                        const at::Scalar& other,
                                        at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::GetLtcTensor(self), other);
  return out;
}

at::Tensor& AtenXlaType::bitwise_or_out(const at::Tensor& self,
                                        const at::Tensor& other,
                                        at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_or_out(out_tensor, bridge::GetLtcTensor(self),
                             bridge::GetLtcTensor(other));
  return out;
}

at::Tensor& AtenXlaType::bitwise_xor_out(const at::Tensor& self,
                                         const at::Scalar& other,
                                         at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::GetLtcTensor(self), other);
  return out;
}

at::Tensor& AtenXlaType::bitwise_xor_out(const at::Tensor& self,
                                         const at::Tensor& other,
                                         at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(out, self, other);
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor::bitwise_xor_out(out_tensor, bridge::GetLtcTensor(self),
                              bridge::GetLtcTensor(other));
  return out;
}

at::Tensor AtenXlaType::ceil(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ceil(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::ceil_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::ceil_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::ceil_(self);
}

at::Tensor AtenXlaType::clamp(const at::Tensor& self,
                              const c10::optional<at::Scalar>& min,
                              const c10::optional<at::Scalar>& max) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min, max));
}

at::Tensor& AtenXlaType::clamp_(at::Tensor& self,
                                const c10::optional<at::Scalar>& min,
                                const c10::optional<at::Scalar>& max) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, min, max);
    return self;
  }
  return AtenXlaTypeDefault::clamp_(self, min, max);
}

at::Tensor AtenXlaType::clamp_max(const at::Tensor& self,
                                  const at::Scalar& max) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), c10::nullopt, max));
}

at::Tensor& AtenXlaType::clamp_max_(at::Tensor& self, const at::Scalar& max) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, c10::nullopt, max);
    return self;
  }
  return AtenXlaTypeDefault::clamp_max_(self, max);
}

at::Tensor AtenXlaType::clamp_min(const at::Tensor& self,
                                  const at::Scalar& min) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min, c10::nullopt));
}

at::Tensor& AtenXlaType::clamp_min_(at::Tensor& self, const at::Scalar& min) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, min, c10::nullopt);
    return self;
  }
  return AtenXlaTypeDefault::clamp_min_(self, min);
}

at::Tensor AtenXlaType::clone(const at::Tensor& self,
                              c10::optional<at::MemoryFormat> memory_format) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (ForceNNC()) {
    return bridge::AtenFromLtcTensor(LazyTensor::clone(self_tensor));
  }
  if (self_tensor.CurrentTensorData()) {
    return AtenXlaTypeDefault::clone(self, memory_format);
  }
  const auto device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  return bridge::CreateLtcTensor(
      bridge::AtenFromLtcTensor(LazyTensor::clone(self_tensor)).to(device_type),
      bridge::GetLtcDevice(self));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  if (groups > 1) {
    std::vector<at::Tensor> grad_input(groups);
    std::vector<at::Tensor> grad_weight(groups);
    std::vector<at::Tensor> grad_bias(groups);
    for (int g = 0; g < groups; ++g) {
      auto grad_output_g = subtensor(grad_output, 1, groups, g);
      auto input_g = subtensor(input, 1, groups, g);
      auto weight_g = subtensor(weight, 0, groups, g);
      auto x_result = convolution_backward_overrideable(
          grad_output_g, input_g, weight_g, stride, padding, dilation,
          transposed, output_padding, 1, output_mask);
      grad_input[g] = std::get<0>(x_result);
      grad_weight[g] = std::get<1>(x_result);
      grad_bias[g] = std::get<2>(x_result);
    }
    return {at::cat(grad_input, 1), at::cat(grad_weight, 0),
            grad_bias[0].defined() ? at::cat(grad_bias, 0) : grad_bias[0]};
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::convolution_backward_overrideable", 1);
  LTC_VLOG(3) << "XLA convolution_backward_overrideable :"
              << " grad_output=" << grad_output.toString()
              << " input=" << input.toString()
              << " weight=" << weight.toString();
  const auto kernel_size = weight.sizes().slice(2);
  LTC_CHECK(kernel_size.size() == 2 || kernel_size.size() == 3);
  const auto device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  if (transposed) {
    at::TensorOptions options = at::TensorOptions().device(device_type);
    auto&& x_result =
        kernel_size.size() == 2
            ? at::slow_conv_transpose2d_backward(
                  grad_output.to(device_type), input.to(device_type),
                  weight.to(device_type), kernel_size, stride, padding,
                  output_padding, dilation,
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Contiguous),
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Contiguous),
                  output_mask)
            : at::slow_conv_transpose3d_backward(
                  grad_output.to(device_type), input.to(device_type),
                  weight.to(device_type), kernel_size, stride, padding,
                  output_padding, dilation,
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Preserve),
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Preserve),
                  output_mask);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        bridge::CreateLtcTensor(std::get<0>(x_result),
                                bridge::GetLtcDevice(grad_output)),
        bridge::CreateLtcTensor(std::get<1>(x_result),
                                bridge::GetLtcDevice(grad_output)),
        bridge::CreateLtcTensor(std::get<2>(x_result),
                                bridge::GetLtcDevice(grad_output)));
  }
  auto&& x_result =
      kernel_size.size() == 2
          ? at::slow_conv_dilated2d_backward(
                grad_output.to(device_type), input.to(device_type),
                weight.to(device_type), kernel_size, stride, padding, dilation,
                output_mask)
          : at::slow_conv_dilated3d_backward(
                grad_output.to(device_type), input.to(device_type),
                weight.to(device_type), kernel_size, stride, padding, dilation,
                output_mask);
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      bridge::CreateLtcTensor(std::get<0>(x_result),
                              bridge::GetLtcDevice(grad_output)),
      bridge::CreateLtcTensor(std::get<1>(x_result),
                              bridge::GetLtcDevice(grad_output)),
      bridge::CreateLtcTensor(std::get<2>(x_result),
                              bridge::GetLtcDevice(grad_output)));
}

at::Tensor AtenXlaType::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  if (groups != 1) {
    std::vector<at::Tensor> outputs(groups);
    for (int g = 0; g < groups; ++g) {
      auto input_g = subtensor(input, 1, groups, g);
      auto weight_g = subtensor(weight, 0, groups, g);
      auto bias_g = bias ? subtensor(*bias, 0, groups, g) : bias;
      outputs[g] =
          convolution_overrideable(input_g, weight_g, bias_g, stride, padding,
                                   dilation, transposed, output_padding, 1);
    }
    return at::cat(outputs, 1);
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::convolution_overrideable", 1);
  LTC_VLOG(3) << "XLA convolution_overrideable :"
              << " input=" << input.toString()
              << " weight=" << weight.toString();
  std::vector<at::Tensor> xlatens_tensors = {input, weight};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  std::vector<c10::optional<at::Tensor>> xlatens_opt_tensors = {bias};
  auto xlatens_opt = bridge::LtcCreateOptTensorList(xlatens_opt_tensors);
  const auto kernel_size = weight.sizes().slice(2);
  LTC_CHECK(kernel_size.size() == 2 || kernel_size.size() == 3);
  const auto device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  if (transposed) {
    auto&& x_result =
        kernel_size.size() == 2
            ? at::slow_conv_transpose2d(
                  input.to(device_type), weight.to(device_type), kernel_size,
                  (bias && bias->defined()) ? bias->to(device_type) : bias,
                  stride, padding, output_padding, dilation)
            : at::slow_conv_transpose3d(
                  input.to(device_type), weight.to(device_type), kernel_size,
                  (bias && bias->defined()) ? bias->to(device_type) : bias,
                  stride, padding, output_padding, dilation);
    return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(input));
  }
  auto&& x_result =
      kernel_size.size() == 2
          ? at::slow_conv_dilated2d(
                input.to(device_type), weight.to(device_type), kernel_size,
                (bias && bias->defined()) ? bias->to(device_type) : bias,
                stride, padding, dilation)
          : at::slow_conv_dilated3d(
                input.to(device_type), weight.to(device_type), kernel_size,
                (bias && bias->defined()) ? bias->to(device_type) : bias,
                stride, padding, dilation);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(input));
}

at::Tensor AtenXlaType::cos(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::cos(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::cos_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::cos_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::cos_(self);
}

at::Tensor AtenXlaType::cosh(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::cosh(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::cosh_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::cosh_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::cosh_(self);
}

at::Tensor AtenXlaType::diagonal(const at::Tensor& self, int64_t offset,
                                 int64_t dim1, int64_t dim2) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::diagonal(bridge::GetLtcTensor(self), offset, dim1, dim2));
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Tensor& other) {
  return div(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Tensor& other,
                            c10::optional<std::string> rounding_mode) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    at::ScalarType dtype = at::result_type(self, other);
    auto operands = GetBinaryOperands(self, other);
    return bridge::AtenFromLtcTensor(
        LazyTensor::div(operands.first, operands.second, rounding_mode, dtype));
  }
  return AtenXlaTypeDefault::div(self, other, rounding_mode);
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::div(bridge::GetLtcTensor(self), other));
  }
  return AtenXlaTypeDefault::div(self, other);
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other) {
  return div_(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other,
                              c10::optional<std::string> rounding_mode) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::div_(
        self_tensor,
        bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()),
        rounding_mode);
    return self;
  }
  return AtenXlaTypeDefault::div_(self, other, rounding_mode);
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::div_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::div_(self, other);
}

at::Tensor AtenXlaType::elu(const at::Tensor& self, const at::Scalar& alpha,
                            const at::Scalar& scale,
                            const at::Scalar& input_scale) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::elu(bridge::GetLtcTensor(self), alpha, scale, input_scale));
}

at::Tensor& AtenXlaType::elu_(at::Tensor& self, const at::Scalar& alpha,
                              const at::Scalar& scale,
                              const at::Scalar& input_scale) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::elu_(self_tensor, alpha, scale, input_scale);
    return self;
  }
  return AtenXlaTypeDefault::elu_(self, alpha, scale, input_scale);
}

at::Tensor AtenXlaType::elu_backward(const at::Tensor& grad_output,
                                     const at::Scalar& alpha,
                                     const at::Scalar& scale,
                                     const at::Scalar& input_scale, bool self,
                                     const at::Tensor& self_or_result) {
  LTC_FN_COUNTER("xla::");
  LTC_CHECK(!self || alpha.to<double>() >= 0.0)
      << "In-place elu backward calculation is triggered with a negative slope "
         "which is not supported.";
  return bridge::AtenFromLtcTensor(LazyTensor::elu_backward(
      bridge::GetLtcTensor(grad_output), alpha, scale, input_scale,
      bridge::GetLtcTensor(self_or_result)));
}

at::Tensor AtenXlaType::empty(at::IntArrayRef size,
                              c10::optional<at::ScalarType> dtype,
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
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(device));
}

at::Tensor AtenXlaType::empty_strided(at::IntArrayRef size,
                                      at::IntArrayRef stride,
                                      c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout,
                                      c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("xla::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::eq(bridge::GetLtcTensor(self), other));
  }
  return AtenXlaTypeDefault::eq(self, other);
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::eq(
        bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
  }
  return AtenXlaTypeDefault::eq(self, other);
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::eq_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::eq_(self, other);
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::eq_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::eq_(self, other);
}

at::Tensor AtenXlaType::erf(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::erf(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::erf_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::erf_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::erf_(self);
}

at::Tensor AtenXlaType::erfc(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::erfc(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::erfc_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::erfc_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::erfc_(self);
}

at::Tensor AtenXlaType::exp(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::exp(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::exp_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::exp_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::exp_(self);
}

at::Tensor AtenXlaType::expand(const at::Tensor& self, at::IntArrayRef size,
                               bool implicit) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::expand(bridge::GetLtcTensor(self),
                           lazy_tensors::util::ToVector<xla::int64>(size)));
  }
  return AtenXlaTypeDefault::expand(self, size, implicit);
}

at::Tensor AtenXlaType::expm1(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::expm1(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::expm1_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::expm1_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::expm1_(self);
}

at::Tensor& AtenXlaType::fill_(at::Tensor& self, const at::Scalar& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::fill_(self_tensor, value);
    return self;
  }
  return AtenXlaTypeDefault::fill_(self, value);
}

at::Tensor& AtenXlaType::fill_(at::Tensor& self, const at::Tensor& value) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LTC_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                                 << "value tensor, but got tensor "
                                 << "with " << value.dim() << " dimension(s).";
    return fill_(self, value.item());
  }
  return AtenXlaTypeDefault::fill_(self, value);
}

at::Tensor AtenXlaType::floor(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::floor(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::floor_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::floor_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::floor_(self);
}

at::Tensor AtenXlaType::fmod(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::fmod(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::fmod(xself, other, dtype);
                    });
}

at::Tensor& AtenXlaType::fmod_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::fmod_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::fmod_(self, other);
}

at::Tensor& AtenXlaType::fmod_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::fmod_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::fmod_(self, other);
}

at::Tensor AtenXlaType::frac(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::frac(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::frac_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::frac_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::frac_(self);
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ge(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ge(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::ge_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::ge_(self, other);
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::ge_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::ge_(self, other);
}

at::Tensor AtenXlaType::gelu(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::gelu(bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::gelu_backward(const at::Tensor& grad,
                                      const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::gelu_backward(
      bridge::GetLtcTensor(grad), bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::gt(bridge::GetLtcTensor(self), other));
  }
  return AtenXlaTypeDefault::gt(self, other);
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::gt(
        bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
  }
  return AtenXlaTypeDefault::gt(self, other);
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::gt_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::gt_(self, other);
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::gt_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::gt_(self, other);
}

at::Tensor AtenXlaType::hardtanh(const at::Tensor& self,
                                 const at::Scalar& min_val,
                                 const at::Scalar& max_val) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::clamp(bridge::GetLtcTensor(self), min_val, max_val));
}

at::Tensor& AtenXlaType::hardtanh_(at::Tensor& self, const at::Scalar& min_val,
                                   const at::Scalar& max_val) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::clamp_(self_tensor, min_val, max_val);
    return self;
  }
  return AtenXlaTypeDefault::hardtanh_(self, min_val, max_val);
}

at::Tensor AtenXlaType::hardtanh_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& min_val,
                                          const at::Scalar& max_val) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::hardtanh_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self), min_val,
      max_val));
}

at::Tensor AtenXlaType::kl_div(const at::Tensor& self, const at::Tensor& target,
                               int64_t reduction, bool log_target) {
  LTC_FN_COUNTER("xla::");
  return at::native::kl_div(self, target, reduction, log_target);
}

at::Tensor AtenXlaType::kl_div_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        int64_t reduction, bool log_target) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::kl_div_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(target), reduction, log_target));
}

at::Tensor AtenXlaType::le(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::le(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::le(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::le(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::le_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::le_(self, other);
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::le_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::le_(self, other);
}

at::Tensor AtenXlaType::leaky_relu(const at::Tensor& self,
                                   const at::Scalar& negative_slope) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::leaky_relu(
      bridge::GetLtcTensor(self), negative_slope.to<double>()));
}

at::Tensor& AtenXlaType::leaky_relu_(at::Tensor& self,
                                     const at::Scalar& negative_slope) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::leaky_relu_(self_tensor, negative_slope.to<double>());
    return self;
  }
  return AtenXlaTypeDefault::leaky_relu_(self, negative_slope);
}

at::Tensor AtenXlaType::leaky_relu_backward(const at::Tensor& grad_output,
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
  return AtenXlaTypeDefault::leaky_relu_backward(
      grad_output, self, negative_slope, self_is_result);
}

at::Tensor AtenXlaType::log(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log(bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::log10(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(
      bridge::GetLtcTensor(self), ir::OpKind(at::aten::log10), 10.0));
}

at::Tensor AtenXlaType::log1p(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::log1p(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::log1p_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::log1p_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::log1p_(self);
}

at::Tensor AtenXlaType::log2(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_base(
      bridge::GetLtcTensor(self), ir::OpKind(at::aten::log2), 2.0));
}

at::Tensor& AtenXlaType::log_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::log_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::log_(self);
}

at::Tensor AtenXlaType::log_sigmoid_backward(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             const at::Tensor& buffer) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::log_sigmoid_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      bridge::GetLtcTensor(buffer)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::log_sigmoid_forward(
    const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  auto result_tuple =
      LazyTensor::log_sigmoid_forward(bridge::GetLtcTensor(self));
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(result_tuple)),
                         bridge::AtenFromLtcTensor(std::get<1>(result_tuple)));
}

at::Tensor AtenXlaType::logsumexp(const at::Tensor& self, at::IntArrayRef dim,
                                  bool keepdim) {
  return AtenXlaTypeDefault::logsumexp(self, dim, keepdim);
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::lt(bridge::GetLtcTensor(self), other));
  }
  return AtenXlaTypeDefault::lt(self, other);
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::lt(
        bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
  }
  return AtenXlaTypeDefault::lt(self, other);
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::lt_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::lt_(self, other);
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::lt_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::lt_(self, other);
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max(const at::Tensor& self,
                                                    int64_t dim, bool keepdim) {
  return AtenXlaTypeDefault::max(self, dim, keepdim);
}

at::Tensor AtenXlaType::maximum(const at::Tensor& self,
                                const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::max(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::max_pool2d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("xla::");
  return aten_autograd_ops_nnc::MaxPool2dAutogradFunctionNNC::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool3d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  LTC_FN_COUNTER("xla::");
  return aten_autograd_ops_nnc::MaxPool3dAutogradFunctionNNC::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::min(const at::Tensor& self,
                                                    int64_t dim, bool keepdim) {
  return AtenXlaTypeDefault::min(self, dim, keepdim);
}

at::Tensor AtenXlaType::minimum(const at::Tensor& self,
                                const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::min(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, const at::Tensor& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const LazyTensor& xother,
                          at::ScalarType dtype) {
                        return LazyTensor::mul(xself, xother, dtype);
                      });
  }
  return AtenXlaTypeDefault::mul(self, other);
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return DoBinaryOp(self, other,
                      [&](const LazyTensor& xself, const at::Scalar& other,
                          at::ScalarType dtype) {
                        return LazyTensor::mul(xself, other, dtype);
                      });
  }
  return AtenXlaTypeDefault::mul(self, other);
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::mul_(self_tensor, bridge::GetOrCreateLtcTensor(
                                      other, self_tensor.GetDevice()));
    return self;
  }
  return AtenXlaTypeDefault::mul_(self, other);
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::mul_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::mul_(self, other);
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, const at::Scalar& other) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::ne(bridge::GetLtcTensor(self), other));
  }
  return AtenXlaTypeDefault::ne(self, other);
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ne(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::ne_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::ne_(self, other);
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::ne_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::ne_(self, other);
}

at::Tensor AtenXlaType::neg(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    LTC_CHECK(self.scalar_type() != at::kBool)
        << "Negation, the `-` operator, on a bool tensor is not supported. If "
           "you are trying to invert a mask, use the `~` or `logical_not()` "
           "operator instead.";
    return bridge::AtenFromLtcTensor(
        LazyTensor::neg(bridge::GetLtcTensor(self)));
  }
  return AtenXlaTypeDefault::neg(self);
}

at::Tensor& AtenXlaType::neg_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::neg_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::neg_(self);
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             const c10::optional<at::Scalar>& p,
                             at::ScalarType dtype) {
  return AtenXlaTypeDefault::norm(self, p, dtype);
}

at::Tensor AtenXlaType::norm(const at::Tensor& self, const at::Scalar& p) {
  return AtenXlaTypeDefault::norm(self, p);
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             const c10::optional<at::Scalar>& p,
                             at::IntArrayRef dim, bool keepdim,
                             at::ScalarType dtype) {
  return AtenXlaTypeDefault::norm(self, p, dim, keepdim, dtype);
}

at::Tensor AtenXlaType::norm(const at::Tensor& self,
                             const c10::optional<at::Scalar>& p,
                             at::IntArrayRef dim, bool keepdim) {
  return AtenXlaTypeDefault::norm(self, p, dim, keepdim);
}

at::Tensor AtenXlaType::permute(const at::Tensor& self, at::IntArrayRef dims) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::permute(self_tensor, Helpers::I64List(dims)));
  }
  auto result = AtenXlaTypeDefault::permute(self, dims);
  MarkAsInteropView(result);
  return result;
}

at::Tensor AtenXlaType::pow(const at::Tensor& self,
                            const at::Scalar& exponent) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    // xla::Pow() doesn't support integer types.
    if (!at::native::is_floating_point(self)) {
      return AtenXlaTypeDefault::pow(self, exponent);
    }
    return bridge::AtenFromLtcTensor(
        LazyTensor::pow(bridge::GetLtcTensor(self), exponent));
  }
  return AtenXlaTypeDefault::pow(self, exponent);
}

at::Tensor AtenXlaType::pow(const at::Tensor& self,
                            const at::Tensor& exponent) {
  LTC_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!at::native::is_floating_point(self)) {
    return AtenXlaTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(LazyTensor::pow(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(exponent)));
}

at::Tensor AtenXlaType::pow(const at::Scalar& self,
                            const at::Tensor& exponent) {
  LTC_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (!self.isFloatingPoint()) {
    return AtenXlaTypeDefault::pow(self, exponent);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::pow(self, bridge::GetLtcTensor(exponent)));
}

at::Tensor& AtenXlaType::pow_(at::Tensor& self, const at::Scalar& exponent) {
  LTC_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC &&
      at::native::is_floating_point(self)) {
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::pow_(self_tensor, exponent);
    return self;
  }
  return AtenXlaTypeDefault::pow_(self, exponent);
}

at::Tensor& AtenXlaType::pow_(at::Tensor& self, const at::Tensor& exponent) {
  LTC_FN_COUNTER("xla::");
  // xla::Pow() doesn't support integer types.
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC &&
      at::native::is_floating_point(self)) {
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::pow_(self_tensor, bridge::GetLtcTensor(exponent));
    return self;
  }
  return AtenXlaTypeDefault::pow_(self, exponent);
}

at::Tensor AtenXlaType::reciprocal(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::reciprocal(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::reciprocal_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::reciprocal_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::reciprocal_(self);
}

at::Tensor AtenXlaType::relu(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::relu(bridge::GetLtcTensor(self)));
  }
  return AtenXlaTypeDefault::relu(self);
}

at::Tensor& AtenXlaType::relu_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::relu_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::relu_(self);
}

at::Tensor AtenXlaType::remainder(const at::Tensor& self,
                                  const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::remainder(
      bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor AtenXlaType::remainder(const at::Tensor& self,
                                  const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::remainder(bridge::GetLtcTensor(self), other));
}

at::Tensor& AtenXlaType::remainder_(at::Tensor& self, const at::Tensor& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::remainder_(self_tensor, bridge::GetLtcTensor(other));
    return self;
  }
  return AtenXlaTypeDefault::remainder_(self, other);
}

at::Tensor& AtenXlaType::remainder_(at::Tensor& self, const at::Scalar& other) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::remainder_(self_tensor, other);
    return self;
  }
  return AtenXlaTypeDefault::remainder_(self, other);
}

at::Tensor AtenXlaType::repeat(const at::Tensor& self,
                               at::IntArrayRef repeats) {
  return AtenXlaTypeDefault::repeat(self, repeats);
}

at::Tensor& AtenXlaType::resize_(
    at::Tensor& self, at::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::resize_(self_tensor, Helpers::I64List(size));
    return self;
  }
  return AtenXlaTypeDefault::resize_(self, size, memory_format);
}

at::Tensor AtenXlaType::round(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::round(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::round_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::round_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::round_(self);
}

at::Tensor AtenXlaType::rrelu_with_noise(
    const at::Tensor& self, const at::Tensor& noise, const at::Scalar& lower,
    const at::Scalar& upper, bool training,
    c10::optional<at::Generator> generator) {
  LTC_FN_COUNTER("xla::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    LTC_CHECK_EQ(training, false);
    return AtenXlaTypeDefault::rrelu_with_noise(self, noise, lower, upper,
                                                training, generator);
  }
  LazyTensor noise_tensor = bridge::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise(
      bridge::GetLtcTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor AtenXlaType::rrelu_with_noise_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, const at::Scalar& lower, const at::Scalar& upper,
    bool training, bool self_is_result) {
  LTC_FN_COUNTER("xla::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  LTC_CHECK(!self_is_result || negative_slope > 0.0);
  LazyTensor noise_tensor = bridge::GetLtcTensor(noise);
  return bridge::AtenFromLtcTensor(LazyTensor::rrelu_with_noise_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor AtenXlaType::rsqrt(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::rsqrt(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::rsqrt_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::rsqrt_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::rsqrt_(self);
}

at::Tensor AtenXlaType::rsub(const at::Tensor& self, const at::Tensor& other,
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
  return AtenXlaTypeDefault::rsub(self, other, alpha);
}

at::Tensor AtenXlaType::rsub(const at::Tensor& self, const at::Scalar& other,
                             const at::Scalar& alpha) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
    return bridge::AtenFromLtcTensor(
        LazyTensor::rsub(bridge::GetLtcTensor(self), other, alpha));
  }
  return AtenXlaTypeDefault::rsub(self, other, alpha);
}

at::Tensor AtenXlaType::select(const at::Tensor& self, int64_t dim,
                               int64_t index) {
  if (ForceNNC()) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::select(bridge::GetLtcTensor(self), dim, index));
  }
  return AtenXlaTypeDefault::select(self, dim, index);
}

at::Tensor& AtenXlaType::silu_out(const at::Tensor& self, at::Tensor& out) {
  LTC_FN_COUNTER("xla::");
  LazyTensor out_tensor = bridge::GetLtcTensor(out);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::silu_out(self_tensor, out_tensor);
  return out;
}

at::Tensor AtenXlaType::sigmoid(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::sigmoid(bridge::GetLtcTensor(self)));
  }
  return AtenXlaTypeDefault::sigmoid(self);
}

at::Tensor& AtenXlaType::sigmoid_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sigmoid_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::sigmoid_(self);
}

at::Tensor AtenXlaType::sigmoid_backward(const at::Tensor& grad_output,
                                         const at::Tensor& output) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::sigmoid_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output)));
  }
  return AtenXlaTypeDefault::sigmoid_backward(grad_output, output);
}

at::Tensor AtenXlaType::sign(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::sign(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::sign_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sign_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::sign_(self);
}

at::Tensor AtenXlaType::sin(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::sin(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::sin_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sin_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::sin_(self);
}

at::Tensor AtenXlaType::sinh(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::sinh(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::sinh_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sinh_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::sinh_(self);
}

at::Tensor AtenXlaType::slice(const at::Tensor& self, int64_t dim,
                              c10::optional<int64_t> start,
                              c10::optional<int64_t> end, int64_t step) {
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

at::Tensor AtenXlaType::softplus(const at::Tensor& self, const at::Scalar& beta,
                                 const at::Scalar& threshold) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softplus(bridge::GetLtcTensor(self), beta, threshold));
}

at::Tensor AtenXlaType::softplus_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Scalar& beta,
                                          const at::Scalar& threshold,
                                          const at::Tensor& output) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::softplus_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self), beta,
      threshold, bridge::GetLtcTensor(output)));
}

at::Tensor AtenXlaType::softshrink(const at::Tensor& self,
                                   const at::Scalar& lambda) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softshrink(bridge::GetLtcTensor(self), lambda));
}

at::Tensor AtenXlaType::softshrink_backward(const at::Tensor& grad_out,
                                            const at::Tensor& self,
                                            const at::Scalar& lambda) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::softshrink_backward(
      bridge::GetLtcTensor(grad_out), bridge::GetLtcTensor(self), lambda));
}

at::Tensor AtenXlaType::sqrt(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::sqrt(bridge::GetLtcTensor(self)));
  }
  return AtenXlaTypeDefault::sqrt(self);
}

at::Tensor& AtenXlaType::sqrt_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sqrt_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::sqrt_(self);
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self) {
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

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self, int64_t dim) {
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

at::Tensor AtenXlaType::sub(const at::Tensor& self, const at::Tensor& other,
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
  return AtenXlaTypeDefault::sub(self, other, alpha);
}

at::Tensor AtenXlaType::sub(const at::Tensor& self, const at::Scalar& other,
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
  return AtenXlaTypeDefault::sub(self, other, alpha);
}

at::Tensor& AtenXlaType::sub_(at::Tensor& self, const at::Tensor& other,
                              const at::Scalar& alpha) {
  if (InPlaceUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    at::native::alpha_check(at::result_type(self, other), alpha);
    CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sub_(
        self_tensor,
        bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()), alpha);
    return self;
  }
  return AtenXlaTypeDefault::sub_(self, other, alpha);
}

at::Tensor& AtenXlaType::sub_(at::Tensor& self, const at::Scalar& other,
                              const at::Scalar& alpha) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    CheckBinaryOpTypePromotion(self, self, other);
    CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::sub_(self_tensor, other, alpha);
    return self;
  }
  return AtenXlaTypeDefault::sub_(self, other, alpha);
}

at::Tensor AtenXlaType::t(const at::Tensor& self) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::transpose(bridge::GetLtcTensor(self), 0, 1));
  }
  auto result = AtenXlaTypeDefault::t(self);
  MarkAsInteropView(result);
  return result;
}

at::Tensor& AtenXlaType::t_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::transpose_(self_tensor, 0, 1);
    return self;
  }
  return AtenXlaTypeDefault::t_(self);
}

at::Tensor AtenXlaType::tan(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::tan(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::tan_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::tan_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::tan_(self);
}

at::Tensor AtenXlaType::tanh(const at::Tensor& self) {
  if (UseNNC(self)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::tanh(bridge::GetLtcTensor(self)));
  }
  return AtenXlaTypeDefault::tanh(self);
}

at::Tensor& AtenXlaType::tanh_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::tanh_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::tanh_(self);
}

at::Tensor AtenXlaType::tanh_backward(const at::Tensor& grad_output,
                                      const at::Tensor& output) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::tanh_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output)));
  }
  return AtenXlaTypeDefault::tanh_backward(grad_output, output);
}

at::Tensor AtenXlaType::threshold(const at::Tensor& self,
                                  const at::Scalar& threshold,
                                  const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::threshold(
      bridge::GetLtcTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor& AtenXlaType::threshold_(at::Tensor& self,
                                    const at::Scalar& threshold,
                                    const at::Scalar& value) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::threshold_(self_tensor, threshold.to<double>(),
                           value.to<double>());
    return self;
  }
  return AtenXlaTypeDefault::threshold_(self, threshold, value);
}

at::Tensor AtenXlaType::threshold_backward(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           const at::Scalar& threshold) {
  if (UseNNC(grad_output)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(LazyTensor::threshold_backward(
        bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(self),
        threshold.to<double>()));
  }
  return AtenXlaTypeDefault::threshold_backward(grad_output, self, threshold);
}

at::Tensor AtenXlaType::transpose(const at::Tensor& self, int64_t dim0,
                                  int64_t dim1) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::transpose(bridge::GetLtcTensor(self), dim0, dim1));
  }
  auto result = AtenXlaTypeDefault::transpose(self, dim0, dim1);
  MarkAsInteropView(result);
  return result;
}

at::Tensor& AtenXlaType::transpose_(at::Tensor& self, int64_t dim0,
                                    int64_t dim1) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

at::Tensor AtenXlaType::trunc(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::trunc(bridge::GetLtcTensor(self)));
}

at::Tensor& AtenXlaType::trunc_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::trunc_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::trunc_(self);
}

at::Tensor AtenXlaType::view(const at::Tensor& self, at::IntArrayRef size) {
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  if (UseNNCViews(self_tensor)) {
    LTC_FN_COUNTER("xla::");
    return bridge::AtenFromLtcTensor(
        LazyTensor::view(self_tensor, Helpers::I64List(size)));
  }
  auto result = AtenXlaTypeDefault::view(self, size);
  MarkAsInteropView(result);
  return result;
}

at::Tensor& AtenXlaType::zero_(at::Tensor& self) {
  if (InPlaceMustUseNNC(self) == ExecutionKind::NNC) {
    LTC_FN_COUNTER("xla::");
    LazyTensor self_tensor = bridge::GetLtcTensor(self);
    LazyTensor::zero_(self_tensor);
    return self;
  }
  return AtenXlaTypeDefault::zero_(self);
}

void AtenXlaType::InitializeAtenBindings() {
  static std::once_flag once;
  std::call_once(once, []() { AtenInitialize(); });
}

}  // namespace torch_lazy_tensors
