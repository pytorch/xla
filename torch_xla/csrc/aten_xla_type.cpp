#include <ATen/ExpandUtils.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpMathType.h>
#include <ATen/Operators.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ops/_embedding_bag_backward_native.h>
#include <ATen/ops/expand_copy.h>
#include <c10/core/Contiguity.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

#include <mutex>
#include <optional>

#include "torch/csrc/lazy/core/helpers.h"
#include "torch/csrc/lazy/core/shape_inference.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/LazyIr.h"
#include "torch_xla/csrc/XLANativeFunctions.h"
#include "torch_xla/csrc/aten_autograd_ops.h"
#include "torch_xla/csrc/aten_fallback.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/as_strided_view_update.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/diagonal_view_update.h"
#include "torch_xla/csrc/ops/einsum_utilities.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/ops/unselect.h"
#include "torch_xla/csrc/ops/update_slice.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "torch_xla/csrc/xla_sharding_util.h"

// [Implementation Guidelines]
// - If you want to call a at::func which doesn't have a kernel registered
// according to xla_native_functions.yaml,
//   you can call a boxed CPU fallback kernel instead.
//   E.g. don't call tensor.op() or at::op(tensor).
//   use at::native::call_fallback_fn<&xla_fallback,
//         ATEN_OP2(op_name, overload_name)>::call(args...)
//   ATEN_OP accepts an operator name without an overload, and
//   ATEN_OP2 accepts an operator name along with its overload name.
//   The description of these macros can be found in
//   https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/templates/Operators.h
//   (You can find some examples below)

namespace torch_xla {
namespace {

using XLAInputVector = std::vector<XLATensorPtr>;

// Calls the inner function by spreading inputs in order, and adding the
// common data-type in the end.
template <class InnerFnType, size_t... Ints>
XLATensorPtr CallInner(const InnerFnType& inner, XLAInputVector inputs,
                       at::ScalarType common_dtype,
                       std::integer_sequence<size_t, Ints...> seq) {
  return inner(inputs[Ints]..., common_dtype);
}

// Computes the number of XLATensorPtr arguments of a given function.
//
// This is used when calling tensor_methods functions, given a list of inputs.
// Specifically, in order to know how many inputs we should get from the list.
template <class T>
struct NumberOfXLATensorArgs {};

template <class... Args>
struct NumberOfXLATensorArgs<XLATensorPtr(Args...)> {
  static constexpr size_t value =
      (std::is_same_v<XLATensorPtr,
                      std::remove_cv_t<std::remove_reference_t<Args>>> +
       ...);
};

// Stateful configuration structure for pre/post-processing the inputs and the
// output.
//
// There are a few checks and preprocessing that PyTorch does, that we are
// mirroring with this class. This should help us get many data-type behavior
// right.
class OpConfig {
 public:
  using InputVector = std::vector<at::Tensor>;
  using ImplFnType =
      std::function<XLATensorPtr(const XLAInputVector&, at::ScalarType)>;

  // Construct an instance from a function of exactly ImplFnType.
  OpConfig(ImplFnType impl) : impl_(impl) {}

  // Construct an instance from a function of the following type:
  //     XLATensorPtr(Tensor..., ScalarType)
  //
  // This is a convenience for wrapping tensor_methods functions.
  template <class InnerFnType>
  static OpConfig From(const InnerFnType& inner_impl) {
    return OpConfig(
        [&](const XLAInputVector& inputs, at::ScalarType common_dtype) {
          constexpr size_t num_tensor_args =
              NumberOfXLATensorArgs<std::remove_pointer_t<InnerFnType>>::value;
          return CallInner(inner_impl, inputs, common_dtype,
                           std::make_index_sequence<num_tensor_args>{});
        });
  }

  OpConfig& add_input(const at::Tensor& input) {
    inputs_.push_back(input);
    return *this;
  }

  OpConfig& cast_inputs_to_common_dtype() {
    cast_inputs_to_common_dtype_ = true;
    return *this;
  }

  OpConfig& use_opmathtype_for_compute() {
    use_opmathtype_for_compute_ = true;
    return *this;
  }

  // Pre-processes the inputs and post-processes the outputs depending on the
  // configured state of this class.
  //
  // In summary, it will:
  //   - Compute the common data-type to be used
  //   - Cast the inputs to the common data-type
  //   - Cast the inputs to its OpMathType (for computation only)
  //   - Run the specified impl
  //   - Cast the output back to the common data-type
  at::Tensor run() {
    at::ScalarType common_dtype = at::native::result_type(inputs_);
    at::ScalarType opmathtype = at::toOpMathType(common_dtype);

    // Pre-process the inputs, given the specified configuration and
    // common_dtype.
    InputVector inputs = maybe_preprocess_inputs(common_dtype, opmathtype);

    // Look for, at least, one tensor already in PyTorch/XLA.
    InputVector::iterator it = std::find_if(
        inputs.begin(), inputs.end(), [](const at::Tensor& tensor) {
          return bridge::TryGetXlaTensor(tensor);
        });
    XLA_CHECK(it != inputs.end());
    // Transform the inputs into a list of XLATensorPtr.
    // For that, either get their corresponding XLATensorPtr, or use the found
    // XLA tensor's BackendDevice for creating a new one.
    torch::lazy::BackendDevice device = bridge::GetXlaTensor(*it)->GetDevice();
    XLAInputVector xla_inputs(inputs.size());
    std::transform(inputs.begin(), inputs.end(), xla_inputs.begin(),
                   [&](const at::Tensor& tensor) {
                     return bridge::GetOrCreateXlaTensor(tensor, device);
                   });

    // Actually call the impl.
    at::ScalarType inner_dtype =
        (use_opmathtype_for_compute_) ? opmathtype : common_dtype;
    XLATensorPtr xla_out = impl_(xla_inputs, inner_dtype);
    at::Tensor out = bridge::AtenFromXlaTensor(xla_out);

    // If we used OpMathType for the computation, cast the result back to its
    // common_dtype.
    if (use_opmathtype_for_compute_) {
      out = out.to(common_dtype);
    }

    return out;
  }

 private:
  // Pre-processes the inputs based on the state of this instance.
  //
  // In summary:
  //   - Cast the inputs to the common data-type (if
  //     cast_inputs_to_common_dtype_ is set)
  //
  //   - Cast the inputs to the OpMathType data-type (if
  //     use_opmathtype_for_compute_ is set)
  InputVector maybe_preprocess_inputs(at::ScalarType common_dtype,
                                      at::ScalarType opmathtype) {
    InputVector inputs = inputs_;

    // Cast only once: either to the common dtype or to OpMathType.
    if (use_opmathtype_for_compute_) {
      std::transform(
          inputs.begin(), inputs.end(), inputs.begin(),
          [=](const at::Tensor& tensor) { return tensor.to(opmathtype); });
    } else if (cast_inputs_to_common_dtype_) {
      std::transform(
          inputs.begin(), inputs.end(), inputs.begin(),
          [=](const at::Tensor& tensor) { return tensor.to(common_dtype); });
    }

    return inputs;
  }

  // Actual implementation of the operation.
  ImplFnType impl_;

  // List of tensor inputs.
  InputVector inputs_;

  // Whether to cast every input to the common data-type.
  // It's analogous to TensorIterator's flag. If the operation you are lowering
  // uses TensorIterator in PyTorch, you can check whether to set this flag or
  // not.
  bool cast_inputs_to_common_dtype_ = false;

  // Whether to use OpMathType for computation.
  // This flag mimics the actual PyTorch kernel implementations. When lowering
  // an operation, take a look at that for deciding whether to set this flag or
  // not.
  bool use_opmathtype_for_compute_ = false;
};

at::Tensor to_meta(const at::Tensor& tensor) {
  // undefined tensors can't be converted to the meta device, since they don't
  // have sizes/strides
  if (!tensor.defined()) return tensor;
  auto out = at::native::empty_strided_meta_symint(
      tensor.sym_sizes(), tensor.sym_strides(),
      /*dtype=*/std::make_optional(tensor.scalar_type()),
      /*layout=*/std::make_optional(tensor.layout()),
      /*device=*/std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/std::nullopt);
  // needs to handle wrapped numbers, so dtype promotion works properly.
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    out.unsafeGetTensorImpl()->set_wrapped_number(true);
  }
  return out;
}

torch::lazy::BackendDevice GetXlaDeviceOrCurrent(
    const std::optional<c10::Device>& device) {
  auto xla_device_opt = bridge::GetXlaDevice(device);
  return xla_device_opt ? *xla_device_opt : bridge::GetCurrentDevice();
}

bool IsOperationOnType(const std::optional<at::ScalarType>& opt_dtype,
                       at::ScalarType tensor_type, at::ScalarType type) {
  if (opt_dtype && *opt_dtype == type) {
    return true;
  }
  return tensor_type == type;
}

bool TensorsAreOfType(std::vector<XLATensorPtr> tensors, at::ScalarType type) {
  for (const XLATensorPtr& tensor : tensors) {
    if (IsOperationOnType(std::optional<at::ScalarType>(std::nullopt),
                          tensor->dtype(), type)) {
      return true;
    }
  }
  return false;
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

std::optional<at::ScalarType> PromoteIntegralType(
    at::ScalarType src_dtype, const std::optional<at::ScalarType>& opt_dtype) {
  return opt_dtype.has_value() ? opt_dtype.value()
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
  xla::PrimitiveType xla_type = XlaTypeFromTorchType(dtype);
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
    min_max = XlaHelpers::MinMaxValues(XlaTypeFromTorchType(dtype));
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
    absl::Span<const int64_t> input_size, const std::optional<double> scales_h,
    const std::optional<double> scales_w,
    const std::vector<int64_t>& output_size) {
  XLA_CHECK(scales_h);
  XLA_CHECK(scales_w);
  // Calculate the output size from input_shape and scale_factors
  XLA_CHECK_EQ(input_size.size(), 4);
  int64_t output_h = input_size[2] * (*scales_h);
  int64_t output_w = input_size[3] * (*scales_w);
  return {output_h, output_w};
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
at::Tensor DoBinaryOp(const at::Scalar& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  XLATensorPtr other_tensor = bridge::GetXlaTensor(other);
  XLATensorPtr result = bin_op(self, other_tensor, dtype);
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::__ilshift__(self_tensor, other);
  return self;
}

at::Tensor& XLANativeFunctions::__ilshift__(at::Tensor& self,
                                            const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::__ilshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor& XLANativeFunctions::__irshift__(at::Tensor& self,
                                            const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::__irshift__(self_tensor, other);
  return self;
}

at::Tensor& XLANativeFunctions::__irshift__(at::Tensor& self,
                                            const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::__irshift__(self_tensor, bridge::GetXlaTensor(other));
  return self;
}

at::Tensor XLANativeFunctions::__lshift__(const at::Tensor& self,
                                          const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::__lshift__(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::__lshift__(const at::Tensor& self,
                                          const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return tensor_methods::__lshift__(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::__rshift__(const at::Tensor& self,
                                          const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::__rshift__(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::__rshift__(const at::Tensor& self,
                                          const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return tensor_methods::__rshift__(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool3d(
    const at::Tensor& self, at::IntArrayRef output_size) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(_adaptive_avg_pool3d)>::call(self, output_size);
  }
  auto common_device = torch_xla::bridge::GetXlaDevice(self);
  XLA_CHECK(common_device);
  torch::lazy::NodePtr node = torch_xla::MakeNode<AdaptiveAvgPool3d>(
      bridge::GetXlaTensor(self)->GetIrValue(),
      std::vector<int64_t>(output_size.begin(), output_size.end()));
  return torch_xla::bridge::AtenFromXlaTensor(
      torch_xla::XLATensor::Create(std::move(node), *common_device));
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 3),
                                   grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()), output_size,
                               /*pool_dim=*/3)) {
    return at::native::call_fallback_fn<
        &xla_fallback,
        ATEN_OP(_adaptive_avg_pool3d_backward)>::call(grad_output, self);
  }
  auto common_device = torch_xla::bridge::GetXlaDevice(grad_output, self);
  XLA_CHECK(common_device);
  torch::lazy::NodePtr node = torch_xla::MakeNode<AdaptiveAvgPool3dBackward>(
      bridge::GetXlaTensor(grad_output)->GetIrValue(),
      bridge::GetXlaTensor(self)->GetIrValue());

  return torch_xla::bridge::AtenFromXlaTensor(
      torch_xla::XLATensor::Create(std::move(node), *common_device));
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool2d(
    const at::Tensor& self, at::IntArrayRef output_size) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(_adaptive_avg_pool2d)>::call(self, output_size);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::_adaptive_avg_pool2d(
      bridge::GetXlaTensor(self), output_size_list));
}

at::Tensor XLANativeFunctions::_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()), output_size,
                               /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_fallback,
        ATEN_OP(_adaptive_avg_pool2d_backward)>::call(grad_output, self);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::_adaptive_avg_pool2d_backward(
          bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::adaptive_max_pool2d(
    const at::Tensor& self, at::IntArrayRef output_size) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto output_size_list = XlaHelpers::I64List(output_size);
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()),
                               output_size_list, /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(adaptive_max_pool2d)>::call(self, output_size);
  }
  std::tuple<XLATensorPtr, XLATensorPtr> res =
      tensor_methods::adaptive_max_pool2d(bridge::GetXlaTensor(self),
                                          output_size_list);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(res)),
                         bridge::AtenFromXlaTensor(std::get<1>(res)));
}

at::Tensor XLANativeFunctions::adaptive_max_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& indices) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  int64_t rank = grad_output.dim();
  std::vector<int64_t> output_size{grad_output.size(rank - 2),
                                   grad_output.size(rank - 1)};
  if (!IsSupportedAdaptivePool(XlaHelpers::I64List(self.sizes()), output_size,
                               /*pool_dim=*/2)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(adaptive_max_pool2d_backward)>::call(grad_output,
                                                                    self,
                                                                    indices);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::adaptive_max_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self)));
}

void XLANativeFunctions::_amp_foreach_non_finite_check_and_unscale_(
    at::TensorList self, at::Tensor& found_inf, const at::Tensor& inv_scale) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr found_inf_tensor = bridge::GetXlaTensor(found_inf);
  tensor_methods::_amp_foreach_non_finite_check_and_unscale_(
      bridge::GetXlaTensors(self), found_inf_tensor,
      bridge::GetXlaTensor(inv_scale));
}

at::Tensor& XLANativeFunctions::_amp_update_scale_(at::Tensor& current_scale,
                                                   at::Tensor& growth_tracker,
                                                   const at::Tensor& found_inf,
                                                   double scale_growth_factor,
                                                   double scale_backoff_factor,
                                                   int64_t growth_interval) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr growth_tracker_tensor = bridge::GetXlaTensor(growth_tracker);
  XLATensorPtr current_scale_tensor = bridge::GetXlaTensor(current_scale);
  tensor_methods::_amp_update_scale_(
      growth_tracker_tensor, current_scale_tensor,
      bridge::GetXlaTensor(found_inf), scale_growth_factor,
      scale_backoff_factor, growth_interval);
  return current_scale;
}

at::Tensor XLANativeFunctions::_copy_from(const at::Tensor& self,
                                          const at::Tensor& dst,
                                          bool /*non_blocking*/) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto dst_tensor = bridge::TryGetXlaTensor(dst);
  auto self_tensor = bridge::TryGetXlaTensor(self);
  if (!self_tensor) {
    static bool sync_update =
        runtime::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true) &&
        !UseVirtualDevice();
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
    XLA_CHECK(dst_tensor);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    tensor_methods::copy_(dst_tensor, self_tensor);
    bridge::ReplaceXlaTensor(dst, dst_tensor);
  }
  return dst;
}

at::Tensor XLANativeFunctions::_copy_from_and_resize(const at::Tensor& self,
                                                     const at::Tensor& dst) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::XlaCreateTensorList(tensors);
}

// TODO(alanwaketan): Improve the error messages.
// Let's rewrite it without reusing other native functions.
at::Tensor XLANativeFunctions::_to_copy(
    const at::Tensor& self, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory, bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  auto options = self.options();
  // I put each of these setters in a conditional instead of doing
  // `self.options().dtype(dtype).layout(layout)... because calling
  // .dtype(nullopt) on an options() that already has dtype appears to wipe it
  if (dtype) {
    options = options.dtype(dtype);
  }
  if (layout) {
    options = options.layout(layout);
  }
  if (device) {
    options = options.device(device);
  }
  if (pin_memory) {
    options = options.pinned_memory(pin_memory);
  }
  if (memory_format) {
    options = options.memory_format(memory_format);
  }

  // Case 1: Materialize the tensor.
  if (device && device->type() != c10::kXLA) {
    XLA_CHECK(device->type() == c10::kCPU)
        << "only cpu device is supported in _to_copy.";
    auto self_tensor = bridge::GetXlaTensor(self);
    auto eager_tensor = self_tensor->ToTensor(/*detached=*/true);

    // Use the eager .to on the eager tensor.
    return eager_tensor.to(options, non_blocking, /*copy=*/true);
  }

  // Case 2: Create a new XLA tensor with the supplied data and options.
  auto new_tensor =
      empty_symint(self.sym_sizes(), at::typeMetaToScalarType(options.dtype()),
                   options.layout(), options.device(), options.pinned_memory(),
                   options.memory_format_opt());
  return _copy_from(self, new_tensor, non_blocking);
}

at::Tensor& XLANativeFunctions::_index_put_impl_(
    at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate, bool /* unsafe */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return torch_xla::XLANativeFunctions::index_put_(self, indices, values,
                                                   accumulate);
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::_linalg_eigh(
    const at::Tensor& self, std::string_view uplo, bool compute_v) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (!compute_v) {
    // Fallback to aten in case of `eigvalsh`.
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP(_linalg_eigh)>::call(self, uplo,
                                                                     compute_v);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto outputs = tensor_methods::eigh(self_tensor, uplo);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::_linalg_slogdet(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto outputs = tensor_methods::slogdet(self_tensor);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)),
                         bridge::AtenFromXlaTensor(XLATensorPtr()),
                         bridge::AtenFromXlaTensor(XLATensorPtr()));
}

at::Tensor XLANativeFunctions::_log_softmax(const at::Tensor& self, int64_t dim,
                                            bool half_to_float) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto self_meta = to_meta(self);
  auto out_meta = at::meta::_log_softmax(self_meta, dim, half_to_float);

  std::vector<torch::lazy::Shape> shapes{
      torch::lazy::Shape(out_meta.scalar_type(), out_meta.sizes().vec())};
  return bridge::AtenFromXlaTensor(tensor_methods::log_softmax(
      bridge::GetXlaTensor(self), dim, std::nullopt, std::move(shapes)));
}

at::Tensor XLANativeFunctions::_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType /* input_dtype */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::log_softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::_pack_padded_sequence(
    const at::Tensor& input, const at::Tensor& lengths, bool batch_first) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::vector<at::Tensor> xla_tensors = {lengths};
  auto cpu_tensors = bridge::XlaCreateTensorList(xla_tensors);
  return at::native::_pack_padded_sequence(input, cpu_tensors[0], batch_first);
}

at::Tensor XLANativeFunctions::_softmax(const at::Tensor& self, int64_t dim,
                                        bool /* half_to_float */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::softmax(bridge::GetXlaTensor(self), dim, std::nullopt));
}

at::Tensor XLANativeFunctions::_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    at::ScalarType input_dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

at::Tensor XLANativeFunctions::_unsafe_view(const at::Tensor& self,
                                            at::IntArrayRef size) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return view_copy_symint(self, c10::fromIntArrayRefSlow(size));
}

at::Tensor XLANativeFunctions::add(const at::Tensor& self,
                                   const at::Tensor& other,
                                   const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Currently, we disallow the case when both operands contain dynamic
  // dimensions. This is consistent with PyTorch's behavior.
  XLA_CHECK(!(tensor_has_dym_dim(self) && tensor_has_dym_dim(other)))
      << "Both operands of torch.add cannot have dynamic dimensions at the "
         "same time. This is not "
         "supported in PyTorch/XLA.";

  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return tensor_methods::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::add(const at::Tensor& self,
                                   const at::Scalar& other,
                                   const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::add(xself, other, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::addmm(const at::Tensor& self,
                                     const at::Tensor& mat1,
                                     const at::Tensor& mat2,
                                     const at::Scalar& beta,
                                     const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (beta.to<double>() != 1 || alpha.to<double>() != 1) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(addmm)>::call(
        self, mat1, mat2, beta, alpha);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::addmm(bridge::GetXlaTensor(mat1),
                            /*weight=*/bridge::GetXlaTensor(mat2),
                            /*bias=*/bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::alias_copy(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::alias(bridge::GetXlaTensor(self)));
}

at::Tensor& XLANativeFunctions::arange_out(const at::Scalar& start,
                                           const at::Scalar& end,
                                           const at::Scalar& step,
                                           at::Tensor& out) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  tensor_methods::arange_out(out_tensor, start, end, step, out.scalar_type());
  return out;
}

at::Tensor XLANativeFunctions::as_strided_copy(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Retrieve the base tensor, if there's one.
  // This function actually operates on the tensor's storage. Since XLA does not
  // expose the actual storage, we use the originally allocated tensor.
  const at::Tensor& base = bridge::GetXlaTensor(self)->Base();
  at::Tensor tensor = base.defined() ? base : self;

  // Fast path: PyTorch/XLA implementation for as_strided works only with
  // non-overlapping and dense tensors.
  if (c10::_compute_non_overlapping_and_dense(size, stride)) {
    // Sets the base tensor as tensor.
    // Even though this function copies (without aliasing) tensor, it's still
    // treated as a view function in the functionalization layer.
    return bridge::AtenFromXlaTensor(bridge::SetBaseTensor(
        tensor_methods::as_strided(bridge::GetXlaTensor(tensor),
                                   XlaHelpers::I64List(size),
                                   XlaHelpers::I64List(stride),
                                   XlaHelpers::I64Optional(storage_offset)),
        tensor));
  }

  // Slow path: decompose as_strided into indexing (we use take, though)
  // operations. We pre-compute the index on CPU, so as to avoid runtime
  // overhead.
  auto dim = size.size();
  auto itemsize = tensor.dtype().itemsize();
  int64_t storage_size =
      at::detail::computeStorageNbytes(size, stride, itemsize);

  XLA_CHECK(tensor.numel() * itemsize >= storage_size)
      << "as_strided: storage not big enough for size " << size << ": "
      << storage_size << " (needed) vs " << tensor.numel() << " (actual).";

  if (dim == 0 && tensor.numel() > 0) {
    // If there's no specified dimension, return the first element of the
    // storage. This behavior is consistent with eager.
    return select_copy(view_copy_symint(tensor, {tensor.numel()}), 0, 0);
  }

  if (storage_size == 0) {
    // Return an empty tensor, if no storage is actually needed.
    return empty_symint(c10::fromIntArrayRefSlow(size), tensor.scalar_type(),
                        /* layout= */ std::nullopt, tensor.device(),
                        /* pin_memory= */ std::nullopt,
                        /*  memory_format= */ std::nullopt);
  }

  // At this point, the following is true:
  XLA_CHECK(storage_size > 0);
  XLA_CHECK(tensor.numel() > 0);
  XLA_CHECK(dim > 0);

  // Index tensor for gathering the needed elements into contiguous data.
  //
  // PyTorch/XLA, by default, assumes dense and contiguous data. However, when
  // specifying strides, that might not be the case.
  //
  // Therefore, we gather the elements selected by following the size, stride,
  // and storage offset, materializing it into contiguous elements.
  //
  // In order to accomplish that, we create an index tensor. Specifically, we
  // create an n-dimensional tensor (n is the number of dimensions of the
  // output) of indices. Each element represent the at which position of the
  // flattened tensor the desired element is in.

  // Example: arange(13).as_strided((2, 2, 2), (3, 4, 5))
  //
  // Start with a 1-element n-dimensional tensor, initialized with 0:
  //
  //     [[[0]]]
  //
  std::vector<int64_t> view_shape(dim, 1);
  auto index_tensor =
      at::tensor({storage_offset.value_or(self.storage_offset())},
                 at::TensorOptions().dtype(at::kLong))
          .view(view_shape);

  // Then, add to the index_tensor the offset value introduced for each possible
  // index of that corresponding dimension.
  //
  //   - Iteration i=0:
  //        [[[0]]] + [[[0 * 3]], [[1 * 3]]]
  //        = [[[0 * 3]], [[1 * 3]]]
  //        = [[[0]], [[3]]]
  //
  //   - Iteration i=1:
  //        [[[0]], [[3]]] + [[[0 * 4], [1 * 4]]]
  //        = [[[0 + 0 * 4], [0 + 1 * 4]], [[3 + 0 * 4], [3 + 1 * 4]]]
  //        = [[[0], [4]], [[3], [7]]]
  //
  //   - Iteration i=2:
  //        [[[0], [4]], [[3], [7]]] + [[[0 * 5, 1 * 5]]]
  //        =[[[0 + 0 * 5, 0 + 1 * 5], [4 + 0 * 5, 4 + 1 * 5]],
  //          [[3 + 0 * 5, 3 + 1 * 5], [7 + 0 * 5, 7 + 1 * 5]]]
  //        =[[[0, 5], [4, 9]], [[3, 8], [7, 12]]]
  for (int i = 0; i < dim; i++) {
    auto vshape = view_shape;
    vshape[i] = size[i];
    index_tensor =
        index_tensor.add((at::arange(size[i]) * stride[i]).view(vshape));
  }

  // Finally, index the tensor with the computed indices.
  return take(tensor, index_tensor.to(tensor.device()));
}

at::Tensor XLANativeFunctions::as_strided_scatter(
    const at::Tensor& base, const at::Tensor& mutated_view,
    at::IntArrayRef size, at::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto base_ = bridge::GetXlaTensor(base);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!AsStrided::StrideIsSupported(base_->shape(), xsize, xstride,
                                    storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(as_strided_scatter)>::call(base, mutated_view,
                                                          size, stride,
                                                          storage_offset);
  }
  auto mutated_view_ = bridge::GetXlaTensor(mutated_view);
  return bridge::AtenFromXlaTensor(
      base_->CreateFrom(torch_xla::MakeNode<AsStridedViewUpdate>(
          base_->GetIrValue(), mutated_view_->GetIrValue(),
          torch::lazy::ToVector<int64_t>(base_->shape().get().dimensions()),
          xstride, storage_offset.value_or(0))));
}

at::Tensor XLANativeFunctions::atan2(const at::Tensor& self,
                                     const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto common_device = torch_xla::bridge::GetXlaDevice(self, other);
  XLA_CHECK(common_device);
  torch::lazy::NodePtr node =
      torch_xla::MakeNode<Atan2>(bridge::GetXlaTensor(self)->GetIrValue(),
                                 bridge::GetXlaTensor(other)->GetIrValue());

  return torch_xla::bridge::AtenFromXlaTensor(
      torch_xla::XLATensor::Create(std::move(node), *common_device));
}

at::Tensor XLANativeFunctions::avg_pool2d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/2,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode, count_include_pad,
      divisor_override));
}

at::Tensor XLANativeFunctions::avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(avg_pool2d_backward)>::call(grad_output, self,
                                                           kernel_size, stride,
                                                           padding, ceil_mode,
                                                           count_include_pad,
                                                           divisor_override);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::avg_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode,
      count_include_pad));
}

at::Tensor XLANativeFunctions::avg_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::avg_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode, count_include_pad,
      divisor_override));
}

at::Tensor XLANativeFunctions::avg_pool3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if ((ceil_mode && count_include_pad) || divisor_override) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(avg_pool3d_backward)>::call(grad_output, self,
                                                           kernel_size, stride,
                                                           padding, ceil_mode,
                                                           count_include_pad,
                                                           divisor_override);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::avg_pool_nd_backward(
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  return bridge::AtenFromXlaTensor(tensor_methods::baddbmm(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(batch1),
      bridge::GetXlaTensor(batch2), beta, alpha));
}

at::Tensor XLANativeFunctions::bernoulli(
    const at::Tensor& self, std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP(bernoulli)>::call(self,
                                                                  generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::bernoulli(self_tensor));
}

at::Tensor XLANativeFunctions::bernoulli(
    const at::Tensor& self, double p, std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(bernoulli, p)>::call(self, p, generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::bernoulli(self_tensor, p));
}

at::Tensor& XLANativeFunctions::bernoulli_(
    at::Tensor& self, const at::Tensor& p,
    std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(bernoulli_, Tensor)>::call(self, p, generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::bernoulli_(self_tensor, bridge::GetXlaTensor(p));
  return self;
}

at::Tensor XLANativeFunctions::binary_cross_entropy_with_logits(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& pos_weight, int64_t reduction) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return at::native::binary_cross_entropy_with_logits(
      self, target, IsDefined(weight) ? *weight : at::Tensor(),
      IsDefined(pos_weight) ? *pos_weight : at::Tensor(), reduction);
}

at::Tensor XLANativeFunctions::bitwise_and(const at::Tensor& self,
                                           const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const XLATensorPtr& other) {
        return tensor_methods::bitwise_and(xself, other);
      });
}

at::Tensor XLANativeFunctions::bitwise_or(const at::Tensor& self,
                                          const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const XLATensorPtr& xother) {
        return tensor_methods::bitwise_or(xself, xother);
      });
}

at::Tensor XLANativeFunctions::bitwise_xor(const at::Tensor& self,
                                           const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOpWithoutPromo(
      self, other, [&](const XLATensorPtr& xself, const XLATensorPtr& xother) {
        return tensor_methods::bitwise_xor(xself, xother);
      });
}

at::Tensor XLANativeFunctions::bmm(const at::Tensor& self,
                                   const at::Tensor& mat2) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::bmm(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(mat2)));
}

at::Tensor XLANativeFunctions::cat(const at::ITensorListRef& tensors,
                                   int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::cat(
      bridge::GetXlaTensors(tensors), dim, at::native::result_type(tensors)));
}

at::Tensor XLANativeFunctions::celu(const at::Tensor& self,
                                    const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::celu(bridge::GetXlaTensor(self), alpha));
}

at::Tensor& XLANativeFunctions::celu_(at::Tensor& self,
                                      const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::celu_(self_tensor, alpha);
  return self;
}

at::Tensor XLANativeFunctions::clamp(const at::Tensor& self,
                                     const std::optional<at::Scalar>& min,
                                     const std::optional<at::Scalar>& max) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::clamp(bridge::GetXlaTensor(self), min, max));
}

at::Tensor XLANativeFunctions::clamp_max(const at::Tensor& self,
                                         const at::Scalar& max) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::clamp(bridge::GetXlaTensor(self), std::nullopt, max));
}

at::Tensor XLANativeFunctions::clamp_min(const at::Tensor& self,
                                         const at::Scalar& min) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::clamp(bridge::GetXlaTensor(self), min, std::nullopt));
}

at::Tensor XLANativeFunctions::clone(
    const at::Tensor& self,
    std::optional<at::MemoryFormat> /* memory_format */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::clone(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::constant_pad_nd(const at::Tensor& self,
                                               at::IntArrayRef pad,
                                               const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::constant_pad_nd(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(pad), value));
}

// This functions covers the whole convolution lowering.
at::Tensor XLANativeFunctions::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (IsDefined(bias)) {
    return bridge::AtenFromXlaTensor(tensor_methods::convolution_overrideable(
        bridge::GetXlaTensor(input), bridge::GetXlaTensor(weight),
        bridge::GetXlaTensor(*bias), XlaHelpers::I64List(stride),
        XlaHelpers::I64List(padding), XlaHelpers::I64List(dilation), transposed,
        XlaHelpers::I64List(output_padding), groups));
  } else {
    return bridge::AtenFromXlaTensor(tensor_methods::convolution_overrideable(
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto gradients = tensor_methods::convolution_backward_overrideable(
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

at::Tensor XLANativeFunctions::copy(const at::Tensor& self,
                                    const at::Tensor& src, bool non_blocking) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return _copy_from(src, self, non_blocking);
}

at::Tensor& XLANativeFunctions::copy_(at::Tensor& self, const at::Tensor& src,
                                      bool non_blocking) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  _copy_from(src, self, non_blocking);
  return self;
}

at::Tensor XLANativeFunctions::cross(const at::Tensor& self,
                                     const at::Tensor& other,
                                     std::optional<int64_t> dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::cross(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other),
      XlaHelpers::I64Optional(dim)));
}

at::Tensor XLANativeFunctions::cumprod(const at::Tensor& self, int64_t dim,
                                       std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  std::optional<at::ScalarType> promoted_dtype =
      PromoteIntegralType(self_tensor->dtype(), dtype);
  if (IsOperationOnType(promoted_dtype, self_tensor->dtype(),
                        at::ScalarType::Long)) {
    // XLA reduce-window does not support S64 mode.
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(cumprod)>::call(
        self, dim, dtype);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::cumprod(self_tensor, dim, promoted_dtype));
}

at::Tensor XLANativeFunctions::cumsum(const at::Tensor& self, int64_t dim,
                                      std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      tensor_methods::cumsum(self_tensor, dim, dtype));
}

// TODO(alanwaketan): Let's rewrite a without reusing other native functions.
at::Tensor XLANativeFunctions::detach_copy(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self));
}

at::Tensor XLANativeFunctions::diag(const at::Tensor& self, int64_t diagonal) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::diag(bridge::GetXlaTensor(self), diagonal));
}

at::Tensor XLANativeFunctions::diagonal_copy(const at::Tensor& self,
                                             int64_t offset, int64_t dim1,
                                             int64_t dim2) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::diagonal(bridge::GetXlaTensor(self), offset, dim1, dim2));
}

at::Tensor XLANativeFunctions::diagonal_scatter(const at::Tensor& base,
                                                const at::Tensor& mutated_view,
                                                int64_t offset, int64_t dim1,
                                                int64_t dim2) {
  auto base_ = bridge::GetXlaTensor(base);
  auto mutated_view_ = bridge::GetXlaTensor(mutated_view);
  int64_t base_rank = bridge::GetXlaTensor(base)->shape().get().rank();
  int64_t canonical_dim1 =
      torch::lazy::GetCanonicalDimensionIndex(dim1, base_rank);
  int64_t canonical_dim2 =
      torch::lazy::GetCanonicalDimensionIndex(dim2, base_rank);
  return bridge::AtenFromXlaTensor(
      base_->CreateFrom(torch_xla::MakeNode<DiagonalViewUpdate>(
          base_->GetIrValue(), mutated_view_->GetIrValue(), offset,
          canonical_dim1, canonical_dim2)));
}

at::Tensor XLANativeFunctions::div(const at::Tensor& self,
                                   const at::Tensor& other) {
  return torch_xla::XLANativeFunctions::div(self, other,
                                            /*rounding_mode=*/std::nullopt);
}

at::Tensor XLANativeFunctions::div(
    const at::Tensor& self, const at::Tensor& other,
    std::optional<std::string_view> rounding_mode) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  at::ScalarType dtype = at::result_type(self, other);
  auto operands = GetBinaryOperands(self, UnwrapNumber(other, dtype));
  return bridge::AtenFromXlaTensor(tensor_methods::div(
      operands.first, operands.second, rounding_mode, dtype));
}

at::Tensor XLANativeFunctions::div(const at::Tensor& self,
                                   const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::div(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::dot(const at::Tensor& self,
                                   const at::Tensor& tensor) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK_EQ(self.dim(), 1)
      << "dot: Expected 1-D argument self, but got " << self.dim() << "-D";
  XLA_CHECK_EQ(tensor.dim(), 1)
      << "dot: Expected 1-D argument tensor, but got " << tensor.dim() << "-D";
  // Fallback to CPU if both tensor types are integral and atleast one of them
  // is a long, as int64 and uint64 dot products are not supported for TPUs.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetCurrentDevice().type());
  if (CheckTpuDevice(hw_type) &&
      (at::isIntegralType(self.scalar_type(), /*include_bool=*/true) &&
       at::isIntegralType(tensor.scalar_type(), /*include_bool=*/true) &&
       (at::elementSize(self.scalar_type()) == 8 ||
        at::elementSize(tensor.scalar_type()) == 8))) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(dot)>::call(
        self, tensor);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::matmul(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(tensor)));
}

at::Tensor XLANativeFunctions::einsum(std::string_view equation,
                                      at::TensorList tensors,
                                      at::OptionalIntArrayRef path) {
  std::string cleansed_equation = std::string(equation);

  cleansed_equation.erase(
      std::remove_if(cleansed_equation.begin(), cleansed_equation.end(),
                     [](unsigned char x) { return std::isspace(x); }),
      cleansed_equation.end());

  std::vector<XLATensorPtr> xla_tensors = bridge::TryGetXlaTensors(tensors);
  bool all_xla_tensors_are_valid = true;
  for (const XLATensorPtr xla_tensor : xla_tensors) {
    if (!xla_tensor) {
      all_xla_tensors_are_valid = false;
      break;
    }
  }

  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Einsum operations with more than 2 operands, like bilinear operations, are
  // not currently supported in XLA
  if (tensors.size() < 1 || tensors.size() > 2 || !all_xla_tensors_are_valid ||
      !EinsumUtilities::EquationIsValid(cleansed_equation) ||
      TensorsAreOfType(xla_tensors, at::ScalarType::Long)) {
    TORCH_LAZY_COUNTER("EinsumFallback", 1);
    return at::native::einsum(equation, tensors, path);
  }
  return aten_autograd_ops::EinsumAutogradFunction::apply(cleansed_equation,
                                                          tensors);
}

at::Tensor XLANativeFunctions::elu_backward(const at::Tensor& grad_output,
                                            const at::Scalar& alpha,
                                            const at::Scalar& scale,
                                            const at::Scalar& input_scale,
                                            bool self,
                                            const at::Tensor& self_or_result) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK(!self || alpha.to<double>() >= 0.0)
      << "In-place elu backward calculation is triggered with a negative slope "
         "which is not supported.";
  return bridge::AtenFromXlaTensor(tensor_methods::elu_backward(
      bridge::GetXlaTensor(grad_output), alpha, scale, input_scale,
      bridge::GetXlaTensor(self_or_result)));
}

at::Tensor XLANativeFunctions::embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::embedding_dense_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(indices),
      num_weights, padding_idx, scale_grad_by_freq));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::_embedding_bag_forward_only(
    const at::Tensor& weight, const at::Tensor& indices,
    const at::Tensor& offsets, bool scale_grad_by_freq, int64_t mode,
    bool sparse, const std::optional<at::Tensor>& per_sample_weights,
    bool include_last_offset, int64_t padding_idx) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (mode == 1 || scale_grad_by_freq || sparse || padding_idx != -1) {
    return at::native::call_fallback_fn<
        &xla_fallback,
        ATEN_OP(_embedding_bag_forward_only)>::call(weight, indices, offsets,
                                                    scale_grad_by_freq, mode,
                                                    sparse, per_sample_weights,
                                                    include_last_offset,
                                                    padding_idx);
  }
  auto indices_tensor = bridge::GetXlaTensor(indices);
  auto sample_weights =
      per_sample_weights.has_value() && per_sample_weights.value().defined()
          ? bridge::GetXlaTensor(per_sample_weights.value())
          : tensor_methods::full_like(indices_tensor, 1.0,
                                      *torch_xla::bridge::GetXlaDevice(weight),
                                      at::ScalarType::Float);
  auto result = tensor_methods::embedding_bag(
      bridge::GetXlaTensor(weight), indices_tensor,
      bridge::GetXlaTensor(offsets), mode, sample_weights, include_last_offset);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(result)),
                         bridge::AtenFromXlaTensor(std::get<1>(result)),
                         bridge::AtenFromXlaTensor(std::get<2>(result)),
                         bridge::AtenFromXlaTensor(std::get<3>(result)));
}

at::Tensor XLANativeFunctions::_embedding_bag_backward(
    const at::Tensor& grad, const at::Tensor& indices_,
    const at::Tensor& offsets_, const at::Tensor& offset2bag,
    const at::Tensor& bag_size_, const at::Tensor& max_indices_,
    int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse,
    const std::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (sparse) {
    TORCH_WARN(
        "XLA does not support EmbeddingBag sparse backward function. "
        "Falling back to the dense function.");
  }
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::_embedding_bag_backward_symint(
        grad, indices_, offsets_, offset2bag, bag_size_, max_indices_,
        num_weights, scale_grad_by_freq, mode, /*sparse=*/false,
        per_sample_weights_opt, padding_idx);
  }
  return at::native::
      call_fallback_fn<&xla_fallback, ATEN_OP(_embedding_bag_backward)>::call(
          grad, indices_, offsets_, offset2bag, bag_size_, max_indices_,
          num_weights, scale_grad_by_freq, mode, /*sparse=*/false,
          per_sample_weights_opt, padding_idx);
}

at::Tensor XLANativeFunctions::empty_symint(
    at::SymIntArrayRef sym_size, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> /* memory_format */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::optional<at::IntArrayRef> int_sizes =
      c10::asIntArrayRefSlowOpt(sym_size);
  bool all_dims_static = int_sizes.has_value();
  // PT empty*() are optimizations to avoid initializing the data when it is
  // known it will be completely rewritten. But since for us doing a zero*()
  // does not actually end up doing any memory initialization, we use that and
  // avoid going to CPU for it. A common PT pattern is indeed doing empty() plus
  // s_copy_().
  XLATensorPtr xla_tensor;
  if (all_dims_static) {
    xla_tensor = tensor_methods::full(XlaHelpers::I64List(int_sizes.value()), 0,
                                      GetXlaDeviceOrCurrent(device),
                                      at::dtype_or_default(dtype));
  } else {
    xla_tensor =
        tensor_methods::full_symint(sym_size, 0, GetXlaDeviceOrCurrent(device),
                                    at::dtype_or_default(dtype));
  }
  // `tensor.to` will trigger an `empty` + `_to_copy`. In the egaer mode, the
  // `full` will be evulated eagerly and got a replicated sharding. We should
  // leave the sharding to be empty.
  if (XLAGraphExecutor::Get()->UseEagerMode() && UseVirtualDevice()) {
    xla_tensor->ClearShardingSpec();
  }
  return bridge::AtenFromXlaTensor(xla_tensor);
}

at::Tensor XLANativeFunctions::empty_strided_symint(
    at::SymIntArrayRef sym_size, at::SymIntArrayRef sym_stride,
    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
    std::optional<at::Device> device, std::optional<bool> pin_memory) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::optional<at::IntArrayRef> size = c10::asIntArrayRefSlowOpt(sym_size);
  bool is_size_dynamic = !size.has_value();
  std::optional<at::IntArrayRef> stride = c10::asIntArrayRefSlowOpt(sym_stride);
  bool is_stride_dynamic = !stride.has_value();
  // As XLATensor doesn't have a storage, it should not care about the memory
  // format or how to jump to the next element (strides). So the term stride
  // does not mean much to us. The size of the tensor has been set by the
  // above `empty_symint` so we feel it is ok to return here.
  return empty_symint(sym_size, dtype, layout, device, pin_memory,
                      std::nullopt);
}

at::Tensor XLANativeFunctions::expand_copy_symint(const at::Tensor& self,
                                                  at::SymIntArrayRef sym_size,
                                                  bool implicit) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::optional<at::IntArrayRef> size = c10::asIntArrayRefSlowOpt(sym_size);
  if (size.has_value()) {
    return bridge::AtenFromXlaTensor(tensor_methods::expand(
        bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(*size)));
  } else {
    // at least one of the dimension is symbolic, use the sym_int version of the
    // node
    return bridge::AtenFromXlaTensor(
        tensor_methods::expand_symint(bridge::GetXlaTensor(self), sym_size));
  }
}

at::Tensor& XLANativeFunctions::exponential_(
    at::Tensor& self, double lambd, std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP(exponential_)>::call(self,
                                                                     lambd,
                                                                     generator);
  }
  XLA_CHECK_GE(lambd, 0.0);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::exponential_(self_tensor, lambd);
  return self;
}

at::Tensor& XLANativeFunctions::eye_out(int64_t n, at::Tensor& out) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  tensor_methods::eye_out(out_tensor, n, n);
  return out;
}

at::Tensor& XLANativeFunctions::eye_out(int64_t n, int64_t m, at::Tensor& out) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  tensor_methods::eye_out(out_tensor, n, m);
  return out;
}

at::Tensor& XLANativeFunctions::fill_(at::Tensor& self,
                                      const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::fill_(self_tensor, value);
  return self;
}

at::Tensor& XLANativeFunctions::fill_(at::Tensor& self,
                                      const at::Tensor& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK_EQ(value.dim(), 0) << "fill_ only supports a 0-dimensional "
                               << "value tensor, but got tensor "
                               << "with " << value.dim() << " dimension(s).";
  return torch_xla::XLANativeFunctions::fill_(self, value.item());
}

at::Tensor XLANativeFunctions::flip(const at::Tensor& self,
                                    at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::flip(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor XLANativeFunctions::floor_divide(const at::Tensor& self,
                                            const at::Tensor& other) {
  return torch_xla::XLANativeFunctions::div(self, other,
                                            /*rounding_mode=*/"floor");
}

at::Tensor XLANativeFunctions::fmod(const at::Tensor& self,
                                    const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return tensor_methods::fmod(xself, xother, dtype);
                    });
}

at::Tensor XLANativeFunctions::fmod(const at::Tensor& self,
                                    const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::fmod(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::full(at::IntArrayRef size,
                                    const at::Scalar& fill_value,
                                    std::optional<at::ScalarType> dtype,
                                    std::optional<at::Layout> layout,
                                    std::optional<at::Device> device,
                                    std::optional<bool> pin_memory) {
  TORCH_LAZY_FN_COUNTER("xla::");
  // Fall back to CPU if layout or pin_memory are not default
  if (layout.value_or(at::Layout::Strided) != at::Layout::Strided ||
      pin_memory.value_or(false)) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(full)>::call(
        size, fill_value, dtype, layout, device, pin_memory);
  }
  at::ScalarType intend_dtype;
  if (dtype || fill_value.isFloatingPoint()) {
    // Respect the dtype if it is being explictlly passed in.
    // All python scalar will be passed in as float64 to the backend, but the
    // default behavior for pytorch is to return a float32 tensor in this case.
    intend_dtype = at::dtype_or_default(dtype);
  } else {
    intend_dtype = fill_value.type();
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::full(absl::Span<const int64_t>(size), fill_value,
                           GetXlaDeviceOrCurrent(device), intend_dtype));
}

at::Tensor XLANativeFunctions::gather(const at::Tensor& self, int64_t dim,
                                      const at::Tensor& index,
                                      bool /* sparse_grad */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::gather(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor XLANativeFunctions::gelu(const at::Tensor& self,
                                    std::string_view approximate) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::gelu(bridge::GetXlaTensor(self), approximate));
}

at::Tensor XLANativeFunctions::gelu_backward(const at::Tensor& grad,
                                             const at::Tensor& self,
                                             std::string_view approximate) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  at::ScalarType result_type = at::result_type(grad, self);
  return bridge::AtenFromXlaTensor(tensor_methods::gelu_backward(
      bridge::GetXlaTensor(grad.to(result_type)),
      bridge::GetXlaTensor(self.to(result_type)), approximate));
}

at::Tensor XLANativeFunctions::hardtanh(const at::Tensor& self,
                                        const at::Scalar& min_val,
                                        const at::Scalar& max_val) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::clamp(bridge::GetXlaTensor(self), min_val, max_val));
}

at::Tensor XLANativeFunctions::hardtanh_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 const at::Scalar& min_val,
                                                 const at::Scalar& max_val) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::hardtanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), min_val,
      max_val));
}

at::Tensor XLANativeFunctions::index(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  bool indices_on_cpu_or_xla =
      std::all_of(indices.begin(), indices.end(),
                  [=](const std::optional<at::Tensor>& opt) {
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
  std::optional<torch::lazy::BackendDevice> device =
      bridge::GetXlaDevice(canonical_index_info.base);
  if (!device.has_value()) {
    device = bridge::GetXlaDevice(canonical_index_info.indices);
  }
  XLA_CHECK(device.has_value());
  return bridge::AtenFromXlaTensor(tensor_methods::index(
      bridge::GetOrCreateXlaTensor(canonical_index_info.base, *device),
      bridge::GetOrCreateXlaTensors(canonical_index_info.indices, *device),
      canonical_index_info.start_dim));
}

at::Tensor XLANativeFunctions::index_add(const at::Tensor& self, int64_t dim,
                                         const at::Tensor& index,
                                         const at::Tensor& source,
                                         const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::index_add(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
      bridge::GetXlaTensor(source), alpha));
}

at::Tensor XLANativeFunctions::index_copy(const at::Tensor& self, int64_t dim,
                                          const at::Tensor& index,
                                          const at::Tensor& source) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      tensor_methods::index_copy(self_tensor, dim, bridge::GetXlaTensor(index),
                                 bridge::GetXlaTensor(source)));
}

at::Tensor& XLANativeFunctions::index_fill_(at::Tensor& self, int64_t dim,
                                            const at::Tensor& index,
                                            const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index),
                              value);
  return self;
}

at::Tensor& XLANativeFunctions::index_fill_(at::Tensor& self, int64_t dim,
                                            const at::Tensor& index,
                                            const at::Tensor& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::index_fill_(self_tensor, dim, bridge::GetXlaTensor(index),
                              bridge::GetXlaTensor(value));
  return self;
}

at::Tensor& XLANativeFunctions::index_put_(
    at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
    const at::Tensor& values, bool accumulate) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  bool indices_on_cpu_or_xla =
      std::all_of(indices.begin(), indices.end(),
                  [=](const std::optional<at::Tensor>& opt) {
                    return opt.has_value() && opt->defined()
                               ? (opt->is_cpu() || bridge::IsXlaTensor(*opt))
                               : true;
                  });
  XLA_CHECK(bridge::IsXlaTensor(self) && indices_on_cpu_or_xla)
      << "indices should be either on cpu or on the same"
      << " device as the indexed tensor (XLA)."
      << " When using XLA, the indexed tensor must be an XLA tensor.";
  XLA_CHECK(self.scalar_type() == values.scalar_type());
  CanonicalIndexInfo canonical_index_info =
      GetCanonicalIndexInfo(self, indices);
  std::optional<torch::lazy::BackendDevice> device =
      bridge::GetXlaDevice(canonical_index_info.base);
  if (!device.has_value()) {
    device = bridge::GetXlaDevice(canonical_index_info.indices);
  }
  XLA_CHECK(device.has_value());
  XLATensorPtr self_tensor = bridge::GetOrCreateXlaTensor(self, *device);
  tensor_methods::index_put_(
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::index_select(
      bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index)));
}

at::Tensor XLANativeFunctions::kl_div(const at::Tensor& self,
                                      const at::Tensor& target,
                                      int64_t reduction, bool log_target) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return at::native::kl_div(self, target, reduction, log_target);
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::kthvalue(
    const at::Tensor& self, int64_t k, int64_t dim, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto results =
      tensor_methods::kthvalue(bridge::GetXlaTensor(self), k, dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::leaky_relu_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Scalar& negative_slope, bool self_is_result) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK(!self_is_result || negative_slope.to<double>() >= 0.0);
  auto common_device = torch_xla::bridge::GetXlaDevice(self);
  XLA_CHECK(common_device);
  auto node_negative_slope =
      torch::lazy::LazyGraphExecutor::Get()->GetIrValueForScalarFromCodegen(
          negative_slope, *common_device);
  torch::lazy::NodePtr node = torch_xla::MakeNode<LeakyReluBackward>(
      bridge::GetXlaTensor(grad_output)->GetIrValue(),
      bridge::GetXlaTensor(self)->GetIrValue(), node_negative_slope,
      self_is_result);
  return torch_xla::bridge::AtenFromXlaTensor(
      torch_xla::XLATensor::Create(std::move(node), *common_device));
}

at::Tensor XLANativeFunctions::lerp(const at::Tensor& self,
                                    const at::Tensor& end,
                                    const at::Tensor& weight) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK_EQ(self.dtype(), end.dtype())
      << "expected dtype " << self.dtype() << " for `end` but got dtype "
      << end.dtype();
  XLA_CHECK_EQ(self.dtype(), weight.dtype())
      << "expected dtype " << self.dtype() << " for `weight` but got dtype "
      << weight.dtype();
  return bridge::AtenFromXlaTensor(tensor_methods::lerp(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(end),
      bridge::GetXlaTensor(weight)));
}

at::Tensor XLANativeFunctions::lerp(const at::Tensor& self,
                                    const at::Tensor& end,
                                    const at::Scalar& weight) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK_EQ(self.dtype(), end.dtype())
      << "expected dtype " << self.dtype() << " for `end` but got dtype "
      << end.dtype();
  return bridge::AtenFromXlaTensor(tensor_methods::lerp(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(end), weight));
}

at::Tensor XLANativeFunctions::lift(const at::Tensor& tensor) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  return MaybeWrapTensorToFunctional(tensor);
}

at::Tensor XLANativeFunctions::lift_fresh(const at::Tensor& tensor) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(tensor));
  return MaybeWrapTensorToFunctional(tensor);
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::linalg_inv_ex(
    const at::Tensor& self, bool check_errors) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // The default value for `check_errors` is False. And for now, we don't
  // do anything differently based on this flag. So when it's set to True,
  // we'll fallback to CPU.
  if (check_errors) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(linalg_inv_ex)>::call(self, check_errors);
  }
  auto common_device = torch_xla::bridge::GetXlaDevice(self);
  TORCH_INTERNAL_ASSERT(common_device);
  torch::lazy::NodePtr node =
      torch_xla::MakeNode<Inverse>(bridge::GetXlaTensor(self)->GetIrValue());
  auto result = torch_xla::XLATensor::Create(std::move(node), *common_device);
  auto info = tensor_methods::full_like(result, 0, result->GetDevice(),
                                        at::ScalarType::Int);
  return std::make_tuple(bridge::AtenFromXlaTensor(result),
                         bridge::AtenFromXlaTensor(info));
}

at::Tensor XLANativeFunctions::linspace(const at::Scalar& start,
                                        const at::Scalar& end, int64_t steps,
                                        std::optional<at::ScalarType> dtype,
                                        std::optional<at::Layout> layout,
                                        std::optional<at::Device> device,
                                        std::optional<bool> pin_memory) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Fall back to CPU if layout or pin_memory are not default
  if (layout.value_or(at::Layout::Strided) != at::Layout::Strided ||
      pin_memory.value_or(false)) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(linspace)>::call(
        start, end, steps, dtype, layout, device, pin_memory);
  }

  return bridge::AtenFromXlaTensor(
      tensor_methods::linspace(start, end, steps, at::dtype_or_default(dtype),
                               GetXlaDeviceOrCurrent(device)));
}

at::Tensor XLANativeFunctions::log(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::log(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::logit(const at::Tensor& self,
                                     std::optional<double> eps) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::logit(bridge::GetXlaTensor(self), eps));
}

at::Tensor XLANativeFunctions::log10(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::log_base(
      bridge::GetXlaTensor(self), torch::lazy::OpKind(at::aten::log10), 10.0));
}

at::Tensor XLANativeFunctions::log1p(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::log1p(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::log2(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::log_base(
      bridge::GetXlaTensor(self), torch::lazy::OpKind(at::aten::log2), 2.0));
}

at::Tensor XLANativeFunctions::logsumexp(const at::Tensor& self,
                                         at::IntArrayRef dim, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::logsumexp(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(dim),
      /*keep_reduced_dimensions=*/keepdim));
}

at::Tensor XLANativeFunctions::xlogy(const at::Tensor& self,
                                     const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::xlogy(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::masked_scatter(const at::Tensor& self,
                                              const at::Tensor& mask,
                                              const at::Tensor& source) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::masked_scatter(
      self_tensor, bridge::GetXlaTensor(mask), bridge::GetXlaTensor(source)));
}

at::Tensor XLANativeFunctions::masked_select(const at::Tensor& self,
                                             const at::Tensor& mask) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  // Initially make XLA handled masked_select() handling experimental, and
  // opt-in.
  if (!DebugUtil::ExperimentEnabled("masked_select")) {
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP(masked_select)>::call(self,
                                                                      mask);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::masked_select(self_tensor, bridge::GetXlaTensor(mask)));
}

at::Tensor XLANativeFunctions::max(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::max(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::max(
    const at::Tensor& self, int64_t dim, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto outputs = tensor_methods::max(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

std::tuple<at::Tensor&, at::Tensor&> XLANativeFunctions::max_out(
    const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& max,
    at::Tensor& max_values) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr max_tensor = bridge::GetXlaTensor(max);
  XLATensorPtr max_values_tensor = bridge::GetXlaTensor(max_values);
  tensor_methods::max_out(max_tensor, max_values_tensor,
                          bridge::GetXlaTensor(self), dim, keepdim);
  return std::forward_as_tuple(max, max_values);
}

at::Tensor XLANativeFunctions::max_pool2d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return aten_autograd_ops::MaxPool2dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(max_pool2d_with_indices)>::call(self,
                                                               kernel_size,
                                                               stride, padding,
                                                               dilation,
                                                               ceil_mode);
  }
  auto outputs = tensor_methods::max_pool_nd(
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
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_fallback,
        ATEN_OP(max_pool2d_with_indices_backward)>::call(grad_output, self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/2, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
}

at::Tensor XLANativeFunctions::max_pool3d(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return aten_autograd_ops::MaxPool3dAutogradFunction::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor XLANativeFunctions::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_fallback,
        ATEN_OP(max_pool3d_with_indices_backward)>::call(grad_output, self,
                                                         kernel_size, stride,
                                                         padding, dilation,
                                                         ceil_mode, indices);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::max_pool_nd_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      /*spatial_dim_count=*/3, XlaHelpers::I64List(kernel_size),
      XlaHelpers::I64List(stride), XlaHelpers::I64List(padding), ceil_mode));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (IsNonTrivialDilation(dilation)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(max_pool3d_with_indices)>::call(self,
                                                               kernel_size,
                                                               stride, padding,
                                                               dilation,
                                                               ceil_mode);
  }
  auto outputs = tensor_methods::max_pool_nd(
      bridge::GetXlaTensor(self), /*spatial_dim_count=*/3,
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), ceil_mode);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::max_unpool2d(const at::Tensor& self,
                                            const at::Tensor& indices,
                                            at::IntArrayRef output_size) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::max_unpool(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(indices),
      torch::lazy::ToVector<int64_t>(output_size)));
}

at::Tensor XLANativeFunctions::max_unpool3d(const at::Tensor& self,
                                            const at::Tensor& indices,
                                            at::IntArrayRef output_size,
                                            at::IntArrayRef stride,
                                            at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::max_unpool(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(indices),
      torch::lazy::ToVector<int64_t>(output_size)));
}

at::Tensor XLANativeFunctions::mean(const at::Tensor& self,
                                    std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::mean(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor XLANativeFunctions::mean(const at::Tensor& self,
                                    at::OptionalIntArrayRef dim, bool keepdim,
                                    std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::mean(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, dtype));
}

at::Tensor XLANativeFunctions::min(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::min(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::min(
    const at::Tensor& self, int64_t dim, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto outputs = tensor_methods::min(bridge::GetXlaTensor(self), dim, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::mish(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::mish(bridge::GetXlaTensor(self)));
}

std::tuple<at::Tensor&, at::Tensor&> XLANativeFunctions::min_out(
    const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& min,
    at::Tensor& min_indices) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr min_tensor = bridge::GetXlaTensor(min);
  XLATensorPtr min_indices_tensor = bridge::GetXlaTensor(min_indices);
  tensor_methods::min_out(min_tensor, min_indices_tensor,
                          bridge::GetXlaTensor(self), dim, keepdim);
  return std::forward_as_tuple(min, min_indices);
}

at::Tensor XLANativeFunctions::mm(const at::Tensor& self,
                                  const at::Tensor& mat2) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::mm(/*input=*/bridge::GetXlaTensor(self),
                         /*weight=*/bridge::GetXlaTensor(mat2)));
}

at::Tensor XLANativeFunctions::mse_loss(const at::Tensor& self,
                                        const at::Tensor& target,
                                        int64_t reduction) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::mse_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target), reduction));
}

at::Tensor XLANativeFunctions::mse_loss_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 const at::Tensor& target,
                                                 int64_t reduction) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::mse_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction));
}

at::Tensor XLANativeFunctions::mul(const at::Tensor& self,
                                   const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  using FnType = XLATensorPtr(const XLATensorPtr&, const XLATensorPtr&,
                              std::optional<at::ScalarType>);
  return OpConfig::From(static_cast<FnType*>(tensor_methods::mul))
      .add_input(self)
      .add_input(other)
      .cast_inputs_to_common_dtype()
      .use_opmathtype_for_compute()
      .run();
}

at::Tensor XLANativeFunctions::mul(const at::Tensor& self,
                                   const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::mul(xself, other, dtype);
                    });
}

at::Tensor XLANativeFunctions::multinomial(
    const at::Tensor& self, int64_t num_samples, bool replacement,
    std::optional<at::Generator> generator) {
  XLA_CHECK(num_samples > 0)
      << "Multinomial number of samples must be greater than 0";
  XLA_CHECK(at::isFloatingType(self.scalar_type()))
      << "Multinomial input must be a floating type";
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Fallback when sampling is not replaced because it is challenging to
  // parallelize. See https://github.com/pytorch/xla/issues/4865
  if ((generator.has_value() && generator->defined()) ||
      (!replacement && num_samples != 1)) {
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP(multinomial)>::call(self,
                                                                    num_samples,
                                                                    replacement,
                                                                    generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      tensor_methods::multinomial(self_tensor, num_samples, replacement));
}

at::Tensor XLANativeFunctions::mv(const at::Tensor& self,
                                  const at::Tensor& vec) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::mv(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(vec)));
}

at::Tensor& XLANativeFunctions::mv_out(const at::Tensor& self,
                                       const at::Tensor& vec, at::Tensor& out) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr out_tensor = bridge::GetXlaTensor(out);
  tensor_methods::mv_out(out_tensor, bridge::GetXlaTensor(self),
                         bridge::GetXlaTensor(vec));
  return out;
}

at::Tensor XLANativeFunctions::nan_to_num(const at::Tensor& self,
                                          std::optional<double> nan,
                                          std::optional<double> posinf,
                                          std::optional<double> neginf) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
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
  return bridge::AtenFromXlaTensor(tensor_methods::nan_to_num(
      input_tensor, nan_replacement, posinf_replacement, neginf_replacement));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_batch_norm(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr input_tensor = bridge::GetXlaTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  XLATensorPtr running_mean_tensor =
      bridge::GetOrCreateXlaTensor(running_mean, device);
  XLATensorPtr running_var_tensor =
      bridge::GetOrCreateXlaTensor(running_var, device);
  auto outputs = tensor_methods::native_batch_norm(
      bridge::GetXlaTensor(input), bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::_native_batch_norm_legit(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, at::Tensor& running_mean,
    at::Tensor& running_var, bool training, double momentum, double eps) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr input_tensor = bridge::GetXlaTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  XLATensorPtr running_mean_tensor = bridge::GetXlaTensor(running_mean);
  XLATensorPtr running_var_tensor = bridge::GetXlaTensor(running_var);
  auto outputs = tensor_methods::native_batch_norm(
      bridge::GetXlaTensor(input), bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::_native_batch_norm_legit(
    const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, bool training, double momentum,
    double eps) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr input_tensor = bridge::GetXlaTensor(input);
  const torch::lazy::BackendDevice& device = input_tensor->GetDevice();
  XLATensorPtr null_running_mean_tensor = XLATensorPtr();
  XLATensorPtr null_running_var_tensor = XLATensorPtr();
  auto outputs = tensor_methods::native_batch_norm(
      bridge::GetXlaTensor(input), bridge::GetOrCreateXlaTensor(weight, device),
      bridge::GetOrCreateXlaTensor(bias, device), null_running_mean_tensor,
      null_running_var_tensor, training, momentum, eps);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean,
    const std::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr grad_out_tensor = bridge::GetXlaTensor(grad_out);
  const torch::lazy::BackendDevice& device = grad_out_tensor->GetDevice();
  auto gradients = tensor_methods::native_batch_norm_backward(
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

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::native_dropout(
    const at::Tensor& self, double p, std::optional<bool> train) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto results = tensor_methods::native_dropout(self_tensor, p, train);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::neg(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK(self.scalar_type() != at::kBool)
      << "Negation, the `-` operator, on a bool tensor is not supported. If "
         "you are trying to invert a mask, use the `~` or `logical_not()` "
         "operator instead.";
  return bridge::AtenFromXlaTensor(
      tensor_methods::neg(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::nll_loss2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice());
  XLATensorPtr total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateXlaTensor(total_weight, self_tensor->GetDevice());
  }
  return bridge::AtenFromXlaTensor(tensor_methods::nll_loss2d_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::nll_loss2d_forward(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr total_weight = tensor_methods::full(
      {}, 1, self_tensor->GetDevice(), self_tensor->dtype());
  return std::make_tuple(
      bridge::AtenFromXlaTensor(tensor_methods::nll_loss2d(
          self_tensor, bridge::GetXlaTensor(target),
          bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromXlaTensor(total_weight));
}

at::Tensor XLANativeFunctions::nll_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor =
      bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice());
  XLATensorPtr total_weight_tensor;
  if (IsDefined(weight)) {
    total_weight_tensor =
        bridge::GetOrCreateXlaTensor(total_weight, self_tensor->GetDevice());
  }
  return bridge::AtenFromXlaTensor(tensor_methods::nll_loss_backward(
      bridge::GetXlaTensor(grad_output), self_tensor,
      bridge::GetXlaTensor(target), weight_tensor, reduction, ignore_index,
      total_weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr total_weight = tensor_methods::full(
      {}, 1, self_tensor->GetDevice(), self_tensor->dtype());
  return std::make_tuple(
      bridge::AtenFromXlaTensor(tensor_methods::nll_loss(
          self_tensor, bridge::GetXlaTensor(target),
          bridge::GetOrCreateXlaTensor(weight, self_tensor->GetDevice()),
          reduction, ignore_index)),
      bridge::AtenFromXlaTensor(total_weight));
}

at::Tensor XLANativeFunctions::nonzero(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  // Initially make XLA handled nonzero() handling experimental, and opt-in.
  if (!DebugUtil::ExperimentEnabled("nonzero")) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(nonzero)>::call(
        self);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::nonzero(self_tensor));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const std::optional<at::Scalar>& p,
                                    at::ScalarType dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(norm, ScalarOpt_dtype)>::call(self, p, dtype);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::norm(
      bridge::GetXlaTensor(self), p, dtype, {}, /*keepdim=*/false));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const at::Scalar& p) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.toDouble() == 0) {
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP2(norm, Scalar)>::call(self, p);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::norm(
      bridge::GetXlaTensor(self), p, std::nullopt, {}, /*keepdim=*/false));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const std::optional<at::Scalar>& p,
                                    at::IntArrayRef dim, bool keepdim,
                                    at::ScalarType dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(norm, ScalarOpt_dim_dtype)>::call(self, p, dim,
                                                                  keepdim,
                                                                  dtype);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::norm(bridge::GetXlaTensor(self), p, dtype, dim, keepdim));
}

at::Tensor XLANativeFunctions::norm(const at::Tensor& self,
                                    const std::optional<at::Scalar>& p,
                                    at::IntArrayRef dim, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // If p==0 it is a torch.nonzero(), which is not lowered to XLA due to dynamic
  // shapes issue.
  if (p.has_value() && p->toDouble() == 0) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(norm, ScalarOpt_dim)>::call(self, p, dim,
                                                            keepdim);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::norm(
      bridge::GetXlaTensor(self), p, std::nullopt, dim, keepdim));
}

at::Tensor XLANativeFunctions::normal(const at::Tensor& mean, double std,
                                      std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(normal, Tensor_float)>::call(mean, std,
                                                             generator);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::normal(bridge::GetXlaTensor(mean), std));
}

at::Tensor XLANativeFunctions::normal(double mean, const at::Tensor& std,
                                      std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(normal, float_Tensor)>::call(mean, std,
                                                             generator);
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::normal(mean, bridge::GetXlaTensor(std)));
}

at::Tensor XLANativeFunctions::normal(const at::Tensor& mean,
                                      const at::Tensor& std,
                                      std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(normal, Tensor_Tensor)>::call(mean, std,
                                                              generator);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::normal(
      bridge::GetXlaTensor(mean), bridge::GetXlaTensor(std)));
}

at::Tensor& XLANativeFunctions::normal_(
    at::Tensor& self, double mean, double std,
    std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(normal_)>::call(
        self, mean, std, generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::normal_(self_tensor, mean, std);
  return self;
}

at::Tensor XLANativeFunctions::permute_copy(const at::Tensor& self,
                                            at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::permute(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(dims)));
}

at::Tensor XLANativeFunctions::pow(const at::Tensor& self,
                                   const at::Scalar& exponent) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr (*method_pow)(const XLATensorPtr&, const at::Scalar&,
                             std::optional<at::ScalarType>) =
      tensor_methods::pow;
  return DoBinaryOp(self, exponent, method_pow);
}

at::Tensor XLANativeFunctions::pow(const at::Tensor& self,
                                   const at::Tensor& exponent) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr (*method_pow)(const XLATensorPtr&, const XLATensorPtr&,
                             std::optional<at::ScalarType>) =
      tensor_methods::pow;
  return DoBinaryOp(self, exponent, method_pow);
}

at::Tensor XLANativeFunctions::pow(const at::Scalar& self,
                                   const at::Tensor& exponent) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr (*method_pow)(const at::Scalar&, const XLATensorPtr&,
                             std::optional<at::ScalarType>) =
      tensor_methods::pow;
  return DoBinaryOp(self, exponent, method_pow);
}

at::Tensor XLANativeFunctions::_prelu_kernel(const at::Tensor& self,
                                             const at::Tensor& weight) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
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
      tensor_methods::prelu(self_tensor, weight_tensor));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::_prelu_kernel_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& weight) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLATensorPtr weight_tensor = bridge::GetXlaTensor(weight);

  auto outputs = tensor_methods::prelu_backward(grad_output_tensor, self_tensor,
                                                weight_tensor);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(outputs)),
                         bridge::AtenFromXlaTensor(std::get<1>(outputs)));
}

at::Tensor XLANativeFunctions::prod(const at::Tensor& self,
                                    std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::prod(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false,
      PromoteIntegralType(self.scalar_type(), dtype)));
}

at::Tensor XLANativeFunctions::prod(const at::Tensor& self, int64_t dim,
                                    bool keepdim,
                                    std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::prod(bridge::GetXlaTensor(self), {dim}, keepdim,
                           PromoteIntegralType(self.scalar_type(), dtype)));
}

void XLANativeFunctions::_propagate_xla_data(const at::Tensor& input,
                                             const at::Tensor& output) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // This op is only called when functionalize pass is transforming an in-place
  // op. Therefore, we can populate some meta data to maintain any optimization
  // for in-place ops we have in hands.

  // 1) Aid XLA's InputOutputAlias.
  auto input_tensor = bridge::GetXlaTensor(input);
  auto output_tensor = bridge::GetXlaTensor(output);
  if (input_tensor->CurrentDataHandle() != nullptr ||
      (input_tensor->CurrentIrValue().node != nullptr &&
       torch_xla::DeviceData::Cast(
           input_tensor->CurrentIrValue().node.get()))) {
    /*
    if input has a XLAData or holds a devicedata node, set alias_id to
    tensor_id. Consider the case.

    // x.tensor_id = 1, x.alias_id = 1
    x = torch.randn(5,5).to(xla_device())
    // x.tensor_id = 2, x.alias_id should be 1
    x += 1
    xm.mark_step()
    // x.tensor_id =3, x.alias_id should be 2 since input tensor id will be 2
    // for this graph
    x *= 1 of 1
    */
    output_tensor->data()->alias_id = input_tensor->GetUniqueId();
  } else {
    /*
    Consider the case

    // x.tensor_id = 1, x.alias_id = 1
    x = torch.randn(5,5).to(xla_device())
    // x.tensor_id = 2, x.alias_id should be 1
    x += 1
    // x.tensor_id = 3, x.alias_id should still be 1
    x * = 2
    xm.mark_step()
    */
    output_tensor->data()->alias_id = input_tensor->data()->alias_id;
  }

  // 2) Aid SPMD.
  XLATensor::ShardingSpecPtr sharding = input_tensor->sharding_spec();
  // don't propagate sharding in eager mode.
  if (!XLAGraphExecutor::Get()->UseEagerMode() && sharding &&
      sharding->sharding.type() != xla::OpSharding::UNKNOWN) {
    tensor_methods::custom_sharding_(output_tensor,
                                     input_tensor->sharding_spec());
  }
}

at::Tensor& XLANativeFunctions::put_(at::Tensor& self, const at::Tensor& index,
                                     const at::Tensor& source,
                                     bool accumulate) {
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::put_(self_tensor, bridge::GetXlaTensor(index),
                       bridge::GetXlaTensor(source), accumulate);
  return self;
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::qr(
    const at::Tensor& self, bool some) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto results = tensor_methods::qr(bridge::GetXlaTensor(self), some);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

// The value generated should be within (from, to].
at::Tensor& XLANativeFunctions::random_(
    at::Tensor& self, int64_t from, std::optional<int64_t> to,
    std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(random_, from)>::call(self, from, to,
                                                      generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  at::ScalarType dtype = self_tensor->dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  int64_t to_val = (to) ? *to : GetIntegerUpperLimitForType(dtype) + inc;
  XLA_CHECK_LE(from, to_val);
  CheckRangeValues(self_tensor->dtype(), from, to_val - 1);
  tensor_methods::random_(self_tensor, from, to_val);
  return self;
}

// The value generated should be in (0, to].
at::Tensor& XLANativeFunctions::random_(
    at::Tensor& self, int64_t to, std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_fallback,
                                        ATEN_OP2(random_, to)>::call(self, to,
                                                                     generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  XLA_CHECK_GT(to, 0);
  CheckRangeValues(self_tensor->dtype(), 0, to - 1);
  tensor_methods::random_(self_tensor, 0, to);
  return self;
}

// The value generated should be in (self_type_min, self_type_max).
at::Tensor& XLANativeFunctions::random_(
    at::Tensor& self, std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(random_)>::call(
        self, generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  at::ScalarType dtype = self_tensor->dtype();
  // Prevent "to_val" from overflowing with at::ScalarType::Long.
  int64_t inc = (dtype == at::ScalarType::Long) ? 0 : 1;
  tensor_methods::random_(self_tensor, 0,
                          GetIntegerUpperLimitForType(dtype) + inc);
  return self;
}

at::Tensor XLANativeFunctions::randperm(int64_t n,
                                        std::optional<at::ScalarType> dtype,
                                        std::optional<at::Layout> layout,
                                        std::optional<at::Device> device,
                                        std::optional<bool> pin_memory) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  // Only support the basic version of randperm(int64_t) to start. If there are
  // any other parameters, fallback to CPU.
  bool fallback_to_cpu = false;
  fallback_to_cpu |= layout.has_value();
  fallback_to_cpu |= pin_memory.has_value() && pin_memory.value() == true;
  fallback_to_cpu |= dtype.value() != at::ScalarType::Long;
  fallback_to_cpu |= n == 0;

  if (fallback_to_cpu) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(randperm)>::call(
        n, dtype, layout, device, pin_memory);
  }

  return bridge::AtenFromXlaTensor(tensor_methods::randperm(
      n, GetXlaDeviceOrCurrent(device), at::ScalarType::Long));
}

at::Tensor XLANativeFunctions::reflection_pad1d(const at::Tensor& self,
                                                at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::reflection_pad1d(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::reflection_pad1d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::reflection_pad1d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::reflection_pad2d(const at::Tensor& self,
                                                at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::reflection_pad2d(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::reflection_pad2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::reflection_pad2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::reflection_pad3d(const at::Tensor& self,
                                                at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::reflection_pad3d(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::reflection_pad3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::reflection_pad3d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      torch::lazy::ToVector<int64_t>(padding)));
}

at::Tensor XLANativeFunctions::remainder(const at::Tensor& self,
                                         const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::remainder(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(other)));
}

at::Tensor XLANativeFunctions::remainder(const at::Tensor& self,
                                         const at::Scalar& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::remainder(bridge::GetXlaTensor(self), other));
}

at::Tensor XLANativeFunctions::replication_pad1d(const at::Tensor& self,
                                                 at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::replication_pad1d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad1d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::replication_pad1d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad2d(const at::Tensor& self,
                                                 at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::replication_pad2d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::replication_pad2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad3d(const at::Tensor& self,
                                                 at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::replication_pad3d(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(padding)));
}

at::Tensor XLANativeFunctions::replication_pad3d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::replication_pad3d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(padding)));
}

const at::Tensor& XLANativeFunctions::resize_(
    const at::Tensor& self, at::IntArrayRef size,
    std::optional<at::MemoryFormat> /* memory_format */) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::resize_(self_tensor, XlaHelpers::I64List(size));
  return self;
}

at::Tensor XLANativeFunctions::roll(const at::Tensor& self,
                                    at::IntArrayRef shifts,
                                    at::IntArrayRef dims) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::roll(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(shifts),
      XlaHelpers::I64List(dims)));
}

at::Tensor XLANativeFunctions::rrelu_with_noise(
    const at::Tensor& self, at::Tensor& noise, const at::Scalar& lower,
    const at::Scalar& upper, bool training,
    std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    // The fallback path for rrelu_with_noise when training=true is wrong
    XLA_CHECK_EQ(training, false);
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(rrelu_with_noise)>::call(self, noise, lower,
                                                        upper, training,
                                                        generator);
  }
  XLATensorPtr noise_tensor = bridge::GetXlaTensor(noise);
  return bridge::AtenFromXlaTensor(tensor_methods::rrelu_with_noise(
      bridge::GetXlaTensor(self), noise_tensor, lower, upper, training));
}

at::Tensor XLANativeFunctions::rrelu_with_noise_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& noise, const at::Scalar& lower, const at::Scalar& upper,
    bool training, bool self_is_result) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  double negative_slope = (lower.to<double>() + upper.to<double>()) / 2;
  XLA_CHECK(!self_is_result || negative_slope > 0.0);
  XLATensorPtr noise_tensor = bridge::GetXlaTensor(noise);
  return bridge::AtenFromXlaTensor(tensor_methods::rrelu_with_noise_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      noise_tensor, lower, upper, training));
}

at::Tensor XLANativeFunctions::rsub(const at::Tensor& self,
                                    const at::Tensor& other,
                                    const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return tensor_methods::rsub(xself, xother, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::rsub(const at::Tensor& self,
                                    const at::Scalar& other,
                                    const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::rsub(xself, other, alpha, dtype);
                    });
}

at::Tensor scatter_reduce_helper(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, const at::Tensor& src,
                                 std::optional<std::string_view> reduce) {
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  if (!reduce.has_value()) {
    return bridge::AtenFromXlaTensor(
        tensor_methods::scatter(self_tensor, dim, bridge::GetXlaTensor(index),
                                bridge::GetXlaTensor(src)));
  } else if (*reduce == "add") {
    return bridge::AtenFromXlaTensor(tensor_methods::scatter_add(
        self_tensor, dim, bridge::GetXlaTensor(index),
        bridge::GetXlaTensor(src)));
  } else {
    // TODO: implement scatter_mul
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(scatter, reduce)>::call(self, dim, index, src,
                                                        *reduce);
  }
}

at::Tensor scatter_reduce_helper(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index,
                                 const at::Scalar& value,
                                 std::optional<std::string_view> reduce) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  if (!reduce.has_value()) {
    return bridge::AtenFromXlaTensor(tensor_methods::scatter(
        self_tensor, dim, bridge::GetXlaTensor(index), value));
  } else if (*reduce == "add") {
    return bridge::AtenFromXlaTensor(tensor_methods::scatter_add(
        self_tensor, dim, bridge::GetXlaTensor(index), value));
  } else {
    // TODO: implement scatter_mul
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(scatter, value_reduce)>::call(self, dim, index,
                                                              value, *reduce);
  }
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Tensor& src) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return scatter_reduce_helper(self, dim, index, src, std::nullopt);
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return scatter_reduce_helper(self, dim, index, value, std::nullopt);
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Tensor& src,
                                       std::string_view reduce) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return scatter_reduce_helper(self, dim, index, src, reduce);
}

at::Tensor XLANativeFunctions::scatter(const at::Tensor& self, int64_t dim,
                                       const at::Tensor& index,
                                       const at::Scalar& value,
                                       std::string_view reduce) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return scatter_reduce_helper(self, dim, index, value, reduce);
}

at::Tensor XLANativeFunctions::scatter_add(const at::Tensor& self, int64_t dim,
                                           const at::Tensor& index,
                                           const at::Tensor& src) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return scatter_reduce_helper(self, dim, index, src, "add");
}

// TODO(sranlatais): mean is not supported; include_self=false also not
// supported
at::Tensor XLANativeFunctions::scatter_reduce(
    const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& src, std::string_view reduce, bool include_self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if ((reduce == "sum" || reduce == "prod" || reduce == "amin" ||
       reduce == "amax") &&
      include_self) {
    return bridge::AtenFromXlaTensor(tensor_methods::scatter_reduce(
        bridge::GetXlaTensor(self), dim, bridge::GetXlaTensor(index),
        bridge::GetXlaTensor(src), reduce, include_self));
  } else {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP2(scatter_reduce, two)>::call(self, dim, index,
                                                            src, reduce,
                                                            include_self);
  }
}

at::Tensor XLANativeFunctions::select_copy(const at::Tensor& self, int64_t dim,
                                           int64_t index) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::select(bridge::GetXlaTensor(self), dim, index));
}

at::Tensor XLANativeFunctions::select_scatter(const at::Tensor& base,
                                              const at::Tensor& mutated_view,
                                              int64_t dim, int64_t index) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto base_tensor = bridge::GetXlaTensor(base);
  auto base_tensor_shape = base_tensor->shape();
  auto mutated_view_tensor = bridge::GetXlaTensor(mutated_view);
  auto mutated_view_tensor_shape = mutated_view_tensor->shape();
  auto common_device = torch_xla::bridge::GetXlaDevice(base);

  dim = torch::lazy::GetCanonicalDimensionIndex(dim,
                                                base_tensor_shape.get().rank());
  xla::Shape narrow_shape = base_tensor_shape;
  narrow_shape.set_dimensions(dim, 1);
  torch::lazy::NodePtr mutated_view_tensor_reshaped_node =
      torch_xla::MakeNode<ViewOp>(
          mutated_view_tensor->GetIrValue(),
          torch::lazy::ToVector<int64_t>(narrow_shape.dimensions()));

  std::vector<int64_t> indices(base_tensor_shape.get().rank(), 0);
  indices[dim] = torch::lazy::GetCanonicalPosition(
      runtime::util::ToVector<int64_t>(base_tensor_shape.get().dimensions()),
      dim, index);
  return bridge::AtenFromXlaTensor(
      base_tensor->CreateFrom(torch_xla::MakeNode<UpdateSlice>(
          base_tensor->GetIrValue(), mutated_view_tensor_reshaped_node,
          indices)));
}

// TODO(JackCaoG): Remove after elu being codegened
at::Tensor& XLANativeFunctions::selu_(at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::selu_(self_tensor);
  return self;
}

at::Tensor& XLANativeFunctions::set_(at::Tensor& self,
                                     const at::Tensor& source) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr source_tensor = bridge::GetXlaTensor(source);
  bridge::ReplaceXlaTensor(self, source_tensor);
  return self;
}

at::Tensor XLANativeFunctions::sigmoid_backward(const at::Tensor& grad_output,
                                                const at::Tensor& output) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::sigmoid_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor XLANativeFunctions::slice_copy(const at::Tensor& self, int64_t dim,
                                          std::optional<int64_t> start,
                                          std::optional<int64_t> end,
                                          int64_t step) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;
  return bridge::AtenFromXlaTensor(bridge::SetBaseTensor(
      tensor_methods::slice(bridge::GetXlaTensor(self), dim, start_val, end_val,
                            step),
      self));
}

at::Tensor XLANativeFunctions::slice_scatter(
    const at::Tensor& base, const at::Tensor& mutated_view, int64_t dim,
    std::optional<int64_t> start, std::optional<int64_t> end, int64_t step) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto base_ = bridge::GetXlaTensor(base);
  auto mutated_view_ = bridge::GetXlaTensor(mutated_view);
  int64_t start_val = start.has_value() ? start.value() : 0;
  int64_t end_val = end.has_value() ? end.value() : INT64_MAX;

  auto input_shape = base_->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start_val = torch::lazy::GetCanonicalPosition(
      runtime::util::ToVector<int64_t>(input_shape.get().dimensions()), dim,
      start_val);
  end_val = torch::lazy::GetCanonicalPosition(
      runtime::util::ToVector<int64_t>(input_shape.get().dimensions()), dim,
      end_val);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start_val > end_val) {
    end_val = start_val;
  }
  step = std::min(step, end_val - start_val);

  return bridge::AtenFromXlaTensor(
      base_->CreateFrom(torch_xla::MakeNode<Unselect>(
          base_->GetIrValue(), mutated_view_->GetIrValue(), dim, start_val,
          end_val, step)));
}

at::Tensor XLANativeFunctions::smooth_l1_loss(const at::Tensor& self,
                                              const at::Tensor& target,
                                              int64_t reduction, double beta) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::smooth_l1_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target), reduction,
      beta));
}

at::Tensor XLANativeFunctions::smooth_l1_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, int64_t reduction, double beta) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::smooth_l1_loss_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      bridge::GetXlaTensor(target), reduction, beta));
}

at::Tensor XLANativeFunctions::softplus(const at::Tensor& self,
                                        const at::Scalar& beta,
                                        const at::Scalar& threshold) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::softplus(bridge::GetXlaTensor(self), beta, threshold));
}

at::Tensor XLANativeFunctions::softplus_backward(const at::Tensor& grad_output,
                                                 const at::Tensor& self,
                                                 const at::Scalar& beta,
                                                 const at::Scalar& threshold) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::softplus_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self), beta,
      threshold));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::sort(
    const at::Tensor& self, int64_t dim, bool descending) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto results =
      tensor_methods::topk(bridge::GetXlaTensor(self), self.size(dim), dim,
                           descending, /*sorted=*/true, /*stable=*/false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::sort(
    const at::Tensor& self, std::optional<bool> stable, int64_t dim,
    bool descending) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto results = tensor_methods::topk(
      bridge::GetXlaTensor(self), self.size(dim), dim, descending,
      /*sorted=*/false,
      /*stable=*/stable.has_value() ? stable.value() : false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::vector<at::Tensor> XLANativeFunctions::split_copy(const at::Tensor& self,
                                                       int64_t split_size,
                                                       int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto xla_tensors =
      tensor_methods::split(bridge::GetXlaTensor(self), split_size, dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

std::vector<at::Tensor> XLANativeFunctions::split_with_sizes_copy(
    const at::Tensor& self, at::IntArrayRef split_sizes, int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto xla_tensors = tensor_methods::split_with_sizes(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(split_sizes), dim);
  return bridge::AtenFromXlaTensors(xla_tensors);
}

at::Tensor XLANativeFunctions::squeeze_copy(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::squeeze(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::squeeze_copy(const at::Tensor& self,
                                            int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::squeeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor XLANativeFunctions::squeeze_copy(const at::Tensor& self,
                                            at::IntArrayRef dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::squeeze(
      bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(dim)));
}

at::Tensor XLANativeFunctions::stack(at::TensorList tensors, int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  at::ScalarType result_type = at::native::result_type(tensors);
  std::vector<at::Tensor> c_tensors(tensors.size());
  std::transform(tensors.begin(), tensors.end(), c_tensors.begin(),
                 [=](const at::Tensor& t) { return t.to(result_type); });
  return bridge::AtenFromXlaTensor(
      tensor_methods::stack(bridge::GetXlaTensors(c_tensors), dim));
}

at::Tensor XLANativeFunctions::std(const at::Tensor& self, bool unbiased) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::std(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false, /*correction=*/unbiased ? 1.0 : 0.0));
}

at::Tensor XLANativeFunctions::std(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim, bool unbiased,
                                   bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::std(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, /*correction=*/unbiased ? 1.0 : 0.0));
}

at::Tensor XLANativeFunctions::std(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim,
                                   const std::optional<c10::Scalar>& correction,
                                   bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::std(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, correction ? correction->toDouble() : 1.0));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::std_mean(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<c10::Scalar>& correction, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto results = tensor_methods::std_mean(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      correction ? correction->toDouble() : 1.0, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::sub(const at::Tensor& self,
                                   const at::Tensor& other,
                                   const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Currently, we disallow the case when both operands contain dynamic
  // dimensions. This is consistent with PyTorch's behavior.
  XLA_CHECK(!(tensor_has_dym_dim(self) && tensor_has_dym_dim(other)))
      << "Both operands of torch.sub cannot have dynamic dimensions at the "
         "same time. This is not "
         "supported in PyTorch/XLA.";

  CheckSubOperandTypes(self.scalar_type(), other.scalar_type());
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const XLATensorPtr& xother,
                        at::ScalarType dtype) {
                      return tensor_methods::sub(xself, xother, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::sub(const at::Tensor& self,
                                   const at::Scalar& other,
                                   const at::Scalar& alpha) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  CheckSubOperandTypes(self.scalar_type(), GetScalarType(other));
  return DoBinaryOp(self, other,
                    [&](const XLATensorPtr& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return tensor_methods::sub(xself, other, alpha, dtype);
                    });
}

at::Tensor XLANativeFunctions::sum(const at::Tensor& self,
                                   std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::sum(
      self_tensor,
      torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      /*keep_reduced_dimensions=*/false, dtype));
}

at::Tensor XLANativeFunctions::sum(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim, bool keepdim,
                                   std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::sum(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> XLANativeFunctions::svd(
    const at::Tensor& self, bool some, bool compute_uv) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto results =
      tensor_methods::svd(bridge::GetXlaTensor(self), some, compute_uv);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)),
                         bridge::AtenFromXlaTensor(std::get<2>(results)));
}

at::Tensor XLANativeFunctions::t_copy(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::transpose(bridge::GetXlaTensor(self), 0, 1));
}

at::Tensor XLANativeFunctions::tanh_backward(const at::Tensor& grad_output,
                                             const at::Tensor& output) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::tanh_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output)));
}

at::Tensor XLANativeFunctions::threshold(const at::Tensor& self,
                                         const at::Scalar& threshold,
                                         const at::Scalar& value) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::threshold(
      bridge::GetXlaTensor(self), threshold.to<double>(), value.to<double>()));
}

at::Tensor XLANativeFunctions::threshold_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self,
                                                  const at::Scalar& threshold) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::threshold_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      threshold.to<double>()));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::topk(
    const at::Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  auto results = tensor_methods::topk(bridge::GetXlaTensor(self), k, dim,
                                      largest, sorted, /*stable=*/false);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::trace(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::trace(bridge::GetXlaTensor(self)));
}

at::Tensor XLANativeFunctions::transpose_copy(const at::Tensor& self,
                                              int64_t dim0, int64_t dim1) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::transpose(bridge::GetXlaTensor(self), dim0, dim1));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::triangular_solve(
    const at::Tensor& b, const at::Tensor& A, bool upper, bool transpose,
    bool unitriangular) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // Currently, ATen doesn't have a left_side option. Once this
  // is added, this API will have to be changed.
  auto results = tensor_methods::triangular_solve(
      bridge::GetXlaTensor(b), bridge::GetXlaTensor(A), /*left_side=*/true,
      upper, transpose, unitriangular);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

std::vector<at::Tensor> XLANativeFunctions::unbind_copy(const at::Tensor& self,
                                                        int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensors(
      tensor_methods::unbind(bridge::GetXlaTensor(self), dim));
}

at::Tensor& XLANativeFunctions::uniform_(
    at::Tensor& self, double from, double to,
    std::optional<at::Generator> generator) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  if (generator.has_value() && generator->defined()) {
    return at::native::call_fallback_fn<&xla_fallback, ATEN_OP(uniform_)>::call(
        self, from, to, generator);
  }
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::uniform_(self_tensor, from, to);
  return self;
}

at::Tensor XLANativeFunctions::unsqueeze_copy(const at::Tensor& self,
                                              int64_t dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::unsqueeze(bridge::GetXlaTensor(self), dim));
}

at::Tensor XLANativeFunctions::upsample_bilinear2d(
    const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  absl::Span<const int64_t> input_dims =
      self_tensor->shape().get().dimensions();
  std::vector<int64_t> scaled_output_size =
      torch::lazy::ToVector<int64_t>(output_size);
  if ((scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    scaled_output_size = GetOutputSizeWithScale(input_dims, scales_h, scales_w,
                                                scaled_output_size);
    if (!output_size.empty()) {
      XLA_CHECK(scaled_output_size.at(0) == output_size.at(0) &&
                scaled_output_size.at(1) == output_size.at(1))
          << "Inferred output size and output_size from upstream are different";
    }
  }
  return bridge::AtenFromXlaTensor(tensor_methods::upsample_bilinear2d(
      self_tensor, scaled_output_size, align_corners));
}

at::Tensor XLANativeFunctions::upsample_bilinear2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(grad_output_tensor->GetDevice().type());
  if (!CheckTpuDevice(hw_type)) {
    return at::native::call_fallback_fn<
        &xla_fallback,
        ATEN_OP(upsample_bilinear2d_backward)>::call(grad_output, output_size,
                                                     input_size, align_corners,
                                                     scales_h, scales_w);
  }
  std::vector<int64_t> scaled_output_size =
      torch::lazy::ToVector<int64_t>(output_size);
  if ((scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    scaled_output_size = GetOutputSizeWithScale(input_size, scales_h, scales_w,
                                                scaled_output_size);
    if (!output_size.empty()) {
      XLA_CHECK(scaled_output_size.at(0) == output_size.at(0) &&
                scaled_output_size.at(1) == output_size.at(1))
          << "Inferred output size and output_size from upstream are different";
    }
  }
  return bridge::AtenFromXlaTensor(tensor_methods::upsample_bilinear2d_backward(
      grad_output_tensor, torch::lazy::ToVector<int64_t>(scaled_output_size),
      torch::lazy::ToVector<int64_t>(input_size), align_corners));
}

at::Tensor XLANativeFunctions::upsample_nearest2d(
    const at::Tensor& self, at::IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  absl::Span<const int64_t> input_dims =
      self_tensor->shape().get().dimensions();
  std::vector<int64_t> scaled_output_size =
      torch::lazy::ToVector<int64_t>(output_size);
  if ((scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    scaled_output_size = GetOutputSizeWithScale(input_dims, scales_h, scales_w,
                                                scaled_output_size);
    if (!output_size.empty()) {
      XLA_CHECK(scaled_output_size.at(0) == output_size.at(0) &&
                scaled_output_size.at(1) == output_size.at(1))
          << "Inferred output size and output_size from upstream are different";
    }
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::upsample_nearest2d(self_tensor, scaled_output_size));
}

at::Tensor XLANativeFunctions::upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr grad_output_tensor = bridge::GetXlaTensor(grad_output);
  // Only the XLA TPU backend for now implements the CustomCall required by
  // our XLA lowering.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(grad_output_tensor->GetDevice().type());
  if (!CheckTpuDevice(hw_type) && !CheckNeuronDevice(hw_type)) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(upsample_nearest2d_backward)>::call(grad_output,
                                                                   output_size,
                                                                   input_size,
                                                                   scales_h,
                                                                   scales_w);
  }
  std::vector<int64_t> scaled_output_size =
      torch::lazy::ToVector<int64_t>(output_size);
  if ((scales_h && *scales_h != 1.0) || (scales_w && *scales_w != 1.0)) {
    scaled_output_size = GetOutputSizeWithScale(input_size, scales_h, scales_w,
                                                scaled_output_size);
    if (!output_size.empty()) {
      XLA_CHECK(scaled_output_size.at(0) == output_size.at(0) &&
                scaled_output_size.at(1) == output_size.at(1))
          << "Inferred output size and output_size from upstream are different";
    }
  }
  return bridge::AtenFromXlaTensor(tensor_methods::upsample_nearest2d_backward(
      grad_output_tensor, torch::lazy::ToVector<int64_t>(scaled_output_size),
      torch::lazy::ToVector<int64_t>(input_size)));
}

at::Tensor XLANativeFunctions::var(const at::Tensor& self,
                                   at::OptionalIntArrayRef dim,
                                   const std::optional<c10::Scalar>& correction,
                                   bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::var(
      self_tensor,
      dim ? XlaHelpers::I64List(*dim)
          : torch::lazy::Iota<int64_t>(
                bridge::GetXlaTensor(self)->shape().get().rank()),
      correction ? correction->toDouble() : 1.0, keepdim));
}

std::tuple<at::Tensor, at::Tensor> XLANativeFunctions::var_mean(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    const std::optional<c10::Scalar>& correction, bool keepdim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto results = tensor_methods::var_mean(
      self_tensor,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      correction ? correction->toDouble() : 1.0, keepdim);
  return std::make_tuple(bridge::AtenFromXlaTensor(std::get<0>(results)),
                         bridge::AtenFromXlaTensor(std::get<1>(results)));
}

at::Tensor XLANativeFunctions::view_as_complex_copy(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  XLA_CHECK(self.scalar_type() == at::kFloat ||
            self.scalar_type() == at::kDouble ||
            self.scalar_type() == at::kHalf)
      << "view_as_complex is only supported for half, float and double "
         "tensors, but got a tensor of scalar type: "
      << self.scalar_type();

  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      tensor_methods::view_as_complex_copy(self_tensor));
}

at::Tensor XLANativeFunctions::view_as_real_copy(const at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  XLA_CHECK(self.is_complex()) << "view_as_real is only supported for complex "
                                  "tensors, but got a tensor of scalar type: "
                               << self.scalar_type();

  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      tensor_methods::view_as_real_copy(self_tensor));
}

at::Tensor XLANativeFunctions::view_copy_symint(const at::Tensor& self,
                                                at::SymIntArrayRef shape) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::optional<at::IntArrayRef> int_shape = c10::asIntArrayRefSlowOpt(shape);
  bool input_shape_static = int_shape.has_value();
  XLATensorPtr xla_input = bridge::GetXlaTensor(self);
  bool input_has_dyn_shape = xla_input->shape().get().is_dynamic();

  XLA_CHECK(!(input_has_dyn_shape && input_shape_static))
      << "This view op has dynamic input tensor but static input shape. This "
         "behavior is currently unsupported; if the user believes this must be "
         "supported, please file a feature request against PyTorch/XLA.";
  return bridge::AtenFromXlaTensor(
      tensor_methods::view_symint(xla_input, shape));
}

at::Tensor XLANativeFunctions::where(const at::Tensor& condition,
                                     const at::Tensor& self,
                                     const at::Tensor& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  c10::MaybeOwned<at::Tensor> b_condition, b_self, b_other;
  std::tie(b_condition, b_self, b_other) =
      xla_expand_outplace(condition, self, other, "where");
  return bridge::AtenFromXlaTensor(tensor_methods::where(
      bridge::GetXlaTensor(*b_condition), bridge::GetXlaTensor(*b_self),
      bridge::GetXlaTensor(*b_other)));
}

at::Tensor& XLANativeFunctions::zero_(at::Tensor& self) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  tensor_methods::zero_(self_tensor);
  return self;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> XLANativeFunctions::_linalg_svd(
    const at::Tensor& self, bool full_matrices, bool compute_uv,
    std::optional<std::string_view> /* driver */) {
  // The optional driver string is only for CUDA with a cuSOLVER backend.
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  // As per https://pytorch.org/docs/stable/generated/torch.svd.html,
  // The second boolean argument is exactly opposite between
  // torch::svd and torch::_linalg_svd, hence the negation of full_matrices.
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto results = tensor_methods::svd(self_tensor, !full_matrices, compute_uv);
  auto u = std::get<0>(results);
  auto s = std::get<1>(results);
  auto vh = tensor_methods::transpose(std::get<2>(results), 0, 1);
  if (!compute_uv) {
    // When compute_uv is false, torch::_linalg_svd returns an empty tensor for
    // u and vh.
    u = tensor_methods::full({0}, 0, self_tensor->GetDevice(),
                             self_tensor->dtype());
    vh = tensor_methods::full({0}, 0, self_tensor->GetDevice(),
                              self_tensor->dtype());
  }
  return std::make_tuple(bridge::AtenFromXlaTensor(u),
                         bridge::AtenFromXlaTensor(s),
                         bridge::AtenFromXlaTensor(vh));
}

at::Scalar XLANativeFunctions::_local_scalar_dense(const at::Tensor& self) {
  if (DebugUtil::ExperimentEnabled("early_sync")) {
    // sync tensors in order to save computation when step is marked later.
    XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
    XLAGraphExecutor::Get()->SyncLiveTensorsGraph(&self_tensor->GetDevice(),
                                                  /*devices=*/{},
                                                  /*wait=*/true);
    TORCH_LAZY_COUNTER("EarlySyncLiveTensorsCount", 1);
  }
  return at::native::call_fallback_fn<&xla_fallback,
                                      ATEN_OP(_local_scalar_dense)>::call(self);
}

// re-use the composite kernel from core, that way we don't need to provide a
// backwards formula for native_layer_norm
std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_layer_norm(const at::Tensor& input,
                                      at::IntArrayRef normalized_shape,
                                      const std::optional<at::Tensor>& weight,
                                      const std::optional<at::Tensor>& bias,
                                      double eps) {
  return at::native::math_native_layer_norm(input, normalized_shape, weight,
                                            bias, eps);
}

// re-use the composite kernel from core, that way we don't need to provide a
// backwards formula for native_group_norm
std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::native_group_norm(const at::Tensor& input,
                                      const std::optional<at::Tensor>& weight,
                                      const std::optional<at::Tensor>& bias,
                                      int64_t N, int64_t C, int64_t HxW,
                                      int64_t group, double eps) {
  return at::native::math_group_norm(input, weight, bias, N, C, HxW, group,
                                     eps);
}

at::Tensor XLANativeFunctions::_cdist_forward(
    const at::Tensor& x1, const at::Tensor& x2, double p,
    std::optional<int64_t> compute_mode) {
  // compute_mode is ignored because the use_mm_for_euclid_dist lowering
  // (compute_mode is 0 or 1) is achieved through composite ops from
  // native pytorch.
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK(p >= 0) << "p value for the p-norm distance must be >= 0";
  return bridge::AtenFromXlaTensor(tensor_methods::cdist_forward(
      bridge::GetXlaTensor(x1), bridge::GetXlaTensor(x2), p));
}

at::Tensor XLANativeFunctions::_pdist_forward(const at::Tensor& self,
                                              double p) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK(p >= 0) << "p value for the p-norm distance must be >= 0";
  XLA_CHECK(bridge::GetXlaTensor(self)->shape().get().rank() == 2)
      << "pdist only support 2d dimension";
  return bridge::AtenFromXlaTensor(
      tensor_methods::pdist_forward(bridge::GetXlaTensor(self), p));
}

// All of the below ops correspond to CompositeExplicitAutograd kernels from
// core that call into view operators internally. These are all composite ops
// that LTC can technically re-use / get for free, but we need to
// "functionalize" them to remove the view ops before we can use them.
at::Tensor XLANativeFunctions::affine_grid_generator(const at::Tensor& theta,
                                                     at::IntArrayRef size,
                                                     bool align_corners) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      affine_grid_generator)>::call(theta, size, align_corners);
}

at::Tensor XLANativeFunctions::block_diag(at::TensorList tensors) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      block_diag)>::call(tensors);
}

at::Tensor XLANativeFunctions::_convolution(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups, bool benchmark,
    bool deterministic, bool cudnn_enabled, bool allow_tf32) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::_convolution(input, weight, bias, stride, padding,
                                    dilation, transposed, output_padding,
                                    groups, benchmark, deterministic,
                                    cudnn_enabled, allow_tf32);
  }
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      _convolution)>::call(input, weight, bias, stride, padding, dilation,
                           transposed, output_padding, groups, benchmark,
                           deterministic, cudnn_enabled, allow_tf32);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor>
XLANativeFunctions::convolution_backward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::OptionalIntArrayRef bias_sizes,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    bool transposed, at::IntArrayRef output_padding, int64_t groups,
    ::std::array<bool, 3> output_mask) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::convolution_backward(
        grad_output, input, weight, bias_sizes, stride, padding, dilation,
        transposed, output_padding, groups, output_mask);
  }
  // TODO (alanwaketan): Let's resuse
  // `at::functionalization::functionalize_aten_op` after upstream has solved
  // its issue.
  // The following is adopted from aten/src/ATen/FunctionalTensorWrapper.cpp:
  // functionalize_op_helper.
  auto func_grad_output = MaybeWrapTensorToFunctional(grad_output);
  auto func_input = MaybeWrapTensorToFunctional(input);
  auto func_weight = MaybeWrapTensorToFunctional(weight);

  auto curr_tls = c10::impl::tls_local_dispatch_key_set();
  auto tls_reenable_functionalize = c10::impl::PODLocalDispatchKeySet();
  tls_reenable_functionalize.set_included(curr_tls.included_);
  tls_reenable_functionalize.set_excluded(
      curr_tls.excluded_.remove(c10::DispatchKey::Functionalize));
  c10::impl::ForceDispatchKeyGuard guard_(tls_reenable_functionalize);
  auto results = at::native::convolution_backward(
      func_grad_output, func_input, func_weight, bias_sizes, stride, padding,
      dilation, transposed, output_padding, groups, output_mask);

  return std::make_tuple(
      at::functionalization::impl::from_functional_tensor(std::get<0>(results)),
      at::functionalization::impl::from_functional_tensor(std::get<1>(results)),
      at::functionalization::impl::from_functional_tensor(
          std::get<2>(results)));
}

at::Tensor XLANativeFunctions::count_nonzero(const at::Tensor& self,
                                             std::optional<int64_t> dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr xla_tensor = bridge::GetXlaTensor(self);
  std::vector<int64_t> dims;
  if (dim) {
    dims = torch::lazy::GetCanonicalDimensionIndices(
        {dim.value()}, xla_tensor->shape().get().rank());
  }
  return bridge::AtenFromXlaTensor(
      tensor_methods::count_nonzero(xla_tensor, dims));
}

at::Tensor XLANativeFunctions::count_nonzero(const at::Tensor& self,
                                             at::IntArrayRef dim) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr xla_tensor = bridge::GetXlaTensor(self);

  std::vector<int64_t> canonical_dims =
      torch::lazy::GetCanonicalDimensionIndices(
          dim, xla_tensor->shape().get().rank());
  std::unordered_set<int64_t> dims_set;
  for (int dim : canonical_dims) {
    XLA_CHECK(dims_set.find(dim) == dims_set.end())
        << "dim " << dim << " appears multiple times in the list of dims";
    dims_set.insert(dim);
  }

  return bridge::AtenFromXlaTensor(
      tensor_methods::count_nonzero(xla_tensor, XlaHelpers::I64List(dim)));
}

at::Tensor XLANativeFunctions::diag_embed(const at::Tensor& self,
                                          int64_t offset, int64_t dim1,
                                          int64_t dim2) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      diag_embed)>::call(self, offset, dim1, dim2);
}

at::Tensor XLANativeFunctions::embedding_symint(const at::Tensor& weight,
                                                const at::Tensor& indices,
                                                c10::SymInt padding_idx,
                                                bool scale_grad_by_freq,
                                                bool sparse) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::embedding_symint(weight, indices, padding_idx,
                                        scale_grad_by_freq, sparse);
  }

  // TODO: We need to make use of the TPU embedding core here eventually.
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::embedding(
      bridge::GetXlaTensor(weight), bridge::GetXlaTensor(indices)));
}

at::Tensor XLANativeFunctions::_euclidean_dist(const at::Tensor& x1,
                                               const at::Tensor& x2) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      _euclidean_dist)>::call(x1, x2);
}

at::Tensor XLANativeFunctions::new_empty_strided_symint(
    const at::Tensor& self, at::SymIntArrayRef size, at::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
    std::optional<at::Device> device, std::optional<bool> pin_memory) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::new_empty_strided_symint(self, size, stride, dtype,
                                                layout, device, pin_memory);
  }
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      new_empty_strided)>::call(self, size, stride, dtype, layout, device,
                                pin_memory);
}

at::Tensor XLANativeFunctions::narrow_copy_symint(const at::Tensor& self,
                                                  int64_t dim,
                                                  c10::SymInt start,
                                                  c10::SymInt length) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      narrow_copy)>::call(self, dim, start, length);
}

at::Tensor XLANativeFunctions::pixel_shuffle(const at::Tensor& self,
                                             int64_t upscale_factor) {
  return bridge::AtenFromXlaTensor(tensor_methods::pixel_shuffle(
      bridge::GetXlaTensor(self), upscale_factor));
}

at::Tensor XLANativeFunctions::pixel_unshuffle(const at::Tensor& self,
                                               int64_t downscale_factor) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      pixel_unshuffle)>::call(self, downscale_factor);
}

at::Tensor XLANativeFunctions::select_backward_symint(
    const at::Tensor& grad_output, c10::SymIntArrayRef input_sizes, int64_t dim,
    c10::SymInt index) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::select_backward_symint(grad_output, input_sizes, dim,
                                              index);
  }
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      select_backward)>::call(grad_output, input_sizes, dim, index);
}

at::Tensor XLANativeFunctions::select_symint(const at::Tensor& self,
                                             int64_t dim, c10::SymInt index) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return select_copy(self, dim, index.expect_int());
  }
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP2(
      select, int)>::call(self, dim, index);
}

at::Tensor XLANativeFunctions::slice(const at::Tensor& self, int64_t dim,
                                     std::optional<int64_t> start,
                                     std::optional<int64_t> end, int64_t step) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return slice_copy(self, dim, start, end, step);
  }
  return at::functionalization::functionalize_aten_op<ATEN_OP2(
      slice, Tensor)>::call(self, dim, start, end, step);
}

at::Tensor XLANativeFunctions::t(const at::Tensor& self) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return transpose_copy(self, 0, 1);
  }
  return at::functionalization::functionalize_aten_op<ATEN_OP(t)>::call(self);
}

at::Tensor XLANativeFunctions::_trilinear(
    const at::Tensor& i1, const at::Tensor& i2, const at::Tensor& i3,
    at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3,
    at::IntArrayRef sumdim, int64_t unroll_dim) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      _trilinear)>::call(i1, i2, i3, expand1, expand2, expand3, sumdim,
                         unroll_dim);
}

at::Tensor XLANativeFunctions::linalg_pinv(
    const at::Tensor& self, const std::optional<at::Tensor>& atol,
    const std::optional<at::Tensor>& rtol, bool hermitian) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP2(
      linalg_pinv, atol_rtol_tensor)>::call(self, atol, rtol, hermitian);
}

at::Tensor XLANativeFunctions::mvlgamma(const at::Tensor& self, int64_t p) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op<ATEN_OP(mvlgamma)>::call(
      self, p);
}

at::Tensor XLANativeFunctions::linalg_vector_norm(
    const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLA_CHECK(at::isFloatingType(self.scalar_type()))
      << "Input must be a floating type";
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(tensor_methods::linalg_vector_norm(
      self_tensor, ord,
      dim ? torch::lazy::ToVector<int64_t>(*dim)
          : torch::lazy::Iota<int64_t>(self_tensor->shape().get().rank()),
      keepdim, dtype));
}

at::Tensor XLANativeFunctions::diagonal_backward_symint(
    const at::Tensor& grad_output, at::SymIntArrayRef input_sizes,
    int64_t offset, int64_t dim1, int64_t dim2) {
  XLA_CHECK(
      !runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false));
  return at::functionalization::functionalize_aten_op_symint<ATEN_OP(
      diagonal_backward)>::call(grad_output, input_sizes, offset, dim1, dim2);
}

at::Tensor XLANativeFunctions::slice_backward(const at::Tensor& grad_output,
                                              at::IntArrayRef input_sizes,
                                              int64_t dim, int64_t start,
                                              int64_t end, int64_t step) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return at::native::slice_backward(grad_output, input_sizes, dim, start, end,
                                      step);
  }
  return at::functionalization::functionalize_aten_op<ATEN_OP(
      slice_backward)>::call(grad_output, input_sizes, dim, start, end, step);
}

at::Tensor XLANativeFunctions::permute(const at::Tensor& self,
                                       at::IntArrayRef dims) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    return permute_copy(self, dims);
  }
  return at::functionalization::functionalize_aten_op<ATEN_OP(permute)>::call(
      self, dims);
}

// For ops below, see note [Disabling Functionalization]
at::Tensor XLANativeFunctions::as_strided(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!AsStrided::StrideIsSupported(self_tensor->shape(), xsize, xstride,
                                    storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(as_strided)>::call(self, size, stride,
                                                  storage_offset);
  }
  return bridge::AtenFromXlaTensor(tensor_methods::as_strided(
      self_tensor, std::move(xsize), std::move(xstride),
      XlaHelpers::I64Optional(storage_offset)));
}

const at::Tensor& XLANativeFunctions::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  XLATensorPtr self_tensor = bridge::GetXlaTensor(self);
  auto xsize = XlaHelpers::I64List(size);
  auto xstride = XlaHelpers::I64List(stride);
  if (!AsStrided::StrideIsSupported(self_tensor->shape(), xsize, xstride,
                                    storage_offset.value_or(0))) {
    return at::native::call_fallback_fn<
        &xla_fallback, ATEN_OP(as_strided_)>::call(self, size, stride,
                                                   storage_offset);
  }
  tensor_methods::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                              XlaHelpers::I64Optional(storage_offset));
  return self;
}

at::Tensor XLANativeFunctions::diagonal(const at::Tensor& self, int64_t offset,
                                        int64_t dim1, int64_t dim2) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(
      tensor_methods::diagonal(bridge::GetXlaTensor(self), offset, dim1, dim2));
}

at::Tensor XLANativeFunctions::expand_symint(const at::Tensor& self,
                                             at::SymIntArrayRef sym_size,
                                             bool implicit) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::optional<at::IntArrayRef> size = c10::asIntArrayRefSlowOpt(sym_size);
  if (size.has_value()) {
    return bridge::AtenFromXlaTensor(tensor_methods::expand(
        bridge::GetXlaTensor(self), torch::lazy::ToVector<int64_t>(*size)));
  } else {
    // at least one of the dimension is symbolic, use the sym_int version of the
    // node
    return bridge::AtenFromXlaTensor(
        tensor_methods::expand_symint(bridge::GetXlaTensor(self), sym_size));
  }
}

at::Tensor XLANativeFunctions::view_symint(const at::Tensor& self,
                                           at::SymIntArrayRef sym_size) {
  // Dynamic shape is only supported when the functionalization is enabled.
  // So only the functionalization version of this function view_copy_symint
  // support dynamic shape.
  auto size = C10_AS_INTARRAYREF_SLOW(sym_size);
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  return bridge::AtenFromXlaTensor(tensor_methods::view(
      bridge::GetXlaTensor(self), XlaHelpers::I64List(size)));
}

}  // namespace torch_xla
