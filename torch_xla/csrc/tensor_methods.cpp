#include <algorithm>
#include <functional>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/adaptive_avg_pool2d.h"
#include "torch_xla/csrc/ops/all.h"
#include "torch_xla/csrc/ops/any.h"
#include "torch_xla/csrc/ops/arg_max.h"
#include "torch_xla/csrc/ops/arg_min.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/avg_pool_nd.h"
#include "torch_xla/csrc/ops/avg_pool_nd_backward.h"
#include "torch_xla/csrc/ops/bitwise_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/cat.h"
#include "torch_xla/csrc/ops/cholesky.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/constant_pad_nd.h"
#include "torch_xla/csrc/ops/conv2d.h"
#include "torch_xla/csrc/ops/conv2d_backward.h"
#include "torch_xla/csrc/ops/conv_transpose2d.h"
#include "torch_xla/csrc/ops/cross_replica_sum.h"
#include "torch_xla/csrc/ops/cumprod.h"
#include "torch_xla/csrc/ops/cumsum.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/dropout.h"
#include "torch_xla/csrc/ops/einsum.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/flip.h"
#include "torch_xla/csrc/ops/gather.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/hardshrink.h"
#include "torch_xla/csrc/ops/hardtanh_backward.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/ops/index_select.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/kth_value.h"
#include "torch_xla/csrc/ops/leaky_relu.h"
#include "torch_xla/csrc/ops/leaky_relu_backward.h"
#include "torch_xla/csrc/ops/linear_interpolation.h"
#include "torch_xla/csrc/ops/log_softmax.h"
#include "torch_xla/csrc/ops/masked_fill.h"
#include "torch_xla/csrc/ops/max_in_dim.h"
#include "torch_xla/csrc/ops/max_pool_nd.h"
#include "torch_xla/csrc/ops/max_pool_nd_backward.h"
#include "torch_xla/csrc/ops/mean.h"
#include "torch_xla/csrc/ops/min_in_dim.h"
#include "torch_xla/csrc/ops/native_batch_norm_backward.h"
#include "torch_xla/csrc/ops/native_batch_norm_forward.h"
#include "torch_xla/csrc/ops/nll_loss.h"
#include "torch_xla/csrc/ops/nll_loss_backward.h"
#include "torch_xla/csrc/ops/not_supported.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/prod.h"
#include "torch_xla/csrc/ops/qr.h"
#include "torch_xla/csrc/ops/randperm.h"
#include "torch_xla/csrc/ops/repeat.h"
#include "torch_xla/csrc/ops/resize.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/scatter.h"
#include "torch_xla/csrc/ops/scatter_add.h"
#include "torch_xla/csrc/ops/shrink_backward.h"
#include "torch_xla/csrc/ops/slow_conv_transpose2d_backward.h"
#include "torch_xla/csrc/ops/softmax.h"
#include "torch_xla/csrc/ops/softshrink.h"
#include "torch_xla/csrc/ops/split.h"
#include "torch_xla/csrc/ops/squeeze.h"
#include "torch_xla/csrc/ops/stack.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/ops/svd.h"
#include "torch_xla/csrc/ops/symeig.h"
#include "torch_xla/csrc/ops/threshold.h"
#include "torch_xla/csrc/ops/threshold_backward.h"
#include "torch_xla/csrc/ops/topk.h"
#include "torch_xla/csrc/ops/triangular_solve.h"
#include "torch_xla/csrc/ops/tril.h"
#include "torch_xla/csrc/ops/triu.h"
#include "torch_xla/csrc/ops/unsqueeze.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct MinMaxValues {
  ir::Value min;
  ir::Value max;
};

MinMaxValues GetMinMaxValues(const XLATensor& tensor,
                             c10::optional<at::Scalar> min,
                             c10::optional<at::Scalar> max) {
  auto shape = tensor.shape();
  XlaHelpers::MinMax min_max =
      XlaHelpers::MinMaxValues(shape.get().element_type());
  if (!min) {
    min = min_max.min;
  }
  if (!max) {
    max = min_max.max;
  }
  return {XLATensor::GetIrValueForScalar(*min, shape.get().element_type(),
                                         tensor.GetDevice()),
          XLATensor::GetIrValueForScalar(*max, shape.get().element_type(),
                                         tensor.GetDevice())};
}

void CheckRank(const XLATensor& t, xla::int64 expected_rank,
               const std::string& tag, const std::string& arg_name,
               int arg_number) {
  xla::int64 actual_rank = t.shape().get().rank();
  XLA_CHECK_EQ(actual_rank, expected_rank)
      << "Expected " << expected_rank << "-dimensional tensor, but got "
      << actual_rank << "-dimensional tensor for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

void CheckDimensionSize(const XLATensor& t, xla::int64 dim,
                        xla::int64 expected_size, const std::string& tag,
                        const std::string& arg_name, int arg_number) {
  xla::int64 dim_size = t.size(dim);
  XLA_CHECK_EQ(t.size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

std::vector<xla::int64> GetExpandDimanesions(
    const xla::Shape& shape, std::vector<xla::int64> dimensions) {
  XLA_CHECK_GE(dimensions.size(), shape.rank()) << shape;
  xla::int64 base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

// Resizes and / or checks whether a list is of the given size. The list is only
// resized if its size is 1. If it's empty, it's replaced with the provided
// default first.
std::vector<xla::int64> CheckIntList(
    tensorflow::gtl::ArraySlice<const xla::int64> list, size_t length,
    const std::string& name, std::vector<xla::int64> def = {}) {
  std::vector<xla::int64> result;
  if (list.empty()) {
    result = std::move(def);
  } else {
    result = xla::util::ToVector<xla::int64>(list);
  }
  if (result.size() == 1 && length > 1) {
    result.resize(length, result[0]);
    return result;
  }
  XLA_CHECK_EQ(result.size(), length)
      << "Invalid length for the '" << name << "' attribute";
  return result;
}

// Returns a 1-D shape for batch norm weight or bias based on the input shape.
xla::Shape BatchNormFeaturesShape(const XLATensor& input) {
  xla::PrimitiveType input_element_type =
      MakeXlaPrimitiveType(input.dtype(), &input.GetDevice());
  return xla::ShapeUtil::MakeShape(input_element_type, {input.size(1)});
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
ir::Value GetIrValueOrDefault(const XLATensor& input, at::Scalar default_value,
                              const xla::Shape& default_shape,
                              const Device& device) {
  return input.is_null() ? XLATensor::GetIrValueForScalar(default_value,
                                                          default_shape, device)
                         : input.GetIrValue();
}

void CheckIsIntegralOrPred(const xla::Shape& shape,
                           const std::string& op_name) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(shape) ||
            shape.element_type() == xla::PrimitiveType::PRED)
      << "Operator " << op_name
      << " is only supported for integer or boolean type tensors, got: "
      << shape;
}

ViewInfo CreateAsStridedViewInfo(
    const xla::Shape& input_shape,
    tensorflow::gtl::ArraySlice<const xla::int64> size,
    c10::optional<xla::int64> storage_offset) {
  xla::Shape result_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), size);
  AsStridedInfo as_strided_info;
  if (storage_offset) {
    as_strided_info.offset = *storage_offset;
  }
  return ViewInfo(ViewInfo::Type::kAsStrided, std::move(result_shape),
                  input_shape.dimensions(), std::move(as_strided_info));
}

}  // namespace

XLATensor XLATensor::__and__(const XLATensor& input, at::Scalar other) {
  CheckIsIntegralOrPred(input.shape(), "__and__");
  ir::Value other_broadcasted_ir =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__and__(const XLATensor& input, const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__and__");
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__iand__(XLATensor& input, at::Scalar other) {
  CheckIsIntegralOrPred(input.shape(), "__iand__");
  ir::Value other_broadcasted_ir =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(
      ir::ops::BitwiseAnd(input.GetIrValue(), other_broadcasted_ir));
}

void XLATensor::__iand__(XLATensor& input, const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__iand__");
  input.SetIrValue(ir::ops::BitwiseAnd(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__ilshift__(XLATensor& input, at::Scalar other) {
  input.SetIrValue(ir::ops::Lshift(input.GetIrValue(), other));
}

void XLATensor::__ilshift__(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__ior__(XLATensor& input, at::Scalar other) {
  CheckIsIntegralOrPred(input.shape(), "__ior__");
  ir::Value other_broadcasted_ir =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(
      ir::ops::BitwiseOr(input.GetIrValue(), other_broadcasted_ir));
}

void XLATensor::__ior__(XLATensor& input, const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__ior__");
  return input.SetIrValue(
      ir::ops::BitwiseOr(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__irshift__(XLATensor& input, at::Scalar other) {
  input.SetIrValue(ir::ops::Rshift(input.GetIrValue(), other));
}

void XLATensor::__irshift__(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Rshift(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__ixor__(XLATensor& input, at::Scalar other) {
  CheckIsIntegralOrPred(input.shape(), "__ixor__");
  ir::Value other_broadcasted_ir =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(
      ir::ops::BitwiseXor(input.GetIrValue(), other_broadcasted_ir));
}

void XLATensor::__ixor__(XLATensor& input, const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__ixor__");
  input.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__lshift__(const XLATensor& input, at::Scalar other) {
  return input.CreateFrom(ir::ops::Lshift(input.GetIrValue(), other));
}

XLATensor XLATensor::__lshift__(const XLATensor& input,
                                const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__or__(const XLATensor& input, const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__or__");
  return input.CreateFrom(
      ir::ops::BitwiseOr(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__or__(const XLATensor& input, at::Scalar other) {
  CheckIsIntegralOrPred(input.shape(), "__or__");
  ir::Value other_broadcasted_ir =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::BitwiseOr(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__rshift__(const XLATensor& input, at::Scalar other) {
  return input.CreateFrom(ir::ops::Rshift(input.GetIrValue(), other));
}

XLATensor XLATensor::__rshift__(const XLATensor& input,
                                const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Rshift(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__xor__(const XLATensor& input, at::Scalar other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  ir::Value other_broadcasted_ir =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::BitwiseXor(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__xor__(const XLATensor& input, const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  return input.CreateFrom(
      ir::ops::BitwiseXor(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::_adaptive_avg_pool2d(const XLATensor& input,
                                          std::vector<xla::int64> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AdaptiveAvgPool2d>(
      input.GetIrValue(), std::move(output_size)));
}

XLATensor XLATensor::_adaptive_avg_pool2d_backward(const XLATensor& grad_output,
                                                   const XLATensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool2dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

XLATensor XLATensor::abs(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Abs(input.GetIrValue()));
}

void XLATensor::abs_(XLATensor& input) {
  input.SetIrValue(ir::ops::Abs(input.GetIrValue()));
}

XLATensor XLATensor::acos(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Acos(input.GetIrValue()));
}

void XLATensor::acos_(XLATensor& input) {
  input.SetIrValue(ir::ops::Acos(input.GetIrValue()));
}

XLATensor XLATensor::add(const XLATensor& input, const XLATensor& other,
                         at::Scalar alpha) {
  ir::Value constant =
      GetIrValueForScalar(alpha, other.shape(), input.GetDevice());
  return input.CreateFrom(input.GetIrValue() + other.GetIrValue() * constant);
}

void XLATensor::add_(XLATensor& input, const XLATensor& other,
                     at::Scalar alpha) {
  ir::Value constant =
      GetIrValueForScalar(alpha, other.shape(), input.GetDevice());
  input.SetIrValue(input.GetIrValue() + other.GetIrValue() * constant);
}

XLATensor XLATensor::add(const XLATensor& input, at::Scalar other,
                         at::Scalar alpha) {
  ir::Value other_constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  ir::Value alpha_constant =
      GetIrValueForScalar(alpha, input.shape(), input.GetDevice());
  return input.CreateFrom(input.GetIrValue() + other_constant * alpha_constant);
}

void XLATensor::add_(XLATensor& input, at::Scalar other, at::Scalar alpha) {
  ir::Value other_constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  ir::Value alpha_constant =
      GetIrValueForScalar(alpha, input.shape(), input.GetDevice());
  input.SetIrValue(input.GetIrValue() + other_constant * alpha_constant);
}

void XLATensor::addcmul_(XLATensor& input, at::Scalar value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::Value constant = GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value mul = tensor1.GetIrValue() * tensor2.GetIrValue();
  input.SetIrValue(input.GetIrValue() + mul * constant);
}

XLATensor XLATensor::addcdiv(const XLATensor& input, at::Scalar value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::Value constant = GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  return input.CreateFrom(input.GetIrValue() + div * constant);
}

void XLATensor::addcdiv_(XLATensor& input, at::Scalar value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::Value constant = GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  input.SetIrValue(input.GetIrValue() + div * constant);
}

XLATensor XLATensor::addcmul(const XLATensor& input, at::Scalar value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::Value constant = GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value mul = tensor1.GetIrValue() * tensor2.GetIrValue();
  return input.CreateFrom(input.GetIrValue() + mul * constant);
}

XLATensor XLATensor::addmm(const XLATensor& input, const XLATensor& weight,
                           const XLATensor& bias) {
  return input.CreateFrom(ir::ops::AddMatMulOp(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue()));
}

XLATensor XLATensor::all(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::All>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions));
}

XLATensor XLATensor::any(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Any>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions));
}

XLATensor XLATensor::arange(at::Scalar start, at::Scalar end, at::Scalar step,
                            const Device& device, at::ScalarType scalar_type) {
  return Create(ir::ops::ARange(start, end, step, scalar_type), device,
                scalar_type);
}

XLATensor XLATensor::argmax(const XLATensor& input, xla::int64 dim,
                            bool keepdim) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMax>(input.GetIrValue(), canonical_dim, keepdim),
      at::ScalarType::Long);
}

XLATensor XLATensor::argmax(const XLATensor& input) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMax>(input.GetIrValue(), -1, false),
      at::ScalarType::Long);
}

XLATensor XLATensor::argmin(const XLATensor& input, xla::int64 dim,
                            bool keepdim) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMin>(input.GetIrValue(), canonical_dim, keepdim),
      at::ScalarType::Long);
}

XLATensor XLATensor::argmin(const XLATensor& input) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMin>(input.GetIrValue(), -1, false),
      at::ScalarType::Long);
}

XLATensor XLATensor::as_strided(const XLATensor& input,
                                std::vector<xla::int64> size,
                                c10::optional<xla::int64> storage_offset) {
  auto input_shape = input.shape();
  return input.CreateViewTensor(
      CreateAsStridedViewInfo(input_shape, size, storage_offset));
}

void XLATensor::as_strided_(XLATensor& input, std::vector<xla::int64> size,
                            c10::optional<xla::int64> storage_offset) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(ir::MakeNode<ir::ops::AsStrided>(
        input.GetIrValue(), std::move(size), storage_offset.value_or(0)));
  } else {
    auto input_shape = input.shape();
    input.SetSubView(
        CreateAsStridedViewInfo(input_shape, size, storage_offset));
  }
}

XLATensor XLATensor::asin(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Asin(input.GetIrValue()));
}

void XLATensor::asin_(XLATensor& input) {
  input.SetIrValue(ir::ops::Asin(input.GetIrValue()));
}

XLATensor XLATensor::atan(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Atan(input.GetIrValue()));
}

void XLATensor::atan_(XLATensor& input) {
  input.SetIrValue(ir::ops::Atan(input.GetIrValue()));
}

XLATensor XLATensor::atan2(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Atan2(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::atan2_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Atan2(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::avg_pool_nd(const XLATensor& input,
                                 xla::int64 spatial_dim_count,
                                 std::vector<xla::int64> kernel_size,
                                 std::vector<xla::int64> stride,
                                 std::vector<xla::int64> padding,
                                 bool ceil_mode, bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return input.CreateFrom(ir::MakeNode<ir::ops::AvgPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode, count_include_pad));
}

XLATensor XLATensor::avg_pool_nd_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    xla::int64 spatial_dim_count, std::vector<xla::int64> kernel_size,
    std::vector<xla::int64> stride, std::vector<xla::int64> padding,
    bool ceil_mode, bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::AvgPoolNdBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode,
      count_include_pad));
}

XLATensor XLATensor::bernoulli(const XLATensor& input, double probability) {
  return input.CreateFrom(ir::ops::Bernoulli(
      input.GetIrValue(),
      GetIrValueForScalar(probability, input.shape(), input.GetDevice())));
}

XLATensor XLATensor::bernoulli(const XLATensor& input) {
  return input.CreateFrom(
      ir::ops::Bernoulli(input.GetIrValue(), input.GetIrValue()));
}

void XLATensor::bernoulli_(XLATensor& input, double probability) {
  input.SetIrValue(ir::ops::Bernoulli(
      input.GetIrValue(),
      GetIrValueForScalar(probability, input.shape(), input.GetDevice())));
}

void XLATensor::bernoulli_(XLATensor& input, const XLATensor& probability) {
  input.SetIrValue(
      ir::ops::Bernoulli(input.GetIrValue(), probability.GetIrValue()));
}

XLATensor XLATensor::bmm(const XLATensor& batch1, const XLATensor& batch2) {
  // Consistent with the checks in bmm_out_or_baddbmm_.
  std::string tag = "bmm";
  CheckRank(batch1, 3, tag, "batch1", 1);
  CheckRank(batch2, 3, tag, "batch2", 2);
  xla::int64 batch_size = batch1.size(0);
  CheckDimensionSize(batch2, 0, batch_size, tag, "batch2", 2);
  xla::int64 contraction_size = batch1.size(2);
  CheckDimensionSize(batch2, 1, contraction_size, tag, "batch2", 2);
  return matmul(batch1, batch2);
}

std::vector<XLATensor> XLATensor::broadcast_tensors(
    tensorflow::gtl::ArraySlice<const XLATensor> tensors) {
  XLA_CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  ir::NodePtr node = ir::ops::BroadcastTensors(tensor_ir_values);
  return tensors.front().MakeOutputTensors(node);
}

XLATensor XLATensor::cast(const XLATensor& input, at::ScalarType dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Cast>(input.GetIrValue(), dtype), dtype);
}

XLATensor XLATensor::cat(tensorflow::gtl::ArraySlice<const XLATensor> tensors,
                         xla::int64 dim) {
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    if (xla::ShapeUtil::ElementsIn(tensor.shape()) > 0) {
      dim = XlaHelpers::GetCanonicalDimensionIndex(dim,
                                                   tensor.shape().get().rank());
      values.push_back(tensor.GetIrValue());
    }
  }
  return tensors[0].CreateFrom(ir::MakeNode<ir::ops::Cat>(values, dim));
}

XLATensor XLATensor::ceil(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Ceil(input.GetIrValue()));
}

void XLATensor::ceil_(XLATensor& input) {
  input.SetIrValue(ir::ops::Ceil(input.GetIrValue()));
}

XLATensor XLATensor::cholesky(const XLATensor& input, bool upper) {
  // Cholesky takes lower instead of upper, hence the negation.
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Cholesky>(input.GetIrValue(), !upper));
}

XLATensor XLATensor::clamp(const XLATensor& input,
                           c10::optional<at::Scalar> min,
                           c10::optional<at::Scalar> max) {
  MinMaxValues min_max = GetMinMaxValues(input, min, max);
  return input.CreateFrom(
      ir::ops::Clamp(input.GetIrValue(), min_max.min, min_max.max));
}

void XLATensor::clamp_(XLATensor& input, c10::optional<at::Scalar> min,
                       c10::optional<at::Scalar> max) {
  MinMaxValues min_max = GetMinMaxValues(input, min, max);
  input.SetIrValue(
      ir::ops::Clamp(input.GetIrValue(), min_max.min, min_max.max));
}

XLATensor XLATensor::clone(const XLATensor& input) {
  return input.CreateFrom(input.GetIrValue());
}

XLATensor XLATensor::constant_pad_nd(
    const XLATensor& input, tensorflow::gtl::ArraySlice<const xla::int64> pad,
    at::Scalar value) {
  std::vector<xla::int64> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().get().rank());
  return input.CreateFrom(ir::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value));
}

XLATensor XLATensor::conv2d(const XLATensor& input, const XLATensor& weight,
                            const XLATensor& bias,
                            std::vector<xla::int64> stride,
                            std::vector<xla::int64> padding) {
  ir::NodePtr ir_value = ir::MakeNode<ir::ops::Conv2d>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      std::move(stride), std::move(padding));
  return input.CreateFrom(ir_value);
}

XLATensor XLATensor::conv2d(const XLATensor& input, const XLATensor& weight,
                            std::vector<xla::int64> stride,
                            std::vector<xla::int64> padding) {
  ir::NodePtr ir_value =
      ir::MakeNode<ir::ops::Conv2d>(input.GetIrValue(), weight.GetIrValue(),
                                    std::move(stride), std::move(padding));
  return input.CreateFrom(ir_value);
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::conv2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    const XLATensor& weight, std::vector<xla::int64> stride,
    std::vector<xla::int64> padding) {
  ir::NodePtr node = ir::MakeNode<ir::ops::Conv2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      std::move(stride), std::move(padding));
  XLATensor grad_input = out_backprop.CreateFrom(ir::Value(node, 0));
  XLATensor grad_weight = out_backprop.CreateFrom(ir::Value(node, 1));
  XLATensor grad_bias = out_backprop.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::conv_transpose2d(const XLATensor& input,
                                      const XLATensor& weight,
                                      const XLATensor& bias,
                                      std::vector<xla::int64> stride,
                                      std::vector<xla::int64> padding) {
  ir::NodePtr node = ir::MakeNode<ir::ops::ConvTranspose2d>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      std::move(stride), std::move(padding));
  return input.CreateFrom(node);
}

XLATensor XLATensor::cos(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Cos(input.GetIrValue()));
}

void XLATensor::cos_(XLATensor& input) {
  input.SetIrValue(ir::ops::Cos(input.GetIrValue()));
}

XLATensor XLATensor::cosh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Cosh(input.GetIrValue()));
}

void XLATensor::cosh_(XLATensor& input) {
  input.SetIrValue(ir::ops::Cosh(input.GetIrValue()));
}

XLATensor XLATensor::cross(const XLATensor& input, const XLATensor& other,
                           c10::optional<xla::int64> dim) {
  return tensor_ops::Cross(input, other, dim);
}

XLATensor XLATensor::cross_replica_sum(
    const XLATensor& input, double scale,
    const std::vector<std::vector<xla::int64>>& groups) {
  return input.CreateFrom(
      ir::ops::CrossReplicaSumOp(input.GetIrValue(), scale, groups));
}

void XLATensor::cross_replica_sum_(
    XLATensor& input, double scale,
    const std::vector<std::vector<xla::int64>>& groups) {
  input.SetIrValue(
      ir::ops::CrossReplicaSumOp(input.GetIrValue(), scale, groups));
}

XLATensor XLATensor::cumprod(const XLATensor& input, xla::int64 dim,
                             c10::optional<at::ScalarType> dtype) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::CumProd>(input.GetIrValue(), canonical_dim, dtype),
      dtype);
}

XLATensor XLATensor::cumsum(const XLATensor& input, xla::int64 dim,
                            c10::optional<at::ScalarType> dtype) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::CumSum>(input.GetIrValue(), canonical_dim, dtype),
      dtype);
}

XLATensor XLATensor::diag(const XLATensor& input, xla::int64 offset) {
  xla::int64 rank = input.shape().get().rank();
  XLA_CHECK(rank == 1 || rank == 2)
      << "Invalid argument for diag: matrix or a vector expected";
  if (rank == 1) {
    return tensor_ops::MakeMatrixWithDiagonal(input, offset);
  }
  return diagonal(input, offset, /*dim1=*/-2, /*dim2=*/-1);
}

XLATensor XLATensor::diagonal(const XLATensor& input, xla::int64 offset,
                              xla::int64 dim1, xla::int64 dim2) {
  xla::int64 rank = input.shape().get().rank();
  xla::int64 canonical_dim1 =
      XlaHelpers::GetCanonicalDimensionIndex(dim1, rank);
  xla::int64 canonical_dim2 =
      XlaHelpers::GetCanonicalDimensionIndex(dim2, rank);
  return input.CreateFrom(ir::MakeNode<ir::ops::Diagonal>(
      input.GetIrValue(), offset, canonical_dim1, canonical_dim2));
}

XLATensor XLATensor::div(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(input.GetIrValue() / other.GetIrValue());
}

XLATensor XLATensor::div(const XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(input.GetIrValue() / constant);
}

void XLATensor::div_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(input.GetIrValue() / other.GetIrValue());
}

void XLATensor::div_(XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(input.GetIrValue() / constant);
}

XLATensor XLATensor::dropout(const XLATensor& input, double p) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Dropout>(input.GetIrValue(), p));
}

void XLATensor::dropout_(XLATensor& input, double p) {
  input.SetIrValue(ir::MakeNode<ir::ops::Dropout>(input.GetIrValue(), p));
}

XLATensor XLATensor::eq(const XLATensor& input, at::Scalar other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

void XLATensor::eq_(XLATensor& input, at::Scalar other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::eq, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::eq(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

void XLATensor::eq_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::eq, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::einsum(
    const std::string& equation,
    tensorflow::gtl::ArraySlice<const XLATensor> tensors) {
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  XLA_CHECK_EQ(tensors.size(), 2);
  return tensors[0].CreateFrom(
      ir::MakeNode<ir::ops::Einsum>(equation, tensor_ir_values));
}

XLATensor XLATensor::elu(const XLATensor& input, at::Scalar alpha,
                         at::Scalar scale, at::Scalar input_scale) {
  return input.CreateFrom(
      ir::ops::Elu(input.GetIrValue(), alpha, scale, input_scale));
}

void XLATensor::elu_(XLATensor& input, at::Scalar alpha, at::Scalar scale,
                     at::Scalar input_scale) {
  input.SetIrValue(ir::ops::Elu(input.GetIrValue(), alpha, scale, input_scale));
}

XLATensor XLATensor::elu_backward(const XLATensor& grad_output,
                                  at::Scalar alpha, at::Scalar scale,
                                  at::Scalar input_scale,
                                  const XLATensor& output) {
  return grad_output.CreateFrom(ir::ops::EluBackward(grad_output.GetIrValue(),
                                                     output.GetIrValue(), alpha,
                                                     scale, input_scale));
}

XLATensor XLATensor::embedding_dense_backward(const XLATensor& grad_output,
                                              const XLATensor& indices,
                                              xla::int64 num_weights,
                                              xla::int64 padding_idx,
                                              bool scale_grad_by_freq) {
  return tensor_ops::EmbeddingDenseBackward(grad_output, indices, num_weights,
                                            padding_idx, scale_grad_by_freq);
}

XLATensor XLATensor::erf(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erf(input.GetIrValue()));
}

void XLATensor::erf_(XLATensor& input) {
  input.SetIrValue(ir::ops::Erf(input.GetIrValue()));
}

XLATensor XLATensor::erfc(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erfc(input.GetIrValue()));
}

void XLATensor::erfc_(XLATensor& input) {
  input.SetIrValue(ir::ops::Erfc(input.GetIrValue()));
}

XLATensor XLATensor::erfinv(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erfinv(input.GetIrValue()));
}

void XLATensor::erfinv_(XLATensor& input) {
  input.SetIrValue(ir::ops::Erfinv(input.GetIrValue()));
}

XLATensor XLATensor::exp(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Exp(input.GetIrValue()));
}

void XLATensor::exp_(XLATensor& input) {
  input.SetIrValue(ir::ops::Exp(input.GetIrValue()));
}

XLATensor XLATensor::expand(const XLATensor& input,
                            std::vector<xla::int64> size) {
  auto input_shape = input.shape();
  return input.CreateFrom(ir::MakeNode<ir::ops::Expand>(
      input.GetIrValue(),
      GetExpandDimanesions(input_shape.get(), std::move(size))));
}

XLATensor XLATensor::expm1(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Expm1(input.GetIrValue()));
}

void XLATensor::expm1_(XLATensor& input) {
  input.SetIrValue(ir::ops::Expm1(input.GetIrValue()));
}

XLATensor XLATensor::eye(xla::int64 lines, xla::int64 cols,
                         const Device& device, at::ScalarType element_type) {
  return XLATensor::Create(
      ir::ops::Identity(lines, cols,
                        MakeXlaPrimitiveType(element_type, &device)),
      device, element_type);
}

void XLATensor::fill_(XLATensor& input, at::Scalar value) {
  ir::Value constant =
      GetIrValueForScalar(value, input.shape(), input.GetDevice());
  input.SetIrValue(std::move(constant));
}

XLATensor XLATensor::flip(const XLATensor& input,
                          tensorflow::gtl::ArraySlice<const xla::int64> dims) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Flip>(
      input.GetIrValue(), XlaHelpers::GetCanonicalDimensionIndices(
                              dims, input.shape().get().rank())));
}

XLATensor XLATensor::floor(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Floor(input.GetIrValue()));
}

void XLATensor::floor_(XLATensor& input) {
  input.SetIrValue(ir::ops::Floor(input.GetIrValue()));
}

XLATensor XLATensor::fmod(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::fmod(const XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), constant));
}

void XLATensor::fmod_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::fmod_(XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(ir::ops::Fmod(input.GetIrValue(), constant));
}

XLATensor XLATensor::frac(const XLATensor& input) {
  return input.CreateFrom(ir::ops::FracOp(input.GetIrValue()));
}

void XLATensor::frac_(XLATensor& input) {
  input.SetIrValue(ir::ops::FracOp(input.GetIrValue()));
}

XLATensor XLATensor::full(tensorflow::gtl::ArraySlice<const xla::int64> size,
                          at::Scalar fill_value, const Device& device,
                          at::ScalarType scalar_type) {
  xla::Shape shape = MakeArrayShapeFromDimensions(
      size, MakeXlaPrimitiveType(scalar_type, &device), device.hw_type);
  return Create(GetIrValueForScalar(fill_value, shape, device), device,
                scalar_type);
}

XLATensor XLATensor::full_like(const XLATensor& input, at::Scalar fill_value,
                               const Device& device,
                               c10::optional<at::ScalarType> scalar_type) {
  xla::Shape tensor_shape = input.shape();
  if (scalar_type) {
    tensor_shape.set_element_type(MakeXlaPrimitiveType(*scalar_type, &device));
  } else {
    scalar_type = input.dtype();
  }
  return input.CreateFrom(GetIrValueForScalar(fill_value, tensor_shape, device),
                          device, *scalar_type);
}

XLATensor XLATensor::gather(const XLATensor& input, xla::int64 dim,
                            const XLATensor& index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Gather>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue()));
}

XLATensor XLATensor::ge(const XLATensor& input, at::Scalar other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

void XLATensor::ge_(XLATensor& input, at::Scalar other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::ge, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::ge(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

void XLATensor::ge_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::ge, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::gt(const XLATensor& input, at::Scalar other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

void XLATensor::gt_(XLATensor& input, at::Scalar other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::gt, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::gt(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

void XLATensor::gt_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::gt, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::index(const XLATensor& input,
                           tensorflow::gtl::ArraySlice<const XLATensor> indices,
                           xla::int64 start_dim) {
  return IndexByTensors(input, indices, start_dim);
}

XLATensor XLATensor::index_add(const XLATensor& input, xla::int64 dim,
                               const XLATensor& index,
                               const XLATensor& source) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexAdd(input, canonical_dim, index, source));
}

void XLATensor::index_add_(XLATensor& input, xla::int64 dim,
                           const XLATensor& index, const XLATensor& source) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexAdd(input, canonical_dim, index, source));
}

XLATensor XLATensor::index_copy(const XLATensor& input, xla::int64 dim,
                                const XLATensor& index,
                                const XLATensor& source) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexCopy(input, canonical_dim, index, source));
}

void XLATensor::index_copy_(XLATensor& input, xla::int64 dim,
                            const XLATensor& index, const XLATensor& source) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexCopy(input, canonical_dim, index, source));
}

XLATensor XLATensor::index_fill(const XLATensor& input, xla::int64 dim,
                                const XLATensor& index, at::Scalar value) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

XLATensor XLATensor::index_fill(const XLATensor& input, xla::int64 dim,
                                const XLATensor& index,
                                const XLATensor& value) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

void XLATensor::index_fill_(XLATensor& input, xla::int64 dim,
                            const XLATensor& index, const XLATensor& value) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

void XLATensor::index_fill_(XLATensor& input, xla::int64 dim,
                            const XLATensor& index, at::Scalar value) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

XLATensor XLATensor::index_put(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const XLATensor> indices, xla::int64 start_dim,
    const XLATensor& values, bool accumulate,
    tensorflow::gtl::ArraySlice<const xla::int64> result_permutation) {
  return input.CreateFrom(IndexPutByTensors(input, indices, start_dim, values,
                                            accumulate, result_permutation));
}

void XLATensor::index_put_(
    XLATensor& input, const XLATensor& canonical_base,
    tensorflow::gtl::ArraySlice<const XLATensor> indices, xla::int64 start_dim,
    const XLATensor& values, bool accumulate,
    tensorflow::gtl::ArraySlice<const xla::int64> result_permutation) {
  input.SetIrValue(IndexPutByTensors(canonical_base, indices, start_dim, values,
                                     accumulate, result_permutation));
}

XLATensor XLATensor::index_select(const XLATensor& input, xla::int64 dim,
                                  const XLATensor& index) {
  ir::Value index_value = EnsureRank1(index.GetIrValue());
  return input.CreateFrom(ir::MakeNode<ir::ops::IndexSelect>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index_value));
}

XLATensor XLATensor::kl_div_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const XLATensor& target,
                                     xla::int64 reduction) {
  return tensor_ops::KlDivBackward(grad_output, input, target, reduction);
}

std::tuple<XLATensor, XLATensor> XLATensor::kthvalue(const XLATensor& input,
                                                     xla::int64 k,
                                                     xla::int64 dim,
                                                     bool keepdim) {
  ir::NodePtr node = ir::MakeNode<ir::ops::KthValue>(
      input.GetIrValue(), k,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::le(const XLATensor& input, at::Scalar other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

void XLATensor::le_(XLATensor& input, at::Scalar other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::le, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::le(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

void XLATensor::le_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::le, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::hardshrink(const XLATensor& input, at::Scalar lambda) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Hardshrink>(input.GetIrValue(), lambda));
}

XLATensor XLATensor::hardshrink_backward(const XLATensor& grad_out,
                                         const XLATensor& input,
                                         at::Scalar lambda) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ShrinkBackward>(
      ir::OpKind(at::aten::hardshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

XLATensor XLATensor::hardtanh_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       at::Scalar min_val, at::Scalar max_val) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::HardtanhBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), min_val, max_val));
}

XLATensor XLATensor::leaky_relu(const XLATensor& input, double negative_slope) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LeakyRelu>(input.GetIrValue(), negative_slope));
}

XLATensor XLATensor::leaky_relu_backward(const XLATensor& grad_output,
                                         const XLATensor& input,
                                         double negative_slope) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::LeakyReluBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), negative_slope));
}

void XLATensor::leaky_relu_(XLATensor& input, double negative_slope) {
  input.SetIrValue(
      ir::MakeNode<ir::ops::LeakyRelu>(input.GetIrValue(), negative_slope));
}

XLATensor XLATensor::log(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Log(input.GetIrValue()));
}

void XLATensor::log_(XLATensor& input) {
  input.SetIrValue(ir::ops::Log(input.GetIrValue()));
}

XLATensor XLATensor::log_base(const XLATensor& input, ir::OpKind op,
                              double base) {
  return input.CreateFrom(ir::ops::LogBase(input.GetIrValue(), op, base));
}

void XLATensor::log_base_(XLATensor& input, ir::OpKind op, double base) {
  input.SetIrValue(ir::ops::LogBase(input.GetIrValue(), op, base));
}

XLATensor XLATensor::log_sigmoid(const XLATensor& input) {
  return input.CreateFrom(std::get<0>(ir::ops::LogSigmoid(input.GetIrValue())));
}

std::tuple<XLATensor, XLATensor> XLATensor::log_sigmoid_forward(
    const XLATensor& input) {
  auto output_and_buffer = ir::ops::LogSigmoid(input.GetIrValue());
  return std::make_tuple(input.CreateFrom(std::get<0>(output_and_buffer)),
                         input.CreateFrom(std::get<1>(output_and_buffer)));
}

XLATensor XLATensor::log_sigmoid_backward(const XLATensor& grad_output,
                                          const XLATensor& input,
                                          const XLATensor& buffer) {
  return grad_output.CreateFrom(ir::ops::LogSigmoidBackward(
      grad_output.GetIrValue(), input.GetIrValue(), buffer.GetIrValue()));
}

XLATensor XLATensor::log_softmax(const XLATensor& input, xla::int64 dim,
                                 c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LogSoftmax>(input.GetIrValue(),
                                        XlaHelpers::GetCanonicalDimensionIndex(
                                            dim, input.shape().get().rank()),
                                        dtype),
      dtype);
}

XLATensor XLATensor::log_softmax_backward(const XLATensor& grad_output,
                                          const XLATensor& output,
                                          xla::int64 dim) {
  return grad_output.CreateFrom(ir::ops::LogSoftmaxBackwardOp(
      grad_output.GetIrValue(), output.GetIrValue(), dim));
}

XLATensor XLATensor::log1p(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Log1p(input.GetIrValue()));
}

void XLATensor::log1p_(XLATensor& input) {
  input.SetIrValue(ir::ops::Log1p(input.GetIrValue()));
}

XLATensor XLATensor::lt(const XLATensor& input, at::Scalar other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

void XLATensor::lt_(XLATensor& input, at::Scalar other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::lt, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::lt(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

void XLATensor::lt_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::lt, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::masked_fill(const XLATensor& input, const XLATensor& mask,
                                 at::Scalar value) {
  // Expand mask to be the same size as input.
  ir::NodePtr expanded_mask = ir::MakeNode<ir::ops::Expand>(
      mask.GetIrValue(), input.shape().get().dimensions());
  return input.CreateFrom(ir::MakeNode<ir::ops::MaskedFill>(
      input.GetIrValue(), expanded_mask, value));
}

void XLATensor::masked_fill_(XLATensor& input, const XLATensor& mask,
                             at::Scalar value) {
  // Expand mask to be the same size as input.
  ir::NodePtr expanded_mask = ir::MakeNode<ir::ops::Expand>(
      mask.GetIrValue(), input.shape().get().dimensions());
  input.SetIrValue(ir::MakeNode<ir::ops::MaskedFill>(input.GetIrValue(),
                                                     expanded_mask, value));
}

XLATensor XLATensor::matmul(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::MatMul(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::max(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(ir::ops::Max(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::max(const XLATensor& input) {
  return input.CreateFrom(ir::ops::MaxUnary(input.GetIrValue()), input.dtype());
}

std::tuple<XLATensor, XLATensor> XLATensor::max(const XLATensor& input,
                                                xla::int64 dim, bool keepdim) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::max_pool_nd(const XLATensor& input,
                                 xla::int64 spatial_dim_count,
                                 std::vector<xla::int64> kernel_size,
                                 std::vector<xla::int64> stride,
                                 std::vector<xla::int64> padding,
                                 bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return input.CreateFrom(ir::MakeNode<ir::ops::MaxPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode));
}

XLATensor XLATensor::max_pool_nd_backward(const XLATensor& out_backprop,
                                          const XLATensor& input,
                                          xla::int64 spatial_dim_count,
                                          std::vector<xla::int64> kernel_size,
                                          std::vector<xla::int64> stride,
                                          std::vector<xla::int64> padding,
                                          bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::MaxPoolNdBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding),
      ceil_mode));
}

XLATensor XLATensor::mean(const XLATensor& input,
                          std::vector<xla::int64> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Mean>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions, dtype),
      dtype);
}

XLATensor XLATensor::min(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(ir::ops::Min(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::min(const XLATensor& input) {
  return input.CreateFrom(ir::ops::MinUnary(input.GetIrValue()), input.dtype());
}

std::tuple<XLATensor, XLATensor> XLATensor::min(const XLATensor& input,
                                                xla::int64 dim, bool keepdim) {
  xla::int64 canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MinInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::mm(const XLATensor& input, const XLATensor& weight) {
  return input.CreateFrom(
      ir::ops::Dot(input.GetIrValue(), weight.GetIrValue()));
}

XLATensor XLATensor::mul(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(input.GetIrValue() * other.GetIrValue());
}

XLATensor XLATensor::mul(const XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(input.GetIrValue() * constant);
}

void XLATensor::mul_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(input.GetIrValue() * other.GetIrValue());
}

void XLATensor::mul_(XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(input.GetIrValue() * constant);
}

XLATensor XLATensor::narrow(const XLATensor& input, xla::int64 dim,
                            xla::int64 start, xla::int64 length) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  xla::Shape narrow_shape = input_shape;
  narrow_shape.set_dimensions(dim, length);

  ViewInfo::Type view_type = (xla::ShapeUtil::ElementsIn(input_shape) ==
                              xla::ShapeUtil::ElementsIn(narrow_shape))
                                 ? ViewInfo::Type::kReshape
                                 : ViewInfo::Type::kNarrow;
  ViewInfo view_info(view_type, std::move(narrow_shape),
                     input_shape.get().dimensions());
  view_info.indices[dim] = XlaHelpers::GetCanonicalPosition(
      input_shape.get().dimensions(), dim, start);
  return input.CreateViewTensor(std::move(view_info));
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::native_batch_norm(
    const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
    XLATensor& running_mean, XLATensor& running_var, bool training,
    double momentum, double eps) {
  xla::Shape features_shape = BatchNormFeaturesShape(input);
  ir::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  ir::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input.GetDevice());
  ir::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input.GetDevice());
  ir::Value running_var_value =
      GetIrValueOrDefault(running_var, 1, features_shape, input.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormForward>(
      input.GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, eps);
  XLATensor output = input.CreateFrom(ir::Value(node, 0));
  XLATensor mean;
  XLATensor variance_inverse;
  if (training) {
    mean = input.CreateFrom(ir::Value(node, 1));
    XLATensor variance = input.CreateFrom(ir::Value(node, 2));
    variance_inverse = input.CreateFrom(ir::Value(node, 3));
    if (!running_mean.is_null()) {
      running_mean.SetIrValue(ir::MakeNode<ir::ops::LinearInterpolation>(
          mean.GetIrValue(), running_mean.GetIrValue(), momentum));
    }
    if (!running_var.is_null()) {
      running_var.SetIrValue(ir::MakeNode<ir::ops::LinearInterpolation>(
          variance.GetIrValue(), running_var.GetIrValue(), momentum));
    }
  }
  return std::make_tuple(std::move(output), std::move(mean),
                         std::move(variance_inverse));
}

std::tuple<XLATensor, XLATensor, XLATensor>
XLATensor::native_batch_norm_backward(const XLATensor& grad_out,
                                      const XLATensor& input,
                                      const XLATensor& weight,
                                      const XLATensor& save_mean,
                                      const XLATensor& save_invstd,
                                      bool training, double eps) {
  xla::Shape features_shape = BatchNormFeaturesShape(input);
  ir::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormBackward>(
      grad_out.GetIrValue(), input.GetIrValue(), weight_value,
      save_mean.GetIrValue(), save_invstd.GetIrValue(), training, eps);
  XLATensor grad_input = input.CreateFrom(ir::Value(node, 0));
  XLATensor grad_weight = input.CreateFrom(ir::Value(node, 1));
  XLATensor grad_bias = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::ne(const XLATensor& input, at::Scalar other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

void XLATensor::ne_(XLATensor& input, at::Scalar other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::ne, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::ne(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

void XLATensor::ne_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::ne, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::neg(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Neg(input.GetIrValue()));
}

void XLATensor::neg_(XLATensor& input) {
  input.SetIrValue(ir::ops::Neg(input.GetIrValue()));
}

XLATensor XLATensor::nll_loss(const XLATensor& input, const XLATensor& target,
                              int ignore_index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLoss>(
      input.GetIrValue(), target.GetIrValue(), ignore_index));
}

XLATensor XLATensor::nll_loss_backward(const XLATensor& input,
                                       const XLATensor& target,
                                       int ignore_index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLossBackward>(
      input.GetIrValue(), target.GetIrValue(), ignore_index));
}

XLATensor XLATensor::not_supported(std::string description, xla::Shape shape,
                                   const Device& device) {
  return Create(ir::MakeNode<ir::ops::NotSupported>(std::move(description),
                                                    std::move(shape)),
                device);
}

XLATensor XLATensor::norm(const XLATensor& input, c10::optional<at::Scalar> p,
                          c10::optional<at::ScalarType> dtype,
                          at::IntArrayRef dim, bool keepdim) {
  auto canonical_dims = XlaHelpers::GetCanonicalDimensionIndices(
      XlaHelpers::I64List(dim), input.shape().get().rank());
  return input.CreateFrom(
      ir::ops::Norm(input.GetIrValue(), p, dtype, canonical_dims, keepdim));
}

XLATensor XLATensor::permute(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> dims) {
  auto input_shape = input.shape();
  ViewInfo view_info(
      ViewInfo::Type::kPermute,
      xla::util::ToVector<xla::int64>(input_shape.get().dimensions()),
      XlaHelpers::GetCanonicalDimensionIndices(dims, input_shape.get().rank()),
      input_shape.get().element_type());
  return input.CreateViewTensor(std::move(view_info));
}

XLATensor XLATensor::pow(const XLATensor& input, at::Scalar exponent) {
  ir::Value exponent_node =
      GetIrValueForScalar(exponent, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

XLATensor XLATensor::pow(const XLATensor& input, const XLATensor& exponent) {
  return input.CreateFrom(
      ir::ops::Pow(input.GetIrValue(), exponent.GetIrValue()));
}

XLATensor XLATensor::pow(at::Scalar input, const XLATensor& exponent) {
  ir::Value input_node =
      GetIrValueForScalar(input, exponent.shape(), exponent.GetDevice());
  return exponent.CreateFrom(ir::ops::Pow(input_node, exponent.GetIrValue()));
}

void XLATensor::pow_(XLATensor& input, at::Scalar exponent) {
  ir::Value exponent_node =
      GetIrValueForScalar(exponent, input.shape(), input.GetDevice());
  input.SetIrValue(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

void XLATensor::pow_(XLATensor& input, const XLATensor& exponent) {
  input.SetIrValue(ir::ops::Pow(input.GetIrValue(), exponent.GetIrValue()));
}

XLATensor XLATensor::prod(const XLATensor& input,
                          std::vector<xla::int64> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Prod>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions, dtype),
      dtype);
}

std::tuple<XLATensor, XLATensor> XLATensor::qr(const XLATensor& input,
                                               bool some) {
  ir::NodePtr node = ir::MakeNode<ir::ops::QR>(input.GetIrValue(), some);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

XLATensor XLATensor::randperm(xla::int64 n, const Device& device,
                              at::ScalarType element_type) {
  xla::PrimitiveType xla_element_type =
      MakeXlaPrimitiveType(element_type, &device);
  return Create(ir::MakeNode<ir::ops::Randperm>(n, xla_element_type), device,
                element_type);
}

XLATensor XLATensor::reciprocal(const XLATensor& input) {
  return input.CreateFrom(ir::ops::ReciprocalOp(input.GetIrValue()));
}

void XLATensor::reciprocal_(XLATensor& input) {
  input.SetIrValue(ir::ops::ReciprocalOp(input.GetIrValue()));
}

XLATensor XLATensor::relu(const XLATensor& input) {
  return input.CreateFrom(ir::ops::ReluOp(input.GetIrValue()));
}

void XLATensor::relu_(XLATensor& input) {
  input.SetIrValue(ir::ops::ReluOp(input.GetIrValue()));
}

XLATensor XLATensor::remainder(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Remainder(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::remainder(const XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Remainder(input.GetIrValue(), constant));
}

void XLATensor::remainder_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Remainder(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::remainder_(XLATensor& input, at::Scalar other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  input.SetIrValue(ir::ops::Remainder(input.GetIrValue(), constant));
}

XLATensor XLATensor::repeat(const XLATensor& input,
                            std::vector<xla::int64> repeats) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Repeat>(input.GetIrValue(), std::move(repeats)));
}

XLATensor XLATensor::reshape(const XLATensor& input,
                             std::vector<xla::int64> output_size) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::View>(input.GetIrValue(), std::move(output_size)));
}

void XLATensor::resize_(XLATensor& input, std::vector<xla::int64> size) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(
        ir::MakeNode<ir::ops::Resize>(input.GetIrValue(), std::move(size)));
  } else {
    auto input_shape = input.shape();
    xla::Shape resize_shape =
        xla::ShapeUtil::MakeShape(input_shape.get().element_type(), size);
    ViewInfo view_info(ViewInfo::Type::kResize, std::move(resize_shape),
                       input_shape.get().dimensions());
    input.SetSubView(std::move(view_info));
  }
}

XLATensor XLATensor::rsqrt(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Rsqrt(input.GetIrValue()));
}

void XLATensor::rsqrt_(XLATensor& input) {
  input.SetIrValue(ir::ops::Rsqrt(input.GetIrValue()));
}

XLATensor XLATensor::rsub(const XLATensor& input, const XLATensor& other,
                          at::Scalar alpha) {
  ir::Value alpha_xla =
      GetIrValueForScalar(alpha, other.shape(), other.GetDevice());
  return input.CreateFrom(other.GetIrValue() - alpha_xla * input.GetIrValue());
}

XLATensor XLATensor::rsub(const XLATensor& input, at::Scalar other,
                          at::Scalar alpha) {
  ir::Value alpha_xla =
      GetIrValueForScalar(alpha, input.shape(), input.GetDevice());
  ir::Value other_xla =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(other_xla - alpha_xla * input.GetIrValue());
}

void XLATensor::copy_(XLATensor& input, XLATensor& src) {
  if (input.GetDevice() == src.GetDevice()) {
    if (input.dtype() == src.dtype()) {
      input.SetIrValue(src.GetIrValue());
    } else {
      input.SetIrValue(
          ir::MakeNode<ir::ops::Cast>(src.GetIrValue(), input.dtype()));
    }
  } else {
    XLATensor new_tensor = src.CopyTensorToDevice(input.GetDevice());
    if (input.dtype() != src.dtype()) {
      new_tensor.SetScalarType(input.dtype());
    }
    std::swap(input, new_tensor);
  }
}

void XLATensor::scatter_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, const XLATensor& src) {
  input.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void XLATensor::scatter_add_(XLATensor& input, xla::int64 dim,
                             const XLATensor& index, const XLATensor& src) {
  input.SetIrValue(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::scatter(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, const XLATensor& src) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::scatter_add(const XLATensor& input, xla::int64 dim,
                                 const XLATensor& index, const XLATensor& src) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void XLATensor::scatter_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, at::Scalar value) {
  ir::Value constant =
      GetIrValueForScalar(value, input.shape(), input.GetDevice());
  input.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), constant,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::scatter(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, at::Scalar value) {
  ir::Value constant =
      GetIrValueForScalar(value, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), constant,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::select(const XLATensor& input, xla::int64 dim,
                            xla::int64 index) {
  return tensor_ops::Select(input, dim, index);
}

XLATensor XLATensor::sigmoid(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sigmoid(input.GetIrValue()));
}

void XLATensor::sigmoid_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sigmoid(input.GetIrValue()));
}

XLATensor XLATensor::sigmoid_backward(const XLATensor& grad_output,
                                      const XLATensor& output) {
  return grad_output.CreateFrom(
      ir::ops::SigmoidBackward(grad_output.GetIrValue(), output.GetIrValue()));
}

XLATensor XLATensor::sign(const XLATensor& input) {
  return input.CreateFrom(ir::ops::SignOp(input.GetIrValue()));
}

void XLATensor::sign_(XLATensor& input) {
  input.SetIrValue(ir::ops::SignOp(input.GetIrValue()));
}

XLATensor XLATensor::sin(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sin(input.GetIrValue()));
}

void XLATensor::sin_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sin(input.GetIrValue()));
}

XLATensor XLATensor::sinh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sinh(input.GetIrValue()));
}

void XLATensor::sinh_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sinh(input.GetIrValue()));
}

XLATensor XLATensor::slice(const XLATensor& input, xla::int64 dim,
                           xla::int64 start, xla::int64 end, xla::int64 step) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                           start);
  end = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                         end);
  step = std::min(step, end - start);

  SelectInfo select = {dim, start, end, step};
  ViewInfo view_info(ViewInfo::Type::kSelect, input_shape, std::move(select));
  return input.CreateViewTensor(std::move(view_info));
}

XLATensor XLATensor::slow_conv_transpose2d(const XLATensor& input,
                                           const XLATensor& weight,
                                           std::vector<xla::int64> stride,
                                           std::vector<xla::int64> padding) {
  ir::NodePtr node = ir::MakeNode<ir::ops::ConvTranspose2d>(
      input.GetIrValue(), weight.GetIrValue(), std::move(stride),
      std::move(padding));
  return input.CreateFrom(node);
}

std::tuple<XLATensor, XLATensor, XLATensor>
XLATensor::slow_conv_transpose2d_backward(const XLATensor& out_backprop,
                                          const XLATensor& input,
                                          const XLATensor& weight,
                                          std::vector<xla::int64> stride,
                                          std::vector<xla::int64> padding) {
  ir::NodePtr node = ir::MakeNode<ir::ops::ConvTranspose2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      std::move(stride), std::move(padding));
  XLATensor grad_input = out_backprop.CreateFrom(ir::Value(node, 0));
  XLATensor grad_weight = out_backprop.CreateFrom(ir::Value(node, 1));
  XLATensor grad_bias = out_backprop.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::smooth_l1_loss(const XLATensor& input,
                                    const XLATensor& target,
                                    xla::int64 reduction) {
  return tensor_ops::SmoothL1Loss(input, target, reduction);
}

XLATensor XLATensor::smooth_l1_loss_backward(const XLATensor& grad_output,
                                             const XLATensor& input,
                                             const XLATensor& target,
                                             xla::int64 reduction) {
  return tensor_ops::SmoothL1LossBackward(grad_output, input, target,
                                          reduction);
}

XLATensor XLATensor::softmax(const XLATensor& input, xla::int64 dim,
                             c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Softmax>(input.GetIrValue(),
                                     XlaHelpers::GetCanonicalDimensionIndex(
                                         dim, input.shape().get().rank()),
                                     dtype),
      dtype);
}

XLATensor XLATensor::softmax_backward(const XLATensor& grad_output,
                                      const XLATensor& output, xla::int64 dim) {
  return grad_output.CreateFrom(ir::ops::SoftmaxBackwardOp(
      grad_output.GetIrValue(), output.GetIrValue(), dim));
}

XLATensor XLATensor::softplus(const XLATensor& input, at::Scalar beta,
                              at::Scalar threshold) {
  return tensor_ops::Softplus(input, beta, threshold);
}

XLATensor XLATensor::softplus_backward(const XLATensor& grad_output,
                                       const XLATensor& input, at::Scalar beta,
                                       at::Scalar threshold,
                                       const XLATensor& output) {
  return tensor_ops::SoftplusBackward(grad_output, input, beta, threshold,
                                      output);
}

XLATensor XLATensor::softshrink(const XLATensor& input, at::Scalar lambda) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Softshrink>(input.GetIrValue(), lambda));
}

XLATensor XLATensor::softshrink_backward(const XLATensor& grad_out,
                                         const XLATensor& input,
                                         at::Scalar lambda) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ShrinkBackward>(
      ir::OpKind(at::aten::softshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

std::vector<XLATensor> XLATensor::split(const XLATensor& input,
                                        xla::int64 split_size, xla::int64 dim) {
  auto input_shape = input.shape();
  int split_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  xla::int64 dim_size = input_shape.get().dimensions(split_dim);
  if (split_size == 0) {
    // Deal with 0 split size, it's a corner case which is only allowed when the
    // dimension size is 0 as well.
    XLA_CHECK_EQ(dim_size, 0);
    xla::Literal literal(input_shape.get());
    return {
        input.CreateFrom(ir::MakeNode<ir::ops::Constant>(std::move(literal)))};
  }
  std::vector<xla::int64> split_sizes;
  for (; dim_size > 0; dim_size -= split_size) {
    split_sizes.push_back(std::min<xla::int64>(dim_size, split_size));
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_sizes), split_dim);
  return input.MakeOutputTensors(node);
}

std::vector<XLATensor> XLATensor::split_with_sizes(
    const XLATensor& input, std::vector<xla::int64> split_size,
    xla::int64 dim) {
  auto input_shape = input.shape();
  int split_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_size), split_dim);
  return input.MakeOutputTensors(node);
}

XLATensor XLATensor::sqrt(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sqrt(input.GetIrValue()));
}

void XLATensor::sqrt_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sqrt(input.GetIrValue()));
}

XLATensor XLATensor::squeeze(const XLATensor& input) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

XLATensor XLATensor::squeeze(const XLATensor& input, xla::int64 dim) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

void XLATensor::squeeze_(XLATensor& input) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void XLATensor::squeeze_(XLATensor& input, xla::int64 dim) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::stack(tensorflow::gtl::ArraySlice<const XLATensor> tensors,
                           xla::int64 dim) {
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  xla::int64 canonical_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, tensors.front().shape().get().rank() + 1);
  return tensors[0].CreateFrom(
      ir::MakeNode<ir::ops::Stack>(values, canonical_dim));
}

XLATensor XLATensor::sub(const XLATensor& input, const XLATensor& other,
                         at::Scalar alpha) {
  ir::Value constant =
      GetIrValueForScalar(alpha, other.shape(), other.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant);
}

XLATensor XLATensor::sum(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions,
                         c10::optional<at::ScalarType> dtype) {
  if ((at::isIntegralType(input.dtype()) || input.dtype() == at::kBool) &&
      !dtype) {
    dtype = at::ScalarType::Long;
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Sum>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions, dtype),
      dtype);
}

void XLATensor::sub_(XLATensor& input, const XLATensor& other,
                     at::Scalar alpha) {
  ir::Value constant =
      GetIrValueForScalar(alpha, other.shape(), other.GetDevice());
  input.SetIrValue(input.GetIrValue() - other.GetIrValue() * constant);
}

XLATensor XLATensor::sub(const XLATensor& input, at::Scalar other,
                         at::Scalar alpha) {
  ir::Value other_constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  ir::Value alpha_constant =
      GetIrValueForScalar(alpha, input.shape(), input.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant);
}

void XLATensor::sub_(XLATensor& input, at::Scalar other, at::Scalar alpha) {
  ir::Value other_constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  ir::Value alpha_constant =
      GetIrValueForScalar(alpha, input.shape(), input.GetDevice());
  input.SetIrValue(input.GetIrValue() - other_constant * alpha_constant);
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::svd(
    const XLATensor& input, bool some, bool compute_uv) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::SVD>(input.GetIrValue(), some, compute_uv);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)),
                         input.CreateFrom(ir::Value(node, 2)));
}

std::tuple<XLATensor, XLATensor> XLATensor::symeig(const XLATensor& input,
                                                   bool eigenvectors,
                                                   bool upper) {
  // SymEig takes lower instead of upper, hence the negation.
  ir::NodePtr node =
      ir::MakeNode<ir::ops::SymEig>(input.GetIrValue(), eigenvectors, !upper);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

XLATensor XLATensor::tan(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Tan(input.GetIrValue()));
}

void XLATensor::tan_(XLATensor& input) {
  input.SetIrValue(ir::ops::Tan(input.GetIrValue()));
}

XLATensor XLATensor::tanh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Tanh(input.GetIrValue()));
}

void XLATensor::tanh_(XLATensor& input) {
  input.SetIrValue(ir::ops::Tanh(input.GetIrValue()));
}

XLATensor XLATensor::tanh_backward(const XLATensor& grad_output,
                                   const XLATensor& output) {
  return XLATensor::mul(grad_output,
                        XLATensor::rsub(XLATensor::pow(output, 2), 1, 1));
}

XLATensor XLATensor::threshold(const XLATensor& input, float threshold,
                               float value) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Threshold>(input.GetIrValue(), threshold, value));
}

void XLATensor::threshold_(XLATensor& input, float threshold, float value) {
  input.SetIrValue(
      ir::MakeNode<ir::ops::Threshold>(input.GetIrValue(), threshold, value));
}

XLATensor XLATensor::threshold_backward(const XLATensor& grad_output,
                                        const XLATensor& input,
                                        float threshold) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::ThresholdBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), threshold));
}

XLATensor XLATensor::to(XLATensor& input, c10::optional<Device> device,
                        c10::optional<at::ScalarType> scalar_type) {
  if (!device) {
    device = input.GetDevice();
  }
  if (!scalar_type) {
    scalar_type = input.dtype();
  }
  if (input.GetDevice() == *device) {
    return input.dtype() == *scalar_type
               ? input.CreateFrom(input.GetIrValue())
               : input.CreateFrom(input.GetIrValue(), *scalar_type);
  }
  XLATensor new_tensor = input.CopyTensorToDevice(*device);
  if (input.dtype() != *scalar_type) {
    new_tensor.SetScalarType(*scalar_type);
  }
  return new_tensor;
}

std::tuple<XLATensor, XLATensor> XLATensor::topk(const XLATensor& input,
                                                 xla::int64 k, xla::int64 dim,
                                                 bool largest, bool sorted) {
  ir::NodePtr node = ir::MakeNode<ir::ops::TopK>(
      input.GetIrValue(), k,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      largest, sorted);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::trace(const XLATensor& input) {
  auto input_shape_ref = input.shape();
  XLA_CHECK_EQ((*input_shape_ref).rank(), 2)
      << "invalid argument for trace: expected a matrix";
  ir::NodePtr eye = ir::ops::Identity((*input_shape_ref).dimensions(0),
                                      (*input_shape_ref).dimensions(1),
                                      (*input_shape_ref).element_type());
  return XLATensor::sum(input.CreateFrom(eye * input.GetIrValue()), {0, 1},
                        false, input.dtype());
}

XLATensor XLATensor::transpose(const XLATensor& input, xla::int64 dim0,
                               xla::int64 dim1) {
  auto input_shape = input.shape();
  auto permute_dims = XlaHelpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
  ViewInfo view_info(
      ViewInfo::Type::kPermute,
      xla::util::ToVector<xla::int64>(input_shape.get().dimensions()),
      permute_dims, input_shape.get().element_type());
  return input.CreateViewTensor(std::move(view_info));
}

void XLATensor::transpose_(XLATensor& input, xla::int64 dim0, xla::int64 dim1) {
  input.SetIrValue(ir::ops::TransposeOp(input.GetIrValue(), dim0, dim1));
}

std::tuple<XLATensor, XLATensor> XLATensor::triangular_solve(
    const XLATensor& rhs, const XLATensor& lhs, bool left_side, bool upper,
    bool transpose, bool unitriangular) {
  // TriangularSolve takes lower instead of upper, hence the negation.
  ir::NodePtr node = ir::MakeNode<ir::ops::TriangularSolve>(
      rhs.GetIrValue(), lhs.GetIrValue(), left_side, !upper, transpose,
      unitriangular);
  return std::make_tuple(rhs.CreateFrom(ir::Value(node, 0)),
                         rhs.CreateFrom(ir::Value(node, 1)));
}

XLATensor XLATensor::tril(const XLATensor& input, xla::int64 diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

void XLATensor::tril_(XLATensor& input, xla::int64 diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

XLATensor XLATensor::triu(const XLATensor& input, xla::int64 diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

void XLATensor::triu_(XLATensor& input, xla::int64 diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

XLATensor XLATensor::trunc(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Trunc(input.GetIrValue()));
}

void XLATensor::trunc_(XLATensor& input) {
  input.SetIrValue(ir::ops::Trunc(input.GetIrValue()));
}

std::vector<XLATensor> XLATensor::unbind(const XLATensor& input,
                                         xla::int64 dim) {
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  xla::int64 dim_size = input.size(dim);
  std::vector<XLATensor> slices;
  slices.reserve(dim_size);
  for (xla::int64 index = 0; index < dim_size; ++index) {
    slices.push_back(select(input, dim, index));
  }
  return slices;
}

XLATensor XLATensor::unsqueeze(const XLATensor& input, xla::int64 dim) {
  auto input_shape = input.shape();
  xla::int64 squeeze_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank() + 1);
  auto dimensions =
      BuildUnsqueezeDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, dimensions);
}

void XLATensor::unsqueeze_(XLATensor& input, xla::int64 dim) {
  int squeeze_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, input.shape().get().rank() + 1);
  input.SetIrValue(
      ir::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(), squeeze_dim));
}

XLATensor XLATensor::view(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  auto input_shape = input.shape();
  std::vector<xla::int64> complete_dimensions =
      GetCompleteShape(output_size, input_shape.get().dimensions());
  xla::Shape shape = MakeArrayShapeFromDimensions(
      complete_dimensions, input_shape.get().element_type(),
      input.GetDevice().hw_type);
  ViewInfo view_info(ViewInfo::Type::kReshape, std::move(shape),
                     input_shape.get().dimensions());
  return input.CreateViewTensor(std::move(view_info));
}

void XLATensor::zero_(XLATensor& input) {
  ir::Value constant =
      GetIrValueForScalar(0.0, input.shape(), input.GetDevice());
  input.SetIrValue(std::move(constant));
}

XLATensor XLATensor::where(const XLATensor& condition, const XLATensor& input,
                           const XLATensor& other) {
  return input.CreateFrom(ir::ops::Where(
      condition.GetIrValue(), input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::DispatchComparisonOp(c10::Symbol kind,
                                          const XLATensor& input,
                                          at::Scalar other) {
  ir::NodePtr node = ir::ops::ComparisonOp(kind, input.GetIrValue(), other);
  return Create(node, input.GetDevice(), at::ScalarType::Bool);
}

XLATensor XLATensor::DispatchComparisonOp(c10::Symbol kind,
                                          const XLATensor& input,
                                          const XLATensor& other) {
  ir::NodePtr node =
      ir::ops::ComparisonOp(kind, input.GetIrValue(), other.GetIrValue());
  return Create(node, input.GetDevice(), at::ScalarType::Bool);
}

}  // namespace torch_xla
