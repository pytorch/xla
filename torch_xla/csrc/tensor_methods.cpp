#include "torch_xla/csrc/tensor.h"

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
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/adaptive_avg_pool2d.h"
#include "torch_xla/csrc/ops/all.h"
#include "torch_xla/csrc/ops/any.h"
#include "torch_xla/csrc/ops/arg_max.h"
#include "torch_xla/csrc/ops/arg_min.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/avg_pool2d.h"
#include "torch_xla/csrc/ops/avg_pool2d_backward.h"
#include "torch_xla/csrc/ops/batch_norm_forward.h"
#include "torch_xla/csrc/ops/bitwise_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/cat.h"
#include "torch_xla/csrc/ops/cholesky.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/constant_pad_nd.h"
#include "torch_xla/csrc/ops/conv2d.h"
#include "torch_xla/csrc/ops/conv2d_backward.h"
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
#include "torch_xla/csrc/ops/index_op.h"
#include "torch_xla/csrc/ops/index_select.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/kth_value.h"
#include "torch_xla/csrc/ops/leaky_relu.h"
#include "torch_xla/csrc/ops/log_softmax.h"
#include "torch_xla/csrc/ops/log_softmax_backward.h"
#include "torch_xla/csrc/ops/masked_fill.h"
#include "torch_xla/csrc/ops/max_pool2d.h"
#include "torch_xla/csrc/ops/max_pool2d_backward.h"
#include "torch_xla/csrc/ops/mean.h"
#include "torch_xla/csrc/ops/native_batch_norm_backward.h"
#include "torch_xla/csrc/ops/native_batch_norm_forward.h"
#include "torch_xla/csrc/ops/not_supported.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/prod.h"
#include "torch_xla/csrc/ops/qr.h"
#include "torch_xla/csrc/ops/repeat.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/scatter.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/slice.h"
#include "torch_xla/csrc/ops/softmax.h"
#include "torch_xla/csrc/ops/split.h"
#include "torch_xla/csrc/ops/squeeze.h"
#include "torch_xla/csrc/ops/stack.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/ops/svd.h"
#include "torch_xla/csrc/ops/symeig.h"
#include "torch_xla/csrc/ops/threshold.h"
#include "torch_xla/csrc/ops/threshold_backward.h"
#include "torch_xla/csrc/ops/topk.h"
#include "torch_xla/csrc/ops/tril.h"
#include "torch_xla/csrc/ops/triu.h"
#include "torch_xla/csrc/ops/unsqueeze.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/tensor_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/translator.h"

namespace torch_xla {
namespace {

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

}  // namespace

XLATensor XLATensor::__and__(const XLATensor& input, at::Scalar other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise and is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__and__(const XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise and is only supported for integer type tensors";
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__iand__(XLATensor& input, at::Scalar other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise and is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(
      ir::ops::BitwiseAnd(input.GetIrValue(), other_broadcasted_ir));
}

void XLATensor::__iand__(XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise and is only supported for integer type tensors";
  input.SetIrValue(ir::ops::BitwiseAnd(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__ilshift__(XLATensor& input, at::Scalar other) {
  input.SetIrValue(ir::ops::Lshift(input.GetIrValue(), other));
}

void XLATensor::__ilshift__(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__ior__(XLATensor& input, at::Scalar other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise or is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(
      ir::ops::BitwiseOr(input.GetIrValue(), other_broadcasted_ir));
}

void XLATensor::__ior__(XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise or is only supported for integer type tensors";
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
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise xor is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(
      ir::ops::BitwiseXor(input.GetIrValue(), other_broadcasted_ir));
}

void XLATensor::__ixor__(XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise xor is only supported for integer type tensors";
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
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise or is only supported for integer type tensors";
  return input.CreateFrom(
      ir::ops::BitwiseOr(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__or__(const XLATensor& input, at::Scalar other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise or is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
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
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise xor is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(
      ir::ops::BitwiseXor(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__xor__(const XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise xor is only supported for integer type tensors";
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
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  return input.CreateFrom(input.GetIrValue() + other.GetIrValue() * constant);
}

void XLATensor::add_(XLATensor& input, const XLATensor& other,
                     at::Scalar alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  input.SetIrValue(input.GetIrValue() + other.GetIrValue() * constant);
}

XLATensor XLATensor::add(const XLATensor& input, at::Scalar other,
                         at::Scalar alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  return input.CreateFrom(input.GetIrValue() + other_constant * alpha_constant);
}

void XLATensor::add_(XLATensor& input, at::Scalar other, at::Scalar alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  input.SetIrValue(input.GetIrValue() + other_constant * alpha_constant);
}

void XLATensor::addcmul_(XLATensor& input, at::Scalar value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value, tensor1.shape().get().element_type());
  ir::Value mul = tensor1.GetIrValue() * tensor2.GetIrValue();
  input.SetIrValue(input.GetIrValue() + mul * constant);
}

XLATensor XLATensor::addcdiv(const XLATensor& input, at::Scalar value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  return input.CreateFrom(input.GetIrValue() + div * constant);
}

void XLATensor::addcdiv_(XLATensor& input, at::Scalar value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  input.SetIrValue(input.GetIrValue() + div * constant);
}

XLATensor XLATensor::addcmul(const XLATensor& input, at::Scalar value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value, tensor1.shape().get().element_type());
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
  return input.CreateFrom(ir::MakeNode<ir::ops::AsStrided>(
      input.GetIrValue(), size, storage_offset));
}

void XLATensor::as_strided_(XLATensor& input, std::vector<xla::int64> size,
                            c10::optional<xla::int64> storage_offset) {
  input.SetIrValue(ir::MakeNode<ir::ops::AsStrided>(input.GetIrValue(), size,
                                                    storage_offset));
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

XLATensor XLATensor::avg_pool2d(const XLATensor& input,
                                std::vector<xla::int64> kernel_size,
                                std::vector<xla::int64> stride,
                                std::vector<xla::int64> padding,
                                bool count_include_pad) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AvgPool2d>(
      input.GetIrValue(), std::move(kernel_size), std::move(stride),
      std::move(padding), count_include_pad));
}

XLATensor XLATensor::avg_pool2d_backward(const XLATensor& out_backprop,
                                         const XLATensor& input,
                                         std::vector<xla::int64> kernel_size,
                                         std::vector<xla::int64> stride,
                                         std::vector<xla::int64> padding,
                                         bool count_include_pad) {
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::AvgPool2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), std::move(kernel_size),
      std::move(stride), std::move(padding), count_include_pad));
}

XLATensor XLATensor::batch_norm(const XLATensor& input, const XLATensor& weight,
                                const XLATensor& bias,
                                const XLATensor& running_mean,
                                const XLATensor& running_var, double momentum,
                                double eps) {
  return input.CreateFrom(ir::MakeNode<ir::ops::BatchNormForward>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(), momentum, eps));
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
  return input.CreateFrom(ir::ops::Clamp(input.GetIrValue(), min, max));
}

void XLATensor::clamp_(XLATensor& input, c10::optional<at::Scalar> min,
                       c10::optional<at::Scalar> max) {
  input.SetIrValue(ir::ops::Clamp(input.GetIrValue(), min, max));
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
                           xla::int64 dim) {
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
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(input.GetIrValue() / constant);
}

void XLATensor::div_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(input.GetIrValue() / other.GetIrValue());
}

void XLATensor::div_(XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(input.GetIrValue() / constant);
}

XLATensor XLATensor::dropout(const XLATensor& input, double p) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Dropout>(input.GetIrValue(), p));
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
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Expand>(input.GetIrValue(), std::move(size)));
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
  input.SetIrValue(ir::ops::ScalarOp(value, input.shape()));
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
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), constant));
}

void XLATensor::fmod_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::fmod_(XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
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
  return Create(ir::MakeNode<ir::ops::Scalar>(fill_value, std::move(shape)),
                device, scalar_type);
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
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Scalar>(fill_value, tensor_shape), device,
      *scalar_type);
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

XLATensor XLATensor::index(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const XLATensor> indices) {
  return IndexByTensors(input, indices);
}

XLATensor XLATensor::index_select(const XLATensor& input, xla::int64 dim,
                                  const XLATensor& index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::IndexSelect>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue()));
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

XLATensor XLATensor::leaky_relu(const XLATensor& input, double negative_slope) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LeakyRelu>(input.GetIrValue(), negative_slope));
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

XLATensor XLATensor::log_softmax(const XLATensor& input, xla::int64 dim) {
  return input.CreateFrom(ir::MakeNode<ir::ops::LogSoftmax>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::log_softmax_backward(const XLATensor& grad_output,
                                          const XLATensor& output,
                                          xla::int64 dim) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::LogSoftmaxBackward>(
      grad_output.GetIrValue(), output.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(
          dim, grad_output.shape().get().rank())));
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

XLATensor XLATensor::max_pool2d(const XLATensor& input,
                                std::vector<xla::int64> kernel_size,
                                std::vector<xla::int64> stride,
                                std::vector<xla::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MaxPool2d>(
      input.GetIrValue(), std::move(kernel_size), std::move(stride),
      std::move(padding)));
}

XLATensor XLATensor::max_pool2d_backward(const XLATensor& out_backprop,
                                         const XLATensor& input,
                                         std::vector<xla::int64> kernel_size,
                                         std::vector<xla::int64> stride,
                                         std::vector<xla::int64> padding) {
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::MaxPool2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), std::move(kernel_size),
      std::move(stride), std::move(padding)));
}

XLATensor XLATensor::mean(const XLATensor& input,
                          std::vector<xla::int64> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Mean>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions, dtype));
}

XLATensor XLATensor::min(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(ir::ops::Min(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::mm(const XLATensor& input, const XLATensor& weight) {
  return input.CreateFrom(
      ir::ops::Dot(input.GetIrValue(), weight.GetIrValue()));
}

XLATensor XLATensor::mul(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(input.GetIrValue() * other.GetIrValue());
}

XLATensor XLATensor::mul(const XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(input.GetIrValue() * constant);
}

void XLATensor::mul_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(input.GetIrValue() * other.GetIrValue());
}

void XLATensor::mul_(XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(input.GetIrValue() * constant);
}

XLATensor XLATensor::narrow(const XLATensor& input, xla::int64 dim,
                            xla::int64 start, xla::int64 length) {
  auto input_shape = input.shape();
  xla::Shape narrow_shape = input_shape;
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  narrow_shape.set_dimensions(dim, length);
  ViewInfo view_info(std::move(narrow_shape), input_shape.get().dimensions());
  view_info.indices[dim] = XlaHelpers::GetCanonicalPosition(
      input_shape.get().dimensions(), dim, start);
  return input.CreateView(std::move(view_info));
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::native_batch_norm(
    const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
    const XLATensor& running_mean, const XLATensor& running_var,
    double momentum, double eps) {
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormForward>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(), momentum, eps);
  XLATensor output = input.CreateFrom(ir::Value(node, 0));
  XLATensor save_mean = input.CreateFrom(ir::Value(node, 1));
  XLATensor save_invstd = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(output), std::move(save_mean),
                         std::move(save_invstd));
}

std::tuple<XLATensor, XLATensor, XLATensor>
XLATensor::native_batch_norm_backward(
    const XLATensor& grad_out, const XLATensor& input, const XLATensor& weight,
    const XLATensor& running_mean, const XLATensor& running_var,
    const XLATensor& save_mean, const XLATensor& save_invstd, double eps) {
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormBackward>(
      grad_out.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(),
      save_mean.GetIrValue(), save_invstd.GetIrValue(), eps);
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

XLATensor XLATensor::nll_loss(const XLATensor& input, const XLATensor& target) {
  return input.CreateFrom(
      ir::ops::NllLossOp(input.GetIrValue(), target.GetIrValue()));
}

XLATensor XLATensor::nll_loss_backward(const XLATensor& input,
                                       const XLATensor& target) {
  return input.CreateFrom(
      ir::ops::NllLossBackwardOp(input.GetIrValue(), target.GetIrValue()));
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
      xla::util::ToVector<xla::int64>(input_shape.get().dimensions()),
      XlaHelpers::GetCanonicalDimensionIndices(dims, input_shape.get().rank()),
      input_shape.get().element_type());
  return input.CreateView(std::move(view_info));
}

XLATensor XLATensor::pow(const XLATensor& input, at::Scalar exponent) {
  ir::NodePtr exponent_node = ir::ops::ScalarOp(exponent, input.shape());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

XLATensor XLATensor::pow(const XLATensor& input, const XLATensor& exponent) {
  return input.CreateFrom(
      ir::ops::Pow(input.GetIrValue(), exponent.GetIrValue()));
}

XLATensor XLATensor::pow(at::Scalar input, const XLATensor& exponent) {
  ir::NodePtr input_node = ir::ops::ScalarOp(input, exponent.shape());
  return exponent.CreateFrom(ir::ops::Pow(input_node, exponent.GetIrValue()));
}

void XLATensor::pow_(XLATensor& input, at::Scalar exponent) {
  ir::NodePtr exponent_node = ir::ops::ScalarOp(exponent, input.shape());
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
                                  keep_reduced_dimensions, dtype));
}

std::tuple<XLATensor, XLATensor> XLATensor::qr(const XLATensor& input,
                                               bool full_matrices) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::QR>(input.GetIrValue(), full_matrices);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
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
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(ir::ops::Remainder(input.GetIrValue(), constant));
}

void XLATensor::remainder_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Remainder(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::remainder_(XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
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

XLATensor XLATensor::rsqrt(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Rsqrt(input.GetIrValue()));
}

void XLATensor::rsqrt_(XLATensor& input) {
  input.SetIrValue(ir::ops::Rsqrt(input.GetIrValue()));
}

XLATensor XLATensor::rsub(const XLATensor& input, const XLATensor& other,
                          at::Scalar alpha) {
  ir::NodePtr alpha_xla = ir::ops::ScalarOp(alpha, other.shape());
  return input.CreateFrom(other.GetIrValue() - alpha_xla * input.GetIrValue());
}

XLATensor XLATensor::rsub(const XLATensor& input, at::Scalar other,
                          at::Scalar alpha) {
  ir::NodePtr alpha_xla = ir::ops::ScalarOp(alpha, input.shape());
  ir::NodePtr other_xla = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(other_xla - alpha_xla * input.GetIrValue());
}

void XLATensor::s_copy_(XLATensor& input, XLATensor& src) {
  if (input.GetDevice() == src.GetDevice()) {
    input.SetIrValue(src.GetIrValue());
  } else {
    // TODO: This can be optimized via proper XRT/XLA computation.
    XLATensor new_tensor = XLATensor::Create(src.ToTensor(), input.GetDevice(),
                                             src.RequiresGrad());
    std::swap(input, new_tensor);
  }
}

void XLATensor::scatter_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, const XLATensor& src) {
  input.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue(), src.GetIrValue()));
}

XLATensor XLATensor::scatter(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, const XLATensor& src) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue(), src.GetIrValue()));
}

void XLATensor::scatter_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, at::Scalar value) {
  ir::NodePtr constant = ir::ops::ScalarOp(value, input.shape());
  input.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue(), constant));
}

XLATensor XLATensor::scatter(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, at::Scalar value) {
  ir::NodePtr constant = ir::ops::ScalarOp(value, input.shape());
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue(), constant));
}

XLATensor XLATensor::select(const XLATensor& input, xla::int64 dim,
                            xla::int64 index) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  index = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                           index);
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Select>(input.GetIrValue(), dim, index));
}

XLATensor XLATensor::sigmoid(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sigmoid(input.GetIrValue()));
}

void XLATensor::sigmoid_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sigmoid(input.GetIrValue()));
}

XLATensor XLATensor::slice(const XLATensor& input, xla::int64 dim,
                           xla::int64 start, xla::int64 end, xla::int64 step) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                           start);
  end = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                         end);
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Slice>(input.GetIrValue(), dim, start, end, step));
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

XLATensor XLATensor::softmax(const XLATensor& input, xla::int64 dim) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Softmax>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
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
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant);
}

XLATensor XLATensor::sum(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions,
                         c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Sum>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions, dtype));
}

void XLATensor::sub_(XLATensor& input, const XLATensor& other,
                     at::Scalar alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  input.SetIrValue(input.GetIrValue() - other.GetIrValue() * constant);
}

XLATensor XLATensor::sub(const XLATensor& input, at::Scalar other,
                         at::Scalar alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant);
}

void XLATensor::sub_(XLATensor& input, at::Scalar other, at::Scalar alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
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
      xla::util::ToVector<xla::int64>(input_shape.get().dimensions()),
      permute_dims, input_shape.get().element_type());
  return input.CreateView(std::move(view_info));
}

void XLATensor::transpose_(XLATensor& input, xla::int64 dim0, xla::int64 dim1) {
  input.SetIrValue(ir::ops::TransposeOp(input.GetIrValue(), dim0, dim1));
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
  int squeeze_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, input.shape().get().rank() + 1);
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(), squeeze_dim));
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
  ViewInfo view_info(std::move(shape), input_shape.get().dimensions());
  return input.CreateView(std::move(view_info));
}

void XLATensor::zero_(XLATensor& input) {
  input.SetIrValue(ir::ops::ScalarOp(0.0, input.shape()));
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
  return Create(node, input.GetDevice(), at::ScalarType::Byte);
}

XLATensor XLATensor::DispatchComparisonOp(c10::Symbol kind,
                                          const XLATensor& input,
                                          const XLATensor& other) {
  ir::NodePtr node =
      ir::ops::ComparisonOp(kind, input.GetIrValue(), other.GetIrValue());
  return Create(node, input.GetDevice(), at::ScalarType::Byte);
}

}  // namespace torch_xla
