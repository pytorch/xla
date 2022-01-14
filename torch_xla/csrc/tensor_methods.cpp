#include <ATen/core/Reduction.h>

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
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/adaptive_avg_pool2d.h"
#include "torch_xla/csrc/ops/adaptive_avg_pool3d.h"
#include "torch_xla/csrc/ops/adaptive_max_pool2d.h"
#include "torch_xla/csrc/ops/all.h"
#include "torch_xla/csrc/ops/all_gather.h"
#include "torch_xla/csrc/ops/all_reduce.h"
#include "torch_xla/csrc/ops/all_to_all.h"
#include "torch_xla/csrc/ops/amax.h"
#include "torch_xla/csrc/ops/amin.h"
#include "torch_xla/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "torch_xla/csrc/ops/amp_update_scale.h"
#include "torch_xla/csrc/ops/any.h"
#include "torch_xla/csrc/ops/arg_max.h"
#include "torch_xla/csrc/ops/arg_min.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/avg_pool_nd.h"
#include "torch_xla/csrc/ops/avg_pool_nd_backward.h"
#include "torch_xla/csrc/ops/bernoulli.h"
#include "torch_xla/csrc/ops/binary_cross_entropy.h"
#include "torch_xla/csrc/ops/binary_cross_entropy_backward.h"
#include "torch_xla/csrc/ops/bitwise_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/cat.h"
#include "torch_xla/csrc/ops/cholesky.h"
#include "torch_xla/csrc/ops/collective_permute.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/constant_pad_nd.h"
#include "torch_xla/csrc/ops/convolution_backward_overrideable.h"
#include "torch_xla/csrc/ops/convolution_overrideable.h"
#include "torch_xla/csrc/ops/cumprod.h"
#include "torch_xla/csrc/ops/cumsum.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/discrete_uniform.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/exponential.h"
#include "torch_xla/csrc/ops/flip.h"
#include "torch_xla/csrc/ops/gather.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/get_dimensions_size.h"
#include "torch_xla/csrc/ops/hardshrink.h"
#include "torch_xla/csrc/ops/hardtanh_backward.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/ops/index_select.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/kth_value.h"
#include "torch_xla/csrc/ops/l1_loss.h"
#include "torch_xla/csrc/ops/l1_loss_backward.h"
#include "torch_xla/csrc/ops/leaky_relu.h"
#include "torch_xla/csrc/ops/leaky_relu_backward.h"
#include "torch_xla/csrc/ops/linear_interpolation.h"
#include "torch_xla/csrc/ops/log_softmax.h"
#include "torch_xla/csrc/ops/logsumexp.h"
#include "torch_xla/csrc/ops/masked_fill.h"
#include "torch_xla/csrc/ops/masked_scatter.h"
#include "torch_xla/csrc/ops/masked_select.h"
#include "torch_xla/csrc/ops/max_in_dim.h"
#include "torch_xla/csrc/ops/max_pool_nd.h"
#include "torch_xla/csrc/ops/max_pool_nd_backward.h"
#include "torch_xla/csrc/ops/max_unpool_nd.h"
#include "torch_xla/csrc/ops/max_unpool_nd_backward.h"
#include "torch_xla/csrc/ops/mean.h"
#include "torch_xla/csrc/ops/min_in_dim.h"
#include "torch_xla/csrc/ops/mse_loss.h"
#include "torch_xla/csrc/ops/mse_loss_backward.h"
#include "torch_xla/csrc/ops/native_batch_norm_backward.h"
#include "torch_xla/csrc/ops/native_batch_norm_forward.h"
#include "torch_xla/csrc/ops/nll_loss.h"
#include "torch_xla/csrc/ops/nll_loss2d.h"
#include "torch_xla/csrc/ops/nll_loss2d_backward.h"
#include "torch_xla/csrc/ops/nll_loss_backward.h"
#include "torch_xla/csrc/ops/nms.h"
#include "torch_xla/csrc/ops/nonzero.h"
#include "torch_xla/csrc/ops/normal.h"
#include "torch_xla/csrc/ops/not_supported.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/prod.h"
#include "torch_xla/csrc/ops/put.h"
#include "torch_xla/csrc/ops/qr.h"
#include "torch_xla/csrc/ops/reduce_scatter.h"
#include "torch_xla/csrc/ops/reflection_pad2d.h"
#include "torch_xla/csrc/ops/reflection_pad2d_backward.h"
#include "torch_xla/csrc/ops/repeat.h"
#include "torch_xla/csrc/ops/replication_pad.h"
#include "torch_xla/csrc/ops/replication_pad_backward.h"
#include "torch_xla/csrc/ops/resize.h"
#include "torch_xla/csrc/ops/rrelu_with_noise.h"
#include "torch_xla/csrc/ops/rrelu_with_noise_backward.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/scatter.h"
#include "torch_xla/csrc/ops/scatter_add.h"
#include "torch_xla/csrc/ops/sgd_optimizer_step.h"
#include "torch_xla/csrc/ops/shrink_backward.h"
#include "torch_xla/csrc/ops/softmax.h"
#include "torch_xla/csrc/ops/softshrink.h"
#include "torch_xla/csrc/ops/split.h"
#include "torch_xla/csrc/ops/squeeze.h"
#include "torch_xla/csrc/ops/stack.h"
#include "torch_xla/csrc/ops/std.h"
#include "torch_xla/csrc/ops/std_mean.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/ops/svd.h"
#include "torch_xla/csrc/ops/symeig.h"
#include "torch_xla/csrc/ops/threshold.h"
#include "torch_xla/csrc/ops/threshold_backward.h"
#include "torch_xla/csrc/ops/topk.h"
#include "torch_xla/csrc/ops/triangular_solve.h"
#include "torch_xla/csrc/ops/tril.h"
#include "torch_xla/csrc/ops/triu.h"
#include "torch_xla/csrc/ops/uniform.h"
#include "torch_xla/csrc/ops/unsqueeze.h"
#include "torch_xla/csrc/ops/upsample_bilinear2d.h"
#include "torch_xla/csrc/ops/upsample_bilinear2d_backward.h"
#include "torch_xla/csrc/ops/upsample_nearest2d.h"
#include "torch_xla/csrc/ops/upsample_nearest2d_backward.h"
#include "torch_xla/csrc/ops/user_computation.h"
#include "torch_xla/csrc/ops/var.h"
#include "torch_xla/csrc/ops/var_mean.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/shape_builder.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct MinMaxValues {
  ir::Value min;
  ir::Value max;
};

ir::Value MaybeExpand(const ir::Value& input, const xla::Shape& target_shape) {
  if (input.shape().dimensions() == target_shape.dimensions()) {
    return input;
  }
  return ir::MakeNode<ir::ops::Expand>(
      input, xla::util::ToVector<int64_t>(target_shape.dimensions()));
}

MinMaxValues GetMinMaxValues(const XLATensor& tensor,
                             const c10::optional<at::Scalar>& min,
                             const c10::optional<at::Scalar>& max) {
  XLA_CHECK(min || max)
      << "At least one of \'min\' or \'max\' must not be None";
  xla::PrimitiveType raw_element_type = TensorTypeToRawXlaType(tensor.dtype());
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(raw_element_type);
  auto shape = tensor.shape();
  return {XLATensor::GetIrValueForScalar(min ? *min : min_max.min,
                                         shape.get().element_type(),
                                         tensor.GetDevice()),
          XLATensor::GetIrValueForScalar(max ? *max : min_max.max,
                                         shape.get().element_type(),
                                         tensor.GetDevice())};
}

void CheckRank(const XLATensor& t, int64_t expected_rank,
               const std::string& tag, const std::string& arg_name,
               int arg_number) {
  int64_t actual_rank = t.shape().get().rank();
  XLA_CHECK_EQ(actual_rank, expected_rank)
      << "Expected " << expected_rank << "-dimensional tensor, but got "
      << actual_rank << "-dimensional tensor for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

template <typename T>
void CheckShapeDimensions(const T& size) {
  XLA_CHECK(std::all_of(size.begin(), size.end(), [](int64_t dim) {
    return dim >= 0;
  })) << "Dimensions cannot be negative numbers";
}

void CheckDimensionSize(const XLATensor& t, int64_t dim, int64_t expected_size,
                        const std::string& tag, const std::string& arg_name,
                        int arg_number) {
  int64_t dim_size = t.size(dim);
  XLA_CHECK_EQ(t.size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

void CheckBmmDimension(const std::string& tag, const XLATensor& batch1,
                       const XLATensor& batch2) {
  // Consistent with the checks in bmm_out_or_baddbmm_.
  CheckRank(batch1, 3, tag, "batch1", 1);
  CheckRank(batch2, 3, tag, "batch2", 2);
  CheckDimensionSize(batch2, 0, /*batch_size=*/batch1.size(0), tag, "batch2",
                     2);
  CheckDimensionSize(batch2, 1, /*contraction_size=*/batch1.size(2), tag,
                     "batch2", 2);
}

std::vector<int64_t> GetExpandDimensions(const xla::Shape& shape,
                                         std::vector<int64_t> dimensions) {
  XLA_CHECK_GE(dimensions.size(), shape.rank()) << shape;
  int64_t base = dimensions.size() - shape.rank();
  for (size_t i = 0; i < shape.rank(); ++i) {
    if (dimensions[base + i] == -1) {
      dimensions[base + i] = shape.dimensions(i);
    }
  }
  return dimensions;
}

ReductionMode GetXlaReductionMode(int64_t reduction) {
  switch (reduction) {
    case at::Reduction::Mean:
      return ReductionMode::kMean;
    case at::Reduction::None:
      return ReductionMode::kNone;
    case at::Reduction::Sum:
      return ReductionMode::kSum;
  }
  XLA_ERROR() << "Unknown reduction mode: " << reduction;
}

// Resizes and / or checks whether a list is of the given size. The list is only
// resized if its size is 1. If it's empty, it's replaced with the provided
// default first.
std::vector<int64_t> CheckIntList(absl::Span<const int64_t> list, size_t length,
                                  const std::string& name,
                                  std::vector<int64_t> def = {}) {
  std::vector<int64_t> result;
  if (list.empty()) {
    result = std::move(def);
  } else {
    result = xla::util::ToVector<int64_t>(list);
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
  auto input_shape = input.shape();
  return ShapeBuilder(input_element_type).Add(input_shape.get(), 1).Build();
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
ir::Value GetIrValueOrDefault(const XLATensor& input,
                              const at::Scalar& default_value,
                              const xla::Shape& default_shape,
                              const Device& device) {
  return input.is_null() ? XLATensor::GetIrValueForScalar(default_value,
                                                          default_shape, device)
                         : input.GetIrValue();
}

// Returns the IR for the given input. If the IR is not a floating point value,
// cast it to the float_type.
ir::Value GetFloatingIrValue(const XLATensor& input,
                             at::ScalarType float_type) {
  ir::Value input_value = input.GetIrValue();
  if (xla::primitive_util::IsIntegralType(input_value.shape().element_type())) {
    input_value = ir::MakeNode<ir::ops::Cast>(input_value, float_type);
  }
  return input_value;
}

ir::Value GetBooleanIrValue(ir::Value input_value) {
  if (input_value.shape().element_type() != xla::PrimitiveType::PRED) {
    input_value =
        ir::MakeNode<ir::ops::Cast>(input_value, xla::PrimitiveType::PRED);
  }
  return input_value;
}

absl::optional<ir::Value> GetOptionalIrValue(const XLATensor& tensor) {
  absl::optional<ir::Value> value;
  if (!tensor.is_null()) {
    value = tensor.GetIrValue();
  }
  return value;
}

void CheckIsIntegralOrPred(const xla::Shape& shape,
                           const std::string& op_name) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(shape) ||
            shape.element_type() == xla::PrimitiveType::PRED)
      << "Operator " << op_name
      << " is only supported for integer or boolean type tensors, got: "
      << shape;
}

ViewInfo CreateAsStridedViewInfo(const xla::Shape& input_shape,
                                 std::vector<int64_t> size,
                                 std::vector<int64_t> stride,
                                 c10::optional<int64_t> storage_offset) {
  xla::Shape result_shape = XlaHelpers::GetDynamicReshape(input_shape, size);
  AsStridedInfo as_strided_info;
  as_strided_info.stride = std::move(stride);
  if (storage_offset) {
    as_strided_info.offset = *storage_offset;
  }
  return ViewInfo(ViewInfo::Type::kAsStrided, std::move(result_shape),
                  input_shape, std::move(as_strided_info));
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////
// XLA dedicated operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
std::pair<XLATensor, ir::Value> XLATensor::all_reduce(
    const XLATensor& input, const ir::Value& token, AllReduceType reduce_type,
    double scale, std::vector<std::vector<int64_t>> groups) {
  std::vector<ir::Value> input_values({input.GetIrValue()});
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

ir::Value XLATensor::all_reduce_(XLATensor& input, const ir::Value& token,
                                 AllReduceType reduce_type, double scale,
                                 std::vector<std::vector<int64_t>> groups) {
  std::vector<ir::Value> input_values({input.GetIrValue()});
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  input.SetInPlaceIrValue(ir::Value(node, 0));
  return ir::Value(node, 1);
}

ir::Value XLATensor::all_reduce(std::vector<XLATensor>* inputs,
                                const ir::Value& token,
                                AllReduceType reduce_type, double scale,
                                std::vector<std::vector<int64_t>> groups) {
  std::vector<ir::Value> input_values;
  input_values.reserve(inputs->size());
  for (auto& input : *inputs) {
    input_values.push_back(input.GetIrValue());
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  for (size_t i = 0; i < inputs->size(); ++i) {
    (*inputs)[i].SetInPlaceIrValue(ir::Value(node, i));
  }
  return ir::Value(node, inputs->size());
}

std::pair<XLATensor, ir::Value> XLATensor::reduce_scatter(
    const XLATensor& input, const ir::Value& token, AllReduceType reduce_type,
    double scale, int64_t scatter_dim, int64_t shard_count,
    std::vector<std::vector<int64_t>> groups) {
  ir::NodePtr node = ir::MakeNode<ir::ops::ReduceScatter>(
      reduce_type, input.GetIrValue(), token, scale, scatter_dim, shard_count,
      std::move(groups));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

std::pair<XLATensor, ir::Value> XLATensor::all_to_all(
    const XLATensor& input, const ir::Value& token, int64_t split_dimension,
    int64_t concat_dimension, int64_t split_count,
    std::vector<std::vector<int64_t>> groups) {
  ir::NodePtr node = ir::MakeNode<ir::ops::AllToAll>(
      input.GetIrValue(), token, split_dimension, concat_dimension, split_count,
      std::move(groups));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

std::pair<XLATensor, ir::Value> XLATensor::all_gather(
    const XLATensor& input, const ir::Value& token, xla::int64_t dim,
    xla::int64_t shard_count, std::vector<std::vector<xla::int64_t>> groups) {
  ir::NodePtr node = ir::MakeNode<ir::ops::AllGather>(
      input.GetIrValue(), token, dim, shard_count, std::move(groups));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

std::pair<XLATensor, ir::Value> XLATensor::collective_permute(
    const XLATensor& input, const ir::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs) {
  ir::NodePtr node = ir::MakeNode<ir::ops::CollectivePermute>(
      input.GetIrValue(), token, std::move(source_target_pairs));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

XLATensor XLATensor::get_dimensions_size(const XLATensor& input,
                                         std::vector<int64_t> dimensions) {
  return input.CreateFrom(ir::MakeNode<ir::ops::GetDimensionsSize>(
                              input.GetIrValue(), std::move(dimensions)),
                          at::ScalarType::Int);
}

void XLATensor::sgd_optimizer_step_(XLATensor& step, XLATensor& param,
                                    XLATensor& buf, const XLATensor& found_inf,
                                    const XLATensor& d_p, double weight_decay,
                                    double momentum, double lr,
                                    double dampening, bool nesterov) {
  ir::Value weight_decay_value =
      GetIrValueForScalar(weight_decay, param.shape(), param.GetDevice());
  ir::Value momentum_value =
      GetIrValueForScalar(momentum, param.shape(), param.GetDevice());
  ir::Value lr_value =
      GetIrValueForScalar(lr, param.shape(), param.GetDevice());
  ir::Value dampening_value =
      GetIrValueForScalar(dampening, param.shape(), param.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::SgdOptimizerStep>(
      step.GetIrValue(), param.GetIrValue(), buf.GetIrValue(),
      found_inf.GetIrValue(), d_p.GetIrValue(), weight_decay_value,
      momentum_value, lr_value, dampening_value,
      /*use_weight_decay=*/weight_decay != 0,
      /*use_momentum=*/momentum != 0, /*use_nesterov=*/nesterov);
  step.SetInPlaceIrValue(ir::Value(node, 0));
  param.SetInPlaceIrValue(ir::Value(node, 1));
  buf.SetInPlaceIrValue(ir::Value(node, 2));
}

std::vector<XLATensor> XLATensor::user_computation(
    const std::string& opname, absl::Span<const XLATensor> inputs,
    ComputationPtr computation) {
  XLA_CHECK(!inputs.empty());
  std::vector<ir::Value> input_values;
  for (auto& input : inputs) {
    input_values.push_back(input.GetIrValue());
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::UserComputation>(
      ir::OpKind::Get(opname), input_values, std::move(computation));
  // Cast can be one of the user computation and we don't want to inherit the
  // logical_element_type in this case
  return inputs.front().MakeOutputTensors(node, /*inherit_logical_type=*/false);
}

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
void XLATensor::__ilshift__(XLATensor& input, const at::Scalar& other) {
  input.SetInPlaceIrValue(ir::ops::Lshift(input.GetIrValue(), other));
}

void XLATensor::__ilshift__(XLATensor& input, const XLATensor& other) {
  input.SetInPlaceIrValue(
      ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::__irshift__(XLATensor& input, const at::Scalar& other) {
  input.SetInPlaceIrValue(ir::ops::Rshift(input.GetIrValue(), other));
}

void XLATensor::__irshift__(XLATensor& input, const XLATensor& other) {
  input.SetInPlaceIrValue(
      ir::ops::Rshift(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__lshift__(
    const XLATensor& input, const at::Scalar& other,
    c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Lshift(input.GetIrValue(), other),
                          logical_element_type);
}

XLATensor XLATensor::__lshift__(
    const XLATensor& input, const XLATensor& other,
    c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(
      ir::ops::Lshift(input.GetIrValue(), other.GetIrValue()),
      logical_element_type);
}

XLATensor XLATensor::__rshift__(
    const XLATensor& input, const at::Scalar& other,
    c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Rshift(input.GetIrValue(), other),
                          logical_element_type);
}

XLATensor XLATensor::__rshift__(
    const XLATensor& input, const XLATensor& other,
    c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(
      ir::ops::Rshift(input.GetIrValue(), other.GetIrValue()),
      logical_element_type);
}

std::tuple<XLATensor, XLATensor> XLATensor::adaptive_max_pool2d(
    const XLATensor& input, std::vector<int64_t> output_size) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::AdaptiveMaxPool2d>(input.GetIrValue(), output_size);
  XLATensor out = input.CreateFrom(ir::Value(node, 0));
  XLATensor indices =
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long);
  return std::make_tuple(std::move(out), std::move(indices));
}

XLATensor XLATensor::adaptive_max_pool2d_backward(const XLATensor& grad_output,
                                                  const XLATensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveMaxPool2dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

XLATensor XLATensor::adaptive_avg_pool3d(const XLATensor& input,
                                         std::vector<int64_t> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AdaptiveAvgPool3d>(
      input.GetIrValue(), std::move(output_size)));
}

XLATensor XLATensor::adaptive_avg_pool3d_backward(const XLATensor& grad_output,
                                                  const XLATensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool3dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

XLATensor XLATensor::_adaptive_avg_pool2d(const XLATensor& input,
                                          std::vector<int64_t> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AdaptiveAvgPool2d>(
      input.GetIrValue(), std::move(output_size)));
}

XLATensor XLATensor::_adaptive_avg_pool2d_backward(const XLATensor& grad_output,
                                                   const XLATensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool2dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

void XLATensor::_amp_foreach_non_finite_check_and_unscale_(
    std::vector<XLATensor> self, XLATensor& found_inf,
    const XLATensor& inv_scale) {
  std::vector<ir::Value> inputs;
  XLATensor new_inv_scale = XLATensor::max(inv_scale);
  for (const auto& x : self) {
    inputs.push_back(x.GetIrValue());
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::AmpForachNonFiniteCheckAndUnscale>(
      inputs, found_inf.GetIrValue(), new_inv_scale.GetIrValue());
  for (size_t i = 0; i < self.size(); ++i) {
    self[i].SetInPlaceIrValue(ir::Value(node, i));
  }
  found_inf.SetInPlaceIrValue(ir::Value(node, self.size()));
}

void XLATensor::_amp_update_scale_(XLATensor& current_scale,
                                   XLATensor& growth_tracker,
                                   const XLATensor& found_inf,
                                   double scale_growth_factor,
                                   double scale_backoff_factor,
                                   int growth_interval) {
  ir::NodePtr node = ir::MakeNode<ir::ops::AmpUpdateScale>(
      growth_tracker.GetIrValue(), current_scale.GetIrValue(),
      found_inf.GetIrValue(), scale_growth_factor, scale_backoff_factor,
      growth_interval);
  growth_tracker.SetInPlaceIrValue(ir::Value(node, 1));
  current_scale.SetInPlaceIrValue(ir::Value(node, 0));
}

XLATensor XLATensor::abs(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Abs(input.GetIrValue()));
}

XLATensor XLATensor::acos(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Acos(input.GetIrValue()));
}

XLATensor XLATensor::acosh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Acosh(input.GetIrValue()));
}

XLATensor XLATensor::add(const XLATensor& input, const XLATensor& other,
                         const at::Scalar& alpha,
                         c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() + other.GetIrValue() * constant,
                          logical_element_type);
}

XLATensor XLATensor::add(const XLATensor& input, const at::Scalar& other,
                         const at::Scalar& alpha,
                         c10::optional<at::ScalarType> logical_element_type) {
  ir::Value other_constant = GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  ir::Value alpha_constant = GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() + other_constant * alpha_constant,
                          logical_element_type);
}

XLATensor XLATensor::addcdiv(const XLATensor& input, const at::Scalar& value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::Value constant = GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  return input.CreateFrom(input.GetIrValue() + div * constant);
}

void XLATensor::addcdiv_(XLATensor& input, const at::Scalar& value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::Value constant = GetIrValueForScalar(
      value, tensor1.shape().get().element_type(), input.GetDevice());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  input.SetInPlaceIrValue(input.GetIrValue() + div * constant);
}

XLATensor XLATensor::addcmul(const XLATensor& input, const at::Scalar& value,
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
                         std::vector<int64_t> dimensions,
                         bool keep_reduced_dimensions) {
  at::ScalarType result_type = input.dtype() == at::ScalarType::Byte
                                   ? at::ScalarType::Byte
                                   : at::ScalarType::Bool;
  return input.CreateFrom(
      ir::MakeNode<ir::ops::All>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions),
      result_type);
}

XLATensor XLATensor::amax(const XLATensor& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Amax>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions));
}

XLATensor XLATensor::amin(const XLATensor& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Amin>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions));
}

XLATensor XLATensor::any(const XLATensor& input,
                         std::vector<int64_t> dimensions,
                         bool keep_reduced_dimensions) {
  at::ScalarType result_type = input.dtype() == at::ScalarType::Byte
                                   ? at::ScalarType::Byte
                                   : at::ScalarType::Bool;
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Any>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions),
      result_type);
}

void XLATensor::arange_out(XLATensor& out, const at::Scalar& start,
                           const at::Scalar& end, const at::Scalar& step,
                           at::ScalarType scalar_type) {
  out.SetIrValue(ir::ops::ARange(start, end, step, scalar_type));
  out.SetScalarType(scalar_type);
}

XLATensor XLATensor::argmax(const XLATensor& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
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

XLATensor XLATensor::argmin(const XLATensor& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
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
                                std::vector<int64_t> size,
                                std::vector<int64_t> stride,
                                c10::optional<int64_t> storage_offset) {
  auto input_shape = input.shape();
  return input.CreateViewTensor(CreateAsStridedViewInfo(
      input_shape, std::move(size), std::move(stride), storage_offset));
}

void XLATensor::as_strided_(XLATensor& input, std::vector<int64_t> size,
                            std::vector<int64_t> stride,
                            c10::optional<int64_t> storage_offset) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(ir::MakeNode<ir::ops::AsStrided>(
        input.GetIrValue(), std::move(size), std::move(stride),
        storage_offset.value_or(0)));
  } else {
    auto input_shape = input.shape();
    input.SetSubView(CreateAsStridedViewInfo(
        input_shape, std::move(size), std::move(stride), storage_offset));
  }
}

XLATensor XLATensor::asin(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Asin(input.GetIrValue()));
}

XLATensor XLATensor::asinh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Asinh(input.GetIrValue()));
}

XLATensor XLATensor::atan(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Atan(input.GetIrValue()));
}

XLATensor XLATensor::atanh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Atanh(input.GetIrValue()));
}

XLATensor XLATensor::atan2(const XLATensor& input, const XLATensor& other,
                           c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(
      ir::ops::Atan2(input.GetIrValue(), other.GetIrValue()),
      logical_element_type);
}

XLATensor XLATensor::avg_pool_nd(const XLATensor& input,
                                 int64_t spatial_dim_count,
                                 std::vector<int64_t> kernel_size,
                                 std::vector<int64_t> stride,
                                 std::vector<int64_t> padding, bool ceil_mode,
                                 bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return input.CreateFrom(ir::MakeNode<ir::ops::AvgPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode, count_include_pad));
}

XLATensor XLATensor::avg_pool_nd_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    int64_t spatial_dim_count, std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode,
    bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::AvgPoolNdBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode,
      count_include_pad));
}

XLATensor XLATensor::baddbmm(const XLATensor& input, const XLATensor& batch1,
                             const XLATensor& batch2, const at::Scalar& beta,
                             const at::Scalar& alpha) {
  CheckBmmDimension(/*tag=*/"baddbmm", batch1, batch2);
  ir::Value product_multiplier = XLATensor::GetIrValueForScalar(
      alpha, batch1.shape().get().element_type(), batch1.GetDevice());
  ir::Value bias_multiplier = XLATensor::GetIrValueForScalar(
      beta, input.shape().get().element_type(), input.GetDevice());
  return input.CreateFrom(ir::ops::BaddBmm(
      batch1.GetIrValue(), batch2.GetIrValue(), input.GetIrValue(),
      product_multiplier, bias_multiplier));
}

XLATensor XLATensor::bernoulli(const XLATensor& input, double probability) {
  auto input_shape = input.shape();
  return input.CreateFrom(ir::MakeNode<ir::ops::Bernoulli>(
      GetIrValueForScalar(probability, input_shape, input.GetDevice()),
      GetRngSeed(input.GetDevice()), input_shape.get()));
}

XLATensor XLATensor::bernoulli(const XLATensor& input) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Bernoulli>(
      input.GetIrValue(), GetRngSeed(input.GetDevice()), input.shape().get()));
}

void XLATensor::bernoulli_(XLATensor& input, double probability) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Bernoulli>(
      GetIrValueForScalar(probability, input_shape, input.GetDevice()),
      GetRngSeed(input.GetDevice()), input_shape.get()));
}

void XLATensor::bernoulli_(XLATensor& input, const XLATensor& probability) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Bernoulli>(
      probability.GetIrValue(), GetRngSeed(input.GetDevice()),
      input.shape().get()));
}

XLATensor XLATensor::binary_cross_entropy(const XLATensor& input,
                                          const XLATensor& target,
                                          const XLATensor& weight,
                                          int64_t reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::BinaryCrossEntropy>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetXlaReductionMode(reduction)));
}

XLATensor XLATensor::binary_cross_entropy_backward(const XLATensor& grad_output,
                                                   const XLATensor& input,
                                                   const XLATensor& target,
                                                   const XLATensor& weight,
                                                   int64_t reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::BinaryCrossEntropyBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetOptionalIrValue(weight), GetXlaReductionMode(reduction)));
}

XLATensor XLATensor::bitwise_and(const XLATensor& input,
                                 const at::Scalar& other) {
  CheckIsIntegralOrPred(input.shape(), "__and__");
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::BitwiseAnd(input.GetIrValue(), constant));
}

XLATensor XLATensor::bitwise_and(const XLATensor& input,
                                 const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__and__");
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::bitwise_not_out(XLATensor& out, const XLATensor& input) {
  out.SetIrValue(ir::ops::Not(input.GetIrValue()));
}

void XLATensor::bitwise_or_out(XLATensor& out, const XLATensor& input,
                               const at::Scalar& other) {
  CheckIsIntegralOrPred(input.shape(), "__or__");
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  out.SetIrValue(ir::ops::BitwiseOr(input.GetIrValue(), constant));
}

void XLATensor::bitwise_or_out(XLATensor& out, const XLATensor& input,
                               const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__or__");
  out.SetIrValue(ir::ops::BitwiseOr(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::bitwise_xor_out(XLATensor& out, const XLATensor& input,
                                const at::Scalar& other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  out.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), constant));
}

void XLATensor::bitwise_xor_out(XLATensor& out, const XLATensor& input,
                                const XLATensor& other) {
  CheckIsIntegralOrPred(input.shape(), "__xor__");
  out.SetIrValue(ir::ops::BitwiseXor(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::bmm(const XLATensor& batch1, const XLATensor& batch2) {
  CheckBmmDimension(/*tag=*/"bmm", batch1, batch2);
  return matmul(batch1, batch2);
}

std::vector<XLATensor> XLATensor::broadcast_tensors(
    absl::Span<const XLATensor> tensors) {
  XLA_CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  ir::NodePtr node = ir::ops::BroadcastTensors(tensor_ir_values);
  return tensors.front().MakeOutputTensors(node);
}

XLATensor XLATensor::cat(absl::Span<const XLATensor> tensors, int64_t dim) {
  // Shape checks for cat:
  // - If not empty, every tensor shape must be the same.
  // - Empty tensor passes but is simply ignore in implementation,
  //   e.g. ([2, 3, 5], [])
  // - If empty dimension, other dimensions must be the same.
  //   e.g. ([4, 0, 32, 32], [4, 2, 32, 32], dim=1) passes.
  //   ([4, 0, 32, 32], [4, 2, 31, 32], dim=1) throws.
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  std::vector<xla::Shape> shapes;
  for (size_t i = 0; i < tensors.size(); ++i) {
    xla::Shape tensor_shape = tensors[i].shape();
    if (tensor_shape.rank() == 1 && tensor_shape.dimensions()[0] == 0) {
      continue;
    }
    dim = XlaHelpers::GetCanonicalDimensionIndex(dim, tensor_shape.rank());
    tensor_shape.DeleteDimension(dim);
    if (!shapes.empty()) {
      XLA_CHECK(xla::ShapeUtil::Compatible(shapes.back(), tensor_shape))
          << shapes.back() << " vs. " << tensor_shape;
    }
    shapes.push_back(tensor_shape);
    values.push_back(tensors[i].GetIrValue());
  }
  if (values.empty()) {
    return tensors[0];
  }
  return tensors[0].CreateFrom(ir::MakeNode<ir::ops::Cat>(values, dim));
}

XLATensor XLATensor::ceil(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Ceil(input.GetIrValue()));
}

XLATensor XLATensor::cholesky(const XLATensor& input, bool upper) {
  // Cholesky takes lower instead of upper, hence the negation.
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Cholesky>(input.GetIrValue(), !upper));
}

XLATensor XLATensor::clamp(const XLATensor& input,
                           const c10::optional<at::Scalar>& min,
                           const c10::optional<at::Scalar>& max) {
  MinMaxValues min_max = GetMinMaxValues(input, min, max);
  return input.CreateFrom(
      ir::ops::Clamp(input.GetIrValue(), min_max.min, min_max.max));
}

XLATensor XLATensor::clamp(const XLATensor& input,
                           const c10::optional<at::Tensor>& min,
                           const c10::optional<at::Tensor>& max) {
  XLA_CHECK(min || max)
      << "At least one of \'min\' or \'max\' must not be None";
  ir::Value res = input.GetIrValue();
  if (min) {
    res = ir::ops::Max(res, bridge::GetXlaTensor(*min).GetIrValue());
  }
  if (max) {
    res = ir::ops::Min(res, bridge::GetXlaTensor(*max).GetIrValue());
  }
  return input.CreateFrom(res);
}

void XLATensor::clamp_out(XLATensor& out, const XLATensor& input,
                          const c10::optional<at::Tensor>& min,
                          const c10::optional<at::Tensor>& max) {
  XLA_CHECK(min || max)
      << "At least one of \'min\' or \'max\' must not be None";
  ir::Value res = input.GetIrValue();
  if (min) {
    res = ir::ops::Max(res, bridge::GetXlaTensor(*min).GetIrValue());
  }
  if (max) {
    res = ir::ops::Min(res, bridge::GetXlaTensor(*max).GetIrValue());
  }
  out.SetInPlaceIrValue(res);
}

XLATensor XLATensor::clone(const XLATensor& input) {
  return input.CreateFrom(input.GetIrValue());
}

XLATensor XLATensor::constant_pad_nd(const XLATensor& input,
                                     absl::Span<const int64_t> pad,
                                     const at::Scalar& value) {
  std::vector<int64_t> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().get().rank());
  return input.CreateFrom(ir::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value));
}

XLATensor XLATensor::convolution_overrideable(
    const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  ir::NodePtr ir_value = ir::MakeNode<ir::ops::ConvolutionOverrideable>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      std::move(stride), std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups);
  return input.CreateFrom(ir_value);
}

XLATensor XLATensor::convolution_overrideable(
    const XLATensor& input, const XLATensor& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  ir::NodePtr ir_value = ir::MakeNode<ir::ops::ConvolutionOverrideable>(
      input.GetIrValue(), weight.GetIrValue(), std::move(stride),
      std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups);
  return input.CreateFrom(ir_value);
}

std::tuple<XLATensor, XLATensor, XLATensor>
XLATensor::convolution_backward_overrideable(
    const XLATensor& out_backprop, const XLATensor& input,
    const XLATensor& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups) {
  ir::NodePtr node = ir::MakeNode<ir::ops::ConvolutionBackwardOverrideable>(
      out_backprop.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      std::move(stride), std::move(padding), std::move(dilation), transposed,
      std::move(output_padding), groups);
  XLATensor grad_input = out_backprop.CreateFrom(ir::Value(node, 0));
  XLATensor grad_weight = out_backprop.CreateFrom(ir::Value(node, 1));
  XLATensor grad_bias = out_backprop.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::cos(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Cos(input.GetIrValue()));
}

XLATensor XLATensor::cosh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Cosh(input.GetIrValue()));
}

XLATensor XLATensor::cross(const XLATensor& input, const XLATensor& other,
                           c10::optional<int64_t> dim) {
  return tensor_ops::Cross(input, other, dim);
}

XLATensor XLATensor::cumprod(const XLATensor& input, int64_t dim,
                             c10::optional<at::ScalarType> dtype) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::CumProd>(input.GetIrValue(), canonical_dim, dtype),
      dtype);
}

XLATensor XLATensor::cumsum(const XLATensor& input, int64_t dim,
                            c10::optional<at::ScalarType> dtype) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::CumSum>(input.GetIrValue(), canonical_dim, dtype),
      dtype);
}

XLATensor XLATensor::diag(const XLATensor& input, int64_t offset) {
  int64_t rank = input.shape().get().rank();
  XLA_CHECK(rank == 1 || rank == 2)
      << "Invalid argument for diag: matrix or a vector expected";
  if (rank == 1) {
    return tensor_ops::MakeMatrixWithDiagonal(input, offset);
  }
  return diagonal(input, offset, /*dim1=*/-2, /*dim2=*/-1);
}

XLATensor XLATensor::diagonal(const XLATensor& input, int64_t offset,
                              int64_t dim1, int64_t dim2) {
  auto input_shape = input.shape();
  int64_t canonical_dim1 =
      XlaHelpers::GetCanonicalDimensionIndex(dim1, input.shape().get().rank());
  int64_t canonical_dim2 =
      XlaHelpers::GetCanonicalDimensionIndex(dim2, input.shape().get().rank());
  DiagonalInfo diagonal_info;
  diagonal_info.offset = offset;
  diagonal_info.dim1 = canonical_dim1;
  diagonal_info.dim2 = canonical_dim2;
  ViewInfo view_info(ViewInfo::Type::kDiagonal, input_shape,
                     std::move(diagonal_info));
  return input.CreateViewTensor(std::move(view_info));
}

XLATensor XLATensor::div(const XLATensor& input, const XLATensor& other,
                         const c10::optional<c10::string_view>& rounding_mode,
                         c10::optional<at::ScalarType> logical_element_type) {
  at::ScalarType scalar_type =
      at::typeMetaToScalarType(c10::get_default_dtype());
  xla::PrimitiveType input_type = input.shape().get().element_type();
  xla::PrimitiveType other_type = other.shape().get().element_type();
  bool input_is_float = xla::primitive_util::IsFloatingPointType(input_type);
  bool other_is_float = xla::primitive_util::IsFloatingPointType(other_type);
  if (input_is_float && !other_is_float) {
    scalar_type = TensorTypeFromXlaType(input_type);
  } else if (!input_is_float && other_is_float) {
    scalar_type = TensorTypeFromXlaType(other_type);
  }
  // We need to cast both input and other to float to perform true divide, floor
  // divide and trunc divide.
  ir::Value input_value = GetFloatingIrValue(input, scalar_type);
  ir::Value other_value = GetFloatingIrValue(other, scalar_type);
  ir::Value res = input_value / other_value;

  if (rounding_mode.has_value()) {
    if (*rounding_mode == "trunc") {
      res = ir::ops::Trunc(res);
    } else if (*rounding_mode == "floor") {
      res = ir::ops::Floor(res);
    } else {
      XLA_CHECK(false)
          << "rounding_mode must be one of None, 'trunc', or 'floor'";
    }
  }

  // Promote the result to the logical_element_type if one of the
  // input and the other is float. If that is not the case logical_element_type
  // will be non-floating-point type, we should only promote the result to that
  // when rounding_mode is not nullopt.
  if (input_is_float || other_is_float || rounding_mode.has_value()) {
    if (logical_element_type.has_value()) {
      xla::PrimitiveType res_intended_type =
          MakeXlaPrimitiveType(*logical_element_type, &input.GetDevice());
      if (res.shape().element_type() != res_intended_type) {
        res = ir::MakeNode<ir::ops::Cast>(res, res_intended_type);
      }
    }
    return input.CreateFrom(res, logical_element_type);
  } else {
    // We don't need to typecheck the res IR here since we cast both input and
    // output to the scalar_type. Res type must also be scalar_type here.
    return input.CreateFrom(res, scalar_type);
  }
}

XLATensor XLATensor::div(const XLATensor& input, const at::Scalar& other) {
  at::ScalarType scalar_type =
      at::typeMetaToScalarType(c10::get_default_dtype());
  ir::Value input_value = GetFloatingIrValue(input, scalar_type);
  ir::Value other_value = GetIrValueForScalar(
      other, input_value.shape().element_type(), input.GetDevice());
  return input.CreateFrom(input_value / other_value, scalar_type);
}

XLATensor XLATensor::eq(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

XLATensor XLATensor::eq(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

XLATensor XLATensor::elu(const XLATensor& input, const at::Scalar& alpha,
                         const at::Scalar& scale,
                         const at::Scalar& input_scale) {
  return input.CreateFrom(
      ir::ops::Elu(input.GetIrValue(), alpha, scale, input_scale));
}

void XLATensor::elu_(XLATensor& input, const at::Scalar& alpha,
                     const at::Scalar& scale, const at::Scalar& input_scale) {
  input.SetInPlaceIrValue(
      ir::ops::Elu(input.GetIrValue(), alpha, scale, input_scale));
}

XLATensor XLATensor::elu_backward(const XLATensor& grad_output,
                                  const at::Scalar& alpha,
                                  const at::Scalar& scale,
                                  const at::Scalar& input_scale,
                                  const XLATensor& output) {
  return grad_output.CreateFrom(ir::ops::EluBackward(grad_output.GetIrValue(),
                                                     output.GetIrValue(), alpha,
                                                     scale, input_scale));
}

XLATensor XLATensor::embedding_dense_backward(const XLATensor& grad_output,
                                              const XLATensor& indices,
                                              int64_t num_weights,
                                              int64_t padding_idx,
                                              bool scale_grad_by_freq) {
  return tensor_ops::EmbeddingDenseBackward(grad_output, indices, num_weights,
                                            padding_idx, scale_grad_by_freq);
}

XLATensor XLATensor::erf(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erf(input.GetIrValue()));
}

XLATensor XLATensor::erfc(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erfc(input.GetIrValue()));
}

XLATensor XLATensor::erfinv(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erfinv(input.GetIrValue()));
}

XLATensor XLATensor::exp(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Exp(input.GetIrValue()));
}

XLATensor XLATensor::expand(const XLATensor& input, std::vector<int64_t> size) {
  auto input_shape = input.shape();
  return input.CreateFrom(ir::MakeNode<ir::ops::Expand>(
      input.GetIrValue(),
      GetExpandDimensions(input_shape.get(), std::move(size))));
}

XLATensor XLATensor::expm1(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Expm1(input.GetIrValue()));
}

void XLATensor::exponential_(XLATensor& input, double lambd) {
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Exponential>(
      GetIrValueForScalar(lambd, input_shape.get().element_type(),
                          input.GetDevice()),
      GetRngSeed(input.GetDevice()), input_shape.get()));
}

XLATensor XLATensor::eye(int64_t lines, int64_t cols, const Device& device,
                         at::ScalarType element_type) {
  return XLATensor::Create(
      ir::ops::Identity(lines, cols,
                        MakeXlaPrimitiveType(element_type, &device)),
      device, element_type);
}

void XLATensor::eye_out(XLATensor& out, int64_t lines, int64_t cols) {
  out.SetIrValue(
      ir::ops::Identity(lines, cols >= 0 ? cols : lines,
                        GetDevicePrimitiveType(out.shape().get().element_type(),
                                               &out.GetDevice())));
}

void XLATensor::fill_(XLATensor& input, const at::Scalar& value) {
  ir::Value constant =
      GetIrValueForScalar(value, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

XLATensor XLATensor::flip(const XLATensor& input,
                          absl::Span<const int64_t> dims) {
  auto dimensions = XlaHelpers::GetCanonicalDimensionIndices(
      dims, input.shape().get().rank());
  std::set<int64_t> unique_dims(dimensions.begin(), dimensions.end());
  XLA_CHECK_EQ(unique_dims.size(), dimensions.size());
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Flip>(input.GetIrValue(), dimensions));
}

XLATensor XLATensor::floor(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Floor(input.GetIrValue()));
}

XLATensor XLATensor::fmod(const XLATensor& input, const XLATensor& other,
                          c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

XLATensor XLATensor::fmod(const XLATensor& input, const at::Scalar& other,
                          c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), constant),
                          logical_element_type);
}

XLATensor XLATensor::frac(const XLATensor& input) {
  return input.CreateFrom(ir::ops::FracOp(input.GetIrValue()));
}

XLATensor XLATensor::full(absl::Span<const int64_t> size,
                          const at::Scalar& fill_value, const Device& device,
                          at::ScalarType scalar_type) {
  CheckShapeDimensions(size);
  xla::Shape shape = MakeArrayShapeFromDimensions(
      size, /*dynamic_dimensions=*/{},
      MakeXlaPrimitiveType(scalar_type, &device), device.hw_type);
  return Create(GetIrValueForScalar(fill_value, shape, device), device,
                scalar_type);
}

XLATensor XLATensor::full_like(const XLATensor& input,
                               const at::Scalar& fill_value,
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

XLATensor XLATensor::gather(const XLATensor& input, int64_t dim,
                            const XLATensor& index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Gather>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index.GetIrValue()));
}

XLATensor XLATensor::ge(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

XLATensor XLATensor::ge(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

XLATensor XLATensor::gelu(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Gelu(input.GetIrValue()));
}

XLATensor XLATensor::gelu_backward(const XLATensor& grad,
                                   const XLATensor& input) {
  return input.CreateFrom(
      ir::ops::GeluBackward(grad.GetIrValue(), input.GetIrValue()));
}

XLATensor XLATensor::ger(const XLATensor& input, const XLATensor& vec2) {
  return input.CreateFrom(ir::ops::Ger(input.GetIrValue(), vec2.GetIrValue()));
}

XLATensor XLATensor::gt(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

XLATensor XLATensor::gt(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

XLATensor XLATensor::index(const XLATensor& input,
                           absl::Span<const XLATensor> indices,
                           int64_t start_dim) {
  return IndexByTensors(input, indices, start_dim);
}

XLATensor XLATensor::index_add(const XLATensor& input, int64_t dim,
                               const XLATensor& index, const XLATensor& source,
                               const at::Scalar& alpha) {
  ir::Value constant = GetIrValueForScalar(
      alpha, source.shape().get().element_type(), input.GetDevice());
  auto scaled_source = input.CreateFrom(source.GetIrValue() * constant);
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexAdd(input, canonical_dim, index, scaled_source));
}

XLATensor XLATensor::index_copy(const XLATensor& input, int64_t dim,
                                const XLATensor& index,
                                const XLATensor& source) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexCopy(input, canonical_dim, index, source));
}

void XLATensor::index_copy_(XLATensor& input, int64_t dim,
                            const XLATensor& index, const XLATensor& source) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexCopy(input, canonical_dim, index, source));
}

XLATensor XLATensor::index_fill(const XLATensor& input, int64_t dim,
                                const XLATensor& index,
                                const at::Scalar& value) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

XLATensor XLATensor::index_fill(const XLATensor& input, int64_t dim,
                                const XLATensor& index,
                                const XLATensor& value) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  return input.CreateFrom(IndexFill(input, canonical_dim, index, value));
}

void XLATensor::index_fill_(XLATensor& input, int64_t dim,
                            const XLATensor& index, const XLATensor& value) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

void XLATensor::index_fill_(XLATensor& input, int64_t dim,
                            const XLATensor& index, const at::Scalar& value) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  input.SetIrValue(IndexFill(input, canonical_dim, index, value));
}

XLATensor XLATensor::index_put(const XLATensor& input,
                               absl::Span<const XLATensor> indices,
                               int64_t start_dim, const XLATensor& values,
                               bool accumulate,
                               absl::Span<const int64_t> result_permutation) {
  return input.CreateFrom(IndexPutByTensors(input, indices, start_dim, values,
                                            accumulate, result_permutation));
}

void XLATensor::index_put_(XLATensor& input, const XLATensor& canonical_base,
                           absl::Span<const XLATensor> indices,
                           int64_t start_dim, const XLATensor& values,
                           bool accumulate,
                           absl::Span<const int64_t> result_permutation) {
  input.SetIrValue(IndexPutByTensors(canonical_base, indices, start_dim, values,
                                     accumulate, result_permutation));
}

XLATensor XLATensor::index_select(const XLATensor& input, int64_t dim,
                                  const XLATensor& index) {
  ir::Value index_value = EnsureRank1(index.GetIrValue());
  return input.CreateFrom(ir::MakeNode<ir::ops::IndexSelect>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      index_value));
}

XLATensor XLATensor::inverse(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Inverse(input.GetIrValue()));
}

XLATensor XLATensor::isnan(const XLATensor& input) {
  ir::Value result = ir::ops::IsNan(input.GetIrValue());
  ir::Value casted = GetBooleanIrValue(result);
  return input.CreateFrom(casted, at::ScalarType::Bool);
}

XLATensor XLATensor::kl_div_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const XLATensor& target, int64_t reduction,
                                     bool log_target) {
  return tensor_ops::KlDivBackward(grad_output, input, target,
                                   GetXlaReductionMode(reduction), log_target);
}

std::tuple<XLATensor, XLATensor> XLATensor::kthvalue(const XLATensor& input,
                                                     int64_t k, int64_t dim,
                                                     bool keepdim) {
  ir::NodePtr node = ir::MakeNode<ir::ops::KthValue>(
      input.GetIrValue(), k,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::l1_loss(const XLATensor& input, const XLATensor& target,
                             int64_t reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::L1Loss>(
      input.GetIrValue(), target.GetIrValue(), GetXlaReductionMode(reduction)));
}

XLATensor XLATensor::l1_loss_backward(const XLATensor& grad_output,
                                      const XLATensor& input,
                                      const XLATensor& target,
                                      int64_t reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::L1LossBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetXlaReductionMode(reduction)));
}

XLATensor XLATensor::le(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

XLATensor XLATensor::le(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

XLATensor XLATensor::hardshrink(const XLATensor& input,
                                const at::Scalar& lambda) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Hardshrink>(input.GetIrValue(), lambda));
}

XLATensor XLATensor::hardshrink_backward(const XLATensor& grad_out,
                                         const XLATensor& input,
                                         const at::Scalar& lambda) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ShrinkBackward>(
      ir::OpKind(at::aten::hardshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

XLATensor XLATensor::hardsigmoid(const XLATensor& input) {
  return input.CreateFrom(ir::ops::HardSigmoid(input.GetIrValue()));
}

XLATensor XLATensor::hardsigmoid_backward(const XLATensor& grad_output,
                                          const XLATensor& input) {
  return input.CreateFrom(ir::ops::HardSigmoidBackward(grad_output.GetIrValue(),
                                                       input.GetIrValue()));
}

XLATensor XLATensor::hardtanh_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const at::Scalar& min_val,
                                       const at::Scalar& max_val) {
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

XLATensor XLATensor::lerp(const XLATensor& input, const XLATensor& end,
                          const XLATensor& weight) {
  return input.CreateFrom(
      ir::ops::Lerp(input.GetIrValue(), end.GetIrValue(), weight.GetIrValue()));
}

XLATensor XLATensor::lerp(const XLATensor& input, const XLATensor& end,
                          const at::Scalar& weight) {
  ir::Value weight_val = GetIrValueForScalar(
      weight, input.shape().get().element_type(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::Lerp(input.GetIrValue(), end.GetIrValue(), weight_val));
}

XLATensor XLATensor::log(const XLATensor& input) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of log(int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input.CreateFrom(
      ir::ops::Log(GetFloatingIrValue(input, at::ScalarType::Float)),
      c10::nullopt);
}

XLATensor XLATensor::log_base(const XLATensor& input, ir::OpKind op,
                              double base) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of logbase(int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input.CreateFrom(
      ir::ops::LogBase(GetFloatingIrValue(input, at::ScalarType::Float), op,
                       base),
      c10::nullopt);
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

XLATensor XLATensor::log_softmax(const XLATensor& input, int64_t dim,
                                 c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LogSoftmax>(input.GetIrValue(),
                                        XlaHelpers::GetCanonicalDimensionIndex(
                                            dim, input.shape().get().rank()),
                                        dtype),
      dtype);
}

XLATensor XLATensor::log_softmax_backward(const XLATensor& grad_output,
                                          const XLATensor& output,
                                          int64_t dim) {
  return grad_output.CreateFrom(ir::ops::LogSoftmaxBackwardOp(
      grad_output.GetIrValue(), output.GetIrValue(), dim));
}

XLATensor XLATensor::log1p(const XLATensor& input) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of log1p(int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input.CreateFrom(
      ir::ops::Log1p(GetFloatingIrValue(input, at::ScalarType::Float)),
      c10::nullopt);
}

void XLATensor::log1p_(XLATensor& input) {
  input.SetInPlaceIrValue(ir::ops::Log1p(input.GetIrValue()));
}

XLATensor XLATensor::logdet(const XLATensor& input) {
  return input.CreateFrom(ir::ops::LogDet(input.GetIrValue()));
}

XLATensor XLATensor::logical_not(const XLATensor& input) {
  return input.CreateFrom(ir::ops::LogicalNot(input.GetIrValue()),
                          at::ScalarType::Bool);
}

XLATensor XLATensor::logical_xor(const XLATensor& input,
                                 const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::LogicalXor(input.GetIrValue(), other.GetIrValue()),
      at::ScalarType::Bool);
}

XLATensor XLATensor::logical_and(const XLATensor& input,
                                 const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::LogicalAnd(input.GetIrValue(), other.GetIrValue()),
      at::ScalarType::Bool);
}

XLATensor XLATensor::logical_or(const XLATensor& input,
                                const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::LogicalOr(input.GetIrValue(), other.GetIrValue()),
      at::ScalarType::Bool);
}

XLATensor XLATensor::logsumexp(const XLATensor& input,
                               std::vector<int64_t> dimensions,
                               bool keep_reduced_dimensions) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Logsumexp>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndices(dimensions,
                                               input.shape().get().rank()),
      keep_reduced_dimensions));
}

XLATensor XLATensor::xlogy(const XLATensor& input, const XLATensor& other) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of xlogy(int,int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input.CreateFrom(
      ir::ops::XLogY(input.GetIrValue(),
                     GetFloatingIrValue(other, at::ScalarType::Float)),
      c10::nullopt);
}

XLATensor XLATensor::lt(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

XLATensor XLATensor::lt(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

void XLATensor::masked_fill_(XLATensor& input, const XLATensor& mask,
                             const at::Scalar& value) {
  ir::ScopePusher ir_scope(at::aten::masked_fill.toQualString());
  input.SetIrValue(ir::MakeNode<ir::ops::MaskedFill>(
      input.GetIrValue(), MaybeExpand(mask.GetIrValue(), input.shape()),
      value));
}

void XLATensor::masked_scatter_(XLATensor& input, const XLATensor& mask,
                                const XLATensor& source) {
  ir::ScopePusher ir_scope(at::aten::masked_scatter.toQualString());
  input.SetIrValue(ir::MakeNode<ir::ops::MaskedScatter>(
      input.GetIrValue(), MaybeExpand(mask.GetIrValue(), input.shape()),
      source.GetIrValue()));
}

XLATensor XLATensor::masked_select(const XLATensor& input,
                                   const XLATensor& mask) {
  ir::NodePtr node = ir::MakeNode<ir::ops::MaskedSelect>(input.GetIrValue(),
                                                         mask.GetIrValue());
  return input.CreateFrom(ir::Value(node, 0));
}

XLATensor XLATensor::matmul(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::MatMul(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::max(const XLATensor& input, const XLATensor& other,
                         c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Max(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

XLATensor XLATensor::max(const XLATensor& input) {
  return input.CreateFrom(ir::ops::MaxUnary(input.GetIrValue()), input.dtype());
}

std::tuple<XLATensor, XLATensor> XLATensor::max(const XLATensor& input,
                                                int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

void XLATensor::max_out(XLATensor& max, XLATensor& max_values,
                        const XLATensor& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  max.SetIrValue(ir::Value(node, 0));
  max_values.SetIrValue(ir::Value(node, 1));
}

std::tuple<XLATensor, XLATensor> XLATensor::max_pool_nd(
    const XLATensor& input, int64_t spatial_dim_count,
    std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
    std::vector<int64_t> padding, bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  ir::NodePtr node = ir::MakeNode<ir::ops::MaxPoolNd>(
      input.GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::max_pool_nd_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    int64_t spatial_dim_count, std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::MaxPoolNdBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding),
      ceil_mode));
}

XLATensor XLATensor::max_unpool(const XLATensor& input,
                                const XLATensor& indices,
                                std::vector<int64_t> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MaxUnpoolNd>(
      input.GetIrValue(), indices.GetIrValue(), std::move(output_size)));
}

XLATensor XLATensor::max_unpool_backward(const XLATensor& grad_output,
                                         const XLATensor& input,
                                         const XLATensor& indices,
                                         std::vector<int64_t> output_size) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::MaxUnpoolNdBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), indices.GetIrValue(),
      std::move(output_size)));
}

XLATensor XLATensor::mean(const XLATensor& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Mean>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions, dtype),
      dtype);
}

XLATensor XLATensor::min(const XLATensor& input, const XLATensor& other,
                         c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(ir::ops::Min(input.GetIrValue(), other.GetIrValue()),
                          logical_element_type);
}

XLATensor XLATensor::min(const XLATensor& input) {
  return input.CreateFrom(ir::ops::MinUnary(input.GetIrValue()), input.dtype());
}

std::tuple<XLATensor, XLATensor> XLATensor::min(const XLATensor& input,
                                                int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MinInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

void XLATensor::min_out(XLATensor& min, XLATensor& min_indices,
                        const XLATensor& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::MinInDim>(input.GetIrValue(),
                                                     canonical_dim, keepdim);
  min.SetIrValue(ir::Value(node, 0));
  min_indices.SetIrValue(ir::Value(node, 1));
}

XLATensor XLATensor::mish(const XLATensor& input) {
  return input.CreateFrom(
      input.GetIrValue() *
      ir::ops::Tanh(tensor_ops::Softplus(input, 1, 20).GetIrValue()));
}

XLATensor XLATensor::mm(const XLATensor& input, const XLATensor& weight) {
  return input.CreateFrom(
      ir::ops::Dot(input.GetIrValue(), weight.GetIrValue()));
}

XLATensor XLATensor::mse_loss(const XLATensor& input, const XLATensor& target,
                              int64_t reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MseLoss>(
      input.GetIrValue(), target.GetIrValue(), GetXlaReductionMode(reduction)));
}

XLATensor XLATensor::mse_loss_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const XLATensor& target,
                                       int64_t reduction) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MseLossBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetXlaReductionMode(reduction)));
}

XLATensor XLATensor::mul(const XLATensor& input, const XLATensor& other,
                         c10::optional<at::ScalarType> logical_element_type) {
  return input.CreateFrom(input.GetIrValue() * other.GetIrValue(),
                          logical_element_type);
}

XLATensor XLATensor::mul(const XLATensor& input, const at::Scalar& other,
                         c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() * constant, logical_element_type);
}

XLATensor XLATensor::mv(const XLATensor& input, const XLATensor& vec) {
  return input.CreateFrom(ir::ops::Dot(input.GetIrValue(), vec.GetIrValue()));
}

void XLATensor::mv_out(XLATensor& out, const XLATensor& input,
                       const XLATensor& vec) {
  out.SetIrValue(ir::ops::Dot(input.GetIrValue(), vec.GetIrValue()));
}

XLATensor XLATensor::nan_to_num(const XLATensor& input, const at::Scalar& nan,
                                const at::Scalar& posinf,
                                const at::Scalar& neginf) {
  ir::Value nan_value =
      GetIrValueForScalar(nan, input.shape(), input.GetDevice());
  ir::Value posinf_value =
      GetIrValueForScalar(posinf, input.shape(), input.GetDevice());
  ir::Value neginf_value =
      GetIrValueForScalar(neginf, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::NanToNum(input.GetIrValue(), nan_value,
                                            posinf_value, neginf_value));
}

XLATensor XLATensor::narrow(const XLATensor& input, int64_t dim, int64_t start,
                            int64_t length) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  xla::Shape narrow_shape = input_shape;
  narrow_shape.set_dimensions(dim, length);

  ViewInfo::Type view_type = (xla::ShapeUtil::ElementsIn(input_shape) ==
                              xla::ShapeUtil::ElementsIn(narrow_shape))
                                 ? ViewInfo::Type::kReshape
                                 : ViewInfo::Type::kNarrow;
  ViewInfo view_info(view_type, std::move(narrow_shape), input_shape);
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
      GetIrValueOrDefault(running_var, 0, features_shape, input.GetDevice());
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormForward>(
      input.GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, eps);
  XLATensor output = input.CreateFrom(ir::Value(node, 0));
  XLATensor mean;
  XLATensor variance_inverse;
  if (training) {
    mean = input.CreateFrom(ir::Value(node, 1));
    variance_inverse = input.CreateFrom(ir::Value(node, 3));
    if (!running_mean.is_null()) {
      running_mean.SetIrValue(ir::MakeNode<ir::ops::LinearInterpolation>(
          mean.GetIrValue(), running_mean.GetIrValue(), momentum));
    }
    if (!running_var.is_null()) {
      running_var.SetIrValue(ir::MakeNode<ir::ops::LinearInterpolation>(
          ir::Value(node, 2), running_var.GetIrValue(), momentum));
    }
  } else {
    at::Tensor at_input = bridge::AtenFromXlaTensor(input);
    mean = bridge::GetXlaTensor(at::empty({0}, at_input.options()));
    variance_inverse = bridge::GetXlaTensor(at::empty({0}, at_input.options()));
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

XLATensor XLATensor::ne(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

XLATensor XLATensor::ne(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

XLATensor XLATensor::neg(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Neg(input.GetIrValue()));
}

XLATensor XLATensor::nll_loss(const XLATensor& input, const XLATensor& target,
                              const XLATensor& weight, int64_t reduction,
                              int ignore_index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLoss>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetXlaReductionMode(reduction), ignore_index));
}

XLATensor XLATensor::nll_loss2d(const XLATensor& input, const XLATensor& target,
                                const XLATensor& weight, int64_t reduction,
                                int ignore_index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLoss2d>(
      input.GetIrValue(), target.GetIrValue(), GetOptionalIrValue(weight),
      GetXlaReductionMode(reduction), ignore_index));
}

XLATensor XLATensor::nll_loss2d_backward(const XLATensor& grad_output,
                                         const XLATensor& input,
                                         const XLATensor& target,
                                         const XLATensor& weight,
                                         int64_t reduction, int ignore_index,
                                         const XLATensor& total_weight) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLoss2dBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetXlaReductionMode(reduction), ignore_index));
}

XLATensor XLATensor::nll_loss_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const XLATensor& target,
                                       const XLATensor& weight,
                                       int64_t reduction, int ignore_index,
                                       const XLATensor& total_weight) {
  return input.CreateFrom(ir::MakeNode<ir::ops::NllLossBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), target.GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetXlaReductionMode(reduction), ignore_index));
}

std::pair<XLATensor, XLATensor> XLATensor::nms(const XLATensor& boxes,
                                               const XLATensor& scores,
                                               const XLATensor& score_threshold,
                                               const XLATensor& iou_threshold,
                                               int64_t output_size) {
  ir::NodePtr node = ir::MakeNode<ir::ops::Nms>(
      boxes.GetIrValue(), scores.GetIrValue(), score_threshold.GetIrValue(),
      iou_threshold.GetIrValue(), output_size);
  return std::pair<XLATensor, XLATensor>(
      Create(ir::Value(node, 0), boxes.GetDevice(), at::ScalarType::Int),
      Create(ir::Value(node, 1), boxes.GetDevice(), at::ScalarType::Int));
}

XLATensor XLATensor::nonzero(const XLATensor& input) {
  ir::NodePtr node = ir::MakeNode<ir::ops::NonZero>(input.GetIrValue());
  return input.CreateFrom(ir::Value(node, 0), at::ScalarType::Long);
}

XLATensor XLATensor::norm(const XLATensor& input,
                          const c10::optional<at::Scalar>& p,
                          c10::optional<at::ScalarType> dtype,
                          at::IntArrayRef dim, bool keepdim) {
  auto canonical_dims = XlaHelpers::GetCanonicalDimensionIndices(
      XlaHelpers::I64List(dim), input.shape().get().rank());
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::ops::Norm(input.GetIrValue(), p, dtype, canonical_dims, keepdim));
}

XLATensor XLATensor::normal(double mean, const XLATensor& std) {
  return std.CreateFrom(ir::MakeNode<ir::ops::Normal>(
      GetIrValueForScalar(mean, std.shape(), std.GetDevice()), std.GetIrValue(),
      GetRngSeed(std.GetDevice())));
}

XLATensor XLATensor::normal(const XLATensor& mean, double std) {
  return mean.CreateFrom(ir::MakeNode<ir::ops::Normal>(
      mean.GetIrValue(),
      GetIrValueForScalar(std, mean.shape(), mean.GetDevice()),
      GetRngSeed(mean.GetDevice())));
}

XLATensor XLATensor::normal(const XLATensor& mean, const XLATensor& std) {
  return mean.CreateFrom(ir::MakeNode<ir::ops::Normal>(
      mean.GetIrValue(), MaybeExpand(std.GetIrValue(), mean.shape()),
      GetRngSeed(mean.GetDevice())));
}

void XLATensor::normal_(XLATensor& input, double mean, double std) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Normal>(
      GetIrValueForScalar(mean, input.shape(), input.GetDevice()),
      GetIrValueForScalar(std, input.shape(), input.GetDevice()),
      GetRngSeed(input.GetDevice())));
}

XLATensor XLATensor::not_supported(std::string description, xla::Shape shape,
                                   const Device& device) {
  return Create(ir::MakeNode<ir::ops::NotSupported>(std::move(description),
                                                    std::move(shape)),
                device);
}

XLATensor XLATensor::permute(const XLATensor& input,
                             absl::Span<const int64_t> dims) {
  auto input_shape = input.shape();
  ViewInfo view_info(
      ViewInfo::Type::kPermute, input_shape,
      XlaHelpers::GetCanonicalDimensionIndices(dims, input_shape.get().rank()));
  return input.CreateViewTensor(std::move(view_info));
}

XLATensor XLATensor::pow(const XLATensor& input, const at::Scalar& exponent) {
  // We want to pass exponent_node as a constant to give XLA more room to
  // optimize
  ir::Value exponent_node = GetIrValueForConstant(exponent, input.shape());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

XLATensor XLATensor::pow(const XLATensor& input, const XLATensor& exponent) {
  return input.CreateFrom(
      ir::ops::Pow(input.GetIrValue(), exponent.GetIrValue()));
}

XLATensor XLATensor::pow(const at::Scalar& input, const XLATensor& exponent) {
  ir::Value input_node =
      GetIrValueForScalar(input, exponent.shape(), exponent.GetDevice());
  return exponent.CreateFrom(ir::ops::Pow(input_node, exponent.GetIrValue()));
}

XLATensor XLATensor::prelu(const XLATensor& input, const XLATensor& weight) {
  return input.CreateFrom(
      ir::ops::Prelu(input.GetIrValue(), weight.GetIrValue()));
}

XLATensor XLATensor::prod(const XLATensor& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Prod>(input.GetIrValue(),
                                  XlaHelpers::GetCanonicalDimensionIndices(
                                      dimensions, input.shape().get().rank()),
                                  keep_reduced_dimensions, dtype),
      dtype);
}

void XLATensor::put_(XLATensor& input, const XLATensor& index,
                     const XLATensor& source, bool accumulate) {
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Put>(
      input.GetIrValue(), index.GetIrValue(), source.GetIrValue(), accumulate));
}

std::tuple<XLATensor, XLATensor> XLATensor::qr(const XLATensor& input,
                                               bool some) {
  ir::NodePtr node = ir::MakeNode<ir::ops::QR>(input.GetIrValue(), some);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

void XLATensor::random_(XLATensor& input, int64_t from, int64_t to) {
  XLA_CHECK_LE(from, to);
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::DiscreteUniform>(
      GetIrValueForScalar(from, xla::PrimitiveType::S64, input.GetDevice()),
      GetIrValueForScalar(to, xla::PrimitiveType::S64, input.GetDevice()),
      GetRngSeed(input.GetDevice()), input_shape));
}

XLATensor XLATensor::reciprocal(const XLATensor& input) {
  return input.CreateFrom(ir::ops::ReciprocalOp(input.GetIrValue()));
}

XLATensor XLATensor::reflection_pad2d(const XLATensor& input,
                                      std::vector<int64_t> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReflectionPad2d>(
      input.GetIrValue(), std::move(padding)));
}

XLATensor XLATensor::reflection_pad2d_backward(const XLATensor& grad_output,
                                               const XLATensor& input,
                                               std::vector<int64_t> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReflectionPad2dBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

XLATensor XLATensor::relu(const XLATensor& input) {
  return input.CreateFrom(ir::ops::ReluOp(input.GetIrValue()));
}

void XLATensor::relu_(XLATensor& input) {
  input.SetInPlaceIrValue(ir::ops::ReluOp(input.GetIrValue()));
}

XLATensor XLATensor::remainder(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Remainder(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::remainder(const XLATensor& input,
                               const at::Scalar& other) {
  ir::Value constant =
      GetIrValueForScalar(other, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::ops::Remainder(input.GetIrValue(), constant));
}

XLATensor XLATensor::repeat(const XLATensor& input,
                            std::vector<int64_t> repeats) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Repeat>(input.GetIrValue(), std::move(repeats)));
}

XLATensor XLATensor::replication_pad1d(const XLATensor& input,
                                       std::vector<int64_t> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPad>(
      input.GetIrValue(), std::move(padding)));
}

XLATensor XLATensor::replication_pad1d_backward(const XLATensor& grad_output,
                                                const XLATensor& input,
                                                std::vector<int64_t> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPadBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

XLATensor XLATensor::replication_pad2d(const XLATensor& input,
                                       std::vector<int64_t> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPad>(
      input.GetIrValue(), std::move(padding)));
}

XLATensor XLATensor::replication_pad2d_backward(const XLATensor& grad_output,
                                                const XLATensor& input,
                                                std::vector<int64_t> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ReplicationPadBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), std::move(padding)));
}

void XLATensor::resize_(XLATensor& input, std::vector<int64_t> size) {
  if (input.data()->view == nullptr) {
    input.SetIrValue(
        ir::MakeNode<ir::ops::Resize>(input.GetIrValue(), std::move(size)));
  } else {
    auto input_shape = input.shape();
    xla::Shape resize_shape =
        xla::ShapeUtil::MakeShape(input_shape.get().element_type(), size);
    ViewInfo view_info(ViewInfo::Type::kResize, std::move(resize_shape),
                       input_shape);
    input.SetSubView(std::move(view_info));
  }
}

XLATensor XLATensor::round(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Round(input.GetIrValue()));
}

XLATensor XLATensor::rrelu_with_noise(const XLATensor& input, XLATensor& noise,
                                      const at::Scalar& lower,
                                      const at::Scalar& upper, bool training) {
  ir::NodePtr output_node = ir::MakeNode<ir::ops::RreluWithNoise>(
      input.GetIrValue(), GetRngSeed(input.GetDevice()), lower, upper,
      training);
  noise.SetIrValue(ir::Value(output_node, 1));
  return input.CreateFrom(ir::Value(output_node, 0));
}

XLATensor XLATensor::rrelu_with_noise_backward(const XLATensor& grad_output,
                                               const XLATensor& input,
                                               const XLATensor& noise,
                                               const at::Scalar& lower,
                                               const at::Scalar& upper,
                                               bool training) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::RreluWithNoiseBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), noise.GetIrValue(), lower,
      upper, training));
}

XLATensor XLATensor::rsqrt(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Rsqrt(input.GetIrValue()));
}

XLATensor XLATensor::rsub(const XLATensor& input, const XLATensor& other,
                          const at::Scalar& alpha,
                          c10::optional<at::ScalarType> logical_element_type) {
  ir::Value alpha_xla = GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, other.GetDevice());
  return input.CreateFrom(other.GetIrValue() - alpha_xla * input.GetIrValue(),
                          logical_element_type);
}

XLATensor XLATensor::rsub(const XLATensor& input, const at::Scalar& other,
                          const at::Scalar& alpha,
                          c10::optional<at::ScalarType> logical_element_type) {
  ir::Value alpha_xla = GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  ir::Value other_xla = GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(other_xla - alpha_xla * input.GetIrValue(),
                          logical_element_type);
}

void XLATensor::copy_(XLATensor& input, XLATensor& src) {
  if (input.GetDevice() == src.GetDevice()) {
    ir::Value copy_value;
    if (input.dtype() == src.dtype()) {
      copy_value = src.GetIrValue();
    } else {
      copy_value = ir::MakeNode<ir::ops::Cast>(src.GetIrValue(), input.dtype(),
                                               src.dtype());
    }
    input.SetIrValue(MaybeExpand(copy_value, input.shape()));
  } else {
    auto input_shape = input.shape();
    at::Tensor src_tensor = src.ToTensor(/*detached=*/true);
    if (!xla::util::Equal(src_tensor.sizes(), input_shape.get().dimensions())) {
      src_tensor = src_tensor.expand(
          xla::util::ToVector<int64_t>(input_shape.get().dimensions()));
    }
    input.UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }
}

XLATensor XLATensor::scatter(const XLATensor& input, int64_t dim,
                             const XLATensor& index, const XLATensor& src) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::scatter(const XLATensor& input, int64_t dim,
                             const XLATensor& index, const at::Scalar& value) {
  ir::Value constant =
      GetIrValueForScalar(value, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), index.GetIrValue(), constant,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::scatter_add(const XLATensor& input, int64_t dim,
                                 const XLATensor& index, const XLATensor& src) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), src.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::scatter_add(const XLATensor& input, int64_t dim,
                                 const XLATensor& index,
                                 const at::Scalar& value) {
  ir::Value constant =
      GetIrValueForScalar(value, input.shape(), input.GetDevice());
  return input.CreateFrom(ir::MakeNode<ir::ops::ScatterAdd>(
      input.GetIrValue(), index.GetIrValue(), constant,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::select(const XLATensor& input, int64_t dim,
                            int64_t index) {
  return tensor_ops::Select(input, dim, index);
}

void XLATensor::silu_out(XLATensor& input, XLATensor& out) {
  out.SetInPlaceIrValue(ir::ops::SiLU(input.GetIrValue()));
}

XLATensor XLATensor::silu_backward(XLATensor& grad_output, XLATensor& input) {
  return input.CreateFrom(
      ir::ops::SiLUBackward(grad_output.GetIrValue(), input.GetIrValue()));
}

XLATensor XLATensor::sigmoid(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sigmoid(input.GetIrValue()));
}

XLATensor XLATensor::sigmoid_backward(const XLATensor& grad_output,
                                      const XLATensor& output) {
  return grad_output.CreateFrom(
      ir::ops::SigmoidBackward(grad_output.GetIrValue(), output.GetIrValue()));
}

XLATensor XLATensor::sgn(const XLATensor& input) {
  return input.CreateFrom(ir::ops::SgnOp(input.GetIrValue()));
}

XLATensor XLATensor::sign(const XLATensor& input) {
  return input.CreateFrom(ir::ops::SignOp(input.GetIrValue()));
}

XLATensor XLATensor::sin(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sin(input.GetIrValue()));
}

XLATensor XLATensor::sinh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sinh(input.GetIrValue()));
}

XLATensor XLATensor::slice(const XLATensor& input, int64_t dim, int64_t start,
                           int64_t end, int64_t step) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                           start);
  end = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                         end);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start > end) {
    end = start;
  }
  step = std::min(step, end - start);

  SelectInfo select = {dim, start, end, step};
  ViewInfo view_info(ViewInfo::Type::kSelect, input_shape, std::move(select));
  return input.CreateViewTensor(std::move(view_info));
}

std::tuple<XLATensor, XLATensor> XLATensor::slogdet(const XLATensor& input) {
  ir::NodePtr node = ir::ops::SLogDet(input.GetIrValue());
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

XLATensor XLATensor::smooth_l1_loss(const XLATensor& input,
                                    const XLATensor& target, int64_t reduction,
                                    double beta) {
  return tensor_ops::SmoothL1Loss(input, target, GetXlaReductionMode(reduction),
                                  beta);
}

XLATensor XLATensor::smooth_l1_loss_backward(const XLATensor& grad_output,
                                             const XLATensor& input,
                                             const XLATensor& target,
                                             int64_t reduction, double beta) {
  return tensor_ops::SmoothL1LossBackward(grad_output, input, target,
                                          GetXlaReductionMode(reduction), beta);
}

XLATensor XLATensor::softmax(const XLATensor& input, int64_t dim,
                             c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Softmax>(input.GetIrValue(),
                                     XlaHelpers::GetCanonicalDimensionIndex(
                                         dim, input.shape().get().rank()),
                                     dtype),
      dtype);
}

XLATensor XLATensor::softmax_backward(const XLATensor& grad_output,
                                      const XLATensor& output, int64_t dim) {
  return grad_output.CreateFrom(ir::ops::SoftmaxBackwardOp(
      grad_output.GetIrValue(), output.GetIrValue(), dim));
}

XLATensor XLATensor::softplus(const XLATensor& input, const at::Scalar& beta,
                              const at::Scalar& threshold) {
  ir::Value beta_value = XLATensor::GetIrValueForScalar(
      beta, input.shape().get().element_type(), input.GetDevice());
  ir::Value threshold_value = XLATensor::GetIrValueForScalar(
      threshold, input.shape().get().element_type(), input.GetDevice());
  return input.CreateFrom(
      ir::ops::Softplus(input.GetIrValue(), beta_value, threshold_value));
}

XLATensor XLATensor::softplus_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const at::Scalar& beta,
                                       const at::Scalar& threshold) {
  return tensor_ops::SoftplusBackward(grad_output, input, beta, threshold);
}

XLATensor XLATensor::softshrink(const XLATensor& input,
                                const at::Scalar& lambda) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Softshrink>(input.GetIrValue(), lambda));
}

XLATensor XLATensor::softshrink_backward(const XLATensor& grad_out,
                                         const XLATensor& input,
                                         const at::Scalar& lambda) {
  return input.CreateFrom(ir::MakeNode<ir::ops::ShrinkBackward>(
      ir::OpKind(at::aten::softshrink_backward), grad_out.GetIrValue(),
      input.GetIrValue(), lambda));
}

std::vector<XLATensor> XLATensor::split(const XLATensor& input,
                                        int64_t split_size, int64_t dim) {
  auto input_shape = input.shape();
  int split_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  int64_t dim_size = input_shape.get().dimensions(split_dim);
  if (dim_size == 0) {
    // Deal with dim_size=0, it's a corner case which only return 1 0-dim tensor
    // no matter what split_size is.
    xla::Literal literal(input_shape.get());
    return {
        input.CreateFrom(ir::MakeNode<ir::ops::Constant>(std::move(literal)))};
  }
  std::vector<int64_t> split_sizes;
  for (; dim_size > 0; dim_size -= split_size) {
    split_sizes.push_back(std::min<int64_t>(dim_size, split_size));
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_sizes), split_dim);
  return input.MakeOutputTensors(node);
}

std::vector<XLATensor> XLATensor::split_with_sizes(
    const XLATensor& input, std::vector<int64_t> split_size, int64_t dim) {
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

XLATensor XLATensor::squeeze(const XLATensor& input) {
  auto input_shape = input.shape();
  auto output_dimensions = BuildSqueezedDimensions(
      input_shape.get().dimensions(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

XLATensor XLATensor::squeeze(const XLATensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  auto output_dimensions =
      BuildSqueezedDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, output_dimensions);
}

void XLATensor::squeeze_(XLATensor& input) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void XLATensor::squeeze_(XLATensor& input, int64_t dim) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank())));
}

XLATensor XLATensor::stack(absl::Span<const XLATensor> tensors, int64_t dim) {
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  int64_t canonical_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, tensors.front().shape().get().rank() + 1);
  return tensors[0].CreateFrom(
      ir::MakeNode<ir::ops::Stack>(values, canonical_dim));
}

XLATensor XLATensor::std(const XLATensor& input,
                         std::vector<int64_t> dimensions,
                         bool keep_reduced_dimensions, int64_t correction) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Std>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions, correction));
}

std::tuple<XLATensor, XLATensor> XLATensor::std_mean(
    const XLATensor& input, std::vector<int64_t> dimensions, int64_t correction,
    bool keep_reduced_dimensions) {
  ir::NodePtr node = ir::MakeNode<ir::ops::StdMean>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndices(dimensions,
                                               input.shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

XLATensor XLATensor::sub(const XLATensor& input, const XLATensor& other,
                         const at::Scalar& alpha,
                         c10::optional<at::ScalarType> logical_element_type) {
  ir::Value constant = GetIrValueForScalar(
      alpha, other.shape(), logical_element_type, other.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant,
                          logical_element_type);
}

XLATensor XLATensor::sub(const XLATensor& input, const at::Scalar& other,
                         const at::Scalar& alpha,
                         c10::optional<at::ScalarType> logical_element_type) {
  ir::Value other_constant = GetIrValueForScalar(
      other, input.shape(), logical_element_type, input.GetDevice());
  ir::Value alpha_constant = GetIrValueForScalar(
      alpha, input.shape(), logical_element_type, input.GetDevice());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant,
                          logical_element_type);
}

XLATensor XLATensor::sum(const XLATensor& input,
                         std::vector<int64_t> dimensions,
                         bool keep_reduced_dimensions,
                         c10::optional<at::ScalarType> dtype) {
  if (at::isIntegralType(input.dtype(), /*includeBool=*/true) && !dtype) {
    dtype = at::ScalarType::Long;
  } else if (!dtype) {
    dtype = input.dtype_optional();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Sum>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 keep_reduced_dimensions, dtype),
      dtype);
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

XLATensor XLATensor::take(const XLATensor& input, const XLATensor& index) {
  return input.CreateFrom(
      ir::ops::Take(input.GetIrValue(), index.GetIrValue()));
}

XLATensor XLATensor::tan(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Tan(input.GetIrValue()));
}

XLATensor XLATensor::tanh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Tanh(input.GetIrValue()));
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
                                                 int64_t k, int64_t dim,
                                                 bool largest, bool sorted,
                                                 bool stable) {
  ir::NodePtr node = ir::MakeNode<ir::ops::TopK>(
      input.GetIrValue(), k,
      XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank()),
      largest, sorted, stable);
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

XLATensor XLATensor::transpose(const XLATensor& input, int64_t dim0,
                               int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = XlaHelpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.CreateViewTensor(std::move(view_info));
}

void XLATensor::transpose_(XLATensor& input, int64_t dim0, int64_t dim1) {
  auto input_shape = input.shape();
  auto permute_dims = XlaHelpers::MakeTransposePermutation(
      /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
  ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, permute_dims);
  return input.ModifyCurrentView(std::move(view_info));
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

XLATensor XLATensor::tril(const XLATensor& input, int64_t diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

void XLATensor::tril_(XLATensor& input, int64_t diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

XLATensor XLATensor::triu(const XLATensor& input, int64_t diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

void XLATensor::triu_(XLATensor& input, int64_t diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

XLATensor XLATensor::trunc(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Trunc(input.GetIrValue()));
}

std::vector<XLATensor> XLATensor::unbind(const XLATensor& input, int64_t dim) {
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input.shape().get().rank());
  int64_t dim_size = input.size(dim);
  std::vector<XLATensor> slices;
  slices.reserve(dim_size);
  for (int64_t index = 0; index < dim_size; ++index) {
    slices.push_back(select(input, dim, index));
  }
  return slices;
}

void XLATensor::uniform_(XLATensor& input, double from, double to) {
  XLA_CHECK_LE(from, to);
  auto input_shape = input.shape();
  input.SetInPlaceIrValue(ir::MakeNode<ir::ops::Uniform>(
      GetIrValueForScalar(from, input_shape.get().element_type(),
                          input.GetDevice()),
      GetIrValueForScalar(to, input_shape.get().element_type(),
                          input.GetDevice()),
      GetRngSeed(input.GetDevice()), input_shape));
}

XLATensor XLATensor::unsqueeze(const XLATensor& input, int64_t dim) {
  auto input_shape = input.shape();
  int64_t squeeze_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank() + 1);
  auto dimensions =
      BuildUnsqueezeDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, dimensions);
}

void XLATensor::unsqueeze_(XLATensor& input, int64_t dim) {
  int squeeze_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, input.shape().get().rank() + 1);
  input.SetIrValue(
      ir::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(), squeeze_dim));
}

XLATensor XLATensor::upsample_bilinear2d(const XLATensor& input,
                                         std::vector<int64_t> output_size,
                                         bool align_corners) {
  return input.CreateFrom(ir::MakeNode<ir::ops::UpsampleBilinear>(
      input.GetIrValue(), std::move(output_size), align_corners));
}

XLATensor XLATensor::upsample_bilinear2d_backward(
    const XLATensor& grad_output, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size, bool align_corners) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::UpsampleBilinearBackward>(
      grad_output.GetIrValue(), std::move(output_size), std::move(input_size),
      align_corners));
}

XLATensor XLATensor::upsample_nearest2d(const XLATensor& input,
                                        std::vector<int64_t> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::UpsampleNearest>(
      input.GetIrValue(), std::move(output_size)));
}

XLATensor XLATensor::upsample_nearest2d_backward(
    const XLATensor& grad_output, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::UpsampleNearestBackward>(
      grad_output.GetIrValue(), std::move(output_size), std::move(input_size)));
}

XLATensor XLATensor::view(const XLATensor& input,
                          absl::Span<const int64_t> output_size) {
  auto input_shape = input.shape();
  std::vector<int64_t> complete_dimensions =
      GetCompleteShape(output_size, input_shape.get().dimensions());
  xla::Shape shape =
      XlaHelpers::GetDynamicReshape(input_shape, complete_dimensions);
  ViewInfo view_info(ViewInfo::Type::kReshape, std::move(shape), input_shape);
  return input.CreateViewTensor(std::move(view_info));
}

XLATensor XLATensor::var(const XLATensor& input,
                         std::vector<int64_t> dimensions, int64_t correction,
                         bool keep_reduced_dimensions) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Var>(input.GetIrValue(),
                                 XlaHelpers::GetCanonicalDimensionIndices(
                                     dimensions, input.shape().get().rank()),
                                 correction, keep_reduced_dimensions));
}

std::tuple<XLATensor, XLATensor> XLATensor::var_mean(
    const XLATensor& input, std::vector<int64_t> dimensions, int64_t correction,
    bool keep_reduced_dimensions) {
  ir::NodePtr node = ir::MakeNode<ir::ops::VarMean>(
      input.GetIrValue(),
      XlaHelpers::GetCanonicalDimensionIndices(dimensions,
                                               input.shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)));
}

void XLATensor::zero_(XLATensor& input) {
  ir::Value constant =
      GetIrValueForScalar(0.0, input.shape(), input.GetDevice());
  input.SetInPlaceIrValue(std::move(constant));
}

XLATensor XLATensor::where(const XLATensor& condition, const XLATensor& input,
                           const XLATensor& other) {
  return input.CreateFrom(ir::ops::Where(
      condition.GetIrValue(), input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::DispatchComparisonOp(c10::Symbol kind,
                                          const XLATensor& input,
                                          const at::Scalar& other) {
  ir::NodePtr node = ir::ops::ComparisonOp(
      kind, input.GetIrValue(), GetIrValueForScalar(other, input.GetDevice()));
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
