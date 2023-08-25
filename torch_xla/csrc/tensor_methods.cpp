#include "torch_xla/csrc/tensor_methods.h"

#include <ATen/core/Reduction.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <functional>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "torch_xla/csrc/LazyIr.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/adam_optimizer_step.h"
#include "torch_xla/csrc/ops/adaptive_max_pool2d.h"
#include "torch_xla/csrc/ops/all_gather.h"
#include "torch_xla/csrc/ops/all_reduce.h"
#include "torch_xla/csrc/ops/all_to_all.h"
#include "torch_xla/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "torch_xla/csrc/ops/amp_update_scale.h"
#include "torch_xla/csrc/ops/arg_max.h"
#include "torch_xla/csrc/ops/arg_min.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/avg_pool_nd.h"
#include "torch_xla/csrc/ops/avg_pool_nd_backward.h"
#include "torch_xla/csrc/ops/bernoulli.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/cat.h"
#include "torch_xla/csrc/ops/cdist.h"
#include "torch_xla/csrc/ops/collective_permute.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/constant_pad_nd.h"
#include "torch_xla/csrc/ops/convolution_backward_overrideable.h"
#include "torch_xla/csrc/ops/convolution_overrideable.h"
#include "torch_xla/csrc/ops/count_nonzero.h"
#include "torch_xla/csrc/ops/cumprod.h"
#include "torch_xla/csrc/ops/cumsum.h"
#include "torch_xla/csrc/ops/custom_sharding.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/discrete_uniform.h"
#include "torch_xla/csrc/ops/einsum.h"
#include "torch_xla/csrc/ops/einsum_backward.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/expand_symint.h"
#include "torch_xla/csrc/ops/exponential.h"
#include "torch_xla/csrc/ops/flip.h"
#include "torch_xla/csrc/ops/gather.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/generic_slice.h"
#include "torch_xla/csrc/ops/get_dimensions_size.h"
#include "torch_xla/csrc/ops/hardtanh_backward.h"
#include "torch_xla/csrc/ops/index_ops.h"
#include "torch_xla/csrc/ops/index_select.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/kth_value.h"
#include "torch_xla/csrc/ops/linear_interpolation.h"
#include "torch_xla/csrc/ops/linspace.h"
#include "torch_xla/csrc/ops/log_softmax.h"
#include "torch_xla/csrc/ops/logsumexp.h"
#include "torch_xla/csrc/ops/masked_fill.h"
#include "torch_xla/csrc/ops/masked_scatter.h"
#include "torch_xla/csrc/ops/masked_select.h"
#include "torch_xla/csrc/ops/max_in_dim.h"
#include "torch_xla/csrc/ops/max_pool_nd.h"
#include "torch_xla/csrc/ops/max_pool_nd_backward.h"
#include "torch_xla/csrc/ops/max_unpool_nd.h"
#include "torch_xla/csrc/ops/mean.h"
#include "torch_xla/csrc/ops/min_in_dim.h"
#include "torch_xla/csrc/ops/mse_loss.h"
#include "torch_xla/csrc/ops/mse_loss_backward.h"
#include "torch_xla/csrc/ops/multinomial.h"
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
#include "torch_xla/csrc/ops/optimization_barrier.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/prod.h"
#include "torch_xla/csrc/ops/put.h"
#include "torch_xla/csrc/ops/qr.h"
#include "torch_xla/csrc/ops/recv.h"
#include "torch_xla/csrc/ops/reduce_scatter.h"
#include "torch_xla/csrc/ops/reflection_pad2d.h"
#include "torch_xla/csrc/ops/reflection_pad2d_backward.h"
#include "torch_xla/csrc/ops/replication_pad.h"
#include "torch_xla/csrc/ops/replication_pad_backward.h"
#include "torch_xla/csrc/ops/resize.h"
#include "torch_xla/csrc/ops/roll.h"
#include "torch_xla/csrc/ops/rrelu_with_noise.h"
#include "torch_xla/csrc/ops/rrelu_with_noise_backward.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/scatter.h"
#include "torch_xla/csrc/ops/scatter_add.h"
#include "torch_xla/csrc/ops/scatter_reduce.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/send.h"
#include "torch_xla/csrc/ops/sgd_optimizer_step.h"
#include "torch_xla/csrc/ops/softmax.h"
#include "torch_xla/csrc/ops/split.h"
#include "torch_xla/csrc/ops/squeeze.h"
#include "torch_xla/csrc/ops/stack.h"
#include "torch_xla/csrc/ops/std.h"
#include "torch_xla/csrc/ops/std_mean.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/ops/svd.h"
#include "torch_xla/csrc/ops/threshold.h"
#include "torch_xla/csrc/ops/threshold_backward.h"
#include "torch_xla/csrc/ops/topk.h"
#include "torch_xla/csrc/ops/triangular_solve.h"
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
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/shape_builder.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "xla/literal_util.h"

namespace torch_xla {
namespace tensor_methods {
namespace {

struct MinMaxValues {
  torch::lazy::Value min;
  torch::lazy::Value max;
};

torch::lazy::Value MaybeExpand(const torch::lazy::Value& input,
                               const xla::Shape& target_shape) {
  if (GetXlaShape(input).dimensions() == target_shape.dimensions()) {
    return input;
  }
  return torch::lazy::MakeNode<Expand>(
      input, torch::lazy::ToVector<int64_t>(target_shape.dimensions()));
}

MinMaxValues GetMinMaxValues(const XLATensorPtr& tensor,
                             const c10::optional<at::Scalar>& min,
                             const c10::optional<at::Scalar>& max) {
  XLA_CHECK(min || max)
      << "At least one of \'min\' or \'max\' must not be None";
  xla::PrimitiveType raw_element_type = TensorTypeToRawXlaType(tensor->dtype());
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(raw_element_type);
  auto shape = tensor->shape();
  return {XLAGraphExecutor::Get()->GetIrValueForScalar(
              min ? *min : min_max.min, shape.get().element_type(),
              tensor->GetDevice()),
          XLAGraphExecutor::Get()->GetIrValueForScalar(
              max ? *max : min_max.max, shape.get().element_type(),
              tensor->GetDevice())};
}

void CheckRank(const XLATensorPtr& t, int64_t expected_rank,
               const std::string& tag, const std::string& arg_name,
               int arg_number) {
  int64_t actual_rank = t->shape().get().rank();
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

void CheckDimensionSize(const XLATensorPtr& t, int64_t dim,
                        int64_t expected_size, const std::string& tag,
                        const std::string& arg_name, int arg_number) {
  int64_t dim_size = t->size(dim);
  XLA_CHECK_EQ(t->size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

void CheckBmmDimension(const std::string& tag, const XLATensorPtr& batch1,
                       const XLATensorPtr& batch2) {
  // Consistent with the checks in bmm_out_or_baddbmm_.
  CheckRank(batch1, 3, tag, "batch1", 1);
  CheckRank(batch2, 3, tag, "batch2", 2);
  CheckDimensionSize(batch2, 0, /*batch_size=*/batch1->size(0), tag, "batch2",
                     2);
  CheckDimensionSize(batch2, 1, /*contraction_size=*/batch1->size(2), tag,
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
    result = torch::lazy::ToVector<int64_t>(list);
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
xla::Shape BatchNormFeaturesShape(const XLATensorPtr& input) {
  xla::PrimitiveType input_element_type =
      MakeXlaPrimitiveType(input->dtype(), &input->GetDevice());
  auto input_shape = input->shape();
  return ShapeBuilder(input_element_type).Add(input_shape.get(), 1).Build();
}

// Returns the IR for the given input or the provided default value broadcasted
// to the default shape, if the input is undefined.
torch::lazy::Value GetIrValueOrDefault(
    const XLATensorPtr& input, const at::Scalar& default_value,
    const xla::Shape& default_shape, const torch::lazy::BackendDevice& device) {
  return input ? input->GetIrValue()
               : XLAGraphExecutor::Get()->GetIrValueForScalar(
                     default_value, default_shape, device);
}

// Returns the IR for the given input. If the IR is not a floating point value,
// cast it to the float_type.
torch::lazy::Value GetFloatingIrValue(const XLATensorPtr& input,
                                      at::ScalarType float_type) {
  torch::lazy::Value input_value = input->GetIrValue();
  if (xla::primitive_util::IsIntegralType(
          GetXlaShape(input_value).element_type())) {
    input_value = torch::lazy::MakeNode<Cast>(input_value, float_type);
  }
  return input_value;
}

torch::lazy::Value GetBooleanIrValue(torch::lazy::Value input_value) {
  if (GetXlaShape(input_value).element_type() != xla::PrimitiveType::PRED) {
    input_value =
        torch::lazy::MakeNode<Cast>(input_value, xla::PrimitiveType::PRED);
  }
  return input_value;
}

absl::optional<torch::lazy::Value> GetOptionalIrValue(
    const XLATensorPtr& tensor) {
  absl::optional<torch::lazy::Value> value;
  if (tensor) {
    value = tensor->GetIrValue();
  }
  return value;
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

// Dispatches a comparison operator, setting the logical type of the result
// appropriately.
XLATensorPtr DispatchComparisonOp(c10::Symbol kind, const XLATensorPtr& input,
                                  const at::Scalar& other) {
  torch::lazy::NodePtr node = ComparisonOp(
      kind, input->GetIrValue(),
      XLAGraphExecutor::Get()->GetIrValueForScalar(other, input->GetDevice()));
  return XLATensor::Create(node, input->GetDevice(), at::ScalarType::Bool);
}

// Same as above, with the second input a tensor as well.
XLATensorPtr DispatchComparisonOp(c10::Symbol kind, const XLATensorPtr& input,
                                  const XLATensorPtr& other) {
  torch::lazy::NodePtr node =
      ComparisonOp(kind, input->GetIrValue(), other->GetIrValue());
  return XLATensor::Create(node, input->GetDevice(), at::ScalarType::Bool);
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////
// XLA dedicated operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
XLATensorPtr all_reduce(const XLATensorPtr& input, AllReduceType reduce_type,
                        double scale, std::vector<std::vector<int64_t>> groups,
                        bool pin_layout) {
  std::vector<torch::lazy::Value> input_values({input->GetIrValue()});
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AllReduce>(
      reduce_type, input_values, GetAllReduceToken(input->GetDevice()), scale,
      std::move(groups), pin_layout);
  SetAllReduceToken(input->GetDevice(),
                    std::make_shared<torch::lazy::Value>(node, 1));
  return input->CreateFrom(torch::lazy::Value(node, 0));
}

void all_reduce(const std::vector<XLATensorPtr>& inputs,
                AllReduceType reduce_type, double scale,
                std::vector<std::vector<int64_t>> groups, bool pin_layout) {
  std::vector<torch::lazy::Value> input_values;
  input_values.reserve(inputs.size());
  for (auto& input : inputs) {
    input_values.push_back(input->GetIrValue());
  }
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AllReduce>(
      reduce_type, input_values, GetAllReduceToken(inputs.front()->GetDevice()),
      scale, std::move(groups), pin_layout);
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs[i]->SetInPlaceIrValue(torch::lazy::Value(node, i));
  }
  SetAllReduceToken(inputs.front()->GetDevice(),
                    std::make_shared<torch::lazy::Value>(node, inputs.size()));
}

std::pair<XLATensorPtr, torch::lazy::Value> reduce_scatter(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    AllReduceType reduce_type, double scale, int64_t scatter_dim,
    int64_t shard_count, std::vector<std::vector<int64_t>> groups,
    bool pin_layout) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ReduceScatter>(
      reduce_type, input->GetIrValue(), token, scale, scatter_dim, shard_count,
      std::move(groups), pin_layout);
  return {input->CreateFrom(torch::lazy::Value(node, 0)),
          torch::lazy::Value(node, 1)};
}

torch::lazy::Value reduce_scatter_out(XLATensorPtr& output,
                                      const XLATensorPtr& input,
                                      const torch::lazy::Value& token,
                                      AllReduceType reduce_type, double scale,
                                      int64_t scatter_dim, int64_t shard_count,
                                      std::vector<std::vector<int64_t>> groups,
                                      bool pin_layout) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ReduceScatter>(
      reduce_type, input->GetIrValue(), token, scale, scatter_dim, shard_count,
      std::move(groups), pin_layout);
  output->SetIrValue(torch::lazy::Value(node, 0));
  return torch::lazy::Value(node, 1);
}

std::pair<XLATensorPtr, torch::lazy::Value> all_to_all(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    std::vector<std::vector<int64_t>> groups, bool pin_layout) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AllToAll>(
      input->GetIrValue(), token, split_dimension, concat_dimension,
      split_count, std::move(groups), pin_layout);
  return {input->CreateFrom(torch::lazy::Value(node, 0)),
          torch::lazy::Value(node, 1)};
}

XLATensorPtr all_gather(const XLATensorPtr& input, int64_t dim,
                        int64_t shard_count,
                        std::vector<std::vector<int64_t>> groups,
                        bool pin_layout) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AllGather>(
      input->GetIrValue(), GetAllReduceToken(input->GetDevice()), dim,
      shard_count, std::move(groups), pin_layout);
  SetAllReduceToken(input->GetDevice(),
                    std::make_shared<torch::lazy::Value>(node, 1));
  return input->CreateFrom(torch::lazy::Value(node, 0));
}

torch::lazy::Value all_gather_out(XLATensorPtr& output,
                                  const XLATensorPtr& input,
                                  const torch::lazy::Value& token, int64_t dim,
                                  int64_t shard_count,
                                  std::vector<std::vector<int64_t>> groups,
                                  bool pin_layout) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AllGather>(
      input->GetIrValue(), token, dim, shard_count, std::move(groups),
      pin_layout);
  output->SetIrValue(torch::lazy::Value(node, 0));
  return torch::lazy::Value(node, 1);
}

std::pair<XLATensorPtr, torch::lazy::Value> collective_permute(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<CollectivePermute>(
      input->GetIrValue(), token, std::move(source_target_pairs));
  return {input->CreateFrom(torch::lazy::Value(node, 0)),
          torch::lazy::Value(node, 1)};
}

void custom_sharding_(
    const XLATensorPtr& input,
    const std::shared_ptr<XLATensor::ShardingSpec>& sharding_spec) {
  input->SetInPlaceIrValue(
      torch::lazy::MakeNode<CustomSharding>(input->GetIrValue()));
  input->SetShardingSpec(*sharding_spec);
}

XLATensorPtr get_dimensions_size(const XLATensorPtr& input,
                                 std::vector<int64_t> dimensions) {
  return input->CreateFrom(torch::lazy::MakeNode<GetDimensionsSize>(
                               input->GetIrValue(), std::move(dimensions)),
                           at::ScalarType::Int);
}

std::pair<XLATensorPtr, torch::lazy::Value> recv(
    XLATensorPtr& output, const torch::lazy::Value& token, int64_t channel_id) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::Recv>(
      token, GetXlaShape(output->GetIrValue()), channel_id);
  output->SetIrValue(torch::lazy::Value(node, 0));
  return {output->CreateFrom(torch::lazy::Value(node, 0)),
          torch::lazy::Value(node, 1)};
}

std::pair<XLATensorPtr, torch::lazy::Value> send(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    int64_t channel_id) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::Send>(
      input->GetIrValue(), token, channel_id);
  return {input->CreateFrom(torch::lazy::Value(node, 0)),
          torch::lazy::Value(node, 1)};
}

void sgd_optimizer_step_(const XLATensorPtr& found_inf, XLATensorPtr& step,
                         XLATensorPtr& param, XLATensorPtr& buf,
                         const XLATensorPtr& d_p, double weight_decay,
                         double momentum, double lr, double dampening,
                         bool nesterov, bool maximize) {
  torch::lazy::Value weight_decay_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(weight_decay, param->shape(),
                                                   param->GetDevice());
  torch::lazy::Value momentum_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(momentum, param->shape(),
                                                   param->GetDevice());
  torch::lazy::Value lr_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      maximize ? -lr : lr, param->shape(), param->GetDevice());
  torch::lazy::Value dampening_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(dampening, param->shape(),
                                                   param->GetDevice());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<SgdOptimizerStep>(
      found_inf->GetIrValue(), step->GetIrValue(), param->GetIrValue(),
      buf->GetIrValue(), d_p->GetIrValue(), weight_decay_value, momentum_value,
      lr_value, dampening_value,
      /*use_weight_decay=*/weight_decay != 0,
      /*use_momentum=*/momentum != 0, /*use_nesterov=*/nesterov);
  step->SetInPlaceIrValue(torch::lazy::Value(node, 0));
  param->SetInPlaceIrValue(torch::lazy::Value(node, 1));
  buf->SetInPlaceIrValue(torch::lazy::Value(node, 2));
}

void adam_optimizer_step_(const XLATensorPtr& found_inf, XLATensorPtr& step,
                          XLATensorPtr& param, const XLATensorPtr& grad,
                          XLATensorPtr& exp_avg, XLATensorPtr& exp_avg_sq,
                          XLATensorPtr& max_exp_avg_sq, double beta1,
                          double beta2, double lr, double weight_decay,
                          double eps, bool amsgrad, bool maximize,
                          bool use_adamw) {
  torch::lazy::Value grad_value =
      maximize ? mul(grad, -1)->GetIrValue() : grad->GetIrValue();
  torch::lazy::Value beta1_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      beta1, found_inf->shape(), found_inf->GetDevice());
  torch::lazy::Value beta2_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      beta2, found_inf->shape(), found_inf->GetDevice());
  torch::lazy::Value lr_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      lr, found_inf->shape(), found_inf->GetDevice());
  torch::lazy::Value weight_decay_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(weight_decay, param->shape(),
                                                   param->GetDevice());
  torch::lazy::Value eps_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      eps, param->shape(), param->GetDevice());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AdamOptimizerStep>(
      found_inf->GetIrValue(), step->GetIrValue(), param->GetIrValue(),
      grad_value, exp_avg->GetIrValue(), exp_avg_sq->GetIrValue(),
      max_exp_avg_sq->GetIrValue(), beta1_value, beta2_value, lr_value,
      weight_decay_value, eps_value,
      /*use_weight_decay=*/weight_decay != 0,
      /*use_amsgrad=*/amsgrad, /*use_adamw=*/use_adamw);
  step->SetInPlaceIrValue(torch::lazy::Value(node, 0));
  param->SetInPlaceIrValue(torch::lazy::Value(node, 1));
  exp_avg->SetInPlaceIrValue(torch::lazy::Value(node, 2));
  exp_avg_sq->SetInPlaceIrValue(torch::lazy::Value(node, 3));
  max_exp_avg_sq->SetInPlaceIrValue(torch::lazy::Value(node, 4));
}

std::vector<XLATensorPtr> user_computation(
    const std::string& opname, absl::Span<const XLATensorPtr> inputs,
    ComputationPtr computation) {
  XLA_CHECK(!inputs.empty());
  std::vector<torch::lazy::Value> input_values;
  for (auto& input : inputs) {
    input_values.push_back(input->GetIrValue());
  }
  torch::lazy::NodePtr node = torch::lazy::MakeNode<UserComputation>(
      torch::lazy::OpKind::Get(opname), input_values, std::move(computation));
  // Cast can be one of the user computation and we don't want to inherit the
  // logical_element_type in this case
  return inputs.front()->MakeOutputTensors(node,
                                           /*inherit_logical_type=*/false);
}

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
void __ilshift__(XLATensorPtr& input, const at::Scalar& other) {
  input->SetInPlaceIrValue(Lshift(input->GetIrValue(), other));
}

void __ilshift__(XLATensorPtr& input, const XLATensorPtr& other) {
  input->SetInPlaceIrValue(Lshift(input->GetIrValue(), other->GetIrValue()));
}

void __irshift__(XLATensorPtr& input, const at::Scalar& other) {
  input->SetInPlaceIrValue(Rshift(input->GetIrValue(), other));
}

void __irshift__(XLATensorPtr& input, const XLATensorPtr& other) {
  input->SetInPlaceIrValue(Rshift(input->GetIrValue(), other->GetIrValue()));
}

XLATensorPtr __lshift__(const XLATensorPtr& input, const at::Scalar& other,
                        c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(Lshift(input->GetIrValue(), other),
                           logical_element_type);
}

XLATensorPtr __lshift__(const XLATensorPtr& input, const XLATensorPtr& other,
                        c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(Lshift(input->GetIrValue(), other->GetIrValue()),
                           logical_element_type);
}

XLATensorPtr __rshift__(const XLATensorPtr& input, const at::Scalar& other,
                        c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(Rshift(input->GetIrValue(), other),
                           logical_element_type);
}

XLATensorPtr __rshift__(const XLATensorPtr& input, const XLATensorPtr& other,
                        c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(Rshift(input->GetIrValue(), other->GetIrValue()),
                           logical_element_type);
}

std::tuple<XLATensorPtr, XLATensorPtr> adaptive_max_pool2d(
    const XLATensorPtr& input, std::vector<int64_t> output_size) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AdaptiveMaxPool2d>(
      input->GetIrValue(), output_size);
  XLATensorPtr out = input->CreateFrom(torch::lazy::Value(node, 0));
  XLATensorPtr indices =
      input->CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long);
  return std::make_tuple(std::move(out), std::move(indices));
}

XLATensorPtr adaptive_max_pool2d_backward(const XLATensorPtr& grad_output,
                                          const XLATensorPtr& input) {
  return input->CreateFrom(AdaptiveMaxPool2dBackward(grad_output->GetIrValue(),
                                                     input->GetIrValue()));
}

XLATensorPtr _adaptive_avg_pool2d(const XLATensorPtr& input,
                                  std::vector<int64_t> output_size) {
  return input->CreateFrom(torch::lazy::MakeNode<AdaptiveAvgPool2d>(
      input->GetIrValue(), std::move(output_size)));
}

XLATensorPtr _adaptive_avg_pool2d_backward(const XLATensorPtr& grad_output,
                                           const XLATensorPtr& input) {
  return input->CreateFrom(torch::lazy::MakeNode<AdaptiveAvgPool2dBackward>(
      grad_output->GetIrValue(), input->GetIrValue()));
}

void _amp_foreach_non_finite_check_and_unscale_(std::vector<XLATensorPtr> self,
                                                XLATensorPtr& found_inf,
                                                const XLATensorPtr& inv_scale) {
  std::vector<torch::lazy::Value> inputs;
  XLATensorPtr new_inv_scale = max(inv_scale);
  for (const auto& x : self) {
    inputs.push_back(x->GetIrValue());
  }
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<AmpForachNonFiniteCheckAndUnscale>(
          inputs, found_inf->GetIrValue(), new_inv_scale->GetIrValue());
  for (size_t i = 0; i < self.size(); ++i) {
    self[i]->SetInPlaceIrValue(torch::lazy::Value(node, i));
  }
  found_inf->SetInPlaceIrValue(torch::lazy::Value(node, self.size()));
}

void _amp_update_scale_(XLATensorPtr& current_scale,
                        XLATensorPtr& growth_tracker,
                        const XLATensorPtr& found_inf,
                        double scale_growth_factor, double scale_backoff_factor,
                        int growth_interval) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<AmpUpdateScale>(
      growth_tracker->GetIrValue(), current_scale->GetIrValue(),
      found_inf->GetIrValue(), scale_growth_factor, scale_backoff_factor,
      growth_interval);
  growth_tracker->SetInPlaceIrValue(torch::lazy::Value(node, 1));
  current_scale->SetInPlaceIrValue(torch::lazy::Value(node, 0));
}

XLATensorPtr abs(const XLATensorPtr& input) {
  return input->CreateFrom(torch::lazy::MakeNode<Abs>(input->GetIrValue()));
}

XLATensorPtr add(const XLATensorPtr& input, const XLATensorPtr& other,
                 const at::Scalar& alpha,
                 c10::optional<at::ScalarType> logical_element_type) {
  xla::Shape input_shape = input->shape().get();
  xla::Shape other_shape = other->shape().get();
  torch::lazy::Value constant;
  if (!input_shape.is_dynamic() && !other_shape.is_dynamic()) {
    constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
        alpha, other->shape(), logical_element_type, input->GetDevice());
  } else {
    SymIntElements sym_int_elements(other->GetIrValue());
    constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
        alpha, other->shape(), sym_int_elements, logical_element_type,
        input->GetDevice());
  }

  return input->CreateFrom(input->GetIrValue() + other->GetIrValue() * constant,
                           logical_element_type);
}

XLATensorPtr add(const XLATensorPtr& input, const at::Scalar& other,
                 const at::Scalar& alpha,
                 c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value other_constant =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          other, input->shape(), logical_element_type, input->GetDevice());
  torch::lazy::Value alpha_constant =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          alpha, input->shape(), logical_element_type, input->GetDevice());
  return input->CreateFrom(
      input->GetIrValue() + other_constant * alpha_constant,
      logical_element_type);
}

XLATensorPtr addmm(const XLATensorPtr& input, const XLATensorPtr& weight,
                   const XLATensorPtr& bias) {
  return input->CreateFrom(AddMatMulOp(
      input->GetIrValue(), weight->GetIrValue(), bias->GetIrValue()));
}

void arange_out(XLATensorPtr& out, const at::Scalar& start,
                const at::Scalar& end, const at::Scalar& step,
                at::ScalarType scalar_type) {
  out->SetIrValue(ARange(start, end, step, scalar_type));
  out->SetScalarType(scalar_type);
}

XLATensorPtr argmax(const XLATensorPtr& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  return input->CreateFrom(torch::lazy::MakeNode<ArgMax>(
                               input->GetIrValue(), canonical_dim, keepdim),
                           at::ScalarType::Long);
}

XLATensorPtr argmax(const XLATensorPtr& input) {
  return input->CreateFrom(
      torch::lazy::MakeNode<ArgMax>(input->GetIrValue(), -1, false),
      at::ScalarType::Long);
}

XLATensorPtr argmin(const XLATensorPtr& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  return input->CreateFrom(torch::lazy::MakeNode<ArgMin>(
                               input->GetIrValue(), canonical_dim, keepdim),
                           at::ScalarType::Long);
}

XLATensorPtr argmin(const XLATensorPtr& input) {
  return input->CreateFrom(
      torch::lazy::MakeNode<ArgMin>(input->GetIrValue(), -1, false),
      at::ScalarType::Long);
}

XLATensorPtr as_strided(const XLATensorPtr& input, std::vector<int64_t> size,
                        std::vector<int64_t> stride,
                        c10::optional<int64_t> storage_offset) {
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    auto input_shape = input->shape();
    return input->CreateViewTensor(CreateAsStridedViewInfo(
        input_shape, std::move(size), std::move(stride), storage_offset));
  }
  return input->CreateFrom(torch::lazy::MakeNode<AsStrided>(
      input->GetIrValue(), std::move(size), std::move(stride),
      storage_offset.value_or(0)));
}

void as_strided_(XLATensorPtr& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset) {
  if (input->data()->view == nullptr) {
    input->SetIrValue(torch::lazy::MakeNode<AsStrided>(
        input->GetIrValue(), std::move(size), std::move(stride),
        storage_offset.value_or(0)));
  } else {
    auto input_shape = input->shape();
    input->SetSubView(CreateAsStridedViewInfo(
        input_shape, std::move(size), std::move(stride), storage_offset));
  }
}

XLATensorPtr avg_pool_nd(const XLATensorPtr& input, int64_t spatial_dim_count,
                         std::vector<int64_t> kernel_size,
                         std::vector<int64_t> stride,
                         std::vector<int64_t> padding, bool ceil_mode,
                         bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return input->CreateFrom(torch::lazy::MakeNode<AvgPoolNd>(
      input->GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode, count_include_pad));
}

XLATensorPtr avg_pool_nd_backward(const XLATensorPtr& out_backprop,
                                  const XLATensorPtr& input,
                                  int64_t spatial_dim_count,
                                  std::vector<int64_t> kernel_size,
                                  std::vector<int64_t> stride,
                                  std::vector<int64_t> padding, bool ceil_mode,
                                  bool count_include_pad) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop->CreateFrom(torch::lazy::MakeNode<AvgPoolNdBackward>(
      out_backprop->GetIrValue(), input->GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding), ceil_mode,
      count_include_pad));
}

XLATensorPtr baddbmm(const XLATensorPtr& input, const XLATensorPtr& batch1,
                     const XLATensorPtr& batch2, const at::Scalar& beta,
                     const at::Scalar& alpha) {
  CheckBmmDimension(/*tag=*/"baddbmm", batch1, batch2);
  torch::lazy::Value product_multiplier =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          alpha, batch1->shape().get().element_type(), batch1->GetDevice());
  torch::lazy::Value bias_multiplier =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          beta, input->shape().get().element_type(), input->GetDevice());
  return input->CreateFrom(torch::lazy::MakeNode<Baddbmm>(
      input->GetIrValue(), batch1->GetIrValue(), batch2->GetIrValue(),
      bias_multiplier, product_multiplier));
}

XLATensorPtr bernoulli(const XLATensorPtr& input, double probability) {
  auto input_shape = input->shape();
  return input->CreateFrom(torch::lazy::MakeNode<Bernoulli>(
      XLAGraphExecutor::Get()->GetIrValueForScalar(probability, input_shape,
                                                   input->GetDevice()),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input_shape.get()));
}

XLATensorPtr bernoulli(const XLATensorPtr& input) {
  return input->CreateFrom(torch::lazy::MakeNode<Bernoulli>(
      input->GetIrValue(),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input->shape().get()));
}

void bernoulli_(XLATensorPtr& input, const XLATensorPtr& probability) {
  input->SetInPlaceIrValue(torch::lazy::MakeNode<Bernoulli>(
      probability->GetIrValue(),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input->shape().get()));
}

XLATensorPtr bitwise_and(const XLATensorPtr& input, const XLATensorPtr& other) {
  return input->CreateFrom(torch::lazy::MakeNode<BitwiseAndTensor>(
      input->GetIrValue(), other->GetIrValue()));
}

XLATensorPtr bitwise_or(const XLATensorPtr& input, const XLATensorPtr& other) {
  return input->CreateFrom(torch::lazy::MakeNode<BitwiseOrTensor>(
      input->GetIrValue(), other->GetIrValue()));
}

XLATensorPtr bitwise_xor(const XLATensorPtr& input, const XLATensorPtr& other) {
  return input->CreateFrom(torch::lazy::MakeNode<BitwiseXorTensor>(
      input->GetIrValue(), other->GetIrValue()));
}

XLATensorPtr bmm(const XLATensorPtr& batch1, const XLATensorPtr& batch2) {
  CheckBmmDimension(/*tag=*/"bmm", batch1, batch2);
  return matmul(batch1, batch2);
}

std::vector<XLATensorPtr> broadcast_tensors(
    absl::Span<const XLATensorPtr> tensors) {
  XLA_CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<torch::lazy::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor->GetIrValue());
  }
  torch::lazy::NodePtr node = BroadcastTensors(tensor_ir_values);
  return tensors.front()->MakeOutputTensors(node);
}

XLATensorPtr cat(absl::Span<const XLATensorPtr> tensors, int64_t dim,
                 at::ScalarType dtype) {
  // Shape checks for cat:
  // - If not empty, every tensor shape must be the same.
  // - Empty tensor passes but is simply ignore in implementation,
  //   e.g. ([2, 3, 5], [])
  // - If empty dimension, other dimensions must be the same.
  //   e.g. ([4, 0, 32, 32], [4, 2, 32, 32], dim=1) passes.
  //   ([4, 0, 32, 32], [4, 2, 31, 32], dim=1) throws.
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<torch::lazy::Value> values;
  std::vector<xla::Shape> shapes;
  for (size_t i = 0; i < tensors.size(); ++i) {
    xla::Shape tensor_shape = tensors[i]->shape();
    if (tensor_shape.rank() == 1 && tensor_shape.dimensions()[0] == 0) {
      continue;
    }
    dim = torch::lazy::GetCanonicalDimensionIndex(dim, tensor_shape.rank());
    tensor_shape.DeleteDimension(dim);
    if (!shapes.empty()) {
      XLA_CHECK(xla::ShapeUtil::CompatibleIgnoringElementType(shapes.back(),
                                                              tensor_shape))
          << shapes.back() << " vs. " << tensor_shape;
    }
    shapes.push_back(tensor_shape);
    values.push_back(tensors[i]->GetIrValue());
  }
  if (values.empty()) {
    return tensors[0];
  }
  return tensors[0]->CreateFrom(torch::lazy::MakeNode<Cat>(values, dim, dtype),
                                dtype);
}

XLATensorPtr cdist_forward(const XLATensorPtr& x1, const XLATensorPtr& x2,
                           double p) {
  torch::lazy::Value exponent_node =
      XLAGraphExecutor::Get()->GetIrValueForScalar(p, x1->GetDevice());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<CdistForward>(
      x1->GetIrValue(), x2->GetIrValue(), exponent_node,
      /*use_hamming=*/p == 0.0,
      /*use_chebyshev=*/std::isinf(p));
  return x1->CreateFrom(node);
}

XLATensorPtr celu(const XLATensorPtr& input, const at::Scalar& alpha) {
  return input->CreateFrom(Celu(input->GetIrValue(), alpha));
}

void celu_(XLATensorPtr& input, const at::Scalar& alpha) {
  input->SetInPlaceIrValue(Celu(input->GetIrValue(), alpha));
}

XLATensorPtr clamp(const XLATensorPtr& input,
                   const c10::optional<at::Scalar>& min,
                   const c10::optional<at::Scalar>& max) {
  MinMaxValues min_max = GetMinMaxValues(input, min, max);
  return input->CreateFrom(
      Clamp(input->GetIrValue(), min_max.min, min_max.max));
}

XLATensorPtr clone(const XLATensorPtr& input) {
  XLATensorPtr cloned = input->CreateFrom(input->GetIrValue());
  if (input->sharding_spec() != nullptr) {
    cloned->SetShardingSpec(*input->sharding_spec());
  }
  return cloned;
}

XLATensorPtr constant_pad_nd(const XLATensorPtr& input,
                             absl::Span<const int64_t> pad,
                             const at::Scalar& value) {
  std::vector<int64_t> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input->shape().get().rank());
  return input->CreateFrom(torch::lazy::MakeNode<ConstantPadNd>(
      input->GetIrValue(), complete_pad, value));
}

XLATensorPtr convolution_overrideable(
    const XLATensorPtr& input, const XLATensorPtr& weight,
    const XLATensorPtr& bias, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups) {
  torch::lazy::NodePtr ir_value =
      torch::lazy::MakeNode<ConvolutionOverrideable>(
          input->GetIrValue(), weight->GetIrValue(), bias->GetIrValue(),
          std::move(stride), std::move(padding), std::move(dilation),
          transposed, std::move(output_padding), groups);
  return input->CreateFrom(ir_value);
}

XLATensorPtr convolution_overrideable(
    const XLATensorPtr& input, const XLATensorPtr& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups) {
  torch::lazy::NodePtr ir_value =
      torch::lazy::MakeNode<ConvolutionOverrideable>(
          input->GetIrValue(), weight->GetIrValue(), std::move(stride),
          std::move(padding), std::move(dilation), transposed,
          std::move(output_padding), groups);
  return input->CreateFrom(ir_value);
}

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr>
convolution_backward_overrideable(
    const XLATensorPtr& out_backprop, const XLATensorPtr& input,
    const XLATensorPtr& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<ConvolutionBackwardOverrideable>(
          out_backprop->GetIrValue(), input->GetIrValue(), weight->GetIrValue(),
          std::move(stride), std::move(padding), std::move(dilation),
          transposed, std::move(output_padding), groups);
  XLATensorPtr grad_input =
      out_backprop->CreateFrom(torch::lazy::Value(node, 0));
  XLATensorPtr grad_weight =
      out_backprop->CreateFrom(torch::lazy::Value(node, 1));
  XLATensorPtr grad_bias =
      out_backprop->CreateFrom(torch::lazy::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensorPtr count_nonzero(const XLATensorPtr& input,
                           std::vector<int64_t> dims) {
  torch::lazy::NodePtr ir_value =
      torch::lazy::MakeNode<CountNonzero>(input->GetIrValue(), dims);
  return input->CreateFrom(ir_value);
}

XLATensorPtr cross(const XLATensorPtr& input, const XLATensorPtr& other,
                   c10::optional<int64_t> dim) {
  return tensor_ops::Cross(input, other, dim);
}

XLATensorPtr cumprod(const XLATensorPtr& input, int64_t dim,
                     c10::optional<at::ScalarType> dtype) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<CumProd>(input->GetIrValue(), canonical_dim, dtype),
      dtype);
}

XLATensorPtr cumsum(const XLATensorPtr& input, int64_t dim,
                    c10::optional<at::ScalarType> dtype) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<CumSum>(input->GetIrValue(), canonical_dim, dtype),
      dtype);
}

XLATensorPtr diag(const XLATensorPtr& input, int64_t offset) {
  int64_t rank = input->shape().get().rank();
  XLA_CHECK(rank == 1 || rank == 2)
      << "Invalid argument for diag: matrix or a vector expected";
  if (rank == 1) {
    return tensor_ops::MakeMatrixWithDiagonal(input, offset);
  }
  return diagonal(input, offset, /*dim1=*/-2, /*dim2=*/-1);
}

XLATensorPtr diagonal(const XLATensorPtr& input, int64_t offset, int64_t dim1,
                      int64_t dim2) {
  auto input_shape = input->shape();
  int64_t canonical_dim1 = torch::lazy::GetCanonicalDimensionIndex(
      dim1, input->shape().get().rank());
  int64_t canonical_dim2 = torch::lazy::GetCanonicalDimensionIndex(
      dim2, input->shape().get().rank());
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    DiagonalInfo diagonal_info;
    diagonal_info.offset = offset;
    diagonal_info.dim1 = canonical_dim1;
    diagonal_info.dim2 = canonical_dim2;
    ViewInfo view_info(ViewInfo::Type::kDiagonal, input_shape,
                       std::move(diagonal_info));
    return input->CreateViewTensor(std::move(view_info));
  }

  return input->CreateFrom(torch::lazy::MakeNode<Diagonal>(
      input->GetIrValue(), offset, canonical_dim1, canonical_dim2));
}

XLATensorPtr div(const XLATensorPtr& input, const XLATensorPtr& other,
                 const c10::optional<c10::string_view>& rounding_mode,
                 c10::optional<at::ScalarType> logical_element_type) {
  at::ScalarType scalar_type =
      at::typeMetaToScalarType(c10::get_default_dtype());
  xla::PrimitiveType input_type = input->shape().get().element_type();
  xla::PrimitiveType other_type = other->shape().get().element_type();
  bool input_is_float = xla::primitive_util::IsFloatingPointType(input_type);
  bool other_is_float = xla::primitive_util::IsFloatingPointType(other_type);
  if (input_is_float && !other_is_float) {
    scalar_type = TensorTypeFromXlaType(input_type);
  } else if (!input_is_float && other_is_float) {
    scalar_type = TensorTypeFromXlaType(other_type);
  }
  // We need to cast both input and other to float to perform true divide, floor
  // divide and trunc divide.
  torch::lazy::Value input_value = GetFloatingIrValue(input, scalar_type);
  torch::lazy::Value other_value = GetFloatingIrValue(other, scalar_type);
  torch::lazy::Value res = input_value / other_value;

  if (rounding_mode.has_value()) {
    if (*rounding_mode == "trunc") {
      res = torch::lazy::MakeNode<Trunc>(res);
    } else if (*rounding_mode == "floor") {
      res = torch::lazy::MakeNode<Floor>(res);
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
          MakeXlaPrimitiveType(*logical_element_type, &input->GetDevice());
      if (GetXlaShape(res).element_type() != res_intended_type) {
        res = torch::lazy::MakeNode<Cast>(res, res_intended_type);
      }
    }
    return input->CreateFrom(res, logical_element_type);
  } else {
    // We don't need to typecheck the res IR here since we cast both input and
    // output to the scalar_type. Res type must also be scalar_type here.
    return input->CreateFrom(res, scalar_type);
  }
}

XLATensorPtr div(const XLATensorPtr& input, const at::Scalar& other) {
  at::ScalarType scalar_type =
      at::typeMetaToScalarType(c10::get_default_dtype());
  xla::PrimitiveType input_type = input->shape().get().element_type();
  bool input_is_float = xla::primitive_util::IsFloatingPointType(input_type);
  if (input_is_float) {
    scalar_type = TensorTypeFromXlaType(input_type);
  }
  torch::lazy::Value input_value = GetFloatingIrValue(input, scalar_type);
  torch::lazy::Value other_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      other, GetXlaShape(input_value).element_type(), input->GetDevice());
  return input->CreateFrom(input_value / other_value, scalar_type);
}

XLATensorPtr einsum(const std::string& equation,
                    const absl::Span<const XLATensorPtr> tensors) {
  std::vector<torch::lazy::Value> irs;
  irs.reserve(tensors.size());
  for (const XLATensorPtr& tensor : tensors) {
    irs.push_back(tensor->GetIrValue());
  }

  return tensors[0]->CreateFrom(torch::lazy::MakeNode<Einsum>(irs, equation));
}

std::tuple<XLATensorPtr, XLATensorPtr> einsum_backward(
    const XLATensorPtr& grad_output,
    const absl::Span<const XLATensorPtr> tensors, const std::string& equation) {
  std::vector<torch::lazy::Value> irs;
  irs.reserve(tensors.size());
  for (const XLATensorPtr& tensor : tensors) {
    irs.push_back(tensor->GetIrValue());
  }

  torch::lazy::NodePtr node = torch::lazy::MakeNode<EinsumBackward>(
      grad_output->GetIrValue(), irs, equation);

  if (node->num_outputs() == 2) {
    return std::make_tuple(
        grad_output->CreateFrom(torch::lazy::Value(node, 0)),
        grad_output->CreateFrom(torch::lazy::Value(node, 1)));
  } else {
    return std::make_tuple(grad_output->CreateFrom(torch::lazy::Value(node, 0)),
                           XLATensorPtr());
  }
}

XLATensorPtr eq(const XLATensorPtr& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

XLATensorPtr eq(const XLATensorPtr& input, const XLATensorPtr& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

XLATensorPtr elu_backward(const XLATensorPtr& grad_output,
                          const at::Scalar& alpha, const at::Scalar& scale,
                          const at::Scalar& input_scale,
                          const XLATensorPtr& output) {
  return grad_output->CreateFrom(EluBackward(grad_output->GetIrValue(),
                                             output->GetIrValue(), alpha, scale,
                                             input_scale));
}

XLATensorPtr embedding_dense_backward(const XLATensorPtr& grad_output,
                                      const XLATensorPtr& indices,
                                      int64_t num_weights, int64_t padding_idx,
                                      bool scale_grad_by_freq) {
  return tensor_ops::EmbeddingDenseBackward(grad_output, indices, num_weights,
                                            padding_idx, scale_grad_by_freq);
}

XLATensorPtr exp(const XLATensorPtr& input) {
  return input->CreateFrom(Exp(input->GetIrValue()));
}

XLATensorPtr expand(const XLATensorPtr& input, std::vector<int64_t> size) {
  auto input_shape = input->shape();
  auto output = input->CreateFrom(torch::lazy::MakeNode<Expand>(
      input->GetIrValue(),
      GetExpandDimensions(input_shape.get(), std::move(size))));
  output->SetStorage(input->Storage());
  return output;
}

XLATensorPtr expand_symint(const XLATensorPtr& input,
                           c10::SymIntArrayRef sym_size) {
  SymIntElements size_elements = SymIntElements(sym_size);
  XLATensorPtr output = input->CreateFrom(
      torch::lazy::MakeNode<ExpandSymInt>(input->GetIrValue(), size_elements));
  output->SetStorage(input->Storage());
  return output;
}

void exponential_(XLATensorPtr& input, double lambd) {
  auto input_shape = input->shape();
  input->SetInPlaceIrValue(torch::lazy::MakeNode<Exponential>(
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          lambd, input_shape.get().element_type(), input->GetDevice()),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()),
      input_shape.get()));
}

XLATensorPtr eye(int64_t lines, int64_t cols,
                 const torch::lazy::BackendDevice& device,
                 at::ScalarType element_type) {
  return XLATensor::Create(
      Identity(lines, cols, MakeXlaPrimitiveType(element_type, &device)),
      device, element_type);
}

void eye_out(XLATensorPtr& out, int64_t lines, int64_t cols) {
  out->SetIrValue(
      Identity(lines, cols >= 0 ? cols : lines,
               GetDevicePrimitiveType(out->shape().get().element_type(),
                                      &out->GetDevice())));
}

void fill_(XLATensorPtr& input, const at::Scalar& value) {
  // Fill_ is implemented by expanding the pass in value to the same shape of
  // input and replace input Tensor's IR with the expanded value. In order to
  // support input with dynamic shapes, we need to expand the value to its
  // dynamic dimension hence we need to create a sym_int_elements here.
  SymIntElements sym_int_elements(input->GetIrValue());
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      value, input->shape(), sym_int_elements, c10::nullopt,
      input->GetDevice());
  input->SetInPlaceIrValue(std::move(constant));
}

XLATensorPtr flip(const XLATensorPtr& input, absl::Span<const int64_t> dims) {
  auto dimensions = torch::lazy::GetCanonicalDimensionIndices(
      torch_xla::runtime::util::ToVector<int64_t>(dims),
      input->shape().get().rank());
  std::set<int64_t> unique_dims(dimensions.begin(), dimensions.end());
  XLA_CHECK_EQ(unique_dims.size(), dimensions.size());
  return input->CreateFrom(
      torch::lazy::MakeNode<Flip>(input->GetIrValue(), dimensions));
}

XLATensorPtr fmod(const XLATensorPtr& input, const XLATensorPtr& other,
                  c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(Fmod(input->GetIrValue(), other->GetIrValue()),
                           logical_element_type);
}

XLATensorPtr fmod(const XLATensorPtr& input, const at::Scalar& other,
                  c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      other, input->shape(), logical_element_type, input->GetDevice());
  return input->CreateFrom(Fmod(input->GetIrValue(), constant),
                           logical_element_type);
}

XLATensorPtr full(absl::Span<const int64_t> size, const at::Scalar& fill_value,
                  const torch::lazy::BackendDevice& device,
                  at::ScalarType scalar_type) {
  CheckShapeDimensions(size);
  xla::Shape shape =
      MakeArrayShapeFromDimensions(size, /*dynamic_dimensions=*/{},
                                   MakeXlaPrimitiveType(scalar_type, &device),
                                   static_cast<XlaDeviceType>(device.type()));
  return XLATensor::Create(
      XLAGraphExecutor::Get()->GetIrValueForScalar(fill_value, shape, device),
      device, scalar_type);
}

XLATensorPtr full_like(const XLATensorPtr& input, const at::Scalar& fill_value,
                       const torch::lazy::BackendDevice& device,
                       c10::optional<at::ScalarType> scalar_type) {
  xla::Shape tensor_shape = input->shape();
  if (scalar_type) {
    tensor_shape.set_element_type(MakeXlaPrimitiveType(*scalar_type, &device));
  } else {
    scalar_type = input->dtype();
  }
  return input->CreateFrom(XLAGraphExecutor::Get()->GetIrValueForScalar(
                               fill_value, tensor_shape, device),
                           device, *scalar_type);
}

XLATensorPtr full_symint(at::SymIntArrayRef sym_size,
                         const at::Scalar& fill_value,
                         const torch::lazy::BackendDevice& device,
                         at::ScalarType scalar_type) {
  XLA_CHECK(std::all_of(sym_size.begin(), sym_size.end(), [](at::SymInt dim) {
    // TODO: It should be OK to perform this test on symbolic ints too, not
    // sure why you conditionalized it.
    if (auto c = dim.maybe_as_int()) {
      return *c >= 0;
    }
    return true;
  })) << "Dimensions cannot be negative numbers";

  return XLATensor::Create(
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          fill_value, MakeXlaPrimitiveType(scalar_type, &device), sym_size,
          device),
      device, scalar_type);
}

XLATensorPtr gather(const XLATensorPtr& input, int64_t dim,
                    const XLATensorPtr& index) {
  xla::Shape input_shape = input->shape();
  xla::Shape index_shape = index->shape();
  XLA_CHECK_EQ(input_shape.rank(), index_shape.rank());
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.rank());
  for (size_t dim = 0; dim < input_shape.rank(); dim++) {
    if (dim != canonical_dim) {
      XLA_CHECK_LE(index->size(dim), input->size(dim));
    }
  }
  return input->CreateFrom(torch::lazy::MakeNode<Gather>(
      input->GetIrValue(), canonical_dim, index->GetIrValue()));
}

XLATensorPtr ge(const XLATensorPtr& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

XLATensorPtr ge(const XLATensorPtr& input, const XLATensorPtr& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

XLATensorPtr gelu(const XLATensorPtr& input,
                  const c10::string_view approximate) {
  if (approximate == "none") {
    return input->CreateFrom(Gelu(input->GetIrValue()));
  } else if (approximate == "tanh") {
    return input->CreateFrom(TanhGelu(input->GetIrValue()));
  } else {
    XLA_ERROR() << "Unknown gelu type: " << approximate;
  }
}

XLATensorPtr gelu_backward(const XLATensorPtr& grad, const XLATensorPtr& input,
                           const c10::string_view approximate) {
  if (approximate == "none") {
    return input->CreateFrom(
        GeluBackward(grad->GetIrValue(), input->GetIrValue()));
  } else if (approximate == "tanh") {
    return input->CreateFrom(
        TanhGeluBackward(grad->GetIrValue(), input->GetIrValue()));
  } else {
    XLA_ERROR() << "Unknown gelu type: " << approximate;
  }
}

XLATensorPtr gt(const XLATensorPtr& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

XLATensorPtr gt(const XLATensorPtr& input, const XLATensorPtr& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

XLATensorPtr index(const XLATensorPtr& input,
                   absl::Span<const XLATensorPtr> indices, int64_t start_dim) {
  return IndexByTensors(input, indices, start_dim);
}

XLATensorPtr index_add(const XLATensorPtr& input, int64_t dim,
                       const XLATensorPtr& index, const XLATensorPtr& source,
                       const at::Scalar& alpha) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      alpha, source->shape().get().element_type(), input->GetDevice());
  auto scaled_source = input->CreateFrom(source->GetIrValue() * constant);
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  return input->CreateFrom(
      IndexAdd(input, canonical_dim, index, scaled_source));
}

XLATensorPtr index_copy(const XLATensorPtr& input, int64_t dim,
                        const XLATensorPtr& index, const XLATensorPtr& source) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  return input->CreateFrom(IndexCopy(input, canonical_dim, index, source));
}

XLATensorPtr index_fill(const XLATensorPtr& input, int64_t dim,
                        const XLATensorPtr& index, const at::Scalar& value) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  return input->CreateFrom(IndexFill(input, canonical_dim, index, value));
}

XLATensorPtr index_fill(const XLATensorPtr& input, int64_t dim,
                        const XLATensorPtr& index, const XLATensorPtr& value) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  return input->CreateFrom(IndexFill(input, canonical_dim, index, value));
}

void index_fill_(XLATensorPtr& input, int64_t dim, const XLATensorPtr& index,
                 const XLATensorPtr& value) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  input->SetIrValue(IndexFill(input, canonical_dim, index, value));
}

void index_fill_(XLATensorPtr& input, int64_t dim, const XLATensorPtr& index,
                 const at::Scalar& value) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  input->SetIrValue(IndexFill(input, canonical_dim, index, value));
}

XLATensorPtr index_put(const XLATensorPtr& input,
                       absl::Span<const XLATensorPtr> indices,
                       int64_t start_dim, const XLATensorPtr& values,
                       bool accumulate,
                       absl::Span<const int64_t> result_permutation) {
  return input->CreateFrom(IndexPutByTensors(input, indices, start_dim, values,
                                             accumulate, result_permutation));
}

void index_put_(XLATensorPtr& input, const XLATensorPtr& canonical_base,
                absl::Span<const XLATensorPtr> indices, int64_t start_dim,
                const XLATensorPtr& values, bool accumulate,
                absl::Span<const int64_t> result_permutation) {
  input->SetIrValue(IndexPutByTensors(canonical_base, indices, start_dim,
                                      values, accumulate, result_permutation));
}

XLATensorPtr index_select(const XLATensorPtr& input, int64_t dim,
                          const XLATensorPtr& index) {
  torch::lazy::Value index_value = EnsureRank1(index->GetIrValue());
  return input->CreateFrom(torch::lazy::MakeNode<IndexSelect>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank()),
      index_value));
}

XLATensorPtr isnan(const XLATensorPtr& input) {
  torch::lazy::Value result = torch::lazy::MakeNode<Isnan>(input->GetIrValue());
  torch::lazy::Value casted = GetBooleanIrValue(result);
  return input->CreateFrom(casted, at::ScalarType::Bool);
}

std::tuple<XLATensorPtr, XLATensorPtr> kthvalue(const XLATensorPtr& input,
                                                int64_t k, int64_t dim,
                                                bool keepdim) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<KthValue>(
      input->GetIrValue(), k,
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank()),
      keepdim);
  return std::make_tuple(
      input->CreateFrom(torch::lazy::Value(node, 0)),
      input->CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

XLATensorPtr le(const XLATensorPtr& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

XLATensorPtr le(const XLATensorPtr& input, const XLATensorPtr& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

XLATensorPtr hardtanh_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const at::Scalar& min_val,
                               const at::Scalar& max_val) {
  return grad_output->CreateFrom(torch::lazy::MakeNode<HardtanhBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), min_val, max_val));
}

XLATensorPtr lerp(const XLATensorPtr& input, const XLATensorPtr& end,
                  const XLATensorPtr& weight) {
  return input->CreateFrom(
      Lerp(input->GetIrValue(), end->GetIrValue(), weight->GetIrValue()));
}

XLATensorPtr lerp(const XLATensorPtr& input, const XLATensorPtr& end,
                  const at::Scalar& weight) {
  torch::lazy::Value weight_val = XLAGraphExecutor::Get()->GetIrValueForScalar(
      weight, input->shape().get().element_type(), input->GetDevice());
  return input->CreateFrom(
      Lerp(input->GetIrValue(), end->GetIrValue(), weight_val));
}

XLATensorPtr linalg_vector_norm(const XLATensorPtr& input,
                                const at::Scalar& ord,
                                std::vector<int64_t> dimensions, bool keep_dim,
                                c10::optional<at::ScalarType> dtype) {
  // If the input is a scalar, we have to manually create the dimensions vector.
  auto input_rank = input->shape().get().rank();
  std::vector<int64_t> canonical_dims;
  if (input_rank != 0) {
    canonical_dims = torch::lazy::GetCanonicalDimensionIndices(
        torch_xla::runtime::util::ToVector<int64_t>(dimensions), input_rank);
  } else {
    canonical_dims = {0};
  }
  torch::lazy::Value res = LinalgVectorNorm(input->GetIrValue(), ord,
                                            canonical_dims, keep_dim, dtype);
  if (!dtype) dtype = input->dtype_optional();
  xla::PrimitiveType res_intended_type =
      MakeXlaPrimitiveType(*dtype, &input->GetDevice());
  if (GetXlaShape(res).element_type() != res_intended_type) {
    res = torch::lazy::MakeNode<Cast>(res, res_intended_type);
  }
  return input->CreateFrom(res, dtype);
}

XLATensorPtr linspace(const at::Scalar& start, const at::Scalar& end,
                      const int64_t steps, at::ScalarType element_type,
                      const torch::lazy::BackendDevice& device) {
  torch::lazy::Value start_val = XLAGraphExecutor::Get()->GetIrValueForScalar(
      start, xla::PrimitiveType::F32, device);
  torch::lazy::Value end_val = XLAGraphExecutor::Get()->GetIrValueForScalar(
      end, xla::PrimitiveType::F32, device);
  return XLATensor::Create(
      torch::lazy::MakeNode<Linspace>(start_val, end_val, steps), device,
      element_type);
}

XLATensorPtr log(const XLATensorPtr& input) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of log(int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input->CreateFrom(
      Log(GetFloatingIrValue(input, at::ScalarType::Float)), c10::nullopt);
}

XLATensorPtr log_base(const XLATensorPtr& input, torch::lazy::OpKind op,
                      double base) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of logbase(int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input->CreateFrom(
      LogBase(GetFloatingIrValue(input, at::ScalarType::Float), op, base),
      c10::nullopt);
}

XLATensorPtr log_sigmoid(const XLATensorPtr& input) {
  torch::lazy::NodePtr node = LogSigmoid(input->GetIrValue());
  return input->CreateFrom(torch::lazy::Value(node, 0));
}

XLATensorPtr log_softmax(const XLATensorPtr& input, int64_t dim,
                         c10::optional<at::ScalarType> dtype,
                         std::vector<torch::lazy::Shape>&& shapes) {
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<LogSoftmax>(input->GetIrValue(),
                                        torch::lazy::GetCanonicalDimensionIndex(
                                            dim, input->shape().get().rank()),
                                        dtype, std::move(shapes)),
      dtype);
}

XLATensorPtr log_softmax_backward(const XLATensorPtr& grad_output,
                                  const XLATensorPtr& output, int64_t dim) {
  return grad_output->CreateFrom(LogSoftmaxBackwardOp(
      grad_output->GetIrValue(), output->GetIrValue(), dim));
}

XLATensorPtr log1p(const XLATensorPtr& input) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of log1p(int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input->CreateFrom(
      Log1p(GetFloatingIrValue(input, at::ScalarType::Float)), c10::nullopt);
}

void log1p_(XLATensorPtr& input) {
  input->SetInPlaceIrValue(Log1p(input->GetIrValue()));
}

XLATensorPtr logsumexp(const XLATensorPtr& input,
                       std::vector<int64_t> dimensions,
                       bool keep_reduced_dimensions) {
  return input->CreateFrom(torch::lazy::MakeNode<Logsumexp>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndices(
          torch_xla::runtime::util::ToVector<int64_t>(dimensions),
          input->shape().get().rank()),
      keep_reduced_dimensions));
}

XLATensorPtr xlogy(const XLATensorPtr& input, const XLATensorPtr& other) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of xlogy(int,int) -> float, we want to derive the dtype from IR value
  // instead of input's logical_element_type.
  return input->CreateFrom(
      XLogY(input->GetIrValue(),
            GetFloatingIrValue(other, at::ScalarType::Float)),
      c10::nullopt);
}

XLATensorPtr lt(const XLATensorPtr& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

XLATensorPtr lt(const XLATensorPtr& input, const XLATensorPtr& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

XLATensorPtr masked_fill(XLATensorPtr& input, const XLATensorPtr& mask,
                         const at::Scalar& value) {
  torch::lazy::ScopePusher ir_scope(at::aten::masked_fill.toQualString());
  auto input_value = input->GetIrValue();
  // Expand input tensor to mask if needed (same as masked_scatter below).
  // An additional check makes sure to only expand if the rank of input tensor
  // is less than that of the mask tensor.
  if (input->shape().get().rank() <= mask->shape().get().rank() &&
      input->shape().get().dimensions() < mask->shape().get().dimensions()) {
    input_value = MaybeExpand(input->GetIrValue(), mask->shape());
  }
  return input->CreateFrom(torch::lazy::MakeNode<MaskedFill>(
      input_value, MaybeExpand(mask->GetIrValue(), GetXlaShape(input_value)),
      value));
}

XLATensorPtr masked_scatter(XLATensorPtr& input, const XLATensorPtr& mask,
                            const XLATensorPtr& source) {
  torch::lazy::ScopePusher ir_scope(at::aten::masked_scatter.toQualString());
  auto input_value = input->GetIrValue();
  // This ensures that input tensor is at least the same shape as mask tensor.
  // Note that we can't use the existing MaybeExpand function since
  // input tensor may sometimes be bigger than the mask tensor, and MaybeExpand
  // requires the first parameter to always be less or equal to the second
  // parameter.
  if (input->shape().get().dimensions() < mask->shape().get().dimensions()) {
    input_value = MaybeExpand(input->GetIrValue(), mask->shape());
  }
  return input->CreateFrom(torch::lazy::MakeNode<MaskedScatter>(
      input_value, MaybeExpand(mask->GetIrValue(), GetXlaShape(input_value)),
      source->GetIrValue()));
}

XLATensorPtr masked_select(const XLATensorPtr& input,
                           const XLATensorPtr& mask) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<MaskedSelect>(
      input->GetIrValue(), mask->GetIrValue());
  return input->CreateFrom(torch::lazy::Value(node, 0));
}

XLATensorPtr matmul(const XLATensorPtr& input, const XLATensorPtr& other) {
  return input->CreateFrom(MatMul(input->GetIrValue(), other->GetIrValue()));
}

XLATensorPtr max(const XLATensorPtr& input) {
  return input->CreateFrom(MaxUnary(input->GetIrValue()), input->dtype());
}

std::tuple<XLATensorPtr, XLATensorPtr> max(const XLATensorPtr& input,
                                           int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<MaxInDim>(
      input->GetIrValue(), canonical_dim, keepdim);
  return std::make_tuple(
      input->CreateFrom(torch::lazy::Value(node, 0)),
      input->CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

void max_out(XLATensorPtr& max, XLATensorPtr& max_values,
             const XLATensorPtr& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<MaxInDim>(
      input->GetIrValue(), canonical_dim, keepdim);
  max->SetIrValue(torch::lazy::Value(node, 0));
  max_values->SetIrValue(torch::lazy::Value(node, 1));
}

std::tuple<XLATensorPtr, XLATensorPtr> max_pool_nd(
    const XLATensorPtr& input, int64_t spatial_dim_count,
    std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
    std::vector<int64_t> padding, bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  torch::lazy::NodePtr node = torch::lazy::MakeNode<MaxPoolNd>(
      input->GetIrValue(), spatial_dim_count, std::move(kernel_size),
      std::move(stride), std::move(padding), ceil_mode);
  return std::make_tuple(
      input->CreateFrom(torch::lazy::Value(node, 0)),
      input->CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

XLATensorPtr max_pool_nd_backward(
    const XLATensorPtr& out_backprop, const XLATensorPtr& input,
    int64_t spatial_dim_count, std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode) {
  kernel_size = CheckIntList(kernel_size, spatial_dim_count, "kernel_size");
  stride = CheckIntList(stride, spatial_dim_count, "stride", kernel_size);
  padding = CheckIntList(padding, spatial_dim_count, "padding");
  return out_backprop->CreateFrom(torch::lazy::MakeNode<MaxPoolNdBackward>(
      out_backprop->GetIrValue(), input->GetIrValue(), spatial_dim_count,
      std::move(kernel_size), std::move(stride), std::move(padding),
      ceil_mode));
}

XLATensorPtr max_unpool(const XLATensorPtr& input, const XLATensorPtr& indices,
                        std::vector<int64_t> output_size) {
  return input->CreateFrom(torch::lazy::MakeNode<MaxUnpoolNd>(
      input->GetIrValue(), indices->GetIrValue(), std::move(output_size)));
}

XLATensorPtr mean(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                  bool keep_reduced_dimensions,
                  c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<Mean>(
          input->GetIrValue(),
          torch::lazy::GetCanonicalDimensionIndices(
              torch_xla::runtime::util::ToVector<int64_t>(dimensions),
              input->shape().get().rank()),
          keep_reduced_dimensions, dtype),
      dtype);
}

XLATensorPtr min(const XLATensorPtr& input, const XLATensorPtr& other,
                 c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(Min(input->GetIrValue(), other->GetIrValue()),
                           logical_element_type);
}

XLATensorPtr min(const XLATensorPtr& input) {
  return input->CreateFrom(MinUnary(input->GetIrValue()), input->dtype());
}

std::tuple<XLATensorPtr, XLATensorPtr> min(const XLATensorPtr& input,
                                           int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<MinInDim>(
      input->GetIrValue(), canonical_dim, keepdim);
  return std::make_tuple(
      input->CreateFrom(torch::lazy::Value(node, 0)),
      input->CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

void min_out(XLATensorPtr& min, XLATensorPtr& min_indices,
             const XLATensorPtr& input, int64_t dim, bool keepdim) {
  int64_t canonical_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<MinInDim>(
      input->GetIrValue(), canonical_dim, keepdim);
  min->SetIrValue(torch::lazy::Value(node, 0));
  min_indices->SetIrValue(torch::lazy::Value(node, 1));
}

XLATensorPtr mish(const XLATensorPtr& input) {
  return input->CreateFrom(
      input->GetIrValue() *
      torch::lazy::MakeNode<Tanh>(
          tensor_ops::Softplus(input, 1, 20)->GetIrValue()));
}

XLATensorPtr mm(const XLATensorPtr& input, const XLATensorPtr& weight) {
  return input->CreateFrom(Dot(input->GetIrValue(), weight->GetIrValue()));
}

XLATensorPtr mse_loss(const XLATensorPtr& input, const XLATensorPtr& target,
                      int64_t reduction) {
  // Here we explictly pass c10::nullopt as logical_element_type because
  // otherwise result will inherit the input's logical_element_type. In the
  // case of mse_loss(long, float16) -> float16, we want to derive the dtype
  // from IR value instead of input's logical_element_type.
  return input->CreateFrom(
      torch::lazy::MakeNode<MseLoss>(input->GetIrValue(), target->GetIrValue(),
                                     GetXlaReductionMode(reduction)),
      c10::nullopt);
}

XLATensorPtr mse_loss_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const XLATensorPtr& target, int64_t reduction) {
  return input->CreateFrom(torch::lazy::MakeNode<MseLossBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), target->GetIrValue(),
      GetXlaReductionMode(reduction)));
}

XLATensorPtr mul(const XLATensorPtr& input, const XLATensorPtr& other,
                 c10::optional<at::ScalarType> logical_element_type) {
  return input->CreateFrom(input->GetIrValue() * other->GetIrValue(),
                           logical_element_type);
}

XLATensorPtr mul(const XLATensorPtr& input, const at::Scalar& other,
                 c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      other, input->shape(), logical_element_type, input->GetDevice());
  return input->CreateFrom(input->GetIrValue() * constant,
                           logical_element_type);
}

XLATensorPtr multinomial(const XLATensorPtr& input, int64_t num_samples,
                         bool replacement) {
  auto input_shape = input->shape();
  return input->CreateFrom(
      torch::lazy::MakeNode<Multinomial>(
          input->GetIrValue(),
          XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()), num_samples,
          replacement),
      at::ScalarType::Long);
}

XLATensorPtr mv(const XLATensorPtr& input, const XLATensorPtr& vec) {
  return input->CreateFrom(Dot(input->GetIrValue(), vec->GetIrValue()));
}

void mv_out(XLATensorPtr& out, const XLATensorPtr& input,
            const XLATensorPtr& vec) {
  out->SetIrValue(Dot(input->GetIrValue(), vec->GetIrValue()));
}

XLATensorPtr nan_to_num(const XLATensorPtr& input, const at::Scalar& nan,
                        const at::Scalar& posinf, const at::Scalar& neginf) {
  torch::lazy::Value nan_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      nan, input->shape(), input->GetDevice());
  torch::lazy::Value posinf_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(posinf, input->shape(),
                                                   input->GetDevice());
  torch::lazy::Value neginf_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(neginf, input->shape(),
                                                   input->GetDevice());
  return input->CreateFrom(
      NanToNum(input->GetIrValue(), nan_value, posinf_value, neginf_value));
}

XLATensorPtr narrow(const XLATensorPtr& input, int64_t dim, int64_t start,
                    int64_t length) {
  auto input_shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.get().rank());

  xla::Shape narrow_shape = input_shape;
  narrow_shape.set_dimensions(dim, length);

  std::vector<int64_t> indices(input_shape.get().rank(), 0);
  indices[dim] = torch::lazy::GetCanonicalPosition(
      torch_xla::runtime::util::ToVector<int64_t>(
          input_shape.get().dimensions()),
      dim, start);

  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    ViewInfo::Type view_type = (xla::ShapeUtil::ElementsIn(input_shape) ==
                                xla::ShapeUtil::ElementsIn(narrow_shape))
                                   ? ViewInfo::Type::kReshape
                                   : ViewInfo::Type::kNarrow;
    ViewInfo view_info(view_type, std::move(narrow_shape), input_shape);
    view_info.indices[dim] = indices[dim];
    return input->CreateViewTensor(std::move(view_info));
  }

  return input->CreateFrom(torch::lazy::MakeNode<GenericSlice>(
      input->GetIrValue(), std::move(indices), narrow_shape.dimensions()));
}

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> native_batch_norm(
    const XLATensorPtr& input, const XLATensorPtr& weight,
    const XLATensorPtr& bias, XLATensorPtr& running_mean,
    XLATensorPtr& running_var, bool training, double momentum, double eps) {
  xla::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input->GetDevice());
  torch::lazy::Value bias_value =
      GetIrValueOrDefault(bias, 0, features_shape, input->GetDevice());
  torch::lazy::Value running_mean_value =
      GetIrValueOrDefault(running_mean, 0, features_shape, input->GetDevice());
  torch::lazy::Value running_var_value =
      GetIrValueOrDefault(running_var, 0, features_shape, input->GetDevice());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<NativeBatchNormForward>(
      input->GetIrValue(), weight_value, bias_value, running_mean_value,
      running_var_value, training, eps);
  XLATensorPtr output = input->CreateFrom(torch::lazy::Value(node, 0));
  XLATensorPtr mean;
  XLATensorPtr variance_inverse;
  if (training) {
    mean = input->CreateFrom(torch::lazy::Value(node, 1));
    variance_inverse = input->CreateFrom(torch::lazy::Value(node, 3));
    if (running_mean) {
      running_mean->SetIrValue(torch::lazy::MakeNode<LinearInterpolation>(
          mean->GetIrValue(), running_mean->GetIrValue(), momentum));
    }
    if (running_var) {
      running_var->SetIrValue(torch::lazy::MakeNode<LinearInterpolation>(
          torch::lazy::Value(node, 2), running_var->GetIrValue(), momentum));
    }
  } else {
    at::Tensor at_input = bridge::AtenFromXlaTensor(input);
    mean = bridge::GetXlaTensor(at::empty({0}, at_input.options()));
    variance_inverse = bridge::GetXlaTensor(at::empty({0}, at_input.options()));
  }
  return std::make_tuple(std::move(output), std::move(mean),
                         std::move(variance_inverse));
}

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> native_batch_norm_backward(
    const XLATensorPtr& grad_out, const XLATensorPtr& input,
    const XLATensorPtr& weight, const XLATensorPtr& save_mean,
    const XLATensorPtr& save_invstd, bool training, double eps) {
  xla::Shape features_shape = BatchNormFeaturesShape(input);
  torch::lazy::Value weight_value =
      GetIrValueOrDefault(weight, 1, features_shape, input->GetDevice());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<NativeBatchNormBackward>(
      grad_out->GetIrValue(), input->GetIrValue(), weight_value,
      save_mean->GetIrValue(), save_invstd->GetIrValue(), training, eps);
  XLATensorPtr grad_input = input->CreateFrom(torch::lazy::Value(node, 0));
  XLATensorPtr grad_weight = input->CreateFrom(torch::lazy::Value(node, 1));
  XLATensorPtr grad_bias = input->CreateFrom(torch::lazy::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensorPtr ne(const XLATensorPtr& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

XLATensorPtr ne(const XLATensorPtr& input, const XLATensorPtr& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

XLATensorPtr neg(const XLATensorPtr& input) {
  return input->CreateFrom(Neg(input->GetIrValue()));
}

XLATensorPtr nll_loss(const XLATensorPtr& input, const XLATensorPtr& target,
                      const XLATensorPtr& weight, int64_t reduction,
                      int ignore_index) {
  return input->CreateFrom(torch::lazy::MakeNode<NllLoss>(
      input->GetIrValue(), target->GetIrValue(), GetOptionalIrValue(weight),
      GetXlaReductionMode(reduction), ignore_index));
}

XLATensorPtr nll_loss2d(const XLATensorPtr& input, const XLATensorPtr& target,
                        const XLATensorPtr& weight, int64_t reduction,
                        int ignore_index) {
  return input->CreateFrom(torch::lazy::MakeNode<NllLoss2d>(
      input->GetIrValue(), target->GetIrValue(), GetOptionalIrValue(weight),
      GetXlaReductionMode(reduction), ignore_index));
}

XLATensorPtr nll_loss2d_backward(const XLATensorPtr& grad_output,
                                 const XLATensorPtr& input,
                                 const XLATensorPtr& target,
                                 const XLATensorPtr& weight, int64_t reduction,
                                 int ignore_index,
                                 const XLATensorPtr& total_weight) {
  return input->CreateFrom(torch::lazy::MakeNode<NllLoss2dBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), target->GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetXlaReductionMode(reduction), ignore_index));
}

XLATensorPtr nll_loss_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const XLATensorPtr& target,
                               const XLATensorPtr& weight, int64_t reduction,
                               int ignore_index,
                               const XLATensorPtr& total_weight) {
  return input->CreateFrom(torch::lazy::MakeNode<NllLossBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), target->GetIrValue(),
      GetOptionalIrValue(weight), GetOptionalIrValue(total_weight),
      GetXlaReductionMode(reduction), ignore_index));
}

std::pair<XLATensorPtr, XLATensorPtr> nms(const XLATensorPtr& boxes,
                                          const XLATensorPtr& scores,
                                          const XLATensorPtr& score_threshold,
                                          const XLATensorPtr& iou_threshold,
                                          int64_t output_size) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<Nms>(
      boxes->GetIrValue(), scores->GetIrValue(), score_threshold->GetIrValue(),
      iou_threshold->GetIrValue(), output_size);
  return std::pair<XLATensorPtr, XLATensorPtr>(
      XLATensor::Create(torch::lazy::Value(node, 0), boxes->GetDevice(),
                        at::ScalarType::Int),
      XLATensor::Create(torch::lazy::Value(node, 1), boxes->GetDevice(),
                        at::ScalarType::Int));
}

XLATensorPtr nonzero(const XLATensorPtr& input) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<NonZero>(input->GetIrValue());
  // Nonzero result type should not depend on input type, hence we shouldn't
  // use input->CreateFrom which will inherit the logical_element_type.
  return XLATensor::Create(torch::lazy::Value(node, 0), input->GetDevice());
}

XLATensorPtr norm(const XLATensorPtr& input, const c10::optional<at::Scalar>& p,
                  c10::optional<at::ScalarType> dtype, at::IntArrayRef dim,
                  bool keepdim) {
  auto canonical_dims = torch::lazy::GetCanonicalDimensionIndices(
      XlaHelpers::I64List(dim), input->shape().get().rank());
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      Norm(input->GetIrValue(), p, dtype, canonical_dims, keepdim));
}

XLATensorPtr normal(double mean, const XLATensorPtr& std) {
  return std->CreateFrom(torch::lazy::MakeNode<Normal>(
      XLAGraphExecutor::Get()->GetIrValueForScalar(mean, std->shape(),
                                                   std->GetDevice()),
      std->GetIrValue(),
      XLAGraphExecutor::Get()->GetRngSeed(std->GetDevice())));
}

XLATensorPtr normal(const XLATensorPtr& mean, double std) {
  return mean->CreateFrom(torch::lazy::MakeNode<Normal>(
      mean->GetIrValue(),
      XLAGraphExecutor::Get()->GetIrValueForScalar(std, mean->shape(),
                                                   mean->GetDevice()),
      XLAGraphExecutor::Get()->GetRngSeed(mean->GetDevice())));
}

XLATensorPtr normal(const XLATensorPtr& mean, const XLATensorPtr& std) {
  return mean->CreateFrom(torch::lazy::MakeNode<Normal>(
      mean->GetIrValue(), MaybeExpand(std->GetIrValue(), mean->shape()),
      XLAGraphExecutor::Get()->GetRngSeed(mean->GetDevice())));
}

void normal_(XLATensorPtr& input, double mean, double std) {
  input->SetInPlaceIrValue(torch::lazy::MakeNode<Normal>(
      XLAGraphExecutor::Get()->GetIrValueForScalar(mean, input->shape(),
                                                   input->GetDevice()),
      XLAGraphExecutor::Get()->GetIrValueForScalar(std, input->shape(),
                                                   input->GetDevice()),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice())));
}

XLATensorPtr not_supported(std::string description, xla::Shape shape,
                           const torch::lazy::BackendDevice& device) {
  return XLATensor::Create(torch::lazy::MakeNode<NotSupported>(
                               std::move(description), std::move(shape)),
                           device);
}

void optimization_barrier_(std::vector<XLATensorPtr>& tensors) {
  std::vector<torch::lazy::Value> irs;
  irs.reserve(tensors.size());
  for (XLATensorPtr& tensor : tensors) {
    irs.push_back(tensor->GetIrValue());
  }
  torch::lazy::NodePtr result = torch::lazy::MakeNode<OptimizationBarrier>(irs);
  for (int i = 0; i < tensors.size(); i++) {
    tensors[i]->SetInPlaceIrValue(torch::lazy::Value(result, i));
  }
}

XLATensorPtr permute(const XLATensorPtr& input,
                     absl::Span<const int64_t> dims) {
  auto input_shape = input->shape();
  std::vector<int64_t> dimensions = torch::lazy::GetCanonicalDimensionIndices(
      torch_xla::runtime::util::ToVector<int64_t>(dims),
      input_shape.get().rank());

  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    ViewInfo view_info(ViewInfo::Type::kPermute, input_shape, dimensions);
    return input->CreateViewTensor(std::move(view_info));
  }

  return input->CreateFrom(
      torch::lazy::MakeNode<Permute>(input->GetIrValue(), dimensions));
}

XLATensorPtr pow(const XLATensorPtr& input, const at::Scalar& exponent) {
  // We want to pass exponent_node as a constant to give XLA more room to
  // optimize
  torch::lazy::Value exponent_node =
      XLAGraphExecutor::Get()->GetIrValueForConstant(exponent, input->shape());
  return input->CreateFrom(Pow(input->GetIrValue(), exponent_node));
}

XLATensorPtr pow(const XLATensorPtr& input, const XLATensorPtr& exponent) {
  return input->CreateFrom(Pow(input->GetIrValue(), exponent->GetIrValue()));
}

XLATensorPtr pow(const at::Scalar& input, const XLATensorPtr& exponent) {
  const torch::lazy::BackendDevice& device = exponent->GetDevice();
  torch::lazy::Value input_node = XLAGraphExecutor::Get()->GetIrValueForScalar(
      input, MakeXlaPrimitiveType(GetScalarType(input), &device), device);
  torch::lazy::NodePtr pow_node = Pow(input_node, exponent->GetIrValue());
  at::ScalarType input_dtype = GetScalarType(input);
  at::ScalarType exp_dtype = exponent->dtype();
  at::ScalarType promoted_dtype = TensorTypeFromXlaType(XlaHelpers::PromoteType(
      TensorTypeToRawXlaType(input_dtype), TensorTypeToRawXlaType(exp_dtype)));
  return exponent->CreateFrom(pow_node, promoted_dtype);
}

XLATensorPtr prelu(const XLATensorPtr& input, const XLATensorPtr& weight) {
  return input->CreateFrom(Prelu(input->GetIrValue(), weight->GetIrValue()));
}

XLATensorPtr prod(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                  bool keep_reduced_dimensions,
                  c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<Prod>(
          input->GetIrValue(),
          torch::lazy::GetCanonicalDimensionIndices(
              torch_xla::runtime::util::ToVector<int64_t>(dimensions),
              input->shape().get().rank()),
          keep_reduced_dimensions, dtype),
      dtype);
}

void put_(XLATensorPtr& input, const XLATensorPtr& index,
          const XLATensorPtr& source, bool accumulate) {
  input->SetInPlaceIrValue(
      torch::lazy::MakeNode<Put>(input->GetIrValue(), index->GetIrValue(),
                                 source->GetIrValue(), accumulate));
}

std::tuple<XLATensorPtr, XLATensorPtr> qr(const XLATensorPtr& input,
                                          bool some) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<QR>(input->GetIrValue(), some);
  return std::make_tuple(input->CreateFrom(torch::lazy::Value(node, 0)),
                         input->CreateFrom(torch::lazy::Value(node, 1)));
}

void random_(XLATensorPtr& input, int64_t from, int64_t to) {
  XLA_CHECK_LE(from, to);
  auto input_shape = input->shape();
  input->SetInPlaceIrValue(torch::lazy::MakeNode<DiscreteUniform>(
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          from, xla::PrimitiveType::S64, input->GetDevice()),
      XLAGraphExecutor::Get()->GetIrValueForScalar(to, xla::PrimitiveType::S64,
                                                   input->GetDevice()),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()), input_shape));
}

XLATensorPtr reflection_pad2d(const XLATensorPtr& input,
                              std::vector<int64_t> padding) {
  return input->CreateFrom(torch::lazy::MakeNode<ReflectionPad2d>(
      input->GetIrValue(), std::move(padding)));
}

XLATensorPtr reflection_pad2d_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& input,
                                       std::vector<int64_t> padding) {
  return input->CreateFrom(torch::lazy::MakeNode<ReflectionPad2dBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), std::move(padding)));
}

XLATensorPtr remainder(const XLATensorPtr& input, const XLATensorPtr& other) {
  return input->CreateFrom(Remainder(input->GetIrValue(), other->GetIrValue()));
}

XLATensorPtr remainder(const XLATensorPtr& input, const at::Scalar& other) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      other, input->shape(), input->GetDevice());
  return input->CreateFrom(Remainder(input->GetIrValue(), constant));
}

XLATensorPtr replication_pad1d(const XLATensorPtr& input,
                               std::vector<int64_t> padding) {
  return input->CreateFrom(torch::lazy::MakeNode<ReplicationPad>(
      input->GetIrValue(), std::move(padding)));
}

XLATensorPtr replication_pad1d_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        std::vector<int64_t> padding) {
  return input->CreateFrom(torch::lazy::MakeNode<ReplicationPadBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), std::move(padding)));
}

XLATensorPtr replication_pad2d(const XLATensorPtr& input,
                               std::vector<int64_t> padding) {
  return input->CreateFrom(torch::lazy::MakeNode<ReplicationPad>(
      input->GetIrValue(), std::move(padding)));
}

XLATensorPtr replication_pad2d_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        std::vector<int64_t> padding) {
  return input->CreateFrom(torch::lazy::MakeNode<ReplicationPadBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), std::move(padding)));
}

void resize_(XLATensorPtr& input, std::vector<int64_t> size) {
  if (input->data()->view == nullptr) {
    input->SetIrValue(
        torch::lazy::MakeNode<Resize>(input->GetIrValue(), std::move(size)));
  } else {
    auto input_shape = input->shape();
    xla::Shape resize_shape =
        xla::ShapeUtil::MakeShape(input_shape.get().element_type(), size);
    ViewInfo view_info(ViewInfo::Type::kResize, std::move(resize_shape),
                       input_shape);
    input->SetSubView(std::move(view_info));
  }
}

XLATensorPtr roll(const XLATensorPtr& input, absl::Span<const int64_t> shifts,
                  absl::Span<const int64_t> dims) {
  XLA_CHECK_GT(shifts.size(), 0) << "`shifts` required";
  if (dims.size() != 0) {
    XLA_CHECK_EQ(shifts.size(), dims.size())
        << "shifts and dimensions must align. shifts: " << shifts.size()
        << ", dims:" << dims.size();
  }
  auto canonical_dims = torch::lazy::GetCanonicalDimensionIndices(
      torch::lazy::ToVector<int64_t>(dims), input->shape().get().rank());
  return input->CreateFrom(torch::lazy::MakeNode<Roll>(
      input->GetIrValue(), torch::lazy::ToVector<int64_t>(shifts),
      canonical_dims));
}

XLATensorPtr rrelu_with_noise(const XLATensorPtr& input, XLATensorPtr& noise,
                              const at::Scalar& lower, const at::Scalar& upper,
                              bool training) {
  torch::lazy::NodePtr output_node = torch::lazy::MakeNode<RreluWithNoise>(
      input->GetIrValue(),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()), lower, upper,
      training);
  noise->SetIrValue(torch::lazy::Value(output_node, 1));
  return input->CreateFrom(torch::lazy::Value(output_node, 0));
}

XLATensorPtr rrelu_with_noise_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& input,
                                       const XLATensorPtr& noise,
                                       const at::Scalar& lower,
                                       const at::Scalar& upper, bool training) {
  return grad_output->CreateFrom(torch::lazy::MakeNode<RreluWithNoiseBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), noise->GetIrValue(),
      lower, upper, training));
}

XLATensorPtr rsub(const XLATensorPtr& input, const XLATensorPtr& other,
                  const at::Scalar& alpha,
                  c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value alpha_xla = XLAGraphExecutor::Get()->GetIrValueForScalar(
      alpha, other->shape(), logical_element_type, other->GetDevice());
  return input->CreateFrom(
      other->GetIrValue() - alpha_xla * input->GetIrValue(),
      logical_element_type);
}

XLATensorPtr rsub(const XLATensorPtr& input, const at::Scalar& other,
                  const at::Scalar& alpha,
                  c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value alpha_xla = XLAGraphExecutor::Get()->GetIrValueForScalar(
      alpha, input->shape(), logical_element_type, input->GetDevice());
  torch::lazy::Value other_xla = XLAGraphExecutor::Get()->GetIrValueForScalar(
      other, input->shape(), logical_element_type, input->GetDevice());
  return input->CreateFrom(other_xla - alpha_xla * input->GetIrValue(),
                           logical_element_type);
}

void copy_(XLATensorPtr& input, XLATensorPtr& src) {
  if (input->GetDevice() == src->GetDevice()) {
    torch::lazy::Value copy_value;
    if (input->dtype() == src->dtype()) {
      copy_value = src->GetIrValue();
    } else {
      copy_value = torch::lazy::MakeNode<Cast>(src->GetIrValue(),
                                               input->dtype(), src->dtype());
    }
    input->SetIrValue(MaybeExpand(copy_value, input->shape()));
  } else {
    auto input_shape = input->shape();
    at::Tensor src_tensor = src->ToTensor(/*detached=*/true);
    if (!torch_xla::runtime::util::Equal(src_tensor.sizes(),
                                         input_shape.get().dimensions())) {
      src_tensor = src_tensor.expand(
          torch::lazy::ToVector<int64_t>(input_shape.get().dimensions()));
    }
    input->UpdateFromTensor(std::move(src_tensor), /*sync=*/false);
  }

  // Preserves sharding when copying.
  if (src->sharding_spec() != nullptr) {
    input->SetShardingSpec(*src->sharding_spec());
  }
}

XLATensorPtr scatter(const XLATensorPtr& input, int64_t dim,
                     const XLATensorPtr& index, const XLATensorPtr& src) {
  return input->CreateFrom(torch::lazy::MakeNode<Scatter>(
      input->GetIrValue(), index->GetIrValue(), src->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              input->shape().get().rank())));
}

XLATensorPtr scatter(const XLATensorPtr& input, int64_t dim,
                     const XLATensorPtr& index, const at::Scalar& value) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      value, input->shape(), input->GetDevice());
  return input->CreateFrom(torch::lazy::MakeNode<Scatter>(
      input->GetIrValue(), index->GetIrValue(), constant,
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              input->shape().get().rank())));
}

XLATensorPtr scatter_add(const XLATensorPtr& input, int64_t dim,
                         const XLATensorPtr& index, const XLATensorPtr& src) {
  return input->CreateFrom(torch::lazy::MakeNode<ScatterAdd>(
      input->GetIrValue(), index->GetIrValue(), src->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              input->shape().get().rank())));
}

XLATensorPtr scatter_add(const XLATensorPtr& input, int64_t dim,
                         const XLATensorPtr& index, const at::Scalar& value) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      value, input->shape(), input->GetDevice());
  return input->CreateFrom(torch::lazy::MakeNode<ScatterAdd>(
      input->GetIrValue(), index->GetIrValue(), constant,
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              input->shape().get().rank())));
}

XLATensorPtr scatter_reduce(const XLATensorPtr& input, int64_t dim,
                            const XLATensorPtr& index, const XLATensorPtr& src,
                            c10::string_view reduce, bool include_self) {
  return input->CreateFrom(torch::lazy::MakeNode<ScatterReduce>(
      input->GetIrValue(), index->GetIrValue(), src->GetIrValue(), reduce,
      include_self,
      torch::lazy::GetCanonicalDimensionIndex(dim,
                                              input->shape().get().rank())));
}

XLATensorPtr select(const XLATensorPtr& input, int64_t dim, int64_t index) {
  return tensor_ops::Select(input, dim, index);
}

void selu_(XLATensorPtr& input) {
  input->SetInPlaceIrValue(Selu(input->GetIrValue()));
}

XLATensorPtr sigmoid(const XLATensorPtr& input) {
  return input->CreateFrom(Sigmoid(input->GetIrValue()));
}

XLATensorPtr sigmoid_backward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& output) {
  return grad_output->CreateFrom(
      SigmoidBackward(grad_output->GetIrValue(), output->GetIrValue()));
}

XLATensorPtr slice(const XLATensorPtr& input, int64_t dim, int64_t start,
                   int64_t end, int64_t step) {
  auto input_shape = input->shape();
  dim = torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start = torch::lazy::GetCanonicalPosition(
      torch_xla::runtime::util::ToVector<int64_t>(
          input_shape.get().dimensions()),
      dim, start);
  end = torch::lazy::GetCanonicalPosition(
      torch_xla::runtime::util::ToVector<int64_t>(
          input_shape.get().dimensions()),
      dim, end);
  // PyTorch allows tensor[-1:0] to return a 0-dim tensor.
  if (start > end) {
    end = start;
  }
  step = std::min(step, end - start);

  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    SelectInfo select = {dim, start, end, step};
    ViewInfo view_info(ViewInfo::Type::kSelect, input_shape, std::move(select));
    return input->CreateViewTensor(std::move(view_info));
  }
  return input->CreateFrom(torch::lazy::MakeNode<Select>(
      input->GetIrValue(), dim, start, end, step));
}

std::tuple<XLATensorPtr, XLATensorPtr> slogdet(const XLATensorPtr& input) {
  torch::lazy::NodePtr node = SLogDet(input->GetIrValue());
  return std::make_tuple(input->CreateFrom(torch::lazy::Value(node, 0)),
                         input->CreateFrom(torch::lazy::Value(node, 1)));
}

XLATensorPtr smooth_l1_loss(const XLATensorPtr& input,
                            const XLATensorPtr& target, int64_t reduction,
                            double beta) {
  return tensor_ops::SmoothL1Loss(input, target, GetXlaReductionMode(reduction),
                                  beta);
}

XLATensorPtr smooth_l1_loss_backward(const XLATensorPtr& grad_output,
                                     const XLATensorPtr& input,
                                     const XLATensorPtr& target,
                                     int64_t reduction, double beta) {
  return tensor_ops::SmoothL1LossBackward(grad_output, input, target,
                                          GetXlaReductionMode(reduction), beta);
}

XLATensorPtr softmax(const XLATensorPtr& input, int64_t dim,
                     c10::optional<at::ScalarType> dtype) {
  if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<Softmax>(input->GetIrValue(),
                                     torch::lazy::GetCanonicalDimensionIndex(
                                         dim, input->shape().get().rank()),
                                     dtype),
      dtype);
}

XLATensorPtr softmax_backward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& output, int64_t dim) {
  return grad_output->CreateFrom(
      SoftmaxBackwardOp(grad_output->GetIrValue(), output->GetIrValue(), dim));
}

XLATensorPtr softplus(const XLATensorPtr& input, const at::Scalar& beta,
                      const at::Scalar& threshold) {
  torch::lazy::Value beta_value = XLAGraphExecutor::Get()->GetIrValueForScalar(
      beta, input->shape().get().element_type(), input->GetDevice());
  torch::lazy::Value threshold_value =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          threshold, input->shape().get().element_type(), input->GetDevice());
  return input->CreateFrom(
      Softplus(input->GetIrValue(), beta_value, threshold_value));
}

XLATensorPtr softplus_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const at::Scalar& beta,
                               const at::Scalar& threshold) {
  return tensor_ops::SoftplusBackward(grad_output, input, beta, threshold);
}

std::vector<XLATensorPtr> split(const XLATensorPtr& input, int64_t split_size,
                                int64_t dim) {
  auto input_shape = input->shape();
  int split_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  int64_t dim_size = input_shape.get().dimensions(split_dim);
  if (dim_size == 0) {
    // Deal with dim_size=0, it's a corner case which only return 1 0-dim tensor
    // no matter what split_size is.
    xla::Literal literal(input_shape.get());
    return {
        input->CreateFrom(torch::lazy::MakeNode<Constant>(std::move(literal)))};
  }
  std::vector<int64_t> split_sizes;
  for (; dim_size > 0; dim_size -= split_size) {
    split_sizes.push_back(std::min<int64_t>(dim_size, split_size));
  }
  torch::lazy::NodePtr node = torch::lazy::MakeNode<Split>(
      input->GetIrValue(), std::move(split_sizes), split_dim);
  return input->MakeOutputTensors(node);
}

std::vector<XLATensorPtr> split_with_sizes(const XLATensorPtr& input,
                                           std::vector<int64_t> split_size,
                                           int64_t dim) {
  auto input_shape = input->shape();
  int split_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  torch::lazy::NodePtr node = torch::lazy::MakeNode<Split>(
      input->GetIrValue(), std::move(split_size), split_dim);
  return input->MakeOutputTensors(node);
}

XLATensorPtr sqrt(const XLATensorPtr& input) {
  return input->CreateFrom(Sqrt(input->GetIrValue()));
}

XLATensorPtr squeeze(const XLATensorPtr& input) {
  auto input_shape = input->shape();
  auto output_dimensions = BuildSqueezedDimensions(
      input_shape.get().dimensions(), /*squeeze_dim=*/-1);
  return view(input, output_dimensions);
}

XLATensorPtr squeeze(const XLATensorPtr& input, int64_t dim) {
  auto input_shape = input->shape();
  int64_t squeeze_dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  auto output_dimensions =
      BuildSqueezedDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, output_dimensions);
}

XLATensorPtr squeeze(const XLATensorPtr& input, std::vector<int64_t> dims) {
  auto input_shape = input->shape();
  std::vector<int64_t> input_dimensions =
      torch_xla::runtime::util::ToVector<int64_t>(
          input_shape.get().dimensions());
  std::vector<int64_t> output_dimensions;
  for (int64_t dim : dims) {
    if (dim >= input_dimensions.size()) {
      continue;
    }
    int64_t squeeze_dim =
        torch::lazy::GetCanonicalDimensionIndex(dim, input_dimensions.size());
    output_dimensions = BuildSqueezedDimensions(input_dimensions, squeeze_dim);
    input_dimensions = output_dimensions;
  }
  return view(input, output_dimensions);
}

XLATensorPtr stack(absl::Span<const XLATensorPtr> tensors, int64_t dim) {
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor->GetIrValue());
  }
  int64_t canonical_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, tensors.front()->shape().get().rank() + 1);
  return tensors[0]->CreateFrom(
      torch::lazy::MakeNode<Stack>(values, canonical_dim));
}

XLATensorPtr std(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                 bool keep_reduced_dimensions, double correction) {
  return input->CreateFrom(torch::lazy::MakeNode<Std>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndices(
          torch_xla::runtime::util::ToVector<int64_t>(dimensions),
          input->shape().get().rank()),
      keep_reduced_dimensions, correction));
}

std::tuple<XLATensorPtr, XLATensorPtr> std_mean(const XLATensorPtr& input,
                                                std::vector<int64_t> dimensions,
                                                double correction,
                                                bool keep_reduced_dimensions) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<StdMean>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndices(
          torch_xla::runtime::util::ToVector<int64_t>(dimensions),
          input->shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input->CreateFrom(torch::lazy::Value(node, 0)),
                         input->CreateFrom(torch::lazy::Value(node, 1)));
}

XLATensorPtr sub(const XLATensorPtr& input, const XLATensorPtr& other,
                 const at::Scalar& alpha,
                 c10::optional<at::ScalarType> logical_element_type) {
  xla::Shape input_shape = input->shape().get();
  xla::Shape other_shape = other->shape().get();
  torch::lazy::Value constant;
  if (!input_shape.is_dynamic() && !other_shape.is_dynamic()) {
    constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
        alpha, other->shape(), logical_element_type, input->GetDevice());
  } else {
    SymIntElements sym_int_elements(other->GetIrValue());
    constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
        alpha, other->shape(), sym_int_elements, logical_element_type,
        input->GetDevice());
  }

  return input->CreateFrom(input->GetIrValue() - other->GetIrValue() * constant,
                           logical_element_type);
}

XLATensorPtr sub(const XLATensorPtr& input, const at::Scalar& other,
                 const at::Scalar& alpha,
                 c10::optional<at::ScalarType> logical_element_type) {
  torch::lazy::Value other_constant =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          other, input->shape(), logical_element_type, input->GetDevice());
  torch::lazy::Value alpha_constant =
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          alpha, input->shape(), logical_element_type, input->GetDevice());
  return input->CreateFrom(
      input->GetIrValue() - other_constant * alpha_constant,
      logical_element_type);
}

XLATensorPtr sum(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                 bool keep_reduced_dimensions,
                 c10::optional<at::ScalarType> dtype) {
  if (at::isIntegralType(input->dtype(), /*includeBool=*/true) && !dtype) {
    dtype = at::ScalarType::Long;
  } else if (!dtype) {
    dtype = input->dtype_optional();
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<Sum>(
          input->GetIrValue(),
          torch::lazy::GetCanonicalDimensionIndices(
              torch_xla::runtime::util::ToVector<int64_t>(dimensions),
              input->shape().get().rank()),
          keep_reduced_dimensions, dtype),
      dtype);
}

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> svd(
    const XLATensorPtr& input, bool some, bool compute_uv) {
  torch::lazy::NodePtr node =
      torch::lazy::MakeNode<SVD>(input->GetIrValue(), some, compute_uv);
  return std::make_tuple(input->CreateFrom(torch::lazy::Value(node, 0)),
                         input->CreateFrom(torch::lazy::Value(node, 1)),
                         input->CreateFrom(torch::lazy::Value(node, 2)));
}

XLATensorPtr tanh_backward(const XLATensorPtr& grad_output,
                           const XLATensorPtr& output) {
  return mul(grad_output, rsub(pow(output, 2), 1, 1));
}

XLATensorPtr threshold(const XLATensorPtr& input, float threshold,
                       float value) {
  return input->CreateFrom(
      torch::lazy::MakeNode<Threshold>(input->GetIrValue(), threshold, value));
}

XLATensorPtr threshold_backward(const XLATensorPtr& grad_output,
                                const XLATensorPtr& input, float threshold) {
  return grad_output->CreateFrom(torch::lazy::MakeNode<ThresholdBackward>(
      grad_output->GetIrValue(), input->GetIrValue(), threshold));
}

XLATensorPtr to(XLATensorPtr& input,
                c10::optional<torch::lazy::BackendDevice> device,
                c10::optional<at::ScalarType> scalar_type) {
  if (!device) {
    device = input->GetDevice();
  }
  if (!scalar_type) {
    scalar_type = input->dtype();
  }
  if (input->GetDevice() == *device) {
    return input->dtype() == *scalar_type
               ? input->CreateFrom(input->GetIrValue())
               : input->CreateFrom(input->GetIrValue(), *scalar_type);
  }
  XLATensorPtr new_tensor = input->CopyTensorToDevice(*device);
  if (input->dtype() != *scalar_type) {
    new_tensor->SetScalarType(*scalar_type);
  }
  return new_tensor;
}

std::tuple<XLATensorPtr, XLATensorPtr> topk(const XLATensorPtr& input,
                                            int64_t k, int64_t dim,
                                            bool largest, bool sorted,
                                            bool stable) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<TopK>(
      input->GetIrValue(), k,
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank()),
      largest, sorted, stable);
  return std::make_tuple(
      input->CreateFrom(torch::lazy::Value(node, 0)),
      input->CreateFrom(torch::lazy::Value(node, 1), at::ScalarType::Long));
}

XLATensorPtr trace(const XLATensorPtr& input) {
  auto input_shape_ref = input->shape();
  XLA_CHECK_EQ((*input_shape_ref).rank(), 2)
      << "invalid argument for trace: expected a matrix";
  torch::lazy::NodePtr eye = Identity((*input_shape_ref).dimensions(0),
                                      (*input_shape_ref).dimensions(1),
                                      (*input_shape_ref).element_type());
  return sum(input->CreateFrom(eye * input->GetIrValue()), {0, 1}, false,
             input->dtype());
}

XLATensorPtr transpose(const XLATensorPtr& input, int64_t dim0, int64_t dim1) {
  torch_xla::runtime::util::MaybeRef<xla::Shape> input_shape = input->shape();

  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    ViewInfo view_info;
    if (input_shape.get().rank() <= 1) {
      // return a view of self if input rank <=1
      torch::lazy::Value ir_value = input->GetIrValue();
      view_info = ViewInfo(ViewInfo::Type::kNoOp, GetXlaShape(ir_value),
                           GetXlaShape(ir_value));
    } else {
      std::vector<int64_t> permute_dims = torch::lazy::MakeTransposePermutation(
          /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
      view_info = ViewInfo(ViewInfo::Type::kPermute, input_shape, permute_dims);
    }
    return input->CreateViewTensor(std::move(view_info));
  }

  XLATensorPtr result;
  if (input_shape.get().rank() <= 1) {
    // kNoOp
    result = input;
  } else {
    // kPermute
    std::vector<int64_t> permute_dims = torch::lazy::MakeTransposePermutation(
        /*dim0=*/dim0, /*dim1=*/dim1, /*rank=*/input_shape.get().rank());
    result = input->CreateFrom(
        torch::lazy::MakeNode<Permute>(input->GetIrValue(), permute_dims));
  }

  return result;
}

std::tuple<XLATensorPtr, XLATensorPtr> triangular_solve(
    const XLATensorPtr& rhs, const XLATensorPtr& lhs, bool left_side,
    bool upper, bool transpose, bool unitriangular) {
  // TriangularSolve takes lower instead of upper, hence the negation.
  torch::lazy::NodePtr node = torch::lazy::MakeNode<TriangularSolve>(
      rhs->GetIrValue(), lhs->GetIrValue(), left_side, !upper, transpose,
      unitriangular);
  return std::make_tuple(rhs->CreateFrom(torch::lazy::Value(node, 0)),
                         rhs->CreateFrom(torch::lazy::Value(node, 1)));
}

std::vector<XLATensorPtr> unbind(const XLATensorPtr& input, int64_t dim) {
  dim =
      torch::lazy::GetCanonicalDimensionIndex(dim, input->shape().get().rank());
  int64_t dim_size = input->size(dim);
  std::vector<XLATensorPtr> slices;
  slices.reserve(dim_size);
  for (int64_t index = 0; index < dim_size; ++index) {
    slices.push_back(select(input, dim, index));
  }
  return slices;
}

void uniform_(XLATensorPtr& input, double from, double to) {
  XLA_CHECK_LE(from, to);
  auto input_shape = input->shape();
  input->SetInPlaceIrValue(torch::lazy::MakeNode<Uniform>(
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          from, input_shape.get().element_type(), input->GetDevice()),
      XLAGraphExecutor::Get()->GetIrValueForScalar(
          to, input_shape.get().element_type(), input->GetDevice()),
      XLAGraphExecutor::Get()->GetRngSeed(input->GetDevice()), input_shape));
}

XLATensorPtr unsqueeze(const XLATensorPtr& input, int64_t dim) {
  torch_xla::runtime::util::MaybeRef<xla::Shape> input_shape = input->shape();
  int64_t squeeze_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, input_shape.get().rank() + 1);
  std::vector<int64_t> dimensions =
      BuildUnsqueezeDimensions(input_shape.get().dimensions(), squeeze_dim);
  return view(input, dimensions);
}

void unsqueeze_(XLATensorPtr& input, int64_t dim) {
  int squeeze_dim = torch::lazy::GetCanonicalDimensionIndex(
      dim, input->shape().get().rank() + 1);
  input->SetIrValue(
      torch::lazy::MakeNode<Unsqueeze>(input->GetIrValue(), squeeze_dim));
}

XLATensorPtr upsample_bilinear2d(const XLATensorPtr& input,
                                 std::vector<int64_t> output_size,
                                 bool align_corners) {
  return input->CreateFrom(torch::lazy::MakeNode<UpsampleBilinear>(
      input->GetIrValue(), std::move(output_size), align_corners));
}

XLATensorPtr upsample_bilinear2d_backward(const XLATensorPtr& grad_output,
                                          std::vector<int64_t> output_size,
                                          std::vector<int64_t> input_size,
                                          bool align_corners) {
  return grad_output->CreateFrom(
      torch::lazy::MakeNode<UpsampleBilinearBackward>(
          grad_output->GetIrValue(), std::move(output_size),
          std::move(input_size), align_corners));
}

XLATensorPtr upsample_nearest2d(const XLATensorPtr& input,
                                std::vector<int64_t> output_size) {
  return input->CreateFrom(torch::lazy::MakeNode<UpsampleNearest>(
      input->GetIrValue(), std::move(output_size)));
}

XLATensorPtr upsample_nearest2d_backward(const XLATensorPtr& grad_output,
                                         std::vector<int64_t> output_size,
                                         std::vector<int64_t> input_size) {
  return grad_output->CreateFrom(torch::lazy::MakeNode<UpsampleNearestBackward>(
      grad_output->GetIrValue(), std::move(output_size),
      std::move(input_size)));
}

XLATensorPtr alias(const XLATensorPtr& input) {
  torch::lazy::Value ir_value = input->GetIrValue();
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    ViewInfo view_info(ViewInfo::Type::kNoOp, GetXlaShape(ir_value),
                       GetXlaShape(ir_value));
    return input->CreateViewTensor(std::move(view_info));
  }
  return input->CreateFrom(ir_value);
}

XLATensorPtr view(const XLATensorPtr& input,
                  absl::Span<const int64_t> output_size) {
  auto input_shape = input->shape();
  std::vector<int64_t> complete_dimensions =
      GetCompleteShape(output_size, input_shape.get().dimensions());
  xla::Shape shape =
      XlaHelpers::GetDynamicReshape(input_shape, complete_dimensions);

  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    ViewInfo view_info(ViewInfo::Type::kReshape, std::move(shape), input_shape);
    return input->CreateViewTensor(std::move(view_info));
  }
  return input->CreateFrom(torch::lazy::MakeNode<ViewOp>(
      input->GetIrValue(), torch::lazy::ToVector<int64_t>(shape.dimensions())));
}

XLATensorPtr view_symint(const XLATensorPtr& input,
                         at::SymIntArrayRef sym_size) {
  auto input_shape = input->shape();
  SymIntElements size_elements(sym_size);
  std::vector<int64_t> complete_dimensions = GetCompleteShape(
      size_elements.GetUpperBounds(), input_shape.get().dimensions());
  xla::Shape result_shape = xla::ShapeUtil::MakeShape(
      input_shape.get().element_type(), complete_dimensions,
      size_elements.GetDynamicDims());
  // See Note: [Disabling functionalization]
  if (runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false)) {
    ViewInfo view_info(ViewInfo::Type::kReshape, std::move(result_shape),
                       input_shape);
    return input->CreateViewTensor(std::move(view_info));
  }
  return input->CreateFrom(
      torch::lazy::MakeNode<ViewOp>(input->GetIrValue(), result_shape));
}

XLATensorPtr view_as_complex_copy(const XLATensorPtr& input) {
  return input->CreateFrom(ViewAsComplexCopy(input->GetIrValue()),
                           at::ScalarType::ComplexFloat);
}

XLATensorPtr view_as_real_copy(const XLATensorPtr& input) {
  return input->CreateFrom(ViewAsRealCopy(input->GetIrValue()),
                           at::ScalarType::Float);
}

XLATensorPtr var(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                 double correction, bool keep_reduced_dimensions) {
  return input->CreateFrom(torch::lazy::MakeNode<Var>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndices(
          torch_xla::runtime::util::ToVector<int64_t>(dimensions),
          input->shape().get().rank()),
      correction, keep_reduced_dimensions));
}

std::tuple<XLATensorPtr, XLATensorPtr> var_mean(const XLATensorPtr& input,
                                                std::vector<int64_t> dimensions,
                                                double correction,
                                                bool keep_reduced_dimensions) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<VarMean>(
      input->GetIrValue(),
      torch::lazy::GetCanonicalDimensionIndices(
          torch_xla::runtime::util::ToVector<int64_t>(dimensions),
          input->shape().get().rank()),
      correction, keep_reduced_dimensions);
  return std::make_tuple(input->CreateFrom(torch::lazy::Value(node, 0)),
                         input->CreateFrom(torch::lazy::Value(node, 1)));
}

void zero_(XLATensorPtr& input) {
  torch::lazy::Value constant = XLAGraphExecutor::Get()->GetIrValueForScalar(
      0.0, input->shape(), input->GetDevice());
  input->SetInPlaceIrValue(std::move(constant));
}

XLATensorPtr where(const XLATensorPtr& condition, const XLATensorPtr& input,
                   const XLATensorPtr& other) {
  return input->CreateFrom(
      Where(condition->GetIrValue(), input->GetIrValue(), other->GetIrValue()));
}

}  // namespace tensor_methods
}  // namespace torch_xla
