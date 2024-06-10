#include "torch_xla/csrc/ops/reduce_scatter.h"

#include <torch/csrc/lazy/core/util.h>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(AllReduceType reduce_type,
                           const torch::lazy::Value input,
                           const torch::lazy::Value& token, double scale,
                           int64_t scatter_dim, int64_t shard_count,
                           const std::vector<std::vector<int64_t>>& groups,
                           bool pin_layout) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp inputOp = operands[0];
    xla::XlaOp tokenOp = operands[1];
    ReduceScatterResult result =
        BuildReduceScatter(reduce_type, inputOp, tokenOp, scale, scatter_dim,
                           shard_count, groups, pin_layout);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(token)}, shape_fn);
}

xla::Shape NodeOutputShape(AllReduceType reduce_type,
                           const torch::lazy::Value input, double scale,
                           int64_t scatter_dim, int64_t shard_count,
                           const std::vector<std::vector<int64_t>>& groups) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp inputOp = operands[0];
    return BuildReduceScatter(reduce_type, inputOp, scale, scatter_dim,
                              shard_count, groups);
  };
  return InferOutputShape({GetXlaShape(input)}, shape_fn);
}

xla::Shape NodeOutputShapeCoalesced(
    AllReduceType reduce_type, c10::ArrayRef<torch::lazy::Value> inputs,
    const torch::lazy::Value& token, double scale, int64_t scatter_dim,
    int64_t shard_count, const std::vector<std::vector<int64_t>>& groups,
    bool pin_layout) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    ReduceScatterResultCoalesced result = BuildReduceScatterCoalesced(
        reduce_type, operands.subspan(0, operands.size() - 1), operands.back(),
        scale, scatter_dim, shard_count, groups, pin_layout);
    result.result.emplace_back(result.token);
    return xla::Tuple(operands[0].builder(), result.result);
  };
  std::vector<xla::Shape> input_shapes;
  for (const auto& input : inputs) {
    input_shapes.emplace_back(GetXlaShape(input));
  }
  input_shapes.emplace_back(GetXlaShape(token));
  return InferOutputShape(input_shapes, shape_fn);
}

}  // namespace

ReduceScatter::ReduceScatter(AllReduceType reduce_type,
                             const torch::lazy::Value& input,
                             const torch::lazy::Value& token, double scale,
                             int64_t scatter_dim, int64_t shard_count,
                             std::vector<std::vector<int64_t>> groups,
                             bool pin_layout)
    : XlaNode(
          xla_reduce_scatter, {input, token},
          [&]() {
            return NodeOutputShape(reduce_type, input, token, scale,
                                   scatter_dim, shard_count, groups,
                                   pin_layout);
          },
          /*num_outputs=*/2,
          torch::lazy::MHash(torch::lazy::GetEnumValue(reduce_type), scale,
                             scatter_dim, shard_count, groups, pin_layout)),
      reduce_type_(reduce_type),
      scale_(scale),
      scatter_dim_(scatter_dim),
      shard_count_(shard_count),
      groups_(std::move(groups)),
      pin_layout_(pin_layout) {}

ReduceScatter::ReduceScatter(AllReduceType reduce_type,
                             const torch::lazy::Value& input, double scale,
                             int64_t scatter_dim, int64_t shard_count,
                             std::vector<std::vector<int64_t>> groups)
    : XlaNode(
          xla_reduce_scatter, {input},
          [&]() {
            return NodeOutputShape(reduce_type, input, scale, scatter_dim,
                                   shard_count, groups);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(torch::lazy::GetEnumValue(reduce_type), scale,
                             scatter_dim, shard_count, groups)),
      reduce_type_(reduce_type),
      scale_(scale),
      scatter_dim_(scatter_dim),
      shard_count_(shard_count),
      groups_(std::move(groups)),
      pin_layout_(false),
      has_token_(false) {}

ReduceScatterCoalesced::ReduceScatterCoalesced(
    AllReduceType reduce_type, c10::ArrayRef<torch::lazy::Value> inputs,
    const torch::lazy::Value& token, double scale, int64_t scatter_dim,
    int64_t shard_count, std::vector<std::vector<int64_t>> groups,
    bool pin_layout)
    : XlaNode(
          xla_reduce_scatter, GetOperandListWithToken(inputs, token),
          [&]() {
            return NodeOutputShapeCoalesced(reduce_type, inputs, token, scale,
                                            scatter_dim, shard_count, groups,
                                            pin_layout);
          },
          /*num_outputs=*/inputs.size() + 1,
          torch::lazy::MHash(torch::lazy::GetEnumValue(reduce_type), scale,
                             scatter_dim, shard_count, groups, pin_layout)),
      reduce_type_(reduce_type),
      scale_(scale),
      scatter_dim_(scatter_dim),
      shard_count_(shard_count),
      groups_(std::move(groups)),
      pin_layout_(pin_layout) {}

torch::lazy::NodePtr ReduceScatter::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReduceScatter>(
      reduce_type_, operands.at(0), operands.at(1), scale_, scatter_dim_,
      shard_count_, groups_, pin_layout_);
}

torch::lazy::NodePtr ReduceScatterCoalesced::Clone(
    torch::lazy::OpList operands) const {
  std::vector<torch::lazy::Value> inputs(operands.begin(), operands.end() - 1);
  return torch::lazy::MakeNode<ReduceScatterCoalesced>(
      reduce_type_, inputs, operands.back(), scale_, scatter_dim_, shard_count_,
      groups_, pin_layout_);
}

XlaOpVector ReduceScatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  if (!has_token_) {
    auto result = BuildReduceScatter(reduce_type_, input, scale_, scatter_dim_,
                                     shard_count_, groups_);
    return ReturnOp(result, loctx);
  }
  xla::XlaOp token = loctx->GetOutputOp(operand(1));
  ReduceScatterResult result =
      BuildReduceScatter(reduce_type_, input, token, scale_, scatter_dim_,
                         shard_count_, groups_, pin_layout_);
  return ReturnOps({result.result, result.token}, loctx);
}

XlaOpVector ReduceScatterCoalesced::Lower(LoweringContext* loctx) const {
  auto& operand_list = operands();
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operand_list.size());
  for (size_t i = 0; i + 1 < operand_list.size(); ++i) {
    inputs.push_back(loctx->GetOutputOp(operand_list[i]));
  }
  xla::XlaOp token = loctx->GetOutputOp(operand_list.back());
  ReduceScatterResultCoalesced result = BuildReduceScatterCoalesced(
      reduce_type_, inputs, token, scale_, scatter_dim_, shard_count_, groups_,
      pin_layout_);
  result.result.push_back(result.token);
  return ReturnOps(result.result, loctx);
}

std::string ReduceScatter::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduce_type=" << torch::lazy::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", scatter_dim=" << scatter_dim_
     << ", shard_count=" << shard_count_ << ", pin_layout=" << pin_layout_
     << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

std::string ReduceScatterCoalesced::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduce_type=" << torch::lazy::GetEnumValue(reduce_type_)
     << ", scale=" << scale_ << ", scatter_dim=" << scatter_dim_
     << ", shard_count=" << shard_count_ << ", pin_layout=" << pin_layout_
     << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}
}  // namespace torch_xla
