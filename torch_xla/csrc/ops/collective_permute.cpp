#include "torch_xla/csrc/ops/collective_permute.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(
    const torch::lazy::Value& input, const torch::lazy::Value& token,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    CollectivePermuteResult result =
        BuildCollectivePermute(operands[0], operands[1], source_target_pairs);
    return xla::Tuple(operands[0].builder(), {result.result, result.token});
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(token)}, shape_fn);
}

}  // namespace

CollectivePermute::CollectivePermute(
    const torch::lazy::Value& input, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs)
    : XlaNode(
          xla_collective_permute, {input, token},
          [&]() { return NodeOutputShape(input, token, source_target_pairs); },
          /*num_outputs=*/2, torch::lazy::MHash(source_target_pairs)),
      source_target_pairs_(std::move(source_target_pairs)) {}

CollectivePermute::CollectivePermute(
    c10::ArrayRef<torch::lazy::Value> inputs, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs)
    : XlaNode(
          xla_collective_permute, GetOperandListWithToken(inputs, token),
          [&]() {
            std::vector<xla::Shape> input_shapes;
            for (const auto& input : inputs) {
              input_shapes.push_back(GetXlaShape(input));
            }
            input_shapes.push_back(GetXlaShape(token));
            auto shape_fn =
                [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
              std::vector<xla::XlaOp> input_ops(operands.begin(),
                                                operands.end() - 1);
              xla::XlaOp token_op = operands.back();
              MultiCollectivePermuteResult result = BuildCollectivePermute(
                  input_ops, token_op, source_target_pairs);
              std::vector<xla::XlaOp> outputs = result.results;
              outputs.push_back(result.token);
              return xla::Tuple(operands[0].builder(), outputs);
            };
            return InferOutputShape(input_shapes, shape_fn);
          },
          /*num_outputs=*/inputs.size() + 1,
          torch::lazy::MHash(source_target_pairs)),
      source_target_pairs_(std::move(source_target_pairs)) {}

torch::lazy::NodePtr CollectivePermute::Clone(
    torch::lazy::OpList operands) const {
  if (operands.size() > 2) {
    std::vector<torch::lazy::Value> inputs(operands.begin(),
                                           operands.end() - 1);
    return torch_xla::MakeNode<CollectivePermute>(inputs, operands.back(),
                                                  source_target_pairs_);
  } else {
    return torch_xla::MakeNode<CollectivePermute>(
        operands.at(0), operands.at(1), source_target_pairs_);
  }
}

XlaOpVector CollectivePermute::Lower(LoweringContext* loctx) const {
  auto& operand_list = operands();
  size_t operand_list_size = operand_list.size();
  if (operand_list_size > 2) {
    std::vector<xla::XlaOp> inputs;
    inputs.reserve(operand_list_size);
    for (size_t i = 0; i < operand_list_size - 1; ++i) {
      inputs.push_back(loctx->GetOutputOp(operand(i)));
    }
    xla::XlaOp token = loctx->GetOutputOp(operand_list.back());

    MultiCollectivePermuteResult result =
        BuildCollectivePermute(inputs, token, source_target_pairs_);

    std::vector<xla::XlaOp> outputs = result.results;
    outputs.push_back(result.token);
    return ReturnOps(outputs, loctx);
  } else {
    xla::XlaOp input = loctx->GetOutputOp(operand(0));
    xla::XlaOp token = loctx->GetOutputOp(operand(1));
    CollectivePermuteResult result =
        BuildCollectivePermute(input, token, source_target_pairs_);
    return ReturnOps({result.result, result.token}, loctx);
  }
}

std::string CollectivePermute::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", source_target_pairs=(";
  for (size_t i = 0; i < source_target_pairs_.size(); ++i) {
    ss << (i == 0 ? "(" : ", (");
    ss << source_target_pairs_[i].first << ", "
       << source_target_pairs_[i].second << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace torch_xla
