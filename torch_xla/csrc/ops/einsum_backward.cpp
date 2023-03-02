#include "torch_xla/csrc/ops/einsum_backward.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/einsum.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

std::vector<torch::lazy::Value> GetOperandList(
    c10::ArrayRef<torch::lazy::Value> operands,
    const torch::lazy::Value& grad_output) {
  std::vector<torch::lazy::Value> operand_list(operands.begin(),
                                               operands.end());
  operand_list.insert(operand_list.begin(), grad_output);
  return operand_list;
}

xla::Shape NodeOutputShapes(const torch::lazy::Value& grad_output,
                            const torch::lazy::OpList& inputs,
                            const std::string& equation) {
  auto lower_for_shapes_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> std::vector<xla::XlaOp> {
    return BuildEinsumBackward(
        operands[0],
        std::vector<xla::XlaOp>(operands.begin() + 1, operands.end()),
        equation);
  };

  std::vector<xla::Shape> input_shapes;
  input_shapes.push_back(GetXlaShape(grad_output));
  for (auto const& op : inputs) {
    input_shapes.push_back(GetXlaShape(op));
  }

  return InferOutputShapes(absl::MakeSpan(input_shapes), lower_for_shapes_fn);
}
}  // namespace

torch::lazy::NodePtr EinsumBackward::Clone(torch::lazy::OpList operands) const {
  std::vector<torch::lazy::Value> inputs;
  inputs.reserve(operands.size() - 1);
  for (size_t i = 1; i < operands.size(); ++i) {
    inputs.push_back(operands.at(i));
  }

  return torch::lazy::MakeNode<EinsumBackward>(operands.at(0), inputs,
                                               equation_);
}

EinsumBackward::EinsumBackward(const torch::lazy::Value& grad_output,
                               const torch::lazy::OpList& inputs,
                               const std::string equation)
    : XlaNode(
          xla_einsum_backward, GetOperandList(inputs, grad_output),
          [&]() { return NodeOutputShapes(grad_output, inputs, equation); },
          /*num_outputs=*/inputs.size(), torch::lazy::MHash(equation)),
      equation_(equation) {}

XlaOpVector EinsumBackward::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  auto& operand_list = operands();
  inputs.reserve(operand_list.size() - 1);
  xla::XlaOp grad_output = loctx->GetOutputOp(operand_list[0]);

  for (size_t i = 1; i < operand_list.size(); ++i) {
    inputs.push_back(loctx->GetOutputOp(operand_list[i]));
  }

  std::vector<xla::XlaOp> ops =
      BuildEinsumBackward(grad_output, absl::MakeSpan(inputs), equation_);

  return ReturnOps(absl::MakeSpan(ops), loctx);
}

std::string EinsumBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", equation=(" << equation_ << ")";
  return ss.str();
}
}  // namespace torch_xla