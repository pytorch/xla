#include "torch_xla/csrc/ops/einsum.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::OpList& operands,
                           const std::string& equation) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildEinsum(operands, equation);
  };

  std::vector<xla::Shape> shapes;
  for (auto const& op : operands) {
    shapes.push_back(GetXlaShape(op));
  }

  return InferOutputShape(absl::MakeSpan(shapes), lower_for_shape_fn);
}

}  // namespace

torch::lazy::NodePtr Einsum::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Einsum>(operands, equation_);
}

Einsum::Einsum(const torch::lazy::OpList& operands, const std::string equation)
    : XlaNode(torch::lazy::OpKind(at::aten::einsum), operands,
              NodeOutputShape(operands, equation),
              /*num_outputs=*/1, torch::lazy::MHash(equation)),
      equation_(std::move(equation)) {}

XlaOpVector Einsum::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  auto& operand_list = operands();
  inputs.reserve(operand_list.size());
  for (size_t i = 0; i < operand_list.size(); ++i) {
    inputs.push_back(loctx->GetOutputOp(operand_list[i]));
  }
  return ReturnOp(BuildEinsum(absl::MakeSpan(inputs), equation_), loctx);
}

std::string Einsum::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", equation=(" << equation_ << ")";
  return ss.str();
}

}  // namespace torch_xla