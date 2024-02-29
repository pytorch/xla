#include "torch_xla/csrc/ops/unsqueeze.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int dim) {
  const xla::Shape& shape = GetXlaShape(input);
  auto dimensions = BuildUnsqueezeDimensions(shape.dimensions(), dim);
  return xla::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace

Unsqueeze::Unsqueeze(const torch::lazy::Value& input, int dim)
    : XlaNode(
          torch::lazy::OpKind(at::aten::unsqueeze), {input},
          [&]() { return NodeOutputShape(input, dim); },
          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr Unsqueeze::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Unsqueeze>(operands.at(0), dim_);
}

XlaOpVector Unsqueeze::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildUnsqueeze(input, dim_);
  return ReturnOp(output, loctx);
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
