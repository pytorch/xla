#include "torch_xla/csrc/ops/cholesky.h"

#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

Cholesky::Cholesky(const XlaValue& input, bool lower)
    : XlaNode(torch::lazy::OpKind(at::aten::cholesky), {input},
              input.xla_shape(),
              /*num_outputs=*/1, torch::lazy::MHash(lower)),
      lower_(lower) {}

torch::lazy::NodePtr Cholesky::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Cholesky>(operands.at(0), lower_);
}

XlaOpVector Cholesky::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      xla::Triangle(xla::Cholesky(input, /*lower=*/lower_), /*lower=*/lower_);
  return ReturnOp(output, loctx);
}

std::string Cholesky::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", lower=" << lower_;
  return ss.str();
}

}  // namespace torch_xla
