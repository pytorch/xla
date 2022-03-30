#include "torch_xla/csrc/ops/cholesky.h"

#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Cholesky::Cholesky(const Value& input, bool lower)
    : Node(torch::lazy::OpKind(at::aten::cholesky), {input}, input.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(lower)),
      lower_(lower) {}

NodePtr Cholesky::Clone(OpList operands) const {
  return MakeNode<Cholesky>(operands.at(0), lower_);
}

XlaOpVector Cholesky::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      xla::Triangle(xla::Cholesky(input, /*lower=*/lower_), /*lower=*/lower_);
  return ReturnOp(output, loctx);
}

std::string Cholesky::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
