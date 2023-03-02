#include "torch_xla/csrc/ops/symeig.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

std::vector<xla::XlaOp> LowerSymEig(xla::XlaOp input, bool eigenvectors,
                                    bool lower) {
  xla::SelfAdjointEigResult self_adj_eig_result =
      xla::SelfAdjointEig(input, /*lower=*/lower, /*max_iter=*/100,
                          /*epsilon=*/1e-6);
  xla::XlaOp v = self_adj_eig_result.v;
  xla::XlaOp w = self_adj_eig_result.w;
  if (!eigenvectors) {
    v = xla::Zeros(input.builder(),
                   xla::ShapeUtil::MakeShape(
                       XlaHelpers::ShapeOfXlaOp(input).element_type(), {0}));
  }
  return {w, v};
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input, bool eigenvectors,
                           bool lower) {
  const xla::Shape& input_shape = GetXlaShape(input);
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // W is ..., M
  xla::Shape wshape(input_shape);
  wshape.DeleteDimension(input_shape.rank() - 1);
  xla::Shape vshape;
  if (eigenvectors) {
    // V is ..., M, M
    vshape = input_shape;
  } else {
    // V is 0
    vshape = xla::ShapeUtil::MakeShape(input_shape.element_type(), {0});
  }
  return xla::ShapeUtil::MakeTupleShape({wshape, vshape});
}

}  // namespace

SymEig::SymEig(const torch::lazy::Value& input, bool eigenvectors, bool lower)
    : XlaNode(
          torch::lazy::OpKind(at::aten::linalg_eigh), {input},
          [&]() { return NodeOutputShape(input, eigenvectors, lower); },
          /*num_outputs=*/2, torch::lazy::MHash(eigenvectors, lower)),
      eigenvectors_(eigenvectors),
      lower_(lower) {}

torch::lazy::NodePtr SymEig::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<SymEig>(operands.at(0), eigenvectors_, lower_);
}

XlaOpVector SymEig::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerSymEig(input, eigenvectors_, lower_), loctx);
}

std::string SymEig::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", eigenvectors=" << eigenvectors_
     << ", lower=" << lower_;
  return ss.str();
}

}  // namespace torch_xla
