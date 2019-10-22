#include "torch_xla/csrc/ops/symeig.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

std::vector<xla::XlaOp> LowerSymEig(const xla::XlaOp& input, bool eigenvectors,
                                    bool lower) {
  xla::SelfAdjointEigResult self_adj_eig_result =
      xla::SelfAdjointEig(input, /*lower=*/lower, /*max_iter=*/100,
                          /*epsilon=*/1e-6);
  xla::XlaOp v = self_adj_eig_result.v;
  xla::XlaOp w = self_adj_eig_result.w;
  if (!eigenvectors) {
    v = xla::Zero(input.builder(), XlaHelpers::TypeOfXlaOp(input));
  }
  return {w, v};
}

xla::Shape NodeOutputShape(const Value& input, bool eigenvectors, bool lower) {
  const xla::Shape& input_shape = input.shape();
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

SymEig::SymEig(const Value& input, bool eigenvectors, bool lower)
    : Node(ir::OpKind(at::aten::symeig), {input},
           [&]() { return NodeOutputShape(input, eigenvectors, lower); },
           /*num_outputs=*/2, xla::util::MHash(eigenvectors, lower)),
      eigenvectors_(eigenvectors),
      lower_(lower) {}

NodePtr SymEig::Clone(OpList operands) const {
  return MakeNode<SymEig>(operands.at(0), eigenvectors_, lower_);
}

XlaOpVector SymEig::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerSymEig(input, eigenvectors_, lower_), loctx);
}

std::string SymEig::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", eigenvectors=" << eigenvectors_
     << ", lower=" << lower_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
