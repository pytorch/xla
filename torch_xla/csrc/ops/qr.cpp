#include "torch_xla/csrc/ops/qr.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/matrix.h"
#include "xla/hlo/builder/lib/qr.h"

namespace torch_xla {
namespace {

std::vector<xla::XlaOp> LowerQR(xla::XlaOp input, bool some) {
  xla::XlaOp q, r;
  xla::QrExplicit(input, /*full_matrices=*/!some, q, r);
  return {q, r};
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input, bool some) {
  const xla::Shape& input_shape = GetXlaShape(input);
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ..., M, N
  int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
  int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);
  xla::Shape qshape(input_shape);
  xla::Shape rshape(input_shape);
  if (!some) {
    // Q is M x M
    qshape.set_dimensions(input_shape.rank() - 1, m_dim);
    // R is M x N, so left unchanged
  } else {
    // Q is M x min(M, N)
    qshape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
    // R is min(M, N) x N
    rshape.set_dimensions(input_shape.rank() - 2, std::min(m_dim, n_dim));
  }
  return xla::ShapeUtil::MakeTupleShape({qshape, rshape});
}

}  // namespace

QR::QR(const torch::lazy::Value& input, bool some)
    : XlaNode(
          torch::lazy::OpKind(at::aten::qr), {input},
          [&]() { return NodeOutputShape(input, some); },
          /*num_outputs=*/2, torch::lazy::MHash(some)),
      some_(some) {}

torch::lazy::NodePtr QR::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<QR>(operands.at(0), some_);
}

XlaOpVector QR::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerQR(input, some_), loctx);
}

std::string QR::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", some=" << some_;
  return ss.str();
}

}  // namespace torch_xla
