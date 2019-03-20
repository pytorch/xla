#include "torch_xla/csrc/ops/qr.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

std::vector<xla::XlaOp> LowerQR(const xla::XlaOp& input, bool full_matrices) {
  xla::QRDecompositionResult qr_result =
      xla::QRDecomposition(input, /*full_matrices=*/full_matrices,
                           /*block_size=*/128, XlaHelpers::mat_mul_precision())
          .ValueOrDie();
  xla::XlaOp q = qr_result.q;
  xla::XlaOp r = qr_result.r;
  return {q, r};
}

xla::Shape NodeOutputShape(const Value& input, bool full_matrices) {
  const xla::Shape& input_shape = input.shape();
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ..., M, N
  xla::int64 m_dim = input_shape.dimensions(input_shape.rank() - 2);
  xla::int64 n_dim = input_shape.dimensions(input_shape.rank() - 1);
  xla::Shape qshape(input_shape);
  xla::Shape rshape(input_shape);
  if (full_matrices) {
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

QR::QR(const Value& input, bool full_matrices)
    : Node(
          ir::OpKind(at::aten::qr), {input},
          [&]() { return NodeOutputShape(input, full_matrices); },
          /*num_outputs=*/2, xla::util::MHash(full_matrices)),
      full_matrices_(full_matrices) {}

XlaOpVector QR::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerQR(input, full_matrices_), loctx);
}

std::string QR::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", full_matrices=" << full_matrices_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
