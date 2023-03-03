#include "torch_xla/csrc/ops/triangular_solve.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

// This function plays two roles:
// - Computes the output shape.
// - Computes the broadcasted shape for the operands.
// NB: This currently infers the shape when left_side is true, as done in ATen.
std::pair<xla::Shape, xla::Shape> InferTriangularSolveShape(
    const xla::Shape& rhs_shape, const xla::Shape& lhs_shape) {
  // Obtain the number of right-hand sides, and dimension of the square matrix.
  int64_t nrhs = rhs_shape.dimensions(rhs_shape.rank() - 1);
  int64_t n = lhs_shape.dimensions(lhs_shape.rank() - 1);
  xla::Shape rhs_batch_shape(rhs_shape);
  xla::Shape lhs_batch_shape(lhs_shape);
  rhs_batch_shape.DeleteDimension(rhs_batch_shape.rank() - 1);
  lhs_batch_shape.DeleteDimension(lhs_batch_shape.rank() - 1);
  // If the shapes match in the batch dimensions, then we don't need to get
  // the promoted shape, and can directly add the trailing dimension.
  if (xla::ShapeUtil::Compatible(lhs_batch_shape, rhs_batch_shape)) {
    rhs_batch_shape.add_dimensions(nrhs);
    lhs_batch_shape.add_dimensions(n);
    xla::LayoutUtil::SetToDefaultLayout(&rhs_batch_shape);
    xla::LayoutUtil::SetToDefaultLayout(&lhs_batch_shape);
    return std::pair<xla::Shape, xla::Shape>(rhs_batch_shape, lhs_batch_shape);
  }
  // Obtain the promoted shapes and add back the trailing dimension.
  xla::Shape rhs_batch_promoted_shape =
      XlaHelpers::GetPromotedShape(rhs_batch_shape, lhs_batch_shape);
  xla::Shape lhs_batch_promoted_shape(rhs_batch_promoted_shape);
  rhs_batch_promoted_shape.add_dimensions(nrhs);
  lhs_batch_promoted_shape.add_dimensions(n);
  xla::LayoutUtil::SetToDefaultLayout(&rhs_batch_promoted_shape);
  xla::LayoutUtil::SetToDefaultLayout(&lhs_batch_promoted_shape);
  return std::pair<xla::Shape, xla::Shape>(rhs_batch_promoted_shape,
                                           lhs_batch_promoted_shape);
}

std::vector<xla::XlaOp> LowerTriangularSolve(xla::XlaOp rhs, xla::XlaOp lhs,
                                             bool left_side, bool lower,
                                             bool transpose,
                                             bool unit_diagonal) {
  const xla::Shape& rhs_shape = XlaHelpers::ShapeOfXlaOp(rhs);
  const xla::Shape& lhs_shape = XlaHelpers::ShapeOfXlaOp(lhs);
  std::pair<xla::Shape, xla::Shape> broadcasted_shapes =
      InferTriangularSolveShape(rhs_shape, lhs_shape);
  xla::XlaOp rhs_broadcasted =
      XlaHelpers::ImplicitBroadcast(rhs, rhs_shape, broadcasted_shapes.first);
  xla::XlaOp lhs_broadcasted =
      XlaHelpers::ImplicitBroadcast(lhs, lhs_shape, broadcasted_shapes.second);

  xla::XlaOp solution = xla::TriangularSolve(
      lhs_broadcasted, rhs_broadcasted, left_side, lower, unit_diagonal,
      transpose ? xla::TriangularSolveOptions::TRANSPOSE
                : xla::TriangularSolveOptions::NO_TRANSPOSE);
  return {solution, lhs_broadcasted};
}

xla::Shape NodeOutputShape(const torch::lazy::Value& rhs,
                           const torch::lazy::Value& lhs) {
  std::pair<xla::Shape, xla::Shape> broadcasted_shapes =
      InferTriangularSolveShape(GetXlaShape(rhs), GetXlaShape(lhs));
  return xla::ShapeUtil::MakeTupleShape(
      {broadcasted_shapes.first, broadcasted_shapes.second});
}

}  // namespace

TriangularSolve::TriangularSolve(const torch::lazy::Value& rhs,
                                 const torch::lazy::Value& lhs, bool left_side,
                                 bool lower, bool transpose, bool unit_diagonal)
    : XlaNode(torch::lazy::OpKind(at::aten::triangular_solve), {rhs, lhs},
              [&]() { return NodeOutputShape(rhs, lhs); },
              /*num_outputs=*/2,
              torch::lazy::MHash(left_side, lower, transpose, unit_diagonal)),
      left_side_(left_side),
      lower_(lower),
      transpose_(transpose),
      unit_diagonal_(unit_diagonal) {}

torch::lazy::NodePtr TriangularSolve::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<TriangularSolve>(operands.at(0), operands.at(1),
                                                left_side_, lower_, transpose_,
                                                unit_diagonal_);
}

XlaOpVector TriangularSolve::Lower(LoweringContext* loctx) const {
  xla::XlaOp rhs = loctx->GetOutputOp(operand(0));
  xla::XlaOp lhs = loctx->GetOutputOp(operand(1));
  return ReturnOps(LowerTriangularSolve(rhs, lhs, left_side_, lower_,
                                        transpose_, unit_diagonal_),
                   loctx);
}

std::string TriangularSolve::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", left_side=" << left_side_
     << ", lower=" << lower_ << ", transpose=" << transpose_
     << ", unit_diagonal=" << unit_diagonal_;
  return ss.str();
}

}  // namespace torch_xla
