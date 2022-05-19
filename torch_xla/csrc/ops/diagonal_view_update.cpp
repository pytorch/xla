#include "torch_xla/csrc/ops/diagonal_view_update.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {

DiagonalViewUpdate::DiagonalViewUpdate(const torch::lazy::Value& target,
                                       const torch::lazy::Value& input,
                                       int64_t offset, int64_t dim1,
                                       int64_t dim2)
    : XlaNode(xla_diagonal_view_update, {target, input}, GetXlaShape(target),
              /*num_outputs=*/1, torch::lazy::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

torch::lazy::NodePtr DiagonalViewUpdate::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DiagonalViewUpdate>(
      operands.at(0), operands.at(1), offset_, dim1_, dim2_);
}

XlaOpVector DiagonalViewUpdate::Lower(LoweringContext* loctx) const {
  xla::XlaOp target = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildDiagonalViewUpdate(target, input, offset_, dim1_, dim2_);
  return ReturnOp(output, loctx);
}

std::string DiagonalViewUpdate::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

}  // namespace torch_xla
