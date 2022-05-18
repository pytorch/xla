#include "torch_xla/csrc/ops/unselect.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {

Unselect::Unselect(const torch::lazy::Value& target,
                   const torch::lazy::Value& source, int64_t dim, int64_t start,
                   int64_t end, int64_t stride)
    : XlaNode(xla_unselect, {target, source}, GetXlaShape(target),
              /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

torch::lazy::NodePtr Unselect::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Unselect>(operands.at(0), operands.at(1), dim_,
                                         start_, end_, stride_);
}

XlaOpVector Unselect::Lower(LoweringContext* loctx) const {
  xla::XlaOp target = loctx->GetOutputOp(operand(0));
  xla::XlaOp source = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildUnselect(target, source, dim_, start_, end_,
                                    Select::GetStride(start_, end_, stride_));
  return ReturnOp(output, loctx);
}

std::string Unselect::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

}  // namespace torch_xla
