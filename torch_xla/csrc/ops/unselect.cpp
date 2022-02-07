#include "torch_xla/csrc/ops/unselect.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Unselect::Unselect(const Value& target, const Value& source, int64_t dim,
                   int64_t start, int64_t end, int64_t stride)
    : Node(xla_unselect, {target, source}, target.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

NodePtr Unselect::Clone(OpList operands) const {
  return MakeNode<Unselect>(operands.at(0), operands.at(1), dim_, start_, end_,
                            stride_);
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
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
