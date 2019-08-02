#include "torch_xla/csrc/ops/nll_loss_backward.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& logits, const Value& labels,
                           int ignore_index) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return BuildNllLossBackward(operands[0], operands[1], ignore_index);
  };
  return InferOutputShape({logits.shape(), labels.shape()}, lower_for_shape_fn);
}

}  // namespace

NllLossBackward::NllLossBackward(const Value& logits, const Value& labels,
                                 int ignore_index)
    : Node(ir::OpKind(at::aten::nll_loss_backward), {logits, labels},
           [&]() { return NodeOutputShape(logits, labels, ignore_index); },
           /*num_outputs=*/1, xla::util::MHash(ignore_index)),
      ignore_index_(ignore_index) {}

std::string NllLossBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", ignore_index=" << ignore_index_;
  return ss.str();
}

NodePtr NllLossBackward::Clone(OpList operands) const {
  return MakeNode<NllLossBackward>(operands.at(0), operands.at(1),
                                   ignore_index_);
}

XlaOpVector NllLossBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp logits = loctx->GetOutputOp(operand(0));
  xla::XlaOp labels = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildNllLossBackward(logits, labels, ignore_index_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
