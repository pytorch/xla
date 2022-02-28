#include "torch_xla/csrc/ops/nll_loss2d.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& logits, const Value& labels,
                           const absl::optional<Value>& weight,
                           ReductionMode reduction, int ignore_index) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildNllLoss(operands[0], operands[1], weight, ignore_index,
                        reduction);
  };
  std::vector<xla::Shape> shapes;
  for (auto& input :
       xla::util::GetValuesVector<Value>({logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

NllLoss2d::NllLoss2d(const Value& logits, const Value& labels,
                     const absl::optional<Value>& weight,
                     ReductionMode reduction, int ignore_index)
    : Node(ir::OpKind(at::aten::nll_loss2d),
           xla::util::GetValuesVector<Value>({logits, labels}, {&weight}),
           [&]() {
             return NodeOutputShape(logits, labels, weight, reduction,
                                    ignore_index);
           },
           /*num_outputs=*/1,
           torch::lazy::MHash(torch::lazy::GetEnumValue(reduction),
                              ignore_index)),
      reduction_(reduction),
      ignore_index_(ignore_index) {}

NodePtr NllLoss2d::Clone(OpList operands) const {
  absl::optional<Value> weight;
  if (operands.size() > 2) {
    weight = operands.at(2);
  }
  return MakeNode<NllLoss2d>(operands.at(0), operands.at(1), weight, reduction_,
                             ignore_index_);
}

XlaOpVector NllLoss2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp logits = loctx->GetOutputOp(operand(0));
  xla::XlaOp labels = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight;
  if (operands().size() > 2) {
    weight = loctx->GetOutputOp(operand(2));
  }
  return ReturnOp(
      BuildNllLoss(logits, labels, weight, ignore_index_, reduction_), loctx);
}

std::string NllLoss2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << torch::lazy::GetEnumValue(reduction_)
     << ", ignore_index=" << ignore_index_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
