#include "torch_xla/csrc/ops/nll_loss.h"

#include <torch/csrc/lazy/core/util.h>

#include "absl/types/span.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& logits,
                           const torch::lazy::Value& labels,
                           const absl::optional<torch::lazy::Value>& weight,
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
  for (auto& input : xla::util::GetValuesVector<torch::lazy::Value>(
           {logits, labels}, {&weight})) {
    shapes.push_back(GetXlaShape(input));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

NllLoss::NllLoss(const torch::lazy::Value& logits,
                 const torch::lazy::Value& labels,
                 const absl::optional<torch::lazy::Value>& weight,
                 ReductionMode reduction, int ignore_index)
    : XlaNode(torch::lazy::OpKind(at::aten::nll_loss),
              xla::util::GetValuesVector<torch::lazy::Value>({logits, labels},
                                                             {&weight}),
              [&]() {
                return NodeOutputShape(logits, labels, weight, reduction,
                                       ignore_index);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(torch::lazy::GetEnumValue(reduction),
                                 ignore_index)),
      reduction_(reduction),
      ignore_index_(ignore_index) {}

torch::lazy::NodePtr NllLoss::Clone(torch::lazy::OpList operands) const {
  absl::optional<torch::lazy::Value> weight;
  if (operands.size() > 2) {
    weight = operands.at(2);
  }
  return torch::lazy::MakeNode<NllLoss>(operands.at(0), operands.at(1), weight,
                                        reduction_, ignore_index_);
}

XlaOpVector NllLoss::Lower(LoweringContext* loctx) const {
  xla::XlaOp logits = loctx->GetOutputOp(operand(0));
  xla::XlaOp labels = loctx->GetOutputOp(operand(1));
  xla::XlaOp weight;
  if (operands().size() > 2) {
    weight = loctx->GetOutputOp(operand(2));
  }
  return ReturnOp(
      BuildNllLoss(logits, labels, weight, ignore_index_, reduction_), loctx);
}

std::string NllLoss::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduction=" << torch::lazy::GetEnumValue(reduction_)
     << ", ignore_index=" << ignore_index_;
  return ss.str();
}

}  // namespace torch_xla
