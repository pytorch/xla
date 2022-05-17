#include "torch_xla/csrc/ops/binary_cross_entropy.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& logits,
                           const torch::lazy::Value& labels,
                           const absl::optional<torch::lazy::Value>& weight,
                           ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildBinaryCrossEntropy(operands[0], operands[1], weight, reduction);
  };
  std::vector<xla::Shape> shapes;
  for (auto& input : xla::util::GetValuesVector<torch::lazy::Value>(
           {logits, labels}, {&weight})) {
    shapes.push_back(GetXlaShape(input));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

BinaryCrossEntropy::BinaryCrossEntropy(
    const torch::lazy::Value& logits, const torch::lazy::Value& labels,
    const absl::optional<torch::lazy::Value>& weight, ReductionMode reduction)
    : XlaNode(
          torch::lazy::OpKind(at::aten::binary_cross_entropy),
          xla::util::GetValuesVector<torch::lazy::Value>({logits, labels},
                                                         {&weight}),
          [&]() { return NodeOutputShape(logits, labels, weight, reduction); },
          /*num_outputs=*/1,
          torch::lazy::MHash(torch::lazy::GetEnumValue(reduction))),
      reduction_(reduction) {}

torch::lazy::NodePtr BinaryCrossEntropy::Clone(OpList operands) const {
  absl::optional<torch::lazy::Value> weight;
  if (operands.size() > 2) {
    weight = operands.at(2);
  }
  return torch::lazy::MakeNode<BinaryCrossEntropy>(
      operands.at(0), operands.at(1), weight, reduction_);
}

XlaOpVector BinaryCrossEntropy::Lower(LoweringContext* loctx) const {
  xla::XlaOp logits = loctx->GetOutputOp(operand(0));
  xla::XlaOp labels = loctx->GetOutputOp(operand(1));
  absl::optional<xla::XlaOp> weight;
  if (operands().size() > 2) {
    weight = loctx->GetOutputOp(operand(2));
  }
  return ReturnOp(BuildBinaryCrossEntropy(logits, labels, weight, reduction_),
                  loctx);
}

std::string BinaryCrossEntropy::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduction=" << torch::lazy::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace torch_xla
