#include "torch_xla/csrc/ops/binary_cross_entropy_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output, const Value& logits,
                           const Value& labels,
                           const absl::optional<Value>& weight,
                           ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 3) {
      weight = operands[3];
    }
    return BuildBinaryCrossEntropyBackward(operands[0], operands[1],
                                           operands[2], weight, reduction);
  };
  std::vector<xla::Shape> shapes;
  for (auto& input : xla::util::GetValuesVector<Value>(
           {grad_output, logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

BinaryCrossEntropyBackward::BinaryCrossEntropyBackward(
    const Value& grad_output, const Value& logits, const Value& labels,
    const absl::optional<Value>& weight, ReductionMode reduction)
    : Node(ir::OpKind(at::aten::binary_cross_entropy_backward),
           xla::util::GetValuesVector<Value>({grad_output, logits, labels},
                                             {&weight}),
           [&]() {
             return NodeOutputShape(grad_output, logits, labels, weight,
                                    reduction);
           },
           /*num_outputs=*/1,
           xla::util::MHash(xla::util::GetEnumValue(reduction))),
      reduction_(reduction) {}

NodePtr BinaryCrossEntropyBackward::Clone(OpList operands) const {
  absl::optional<Value> weight;
  if (operands.size() > 3) {
    weight = operands.at(3);
  }
  return MakeNode<BinaryCrossEntropyBackward>(
      operands.at(0), operands.at(1), operands.at(2), weight, reduction_);
}

XlaOpVector BinaryCrossEntropyBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp logits = loctx->GetOutputOp(operand(1));
  xla::XlaOp labels = loctx->GetOutputOp(operand(2));
  absl::optional<xla::XlaOp> weight;
  if (operands().size() > 3) {
    weight = loctx->GetOutputOp(operand(3));
  }
  return ReturnOp(BuildBinaryCrossEntropyBackward(grad_output, logits, labels,
                                                  weight, reduction_),
                  loctx);
}

std::string BinaryCrossEntropyBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", reduction=" << xla::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
