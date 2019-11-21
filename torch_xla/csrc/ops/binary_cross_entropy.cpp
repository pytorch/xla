#include "torch_xla/csrc/ops/binary_cross_entropy.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& logits, const Value& labels,
                           const absl::optional<Value>& weight,
                           ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildBinaryCrossEntropy(operands[0], operands[1], weight, reduction);
  };
  std::vector<xla::Shape> shapes;
  for (auto& input :
       xla::util::GetValuesVector<Value>({logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

BinaryCrossEntropy::BinaryCrossEntropy(const Value& logits, const Value& labels,
                                       const absl::optional<Value>& weight,
                                       ReductionMode reduction)
    : Node(ir::OpKind(at::aten::binary_cross_entropy),
           xla::util::GetValuesVector<Value>({logits, labels}, {&weight}),
           [&]() { return NodeOutputShape(logits, labels, weight, reduction); },
           /*num_outputs=*/1,
           xla::util::MHash(xla::util::GetEnumValue(reduction))),
      reduction_(reduction) {}

NodePtr BinaryCrossEntropy::Clone(OpList operands) const {
  absl::optional<Value> weight;
  if (operands.size() > 2) {
    weight = operands.at(2);
  }
  return MakeNode<BinaryCrossEntropy>(operands.at(0), operands.at(1), weight,
                                      reduction_);
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
  ss << Node::ToString()
     << ", reduction=" << xla::util::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
